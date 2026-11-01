// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Direct, toolchain-free compilation of one qualified rocKE GEMM IR pattern.
//!
//! This is intentionally not a general AMDGPU compiler. It recognizes the
//! complete serialized `ckdsl.ir/v1` graph emitted by rocKE's gfx1151 f16 RCR
//! WMMA GEMM builder, selects a fixed-register wave32 schedule, encodes gfx1151
//! instructions, and writes an ELF/AMDHSA code object. The checked-in HSACO is
//! used only by tests as a qualification oracle.

use std::time::Instant;

use crate::elf::{validate_hsaco, ElfFacts};
use crate::error::CompileError;
use crate::gfx1151_isa::encode_wmma_gemm;
use crate::hsaco_writer::write_gfx1151_wmma_gemm;

pub const DIRECT_GFX1151_WMMA_GEMM_SYMBOL: &str = "rocke_direct_wmma_gemm_wmma16x16x16_fp16_rcr_xm";
pub const DIRECT_BACKEND_VERSION: &str = "gfx1151-wmma-f16-rcr-16x16x16/v2";

const XM_IR: &str = include_str!("gfx1151_wmma_gemm.ir");
const XM_SUFFIX: &str = "_wmma16x16x16_fp16_rcr_xm";
const XN_SUFFIX: &str = "_wmma16x16x16_fp16_rcr_xn";
const XM_BLOCK_X: &str = "%bid8 : i32 = gpu.block_id (  ) { axis = s:\"x\" }";
const XM_BLOCK_Y: &str = "%bid10 : i32 = gpu.block_id (  ) { axis = s:\"y\" }";
const XN_BLOCK_X: &str = "%bid8 : i32 = gpu.block_id (  ) { axis = s:\"y\" }";
const XN_BLOCK_Y: &str = "%bid10 : i32 = gpu.block_id (  ) { axis = s:\"x\" }";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectGemmSpec {
    pub symbol: String,
    pub block_x_is_m: bool,
}

/// A generated code object plus the evidence checked before returning it.
#[derive(Debug)]
pub struct DirectCompilation {
    pub hsaco: Vec<u8>,
    pub elf: ElfFacts,
    pub seconds: f64,
    pub diagnostics: String,
}

fn normalize_ir(text: &str) -> String {
    text.replace("\r\n", "\n")
        .lines()
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n")
        .trim_end()
        .to_string()
}

fn first_difference(expected: &str, actual: &str) -> String {
    let expected_lines: Vec<_> = expected.lines().collect();
    let actual_lines: Vec<_> = actual.lines().collect();
    let common = expected_lines.len().min(actual_lines.len());
    for index in 0..common {
        if expected_lines[index] != actual_lines[index] {
            return format!(
                "line {} differs: expected {:?}, got {:?}",
                index + 1,
                expected_lines[index],
                actual_lines[index]
            );
        }
    }
    format!(
        "line count differs: expected {}, got {}",
        expected_lines.len(),
        actual_lines.len()
    )
}

/// Recognize the complete serialized IR graph for the one supported GEMM.
///
/// Exact graph matching is intentional: a changed operand, attribute, type, or
/// operation is rejected instead of silently receiving code for a different
/// computation.
pub fn parse_gfx1151_wmma_gemm_ir(ir: &str) -> Result<DirectGemmSpec, CompileError> {
    let actual = normalize_ir(ir);
    let header = actual
        .lines()
        .nth(1)
        .ok_or_else(|| CompileError::new("direct-gemm-ir", "serialized IR has no kernel header"))?;
    let symbol = header
        .strip_prefix("kernel @")
        .and_then(|line| line.strip_suffix(" {"))
        .ok_or_else(|| {
            CompileError::new(
                "direct-gemm-ir",
                "serialized IR kernel header is not `kernel @NAME {`",
            )
        })?;
    let block_x_is_m = if symbol.ends_with(XM_SUFFIX) {
        true
    } else if symbol.ends_with(XN_SUFFIX) {
        false
    } else {
        return Err(CompileError::new(
            "direct-gemm-ir",
            format!("kernel {symbol:?} is outside the f16 RCR WMMA 16x16x16 xm/xn slice"),
        ));
    };

    let mut expected = XM_IR.replace(DIRECT_GFX1151_WMMA_GEMM_SYMBOL, symbol);
    if !block_x_is_m {
        expected = expected
            .replace(XM_BLOCK_X, XN_BLOCK_X)
            .replace(XM_BLOCK_Y, XN_BLOCK_Y);
    }
    let expected = normalize_ir(&expected);
    if actual != expected {
        return Err(CompileError::new(
            "direct-gemm-ir",
            format!(
                "kernel {symbol:?} does not match the qualified rocKE GEMM graph: {}",
                first_difference(&expected, &actual)
            ),
        ));
    }
    Ok(DirectGemmSpec {
        symbol: symbol.to_string(),
        block_x_is_m,
    })
}

pub fn compile_gfx1151_wmma_gemm_ir(ir: &str) -> Result<DirectCompilation, CompileError> {
    let started = Instant::now();
    let spec = parse_gfx1151_wmma_gemm_ir(ir)?;
    let code = encode_wmma_gemm(spec.block_x_is_m);
    let hsaco = write_gfx1151_wmma_gemm(&spec.symbol, &code)
        .map_err(|message| CompileError::new("direct-gemm-elf-write", message))?;
    let elf = validate_hsaco(&hsaco, std::slice::from_ref(&spec.symbol))
        .map_err(|message| CompileError::new("direct-gemm-elf", message))?;
    let diagnostics = format!(
        "matched ck.dsl.ir/v1 {DIRECT_BACKEND_VERSION}; encoded {} gfx1151 bytes and \
         wrote {} ELF/AMDHSA bytes in Rust; no COMGR, LLVM, Clang, assembler, linker, \
         subprocess, dynamic compiler library, image template, or cache",
        code.len(),
        hsaco.len()
    );
    Ok(DirectCompilation {
        hsaco,
        elf,
        seconds: started.elapsed().as_secs_f64(),
        diagnostics,
    })
}

pub fn emit_gfx1151_wmma_gemm() -> Result<DirectCompilation, CompileError> {
    compile_gfx1151_wmma_gemm_ir(XM_IR)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_gemm_compiles_ir_into_a_valid_contract() {
        let compiled = emit_gfx1151_wmma_gemm().unwrap();
        let image = &compiled.hsaco;
        assert!(image.starts_with(b"\x7fELF"));
        assert_eq!(compiled.elf.abi_version, 4);
        assert_eq!(compiled.elf.flags, 0x4a);
        assert_eq!(compiled.elf.metadata_notes, 1);
        assert!(compiled
            .elf
            .exported_symbols
            .iter()
            .any(|symbol| symbol == DIRECT_GFX1151_WMMA_GEMM_SYMBOL));
        assert!(compiled.seconds > 0.0);
        assert!(compiled.diagnostics.contains(DIRECT_BACKEND_VERSION));
        assert!(compiled.diagnostics.contains("no COMGR"));
        assert!(compiled.diagnostics.contains("encoded 744 gfx1151 bytes"));
        assert!(!image
            .windows(b"Linker: AMD LLVM".len())
            .any(|window| window == b"Linker: AMD LLVM"));
    }

    #[test]
    fn changed_ir_is_rejected_instead_of_receiving_template_code() {
        let changed = XM_IR.replace("value = i:16", "value = i:8");
        let error = compile_gfx1151_wmma_gemm_ir(&changed).unwrap_err();
        assert_eq!(error.kind, "direct-gemm-ir");
        assert!(error.message.contains("does not match"));
    }

    #[test]
    fn xn_graph_drives_a_distinct_generated_symbol_and_mapping() {
        let symbol = "rocke_direct_wmma_gemm_wmma16x16x16_fp16_rcr_xn";
        let xn = XM_IR
            .replace(DIRECT_GFX1151_WMMA_GEMM_SYMBOL, symbol)
            .replace(XM_BLOCK_X, XN_BLOCK_X)
            .replace(XM_BLOCK_Y, XN_BLOCK_Y);
        let compiled = compile_gfx1151_wmma_gemm_ir(&xn).unwrap();
        assert!(compiled
            .elf
            .exported_symbols
            .iter()
            .any(|item| item == symbol));
        assert!(compiled
            .hsaco
            .windows(symbol.len())
            .any(|window| window == symbol.as_bytes()));
    }
}
