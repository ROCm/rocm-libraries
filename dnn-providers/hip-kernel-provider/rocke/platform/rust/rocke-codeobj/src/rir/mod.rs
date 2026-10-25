// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Direct rocKE IR (`ck.dsl.ir/v1`) to gfx1151 HSACO compiler.

pub mod analysis;
pub mod asm;
pub mod ast;
pub mod fx;
pub mod isel;
pub mod mir;
pub mod opt;
pub mod parse;
mod parse_par;
pub mod post;
pub mod print;
pub mod regalloc;
pub mod vopd;

use std::time::{Duration, Instant};

use crate::error::CompileError;
use crate::hsaco_writer::{write_kernel_object, KernelArgMeta, KernelObjectSpec};
use ast::{Scalar, Type};

/// Conservative-mode switches for bisecting miscompiles, e.g.
/// `ROCKE_DIRECT_DEBUG=waitall,nocoalesce,nolicm,noldsreuse,noldscse,nosimplify,nofuse,nosched,nodelay,novopd`.
pub(crate) fn debug_flag(name: &str) -> bool {
    std::env::var("ROCKE_DIRECT_DEBUG").is_ok_and(|v| v.split(',').any(|f| f.trim() == name))
}

/// Waves per SIMD that can be resident on gfx1151 in wave32 WGP mode: VGPRs
/// come in blocks of 24 out of 1536 per SIMD, and a WGP's four SIMDs share
/// 128 KiB of LDS.
fn occupancy(vgprs: u32, lds_bytes: u32, workgroup_size: u32) -> u32 {
    let by_vgprs = 1536 / vgprs.max(1).next_multiple_of(24);
    let by_lds = match lds_bytes {
        0 => 16,
        lds => ((128 * 1024 / lds) * workgroup_size.div_ceil(32) / 4).max(1),
    };
    by_vgprs.min(by_lds).min(16)
}

#[derive(Debug, Default, Clone)]
pub struct CompileStats {
    pub instructions: usize,
    pub code_bytes: usize,
    pub vgprs: u32,
    pub sgprs: u32,
    pub lds_bytes: u32,
    pub kernarg_bytes: u32,
    pub phases: Vec<(&'static str, Duration)>,
}

pub struct CompiledKernel {
    pub name: String,
    pub hsaco: Vec<u8>,
    pub stats: CompileStats,
    func: mir::MFunc,
}

impl CompiledKernel {
    /// Text listing of the machine code with byte offsets and encodings.
    pub fn listing(&self) -> String {
        asm::listing(&self.func).expect("the code assembled once")
    }
}

/// Compiles one rocKE IR kernel to a loadable gfx1151 code object.
pub fn compile(text: &str) -> Result<CompiledKernel, CompileError> {
    // Merged LDS loads live longer, and there is no spilling: a kernel they
    // push out of registers compiles without them.
    match compile_with(text, true) {
        Err(error) if error.kind == "ROCKE_DIRECT_REGISTERS" => compile_with(text, false),
        result => result,
    }
}

fn compile_with(text: &str, lds_cse: bool) -> Result<CompiledKernel, CompileError> {
    let mut phases = Vec::new();
    let mut mark = Instant::now();
    let mut lap = |name: &'static str, phases: &mut Vec<(&'static str, Duration)>| {
        let now = Instant::now();
        phases.push((name, now - mark));
        mark = now;
    };
    let mut kernel = parse::parse(text)?;
    let (mut lds, lds_size) = analysis::lds_layout(&kernel);
    lap("parse", &mut phases);
    let facts = if lds_cse {
        opt::optimize(&mut kernel, &mut lds)?
    } else {
        opt::optimize_keeping_lds_loads(&mut kernel, &mut lds)?
    };
    lap("optimize", &mut phases);
    let uniform = analysis::classify(&kernel);
    let mut sel = isel::select(&kernel, &facts, uniform, &lds, lds_size)?;
    post::remove_dead_copies(&mut sel.func);
    if !debug_flag("nofuse") {
        post::fuse_valu(&mut sel.func);
    }
    lap("isel", &mut phases);
    // Overlapping latencies costs registers. Take the widest scheduling
    // window that keeps the occupancy of load sinking alone; if even that
    // does not fit, fall back to program order.
    let mut sunk = sel.func.clone();
    post::sink_load_uses(&mut sunk, 32);
    let mut base = sunk.clone();
    let alloc = match regalloc::allocate(&mut base) {
        Ok(base_alloc) => {
            let waves = |vgprs: u32| occupancy(vgprs, sel.lds_size, kernel.max_workgroup_size);
            let base_waves = waves(base_alloc.num_vgprs);
            let fewer_waves = |vgprs: u32| waves(vgprs) < base_waves;
            let windows: &[usize] = if debug_flag("nosched") { &[] } else { &[64, 32, 16] };
            let (func, alloc) = windows
                .iter()
                .find_map(|&window| {
                    let mut f = sunk.clone();
                    post::schedule_window(&mut f, window);
                    let a = regalloc::allocate_unless(&mut f, Some(&fewer_waves)).ok()??;
                    (waves(a.num_vgprs) >= base_waves).then_some((f, a))
                })
                .unwrap_or((base, base_alloc));
            sel.func = func;
            alloc
        }
        Err(_) => regalloc::allocate(&mut sel.func)?,
    };
    lap("regalloc", &mut phases);
    post::expand_pseudos(&mut sel.func)?;
    post::insert_waitcnt(&mut sel.func);
    if !debug_flag("novopd") {
        vopd::form_pairs(&mut sel.func);
    }
    post::fix_wmma_hazards(&mut sel.func);
    if !debug_flag("nodelay") {
        post::insert_delay_alu(&mut sel.func);
    }
    post::add_vgpr_release(&mut sel.func, alloc.num_vgprs);
    lap("post", &mut phases);
    let code = asm::assemble(&sel.func)?;
    lap("encode", &mut phases);

    let vgprs = alloc.num_vgprs.max(1);
    let rsrc1 = 0xe0af_0000 | (vgprs.div_ceil(8) - 1);
    let rsrc2 = (2 << 1)
        | (sel.wg_ids[0] as u32) << 7
        | (sel.wg_ids[1] as u32) << 8
        | (sel.wg_ids[2] as u32) << 9;
    let rsrc3 = (code.len().div_ceil(128).min(63) as u32) << 4;
    let (offsets, kernarg_size) = opt::kernarg_layout(&kernel);
    let args = kernel
        .params
        .iter()
        .zip(&offsets)
        .map(|(p, &offset)| {
            let (size, pointer) = match p.ty {
                Type::Ptr(_) => (8, true),
                Type::Scalar(Scalar::I64) => (8, false),
                _ => (4, false),
            };
            KernelArgMeta {
                name: p.name.clone(),
                offset,
                size,
                pointer,
                access: None,
            }
        })
        .collect();
    let spec = KernelObjectSpec {
        symbol: &kernel.name,
        code: &code,
        args,
        kernarg_size,
        group_segment_size: sel.lds_size,
        max_flat_workgroup_size: kernel.max_workgroup_size,
        sgpr_count: alloc.num_sgprs + 2,
        vgpr_count: vgprs,
        rsrc1,
        rsrc2,
        rsrc3,
    };
    let hsaco = write_kernel_object(&spec).map_err(|e| CompileError::new("ROCKE_DIRECT_ELF", e))?;
    lap("elf", &mut phases);
    let instructions = sel.func.blocks.iter().map(|b| b.insts.len()).sum();
    Ok(CompiledKernel {
        name: kernel.name.clone(),
        hsaco,
        stats: CompileStats {
            instructions,
            code_bytes: code.len(),
            vgprs,
            sgprs: alloc.num_sgprs,
            lds_bytes: sel.lds_size,
            kernarg_bytes: kernarg_size,
            phases,
        },
        func: sel.func,
    })
}
