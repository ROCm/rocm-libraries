// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Microbenchmark packager: `cargo run --release --example mb_hsaco -- <manifest>`.
//!
//! Each manifest line describes one kernel as whitespace-separated `key=value`
//! fields; the crate's own ELF/AMDHSA writer wraps the raw gfx1151 machine
//! code into a loadable code object and the crate's validator checks it:
//!
//! - `code=<path>` raw little-endian machine code, `out=<path>` the code object
//! - `name=<symbol>`, `vgprs=<n>`, `sgprs=<n>`, `lds=<bytes>`, `wg=<threads>`
//! - `kernarg=<bytes>`, `args=<name>:<offset>:<size>:<p|v>,...`
//! - `wgid=<mask>` workgroup ids in SGPRs after the kernarg pointer (bit 0 x,
//!   bit 1 y, bit 2 z)
//! - optional `rsrc1=`, `rsrc2=`, `rsrc3=` (hex) override the derived values
//! - optional `target=gfx950` (default gfx1151); for gfx950, `sgprs=` counts
//!   the SGPRs the code uses, and the descriptor adds the ones it reserves
//!
//! The derived descriptor matches what `rir::compile` emits. This is
//! measurement tooling: it is not part of the compile path.

#[allow(dead_code)]
#[path = "../src/hsaco_writer.rs"]
mod hsaco_writer;

use hsaco_writer::{write_kernel_object, KernelArgMeta, KernelObjectSpec, GFX1151_OBJECT, GFX950_OBJECT};
use rocke_codeobj_ffi::target::gfx9::program_resources;

fn parse_u32(text: &str) -> Result<u32, String> {
    let t = text.trim();
    let parsed = match t.strip_prefix("0x") {
        Some(hex) => u32::from_str_radix(hex, 16),
        None => t.parse::<u32>(),
    };
    parsed.map_err(|e| format!("bad number {t:?}: {e}"))
}

fn parse_args(text: &str) -> Result<Vec<KernelArgMeta>, String> {
    if text.is_empty() {
        return Ok(Vec::new());
    }
    text.split(',')
        .map(|item| {
            let fields: Vec<&str> = item.split(':').collect();
            if fields.len() != 4 {
                return Err(format!("argument {item:?} is not name:offset:size:p|v"));
            }
            Ok(KernelArgMeta {
                name: fields[0].to_string(),
                offset: parse_u32(fields[1])?,
                size: parse_u32(fields[2])?,
                pointer: fields[3] == "p",
                access: None,
            })
        })
        .collect()
}

fn package(line: &str) -> Result<(String, usize), String> {
    let mut fields = std::collections::HashMap::new();
    for item in line.split_whitespace() {
        let (key, value) = item.split_once('=').ok_or_else(|| format!("field {item:?} has no '='"))?;
        fields.insert(key, value);
    }
    let get = |key: &str| fields.get(key).copied().ok_or_else(|| format!("missing {key}="));
    let num = |key: &str, default: u32| fields.get(key).map_or(Ok(default), |v| parse_u32(v));
    let code = std::fs::read(get("code")?).map_err(|e| format!("read code: {e}"))?;
    let out = get("out")?.to_string();
    let vgprs = num("vgprs", 8)?.max(1);
    let wgid = num("wgid", 1)?;
    let gfx950 = match fields.get("target").copied() {
        None | Some("gfx1151") => false,
        Some("gfx950") => true,
        Some(other) => return Err(format!("unknown target {other:?}")),
    };
    let derived = if gfx950 {
        let r = program_resources(vgprs, num("sgprs", 0)?, [wgid & 1 != 0, wgid & 2 != 0, wgid & 4 != 0], false);
        (r.rsrc1, r.rsrc2, r.rsrc3, r.sgpr_count, r.vgpr_count)
    } else {
        let rsrc1 = 0xe0af_0000 | (vgprs.div_ceil(8) - 1);
        let rsrc2 = (2 << 1) | (wgid & 1) << 7 | (wgid >> 1 & 1) << 8 | (wgid >> 2 & 1) << 9;
        (rsrc1, rsrc2, (code.len().div_ceil(128).min(63) as u32) << 4, num("sgprs", 106)?, vgprs)
    };
    let spec = KernelObjectSpec {
        target: if gfx950 { GFX950_OBJECT } else { GFX1151_OBJECT },
        symbol: get("name")?,
        code: &code,
        args: parse_args(fields.get("args").copied().unwrap_or(""))?,
        kernarg_size: num("kernarg", 0)?,
        group_segment_size: num("lds", 0)?,
        max_flat_workgroup_size: num("wg", 1024)?,
        sgpr_count: derived.3,
        vgpr_count: derived.4,
        agpr_count: 0,
        rsrc1: num("rsrc1", derived.0)?,
        rsrc2: num("rsrc2", derived.1)?,
        rsrc3: num("rsrc3", derived.2)?,
        private_segment_size: 0,
        vgpr_spill_count: 0,
        kernarg_preload: 0,
    };
    let hsaco = write_kernel_object(&spec)?;
    rocke_codeobj_ffi::validate_hsaco(&hsaco, &[spec.symbol.to_string()])?;
    std::fs::write(&out, &hsaco).map_err(|e| format!("write {out}: {e}"))?;
    Ok((out, hsaco.len()))
}

fn main() {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: mb_hsaco <manifest>");
        std::process::exit(2);
    };
    let text = std::fs::read_to_string(&path).expect("read manifest");
    let mut failed = 0;
    for line in text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
        match package(line) {
            Ok((out, bytes)) => println!("ok {out} {bytes}"),
            Err(error) => {
                failed += 1;
                println!("error {error} :: {line}");
            }
        }
    }
    std::process::exit(i32::from(failed > 0));
}
