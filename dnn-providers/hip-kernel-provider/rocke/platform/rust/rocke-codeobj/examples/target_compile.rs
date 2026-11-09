// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Developer driver: `cargo run --release --example target_compile -- <gfx> <out-dir> [--max-vgprs N] <in.ir>...`.
//! Compiles each input for `<gfx>` (`gfx1151`, or `gfx950`, which compiles
//! through `rir::compile_any_target` until it is supported) to
//! `<out-dir>/<stem>.hsaco` with its listing in `<stem>.s`, and prints one
//! tab-separated line per input: `<stem> ok <instructions> <code bytes>
//! <vgprs> <sgprs> <lds bytes>`, or `<stem> error <message>`.
//! `--max-vgprs N` allocates within N VGPRs (`rir::compile_any_target_limited`),
//! which makes kernels that need more spill; each line then adds the VGPR
//! dwords spilled to LDS and to scratch, and the SGPR dwords kept in VGPR lanes.

use std::path::Path;

use rocke_codeobj_ffi::rir;
use rocke_codeobj_ffi::target::Target;

fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    let mut limit = None;
    if let Some(at) = args.iter().position(|a| a == "--max-vgprs") {
        limit = args.get(at + 1).and_then(|n| n.parse::<u32>().ok());
        if limit.is_none() {
            eprintln!("target_compile: --max-vgprs needs a count");
            std::process::exit(2);
        }
        args.drain(at..at + 2);
    }
    if args.len() < 4 {
        eprintln!("usage: target_compile <gfx> <out-dir> [--max-vgprs N] <in.ir>...");
        std::process::exit(2);
    }
    let Some(target) = Target::from_name(&args[1]) else {
        eprintln!("target_compile: unknown target {}", args[1]);
        std::process::exit(2);
    };
    let out_dir = Path::new(&args[2]);
    std::fs::create_dir_all(out_dir).expect("create the output directory");
    let mut failed = 0;
    for path in &args[3..] {
        let path = Path::new(path);
        let stem = path.file_name().and_then(|n| n.to_str()).unwrap_or("kernel");
        let stem = stem.strip_suffix(".ir").unwrap_or(stem);
        let text = std::fs::read_to_string(path).expect("read IR");
        let compiled = match limit {
            Some(vgprs) => rir::compile_any_target_limited(&text, target, vgprs),
            None => rir::compile_any_target(&text, target),
        };
        match compiled {
            Ok(compiled) => {
                std::fs::write(out_dir.join(format!("{stem}.hsaco")), &compiled.hsaco).expect("write the code object");
                std::fs::write(out_dir.join(format!("{stem}.s")), compiled.listing()).expect("write the listing");
                let s = &compiled.stats;
                let mut line = format!("{stem}\tok\t{}\t{}\t{}\t{}\t{}", s.instructions, s.code_bytes, s.vgprs, s.sgprs, s.lds_bytes);
                if limit.is_some() {
                    line += &format!("\t{}\t{}\t{}", s.spill.lds_dwords, s.spill.scratch_dwords, s.spill.sgpr_dwords);
                }
                println!("{line}");
            }
            Err(error) => {
                println!("{stem}\terror\t{error}");
                failed += 1;
            }
        }
    }
    if failed > 0 {
        eprintln!("{failed} input(s) failed");
    }
}
