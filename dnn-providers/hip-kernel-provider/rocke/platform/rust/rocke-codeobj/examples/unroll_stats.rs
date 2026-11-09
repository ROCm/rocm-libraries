// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Compiles IR files and prints one JSON line of statistics per kernel:
//! `cargo run --release --example unroll_stats -- <out-dir|-> <in.ir>...`.
//! With an output directory, writes `<stem>.hsaco` there. The debug
//! switches (`ROCKE_DIRECT_DEBUG`, e.g. `nounroll`) apply as in any compile.

use std::path::Path;
use std::time::Instant;

use rocke_codeobj_ffi::rir;
use rocke_codeobj_ffi::rir::mir;

fn mir_text(f: &mir::MFunc) -> String {
    let opnd = |o: &mir::Opnd| match o {
        mir::Opnd::R(v) => format!("r{}.{}.{}", v.reg, v.off, v.n),
        mir::Opnd::P(p, n) => format!("p{}", mir::phys_name(*p, *n)),
        mir::Opnd::Imm(v) => format!("#{v}"),
    };
    let list = |l: &[mir::Opnd]| if l.is_empty() { "-".to_string() } else { l.iter().map(opnd).collect::<Vec<_>>().join(";") };
    let mut out = String::new();
    for (reg, info) in f.vregs.iter().enumerate() {
        out += &format!("v {reg} {} {}\n", if info.class == mir::Class::S { "S" } else { "V" }, info.size);
    }
    for (index, block) in f.blocks.iter().enumerate() {
        let succs: Vec<String> = f.successors(index).iter().map(|s| s.to_string()).collect();
        out += &format!("b {index} {} {}\n", block.loop_depth, if succs.is_empty() { "-".into() } else { succs.join(",") });
        for inst in &block.insts {
            out += &format!("i {} d {} u {} imm {}\n", inst.op.name(), list(&inst.defs), list(&inst.uses), inst.imm);
        }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: unroll_stats <out-dir|-> <in.ir>...");
        std::process::exit(2);
    }
    if args[1] == "--pressure" {
        let text = std::fs::read_to_string(&args[2]).expect("read IR");
        let late = args.get(3).is_some_and(|a| a == "late");
        let mut kernel = rir::parse::parse(&text).unwrap_or_else(|e| panic!("parse: {e}"));
        let (mut lds, lds_size) = rir::analysis::lds_layout(&kernel);
        let mut facts = rir::opt::optimize(&mut kernel, &mut lds).unwrap_or_else(|e| panic!("optimize: {e}"));
        if late {
            let auto = rir::unroll::Auto::DEFAULT;
            facts = rir::unroll::unroll_optimized(&mut kernel, &mut lds, facts, auto, false).expect("unroll").0;
        }
        let (facts, lds_size) = rir::lds_swizzle::run(&mut kernel, facts, &mut lds, lds_size);
        let uniform = rir::analysis::classify(&kernel);
        let mut sel = rir::isel::select(&kernel, &facts, uniform, &lds, lds_size).unwrap_or_else(|e| panic!("isel: {e}"));
        rir::post::remove_dead_copies(&mut sel.func);
        rir::peephole::optimize(&mut sel.func);
        rir::post::fuse_valu(&mut sel.func);
        if let Some(path) = args.get(4) {
            std::fs::write(path, mir_text(&sel.func)).expect("write MIR");
        }
        // Most dwords live at once, and the registers a first-fit assignment needs.
        let report = |what: &str, f: &mir::MFunc| {
            let p = rir::regalloc::pressure(f).expect("pressure");
            let need = rir::regalloc::assignment_need(f).expect("need");
            println!("{what}: live v{} s{}, assignment v{} s{}", p.vgprs, p.sgprs, need.vgprs, need.sgprs);
        };
        report("program order", &sel.func);
        let mut sunk = sel.func.clone();
        rir::post::sink_load_uses(&mut sunk, 32);
        report("load sinking", &sunk);
        for window in [16, 32, 64] {
            let mut f = sunk.clone();
            rir::post::schedule_window(&mut f, window);
            report(&format!("window {window}"), &f);
        }
        return;
    }
    if args[1] == "--ir" {
        let text = std::fs::read_to_string(&args[2]).expect("read IR");
        let mut kernel = rir::parse::parse(&text).unwrap_or_else(|e| panic!("parse: {e}"));
        let (mut lds, _) = rir::analysis::lds_layout(&kernel);
        let facts = rir::opt::optimize(&mut kernel, &mut lds).unwrap_or_else(|e| panic!("optimize: {e}"));
        let auto = rir::unroll::Auto::DEFAULT;
        let (_, n) = rir::unroll::unroll_optimized(&mut kernel, &mut lds, facts, auto, false).expect("unroll");
        eprintln!("unrolled {n}");
        println!("{}", rir::print::dump(&kernel));
        return;
    }
    let out = (args[1] != "-").then(|| Path::new(&args[1]));
    let mut failed = 0;
    for path in &args[2..] {
        let stem = Path::new(path).file_name().and_then(|n| n.to_str()).unwrap_or(path).split('.').next().unwrap_or(path);
        let text = std::fs::read_to_string(path).expect("read IR");
        let start = Instant::now();
        match rir::compile(&text) {
            Ok(compiled) => {
                let ms = start.elapsed().as_secs_f64() * 1e3;
                let s = &compiled.stats;
                if let Some(dir) = out {
                    std::fs::write(dir.join(format!("{stem}.hsaco")), &compiled.hsaco).expect("write hsaco");
                }
                println!(
                    "{{\"ir\": \"{stem}\", \"ok\": true, \"insts\": {}, \"bytes\": {}, \"vgprs\": {}, \"sgprs\": {}, \
                     \"waves\": {}, \"lds\": {}, \"window\": {}, \"unrolled\": {}, \"spill_lds\": {}, \
                     \"spill_scratch\": {}, \"spill_sgpr\": {}, \"pressure_vgprs\": {}, \"ms\": {ms:.3}}}",
                    s.instructions,
                    s.code_bytes,
                    s.vgprs,
                    s.sgprs,
                    s.waves,
                    s.lds_bytes,
                    s.window,
                    s.unrolled,
                    s.spill.lds_dwords,
                    s.spill.scratch_dwords,
                    s.spill.sgpr_dwords,
                    s.pressure_vgprs,
                );
            }
            Err(e) => {
                failed += 1;
                let message = e.to_string().replace('\\', "\\\\").replace('"', "'");
                println!("{{\"ir\": \"{stem}\", \"ok\": false, \"kind\": \"{}\", \"message\": \"{message}\"}}", e.kind);
            }
        }
    }
    if failed > 0 {
        std::process::exit(1);
    }
}
