// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Developer driver: `cargo run --release --example mir_dump -- [--peephole] <out-dir> <in.ir>...`.
//! Writes `<out-dir>/<stem>.mir` for each input: the machine IR after
//! instruction selection, dead-copy removal and VALU fusion, in a line format
//! for offline pattern mining. With `--peephole` the peephole pass runs where
//! the compiler runs it (after dead-copy removal, before VALU fusion, which is
//! then left out): `<stem>.pre.mir` is its input, `<stem>.mir` its output, and
//! how often each rule family fired is printed per input.
//! `MIR_DUMP_TIME_REPS=<n>` also prints the pass's mean time over n runs.
//!
//! ```text
//! kernel <name>
//! v <reg> <S|V> <size> <fixed register or ->
//! b <index> <loop depth> <successors, comma-separated or ->
//! i <mnemonic> d <operands> u <operands> imm <imm> mods <mods>
//! ```
//!
//! Operands are separated by `;` (`-` for none): `r<reg>.<offset>.<dwords>`,
//! `p<physical name>`, or `#<value as u32>`.

use std::fmt::Write;
use std::path::Path;

use rocke_codeobj_ffi::rir::{analysis, isel, mir, opt, parse, peephole, post};

fn operand(o: &mir::Opnd) -> String {
    match o {
        mir::Opnd::R(v) => format!("r{}.{}.{}", v.reg, v.off, v.n),
        mir::Opnd::P(p, n) => format!("p{}", mir::phys_name(*p, *n)),
        mir::Opnd::Imm(v) => format!("#{v}"),
    }
}

fn operands(list: &[mir::Opnd]) -> String {
    if list.is_empty() {
        return "-".into();
    }
    list.iter().map(operand).collect::<Vec<_>>().join(";")
}

fn dump(name: &str, f: &mir::MFunc) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "kernel {name}");
    for (reg, info) in f.vregs.iter().enumerate() {
        let class = if info.class == mir::Class::S { "S" } else { "V" };
        let fixed = info.fixed.map_or("-".to_string(), |p| p.to_string());
        let _ = writeln!(out, "v {reg} {class} {} {fixed}", info.size);
    }
    for (index, block) in f.blocks.iter().enumerate() {
        let succs = f.successors(index);
        let succs = if succs.is_empty() {
            "-".to_string()
        } else {
            succs.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(",")
        };
        let _ = writeln!(out, "b {index} {} {succs}", block.loop_depth);
        for inst in &block.insts {
            let _ = writeln!(
                out,
                "i {} d {} u {} imm {} mods {}",
                inst.op.name(),
                operands(&inst.defs),
                operands(&inst.uses),
                inst.imm,
                inst.mods
            );
        }
    }
    out
}

fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    let with_peephole = args.get(1).is_some_and(|a| a == "--peephole");
    if with_peephole {
        args.remove(1);
    }
    if args.len() < 3 {
        eprintln!("usage: mir_dump [--peephole] <out-dir> <in.ir>...");
        std::process::exit(2);
    }
    let out_dir = Path::new(&args[1]);
    std::fs::create_dir_all(out_dir).expect("create the output directory");
    let mut failed = 0;
    let mut total = [0u32; peephole::FAMILIES.len()];
    for path in &args[2..] {
        let path = Path::new(path);
        let stem = path.file_name().and_then(|n| n.to_str()).unwrap_or("kernel");
        let stem = stem.strip_suffix(".ir").unwrap_or(stem);
        let text = std::fs::read_to_string(path).expect("read IR");
        let selected = (|| {
            let mut kernel = parse::parse(&text)?;
            let (mut lds, lds_size) = analysis::lds_layout(&kernel);
            let facts = opt::optimize(&mut kernel, &mut lds)?;
            let uniform = analysis::classify(&kernel);
            let mut sel = isel::select(&kernel, &facts, uniform, &lds, lds_size)?;
            post::remove_dead_copies(&mut sel.func);
            let mut pre = None;
            let stats = if with_peephole {
                pre = Some(dump(&kernel.name, &sel.func));
                if let Some(reps) = std::env::var("MIR_DUMP_TIME_REPS").ok().and_then(|r| r.parse::<u32>().ok()) {
                    let start = std::time::Instant::now();
                    for _ in 0..reps {
                        let mut g = sel.func.clone();
                        std::hint::black_box(peephole::optimize(&mut g));
                    }
                    let clone_start = std::time::Instant::now();
                    for _ in 0..reps {
                        std::hint::black_box(sel.func.clone());
                    }
                    let pass = (clone_start - start).saturating_sub(clone_start.elapsed());
                    eprintln!("{} us per pass", pass.as_secs_f64() * 1e6 / reps as f64);
                }
                Some(peephole::optimize(&mut sel.func))
            } else {
                post::fuse_valu(&mut sel.func);
                None
            };
            Ok::<_, rocke_codeobj_ffi::error::CompileError>((kernel.name.clone(), sel.func, stats, pre))
        })();
        match selected {
            Ok((name, func, stats, pre)) => {
                std::fs::write(out_dir.join(format!("{stem}.mir")), dump(&name, &func)).expect("write the dump");
                if let Some(pre) = pre {
                    std::fs::write(out_dir.join(format!("{stem}.pre.mir")), pre).expect("write the dump");
                }
                if let Some(stats) = stats {
                    let fired: Vec<String> = peephole::FAMILIES
                        .iter()
                        .zip(stats.fired)
                        .filter(|(_, n)| *n > 0)
                        .map(|(f, n)| format!("{f}={n}"))
                        .collect();
                    println!("{stem}: {}", fired.join(" "));
                    for (t, n) in total.iter_mut().zip(stats.fired) {
                        *t += n;
                    }
                }
            }
            Err(error) => {
                eprintln!("{}: {error}", path.display());
                failed += 1;
            }
        }
    }
    if with_peephole {
        let fired: Vec<String> = peephole::FAMILIES.iter().zip(total).map(|(f, n)| format!("{f}={n}")).collect();
        println!("total: {}", fired.join(" "));
    }
    if failed > 0 {
        eprintln!("{failed} input(s) failed");
    }
}
