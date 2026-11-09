// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Static LDS bank-conflict report:
//! `cargo run --release --example lds_conflicts -- [--time N] <in.ir | corpus-dir> ...`.
//! For every LDS allocation of each kernel prints the weighted LDS cycles per
//! wave (conflict-free, current layout, after the automatic layout), static
//! machine accesses with conflicts before and after, the layouts priced, and
//! the chosen layout. A directory is read through its `index.json`, like
//! `bench_corpus`, or else every `*.ir` file in it. With `--time N`, each
//! kernel is optimized once and its analysis repeated `N` times; the median
//! analysis time is printed. With `--check`, the pass's self-check runs on
//! every kernel instead of the report (exit status 1 on any failure).

use std::path::Path;
use std::time::Instant;

use rocke_codeobj_ffi::rir::{analysis, lds_swizzle, opt, parse};

fn inputs(arg: &str) -> Vec<(String, String)> {
    let path = Path::new(arg);
    if !path.is_dir() {
        let name = path.file_stem().map_or(arg.to_string(), |s| s.to_string_lossy().into_owned());
        return vec![(name, arg.to_string())];
    }
    let Ok(index) = std::fs::read_to_string(path.join("index.json")) else {
        let mut files: Vec<(String, String)> = std::fs::read_dir(path)
            .expect("read directory")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "ir"))
            .map(|p| (p.file_stem().unwrap_or_default().to_string_lossy().into_owned(), p.to_string_lossy().into_owned()))
            .collect();
        files.sort();
        return files;
    };
    index
        .split("\"slug\":")
        .skip(1)
        .filter_map(|rest| rest.split('"').nth(1))
        .map(|slug| (slug.to_string(), path.join(format!("{slug}.ir")).to_string_lossy().into_owned()))
        .collect()
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut reps = 0usize;
    if args.first().map(String::as_str) == Some("--time") {
        reps = args.get(1).and_then(|n| n.parse().ok()).expect("--time N");
        args.drain(..2);
    }
    let check = args.first().map(String::as_str) == Some("--check");
    if check {
        args.remove(0);
    }
    if args.is_empty() {
        eprintln!("usage: lds_conflicts [--time N | --check] <in.ir | dir> ...");
        std::process::exit(2);
    }
    if check {
        let (mut kernels, mut changed, mut checked, mut skipped, mut failed, mut rejected) = (0, 0, 0, 0, 0, 0);
        for arg in &args {
            for (name, path) in inputs(arg) {
                let text = std::fs::read_to_string(&path).expect("read IR");
                let optimized = parse::parse(&text).and_then(|mut kernel| {
                    let (mut lds, lds_size) = analysis::lds_layout(&kernel);
                    opt::optimize(&mut kernel, &mut lds).map(|facts| (kernel, facts, lds, lds_size))
                });
                let Ok((kernel, facts, lds, lds_size)) = optimized else {
                    rejected += 1;
                    continue;
                };
                kernels += 1;
                match lds_swizzle::self_check(&kernel, &facts, &lds, lds_size) {
                    Ok((0, 0)) => {}
                    Ok((c, s)) => {
                        changed += 1;
                        checked += c;
                        skipped += s;
                        println!("{name}: {c} accesses checked, {s} with uniform unknowns checked at zero");
                    }
                    Err(e) => {
                        failed += 1;
                        println!("{name}: FAILED: {e}");
                    }
                }
            }
        }
        println!(
            "self-check: {kernels} kernels ({rejected} rejected before the pass), {changed} changed, \
             {checked} accesses checked exactly, {skipped} with uniform unknowns checked at zero, {failed} failed"
        );
        std::process::exit(i32::from(failed > 0));
    }
    println!(
        "{:<30} {:<16} {:>6} {:>5} {:>10} {:>10} {:>10} {:>9} {:>9} {:>4}  layout / note",
        "kernel", "alloc", "bytes", "acc", "ideal", "before", "after", "confl_b", "confl_a", "cand"
    );
    let mut total_ms = 0.0;
    for arg in &args {
        for (name, path) in inputs(arg) {
            let text = std::fs::read_to_string(&path).expect("read IR");
            let mut kernel = match parse::parse(&text) {
                Ok(k) => k,
                Err(e) => {
                    println!("{name:<30} error: {e}");
                    continue;
                }
            };
            let (mut lds, lds_size) = analysis::lds_layout(&kernel);
            let facts = match opt::optimize(&mut kernel, &mut lds) {
                Ok(f) => f,
                Err(e) => {
                    println!("{name:<30} error: {e}");
                    continue;
                }
            };
            let reports = lds_swizzle::analyze(&kernel, &facts, &lds, lds_size);
            if reps > 0 && !lds.is_empty() {
                let mut times: Vec<f64> = (0..reps)
                    .map(|_| {
                        let start = Instant::now();
                        let r = lds_swizzle::analyze(&kernel, &facts, &lds, lds_size);
                        std::hint::black_box(r);
                        start.elapsed().as_secs_f64() * 1e3
                    })
                    .collect();
                times.sort_by(f64::total_cmp);
                let median = times[times.len() / 2];
                total_ms += median;
                let mut whole: Vec<f64> = (0..reps)
                    .map(|_| {
                        let (mut k, mut l) = (kernel.clone(), lds.clone());
                        let f = facts.clone();
                        let start = Instant::now();
                        let r = lds_swizzle::run(&mut k, f, &mut l, lds_size);
                        let t = start.elapsed().as_secs_f64() * 1e3;
                        std::hint::black_box(r);
                        t
                    })
                    .collect();
                whole.sort_by(f64::total_cmp);
                let mut cold: Vec<f64> = (0..reps)
                    .map(|_| {
                        let (k, l) = (kernel.clone(), lds.clone());
                        let f = facts.clone();
                        let start = Instant::now();
                        let r = lds_swizzle::analyze(&k, &f, &l, lds_size);
                        let t = start.elapsed().as_secs_f64() * 1e3;
                        std::hint::black_box(r);
                        t
                    })
                    .collect();
                cold.sort_by(f64::total_cmp);
                println!(
                    "{name:<30} analysis median {median:.4} ms (on a fresh copy {:.4}), whole pass on a fresh copy {:.4} ms over {reps}",
                    cold[cold.len() / 2],
                    whole[whole.len() / 2]
                );
            }
            if reports.is_empty() {
                println!("{name:<30} (no LDS)");
            }
            for r in reports {
                println!(
                    "{:<30} {:<16} {:>6} {:>5} {:>10.1} {:>10.1} {:>10.1} {:>4}/{:<4} {:>4}/{:<4} {:>4}  {}{}",
                    name,
                    r.name,
                    r.bytes,
                    r.accesses,
                    r.ideal,
                    r.before,
                    r.after,
                    r.conflicted_before,
                    r.machine_accesses,
                    r.conflicted_after,
                    r.machine_accesses,
                    r.candidates,
                    r.note,
                    if r.layout != lds_swizzle::Layout::Identity {
                        format!(
                            " ({} -> {} bytes, workgroups/WGP {} -> {}, {} bases, {} on residue bases)",
                            r.bytes, r.new_bytes, r.groups.0, r.groups.1, r.bases, r.explicit
                        )
                    } else {
                        String::new()
                    }
                );
            }
        }
    }
    if reps > 0 {
        println!("sum of analysis medians {total_ms:.4} ms");
    }
}
