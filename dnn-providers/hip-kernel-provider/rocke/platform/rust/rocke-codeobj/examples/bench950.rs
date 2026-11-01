// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! gfx950 compile-time benchmark:
//! `cargo run --release --example bench950 -- <dir> [reps] [name-filter]`.
//! `bench_corpus` for gfx950: compiles every `<slug>.ir` listed in
//! `<dir>/index.json` in process (`rir::compile_any_target`), checks each code
//! object against `<slug>.hsaco`, then compiles the whole set `reps` more times
//! round-robin and prints the median total and per-phase times in
//! milliseconds, in `bench_corpus`'s table. `other` is the part of the total
//! the phases of the attempt a compile keeps do not cover: the attempts it
//! discarded (a retry, or the rolled and unrolled compiles). Exits non-zero if
//! any code object differs.

use std::path::Path;
use std::time::Instant;

use rocke_codeobj_ffi::rir;
use rocke_codeobj_ffi::target::Target;

fn slugs(dir: &Path) -> Vec<String> {
    let index = std::fs::read_to_string(dir.join("index.json")).expect("read index.json");
    index
        .split("\"slug\":")
        .skip(1)
        .filter_map(|rest| rest.split('"').nth(1).map(str::to_string))
        .collect()
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

struct Row {
    slug: String,
    text: String,
    same: bool,
    totals: Vec<f64>,
    phases: Vec<Vec<f64>>,
    other: Vec<f64>,
}

/// Compiles `text` for gfx950: the time in milliseconds, the code object and
/// each phase's milliseconds by name.
fn compile(text: &str, slug: &str) -> (f64, Vec<u8>, Vec<(&'static str, f64)>) {
    let start = Instant::now();
    let compiled = rir::compile_any_target(text, Target::Gfx950).unwrap_or_else(|e| panic!("{slug}: {e}"));
    let total = start.elapsed().as_secs_f64() * 1e3;
    let phases = compiled.stats.phases.iter().map(|(n, d)| (*n, d.as_secs_f64() * 1e3)).collect();
    (total, compiled.hsaco, phases)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(dir) = args.get(1).map(Path::new) else {
        eprintln!("usage: bench950 <dir> [reps] [name-filter]");
        std::process::exit(2);
    };
    let reps: usize = args.get(2).map_or(20, |r| r.parse().expect("reps"));
    let filter = args.get(3).map(String::as_str).unwrap_or("");
    let mut phase_names: Vec<&'static str> = Vec::new();
    let mut rows: Vec<Row> = Vec::new();
    for slug in slugs(dir).into_iter().filter(|s| s.contains(filter)) {
        let text = std::fs::read_to_string(dir.join(format!("{slug}.ir"))).expect("read .ir");
        let expected = std::fs::read(dir.join(format!("{slug}.hsaco"))).expect("read .hsaco");
        let (_, hsaco, phases) = compile(&text, &slug);
        for (name, _) in &phases {
            if !phase_names.contains(name) {
                phase_names.push(name);
            }
        }
        let same = hsaco == expected;
        rows.push(Row { slug, text, same, totals: Vec::new(), phases: Vec::new(), other: Vec::new() });
    }
    for row in rows.iter_mut() {
        row.phases = vec![Vec::new(); phase_names.len()];
    }
    for _ in 0..reps {
        for row in rows.iter_mut() {
            let (total, _, phases) = compile(&row.text, &row.slug);
            row.totals.push(total);
            let mut covered = 0.0;
            for (slot, name) in row.phases.iter_mut().zip(&phase_names) {
                let ms: f64 = phases.iter().filter(|(n, _)| n == name).map(|(_, ms)| ms).sum();
                covered += ms;
                slot.push(ms);
            }
            row.other.push((total - covered).max(0.0));
        }
    }
    print!("{:<30} {:>6} {:>8} {:>8}", "kernel", "ir_kb", "total", "min");
    for name in &phase_names {
        print!(" {name:>8}");
    }
    println!(" {:>8}  bytes", "other");
    let mut sum_total = 0.0;
    let mut sum_phases = vec![0.0; phase_names.len()];
    let mut sum_other = 0.0;
    for row in rows.iter_mut() {
        let total = median(&mut row.totals);
        sum_total += total;
        print!("{:<30} {:>6.0} {total:>8.3} {:>8.3}", row.slug, row.text.len() as f64 / 1024.0, row.totals[0]);
        for (i, values) in row.phases.iter_mut().enumerate() {
            let m = median(values);
            sum_phases[i] += m;
            print!(" {m:>8.3}");
        }
        let other = median(&mut row.other);
        sum_other += other;
        println!(" {other:>8.3}  {}", if row.same { "same" } else { "DIFFERENT" });
    }
    print!("{:<30} {:>6} {sum_total:>8.3} {:>8}", "sum of medians", "", "");
    for s in &sum_phases {
        print!(" {s:>8.3}");
    }
    println!(" {sum_other:>8.3}");
    let different = rows.iter().filter(|r| !r.same).count();
    if different > 0 {
        eprintln!("{different} code object(s) differ from the expected ones");
        std::process::exit(1);
    }
}
