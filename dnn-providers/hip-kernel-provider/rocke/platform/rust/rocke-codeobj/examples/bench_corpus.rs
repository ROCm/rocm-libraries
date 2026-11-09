//! Compile-time benchmark:
//! `cargo run --release --example bench_corpus -- <corpus-dir> [reps] [name-filter]`.
//! Compiles every `<slug>.ir` listed in `<corpus-dir>/index.json`, checks each
//! HSACO against `<slug>.hsaco`, then compiles the whole set `reps` more times
//! round-robin and prints the median total and per-phase times in
//! milliseconds. Exits non-zero if any HSACO differs.

use std::path::Path;
use std::time::Instant;

use rocke_codeobj_ffi::rir;

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
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(dir) = args.get(1).map(Path::new) else {
        eprintln!("usage: bench_corpus <corpus-dir> [reps] [name-filter]");
        std::process::exit(2);
    };
    let reps: usize = args.get(2).map_or(20, |r| r.parse().expect("reps"));
    let filter = args.get(3).map(String::as_str).unwrap_or("");
    let mut phase_names: Vec<&'static str> = Vec::new();
    let mut rows: Vec<Row> = Vec::new();
    for slug in slugs(dir).into_iter().filter(|s| s.contains(filter)) {
        let text = std::fs::read_to_string(dir.join(format!("{slug}.ir"))).expect("read .ir");
        let expected = std::fs::read(dir.join(format!("{slug}.hsaco"))).expect("read .hsaco");
        let first = rir::compile(&text).unwrap_or_else(|e| panic!("{slug}: {e}"));
        if phase_names.is_empty() {
            phase_names = first.stats.phases.iter().map(|(n, _)| *n).collect();
        }
        let same = first.hsaco == expected;
        rows.push(Row { slug, text, same, totals: Vec::new(), phases: vec![Vec::new(); phase_names.len()] });
    }
    for _ in 0..reps {
        for row in rows.iter_mut() {
            let start = Instant::now();
            let compiled = rir::compile(&row.text).expect("compile");
            row.totals.push(start.elapsed().as_secs_f64() * 1e3);
            for (slot, (_, d)) in row.phases.iter_mut().zip(&compiled.stats.phases) {
                slot.push(d.as_secs_f64() * 1e3);
            }
        }
    }
    print!("{:<30} {:>6} {:>8} {:>8}", "kernel", "ir_kb", "total", "min");
    for name in &phase_names {
        print!(" {name:>8}");
    }
    println!("  bytes");
    let mut sum_total = 0.0;
    let mut sum_phases = vec![0.0; phase_names.len()];
    for row in rows.iter_mut() {
        let total = median(&mut row.totals);
        sum_total += total;
        print!("{:<30} {:>6.0} {total:>8.3} {:>8.3}", row.slug, row.text.len() as f64 / 1024.0, row.totals[0]);
        for (i, values) in row.phases.iter_mut().enumerate() {
            let m = median(values);
            sum_phases[i] += m;
            print!(" {m:>8.3}");
        }
        println!("  {}", if row.same { "same" } else { "DIFFERENT" });
    }
    print!("{:<30} {:>6} {sum_total:>8.3} {:>8}", "sum of medians", "", "");
    for s in &sum_phases {
        print!(" {s:>8.3}");
    }
    println!();
    let different = rows.iter().filter(|r| !r.same).count();
    if different > 0 {
        eprintln!("{different} HSACO(s) differ from the corpus");
        std::process::exit(1);
    }
}
