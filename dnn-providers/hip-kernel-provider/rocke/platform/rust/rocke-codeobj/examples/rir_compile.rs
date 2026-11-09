// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Developer driver: `cargo run --release --example rir_compile -- <in.ckir> [out-prefix]`.
//! Without a prefix, prints the optimized IR. With one, compiles to
//! `<prefix>.hsaco` and writes the listing to `<prefix>.s`.

use std::time::Instant;

use rocke_codeobj_ffi::rir;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1) else {
        eprintln!("usage: rir_compile <in.ckir> [out-prefix]");
        std::process::exit(2);
    };
    let text = std::fs::read_to_string(path).expect("read IR");
    let Some(prefix) = args.get(2) else {
        let start = Instant::now();
        let mut kernel = rir::parse::parse(&text).unwrap_or_else(|e| panic!("parse: {e}"));
        let parsed = rir::print::op_count(&kernel.body);
        let (mut lds, _) = rir::analysis::lds_layout(&kernel);
        rir::opt::optimize(&mut kernel, &mut lds).unwrap_or_else(|e| panic!("optimize: {e}"));
        println!("{}", rir::print::dump(&kernel));
        eprintln!(
            "ops parsed={parsed} optimized={} time={:?}",
            rir::print::op_count(&kernel.body),
            start.elapsed()
        );
        return;
    };
    let start = Instant::now();
    let compiled = match rir::compile(&text) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("compile failed: {e}");
            std::process::exit(1);
        }
    };
    let total = start.elapsed();
    std::fs::write(format!("{prefix}.hsaco"), &compiled.hsaco).expect("write hsaco");
    std::fs::write(format!("{prefix}.s"), compiled.listing()).expect("write listing");
    let s = &compiled.stats;
    let phases: Vec<String> = s.phases.iter().map(|(n, d)| format!("{n}={:.2}ms", d.as_secs_f64() * 1e3)).collect();
    eprintln!(
        "{}: insts={} bytes={} vgpr={} sgpr={} lds={} kernarg={} total={:.2}ms [{}]",
        compiled.name,
        s.instructions,
        s.code_bytes,
        s.vgprs,
        s.sgprs,
        s.lds_bytes,
        s.kernarg_bytes,
        total.as_secs_f64() * 1e3,
        phases.join(" ")
    );
}
