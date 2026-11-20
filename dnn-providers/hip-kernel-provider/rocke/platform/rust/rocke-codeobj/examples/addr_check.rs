// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Developer check: `cargo run --release --example addr_check -- <gfx> [--samples N] <in.ir>...`.
//! Optimizes each input for `<gfx>`'s wave size twice: plainly, and with
//! the address reductions (`TargetInfo::address_reduction`) and the levers
//! `ROCKE_DIRECT_LEVERS` names. Compares the memory accesses one lane of the
//! two makes (`src/rir/access_trace.rs`) over N sampled launches (64 by
//! default): kernel sizes small and large, so that a kernel which picks its
//! addressing by size takes each branch, with a lane and a workgroup drawn
//! for each. Prints one tab-separated line per input:
//! `<stem> equal <samples compared> <samples skipped> <accesses compared>`,
//! `<stem> differs <the first access only one side makes>`, or
//! `<stem> error <message>`. A sample is skipped when its sizes make an nsw
//! op of the IR overflow. Exits non-zero unless every input is equal.

use std::path::Path;

use rocke_codeobj_ffi::rir::{analysis, ast, opt, parse};
use rocke_codeobj_ffi::target::Target;

#[allow(dead_code)]
#[path = "../src/rir/access_trace.rs"]
mod access_trace;

use access_trace::{optimized, trace, Inputs, UNDEFINED};

/// xorshift64: the samples depend only on the input's position.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> i64 {
        (self.next() % n) as i64
    }
}

const SMALL: [i64; 10] = [32, 64, 96, 100, 128, 300, 1000, 1024, 2048, 4096];
const LARGE: [i64; 3] = [40_000, 50_000, 65_536];

fn sample(kernel: &ast::Kernel, rng: &mut Rng, large: bool) -> Inputs {
    let sizes: &[i64] = if large { &LARGE } else { &SMALL };
    let params = kernel
        .params
        .iter()
        .enumerate()
        .map(|(i, p)| match p.ty {
            // Bases 2^40 apart, 16-byte aligned.
            ast::Type::Ptr(_) => ((i as i64 + 1) << 40) + (rng.below(1 << 20) << 4),
            _ => sizes[rng.below(sizes.len() as u64) as usize],
        })
        .collect();
    let lane = rng.below(kernel.max_workgroup_size.max(1) as u64);
    Inputs { params, thread: [lane, 0, 0], block: [rng.below(4), rng.below(4), 0], seed: rng.next() }
}

fn check(text: &str, wave: u32, samples: u32, rng: &mut Rng) -> Result<String, String> {
    // The plain kernel: no reductions and no levers. Nothing else runs in
    // this process while the variable is away.
    let levers = std::env::var_os("ROCKE_DIRECT_LEVERS");
    std::env::remove_var("ROCKE_DIRECT_LEVERS");
    let plain = optimized(text, wave, false);
    if let Some(levers) = levers {
        std::env::set_var("ROCKE_DIRECT_LEVERS", levers);
    }
    let plain = plain.map_err(|e| format!("error\t{e}"))?;
    let reduced = optimized(text, wave, true).map_err(|e| format!("error\t{e}"))?;
    let (mut compared, mut skipped, mut accesses) = (0, 0, 0);
    for i in 0..samples {
        let inputs = sample(&plain, rng, i % 3 == 2);
        let want = match trace(&plain, &inputs) {
            Ok(want) => want,
            Err(e) if e.starts_with(UNDEFINED) => {
                skipped += 1;
                continue;
            }
            Err(e) => return Err(format!("error\t{e}")),
        };
        let got = trace(&reduced, &inputs).map_err(|e| format!("error\twith the reductions: {e}"))?;
        if got != want {
            let only = want.iter().find(|a| !got.contains(a)).or_else(|| got.iter().find(|a| !want.contains(a)));
            return Err(format!("differs\t{only:?} for params {:?} lane {} block {:?}", inputs.params, inputs.thread[0], inputs.block));
        }
        compared += 1;
        accesses += want.len();
    }
    if compared == 0 {
        return Err("error\tno sample without an overflow".into());
    }
    Ok(format!("equal\t{compared}\t{skipped}\t{accesses}"))
}

fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    let mut samples = 64;
    if let Some(at) = args.iter().position(|a| a == "--samples") {
        match args.get(at + 1).and_then(|n| n.parse::<u32>().ok()) {
            Some(n) => samples = n,
            None => {
                eprintln!("addr_check: --samples needs a count");
                std::process::exit(2);
            }
        }
        args.drain(at..at + 2);
    }
    if args.len() < 3 {
        eprintln!("usage: addr_check <gfx> [--samples N] <in.ir>...");
        std::process::exit(2);
    }
    let Some(target) = Target::from_name(&args[1]) else {
        eprintln!("addr_check: unknown target {}", args[1]);
        std::process::exit(2);
    };
    let mut failed = 0;
    for (index, path) in args[2..].iter().enumerate() {
        let path = Path::new(path);
        let stem = path.file_name().and_then(|n| n.to_str()).unwrap_or("kernel");
        let stem = stem.strip_suffix(".ir").unwrap_or(stem);
        let text = std::fs::read_to_string(path).expect("read IR");
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ (index as u64 + 1));
        match check(&text, target.info().wave, samples, &mut rng) {
            Ok(line) => println!("{stem}\t{line}"),
            Err(line) => {
                failed += 1;
                println!("{stem}\t{line}");
            }
        }
    }
    if failed > 0 {
        eprintln!("{failed} input(s) failed");
        std::process::exit(1);
    }
}
