// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! `ROCKE_DIRECT_DEBUG=trace`: where the VGPR pressure of a schedule peaks, and what is live there.

use super::mir::*;
use super::regalloc;

/// `ROCKE_DIRECT_DEBUG=trace`: where in the biggest loop block of `f` the VGPR pressure peaks, and what is live there.
pub(super) fn trace_peak(f: &MFunc, label: &str) {
    let Ok(live) = regalloc::analyze(f) else { return };
    let prof = regalloc::profile(f, &live, Class::V);
    // The deepest block with most instructions.
    for b in 0..f.blocks.len() {
        let peak = (2 * live.first[b] as usize..2 * live.first[b + 1] as usize).map(|s| prof[s]).max().unwrap_or(0);
        eprintln!("trace[{label}]: block {b} depth {} insts {} peak v{peak}", f.blocks[b].loop_depth, f.blocks[b].insts.len());
    }
    let peak_of = |b: usize| (2 * live.first[b] as usize..2 * live.first[b + 1] as usize).map(|s| prof[s]).max().unwrap_or(0);
    let Some(b) = (0..f.blocks.len()).max_by_key(|&b| (peak_of(b), f.blocks[b].insts.len())) else { return };
    let (lo, hi) = (2 * live.first[b] as usize, 2 * live.first[b + 1] as usize);
    let (mut peak, mut at) = (0, lo);
    for s in lo..hi {
        if prof[s] > peak {
            (peak, at) = (prof[s], s);
        }
    }
    eprintln!("trace[{label}]: block {b} ({} insts) peak v{peak} at inst {}", f.blocks[b].insts.len(), at / 2 - live.first[b] as usize);
    // The profile every 64 instructions.
    let line: Vec<String> = (lo..hi).step_by(64).map(|s| prof[s].to_string()).collect();
    eprintln!("trace[{label}]: profile/32 insts {}", line.join(" "));
    // Live virtual registers at the peak, by defining opcode.
    let mut by: std::collections::BTreeMap<String, (u32, u32)> = Default::default();
    for v in (0..f.vregs.len()).filter(|&v| f.vregs[v].class == Class::V) {
        let mut n = 0;
        for u in live.base[v]..live.base[v + 1] {
            if live.segs(u).iter().any(|&(s, e)| (s as usize) <= at && at <= e as usize) {
                n += 1;
            }
        }
        if n == 0 {
            continue;
        }
        let def = f.blocks.iter().flat_map(|b| b.insts.iter()).find(|i| i.defs.iter().any(|o| o.vref().is_some_and(|r| r.reg as usize == v)));
        let key = def.map_or("?".to_string(), |i| i.op.name());
        let e = by.entry(key).or_default();
        e.0 += 1;
        e.1 += n;
    }
    let mut rows: Vec<_> = by.into_iter().collect();
    rows.sort_by_key(|(_, (_, n))| std::cmp::Reverse(*n));
    for (k, (regs, units)) in rows.iter().take(14) {
        eprintln!("trace[{label}]:   {k}: {regs} vregs {units} dwords live at the peak");
    }
}
