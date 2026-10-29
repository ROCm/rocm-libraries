// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Wider schedules with the MFMA accumulators in AGPRs, at the occupancy the
//! LDS allows (gfx950). A function whose LDS (or workgroup) already holds it
//! at the fewest waves gains nothing from fewer VGPRs, yet the scheduling
//! search and `widen_schedule` keep to schedules that fit the VGPRs, and the
//! occupancy trade weighs only fewer waves. In an attention loop at one wave
//! that leaves the K and V loads one or two ahead of their uses, where COMGR,
//! its accumulators in AGPRs, keeps dozens in flight.
//!
//! Here each window wider than the one taken is scheduled past the VGPR
//! limit, its accumulator webs move to AGPRs (`agpr::place_with`, with the
//! moves their in-loop accesses need, as COMGR's code has them), and what is
//! left must fit the VGPRs. A candidate is taken when `sched::cost` gives it
//! a fifth fewer weighted slots per wave than the schedule taken, as the
//! trade asks; the fewest win. `ROCKE_DIRECT_DEBUG=nobankwide` skips it.

use super::agpr::{is_mfma, place_with, rewrite};
use super::{rules, sched};
use crate::rir::mir::MFunc;
use crate::rir::regalloc::{self, Allocation, Limits, Pressure};
use crate::target::Target;

/// The windows weighed, widest first (`rir`'s 512 window and the search's).
const WINDOWS: [usize; 4] = [512, 64, 32, 16];
/// AGPRs per lane: the pressure walks stop past the VGPR limit plus these.
const AGPRS: u32 = 256;

/// `f` with its accumulator webs (moves in loops allowed) in AGPRs, allocated
/// within `limits` without spilling, when that fits.
pub fn allocate_banked_in_loops(f: &MFunc, limits: Limits) -> Option<(MFunc, Allocation)> {
    allocate_placed(f, place_with(f, true)?, limits)
}

/// `allocate_banked_in_loops` with the webs placed.
pub(super) fn allocate_placed(f: &MFunc, (base_of, agprs): (Vec<u32>, u32), limits: Limits) -> Option<(MFunc, Allocation)> {
    let mut g = f.clone();
    rewrite(&mut g, &base_of);
    let pressure = regalloc::pressure(&g).ok()?;
    if pressure.vgprs > limits.vgprs || pressure.sgprs > limits.sgprs {
        return None;
    }
    let a = regalloc::allocate_within(&mut g, limits).ok()?;
    if a.num_vgprs > limits.vgprs || a.num_sgprs > limits.sgprs {
        return None;
    }
    let mut spill = a.spill;
    spill.agpr_dwords = agprs;
    Some((g, Allocation { pressure, spill, ..a }))
}

/// `chosen`, the allocation and window `func` has, or a wider window with its
/// accumulator webs in AGPRs at the same occupancy that `sched::cost` gives a
/// fifth fewer weighted slots, the fewest of those; `func` gets the one taken.
/// `schedule(window, cap)` is the function after load sinking scheduled with
/// `window` and its pressure, or None when the pressure passes `cap`.
pub fn widen_banked(
    func: &mut MFunc,
    chosen: (Allocation, usize),
    schedule: impl Fn(usize, Limits) -> Option<(MFunc, Pressure)>,
    lds_size: u32,
    wg: u32,
) -> (Allocation, usize) {
    if func.target != Target::Gfx950 || func.agpr_limit == Some(0) || crate::rir::debug_flag("nobankwide") {
        return chosen;
    }
    if !func.blocks.iter().flat_map(|b| &b.insts).any(|i| is_mfma(i.op)) {
        return chosen;
    }
    let t = func.target.info();
    let lds = lds_size + chosen.0.spill.lds_bytes;
    let waves = |a: &Allocation| t.occupancy(a.unified_vgprs().max(1), lds, wg);
    let have = waves(&chosen.0);
    // Fewer registers would give more waves: the search and the trade weigh those.
    if t.occupancy(1, lds, wg) > have {
        return chosen;
    }
    let hw = Limits::for_func(func);
    let room = rules::workgroup_registers(t, wg);
    let mut bar = None;
    let cap = Limits { vgprs: hw.vgprs + AGPRS, sgprs: hw.sgprs };
    let mut best: Option<(u64, MFunc, Allocation, usize)> = None;
    for window in WINDOWS.into_iter().filter(|&w| w > chosen.1) {
        let Some((f, p)) = schedule(window, cap) else { continue };
        // A window that fits the VGPRs as it is was weighed without AGPRs.
        if p.vgprs <= hw.vgprs {
            continue;
        }
        let Some((g, a)) = allocate_banked_in_loops(&f, hw) else { continue };
        if a.tight || a.unified_vgprs() > room || waves(&a) < have {
            continue;
        }
        let bar = *bar.get_or_insert_with(|| sched::cost(func).cycles(have) * 4 / 5);
        let cycles = sched::cost(&g).cycles(waves(&a));
        if cycles < bar && best.as_ref().is_none_or(|b| cycles < b.0) {
            best = Some((cycles, g, a, window));
        }
    }
    match best {
        Some((_, g, a, window)) => {
            *func = g;
            (a, window)
        }
        None => chosen,
    }
}
