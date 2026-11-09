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
use crate::rir::mir::{Fmt, MFunc, Opc, F_USE_EXEC};
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
    // Where `func` holds no AGPRs, every candidate is a reordering of its
    // blocks with AGPR moves added, so none issues or holds the pipe less
    // than `issue_floor`: past the bar, nothing below can pass.
    if chosen.0.spill.agpr_dwords == 0 {
        let at = *bar.insert(sched::cost(func).cycles(have) * 4 / 5);
        if issue_floor(func) >= at {
            return chosen;
        }
    }
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

/// `sched::Cost::floor` of `f` without its `v_mov_b32`s: per block, weighted
/// as there, the slots its other instructions issue in or its MFMAs hold the
/// pipe, the longer. `agpr::rewrite` keeps every instruction it is given but
/// a move whose result nothing reads any more, and adds only moves, so no
/// reordering of `f`'s blocks with webs banked gets below this.
fn issue_floor(f: &MFunc) -> u64 {
    f.blocks
        .iter()
        .map(|block| {
            let (mut issue, mut pipe) = (0u64, 0u64);
            for inst in block.insts.iter().filter(|i| !(i.op.fmt() == Fmt::Pseudo && i.op.flags() & F_USE_EXEC == 0)) {
                pipe += u64::from(sched::pipe_slots(inst.op));
                issue += u64::from(inst.op != Opc::VMovB32);
            }
            8u64.pow(block.loop_depth.min(4)) * issue.max(pipe)
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rir::mir::{Class, MBlock, MInst, Opnd, VRef, VRegInfo};
    use crate::target::gfx9::gen_isa::Gfx9Op;

    /// Accumulators started from tuples of moved constants (with a dead move
    /// now and then), MFMA chains on them in a loop that reads or copies some
    /// of them, and reads after it: banked either way, every function issues
    /// and holds the pipe at least `issue_floor` of the one it came from.
    #[test]
    fn banking_keeps_the_issue_floor() {
        let (mut banked, mut dropped) = (0, 0);
        for seed in 0..600u64 {
            let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
            let mut next = |n: u64| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x % n) as u32
            };
            let mut f = MFunc { target: Target::Gfx950, ..MFunc::default() };
            let mut reg = |size: u8| {
                f.vregs.push(VRegInfo { class: Class::V, size, align: 1, fixed: None });
                f.vregs.len() as u32 - 1
            };
            let whole = |r: u32, n: u8| Opnd::R(VRef { reg: r, off: 0, n });
            let dword = |r: u32, k: u8| Opnd::R(VRef { reg: r, off: k, n: 1 });
            let (ab, t) = (reg(4), reg(1));
            let accs: Vec<u32> = (0..1 + next(4)).map(|_| reg(4)).collect();
            let mut entry = vec![MInst::new(Opc::VAddF32, [whole(ab, 4)], [whole(ab, 4), Opnd::Imm(1)])];
            for &a in &accs {
                let c = reg(4);
                for k in 0..4 {
                    entry.push(MInst::new(Opc::VMovB32, [dword(c, k)], [Opnd::Imm(next(3))]));
                }
                entry.push(MInst::new(Opc::Copy, [whole(a, 4)], [whole(c, 4)]));
                if next(3) == 0 {
                    entry.push(MInst::new(Opc::VMovB32, [dword(reg(1), 0)], [Opnd::Imm(7)]));
                }
            }
            let mut body = Vec::new();
            for _ in 0..1 + next(12) {
                let a = accs[next(accs.len() as u64) as usize];
                body.push(match next(6) {
                    0 => MInst::new(Opc::VAddF32, [dword(t, 0)], [dword(a, next(4) as u8), Opnd::Imm(1)]),
                    1 => MInst::new(Opc::Copy, [whole(accs[next(accs.len() as u64) as usize], 4)], [whole(a, 4)]),
                    _ => MInst::new(Opc::Gfx9(Gfx9Op::VMfma16x16x32F16), [whole(a, 4)], [whole(ab, 4), whole(ab, 4), whole(a, 4)]),
                });
            }
            body.push(MInst::new(Opc::SCbranchScc1, [], []).with_imm(1));
            let exit: Vec<MInst> = accs.iter().map(|&a| MInst::new(Opc::VAddF32, [dword(t, 0)], [dword(a, 0), dword(a, 3)])).collect();
            f.blocks = vec![MBlock { insts: entry, loop_depth: 0 }, MBlock { insts: body, loop_depth: 1 + next(2) }, MBlock { insts: exit, loop_depth: 0 }];
            let movs = |f: &MFunc| f.blocks.iter().flat_map(|b| &b.insts).filter(|i| i.op == Opc::VMovB32).count();
            for in_loops in [false, true] {
                let Some((base_of, _)) = place_with(&f, in_loops) else { continue };
                let mut g = f.clone();
                rewrite(&mut g, &base_of);
                assert!(sched::cost(&g).floor() >= issue_floor(&f), "seed {seed}");
                banked += 1;
                dropped += usize::from(movs(&g) < movs(&f));
            }
        }
        // The kernels do bank, and some lose moves of constants doing so.
        assert!(banked > 300 && dropped > 100, "{banked} banked, {dropped} dropped moves");
    }
}
