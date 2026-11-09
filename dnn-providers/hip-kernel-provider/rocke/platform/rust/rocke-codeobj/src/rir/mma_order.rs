// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Program order for register pressure. A matrix loop written as "load
//! every B fragment, then every WMMA" keeps all fragments live next to the
//! accumulators, which can exceed the register file in any schedule the
//! scheduler derives from it: it only delays uses. When program order does
//! not fit the hardware and the reordered function does, each WMMA moves up
//! to just below the last definition of its sources, so that a fragment
//! dies at its WMMA; load sinking then overlaps a bounded number of loads
//! again. `compile` tries it only in the retries of a kernel whose first
//! compile ran out of registers, so that the others pay nothing for it.

use super::mir::*;
use super::regalloc;

/// Whether instruction `x` writes a unit that `units` names.
fn writes_any(x: &MInst, units: &[(u32, u8)]) -> bool {
    x.defs.iter().any(|o| overlaps(o, units))
}

fn overlaps(o: &Opnd, units: &[(u32, u8)]) -> bool {
    match o {
        Opnd::R(v) => (v.off..v.off + v.n).any(|k| units.contains(&(v.reg, k))),
        _ => false,
    }
}

fn units(opnds: &[Opnd]) -> Vec<(u32, u8)> {
    opnds.iter().filter_map(Opnd::vref).flat_map(|v| (v.off..v.off + v.n).map(move |k| (v.reg, k))).collect()
}

/// Whether the WMMA `w` may move above `x`: `x` neither defines what `w`
/// reads nor touches what `w` writes, and orders nothing (no control flow,
/// EXEC change, fence, barrier or physical register).
fn may_pass(w_uses: &[(u32, u8)], w_defs: &[(u32, u8)], x: &MInst) -> bool {
    let flags = x.op.flags();
    if flags & (F_BRANCH | F_END | F_DEF_EXEC | F_FENCE) != 0
        || matches!(x.op, Opc::SBarrier | Opc::SSendmsg)
        || x.defs.iter().chain(&x.uses).any(|o| matches!(o, Opnd::P(p, _) if *p != Phys::Null))
    {
        return false;
    }
    !writes_any(x, w_uses) && !writes_any(x, w_defs) && !x.uses.iter().any(|o| overlaps(o, w_defs))
}

/// Moves each WMMA of every block up past instructions it does not depend
/// on, together with the copies right before it that give it a source (the
/// accumulator it updates in place); returns how many moved.
fn hoist_mma(f: &mut MFunc) -> usize {
    let mut moved = 0;
    for block in f.blocks.iter_mut() {
        let insts = &mut block.insts;
        for i in 0..insts.len() {
            if insts[i].op.flags() & F_WMMA == 0 {
                continue;
            }
            let reads = units(&insts[i].uses);
            let mut first = i;
            while first > 0 && insts[first - 1].op == Opc::Copy && writes_any(&insts[first - 1], &reads) {
                first -= 1;
            }
            let group = &insts[first..=i];
            let uses = units(&group.iter().flat_map(|w| w.uses.iter().copied()).collect::<Vec<_>>());
            let defs = units(&group.iter().flat_map(|w| w.defs.iter().copied()).collect::<Vec<_>>());
            let mut to = first;
            while to > 0 && may_pass(&uses, &defs, &insts[to - 1]) {
                to -= 1;
            }
            if to < first {
                insts[to..=i].rotate_right(i + 1 - first);
                moved += 1;
            }
        }
    }
    moved
}

/// Reorders `f` (see the module notes) when its program order needs more
/// VGPRs than the hardware has and the reordered function does not.
pub(super) fn fit(f: &mut MFunc) {
    let over = |f: &MFunc| regalloc::pressure(f).map_or(true, |p| p.vgprs > regalloc::MAX_VGPRS);
    if super::debug_flag("nommaorder") || !over(f) {
        return;
    }
    let mut g = f.clone();
    if hoist_mma(&mut g) > 0 && !over(&g) {
        *f = g;
    }
}

#[cfg(test)]
mod tests {
    use super::super::spill::tests::effects;
    use super::*;

    fn reg(f: &mut MFunc, size: u8) -> Opnd {
        f.vregs.push(VRegInfo { class: Class::V, size, align: 1, fixed: None });
        Opnd::R(VRef { reg: f.vregs.len() as u32 - 1, off: 0, n: size })
    }

    fn half(o: Opnd, k: u8) -> Opnd {
        let Opnd::R(v) = o else { unreachable!("a virtual register") };
        Opnd::R(VRef { reg: v.reg, off: 4 * k, n: 4 })
    }

    /// `tiles` accumulators (registers 2 on) updated in place by WMMAs that
    /// read one A and one B fragment each, all B fragments loaded first (each
    /// copied from two LDS loads), then every accumulator stored; `between`
    /// goes right before the first WMMA.
    fn matrix_loop(tiles: usize, between: Option<fn(Opnd) -> MInst>) -> MFunc {
        let mut f = MFunc::default();
        let tid = reg(&mut f, 1);
        f.vregs[0].fixed = Some(0);
        let mut insts = Vec::new();
        let a = reg(&mut f, 8);
        insts.push(MInst::new(Opc::ImplicitDef, [a], []));
        let accs: Vec<Opnd> = (0..tiles).map(|_| reg(&mut f, 8)).collect();
        for &acc in &accs {
            insts.push(MInst::new(Opc::ImplicitDef, [acc], []));
        }
        let mut bs = Vec::new();
        for t in 0..tiles {
            let (lo, hi, b) = (reg(&mut f, 4), reg(&mut f, 4), reg(&mut f, 8));
            insts.push(MInst::new(Opc::DsLoadB128, [lo], [tid]).with_imm(32 * t as i64));
            insts.push(MInst::new(Opc::DsLoadB128, [hi], [tid]).with_imm(32 * t as i64 + 16));
            insts.push(MInst::new(Opc::Copy, [half(b, 0)], [lo]));
            insts.push(MInst::new(Opc::Copy, [half(b, 1)], [hi]));
            bs.push(b);
        }
        if let Some(make) = between {
            insts.push(make(accs[0]));
        }
        for (&acc, &b) in accs.iter().zip(&bs) {
            insts.push(MInst::new(Opc::VWmmaF32F16, [acc], [a, b, acc]));
        }
        for &acc in &accs {
            insts.push(MInst::new(Opc::GStoreB128, [], [tid, half(acc, 0)]));
            insts.push(MInst::new(Opc::GStoreB128, [], [tid, half(acc, 1)]));
        }
        insts.push(MInst::new(Opc::SEndpgm, [], []));
        f.blocks.push(MBlock { insts, loop_depth: 0 });
        f
    }

    fn wmma_positions(f: &MFunc) -> Vec<usize> {
        f.blocks[0].insts.iter().enumerate().filter(|(_, i)| i.op.flags() & F_WMMA != 0).map(|(k, _)| k).collect()
    }

    /// Sixteen 8-dword accumulators and sixteen B fragments live at once do
    /// not fit 256 VGPRs; with each WMMA right below its fragment's copies
    /// they do, and the stores see the same values.
    #[test]
    fn wmmas_move_to_their_fragments_when_program_order_does_not_fit() {
        let f = matrix_loop(16, None);
        assert!(regalloc::pressure(&f).unwrap().vgprs > regalloc::MAX_VGPRS);
        let mut g = f.clone();
        fit(&mut g);
        assert!(regalloc::pressure(&g).unwrap().vgprs <= regalloc::MAX_VGPRS);
        // Entry: A, 16 accumulators; then per tile two loads, two copies, the WMMA.
        let want: Vec<usize> = (0..16).map(|t| 17 + 5 * t + 4).collect();
        assert_eq!(wmma_positions(&g), want);
        assert_eq!(effects(&g, 1), effects(&f, 1));
    }

    /// A kernel that fits stays as written, and a WMMA does not pass an
    /// instruction reading the accumulator it overwrites.
    #[test]
    fn fitting_kernels_and_readers_of_the_accumulator_stay() {
        let f = matrix_loop(8, None);
        let mut g = f.clone();
        fit(&mut g);
        assert_eq!(wmma_positions(&g), wmma_positions(&f));
        let reads = |acc: Opnd| MInst::new(Opc::GStoreB128, [], [Opnd::R(VRef { reg: 0, off: 0, n: 1 }), half(acc, 0)]);
        let f = matrix_loop(16, Some(reads));
        let mut g = f.clone();
        hoist_mma(&mut g);
        let insts = &g.blocks[0].insts;
        let reader = insts.iter().position(|i| i.op == Opc::GStoreB128).unwrap();
        let first = insts.iter().position(|i| i.op.flags() & F_WMMA != 0 && matches!(i.defs[0], Opnd::R(v) if v.reg == 2));
        assert_eq!(first, Some(reader + 1));
        assert_eq!(effects(&g, 1), effects(&f, 1));
    }
}
