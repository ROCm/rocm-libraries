// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! SGPR relief for gfx950 schedules. gfx950 hands out 102 SGPRs and a
//! wave64 lane mask takes two. A window that hoists a block's loads also
//! hoists the compares that select between them and their fallbacks, which
//! keeps every mask live from its compare to its select; in the attention
//! backward kernels with a band mask (sixteen rows of LSE and Dsum loads,
//! each masked) no window then fits the SGPRs, and the kernel runs in
//! program order, one load at a time.
//!
//! `sink` takes a window's order of a block and moves each instruction that
//! defines only SGPRs and touches no memory down towards the first
//! instruction in that order that depends on it: a VALU one (a compare)
//! stops two instructions short, which fill the wait states its SGPR result
//! owes a VALU reader (H1), and a SALU one goes right before it. The
//! block's dependence graph has every dependence (registers, SCC, memory,
//! the fences), so the result is an order of the same block with every
//! value defined before it is read: a reordering within blocks, which the
//! liveness every schedule shares stays valid for (`rir::Schedules`).

use crate::rir::mir::*;

/// Whether `inst` may sink: every result is an SGPR (at least one), and it
/// reads or writes no memory and is no fence.
fn sinks(f: &MFunc, inst: &MInst) -> bool {
    let fixed = F_SMEM | F_LDS | F_VMEM_LOAD | F_VMEM_STORE | F_BRANCH | F_END | F_FENCE | F_DEF_EXEC;
    !inst.defs.is_empty()
        && inst.op.flags() & fixed == 0
        && !matches!(inst.op, Opc::SBarrier | Opc::SSendmsg)
        && inst.defs.iter().all(|d| matches!(d, Opnd::R(v) if f.vregs[v.reg as usize].class == Class::S && f.vregs[v.reg as usize].fixed.is_none()))
        && inst.uses.iter().all(|u| !matches!(u, Opnd::P(..)))
}

/// The wait states between a VALU write of an SGPR and a VALU read of it (H1).
const H1: usize = 2;

/// `order`, a valid order of block `block` of `f`, with every instruction
/// that `sinks` moved down towards its first dependent in it (the end if
/// there is none): `H1` instructions short of it for a VALU instruction,
/// right before it for a SALU one, and never up. `succs(i)` gives the
/// instructions that depend on instruction `i` of the block. The last
/// instructions move first, so a chain of them sinks together.
pub fn sink<S: Iterator<Item = usize>>(f: &MFunc, block: usize, order: Vec<u32>, succs: impl Fn(usize) -> S) -> Vec<u32> {
    let insts = &f.blocks[block].insts;
    let n = order.len();
    let mut seq = order;
    let mut pos = vec![0usize; n];
    for (k, &i) in seq.iter().enumerate() {
        pos[i as usize] = k;
    }
    let movable: Vec<u32> = seq.iter().rev().copied().filter(|&i| sinks(f, &insts[i as usize])).collect();
    for x in movable {
        let p = pos[x as usize];
        let q = succs(x as usize).map(|s| pos[s]).min().unwrap_or(n);
        let gap = if insts[x as usize].op.flags() & F_USE_EXEC != 0 { H1 } else { 0 };
        let t = q.saturating_sub(1 + gap).max(p);
        if t == p {
            continue;
        }
        seq[p..=t].rotate_left(1);
        for (k, &i) in seq.iter().enumerate().take(t + 1).skip(p) {
            pos[i as usize] = k;
        }
    }
    seq
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::Target;

    fn r(reg: u32, n: u8) -> Opnd {
        Opnd::R(VRef { reg, off: 0, n })
    }

    /// v0..v3 VGPRs, s4..s7 SGPRs (s6 and s7 lane masks), one block:
    /// two compares hoisted above two loads, each select after its load.
    fn block() -> MFunc {
        let mut f = MFunc { target: Target::Gfx950, ..MFunc::default() };
        for (class, size) in [(Class::V, 2), (Class::V, 1), (Class::V, 1), (Class::V, 1), (Class::S, 1), (Class::S, 1), (Class::S, 2), (Class::S, 2)] {
            f.vregs.push(VRegInfo { class, size, align: 1, fixed: None });
        }
        let load = |d: u32| MInst::new(Opc::GLoadB32, [r(d, 1)], [r(0, 2)]);
        let cmp = |d: u32| MInst::new(Opc::VCmp(0x41), [r(d, 2)], [r(1, 1), r(4, 1)]);
        let sel = |d: u32, m: u32| MInst::new(Opc::VCndmaskB32, [r(d, 1)], [Opnd::Imm(0), r(d, 1), r(m, 2)]);
        f.blocks.push(MBlock { insts: vec![cmp(6), cmp(7), load(2), load(3), sel(2, 6), sel(3, 7)], loop_depth: 0 });
        f
    }

    /// Compare k is read by select k only, and load k by select k.
    fn succs(i: usize) -> std::vec::IntoIter<usize> {
        match i {
            0 | 2 => vec![4],
            1 | 3 => vec![5],
            _ => vec![],
        }
        .into_iter()
    }

    #[test]
    fn lane_masks_sink_to_two_short_of_their_selects() {
        let f = block();
        // Program order is the window's order.
        let order = sink(&f, 0, (0..6).collect(), succs);
        assert_eq!(order, [2, 0, 1, 3, 4, 5]);
        let at = |i: u32| order.iter().position(|&x| x == i).expect("placed");
        // Nothing moves past what reads it; two instructions fill H1 before each select.
        assert!(at(2) < at(4) && at(3) < at(5));
        assert_eq!((at(4) - at(0), at(5) - at(1)), (H1 + 1, H1 + 1));
    }

    #[test]
    fn a_salu_result_sinks_right_before_its_reader() {
        let mut f = block();
        // s5 = s4 + 1 first, read by the second compare.
        f.blocks[0].insts.insert(0, MInst::new(Opc::SAddI32, [r(5, 1)], [r(4, 1), Opnd::Imm(1)]));
        let succs = |i: usize| match i {
            0 => vec![2],
            i => succs(i - 1).map(|j| j + 1).collect(),
        }
        .into_iter();
        let order = sink(&f, 0, (0..7).collect(), succs);
        let at = |i: u32| order.iter().position(|&x| x == i).expect("placed");
        assert_eq!(at(0) + 1, at(2), "{order:?}");
    }

    #[test]
    fn memory_vector_results_and_fences_stay() {
        let f = block();
        let insts = &f.blocks[0].insts;
        assert!(sinks(&f, &insts[0]) && sinks(&f, &insts[1]));
        // A load (memory, a VGPR result) and a select (a VGPR result) stay.
        assert!(!sinks(&f, &insts[2]) && !sinks(&f, &insts[4]));
        let barrier = MInst::new(Opc::SBarrier, [], []);
        assert!(!sinks(&f, &barrier));
        // With nothing depending on them in the block, VALU results end H1
        // short of the end, and none moves up.
        let order = sink(&f, 0, vec![0, 2, 3, 4, 1, 5], |_| std::iter::empty());
        assert_eq!(order, [2, 3, 4, 0, 1, 5]);
    }
}
