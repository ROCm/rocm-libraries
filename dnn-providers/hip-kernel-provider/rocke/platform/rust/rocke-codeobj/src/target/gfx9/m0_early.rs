// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! A SALU write of M0 needs one wait state before the LDS DMA load that
//! reads it (H12). Kernels that place each DMA load right after an MFMA
//! behind a `sched_barrier` get the write between the two, so the hazard
//! pass puts an `s_nop 0` there. hipBLASLt's gfx950 placer instead writes
//! M0 in the MFMA interval before the load (`noM0WithBufferLoad`).
//!
//! With the kernel's `m0_early = b:true` (`MFunc::m0_early`), after register
//! allocation and before the hazard `s_nop`s, each M0 write that has no wait
//! state between it and its reader moves up past the instructions before
//! it until one wait state lies between: past none that reads or writes M0
//! or writes one of its sources, and not out of its block.

use super::hazards::{fact, M0};
use crate::rir::mir::*;

/// Moves M0 writes as the module documentation says; returns how many moved.
pub fn run(f: &mut MFunc) -> usize {
    if !f.m0_early {
        return 0;
    }
    let mut moved = 0;
    for block in &mut f.blocks {
        let mut k = 0;
        while k < block.insts.len() {
            if fact(&block.insts[k]).m0_reader {
                moved += usize::from(hoist_writer(&mut block.insts, k));
            }
            k += 1;
        }
    }
    moved
}

fn touches_m0(inst: &MInst) -> bool {
    let fa = fact(inst);
    fa.m0_reader || fa.defs.iter().chain(fa.uses.iter()).any(|r| r.overlaps(M0))
}

/// For the M0 reader `insts[k]`: the M0 write before it in the block, if no
/// wait state lies between, moved up until one does. Whether it moved.
fn hoist_writer(insts: &mut Vec<MInst>, k: usize) -> bool {
    let mut gap = 0;
    let mut w = None;
    for j in (0..k).rev() {
        let fa = fact(&insts[j]);
        if fa.defs.iter().any(|r| r.overlaps(M0)) {
            w = Some(j);
            break;
        }
        if touches_m0(&insts[j]) {
            return false;
        }
        gap += fa.wait_states;
    }
    let Some(w) = w else { return false };
    if gap > 0 || fact(&insts[w]).kind != super::hazards::Kind::Salu {
        return false;
    }
    let sources = fact(&insts[w]).uses;
    let mut to = w;
    while to > 0 && gap == 0 {
        let p = &insts[to - 1];
        let fp = fact(p);
        if touches_m0(p) || p.op.flags() & (F_BRANCH | F_END) != 0 || fp.defs.iter().any(|d| sources.iter().any(|s| s.overlaps(*d))) {
            break;
        }
        gap += fp.wait_states;
        to -= 1;
    }
    if gap == 0 {
        return false;
    }
    let write = insts.remove(w);
    insts.insert(to, write);
    true
}

#[cfg(test)]
mod tests {
    use super::super::gen_isa::Gfx9Op;
    use super::super::hazards::insert_nops;
    use super::*;
    use crate::target::Target;

    fn s(r: u16) -> Opnd {
        Opnd::P(Phys::S(r), 1)
    }

    fn v(r: u16, n: u8) -> Opnd {
        Opnd::P(Phys::V(r), n)
    }

    fn m0(from: u16) -> MInst {
        MInst::new(Opc::SMovB32, [Opnd::P(Phys::M0, 1)], [s(from)])
    }

    fn dma(voffset: u16) -> MInst {
        MInst::new(Opc::Gfx9(Gfx9Op::BufLoadLdsB128), [], [v(voffset, 1), Opnd::P(Phys::S(4), 4), s(9)]).with_imm(2 << 12)
    }

    fn mfma(acc: u16) -> MInst {
        MInst::new(Opc::Gfx9(Gfx9Op::VMfmaScale32x32x64F8f6f4), [v(256 + acc, 16)], [v(20, 8), v(30, 8), v(256 + acc, 16), v(40, 1), v(41, 1)])
            .with_imm(2 | 2 << 7)
    }

    fn func(insts: Vec<MInst>, on: bool) -> MFunc {
        let mut f = MFunc { target: Target::Gfx950, m0_early: on, ..MFunc::default() };
        f.blocks.push(MBlock { insts, loop_depth: 0 });
        f
    }

    fn nops(f: &MFunc) -> usize {
        f.blocks[0].insts.iter().filter(|i| i.op == Opc::SNop).count()
    }

    #[test]
    fn the_write_moves_above_the_mfma_and_its_nop_goes() {
        let insts = vec![dma(1), mfma(0), m0(15), dma(2), mfma(16)];
        let mut off = func(insts.clone(), false);
        assert_eq!(run(&mut off), 0);
        insert_nops(&mut off);
        assert_eq!(nops(&off), 1);
        let mut on = func(insts, true);
        assert_eq!(run(&mut on), 1);
        let ops: Vec<Opc> = on.blocks[0].insts.iter().map(|i| i.op).collect();
        assert_eq!(ops[..3], [Opc::Gfx9(Gfx9Op::BufLoadLdsB128), Opc::SMovB32, Opc::Gfx9(Gfx9Op::VMfmaScale32x32x64F8f6f4)]);
        insert_nops(&mut on);
        assert_eq!(nops(&on), 0);
    }

    #[test]
    fn a_covered_write_stays() {
        let insts = vec![m0(15), mfma(0), dma(2)];
        let mut f = func(insts.clone(), true);
        assert_eq!(run(&mut f), 0);
        assert_eq!(format!("{:?}", f.blocks[0].insts), format!("{insts:?}"));
    }

    #[test]
    fn nothing_moves_past_an_m0_reader_or_a_def_of_the_source() {
        // Right after the previous DMA load: the write cannot pass it.
        let mut f = func(vec![dma(1), m0(15), dma(2)], true);
        assert_eq!(run(&mut f), 0);
        // The source is written by the instruction before.
        let def = MInst::new(Opc::SAddI32, [s(15)], [s(14), Opnd::Imm(64)]);
        let mut f = func(vec![def, m0(15), dma(2)], true);
        assert_eq!(run(&mut f), 0);
        insert_nops(&mut f);
        assert_eq!(nops(&f), 1);
    }

    #[test]
    fn pseudos_count_no_wait_state() {
        let mut f = func(vec![mfma(0), MInst::new(Opc::SchedBarrier, [], []), m0(15), dma(2)], true);
        assert_eq!(run(&mut f), 1);
        assert_eq!(f.blocks[0].insts[0].op, Opc::SMovB32);
        insert_nops(&mut f);
        assert_eq!(nops(&f), 0);
    }
}
