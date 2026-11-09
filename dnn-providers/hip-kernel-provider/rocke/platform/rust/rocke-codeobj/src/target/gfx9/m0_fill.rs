// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! H12 (a SALU write of M0, then an LDS DMA load reading it) needs one wait
//! state. A kernel that pins every other instruction to an MFMA slot
//! (`sched_barrier` after each MFMA) gets each DMA's M0 write right before
//! the DMA, and the hazard pass puts an `s_nop 0` between them.
//!
//! With the kernel's `m0_fill = b:true` (`MFunc::passes`), after `s_waitcnt`
//! and before the hazard `s_nop`s, a `s_mov_b32 m0, x` directly before a DMA
//! swaps with the instruction before it when that one is an MFMA, a VALU, a
//! SALU or an LDS access other than a DMA that reads and writes neither M0
//! nor `x`, and is no wait, barrier, branch, fence or EXEC or VCC access.
//! The move reads `x` and writes M0 only, so the swap changes no value. It
//! crosses exactly one instruction, where `m0_early` crosses whatever keeps
//! it from its wait state, waits, barriers and `s_setprio` included.

use super::hazards::{fact, regs, Kind, M0};
use super::lds_dma::is_dma;
use crate::rir::mir::*;

fn m0_move(inst: &MInst) -> bool {
    inst.op == Opc::SMovB32 && inst.defs.len() == 1 && regs(&inst.defs[0]).is_some_and(|r| r.overlaps(M0))
}

/// Whether the M0 move `mov` may swap with `p`, the instruction before it.
fn may_cross(p: &MInst, mov: &MInst) -> bool {
    if p.op.flags() & (F_BRANCH | F_END | F_DEF_EXEC | F_USE_VCC | F_FENCE) != 0 || is_dma(p.op) || m0_move(p) {
        return false;
    }
    if matches!(p.op, Opc::SNop | Opc::SBarrier | Opc::SchedBarrier | Opc::SSendmsg) {
        return false;
    }
    let f = fact(p);
    let kind = f.mfma.is_some() || matches!(f.kind, Kind::Valu | Kind::Lds | Kind::Salu);
    if !kind || f.m0_reader || f.wait_states != 1 {
        return false;
    }
    let sources = fact(mov).uses;
    let touches_m0 = f.defs.iter().chain(f.uses.iter()).any(|r| r.overlaps(M0));
    let writes_source = f.defs.iter().any(|d| sources.iter().any(|s| s.overlaps(*d)));
    !touches_m0 && !writes_source
}

/// Swaps the M0 writes as the module documentation says; returns how many
/// moved.
pub fn run(f: &mut MFunc) -> usize {
    if !f.passes.m0_fill {
        return 0;
    }
    let mut moved = 0;
    for block in &mut f.blocks {
        let insts = &mut block.insts;
        for i in 2..insts.len() {
            if is_dma(insts[i].op) && m0_move(&insts[i - 1]) && may_cross(&insts[i - 2], &insts[i - 1]) {
                insts.swap(i - 2, i - 1);
                moved += 1;
            }
        }
    }
    moved
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
        MInst::new(Opc::Gfx9(Gfx9Op::VMfmaScale32x32x64F8f6f4), [v(256 + acc, 16)], [v(20, 6), v(30, 6), v(256 + acc, 16), v(40, 1), v(41, 1)])
            .with_imm(2 | 2 << 7)
    }

    fn func(insts: Vec<MInst>, on: bool) -> MFunc {
        let mut f = MFunc { target: Target::Gfx950, ..MFunc::default() };
        f.passes.m0_fill = on;
        f.blocks.push(MBlock { insts, loop_depth: 0 });
        f
    }

    fn nops(f: &MFunc) -> usize {
        f.blocks[0].insts.iter().filter(|i| i.op == Opc::SNop).count()
    }

    #[test]
    fn the_write_swaps_with_the_mfma_and_its_nop_goes() {
        let insts = vec![dma(1), mfma(0), m0(15), dma(2), mfma(16)];
        let mut off = func(insts.clone(), false);
        assert_eq!(run(&mut off), 0);
        insert_nops(&mut off);
        assert_eq!(nops(&off), 1);
        let mut on = func(insts, true);
        assert_eq!(run(&mut on), 1);
        let ops: Vec<Opc> = on.blocks[0].insts.iter().map(|i| i.op).collect();
        assert_eq!(ops[1..4], [Opc::SMovB32, Opc::Gfx9(Gfx9Op::VMfmaScale32x32x64F8f6f4), Opc::Gfx9(Gfx9Op::BufLoadLdsB128)]);
        insert_nops(&mut on);
        assert_eq!(nops(&on), 0);
    }

    #[test]
    fn it_crosses_one_instruction_and_no_wait_barrier_or_writer_of_its_source() {
        let mov = m0(10);
        assert!(may_cross(&MInst::new(Opc::SAddI32, [s(11)], [s(12), Opnd::Imm(1)]), &mov));
        assert!(may_cross(&MInst::new(Opc::DsLoadB64, [v(50, 2)], [v(60, 1)]), &mov));
        // It reads s10, which this writes.
        assert!(!may_cross(&MInst::new(Opc::SAddI32, [s(10)], [s(12), Opnd::Imm(1)]), &mov));
        assert!(!may_cross(&m0(12), &mov));
        assert!(!may_cross(&dma(3), &mov));
        for op in [Opc::SNop, Opc::SWaitcnt, Opc::SWaitcntExplicit, Opc::SBarrierBare, Opc::SSetprio] {
            assert!(!may_cross(&MInst::new(op, [], []), &mov), "{op:?}");
        }
        // Behind a wait, the write stays (`m0_early` would cross it) and keeps its nop.
        let wait = MInst::new(Opc::SWaitcntExplicit, [], []).with_imm(0x3f70);
        let mut f = func(vec![mfma(0), wait, m0(15), dma(2)], true);
        assert_eq!(run(&mut f), 0);
        insert_nops(&mut f);
        assert_eq!(nops(&f), 1);
    }
}
