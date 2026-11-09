// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! rustgemm lever `m0fill`: fill H12's wait state (SALU writes M0 -> LDS DMA,
//! one wait state) with an independent instruction instead of an `s_nop`.
//!
//! rocKE pins every other instruction to an MFMA slot (`sched_barrier(0)`
//! after each MFMA), so the move of an LDS DMA's M0 lands right before the
//! DMA and the hazard pass puts an `s_nop` between them: one extra
//! instruction per DMA (42 per loop trip in the MXFP6 GEMM). LLVM's code
//! has the move one instruction earlier. After allocation and `s_waitcnt`
//! (physical registers, final waits), before the hazard `s_nop`s: a
//! `s_mov_b32 m0, x` directly before a DMA moves above the instruction
//! before it when that one is an MFMA, an LDS access that is no DMA, a VALU
//! or a SALU instruction, reads and writes neither M0 nor anything the move
//! reads, and is no wait, barrier, branch or EXEC write. The move reads an
//! SGPR and writes M0 only, so the swap changes no value.

use super::hazards::{fact, regs, Kind, M0};
use super::lds_dma::is_dma;
use crate::rir::mir::*;

fn m0_move(inst: &MInst) -> bool {
    inst.op == Opc::SMovB32 && inst.defs.len() == 1 && regs(&inst.defs[0]).is_some_and(|r| r.overlaps(M0))
}

fn may_cross(p: &MInst, mov: &MInst) -> bool {
    let flags = p.op.flags();
    if flags & (F_BRANCH | F_END | F_DEF_EXEC | F_FENCE | F_USE_VCC) != 0 || is_dma(p.op) {
        return false;
    }
    if matches!(p.op, Opc::SNop | Opc::SBarrier | Opc::SchedBarrier | Opc::SSendmsg) || m0_move(p) {
        return false;
    }
    let f = fact(p);
    let kind_ok = f.mfma.is_some() || matches!(f.kind, Kind::Valu | Kind::Lds | Kind::Salu);
    if !kind_ok || f.m0_reader || f.wait_states != 1 {
        return false;
    }
    let mv = fact(mov);
    let touches_m0 = f.defs.iter().chain(f.uses.iter()).any(|r| r.overlaps(M0));
    let clobbers_src = f.defs.iter().any(|d| mv.uses.iter().any(|u| u.overlaps(*d)));
    !touches_m0 && !clobbers_src
}

/// Moves M0 writes above one independent instruction where an LDS DMA follows them directly; returns how many moved.
pub fn run(f: &mut MFunc) -> usize {
    let mut moved = 0;
    for block in &mut f.blocks {
        let insts = &mut block.insts;
        let mut i = 2;
        while i < insts.len() {
            if is_dma(insts[i].op) && m0_move(&insts[i - 1]) && may_cross(&insts[i - 2], &insts[i - 1]) {
                insts.swap(i - 2, i - 1);
                moved += 1;
            }
            i += 1;
        }
    }
    moved
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(r: u16) -> Opnd {
        Opnd::P(Phys::S(r), 1)
    }

    #[test]
    fn moves_above_an_independent_salu_and_not_above_a_writer_of_its_source() {
        let mov = MInst::new(Opc::SMovB32, [Opnd::P(Phys::M0, 1)], [s(10)]);
        let add = MInst::new(Opc::SAddI32, [s(11)], [s(12), Opnd::Imm(1)]);
        assert!(may_cross(&add, &mov));
        let def_src = MInst::new(Opc::SAddI32, [s(10)], [s(12), Opnd::Imm(1)]);
        assert!(!may_cross(&def_src, &mov));
        assert!(!may_cross(&mov.clone(), &mov));
        assert!(!may_cross(&MInst::new(Opc::SNop, [], []), &mov));
    }
}
