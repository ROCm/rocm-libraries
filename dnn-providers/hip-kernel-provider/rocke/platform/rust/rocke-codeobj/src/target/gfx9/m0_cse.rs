// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! rustgemm lever `m0cse`: drop a `s_mov_b32 m0, x` when M0 already holds x.
//!
//! LDS DMA selection writes M0 right before every DMA (`lds_dma::select`), so
//! a group of DMAs that share their LDS base (rocKE's `dtl_imm`: four 1 KiB
//! DTLs differ only in the instruction offset) gets the same move once per
//! DMA: 14 moves per MXFP6 GEMM K tile where 6 distinct values are needed.
//! After allocation and `s_waitcnt`, before the hazard `s_nop`s: within a
//! block, a move to M0 from the same source as the last move to M0 is
//! deleted while nothing in between wrote M0, the source, EXEC, or is a
//! branch / end / message. Only moves are removed; every DMA still reads the
//! value it read before (the move it would have re-executed wrote the same
//! value into M0), and fewer M0 writes only remove H12 wait states.

use super::hazards::{fact, regs, M0};
use crate::rir::mir::*;

fn m0_move_src(inst: &MInst) -> Option<Opnd> {
    if inst.op == Opc::SMovB32 && inst.defs.len() == 1 && regs(&inst.defs[0]).is_some_and(|r| r.overlaps(M0)) {
        inst.uses.first().copied()
    } else {
        None
    }
}

/// Removes redundant moves to M0; returns how many were removed.
pub fn run(f: &mut MFunc) -> usize {
    let mut removed = 0;
    for block in &mut f.blocks {
        let mut known: Option<Opnd> = None;
        let mut keep = Vec::with_capacity(block.insts.len());
        for inst in block.insts.drain(..) {
            if let Some(src) = m0_move_src(&inst) {
                if known == Some(src) {
                    removed += 1;
                    continue;
                }
                known = Some(src);
                keep.push(inst);
                continue;
            }
            let flags = inst.op.flags();
            if flags & (F_BRANCH | F_END | F_DEF_EXEC) != 0 || matches!(inst.op, Opc::SSendmsg) {
                known = None;
            }
            if let Some(src) = known {
                let fa = fact(&inst);
                let src_regs = regs(&src);
                let clobbered = fa
                    .defs
                    .iter()
                    .any(|d| d.overlaps(M0) || src_regs.is_some_and(|s| d.overlaps(s)));
                if clobbered {
                    known = None;
                }
            }
            keep.push(inst);
        }
        block.insts = keep;
    }
    removed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(r: u16) -> Opnd {
        Opnd::P(Phys::S(r), 1)
    }

    fn mov_m0(src: Opnd) -> MInst {
        MInst::new(Opc::SMovB32, [Opnd::P(Phys::M0, 1)], [src])
    }

    #[test]
    fn drops_repeated_moves_and_keeps_moves_after_a_source_write() {
        let mut f = MFunc::default();
        let mut b = MBlock::default();
        b.insts.push(mov_m0(s(14)));
        b.insts.push(MInst::new(Opc::SAddI32, [s(20)], [s(21), Opnd::Imm(1)]));
        b.insts.push(mov_m0(s(14))); // same value: dropped
        b.insts.push(MInst::new(Opc::SAddI32, [s(14)], [s(14), Opnd::Imm(4)]));
        b.insts.push(mov_m0(s(14))); // source rewritten: kept
        b.insts.push(mov_m0(s(30))); // other value: kept
        b.insts.push(mov_m0(s(14))); // M0 changed in between: kept
        f.blocks.push(b);
        assert_eq!(run(&mut f), 1);
        assert_eq!(f.blocks[0].insts.len(), 6);
    }
}
