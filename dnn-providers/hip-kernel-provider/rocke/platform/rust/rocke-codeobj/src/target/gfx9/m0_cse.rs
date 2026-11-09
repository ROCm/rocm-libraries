// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! LDS DMA selection writes M0 right before every DMA load
//! (`lds_dma::select`), so DMA loads that share their LDS base and differ
//! only in their instruction offset (`offset`) each get a move of the same
//! value; allocation may give each its own SGPR copy of it.
//!
//! With the kernel's `m0_cse = b:true` (`MFunc::passes`), after allocation
//! and `s_waitcnt` and before the hazard `s_nop`s: within a block, a
//! `s_mov_b32 m0, x` goes when M0 already holds `x`'s value. Values are
//! numbered per block: an immediate is its own value, `s_mov_b32 y, x` gives
//! `y` the value of `x`, and any other write of a register gives it a new
//! one; a branch, the end, a write of EXEC or a message forgets them all.
//! Every DMA load still reads the value it read before, and each move that
//! goes takes H12's wait state with it.

use super::hazards::{fact, regs, Regs, M0};
use crate::rir::fx::FxHashMap;
use crate::rir::mir::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Val {
    Imm(u32),
    Id(u32),
}

/// The values the SGPRs and M0 of one block hold, by register.
#[derive(Default)]
struct Numbering {
    vals: FxHashMap<u16, Val>,
    next: u32,
}

impl Numbering {
    /// The value of one-dword operand `o`, numbering it on first sight.
    fn of(&mut self, o: &Opnd) -> Option<Val> {
        if let Opnd::Imm(i) = *o {
            return Some(Val::Imm(i));
        }
        let r = regs(o).filter(|r| r.n == 1)?;
        let next = &mut self.next;
        Some(*self.vals.entry(r.start).or_insert_with(|| {
            *next += 1;
            Val::Id(*next)
        }))
    }

    fn forget(&mut self, r: Regs) {
        self.vals.retain(|&k, _| !r.overlaps(Regs { start: k, n: 1 }));
    }
}

/// Removes the moves the module documentation says; returns how many went.
pub fn run(f: &mut MFunc) -> usize {
    if !f.passes.m0_cse {
        return 0;
    }
    let mut removed = 0;
    for block in &mut f.blocks {
        let mut known = Numbering::default();
        let mut keep = Vec::with_capacity(block.insts.len());
        for inst in block.insts.drain(..) {
            let dst = (inst.op == Opc::SMovB32 && inst.defs.len() == 1 && inst.uses.len() == 1)
                .then(|| regs(&inst.defs[0]))
                .flatten()
                .filter(|r| r.n == 1);
            if let Some(dst) = dst {
                let v = known.of(&inst.uses[0]);
                if dst == M0 && v.is_some() && known.vals.get(&M0.start).copied() == v {
                    removed += 1;
                    continue;
                }
                known.forget(dst);
                if let Some(v) = v {
                    known.vals.insert(dst.start, v);
                }
                keep.push(inst);
                continue;
            }
            if inst.op.flags() & (F_BRANCH | F_END | F_DEF_EXEC) != 0 || inst.op == Opc::SSendmsg {
                known.vals.clear();
            }
            for &d in fact(&inst).defs.iter() {
                known.forget(d);
            }
            keep.push(inst);
        }
        block.insts = keep;
    }
    removed
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

    fn mov(dst: Opnd, src: Opnd) -> MInst {
        MInst::new(Opc::SMovB32, [dst], [src])
    }

    fn mov_m0(src: Opnd) -> MInst {
        mov(Opnd::P(Phys::M0, 1), src)
    }

    fn dma(offset: i64) -> MInst {
        MInst::new(Opc::Gfx9(Gfx9Op::BufLoadLdsB128), [], [Opnd::P(Phys::V(1), 1), Opnd::P(Phys::S(4), 4), s(9)]).with_imm(offset)
    }

    fn func(insts: Vec<MInst>, on: bool) -> MFunc {
        let mut f = MFunc { target: Target::Gfx950, ..MFunc::default() };
        f.passes.m0_cse = on;
        f.blocks.push(MBlock { insts, loop_depth: 0 });
        f
    }

    #[test]
    fn repeated_moves_go_and_moves_after_a_write_of_their_source_or_m0_stay() {
        let insts = vec![
            mov_m0(s(14)),
            MInst::new(Opc::SAddI32, [s(20)], [s(21), Opnd::Imm(1)]),
            mov_m0(s(14)), // same value: goes
            MInst::new(Opc::SAddI32, [s(14)], [s(14), Opnd::Imm(4)]),
            mov_m0(s(14)), // source written in between: stays
            mov_m0(s(30)), // another value: stays
            mov_m0(s(14)), // M0 written in between: stays
            mov_m0(Opnd::Imm(64)),
            MInst::new(Opc::SBranch, [], []),
            mov_m0(Opnd::Imm(64)), // past a branch: stays
        ];
        let mut off = func(insts.clone(), false);
        assert_eq!(run(&mut off), 0);
        let mut on = func(insts, true);
        assert_eq!(run(&mut on), 1);
        assert_eq!(on.blocks[0].insts.len(), 9);
        assert_eq!(on.blocks[0].insts[2].op, Opc::SAddI32);
    }

    #[test]
    fn copies_of_the_value_in_m0_count_as_it() {
        let insts = vec![
            mov(s(6), s(4)),
            mov_m0(s(6)),
            mov(s(0), s(4)),
            mov_m0(s(0)), // a copy of the same value: goes
            mov_m0(s(4)), // goes
            mov(s(4), s(5)),
            mov_m0(s(4)), // s4 now holds s5: stays
            mov_m0(s(6)), // stays
        ];
        let mut on = func(insts, true);
        assert_eq!(run(&mut on), 2);
        assert_eq!(on.blocks[0].insts.len(), 6);
    }

    #[test]
    fn dma_loads_sharing_a_base_keep_one_move_and_no_nops() {
        let insts = vec![mov_m0(s(14)), dma(0), mov_m0(s(14)), dma(1024), mov_m0(s(14)), dma(2048)];
        let mut off = func(insts.clone(), false);
        insert_nops(&mut off);
        let nops = |f: &MFunc| f.blocks[0].insts.iter().filter(|i| i.op == Opc::SNop).count();
        assert_eq!(nops(&off), 3);
        let mut on = func(insts, true);
        assert_eq!(run(&mut on), 2);
        insert_nops(&mut on);
        assert_eq!(nops(&on), 1);
        let offsets: Vec<i64> = on.blocks[0].insts.iter().filter(|i| super::super::lds_dma::is_dma(i.op)).map(|i| i.imm).collect();
        assert_eq!(offsets, [0, 1024, 2048]);
    }
}
