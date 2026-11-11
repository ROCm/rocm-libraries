// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! rustgemm lever `licm`: loop-invariant constant moves leave single-block loops.
//!
//! Instruction selection materializes a constant where it is read, so a loop
//! whose body rebuilds a register tuple every trip (rocKE's per-tile buffer
//! descriptor of the canonical MXFP6 GEMM: `s_mov_b32 s74, 0x7ffffff0` and
//! `s_mov_b32 s75, 0x31014000` twice per K tile) re-executes the same moves
//! every trip. After allocation and before `s_waitcnt`, for a loop that is one
//! block branching to itself and entered from one preheader that only falls
//! (or branches) into it: a register (an SGPR, or a VGPR when nothing in the
//! loop writes EXEC) that every instruction of the loop writing it writes with
//! the same `s_mov_b32` / `v_mov_b32` of one constant, and that no instruction
//! of the loop reads before the first such move, gets the move once at the end
//! of the preheader and none in the loop.
//!
//! Exact: the loop is a do-while block, so it runs at least once; every read in
//! the loop came after a move of the constant in the same trip and still reads
//! the constant; the value at the loop's exit is the constant either way; the
//! preheader's own value of the register was dead (nothing in the loop reads it
//! before the move, and the preheader's only successor is the loop). A VGPR
//! move writes the lanes EXEC enables, and EXEC is the same in the preheader's
//! last instruction and through the loop. Only moves are removed.

use super::hazards::{fact, regs, Regs, EXEC};
use crate::rir::mir::*;

/// The register and constant of a hoistable move: `s_mov_b32 sN, imm` or `v_mov_b32 vN, imm` without modifiers.
fn const_move(inst: &MInst) -> Option<(Regs, u32, bool)> {
    if !matches!(inst.op, Opc::SMovB32 | Opc::VMovB32) || inst.mods != 0 || inst.defs.len() != 1 || inst.uses.len() != 1 {
        return None;
    }
    let Opnd::Imm(k) = inst.uses[0] else { return None };
    let r = regs(&inst.defs[0])?;
    let vector = inst.op == Opc::VMovB32;
    let plain = if vector { r.is_vgpr() } else { r.is_sgpr() && r.start < 256 + 102 };
    (plain && r.n == 1).then_some((r, k, vector))
}

/// Hoists the loop-invariant constant moves of every single-block loop; returns how many moves were removed from
/// loops (one move per hoisted register is added to its preheader).
pub fn run(f: &mut MFunc) -> usize {
    let preds = super::hazards::predecessors(f);
    let mut removed = 0;
    for b in 0..f.blocks.len() {
        let Some(last) = f.blocks[b].insts.iter().rev().find(|i| i.is_branch()) else { continue };
        if last.imm as usize != b || f.blocks[b].loop_depth == 0 {
            continue;
        }
        let entries: Vec<usize> = preds[b].iter().copied().filter(|&p| p != b).collect();
        let [p] = entries[..] else { continue };
        if f.successors(p) != [b] {
            continue;
        }
        removed += hoist(f, p, b);
    }
    removed
}

fn hoist(f: &mut MFunc, p: usize, b: usize) -> usize {
    let body = &f.blocks[b].insts;
    let writes_exec = body.iter().any(|i| i.op.flags() & F_DEF_EXEC != 0 || fact(i).defs.iter().any(|d| d.overlaps(EXEC)));
    // Candidates: (register, constant, vector, index of the first move); poisoned candidates are dropped.
    let mut cand: Vec<(Regs, u32, bool, usize)> = Vec::new();
    let mut bad: Vec<Regs> = Vec::new();
    for (i, inst) in body.iter().enumerate() {
        let fa = fact(inst);
        let mv = const_move(inst).filter(|_| f.lever(inst, "licm"));
        // A read of a register before its first move in the body reads the preheader's (or last trip's) value.
        for u in &fa.uses {
            if !cand.iter().any(|c| c.0.overlaps(*u)) && !bad.iter().any(|x| x.overlaps(*u)) {
                bad.push(*u);
            }
        }
        for d in &fa.defs {
            match mv {
                Some((r, k, v)) if r == *d => {
                    if bad.iter().any(|x| x.overlaps(r)) {
                        continue;
                    }
                    match cand.iter().position(|c| c.0.overlaps(r)) {
                        Some(j) if cand[j].0 == r && cand[j].1 == k && cand[j].2 == v => {}
                        Some(j) => {
                            bad.push(cand[j].0);
                            cand.remove(j);
                            bad.push(r);
                        }
                        None => cand.push((r, k, v, i)),
                    }
                }
                _ => {
                    if let Some(j) = cand.iter().position(|c| c.0.overlaps(*d)) {
                        bad.push(cand[j].0);
                        cand.remove(j);
                    }
                    bad.push(*d);
                }
            }
        }
    }
    cand.retain(|c| !bad.iter().any(|x| x.overlaps(c.0)) && !(c.2 && writes_exec));
    if cand.is_empty() {
        return 0;
    }
    let mut moved: Vec<MInst> = Vec::new();
    let mut removed = 0;
    let mut keep = Vec::with_capacity(f.blocks[b].insts.len());
    for inst in f.blocks[b].insts.drain(..) {
        if let Some((r, k, v)) = const_move(&inst) {
            if cand.iter().any(|c| c.0 == r && c.1 == k && c.2 == v) {
                if !moved.iter().any(|m| const_move(m).is_some_and(|(mr, _, _)| mr == r)) {
                    moved.push(inst);
                }
                removed += 1;
                continue;
            }
        }
        keep.push(inst);
    }
    f.blocks[b].insts = keep;
    let pre = &mut f.blocks[p].insts;
    let at = pre.iter().rposition(|i| !i.is_branch()).map_or(0, |i| i + 1);
    for (j, m) in moved.into_iter().enumerate() {
        pre.insert(at + j, m);
    }
    removed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(r: u16) -> Opnd {
        Opnd::P(Phys::S(r), 1)
    }

    fn v(r: u16) -> Opnd {
        Opnd::P(Phys::V(r), 1)
    }

    fn mov(d: Opnd, k: u32) -> MInst {
        let op = if matches!(d, Opnd::P(Phys::V(_), _)) { Opc::VMovB32 } else { Opc::SMovB32 };
        MInst::new(op, [d], [Opnd::Imm(k)])
    }

    /// bb0 (preheader) -> bb1 (self loop, `body` then the counter and back branch) -> bb2 (end).
    fn func(body: Vec<MInst>) -> MFunc {
        let mut f = MFunc::default();
        let mut pre = MBlock::default();
        pre.insts.push(MInst::new(Opc::SMovB32, [s(0)], [Opnd::Imm(0)]));
        let mut lp = MBlock { loop_depth: 1, ..Default::default() };
        lp.insts = body;
        lp.insts.push(MInst::new(Opc::SAddI32, [s(0)], [s(0), Opnd::Imm(1)]));
        lp.insts.push(MInst::new(Opc::SCmpLtI32, [], [s(0), Opnd::Imm(8)]));
        lp.insts.push(MInst::new(Opc::SCbranchScc1, [], []).with_imm(1));
        let mut end = MBlock::default();
        end.insts.push(MInst::new(Opc::SEndpgm, [], []));
        f.blocks = vec![pre, lp, end];
        f
    }

    fn with_lever<T>(body: impl FnOnce() -> T) -> T {
        crate::rir::with_levers(&["licm"], body)
    }

    fn rsrc_use(base: u16) -> MInst {
        MInst::new(Opc::SAddI32, [s(90)], [Opnd::P(Phys::S(base), 1), s(base + 1)])
    }

    #[test]
    fn hoists_the_descriptor_constants_rebuilt_every_tile() {
        let body = vec![
            mov(s(74), 0x7fff_fff0),
            mov(s(75), 0x3101_4000),
            rsrc_use(74),
            mov(s(74), 0x7fff_fff0),
            mov(s(75), 0x3101_4000),
            rsrc_use(74),
        ];
        let mut f = func(body);
        assert_eq!(with_lever(|| run(&mut f)), 4);
        assert_eq!(f.blocks[1].insts.len(), 5);
        let pre: Vec<_> = f.blocks[0].insts.iter().filter_map(const_move).map(|(r, k, _)| (r.start - 256, k)).collect();
        assert_eq!(pre, vec![(0, 0), (74, 0x7fff_fff0), (75, 0x3101_4000)]);
    }

    #[test]
    fn keeps_a_register_read_before_its_first_move() {
        let read74 = MInst::new(Opc::SAddI32, [s(90)], [s(74), Opnd::Imm(1)]);
        let mut f = func(vec![read74, mov(s(74), 7), mov(s(75), 9), rsrc_use(74)]);
        // s74 is read first (the preheader's value on trip 1): kept; s75 is read only after its move: hoisted.
        assert_eq!(with_lever(|| run(&mut f)), 1);
        assert!(f.blocks[1].insts.iter().any(|i| const_move(i).is_some_and(|(r, _, _)| r.start == 256 + 74)));
        assert!(!f.blocks[1].insts.iter().any(|i| const_move(i).is_some_and(|(r, _, _)| r.start == 256 + 75)));
    }

    #[test]
    fn keeps_registers_written_with_two_constants_or_by_other_instructions() {
        let mut f = func(vec![mov(s(74), 7), rsrc_use(74), mov(s(74), 8), rsrc_use(74)]);
        assert_eq!(with_lever(|| run(&mut f)), 0);
        let mut f = func(vec![mov(s(75), 7), rsrc_use(74), MInst::new(Opc::SAddI32, [s(75)], [s(75), Opnd::Imm(1)])]);
        assert_eq!(with_lever(|| run(&mut f)), 0);
        // A tuple write overlapping the register poisons it too.
        let load = MInst::new(Opc::SLoadB64, [Opnd::P(Phys::S(74), 2)], [Opnd::P(Phys::S(0), 2)]);
        let mut f = func(vec![mov(s(75), 7), rsrc_use(74), load]);
        assert_eq!(with_lever(|| run(&mut f)), 0);
    }

    #[test]
    fn keeps_vgpr_moves_when_the_loop_writes_exec_and_needs_a_single_entry() {
        let use_v = MInst::new(Opc::VAddNcU32, [v(3)], [v(1), v(2)]);
        let exec = MInst::new(Opc::SMovB64, [Opnd::P(Phys::ExecLo, 2)], [Opnd::Imm(0)]);
        let mut f = func(vec![mov(v(1), 5), use_v.clone(), exec]);
        assert_eq!(with_lever(|| run(&mut f)), 0);
        let mut f = func(vec![mov(v(1), 5), use_v.clone()]);
        assert_eq!(with_lever(|| run(&mut f)), 1);
        // A preheader that can skip the loop (two successors): the move would run on the skip path.
        let mut f = func(vec![mov(v(1), 5), use_v]);
        f.blocks[0].insts.push(MInst::new(Opc::SCbranchScc0, [], []).with_imm(2));
        assert_eq!(with_lever(|| run(&mut f)), 0);
    }

    #[test]
    fn does_nothing_without_the_lever() {
        let mut f = func(vec![mov(s(74), 7), rsrc_use(74), mov(s(74), 7), rsrc_use(74)]);
        assert_eq!(run(&mut f), 0);
    }
}
