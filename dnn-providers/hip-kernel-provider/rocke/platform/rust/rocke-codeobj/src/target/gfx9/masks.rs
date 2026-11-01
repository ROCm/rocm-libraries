// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Uniform compares read as wave64 masks, evaluated once per loop block on
//! gfx950.
//!
//! Instruction selection keeps a uniform compare unevaluated and evaluates it
//! at each use (`isel::mask`): every lane-mask reader of it gets its own
//! `s_cmp` and `s_cselect_b64 m, -1, 0`, even when the middle end hoisted
//! the compare out of the loop. The attention backward kernels' row tests
//! (`0 <= kv length`) are evaluated eight times per iteration that way. gfx11
//! reuses such masks for address guards (`rir::uniform_masks`); this is the
//! wave64 form, for every reader: within a loop block, a later pair that
//! evaluates the same compare of the same values while SCC is dead after it
//! reads the first pair's mask instead. `ROCKE_DIRECT_DEBUG=nomasks64` keeps
//! them all.
//!
//! Kernels with MFMAs keep them too: there the pairs issue while the MFMAs
//! run, and without them the scheduler puts the selects right behind their
//! loads.

use super::agpr::is_mfma;
use crate::rir::mir::*;

/// A uniform compare of `a` and `b` turned into a lane mask: what two equal
/// evaluations share.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Key {
    op: Opc,
    a: Opnd,
    b: Opnd,
}

/// Evaluations of one compare in a block, the first included, before it is
/// shared, as `rir::uniform_masks` counts them.
const MIN_EVALUATIONS: usize = 3;

/// The key of `cmp` and `sel` and the mask `sel` writes, when `cmp` is a
/// scalar compare of SGPRs or constants that only sets SCC and `sel` is
/// `s_cselect_b64 m, -1, 0` into a two-dword SGPR with no other definition.
fn evaluation(cmp: &MInst, sel: &MInst, vregs: &[VRegInfo], defs: &[u32]) -> Option<(Key, VRef)> {
    let plain = |o: &Opnd| match o {
        Opnd::R(v) => vregs[v.reg as usize].class == Class::S,
        Opnd::Imm(_) => true,
        Opnd::P(..) => false,
    };
    if cmp.op.fmt() != Fmt::Sopc || !cmp.defs.is_empty() || cmp.uses.len() != 2 || !cmp.uses.iter().all(plain) {
        return None;
    }
    let (Opc::SCselectB64, &[Opnd::R(m)], &[Opnd::Imm(u32::MAX), Opnd::Imm(0)]) = (sel.op, &sel.defs[..], &sel.uses[..]) else {
        return None;
    };
    let info = vregs[m.reg as usize];
    if info.class != Class::S || info.size != 2 || m.off != 0 || m.n != 2 || defs[m.reg as usize] != 1 {
        return None;
    }
    Some((Key { op: cmp.op, a: cmp.uses[0], b: cmp.uses[1] }, m))
}

/// Whether the SCC that the instruction before `insts[from]` leaves is
/// overwritten before anything reads it, within the block.
fn scc_dead(insts: &[MInst], from: usize) -> bool {
    for inst in &insts[from..] {
        let flags = inst.op.flags();
        if flags & F_USE_SCC != 0 {
            return false;
        }
        if flags & F_DEF_SCC != 0 {
            return true;
        }
    }
    false
}

/// Whether `inst` writes a dword of a register `key` reads.
fn redefines(inst: &MInst, key: &Key) -> bool {
    inst.defs.iter().any(|d| {
        let Opnd::R(d) = d else { return false };
        [key.a, key.b].iter().any(|o| matches!(o, Opnd::R(v) if v.reg == d.reg && d.off < v.off + v.n && v.off < d.off + d.n))
    })
}

/// The evaluations of a compare in a block while its operands keep their
/// values: how many, the first one's mask, and the later ones that may read
/// it instead, with their positions and masks.
struct Evaluations {
    key: Key,
    count: usize,
    first: VRef,
    later: Vec<(usize, VRef)>,
}

/// Marks the later evaluations of `e` dead and renames their masks to the
/// first's when there are at least `MIN_EVALUATIONS` in all.
fn settle(e: Evaluations, dead: &mut [bool], renamed: &mut [Option<VRef>]) -> usize {
    if e.count < MIN_EVALUATIONS {
        return 0;
    }
    for &(i, m) in &e.later {
        (dead[i], dead[i + 1]) = (true, true);
        renamed[m.reg as usize] = Some(e.first);
    }
    e.later.len()
}

/// Drops each `s_cmp` + `s_cselect_b64` pair that repeats an earlier one of
/// its loop block with the same operands while they still hold the same
/// values and SCC is dead after it, and makes the readers of its mask read
/// the earlier one's, for compares evaluated `MIN_EVALUATIONS` times or more,
/// in a kernel without MFMAs. Returns the pairs dropped.
pub fn reuse(f: &mut MFunc) -> usize {
    if crate::rir::debug_flag("nomasks64") || f.blocks.iter().flat_map(|b| &b.insts).any(|i| is_mfma(i.op)) {
        return 0;
    }
    let n = f.vregs.len();
    let mut defs = vec![0u32; n];
    for inst in f.blocks.iter().flat_map(|b| b.insts.iter()) {
        for o in &inst.defs {
            if let Opnd::R(v) = o {
                defs[v.reg as usize] += 1;
            }
        }
    }
    let mut renamed: Vec<Option<VRef>> = vec![None; n];
    let mut dropped = 0;
    let vregs = &f.vregs;
    for block in f.blocks.iter_mut().filter(|b| b.loop_depth > 0) {
        let insts = &mut block.insts;
        let mut dead = vec![false; insts.len()];
        let mut open: Vec<Evaluations> = Vec::new();
        for i in 0..insts.len() {
            if i + 1 < insts.len() {
                if let Some((key, m)) = evaluation(&insts[i], &insts[i + 1], vregs, &defs) {
                    match open.iter_mut().find(|e| e.key == key) {
                        Some(e) => {
                            e.count += 1;
                            if scc_dead(insts, i + 2) {
                                e.later.push((i, m));
                            }
                        }
                        None => open.push(Evaluations { key, count: 1, first: m, later: Vec::new() }),
                    }
                }
            }
            if !insts[i].defs.is_empty() {
                let inst = &insts[i];
                for e in open.extract_if(.., |e| redefines(inst, &e.key)) {
                    dropped += settle(e, &mut dead, &mut renamed);
                }
            }
        }
        for e in open {
            dropped += settle(e, &mut dead, &mut renamed);
        }
        if dead.iter().any(|&d| d) {
            let mut k = 0;
            insts.retain(|_| {
                k += 1;
                !dead[k - 1]
            });
        }
    }
    if dropped > 0 {
        for inst in f.blocks.iter_mut().flat_map(|b| b.insts.iter_mut()) {
            for o in inst.uses.iter_mut() {
                if let Opnd::R(v) = o {
                    if let Some(first) = renamed[v.reg as usize] {
                        *o = Opnd::R(VRef { reg: first.reg, off: first.off + v.off, n: v.n });
                    }
                }
            }
        }
    }
    dropped
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::gfx9::gen_isa::Gfx9Op;

    fn s(reg: u32, n: u8) -> Opnd {
        Opnd::R(VRef { reg, off: 0, n })
    }

    /// `s_cmp_le_i32 0, s1` and `s_cselect_b64 m, -1, 0`.
    fn test(m: u32) -> [MInst; 2] {
        [
            MInst::new(Opc::SCmpLeI32, [], [Opnd::Imm(0), s(1, 1)]),
            MInst::new(Opc::SCselectB64, [s(m, 2)], [Opnd::Imm(u32::MAX), Opnd::Imm(0)]),
        ]
    }

    /// `acc &= m` into a new mask.
    fn and(d: u32, acc: u32, m: u32) -> MInst {
        MInst::new(Opc::SAndB64, [s(d, 2)], [s(acc, 2), s(m, 2)])
    }

    /// Registers 0..40 are SGPRs (1..9 one dword, the rest pairs); one block at `depth` that ends as a loop
    /// latch does, by setting SCC and branching on it.
    fn func(mut insts: Vec<MInst>, depth: u32) -> MFunc {
        insts.push(MInst::new(Opc::SAddI32, [s(9, 1)], [s(9, 1), Opnd::Imm(1)]));
        insts.push(MInst::new(Opc::SCbranchScc1, [], []));
        let mut vregs = vec![VRegInfo { class: Class::S, size: 1, align: 1, fixed: None }; 10];
        vregs.extend(vec![VRegInfo { class: Class::S, size: 2, align: 2, fixed: None }; 30]);
        MFunc { blocks: vec![MBlock { insts, loop_depth: depth }], vregs, ..Default::default() }
    }

    /// `n` row masks, each its own evaluation of `0 <= s1`, each ANDed into its own mask.
    fn rows(n: u32) -> Vec<MInst> {
        (0..n).flat_map(|k| test(10 + k).into_iter().chain([and(20 + k, 30, 10 + k)])).collect()
    }

    fn masks_read(f: &MFunc) -> Vec<Opnd> {
        f.blocks[0].insts.iter().filter(|i| i.op == Opc::SAndB64).map(|i| i.uses[1]).collect()
    }

    #[test]
    fn a_compare_read_as_eight_masks_is_evaluated_once() {
        let mut f = func(rows(8), 1);
        assert_eq!(reuse(&mut f), 7);
        assert_eq!(f.blocks[0].insts.iter().filter(|i| i.op == Opc::SCmpLeI32).count(), 1);
        assert_eq!(masks_read(&f), [s(10, 2); 8]);
    }

    #[test]
    fn two_evaluations_or_a_straight_line_block_stay() {
        for (n, depth) in [(2, 1), (8, 0)] {
            let mut f = func(rows(n), depth);
            assert_eq!(reuse(&mut f), 0);
            assert_eq!(f.blocks[0].insts.len(), 3 * n as usize + 2);
        }
    }

    #[test]
    fn a_redefined_operand_or_a_read_scc_keeps_the_evaluation() {
        // s1 changes after the third evaluation: the three before share one, the three after another.
        let mut insts = rows(3);
        insts.push(MInst::new(Opc::SAddI32, [s(1, 1)], [s(1, 1), Opnd::Imm(1)]));
        insts.extend((3..6).flat_map(|k| test(10 + k).into_iter().chain([and(20 + k, 30, 10 + k)])));
        let mut f = func(insts, 1);
        assert_eq!(reuse(&mut f), 4);
        assert_eq!(f.blocks[0].insts.iter().filter(|i| i.op == Opc::SCmpLeI32).count(), 2);
        // A branch reading the SCC of the third pair keeps that pair.
        let mut insts = rows(3);
        insts.insert(8, MInst::new(Opc::SCbranchScc1, [], []));
        let mut f = func(insts, 1);
        assert_eq!(reuse(&mut f), 1);
    }

    #[test]
    fn a_kernel_with_an_mfma_keeps_every_evaluation() {
        let mut insts = rows(8);
        insts.push(MInst::new(Opc::Gfx9(Gfx9Op::VMfma16x16x32F16), [s(30, 2)], [s(31, 2), s(32, 2), s(30, 2)]));
        let mut f = func(insts, 1);
        assert_eq!(reuse(&mut f), 0);
        assert_eq!(f.blocks[0].insts.iter().filter(|i| i.op == Opc::SCmpLeI32).count(), 8);
    }
}
