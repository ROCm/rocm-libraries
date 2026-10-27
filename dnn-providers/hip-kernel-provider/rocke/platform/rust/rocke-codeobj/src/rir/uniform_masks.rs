// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Uniform compares read as masks, materialized once per block.
//!
//! Instruction selection keeps a uniform compare unevaluated and evaluates
//! it at each use, so that a branch or a scalar select reads it straight
//! from SCC. A select of VGPR values reads it as a lane mask instead, and
//! each such select gets its own `s_cmp` and `s_cselect_b32 m, -1, 0`: a
//! depthwise convolution's row test, read by the address select of every
//! column of the row, is evaluated once per column. Within a loop block, a
//! later pair that evaluates the same compare of the same values reads the
//! mask of the first instead.
//!
//! Only guards of buffer offsets are reused: compares whose every mask is
//! read only by selects of an offset or the out-of-bounds sentinel, as the
//! convolutions' row tests are. Attention's score masks repeat compares too;
//! they are left as they are.

use super::mir::*;

/// The buffer offset that is out of bounds for every descriptor: a guarded
/// access selects it where its test fails.
const OUT_OF_BOUNDS: u32 = 0x7fff_ffff;

/// A uniform compare of `a` and `b` turned into the scalar `t` or `f` by
/// `s_cselect_b32`: what two equal evaluations share.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Key {
    op: Opc,
    a: Opnd,
    b: Opnd,
    t: u32,
    f: u32,
}

/// The key of `cmp`, `sel` and the one-dword SGPR `sel` writes, when `cmp`
/// is a scalar compare of registers or constants that only sets SCC and
/// `sel` turns SCC into one of two constants, with no other definition of
/// that SGPR.
fn evaluation(cmp: &MInst, sel: &MInst, vregs: &[VRegInfo], defs: &[u32]) -> Option<(Key, VRef)> {
    let plain = |o: &Opnd| match o {
        Opnd::R(v) => vregs[v.reg as usize].class == Class::S,
        Opnd::Imm(_) => true,
        Opnd::P(..) => false,
    };
    if cmp.op.fmt() != Fmt::Sopc || !cmp.defs.is_empty() || cmp.uses.len() != 2 || !cmp.uses.iter().all(plain) {
        return None;
    }
    let (Opc::SCselectB32, &[Opnd::R(m)], &[Opnd::Imm(t), Opnd::Imm(f)]) = (sel.op, &sel.defs[..], &sel.uses[..]) else {
        return None;
    };
    let info = vregs[m.reg as usize];
    if info.class != Class::S || info.size != 1 || m.off != 0 || m.n != 1 || defs[m.reg as usize] != 1 {
        return None;
    }
    Some((Key { op: cmp.op, a: cmp.uses[0], b: cmp.uses[1], t, f }, m))
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

/// Evaluations of one compare in a block, the first included, before the
/// rule applies: fewer repeat a test once, not once per column.
const MIN_EVALUATIONS: usize = 3;

/// The evaluations of a compare in a block while its operands keep their
/// values: how many, the first one's SGPR, and the later ones that may read
/// it instead, with their positions and SGPRs.
struct Evaluations {
    key: Key,
    count: usize,
    first: VRef,
    later: Vec<(usize, VRef)>,
}

/// Marks the later evaluations of `e` dead and renames their SGPRs to the
/// first's when there are at least `MIN_EVALUATIONS` in all and each of
/// their SGPRs only guards buffer offsets.
fn settle(e: Evaluations, dead: &mut [bool], renamed: &mut [Option<VRef>], guards: &[bool]) -> usize {
    let guard = |m: VRef| guards[m.reg as usize];
    if e.count < MIN_EVALUATIONS || !guard(e.first) || !e.later.iter().all(|&(_, m)| guard(m)) {
        return 0;
    }
    for &(i, m) in &e.later {
        (dead[i], dead[i + 1]) = (true, true);
        renamed[m.reg as usize] = Some(e.first);
    }
    e.later.len()
}

/// Drops each `s_cmp` + `s_cselect_b32` pair that repeats an earlier one of
/// its loop block with the same operands while they still hold the same
/// values and SCC is dead after it, and makes the readers of its SGPR read
/// the earlier one's, for compares evaluated `MIN_EVALUATIONS` times or more.
/// Returns the pairs dropped. `ROCKE_DIRECT_DEBUG=nomasks` leaves them all.
pub(crate) fn reuse_masks(f: &mut MFunc) -> usize {
    if super::debug_flag("nomasks") {
        return 0;
    }
    let n = f.vregs.len();
    let mut defs = vec![0u32; n];
    // Registers read, all of them, as the mask of a select between a
    // buffer offset and `OUT_OF_BOUNDS`.
    let (mut guards, mut read) = (vec![true; n], vec![false; n]);
    for inst in f.blocks.iter().flat_map(|b| b.insts.iter()) {
        for o in &inst.defs {
            if let Opnd::R(v) = o {
                defs[v.reg as usize] += 1;
            }
        }
        let select = inst.op == Opc::VCndmaskB32 && inst.uses.len() == 3 && inst.uses[..2].contains(&Opnd::Imm(OUT_OF_BOUNDS));
        for (slot, o) in inst.uses.iter().enumerate() {
            if let Opnd::R(v) = o {
                read[v.reg as usize] = true;
                guards[v.reg as usize] &= select && slot == 2;
            }
        }
    }
    for (g, r) in guards.iter_mut().zip(&read) {
        *g &= *r;
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
                let (stale, live): (Vec<_>, Vec<_>) = open.drain(..).partition(|e| redefines(inst, &e.key));
                open = live;
                for e in stale {
                    dropped += settle(e, &mut dead, &mut renamed, &guards);
                }
            }
        }
        for e in open {
            dropped += settle(e, &mut dead, &mut renamed, &guards);
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
                        *o = Opnd::R(first);
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

    fn r(reg: u32) -> Opnd {
        Opnd::R(VRef { reg, off: 0, n: 1 })
    }

    /// `s_cmp_lt_i32 a, b` and `s_cselect_b32 m, -1, 0`.
    fn test(a: u32, b: u32, m: u32) -> [MInst; 2] {
        [
            MInst::new(Opc::SCmpLtI32, [], [r(a), r(b)]),
            MInst::new(Opc::SCselectB32, [r(m)], [Opnd::Imm(u32::MAX), Opnd::Imm(0)]),
        ]
    }

    /// The offset `off`, or out of bounds where `m` fails, into `dst`.
    fn guard(dst: u32, off: u32, m: u32) -> MInst {
        MInst::new(Opc::VCndmaskB32, [r(dst)], [Opnd::Imm(OUT_OF_BOUNDS), r(off), r(m)])
    }

    /// Registers 0..20 are SGPRs, 20.. VGPRs; one block at `depth` that ends
    /// as a loop latch does, by setting SCC and branching on it.
    fn func(mut insts: Vec<MInst>, depth: u32) -> MFunc {
        insts.push(MInst::new(Opc::SAddI32, [r(19)], [r(19), Opnd::Imm(1)]));
        insts.push(MInst::new(Opc::SCbranchScc1, [], []));
        let mut vregs = vec![VRegInfo { class: Class::S, size: 1, align: 1, fixed: None }; 20];
        vregs.extend(vec![VRegInfo { class: Class::V, size: 1, align: 1, fixed: None }; 20]);
        MFunc { blocks: vec![MBlock { insts, loop_depth: depth }], vregs, ..Default::default() }
    }

    /// A row of `n` column guards, each with its own evaluation of `s1 < s2`.
    fn row(n: u32) -> Vec<MInst> {
        (0..n).flat_map(|k| test(1, 2, 3 + k).into_iter().chain([guard(20 + k, 30 + k, 3 + k)])).collect()
    }

    fn masks_read(f: &MFunc) -> Vec<Opnd> {
        f.blocks[0].insts.iter().filter(|i| i.op == Opc::VCndmaskB32).map(|i| i.uses[2]).collect()
    }

    #[test]
    fn a_row_of_guards_evaluates_its_test_once() {
        let mut f = func(row(4), 1);
        assert_eq!(reuse_masks(&mut f), 3);
        assert_eq!(f.blocks[0].insts.iter().filter(|i| i.op == Opc::SCmpLtI32).count(), 1);
        assert_eq!(masks_read(&f), [r(3); 4]);
    }

    #[test]
    fn two_evaluations_or_a_straight_line_block_stay() {
        for (n, depth) in [(2, 1), (4, 0)] {
            let mut f = func(row(n), depth);
            assert_eq!(reuse_masks(&mut f), 0);
            assert_eq!(f.blocks[0].insts.len(), 3 * n as usize + 2);
        }
    }

    #[test]
    fn a_test_whose_mask_is_not_a_guard_stays() {
        // The second select chooses between two values: a score mask, not an address guard.
        let mut insts = row(3);
        insts[5] = MInst::new(Opc::VCndmaskB32, [r(21)], [Opnd::Imm(0), r(31), r(4)]);
        let mut f = func(insts, 1);
        assert_eq!(reuse_masks(&mut f), 0);
    }

    #[test]
    fn a_redefined_operand_starts_a_new_count() {
        // s1 changes after the second evaluation: two before, two after.
        let mut insts = row(2);
        insts.push(MInst::new(Opc::SAddI32, [r(1)], [r(1), Opnd::Imm(1)]));
        insts.extend(row(2).into_iter().map(|mut i| {
            for o in i.defs.iter_mut().chain(i.uses.iter_mut()) {
                if let Opnd::R(v) = o {
                    if (3..20).contains(&v.reg) || v.reg >= 20 {
                        v.reg += 2;
                    }
                }
            }
            i
        }));
        let mut f = func(insts, 1);
        assert_eq!(reuse_masks(&mut f), 0);
        // Three after the change do share one evaluation.
        let mut insts = row(1);
        insts.push(MInst::new(Opc::SAddI32, [r(1)], [r(1), Opnd::Imm(1)]));
        insts.extend(row(3).into_iter().map(|mut i| {
            for o in i.defs.iter_mut().chain(i.uses.iter_mut()) {
                if let Opnd::R(v) = o {
                    if (3..20).contains(&v.reg) || v.reg >= 20 {
                        v.reg += 1;
                    }
                }
            }
            i
        }));
        let mut f = func(insts, 1);
        assert_eq!(reuse_masks(&mut f), 2);
    }

    #[test]
    fn an_evaluation_whose_scc_is_read_stays() {
        // The branch after the third pair reads its SCC: that pair stays, the second goes.
        let mut insts = row(3);
        insts.push(MInst::new(Opc::SCbranchScc1, [], []));
        let mut f = func(insts, 1);
        assert_eq!(reuse_masks(&mut f), 1);
        assert_eq!(f.blocks[0].insts.iter().filter(|i| i.op == Opc::SCmpLtI32).count(), 2);
    }
}
