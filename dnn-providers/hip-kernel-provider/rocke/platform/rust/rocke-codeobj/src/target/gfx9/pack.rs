// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Packed f32 on gfx950: two f32 VALU instructions of one kind that the
//! shared selection writes back to back (the elements of one vector
//! operation) become one `v_pk_mul_f32`, `v_pk_add_f32` or `v_pk_fma_f32`
//! that computes both lanes. It runs on virtual registers after `select`
//! and before `legalize`.
//!
//! Each source position of the two takes, as one 64-bit source:
//! - the two halves of one register tuple, in order or swapped (`op_sel`);
//! - one register read by both lanes (a broadcast, `op_sel_hi` cleared; at an
//!   odd register the encoder reads it as the high half of its pair);
//! - two one-dword registers written once each, which become the two halves
//!   of a new aligned pair (every read and write of them is renamed).
//!
//! The two results become the halves of a new pair the same way. Constants
//! are left alone (a packed source would need them in registers), and so is
//! anything with modifiers other than the same negation on both, precolored
//! registers, registers written more than once, and a result the second
//! instruction reads.

use super::gen_isa::Gfx9Op;
use crate::rir::mir::*;
use std::collections::HashMap;

/// The lane selects of `imm=pksel`: op_sel of source `k`, and its
/// op_sel_hi cleared.
fn op_sel(k: usize) -> i64 {
    1 << k
}

fn op_sel_hi_clear(k: usize) -> i64 {
    0x10 << k
}

pub fn run(f: &mut MFunc) {
    let mut defs = vec![0u32; f.vregs.len()];
    let mut early = vec![false; f.vregs.len()];
    for inst in f.blocks.iter().flat_map(|b| &b.insts) {
        for d in inst.defs.iter().filter_map(Opnd::vref) {
            defs[d.reg as usize] += 1;
            early[d.reg as usize] |= inst.op.flags() & F_EARLY != 0;
        }
    }
    let mut p = Packer { f_vregs: std::mem::take(&mut f.vregs), defs, early, rename: HashMap::new() };
    for block in f.blocks.iter_mut() {
        let insts = std::mem::take(&mut block.insts);
        block.insts = p.block(insts);
    }
    f.vregs = std::mem::take(&mut p.f_vregs);
    if p.rename.is_empty() {
        return;
    }
    let rename = |o: &mut Opnd, map: &HashMap<u32, (u32, u8)>| {
        if let Opnd::R(v) = o {
            if let Some(&(pair, off)) = map.get(&v.reg) {
                *v = VRef { reg: pair, off: off + v.off, n: v.n };
            }
        }
    };
    for inst in f.blocks.iter_mut().flat_map(|b| b.insts.iter_mut()) {
        for o in inst.defs.iter_mut().chain(inst.uses.iter_mut()) {
            rename(o, &p.rename);
        }
    }
}

struct Packer {
    f_vregs: Vec<VRegInfo>,
    defs: Vec<u32>,
    /// Written by an instruction whose destination must not overlap its
    /// sources: never made half of a pair.
    early: Vec<bool>,
    /// One-dword registers that became the half `off` of a pair.
    rename: HashMap<u32, (u32, u8)>,
}

/// One packed source: the 64-bit operand (or the plan's pair `Joined(j)`
/// once it exists) and its lane selects.
struct Source {
    opnd: Operand,
    sel: i64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Operand {
    Ready(Opnd),
    Joined(usize),
}

impl Packer {
    fn block(&mut self, insts: Vec<MInst>) -> Vec<MInst> {
        let mut out = Vec::with_capacity(insts.len());
        let mut i = 0;
        while i < insts.len() {
            if i + 1 < insts.len() {
                if let Some(pk) = self.pair(&insts[i], &insts[i + 1]) {
                    out.push(pk);
                    i += 2;
                    continue;
                }
            }
            out.push(insts[i].clone());
            i += 1;
        }
        out
    }

    /// `o` with the renames decided so far.
    fn canon(&self, o: Opnd) -> Opnd {
        match o {
            Opnd::R(v) => match self.rename.get(&v.reg) {
                Some(&(pair, off)) => Opnd::R(VRef { reg: pair, off: off + v.off, n: v.n }),
                None => o,
            },
            _ => o,
        }
    }

    /// A whole one-dword VGPR written once, not precolored, never early, and
    /// not yet half of a pair: free to become one.
    fn free(&self, o: Opnd) -> Option<u32> {
        let v = o.vref()?;
        let info = self.f_vregs[v.reg as usize];
        let ok = info.class == Class::V
            && info.size == 1
            && v.off == 0
            && v.n == 1
            && info.fixed.is_none()
            && self.defs[v.reg as usize] == 1
            && !self.early[v.reg as usize]
            && !self.rename.contains_key(&v.reg);
        ok.then_some(v.reg)
    }

    /// A new aligned pair whose halves are `lo` and `hi` (renamed later).
    fn join(&mut self, lo: u32, hi: u32) -> u32 {
        let pair = self.f_vregs.len() as u32;
        self.f_vregs.push(VRegInfo { class: Class::V, size: 2, align: 2, fixed: None });
        self.defs.push(2);
        self.early.push(false);
        self.rename.insert(lo, (pair, 0));
        self.rename.insert(hi, (pair, 1));
        pair
    }

    /// Whether the tuple `reg` starts at an even register: VGPR tuples always
    /// do on gfx950 (`legalize`), SGPR tuples by their alignment.
    fn even_tuple(&self, reg: u32) -> bool {
        let info = self.f_vregs[reg as usize];
        info.size >= 2 && (info.class == Class::V || info.align >= 2)
    }

    /// Source position `k` of the two lanes: `a` for the low lane, `b` for
    /// the high one. Two free registers join as a pair of the plan.
    fn source(&self, k: usize, a: Opnd, b: Opnd, plan: &mut Vec<(u32, u32)>) -> Option<Source> {
        let (a, b) = (self.canon(a), self.canon(b));
        let (va, vb) = (a.vref()?, b.vref()?);
        let ready = |o: Opnd, sel: i64| Some(Source { opnd: Operand::Ready(o), sel });
        if va.n != 1 || vb.n != 1 {
            return None;
        }
        if a == b {
            // Both lanes read one register (the encoder reads it as the high
            // half of its pair when it lands at an odd register).
            return ready(a, op_sel_hi_clear(k));
        }
        if va.reg == vb.reg {
            // The two halves of one pair, in order or swapped.
            if !self.even_tuple(va.reg) {
                return None;
            }
            if vb.off == va.off + 1 && va.off % 2 == 0 {
                return ready(Opnd::R(VRef { reg: va.reg, off: va.off, n: 2 }), 0);
            }
            if va.off == vb.off + 1 && vb.off % 2 == 0 {
                return ready(Opnd::R(VRef { reg: va.reg, off: vb.off, n: 2 }), op_sel(k) | op_sel_hi_clear(k));
            }
            return None;
        }
        let (lo, hi) = (self.free(a)?, self.free(b)?);
        if let Some(j) = plan.iter().position(|&p| p == (lo, hi)) {
            return Some(Source { opnd: Operand::Joined(j), sel: 0 });
        }
        if plan.iter().any(|&(x, y)| [x, y].contains(&lo) || [x, y].contains(&hi)) {
            return None;
        }
        plan.push((lo, hi));
        Some(Source { opnd: Operand::Joined(plan.len() - 1), sel: 0 })
    }

    fn pair(&mut self, i1: &MInst, i2: &MInst) -> Option<MInst> {
        let (op, extra) = match (i1.op, i2.op) {
            (Opc::VMulF32, Opc::VMulF32) => (Gfx9Op::VPkMulF32, 0),
            (Opc::VAddF32, Opc::VAddF32) => (Gfx9Op::VPkAddF32, 0),
            (Opc::VSubF32, Opc::VSubF32) => (Gfx9Op::VPkAddF32, M_NEG1),
            (Opc::VFmaF32, Opc::VFmaF32) => (Gfx9Op::VPkFmaF32, 0),
            _ => return None,
        };
        let neg = M_NEG0 | M_NEG1 | M_NEG2;
        if i1.mods != i2.mods || i1.mods & !neg != 0 || i1.mods & extra != 0 || i1.defs.len() != 1 || i2.defs.len() != 1 {
            return None;
        }
        let (d1, d2) = (self.free(i1.defs[0])?, self.free(i2.defs[0])?);
        if i2.uses.iter().any(|u| u.vref().is_some_and(|v| v.reg == d1)) || i1.uses.len() != i2.uses.len() {
            return None;
        }
        // Commutative sources may pair with the other lane's sources swapped.
        let n = i1.uses.len();
        let swaps: &[bool] = if op == Gfx9Op::VPkFmaF32 || extra != 0 { &[false] } else { &[false, true] };
        for &swap in swaps {
            let mut b_uses: Vec<Opnd> = i2.uses.to_vec();
            if swap {
                b_uses.swap(0, 1);
            }
            let mut plan = vec![(d1, d2)];
            let srcs: Option<Vec<Source>> = (0..n).map(|k| self.source(k, i1.uses[k], b_uses[k], &mut plan)).collect();
            let Some(srcs) = srcs else { continue };
            // One scalar at most: a second would be copied to a VGPR the
            // packed source could not take at an odd register.
            let mut scalars: Vec<u32> = srcs
                .iter()
                .filter_map(|s| match s.opnd {
                    Operand::Ready(o) => o.vref().filter(|v| self.f_vregs[v.reg as usize].class == Class::S).map(|v| v.reg),
                    Operand::Joined(_) => None,
                })
                .collect();
            scalars.dedup();
            if scalars.len() > 1 {
                continue;
            }
            // Commit: the plan's pairs (the results first), the alignments.
            let pairs: Vec<u32> = plan.into_iter().map(|(lo, hi)| self.join(lo, hi)).collect();
            let mut uses = Uses::new();
            let mut sel = 0;
            for s in srcs {
                uses.push(match s.opnd {
                    Operand::Ready(o) => o,
                    Operand::Joined(j) => Opnd::R(VRef { reg: pairs[j], off: 0, n: 2 }),
                });
                sel |= s.sel;
            }
            let pk = MInst::new(Opc::Gfx9(op), [Opnd::R(VRef { reg: pairs[0], off: 0, n: 2 })], uses).with_imm(sel);
            return Some(pk.with_mods(i1.mods | extra));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(reg: u32, off: u8, n: u8) -> Opnd {
        Opnd::R(VRef { reg, off, n })
    }

    fn func(vregs: &[(Class, u8)], insts: Vec<MInst>) -> MFunc {
        let mut f = MFunc { target: crate::target::Target::Gfx950, ..MFunc::default() };
        for &(class, size) in vregs {
            f.vregs.push(VRegInfo { class, size, align: if size > 1 { 2 } else { 1 }, fixed: None });
        }
        f.blocks = vec![MBlock { insts, loop_depth: 0 }];
        f
    }

    const V: Class = Class::V;

    #[test]
    fn element_pairs_become_packed_instructions() {
        // r1, r2 = cvt; r3 = r1 * r1; r4 = r2 * r2; r5 = r3 + r4.
        let insts = vec![
            MInst::new(Opc::VCvtF32F16, [r(1, 0, 1)], [r(0, 0, 1)]),
            MInst::new(Opc::VCvtF32F16, [r(2, 0, 1)], [r(0, 0, 1)]).with_mods(M_OPSEL_SRC0),
            MInst::new(Opc::VMulF32, [r(3, 0, 1)], [r(1, 0, 1), r(1, 0, 1)]),
            MInst::new(Opc::VMulF32, [r(4, 0, 1)], [r(2, 0, 1), r(2, 0, 1)]),
            MInst::new(Opc::VAddF32, [r(5, 0, 1)], [r(3, 0, 1), r(4, 0, 1)]),
        ];
        let mut f = func(&[(V, 1); 6], insts);
        run(&mut f);
        let code = &f.blocks[0].insts;
        assert_eq!(code.len(), 4, "{code:?}");
        // The results are pair 6, the converted values pair 7.
        let (res, src) = (6u32, 7u32);
        assert_eq!((f.vregs[6].size, f.vregs[6].align, f.vregs[7].size), (2, 2, 2));
        assert_eq!((code[0].defs[0], code[1].defs[0]), (r(src, 0, 1), r(src, 1, 1)));
        let pk = &code[2];
        assert_eq!((pk.op, pk.defs[0], &pk.uses[..], pk.imm), (Opc::Gfx9(Gfx9Op::VPkMulF32), r(res, 0, 2), &[r(src, 0, 2), r(src, 0, 2)][..], 0));
        assert_eq!(&code[3].uses[..], &[r(res, 0, 1), r(res, 1, 1)][..]);
    }

    #[test]
    fn a_shared_source_is_broadcast_and_tuple_halves_pair_in_place() {
        // r3, r4 = s * x[0], s * x[1] (x a pair, s one register read by both).
        let insts = vec![
            MInst::new(Opc::VMulF32, [r(3, 0, 1)], [r(1, 0, 1), r(2, 0, 1)]),
            MInst::new(Opc::VMulF32, [r(4, 0, 1)], [r(1, 0, 1), r(2, 1, 1)]),
            MInst::new(Opc::VAddF32, [r(5, 0, 1)], [r(3, 0, 1), r(4, 0, 1)]),
        ];
        let mut f = func(&[(V, 1), (V, 1), (V, 2), (V, 1), (V, 1), (V, 1)], insts);
        run(&mut f);
        let pk = &f.blocks[0].insts[0];
        assert_eq!((pk.op, &pk.uses[..], pk.imm), (Opc::Gfx9(Gfx9Op::VPkMulF32), &[r(1, 0, 1), r(2, 0, 2)][..], 0x10));
        assert_eq!(f.vregs[1].align, 1, "a broadcast source may land at any register");
        // Swapped halves: op_sel and op_sel_hi cleared on that source.
        let insts = vec![
            MInst::new(Opc::VMulF32, [r(3, 0, 1)], [r(2, 1, 1), r(1, 0, 1)]),
            MInst::new(Opc::VMulF32, [r(4, 0, 1)], [r(2, 0, 1), r(1, 0, 1)]),
        ];
        let mut f = func(&[(V, 1), (V, 1), (V, 2), (V, 1), (V, 1)], insts);
        run(&mut f);
        let pk = &f.blocks[0].insts[0];
        assert_eq!((&pk.uses[..], pk.imm), (&[r(2, 0, 2), r(1, 0, 1)][..], 0x11 | 0x20));
    }

    #[test]
    fn what_cannot_pair_stays() {
        let regs = [(V, 1); 6];
        let keep = |insts: Vec<MInst>| {
            let mut f = func(&regs, insts.clone());
            run(&mut f);
            assert!(f.blocks[0].insts.iter().all(|i| !matches!(i.op, Opc::Gfx9(_))), "{:?}", f.blocks[0].insts);
            assert_eq!(f.blocks[0].insts.len(), insts.len());
        };
        // The second reads the first's result.
        keep(vec![
            MInst::new(Opc::VMulF32, [r(3, 0, 1)], [r(1, 0, 1), r(2, 0, 1)]),
            MInst::new(Opc::VMulF32, [r(4, 0, 1)], [r(3, 0, 1), r(2, 0, 1)]),
        ]);
        // A constant source.
        keep(vec![
            MInst::new(Opc::VMulF32, [r(3, 0, 1)], [Opnd::Imm(0x3f00_0000), r(1, 0, 1)]),
            MInst::new(Opc::VMulF32, [r(4, 0, 1)], [Opnd::Imm(0x3f00_0000), r(2, 0, 1)]),
        ]);
        // Different negations, and different kinds.
        keep(vec![
            MInst::new(Opc::VMulF32, [r(3, 0, 1)], [r(1, 0, 1), r(2, 0, 1)]).with_mods(M_NEG0),
            MInst::new(Opc::VMulF32, [r(4, 0, 1)], [r(1, 0, 1), r(2, 0, 1)]),
        ]);
        keep(vec![
            MInst::new(Opc::VMulF32, [r(3, 0, 1)], [r(1, 0, 1), r(2, 0, 1)]),
            MInst::new(Opc::VAddF32, [r(4, 0, 1)], [r(1, 0, 1), r(2, 0, 1)]),
        ]);
        // A result written twice (a loop-carried value).
        keep(vec![
            MInst::new(Opc::VMulF32, [r(3, 0, 1)], [r(1, 0, 1), r(2, 0, 1)]),
            MInst::new(Opc::VMulF32, [r(4, 0, 1)], [r(1, 0, 1), r(5, 0, 1)]),
            MInst::new(Opc::VMovB32, [r(3, 0, 1)], [r(5, 0, 1)]),
        ]);
    }
}
