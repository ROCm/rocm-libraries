// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! gfx950's own instructions (`Opc::Gfx9`) where the shared instruction
//! selection writes a sequence for gfx11, which lacks them. It runs on
//! virtual registers after `rir::isel::select_for` and before `legalize`,
//! within a block, and rewrites a sequence only when its temporaries have
//! one definition and no use outside it:
//!
//! - `PACK_F16`, and a `v_perm_b32` joining the low halves of two
//!   `v_cvt_f16_f32` results (or one and a zero): `v_cvt_pk_f16_f32`;
//! - the gfx11 `fptrunc float to bfloat` (`v_bfe_u32`, `v_or_b32`,
//!   `v_cmp_u_f32`, `v_add3_u32`, `v_cndmask_b32`), whose bf16 is the high
//!   half of its result: `v_cvt_pk_bf16_f32 0, x`, which has the same high
//!   half; and a `v_perm_b32` joining two such high halves (or one and a
//!   zero): one `v_cvt_pk_bf16_f32`;
//! - a 64-bit add (`v_add_co_u32`, then `v_addc_co_u32` on its carry) of a
//!   register pair, shifted left by 0 to 4 (`v_lshlrev_b64`) or not:
//!   `v_lshl_add_u64`.
//!
//! The conversions are LLVM's for gfx950: `fptrunc` to half or bfloat rounds
//! to nearest even in the kernel's float mode, and a NaN stays a NaN.
//!
//! `ROCKE_DIRECT_DEBUG=noselect9` skips these rewrites, `nopack` the packing.

use super::gen_isa::Gfx9Op;
use crate::rir::mir::*;
use std::collections::HashMap;

/// Definitions and uses of each virtual register in the function.
struct Counts {
    defs: Vec<u32>,
    uses: Vec<u32>,
}

impl Counts {
    fn new(f: &MFunc) -> Self {
        let mut c = Counts { defs: vec![0; f.vregs.len()], uses: vec![0; f.vregs.len()] };
        for inst in f.blocks.iter().flat_map(|b| &b.insts) {
            c.add(inst, 1);
        }
        c
    }

    /// Counts `inst`'s operands in (`by` 1) or out (`by` -1).
    fn add(&mut self, inst: &MInst, by: i32) {
        for d in inst.defs.iter().filter_map(Opnd::vref) {
            self.defs[d.reg as usize] = (self.defs[d.reg as usize] as i32 + by) as u32;
        }
        for u in inst.uses.iter().filter_map(Opnd::vref) {
            self.uses[u.reg as usize] = (self.uses[u.reg as usize] as i32 + by) as u32;
        }
    }
}

/// The rewrites above.
pub fn run(f: &mut MFunc) {
    if crate::rir::debug_flag("noselect9") {
        return;
    }
    let mut counts = Counts::new(f);
    let sizes: Vec<u8> = f.vregs.iter().map(|v| v.size).collect();
    for block in f.blocks.iter_mut() {
        let insts = std::mem::take(&mut block.insts);
        block.insts = Block::new(insts, &sizes, &mut counts).run();
    }
}

struct Block<'a> {
    code: Vec<Option<MInst>>,
    /// Where each register written so far in the block was last written.
    def_at: HashMap<u32, usize>,
    sizes: &'a [u8],
    counts: &'a mut Counts,
}

const PERM_LOW_HALVES: u32 = 0x0504_0100;
const PERM_HIGH_HALVES: u32 = 0x0706_0302;
const PERM_HIGH_THEN_LOW: u32 = 0x0504_0302;

impl<'a> Block<'a> {
    fn new(insts: Vec<MInst>, sizes: &'a [u8], counts: &'a mut Counts) -> Self {
        Block { code: insts.into_iter().map(Some).collect(), def_at: HashMap::new(), sizes, counts }
    }

    fn run(mut self) -> Vec<MInst> {
        for i in 0..self.code.len() {
            let Some(inst) = self.code[i].clone() else { continue };
            if let Some(new) = self.rewrite(&inst, i) {
                self.counts.add(&inst, -1);
                self.counts.add(&new, 1);
                self.code[i] = Some(new);
            }
            for d in self.code[i].as_ref().map_or(&[][..], |n| &n.defs[..]).iter().filter_map(Opnd::vref) {
                self.def_at.insert(d.reg, i);
            }
        }
        self.code.into_iter().flatten().collect()
    }

    fn rewrite(&mut self, inst: &MInst, i: usize) -> Option<MInst> {
        let gfx9 = |op: Gfx9Op, uses: [Opnd; 2]| MInst::new(Opc::Gfx9(op), inst.defs, uses);
        match inst.op {
            Opc::PackF16 => Some(gfx9(Gfx9Op::VCvtPkF16F32, [inst.uses[0], inst.uses[1]])),
            Opc::VCndmaskB32 if inst.mods == 0 => self.bf16_round(inst).map(|x| gfx9(Gfx9Op::VCvtPkBf16F32, [Opnd::Imm(0), x])),
            Opc::VPermB32 if inst.mods == 0 => {
                let (hi, lo, sel) = (inst.uses[0], inst.uses[1], inst.uses[2]);
                match sel {
                    Opnd::Imm(PERM_LOW_HALVES) => {
                        let (x, y) = (self.f16_of(lo)?, self.f16_of(hi)?);
                        if x == Opnd::Imm(0) && y == Opnd::Imm(0) {
                            return None;
                        }
                        self.take(lo);
                        self.take(hi);
                        Some(gfx9(Gfx9Op::VCvtPkF16F32, [x, y]))
                    }
                    Opnd::Imm(s @ (PERM_HIGH_HALVES | PERM_HIGH_THEN_LOW)) => {
                        let x = self.bf16_of(lo)?;
                        let y = if s == PERM_HIGH_HALVES { self.bf16_of(hi)? } else { self.zero(hi)? };
                        self.take(lo);
                        self.take(hi);
                        Some(gfx9(Gfx9Op::VCvtPkBf16F32, [x, y]))
                    }
                    _ => None,
                }
            }
            Opc::VAddCoCiU32 if i > 0 && inst.mods == 0 => self.shift_add64(inst, i),
            _ => None,
        }
    }

    /// The instruction last writing `o`, a whole one-dword register written
    /// once, in this block, and read nowhere else.
    fn temp(&self, o: Opnd) -> Option<(usize, &MInst)> {
        let v = o.vref()?;
        let whole = v.off == 0 && v.n == self.sizes[v.reg as usize];
        let reg = v.reg as usize;
        if !whole || self.counts.defs[reg] != 1 || self.counts.uses[reg] != 1 {
            return None;
        }
        let at = *self.def_at.get(&v.reg)?;
        Some((at, self.code[at].as_ref()?))
    }

    /// Removes the instruction writing temporary `o` (`temp` found it).
    fn take(&mut self, o: Opnd) {
        if let Some((at, _)) = self.temp(o) {
            let inst = self.code[at].take().expect("a live instruction");
            self.counts.add(&inst, -1);
        }
    }

    /// The f32 whose f16 conversion is temporary `o` (`Imm(0)` for a zero).
    fn f16_of(&self, o: Opnd) -> Option<Opnd> {
        let (_, d) = self.temp(o)?;
        match d.op {
            Opc::VCvtF16F32 if d.mods == 0 => Some(d.uses[0]),
            _ => self.zero(o),
        }
    }

    /// The f32 whose bf16 is the high half of temporary `o`, written by
    /// `v_cvt_pk_bf16_f32 0, x`.
    fn bf16_of(&self, o: Opnd) -> Option<Opnd> {
        let (_, d) = self.temp(o)?;
        (d.op == Opc::Gfx9(Gfx9Op::VCvtPkBf16F32) && d.mods == 0 && d.uses[0] == Opnd::Imm(0)).then_some(d.uses[1])
    }

    /// `Imm(0)` when temporary `o` is a copy of zero.
    fn zero(&self, o: Opnd) -> Option<Opnd> {
        let (_, d) = self.temp(o)?;
        (matches!(d.op, Opc::Copy | Opc::VMovB32) && d.mods == 0 && d.uses[..] == [Opnd::Imm(0)]).then_some(Opnd::Imm(0))
    }

    /// The x of `v_cndmask_b32 rounded, quiet, nan` when it ends the gfx11
    /// `fptrunc` of x to bfloat: `nan = v_cmp_u_f32 x, x`, `quiet = x |
    /// 0x400000`, `rounded = x + 0x7fff + bit 16 of x` (`v_add3_u32` of
    /// `v_bfe_u32 x, 16, 1`). Its four temporaries are removed.
    fn bf16_round(&mut self, cnd: &MInst) -> Option<Opnd> {
        let (rounded, quiet, nan) = (cnd.uses[0], cnd.uses[1], cnd.uses[2]);
        let (n_at, n) = self.temp(nan)?;
        let x = n.uses[0];
        if n.op != Opc::VCmp(0x18) || n.mods != 0 || n.uses[..] != [x, x] {
            return None;
        }
        let (q_at, q) = self.temp(quiet)?;
        let quiets = q.op == Opc::VOrB32
            && q.mods == 0
            && (q.uses[..] == [Opnd::Imm(0x40_0000), x] || q.uses[..] == [x, Opnd::Imm(0x40_0000)]);
        let (a_at, a) = self.temp(rounded)?;
        if !quiets || a.op != Opc::VAdd3U32 || a.mods != 0 || a.uses.len() != 3 {
            return None;
        }
        // x, 0x7fff and the bit-16 extract, in any order.
        let mut rest: Vec<Opnd> = a.uses.to_vec();
        for want in [x, Opnd::Imm(0x7fff)] {
            let p = rest.iter().position(|&o| o == want)?;
            rest.remove(p);
        }
        let (b_at, b) = self.temp(rest[0])?;
        if b.op != Opc::VBfeU32 || b.mods != 0 || b.uses[..] != [x, Opnd::Imm(16), Opnd::Imm(1)] {
            return None;
        }
        for at in [n_at, q_at, a_at, b_at] {
            let inst = self.code[at].take().expect("a live instruction");
            self.counts.add(&inst, -1);
        }
        Some(x)
    }

    /// `v_lshl_add_u64 d, t, k, b` for the 64-bit add `d = b + (t << k)` of
    /// register pairs that `v_add_co_u32` (just before `addc`) and
    /// `v_addc_co_u32` write, a shift `t << k` by 0-4 written once and read
    /// by the add alone taking its place. The carry is read by `addc` only.
    fn shift_add64(&mut self, addc: &MInst, i: usize) -> Option<MInst> {
        let add = self.code[i - 1].clone()?;
        if add.op != Opc::VAddCoU32 || add.mods != 0 || addc.uses.len() != 3 {
            return None;
        }
        let (lo, carry) = (add.defs[0].vref()?, add.defs[1]);
        let hi = addc.defs[0].vref()?;
        let carry_free = matches!(addc.defs.get(1), Some(Opnd::P(Phys::Null, _)))
            || addc.defs.get(1).and_then(|d| d.vref()).is_some_and(|d| self.counts.uses[d.reg as usize] == 0);
        let whole_carry = carry.vref().is_some_and(|c| c.off == 0 && c.n == self.sizes[c.reg as usize]);
        if !carry_free || !whole_carry || addc.uses[2] != carry || self.counts.uses[carry.vref()?.reg as usize] != 1 {
            return None;
        }
        // The result is one register pair, written by these two alone.
        let d = lo.reg;
        if hi.reg != d || (lo.off, lo.n, hi.off, hi.n) != (0, 1, 1, 1) || self.sizes[d as usize] != 2 || self.counts.defs[d as usize] != 2
        {
            return None;
        }
        // The addends: the pairs whose low halves `add` and high halves `addc` take.
        let pair = |l: Opnd, h: Opnd| -> Option<Opnd> {
            let (l, h) = (l.vref()?, h.vref()?);
            (l.reg == h.reg && l.n == 1 && h.n == 1 && h.off == l.off + 1 && l.off % 2 == 0)
                .then_some(Opnd::R(VRef { reg: l.reg, off: l.off, n: 2 }))
        };
        let orders = [(0, 1), (1, 0)];
        let (t, b) = orders
            .iter()
            .find_map(|&(ti, bi)| Some((pair(add.uses[ti], addc.uses[ti])?, pair(add.uses[bi], addc.uses[bi])?)))?;
        let (t, b) = match (self.shifted(t), self.shifted(b)) {
            (Some(_), _) => (t, b),
            (None, Some(_)) => (b, t),
            (None, None) => (t, b),
        };
        let (src0, k) = match self.shifted(t) {
            Some((at, src, k)) => {
                let shift = self.code[at].take().expect("a live instruction");
                self.counts.add(&shift, -1);
                (src, k)
            }
            None => (t, 0),
        };
        let gone = self.code[i - 1].take().expect("a live instruction");
        self.counts.add(&gone, -1);
        let dst = Opnd::R(VRef { reg: d, off: 0, n: 2 });
        Some(MInst::new(Opc::Gfx9(Gfx9Op::VLshlAddU64), [dst], [src0, Opnd::Imm(k), b]))
    }

    /// (where, source, amount) when pair `t` is `v_lshlrev_b64 k, source`
    /// with `k` at most 4, written once in this block and read only as the
    /// two halves an add takes.
    fn shifted(&self, t: Opnd) -> Option<(usize, Opnd, u32)> {
        let v = t.vref()?;
        let reg = v.reg as usize;
        if (v.off, v.n) != (0, 2) || self.sizes[reg] != 2 || self.counts.defs[reg] != 1 || self.counts.uses[reg] != 2 {
            return None;
        }
        let at = *self.def_at.get(&v.reg)?;
        let s = self.code[at].as_ref()?;
        match (s.op, s.uses.first(), s.uses.get(1)) {
            (Opc::VLshlrevB64, Some(&Opnd::Imm(k)), Some(&src)) if s.mods == 0 && k <= 4 && src.dwords() == 2 => Some((at, src, k)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(reg: u32, off: u8, n: u8) -> Opnd {
        Opnd::R(VRef { reg, off, n })
    }

    /// A function of one block over registers of these (class, size).
    fn func(vregs: &[(Class, u8)], insts: Vec<MInst>) -> MFunc {
        let mut f = MFunc { target: crate::target::Target::Gfx950, ..MFunc::default() };
        for &(class, size) in vregs {
            f.vregs.push(VRegInfo { class, size, align: 1, fixed: None });
        }
        f.blocks = vec![MBlock { insts, loop_depth: 0 }];
        f
    }

    fn selected(mut f: MFunc) -> Vec<MInst> {
        run(&mut f);
        f.blocks.remove(0).insts
    }

    const V: Class = Class::V;
    const S: Class = Class::S;

    #[test]
    fn f16_packs_become_one_conversion() {
        // v_cvt_f16_f32 r2, r0; v_cvt_f16_f32 r3, r1; v_perm_b32 r4, r3, r2, 0x05040100
        let insts = vec![
            MInst::new(Opc::VCvtF16F32, [r(2, 0, 1)], [r(0, 0, 1)]),
            MInst::new(Opc::VCvtF16F32, [r(3, 0, 1)], [r(1, 0, 1)]),
            MInst::new(Opc::VPermB32, [r(4, 0, 1)], [r(3, 0, 1), r(2, 0, 1), Opnd::Imm(PERM_LOW_HALVES)]),
            MInst::new(Opc::PackF16, [r(5, 0, 1)], [r(0, 0, 1), r(1, 0, 1)]),
        ];
        let code = selected(func(&[(V, 1); 6], insts));
        let pk = Opc::Gfx9(Gfx9Op::VCvtPkF16F32);
        assert_eq!(code.len(), 2);
        assert_eq!((code[0].op, code[0].defs[0], &code[0].uses[..]), (pk, r(4, 0, 1), &[r(0, 0, 1), r(1, 0, 1)][..]));
        assert_eq!((code[1].op, &code[1].uses[..]), (pk, &[r(0, 0, 1), r(1, 0, 1)][..]));
        // A zero high half (an odd element count).
        let insts = vec![
            MInst::new(Opc::VCvtF16F32, [r(2, 0, 1)], [r(0, 0, 1)]),
            MInst::new(Opc::Copy, [r(3, 0, 1)], [Opnd::Imm(0)]),
            MInst::new(Opc::VPermB32, [r(4, 0, 1)], [r(3, 0, 1), r(2, 0, 1), Opnd::Imm(PERM_LOW_HALVES)]),
        ];
        let code = selected(func(&[(V, 1); 5], insts));
        assert_eq!((code.len(), &code[0].uses[..]), (1, &[r(0, 0, 1), Opnd::Imm(0)][..]));
        // A conversion read elsewhere stays, and so does the perm.
        let insts = vec![
            MInst::new(Opc::VCvtF16F32, [r(2, 0, 1)], [r(0, 0, 1)]),
            MInst::new(Opc::VCvtF16F32, [r(3, 0, 1)], [r(1, 0, 1)]),
            MInst::new(Opc::VPermB32, [r(4, 0, 1)], [r(3, 0, 1), r(2, 0, 1), Opnd::Imm(PERM_LOW_HALVES)]),
            MInst::new(Opc::VAddF32, [r(5, 0, 1)], [r(2, 0, 1), r(2, 0, 1)]),
        ];
        let code = selected(func(&[(V, 1); 6], insts));
        assert_eq!(code.iter().map(|i| i.op).collect::<Vec<_>>(), [Opc::VCvtF16F32, Opc::VCvtF16F32, Opc::VPermB32, Opc::VAddF32]);
    }

    /// The gfx11 `fptrunc float to bfloat` of one value: (instructions, result).
    fn bf16(x: u32, first: u32) -> (Vec<MInst>, u32) {
        let [lsb, quiet, nan, rounded, res] = [first, first + 1, first + 2, first + 3, first + 4];
        let x = r(x, 0, 1);
        (
            vec![
                MInst::new(Opc::VBfeU32, [r(lsb, 0, 1)], [x, Opnd::Imm(16), Opnd::Imm(1)]),
                MInst::new(Opc::VOrB32, [r(quiet, 0, 1)], [Opnd::Imm(0x40_0000), x]),
                MInst::new(Opc::VCmp(0x18), [r(nan, 0, 2)], [x, x]),
                MInst::new(Opc::VAdd3U32, [r(rounded, 0, 1)], [r(lsb, 0, 1), x, Opnd::Imm(0x7fff)]),
                MInst::new(Opc::VCndmaskB32, [r(res, 0, 1)], [r(rounded, 0, 1), r(quiet, 0, 1), r(nan, 0, 2)]),
            ],
            res,
        )
    }

    /// The registers of `n / 5` roundings: the compare's mask is a pair.
    fn bf16_regs(n: usize) -> Vec<(Class, u8)> {
        (0..n).map(|k| if k % 5 == 2 { (S, 2) } else { (V, 1) }).collect()
    }

    #[test]
    fn bf16_roundings_become_conversions() {
        // Registers 0 and 1 are the f32 values; 2..=11 the two roundings, 12 the pack.
        let mut regs = vec![(V, 1), (V, 1)];
        regs.extend(bf16_regs(10));
        regs.push((V, 1));
        let (mut insts, a) = bf16(0, 2);
        let (more, b) = bf16(1, 7);
        insts.extend(more);
        insts.push(MInst::new(Opc::VPermB32, [r(12, 0, 1)], [r(b, 0, 1), r(a, 0, 1), Opnd::Imm(PERM_HIGH_HALVES)]));
        let code = selected(func(&regs, insts.clone()));
        assert_eq!(code.len(), 1, "{code:?}");
        assert_eq!((code[0].op, &code[0].uses[..]), (Opc::Gfx9(Gfx9Op::VCvtPkBf16F32), &[r(0, 0, 1), r(1, 0, 1)][..]));
        // One rounding stored as a high half: the conversion of 0 and x.
        let (one, res) = bf16(0, 2);
        let mut regs = vec![(V, 1), (V, 1)];
        regs.extend(bf16_regs(5));
        let mut code = one.clone();
        code.push(MInst::new(Opc::GStoreD16HiB16, [], [r(1, 0, 1), r(res, 0, 1)]));
        let code = selected(func(&regs, code));
        assert_eq!(code.len(), 2);
        assert_eq!((code[0].op, &code[0].uses[..]), (Opc::Gfx9(Gfx9Op::VCvtPkBf16F32), &[Opnd::Imm(0), r(0, 0, 1)][..]));
        // A temporary read elsewhere keeps the whole sequence.
        let mut code = one;
        code.push(MInst::new(Opc::VAddF32, [r(1, 0, 1)], [r(2, 0, 1), r(2, 0, 1)]));
        let code = selected(func(&regs, code));
        assert_eq!(code.len(), 6);
        assert!(code.iter().all(|i| !matches!(i.op, Opc::Gfx9(_))));
    }

    #[test]
    fn shifted_64_bit_adds_become_lshl_add() {
        // r2 = r1 << 1; r3 = s0[0:1] + r2 (lo with carry r4, then hi).
        let regs = [(S, 4), (V, 2), (V, 2), (V, 2), (S, 2)];
        let shift = MInst::new(Opc::VLshlrevB64, [r(2, 0, 2)], [Opnd::Imm(1), r(1, 0, 2)]);
        let add = MInst::new(Opc::VAddCoU32, [r(3, 0, 1), r(4, 0, 2)], [r(0, 2, 1), r(2, 0, 1)]);
        let addc =
            MInst::new(Opc::VAddCoCiU32, [r(3, 1, 1), Opnd::P(Phys::Null, 2)], [r(0, 3, 1), r(2, 1, 1), r(4, 0, 2)]);
        let code = selected(func(&regs, vec![shift.clone(), add.clone(), addc.clone()]));
        assert_eq!(code.len(), 1, "{code:?}");
        let want = [r(1, 0, 2), Opnd::Imm(1), r(0, 2, 2)];
        assert_eq!((code[0].op, code[0].defs[0], &code[0].uses[..]), (Opc::Gfx9(Gfx9Op::VLshlAddU64), r(3, 0, 2), &want[..]));
        // Without the shift, an add of two pairs: a shift by 0.
        let code = selected(func(&regs, vec![add.clone(), addc.clone()]));
        assert_eq!((code.len(), &code[0].uses[..]), (1, &[r(0, 2, 2), Opnd::Imm(0), r(2, 0, 2)][..]));
        // A shift by 5, or a carry read elsewhere, stays as it was.
        let wide = MInst::new(Opc::VLshlrevB64, [r(2, 0, 2)], [Opnd::Imm(5), r(1, 0, 2)]);
        let code = selected(func(&regs, vec![wide, add.clone(), addc.clone()]));
        assert_eq!((code.len(), code[1].op, code[1].uses[1]), (2, Opc::Gfx9(Gfx9Op::VLshlAddU64), Opnd::Imm(0)));
        let read = MInst::new(Opc::VCndmaskB32, [r(1, 0, 1)], [Opnd::Imm(0), Opnd::Imm(1), r(4, 0, 2)]);
        let code = selected(func(&regs, vec![shift, add, addc, read]));
        assert_eq!(code.len(), 4);
        assert!(code.iter().all(|i| !matches!(i.op, Opc::Gfx9(_))));
    }
}
