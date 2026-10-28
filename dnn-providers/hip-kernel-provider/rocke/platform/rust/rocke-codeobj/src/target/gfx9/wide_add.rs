// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! The 64-bit add of a register pair and a zero-extended 32-bit VGPR as one
//! `v_lshl_add_u64` of the pair `[y, 0]`, as COMGR forms an address from a
//! base and an unsigned offset.
//!
//! Selection adds the halves with a carry: `v_add_co_u32` then
//! `v_addc_co_u32 hi, base_hi, 0, carry`, which `select` cannot fuse (its
//! `shift_add64` takes two pairs). That costs the carry's SGPR pair, a VGPR
//! copy of `base_hi` (gfx9 reads one SGPR per VALU instruction, and the
//! carry is one), and the wait state a VALU owes the SGPR another wrote.
//! The band-mask attention backward kernels form 16 such addresses per
//! iteration (the LSE and Dsum loads at a clamped row). Each `y` makes one
//! pair per block, reused by the adds of the same `y` until EXEC changes.
//!
//! Not `v_mad_i64_i32 x, 2^k, base`: as the pair `[x, x >> 31]` shifted into
//! the add (COMGR's form, a sign half and often a copy of `x` more) it ran
//! slower on 51 of the 240 GEMMs, `2651251cfd202a67` among them.

use super::gen_isa::Gfx9Op;
use crate::rir::mir::*;

/// `base + zext(y)` with `d` its result: an add and the carry add just
/// after it of the halves of one result pair `d`, the carry read by the
/// second alone and its carry out unused.
fn matched(f: &MFunc, defs: &[u32], uses: &[u32], add: &MInst, addc: &MInst) -> Option<(VRef, Opnd, VRef)> {
    if add.op != Opc::VAddCoU32 || addc.op != Opc::VAddCoCiU32 || add.mods != 0 || addc.mods != 0 {
        return None;
    }
    if add.defs.len() != 2 || add.uses.len() != 2 || addc.uses.len() != 3 {
        return None;
    }
    let (lo, hi, carry) = (add.defs[0].vref()?, addc.defs[0].vref()?, add.defs[1].vref()?);
    let carry_whole = carry.off == 0 && carry.n == f.vregs[carry.reg as usize].size;
    let carry_out_unused = match addc.defs.get(1) {
        None | Some(Opnd::P(Phys::Null, _)) => true,
        Some(o) => o.vref().is_some_and(|c| uses[c.reg as usize] == 0),
    };
    if !carry_whole || !carry_out_unused || addc.uses[2] != Opnd::R(carry) || uses[carry.reg as usize] != 1 {
        return None;
    }
    let d = lo.reg;
    if hi.reg != d || (lo.off, lo.n, hi.off, hi.n) != (0, 1, 1, 1) || f.vregs[d as usize].size != 2 || defs[d as usize] != 2 {
        return None;
    }
    let pair = |l: Opnd, h: Opnd| -> Option<Opnd> {
        let (l, h) = (l.vref()?, h.vref()?);
        (l.reg == h.reg && l.n == 1 && h.n == 1 && h.off == l.off + 1 && l.off % 2 == 0).then_some(Opnd::R(VRef { n: 2, ..l }))
    };
    [(0, 1), (1, 0)].iter().find_map(|&(bi, yi)| {
        let base = pair(add.uses[bi], addc.uses[bi])?;
        let y = add.uses[yi].vref()?;
        let vgpr = y.n == 1 && f.vregs[y.reg as usize].class == Class::V;
        (vgpr && addc.uses[yi] == Opnd::Imm(0)).then_some((VRef { reg: d, off: 0, n: 2 }, base, y))
    })
}

/// Rewrites `f` (after selection) as the module documentation says;
/// returns how many adds became one. `ROCKE_DIRECT_DEBUG=nowideadd` skips it.
pub fn run(f: &mut MFunc) -> usize {
    if crate::rir::debug_flag("nowideadd") {
        return 0;
    }
    let (mut defs, mut uses) = (vec![0u32; f.vregs.len()], vec![0u32; f.vregs.len()]);
    for inst in f.blocks.iter().flat_map(|b| &b.insts) {
        inst.defs.iter().filter_map(Opnd::vref).for_each(|v| defs[v.reg as usize] += 1);
        inst.uses.iter().filter_map(Opnd::vref).for_each(|v| uses[v.reg as usize] += 1);
    }
    let mut fused = 0;
    for b in 0..f.blocks.len() {
        let insts = std::mem::take(&mut f.blocks[b].insts);
        let mut out = Vec::with_capacity(insts.len());
        let mut pairs: std::collections::HashMap<VRef, VRef> = std::collections::HashMap::new();
        let mut k = 0;
        while k < insts.len() {
            let found = insts.get(k + 1).and_then(|addc| matched(f, &defs, &uses, &insts[k], addc));
            let Some((d, base, y)) = found else {
                let inst = insts[k].clone();
                if inst.op.flags() & F_DEF_EXEC != 0 || inst.defs.iter().any(|o| matches!(o, Opnd::P(Phys::ExecLo, _))) {
                    pairs.clear();
                }
                out.push(inst);
                k += 1;
                continue;
            };
            let once = defs[y.reg as usize] <= 1;
            let p = match pairs.get(&y).filter(|_| once) {
                Some(&p) => p,
                None => {
                    f.vregs.push(VRegInfo { class: Class::V, size: 2, align: 1, fixed: None });
                    let p = VRef { reg: f.vregs.len() as u32 - 1, off: 0, n: 2 };
                    out.push(MInst::new(Opc::Copy, [Opnd::R(VRef { n: 1, ..p })], [Opnd::R(y)]));
                    out.push(MInst::new(Opc::VMovB32, [Opnd::R(VRef { off: 1, n: 1, ..p })], [Opnd::Imm(0)]));
                    if once {
                        pairs.insert(y, p);
                    }
                    p
                }
            };
            out.push(MInst::new(Opc::Gfx9(Gfx9Op::VLshlAddU64), [Opnd::R(d)], [Opnd::R(p), Opnd::Imm(0), base]));
            fused += 1;
            k += 2;
        }
        f.blocks[b].insts = out;
    }
    fused
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::Target;

    fn r(reg: u32, off: u8, n: u8) -> Opnd {
        Opnd::R(VRef { reg, off, n })
    }

    /// Registers: base s[0:1] (0), offsets v1 and v2 (1, 2), results (3, 4)
    /// and carries (5, 6).
    fn func(insts: Vec<MInst>) -> MFunc {
        let mut f = MFunc { target: Target::Gfx950, ..MFunc::default() };
        let regs = [(Class::S, 2), (Class::V, 1), (Class::V, 1), (Class::V, 2), (Class::V, 2), (Class::S, 2), (Class::S, 2)];
        for (class, size) in regs {
            f.vregs.push(VRegInfo { class, size, align: 1, fixed: None });
        }
        f.blocks.push(MBlock { insts, loop_depth: 0 });
        f
    }

    /// `d = s[0:1] + zext(y)` as selection writes it, with carry `c`.
    fn add(d: u32, y: u32, c: u32) -> [MInst; 2] {
        [
            MInst::new(Opc::VAddCoU32, [r(d, 0, 1), r(c, 0, 2)], [r(0, 0, 1), r(y, 0, 1)]),
            MInst::new(Opc::VAddCoCiU32, [r(d, 1, 1), Opnd::P(Phys::Null, 2)], [r(0, 1, 1), Opnd::Imm(0), r(c, 0, 2)]),
        ]
    }

    #[test]
    fn an_add_of_a_zero_extended_offset_is_one_shift_add_of_its_pair() {
        // Two bases' worth of adds of one offset: one pair, two shift-adds.
        let mut insts: Vec<MInst> = add(3, 1, 5).into();
        insts.extend(add(4, 1, 6));
        let mut f = func(insts);
        assert_eq!(run(&mut f), 2);
        let ops: Vec<Opc> = f.blocks[0].insts.iter().map(|i| i.op).collect();
        let shift_add = Opc::Gfx9(Gfx9Op::VLshlAddU64);
        assert_eq!(ops, [Opc::Copy, Opc::VMovB32, shift_add, shift_add]);
        let pair = r(7, 0, 2);
        assert_eq!(&f.blocks[0].insts[2].uses[..], &[pair, Opnd::Imm(0), r(0, 0, 2)]);
        assert_eq!((f.blocks[0].insts[0].uses[0], f.blocks[0].insts[1].uses[0]), (r(1, 0, 1), Opnd::Imm(0)));
        assert_eq!((f.blocks[0].insts[2].defs[0], f.blocks[0].insts[3].defs[0]), (r(3, 0, 2), r(4, 0, 2)));
    }

    #[test]
    fn an_exec_write_or_a_carry_read_elsewhere_keeps_them_apart() {
        // EXEC written between: the second add makes its own pair.
        let mut insts: Vec<MInst> = add(3, 1, 5).into();
        insts.push(MInst::new(Opc::SMovB64, [Opnd::P(Phys::ExecLo, 2)], [Opnd::Imm(u32::MAX)]));
        insts.extend(add(4, 1, 6));
        let mut f = func(insts);
        assert_eq!(run(&mut f), 2);
        assert_eq!(f.blocks[0].insts.iter().filter(|i| i.op == Opc::Copy).count(), 2);
        // A carry also read by a select stays a carry.
        let mut insts: Vec<MInst> = add(3, 1, 5).into();
        insts.push(MInst::new(Opc::VCndmaskB32, [r(2, 0, 1)], [Opnd::Imm(0), Opnd::Imm(1), r(5, 0, 2)]));
        let mut f = func(insts);
        assert_eq!(run(&mut f), 0);
        // A high half that is not zero is two pairs' add (`select`'s).
        let [lo, mut hi] = add(3, 1, 5);
        hi.uses[1] = r(2, 0, 1);
        let mut f = func(vec![lo, hi]);
        assert_eq!(run(&mut f), 0);
    }
}
