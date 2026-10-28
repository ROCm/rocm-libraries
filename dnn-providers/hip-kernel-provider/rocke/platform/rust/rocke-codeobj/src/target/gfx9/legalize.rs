// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! gfx9 operand rules on virtual registers, after instruction selection and
//! before allocation: what the shared selection leaves that gfx950 cannot
//! encode becomes what it can (the encoder refuses whatever is left).
//!
//! - A literal only in a VOP1 or VOP2 short form. A VOP3-encoded
//!   instruction (VOP3B, a compare, VOP3P) reads it from an SGPR when
//!   nothing else of it uses the constant bus, else from a VGPR. A block
//!   in a loop moves each literal into a register once and reuses it while
//!   each use is within `REUSE` instructions of the last (a VGPR one up to
//!   the next EXEC write, past which it may lack lanes): the rows of an
//!   attention tile read the same exponent bounds and infinities, which
//!   COMGR keeps in registers across the loop. Straight-line code moves a
//!   literal at each use: reused there, four G1 kernels (elementwise,
//!   `topk_softmax`) ran 3 to 7% slower.
//! - One constant-bus read per VALU instruction: an SGPR tuple, a special
//!   register, a literal, or the implicit VCC of `v_div_fmas`. Sources that
//!   may be VGPRs take copies of the other scalars; scalar-only operands
//!   (lane masks, lane selects) keep the bus.
//! - No null register: a discarded scalar result goes to a dead SGPR pair.
//! - VGPR tuples at even registers: tuple vregs are 2-aligned, and a tuple
//!   operand at an odd offset of one goes through an aligned copy.

use super::asm::{literal_of, valu_lits, Lit};
use super::gen_isa::{self as t, Class as Kind, Form};
use crate::rir::mir::*;

pub fn run(f: &mut MFunc) {
    for info in f.vregs.iter_mut() {
        if info.class == Class::V && info.size >= 2 {
            info.align = info.align.max(2);
        }
    }
    let reuse = !crate::rir::debug_flag("noliteralreuse");
    let mut blocks = std::mem::take(&mut f.blocks);
    for block in &mut blocks {
        let mut out = Vec::with_capacity(block.insts.len());
        let mut cache = Literals { reuse: reuse && block.loop_depth > 0, held: Vec::new() };
        for inst in block.insts.drain(..) {
            legalize(f, inst, &mut out, &mut cache);
        }
        block.insts = out;
    }
    f.blocks = blocks;
}

/// Instructions within which a literal register of the block serves another use.
const REUSE: usize = 64;

/// The literal registers of the block being legalized: (value, dwords, literal kind, class), the register, and
/// the output position of its last use.
struct Literals {
    reuse: bool,
    held: Vec<((u32, u8, Lit, Class), Opnd, usize)>,
}

impl Literals {
    /// A register holding literal `v` for source `k`'s kind: the block's, if a use within `REUSE`
    /// instructions read it, else a fresh one (`materialize`).
    fn get(&mut self, f: &mut MFunc, class: Class, v: u32, dwords: u8, lit: Lit, out: &mut Vec<MInst>) -> Opnd {
        let key = (v, dwords, lit, class);
        let at = out.len();
        if let Some(e) = self.held.iter_mut().find(|e| e.0 == key && at - e.2 <= REUSE) {
            e.2 = at;
            return e.1;
        }
        let reg = materialize(f, class, v, dwords, lit, out);
        if self.reuse {
            self.held.retain(|e| e.0 != key);
            self.held.push((key, reg, out.len()));
        }
        reg
    }

    /// After an EXEC write a VGPR literal may lack the lanes it now runs in.
    fn exec_written(&mut self) {
        self.held.retain(|e| e.0 .3 != Class::V);
    }
}

fn vreg(f: &mut MFunc, class: Class, size: u8) -> VRef {
    let align = match (class, size) {
        (_, 1) => 1,
        (Class::V, _) | (Class::S, 2) => 2,
        (Class::S, _) => 4,
    };
    let reg = f.vregs.len() as u32;
    f.vregs.push(VRegInfo { class, size, align, fixed: None });
    VRef { reg, off: 0, n: size }
}

fn class_of(f: &MFunc, o: &Opnd) -> Option<Class> {
    match o {
        Opnd::R(v) => Some(f.vregs[v.reg as usize].class),
        Opnd::P(Phys::V(_), _) => Some(Class::V),
        Opnd::P(..) => Some(Class::S),
        Opnd::Imm(_) => None,
    }
}

fn legalize(f: &mut MFunc, mut inst: MInst, out: &mut Vec<MInst>, cache: &mut Literals) {
    for d in inst.defs.iter_mut() {
        if matches!(d, Opnd::P(Phys::Null, _)) {
            *d = Opnd::R(vreg(f, Class::S, 2));
        }
    }
    let mut after = Vec::new();
    for k in 0..inst.uses.len() {
        if let Some(tmp) = misaligned(f, &inst.uses[k]) {
            out.push(MInst::new(Opc::Copy, [Opnd::R(tmp)], [inst.uses[k]]));
            let old = inst.uses[k];
            for u in inst.uses.iter_mut().filter(|u| **u == old) {
                *u = Opnd::R(tmp);
            }
        }
    }
    for k in 0..inst.defs.len() {
        if let Some(tmp) = misaligned(f, &inst.defs[k]) {
            after.push(MInst::new(Opc::Copy, [inst.defs[k]], [Opnd::R(tmp)]));
            inst.defs[k] = Opnd::R(tmp);
        }
    }
    if let (Some(info), Some(sig)) = (t::info(inst.op), t::signature(inst.op)) {
        if info.table && matches!(info.fmt, Fmt::Vop1 | Fmt::Vop2 | Fmt::Vop3 | Fmt::Vop3b | Fmt::Vopc | Fmt::Vop3p) {
            constant_bus(f, &mut inst, info, sig, out, cache);
        }
    }
    let exec = inst.op.flags() & F_DEF_EXEC != 0 || inst.defs.iter().any(|d| matches!(d, Opnd::P(Phys::ExecLo, _)));
    out.push(inst);
    out.extend(after);
    if exec {
        cache.exec_written();
    }
}

/// An aligned replacement for `o` when it is a VGPR tuple at an odd offset.
fn misaligned(f: &mut MFunc, o: &Opnd) -> Option<VRef> {
    match *o {
        Opnd::R(v) if v.n >= 2 && v.off % 2 == 1 && f.vregs[v.reg as usize].class == Class::V => Some(vreg(f, Class::V, v.n)),
        _ => None,
    }
}

/// Whether the encoder gives `inst` a VOP1 or VOP2 short form, the only
/// ones with a literal (gfx9 has no VOP3 literal).
fn short_form(f: &MFunc, inst: &MInst, info: t::Info) -> bool {
    let neg = M_NEG0 | M_NEG1 | M_NEG2;
    match info.fmt {
        Fmt::Vop1 => {
            !inst.defs.is_empty()
                && inst.mods & neg == 0
                && info.form != Form::Dpp16
                && !(info.form == Form::SdwaHi && inst.mods & M_OPSEL_SRC0 != 0)
        }
        Fmt::Vop2 => {
            inst.uses.len() == 2 && class_of(f, &inst.uses[1]) == Some(Class::V) && inst.mods == 0 && info.form != Form::Noshort
        }
        _ => false,
    }
}

/// What a source reads over the constant bus: an SGPR range by its first
/// dword, a special register, a literal, or the implicit VCC.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Scalar {
    Reg(u32, u8),
    Phys(Phys),
    Literal(u32),
    ImplicitVcc,
}

fn scalar(f: &MFunc, inst: &MInst, k: usize, lit: Lit) -> Option<Scalar> {
    match inst.uses[k] {
        Opnd::R(v) if f.vregs[v.reg as usize].class == Class::S => Some(Scalar::Reg(v.reg, v.off)),
        Opnd::R(_) | Opnd::P(Phys::V(_), _) => None,
        // v_writelane's M0 lane select does not use the bus.
        Opnd::P(Phys::M0, _) if inst.op == Opc::VWritelaneB32 && k == 1 => None,
        Opnd::P(p, _) => Some(Scalar::Phys(p)),
        o @ Opnd::Imm(_) => literal_of(&o, lit).map(Scalar::Literal),
    }
}

/// A register holding immediate `v` as source `k` reads it: one dword, or
/// the pair a 64-bit source reads (`Opnd::Imm`: an f64's high dword, a
/// zero-extended 64-bit integer).
fn materialize(f: &mut MFunc, class: Class, v: u32, dwords: u8, lit: Lit, out: &mut Vec<MInst>) -> Opnd {
    let dst = vreg(f, class, dwords.max(1));
    if dwords >= 2 {
        let (lo, hi) = if lit == Lit::F64 { (0, v) } else { (v, 0) };
        out.push(MInst::new(Opc::Copy, [Opnd::R(VRef { n: 1, ..dst })], [Opnd::Imm(lo)]));
        out.push(MInst::new(Opc::Copy, [Opnd::R(VRef { off: 1, n: 1, ..dst })], [Opnd::Imm(hi)]));
    } else {
        out.push(MInst::new(Opc::Copy, [Opnd::R(dst)], [Opnd::Imm(v)]));
    }
    Opnd::R(dst)
}

fn constant_bus(f: &mut MFunc, inst: &mut MInst, info: t::Info, sig: &t::Sig, out: &mut Vec<MInst>, cache: &mut Literals) {
    let lits = valu_lits(info.form, info.flags);
    let encoded = if info.form == Form::Writelane { 2 } else { inst.uses.len().min(3) };
    let dwords = |k: usize| sig.uses.get(k).map_or(1, |o| o.dwords);
    let movable = |k: usize| matches!(sig.uses.get(k).map(|o| o.class), Some(Kind::Vs | Kind::Vi | Kind::Vr | Kind::V));
    let reads = |f: &MFunc, inst: &MInst| -> Vec<(usize, Scalar)> {
        (0..encoded).filter_map(|k| scalar(f, inst, k, lits[k]).map(|s| (k, s))).collect()
    };
    let implicit = info.flags & F_USE_VCC != 0;
    // Literals out of a VOP3 encoding: the first into an SGPR when no other
    // scalar needs the bus, the rest into VGPRs.
    if !short_form(f, inst, info) {
        let all = reads(f, inst);
        let mut bus_free = !implicit && all.iter().all(|(_, s)| matches!(s, Scalar::Literal(_)));
        let mut done: Vec<((u32, u8, Lit), Opnd)> = Vec::new();
        for &(k, s) in &all {
            let Scalar::Literal(v) = s else { continue };
            let key = (v, dwords(k), lits[k]);
            let reg = match done.iter().find(|(x, _)| *x == key) {
                Some(&(_, reg)) => reg,
                None => {
                    let class = if bus_free { Class::S } else { Class::V };
                    bus_free = false;
                    let reg = cache.get(f, class, v, dwords(k), lits[k], out);
                    done.push((key, reg));
                    reg
                }
            };
            inst.uses[k] = reg;
        }
    }
    // One read left: the scalar-only operands and the implicit VCC keep it,
    // then the first scalar of a source that may be a VGPR.
    let all = reads(f, inst);
    let mut kept: Vec<Scalar> = Vec::new();
    if implicit {
        kept.push(Scalar::ImplicitVcc);
    }
    for &(k, s) in &all {
        if !movable(k) && !kept.contains(&s) {
            kept.push(s);
        }
    }
    for &(k, s) in &all {
        if kept.contains(&s) {
            continue;
        }
        if kept.is_empty() {
            kept.push(s);
            continue;
        }
        let copy = match s {
            Scalar::Literal(v) => cache.get(f, Class::V, v, dwords(k), lits[k], out),
            _ => {
                let src = inst.uses[k];
                let dst = vreg(f, Class::V, src.dwords());
                out.push(MInst::new(Opc::Copy, [Opnd::R(dst)], [src]));
                Opnd::R(dst)
            }
        };
        for j in 0..encoded {
            if scalar(f, inst, j, lits[j]) == Some(s) {
                inst.uses[j] = copy;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn func(vregs: &[(Class, u8)]) -> MFunc {
        let mut f = MFunc { target: crate::target::Target::Gfx950, ..MFunc::default() };
        for &(class, size) in vregs {
            vreg(&mut f, class, size);
        }
        f
    }

    fn r(reg: u32, n: u8) -> Opnd {
        Opnd::R(VRef { reg, off: 0, n })
    }

    fn run_on(f: &mut MFunc, insts: Vec<MInst>) -> Vec<MInst> {
        f.blocks = vec![MBlock { insts, loop_depth: 0 }];
        run(f);
        f.blocks[0].insts.clone()
    }

    /// `run_on` with the block in a loop.
    fn run_in_loop(f: &mut MFunc, insts: Vec<MInst>) -> Vec<MInst> {
        f.blocks = vec![MBlock { insts, loop_depth: 1 }];
        run(f);
        f.blocks[0].insts.clone()
    }

    fn class(f: &MFunc, o: &Opnd) -> Class {
        class_of(f, o).expect("a register")
    }

    #[test]
    fn literals_leave_vop3_encodings() {
        // v_fma_f32 v0, v1, v2, 0x12345678: the literal moves to an SGPR.
        let mut f = func(&[(Class::V, 1), (Class::V, 1), (Class::V, 1)]);
        let code = run_on(&mut f, vec![MInst::new(Opc::VFmaF32, [r(0, 1)], [r(1, 1), r(2, 1), Opnd::Imm(0x1234_5678)])]);
        assert_eq!(code.len(), 2);
        assert_eq!((code[0].op, code[0].uses[0]), (Opc::Copy, Opnd::Imm(0x1234_5678)));
        assert_eq!(code[1].uses[2], code[0].defs[0]);
        assert_eq!(class(&f, &code[1].uses[2]), Class::S);
        // v_cndmask_b32 with a lane mask in an SGPR pair: the mask has the
        // bus, so the literal goes to a VGPR.
        let mut f = func(&[(Class::V, 1), (Class::V, 1), (Class::S, 2)]);
        let code = run_on(&mut f, vec![MInst::new(Opc::VCndmaskB32, [r(0, 1)], [Opnd::Imm(0x4280_0000), r(1, 1), r(2, 2)])]);
        assert_eq!(class(&f, &code[1].uses[0]), Class::V);
        assert_eq!(code[1].uses[2], r(2, 2));
        // A VOP2 short form keeps its literal; inline constants never move.
        let mut f = func(&[(Class::V, 1), (Class::V, 1)]);
        let code = run_on(&mut f, vec![MInst::new(Opc::VMulF32, [r(0, 1)], [Opnd::Imm(0x4f7f_fffe), r(1, 1)])]);
        assert_eq!(code.len(), 1);
        let code = run_on(&mut f, vec![MInst::new(Opc::VFmaF32, [r(0, 1)], [r(1, 1), Opnd::Imm(0x3f80_0000), Opnd::Imm(64)])]);
        assert_eq!(code.len(), 1);
        // f64 sources read an immediate as the high dword of the pair.
        let mut f = func(&[(Class::V, 2), (Class::V, 2), (Class::V, 2)]);
        let code = run_on(&mut f, vec![MInst::new(Opc::VFmaF64, [r(0, 2)], [r(1, 2), r(2, 2), Opnd::Imm(0x4000_1000)])]);
        assert_eq!(code.iter().map(|i| i.uses[0]).take(2).collect::<Vec<_>>(), [Opnd::Imm(0), Opnd::Imm(0x4000_1000)]);
        assert_eq!(code[2].uses[2].dwords(), 2);
    }

    #[test]
    fn a_block_in_a_loop_moves_a_literal_into_a_register_once() {
        // Two compares and two selects reading one literal: one SGPR and one VGPR for all four.
        let mut f = func(&[(Class::V, 1), (Class::S, 2), (Class::S, 2), (Class::V, 1), (Class::V, 1)]);
        let inf = Opnd::Imm(0x7f80_0000);
        let cmp = |d: u32| MInst::new(Opc::VCmp(0x11), [r(d, 2)], [inf, r(0, 1)]);
        let sel = |d: u32, m: u32| MInst::new(Opc::VCndmaskB32, [r(d, 1)], [inf, r(0, 1), r(m, 2)]);
        let code = run_in_loop(&mut f, vec![cmp(1), sel(3, 1), cmp(2), sel(4, 2)]);
        let copies: Vec<&MInst> = code.iter().filter(|i| i.op == Opc::Copy).collect();
        assert_eq!(copies.len(), 2, "{code:?}");
        let classes: Vec<Class> = copies.iter().map(|c| class(&f, &c.defs[0])).collect();
        assert_eq!(classes, [Class::S, Class::V]);
        let reads = |k: usize| code.iter().filter(|i| i.op != Opc::Copy).map(|i| i.uses[k]).collect::<Vec<_>>();
        assert_eq!(reads(0), [copies[0].defs[0], copies[1].defs[0], copies[0].defs[0], copies[1].defs[0]]);
        // Straight-line code moves it at each use.
        let code = run_on(&mut f, vec![cmp(1), sel(3, 1), cmp(2), sel(4, 2)]);
        assert_eq!(code.iter().filter(|i| i.op == Opc::Copy).count(), 4, "{code:?}");
        // Past an EXEC write the VGPR is moved again; the SGPR stays.
        let exec = MInst::new(Opc::SAndSaveexecB64, [r(2, 2)], [r(1, 2)]);
        let code = run_in_loop(&mut f, vec![cmp(1), sel(3, 1), exec, cmp(2), sel(4, 2)]);
        let copies: Vec<&MInst> = code.iter().filter(|i| i.op == Opc::Copy).collect();
        assert_eq!(copies.iter().map(|c| class(&f, &c.defs[0])).collect::<Vec<_>>(), [Class::S, Class::V, Class::V]);
        // A use more than REUSE instructions after the last moves it again.
        let mut far = vec![cmp(1)];
        far.extend((0..REUSE).map(|_| MInst::new(Opc::VAddF32, [r(3, 1)], [r(0, 1), r(0, 1)])));
        far.push(cmp(2));
        let code = run_in_loop(&mut f, far);
        assert_eq!(code.iter().filter(|i| i.op == Opc::Copy).count(), 2);
    }

    #[test]
    fn one_scalar_reaches_the_constant_bus() {
        // Two SGPRs in a VOP2: the second goes to a VGPR, giving a short form.
        let mut f = func(&[(Class::V, 1), (Class::S, 1), (Class::S, 1)]);
        let code = run_on(&mut f, vec![MInst::new(Opc::VAddF32, [r(0, 1)], [r(1, 1), r(2, 1)])]);
        assert_eq!((code[0].op, code[0].uses[0]), (Opc::Copy, r(2, 1)));
        assert_eq!((code[1].uses[0], class(&f, &code[1].uses[1])), (r(1, 1), Class::V));
        // The same SGPR twice is one read.
        let code = run_on(&mut f, vec![MInst::new(Opc::VFmaF32, [r(0, 1)], [r(1, 1), r(1, 1), r(0, 1)])]);
        assert_eq!(code.len(), 1);
        // v_div_fmas reads VCC: no SGPR source is left.
        let code = run_on(&mut f, vec![MInst::new(Opc::VDivFmasF32, [r(0, 1)], [r(1, 1), r(0, 1), r(0, 1)])]);
        assert_eq!((code.len(), class(&f, &code[1].uses[0])), (2, Class::V));
        // A carry-in mask keeps the bus over an SGPR addend.
        let mut f = func(&[(Class::V, 1), (Class::S, 2), (Class::S, 1), (Class::V, 1), (Class::S, 2)]);
        let code = run_on(&mut f, vec![MInst::new(Opc::VAddCoCiU32, [r(0, 1), r(1, 2)], [r(2, 1), r(3, 1), r(4, 2)])]);
        assert_eq!((class(&f, &code[1].uses[0]), code[1].uses[2]), (Class::V, r(4, 2)));
    }

    #[test]
    fn null_results_and_odd_tuples_get_registers() {
        let mut f = func(&[(Class::V, 1), (Class::V, 1), (Class::V, 4)]);
        let code = run_on(
            &mut f,
            vec![MInst::new(Opc::VDivScaleF32, [r(0, 1), Opnd::P(Phys::Null, 1)], [r(1, 1), r(1, 1), r(0, 1)])],
        );
        let Opnd::R(flag) = code[0].defs[1] else { panic!("{:?}", code[0]) };
        assert_eq!((f.vregs[flag.reg as usize].class, f.vregs[flag.reg as usize].size), (Class::S, 2));
        // The middle pair of a quad starts at an odd VGPR: it is copied out
        // to an aligned pair, and the quad is 2-aligned.
        let mid = Opnd::R(VRef { reg: 2, off: 1, n: 2 });
        let code = run_on(&mut f, vec![MInst::new(Opc::VAddF64, [r(0, 1)], [mid, mid])]);
        assert_eq!((code[0].op, code[0].uses[0]), (Opc::Copy, mid));
        assert_eq!((code[1].uses[0], code[1].uses[1]), (code[0].defs[0], code[0].defs[0]));
        assert_eq!(f.vregs[2].align, 2);
    }
}
