// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! gfx950 binary encoder for physical machine IR, and a text listing.
//!
//! Layouts, opcodes, operand codes and each opcode's form come from the
//! gfx950 description (`gen_isa.rs`, generated from `isa/gfx950.isa`); this
//! file applies gfx9's operand rules. What gfx9 cannot encode is refused,
//! never fixed up: a literal on a VOP3-encoded instruction (the legalizer
//! moves literals out before allocation), more than one constant-bus read
//! in a VALU instruction (SGPRs, special registers, a literal, the implicit
//! VCC of v_div_fmas), a VGPR tuple at an odd register (gfx90a and later
//! align them), the null register (gfx9 has none: code 124 is M0), SGPRs
//! past s101 (s102..s105 are FLAT_SCRATCH and XNACK_MASK), a modifier the
//! opcode does not accept (neg of an integer source among them), an opcode
//! without a gfx950 row, a virtual register, or a pseudo.

use std::fmt::Write;

use super::gen_isa as t;
use super::gen_isa::Form;
use crate::error::CompileError;
use crate::rir::mir::*;

#[cold]
#[inline(never)]
fn bad(message: impl Into<String>) -> CompileError {
    CompileError::new("ROCKE_DIRECT_ENCODE", message)
}

/// SGPRs an operand may name: s0..s101.
const LAST_SGPR: u16 = 101;

/// How a source reads an immediate that is not an inline integer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Lit {
    /// A 32-bit value: an f32 inline constant or a literal.
    B32,
    /// A literal only (a 64-bit integer source, 16-bit pairs).
    Plain,
    /// The high dword of an f64 with a zero low dword.
    F64,
}

/// The literal `o` needs when a source reads it as `lit`: an immediate that
/// is no inline constant there.
pub(super) fn literal_of(o: &Opnd, lit: Lit) -> Option<u32> {
    match *o {
        Opnd::Imm(v) => match t::inline_int(v).or_else(|| match lit {
            Lit::B32 => t::inline_f32(v),
            Lit::F64 => t::inline_f64(v),
            Lit::Plain => None,
        }) {
            Some(_) => None,
            None => Some(v),
        },
        _ => None,
    }
}

fn src(o: &Opnd, lit: Lit) -> Result<(u32, Option<u32>), CompileError> {
    Ok(match *o {
        Opnd::P(Phys::V(r), n) => (t::SRC_VGPR + aligned(r, n)?, None),
        Opnd::P(Phys::S(r), n) => (sgpr(r, n)?, None),
        Opnd::P(Phys::VccLo, _) => (t::REG_VCC_LO, None),
        Opnd::P(Phys::ExecLo, _) => (t::REG_EXEC_LO, None),
        Opnd::P(Phys::M0, _) => (t::REG_M0, None),
        Opnd::P(Phys::Scc, _) => (t::REG_SCC, None),
        Opnd::P(Phys::Null, _) => return Err(bad("gfx950 has no null register")),
        Opnd::Imm(v) => match t::inline_int(v).or_else(|| match lit {
            Lit::B32 => t::inline_f32(v),
            Lit::F64 => t::inline_f64(v),
            Lit::Plain => None,
        }) {
            Some(code) => (code, None),
            None => (t::SRC_LITERAL, Some(v)),
        },
        Opnd::R(_) => return Err(bad("virtual register reached the gfx950 encoder")),
    })
}

/// Operand code of an SGPR tuple starting at `r`, which must end by s101.
fn sgpr(r: u16, n: u8) -> Result<u32, CompileError> {
    if r + n.max(1) as u16 - 1 > LAST_SGPR {
        return Err(bad(format!("s{r} (x{n}) is past s{LAST_SGPR}")));
    }
    Ok(t::SRC_SGPR + r as u32)
}

fn ssrc(o: &Opnd, lit: Lit) -> Result<(u32, Option<u32>), CompileError> {
    let (code, literal) = src(o, lit)?;
    if code >= t::SRC_VGPR {
        return Err(bad(format!("{o:?} is not a scalar source")));
    }
    Ok((code, literal))
}

fn sdst(o: &Opnd) -> Result<u32, CompileError> {
    Ok(match *o {
        Opnd::P(Phys::S(r), n) => sgpr(r, n)?,
        Opnd::P(Phys::VccLo, _) => t::REG_VCC_LO,
        Opnd::P(Phys::ExecLo, _) => t::REG_EXEC_LO,
        Opnd::P(Phys::M0, _) => t::REG_M0,
        Opnd::P(Phys::Null, _) => return Err(bad("gfx950 has no null register")),
        _ => return Err(bad(format!("{o:?} is not a scalar destination"))),
    })
}

fn vgpr(o: &Opnd) -> Result<u32, CompileError> {
    match *o {
        Opnd::P(Phys::V(r), n) => aligned(r, n),
        _ => Err(bad(format!("{o:?} is not a VGPR"))),
    }
}

/// `r`, if a VGPR tuple of `n` registers may start there: gfx90a and later
/// keep tuples at even registers.
fn aligned(r: u16, n: u8) -> Result<u32, CompileError> {
    if n >= 2 && r % 2 != 0 {
        return Err(bad(format!("v{r} (x{n}): gfx950 VGPR tuples start at an even register")));
    }
    Ok(r as u32)
}

/// A scalar a VALU instruction reads over the constant bus: an SGPR tuple by
/// its first register, a special register by its code, a literal, or the
/// implicit 64-bit VCC of v_div_fmas (distinct from an explicit vcc_lo).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Scalar {
    Reg(u32),
    Literal(u32),
    ImplicitVcc,
}

/// The constant-bus reads of a VALU instruction's encoded sources, with the
/// implicit VCC of an opcode that reads it; gfx950 allows one, and
/// v_writelane an M0 lane select besides.
fn constant_bus(op: Opc, srcs: &[Opnd], lits: [Lit; 3], flags: u32) -> Result<(), CompileError> {
    let mut seen: [Option<Scalar>; 4] = [None; 4];
    let mut n = 0;
    let mut add = |s: Scalar| {
        if !seen[..n].contains(&Some(s)) {
            seen[n] = Some(s);
            n += 1;
        }
    };
    if flags & F_USE_VCC != 0 {
        add(Scalar::ImplicitVcc);
    }
    for (k, o) in srcs.iter().take(3).enumerate() {
        match *o {
            Opnd::P(Phys::V(_), _) => {}
            Opnd::P(Phys::M0, _) if op == Opc::VWritelaneB32 && k == 1 => {}
            Opnd::Imm(_) => {
                if let (_, Some(l)) = src(o, lits[k])? {
                    add(Scalar::Literal(l));
                }
            }
            _ => add(Scalar::Reg(src(o, lits[k])?.0)),
        }
    }
    if n > 1 {
        return Err(bad(format!("{}: {n} constant-bus reads (gfx950 allows one)", t::mnemonic(op).unwrap_or("?"))));
    }
    Ok(())
}

/// The neg bits an opcode's sources may take: float sources only.
fn negatable(op: Opc, allowed: u32) -> u32 {
    let neg = M_NEG0 | M_NEG1 | M_NEG2;
    match op {
        Opc::VCmp(code) => match t::vcmp(code) {
            Some((_, "v_cmp_class_f32")) => M_NEG0,
            Some((_, name)) if name.ends_with("_f32") || name.ends_with("_f64") => M_NEG0 | M_NEG1,
            _ => 0,
        },
        Opc::VLdexpF32 | Opc::VLdexpF64 => M_NEG0,
        _ => allowed & neg,
    }
}

fn is_vgpr(o: &Opnd) -> bool {
    matches!(o, Opnd::P(Phys::V(_), _))
}

fn nth(opnds: &[Opnd], i: usize) -> Result<&Opnd, CompileError> {
    opnds.get(i).ok_or_else(|| bad(format!("missing operand {i}")))
}

fn literal(out: &mut Vec<u32>, lits: &[Option<u32>]) -> Result<(), CompileError> {
    let mut lit: Option<u32> = None;
    for l in lits.iter().flatten() {
        match lit {
            Some(x) if x != *l => return Err(bad("two different literals in one instruction")),
            _ => lit = Some(*l),
        }
    }
    if let Some(l) = lit {
        out.push(l);
    }
    Ok(())
}

fn push2(out: &mut Vec<u32>, words: [u32; 2]) {
    out.extend_from_slice(&words);
}

/// A literal of a 16-bit source must fit 16 bits.
fn fits16(flags: u32, literal: Option<u32>, name: &str) -> Result<(), CompileError> {
    match literal {
        Some(l) if flags & F_16BIT_SRC != 0 && l > 0xffff => Err(bad(format!("{name}: literal {l:#x} of a 16-bit source"))),
        _ => Ok(()),
    }
}

/// How sources 0-2 of a VALU instruction read immediates.
pub(super) fn valu_lits(form: Form, flags: u32) -> [Lit; 3] {
    match form {
        Form::F64 => [Lit::F64; 3],
        Form::F64I32 => [Lit::F64, Lit::B32, Lit::B32],
        Form::Dot2 => [Lit::Plain, Lit::Plain, Lit::B32],
        Form::Writelane | Form::Src16 => [Lit::Plain; 3],
        _ if flags & (F_64BIT_SRC | F_16BIT_SRC) != 0 => [Lit::Plain; 3],
        _ => [Lit::B32; 3],
    }
}

fn salu_lit(form: Form, k: usize) -> Lit {
    match form {
        Form::B64 => Lit::Plain,
        Form::B64Shift if k == 0 => Lit::Plain,
        _ => Lit::B32,
    }
}

/// A VOP3 (or, with `sdst`, VOP3B) instruction: up to three sources and
/// no literal.
fn vop3(
    out: &mut Vec<u32>,
    op: u32,
    dst: u32,
    sdst: Option<u32>,
    srcs: &[Opnd],
    mods: u32,
    reads: [Lit; 3],
    name: &str,
) -> Result<(), CompileError> {
    if srcs.len() > 3 {
        return Err(bad("more than three VOP3 sources"));
    }
    let mut codes = [0u32; 3];
    for (i, s) in srcs.iter().enumerate() {
        let (code, lit) = src(s, reads[i])?;
        if lit.is_some() {
            return Err(bad(format!("{name}: a literal in a VOP3 encoding (gfx950 has none)")));
        }
        codes[i] = code;
    }
    let [src0, src1, src2] = codes;
    let neg = mods & (M_NEG0 | M_NEG1 | M_NEG2);
    push2(out, match sdst {
        Some(sdst) => t::EncVop3b { op, vdst: dst, sdst, src0, src1, src2, neg, ..Default::default() }.pack(),
        None => {
            let op_sel = (mods / M_OPSEL_SRC0) & 0xf;
            t::EncVop3 { op, vdst: dst, op_sel, src0, src1, src2, neg, ..Default::default() }.pack()
        }
    });
    Ok(())
}

/// Whether `ctrl`, the DPP control field, is one gfx9 has: quad_perm,
/// row_shl/shr/ror, the wave shifts and rotates, row_mirror,
/// row_half_mirror and row_bcast 15/31 (no row_share or row_xmask).
fn dpp_control_on_gfx9(ctrl: u32) -> bool {
    matches!(ctrl, 0x000..=0x0ff | 0x101..=0x10f | 0x111..=0x11f | 0x121..=0x12f | 0x130 | 0x134 | 0x138 | 0x13c | 0x140..=0x143)
}

/// Encodes one instruction into `out`.
pub fn encode(inst: &MInst, out: &mut Vec<u32>) -> Result<(), CompileError> {
    let op = inst.op;
    let name = || t::mnemonic(op).unwrap_or("?");
    let Some(info) = t::info(op) else {
        return Err(bad(format!("{} has no gfx950 encoding", op.name())));
    };
    if !info.table {
        return Err(bad(format!("pseudo {} reached the gfx950 encoder", op.name())));
    }
    let sig = t::signature(op).ok_or_else(|| bad(format!("{} has no gfx950 signature", op.name())))?;
    if inst.mods & !sig.mods != 0 || inst.mods & (M_NEG0 | M_NEG1 | M_NEG2) & !negatable(op, sig.mods) != 0 {
        return Err(bad(format!("{}: modifiers {:#x} beyond those gfx950 accepts ({:#x})", name(), inst.mods, sig.mods)));
    }
    let (code, vop3_code, flags, form) = (info.code as u32, info.vop3 as u32, info.flags, info.form);
    let (defs, uses, mods) = (&inst.defs[..], &inst.uses[..], inst.mods);
    if matches!(info.fmt, Fmt::Vop1 | Fmt::Vop2 | Fmt::Vop3 | Fmt::Vop3b | Fmt::Vopc | Fmt::Vop3p) {
        let encoded = if form == Form::Writelane { &uses[..uses.len().min(2)] } else { uses };
        constant_bus(op, encoded, valu_lits(form, flags), flags)?;
        if matches!(op, Opc::VReadlaneB32 | Opc::VReadfirstlaneB32) && matches!(defs.first(), Some(Opnd::P(Phys::M0, _))) {
            return Err(bad(format!("{}: M0 is not a lane-read destination on gfx950", name())));
        }
    }
    match info.fmt {
        Fmt::Sop1 => {
            let sdst = sdst(nth(defs, 0)?)?;
            let (ssrc0, l0) = ssrc(nth(uses, 0)?, salu_lit(form, 0))?;
            out.push(t::EncSop1 { op: code, sdst, ssrc0 }.pack());
            literal(out, &[l0])?;
        }
        Fmt::Sop2 => {
            let sdst = sdst(nth(defs, 0)?)?;
            let (ssrc0, l0) = ssrc(nth(uses, 0)?, salu_lit(form, 0))?;
            let (ssrc1, l1) = ssrc(nth(uses, 1)?, salu_lit(form, 1))?;
            out.push(t::EncSop2 { op: code, sdst, ssrc0, ssrc1 }.pack());
            literal(out, &[l0, l1])?;
        }
        Fmt::Sopc => {
            let (ssrc0, l0) = ssrc(nth(uses, 0)?, salu_lit(form, 0))?;
            let (ssrc1, l1) = ssrc(nth(uses, 1)?, salu_lit(form, 1))?;
            out.push(t::EncSopc { op: code, ssrc0, ssrc1 }.pack());
            literal(out, &[l0, l1])?;
        }
        Fmt::Sopk => {
            let sdst = sdst(nth(defs, 0)?)?;
            out.push(t::EncSopk { op: code, sdst, simm16: inst.imm as u32 & 0xffff }.pack());
        }
        Fmt::Sopp => out.push(t::EncSopp { op: code, simm16: inst.imm as u32 & 0xffff }.pack()),
        Fmt::Smem => {
            if matches!(nth(defs, 0)?, Opnd::P(Phys::M0 | Phys::ExecLo, _)) {
                return Err(bad(format!("{}: M0 and EXEC are not SMEM destinations", name())));
            }
            let sdata = sdst(nth(defs, 0)?)?;
            let Opnd::P(Phys::S(base), 2) = *nth(uses, 0)? else {
                return Err(bad("SMEM base is not an SGPR pair"));
            };
            let offset = inst.imm;
            if !(-(1 << 20)..1 << 20).contains(&offset) {
                return Err(bad(format!("SMEM offset {offset} out of range")));
            }
            let sbase = sgpr(base, 2)? >> 1;
            push2(out, t::EncSmem { op: code, sdata, sbase, offset: offset as u32 & 0x1f_ffff, ..Default::default() }.pack());
        }
        Fmt::Vop1 if defs.is_empty() => out.push(t::EncVop1 { op: code, ..Default::default() }.pack()),
        Fmt::Vop1 if form == Form::Dpp16 => {
            let vdst = vgpr(nth(defs, 0)?)?;
            let src0 = vgpr(nth(uses, 0)?)?;
            let control = inst.imm as u32 & 0xff_ffff;
            if !dpp_control_on_gfx9(control & 0x1ff) || control & 1 << 10 != 0 {
                return Err(bad(format!("DPP control {control:#x} is not a gfx9 control")));
            }
            out.push(t::EncVop1 { op: code, vdst, src0: t::SRC_DPP }.pack());
            out.push(t::EncDpp { src0, ctrl: control }.pack());
        }
        Fmt::Vop1 if form == Form::SdwaHi && mods & M_OPSEL_SRC0 != 0 => {
            // The high half of a 16-bit source: SDWA with src0_sel WORD_1,
            // a full-dword result (dst_sel DWORD, dst_unused UNUSED_PAD).
            let vdst = vgpr(nth(defs, 0)?)?;
            let s = nth(uses, 0)?;
            let (src0, s0) = match *s {
                Opnd::P(Phys::V(r), _) => (r as u32, 0),
                Opnd::P(Phys::S(r), n) => (sgpr(r, n)?, 1),
                _ => return Err(bad(format!("{}: an SDWA source is a VGPR or an SGPR", name()))),
            };
            out.push(t::EncVop1 { op: code, vdst, src0: t::SRC_SDWA }.pack());
            let src0_neg = u32::from(mods & M_NEG0 != 0);
            out.push(t::EncSdwa { src0, dst_sel: 6, src0_sel: 5, src0_neg, s0, ..Default::default() }.pack());
        }
        Fmt::Vop1 => {
            let d = nth(defs, 0)?;
            let dst = if is_vgpr(d) { vgpr(d)? } else { sdst(d)? };
            let reads = valu_lits(form, flags);
            if mods & (M_NEG0 | M_NEG1 | M_NEG2) == 0 {
                let (src0, l0) = src(nth(uses, 0)?, reads[0])?;
                fits16(flags, l0, name())?;
                out.push(t::EncVop1 { op: code, vdst: dst, src0 }.pack());
                literal(out, &[l0])?;
            } else {
                vop3(out, vop3_code, dst, None, &uses[..1], mods, reads, name())?;
            }
        }
        Fmt::Vop2 => {
            let vdst = vgpr(nth(defs, 0)?)?;
            let (s0, s1) = (nth(uses, 0)?, nth(uses, 1)?);
            let reads = valu_lits(form, flags);
            if uses.len() == 2 && is_vgpr(s1) && mods == 0 && form != Form::Noshort {
                let (src0, l0) = src(s0, reads[0])?;
                fits16(flags, l0, name())?;
                out.push(t::EncVop2 { op: code, vdst, vsrc1: vgpr(s1)?, src0 }.pack());
                literal(out, &[l0])?;
            } else {
                vop3(out, vop3_code, vdst, None, uses, mods, reads, name())?;
            }
        }
        Fmt::Vop3 if form == Form::Writelane => {
            let vdst = vgpr(nth(defs, 0)?)?;
            nth(uses, 1)?;
            vop3(out, vop3_code, vdst, None, &uses[..2], mods, valu_lits(form, flags), name())?;
        }
        Fmt::Vop3 => {
            // A fourth use is the destination's old value, not encoded.
            let vdst = vgpr(nth(defs, 0)?)?;
            vop3(out, vop3_code, vdst, None, &uses[..uses.len().min(3)], mods, valu_lits(form, flags), name())?;
        }
        Fmt::Vop3b => {
            let vdst = vgpr(nth(defs, 0)?)?;
            let sdst = sdst(nth(defs, 1)?)?;
            vop3(out, vop3_code, vdst, Some(sdst), uses, mods, valu_lits(form, flags), name())?;
        }
        Fmt::Vopc => {
            let sdst = sdst(nth(defs, 0)?)?;
            vop3(out, vop3_code, sdst, None, uses, mods, valu_lits(form, flags), name())?;
        }
        Fmt::Vop3p => {
            let vdst = vgpr(nth(defs, 0)?)?;
            // Packed sources read both halves (op_sel_hi all set, the unused
            // third source's too, as llvm-mc writes it); a mix opcode takes
            // op_sel and op_sel_hi from the modifiers.
            let (op_sel, op_sel_hi) =
                if form == Form::Mix { ((mods / M_OPSEL_SRC0) & 7, (mods / M_MIX_F16_0) & 7) } else { (0, 7) };
            let reads = valu_lits(form, flags);
            let mut codes = [0u32; 3];
            for ((c, o), &read) in codes.iter_mut().zip(uses).zip(&reads) {
                let (code, lit) = src(o, read)?;
                if lit.is_some() {
                    return Err(bad(format!("{}: a literal in a VOP3P encoding (gfx950 has none)", name())));
                }
                *c = code;
            }
            let [src0, src1, src2] = codes;
            let neg = mods & (M_NEG0 | M_NEG1 | M_NEG2);
            let words = t::EncVop3p {
                op: code,
                vdst,
                op_sel,
                op_sel_hi2: op_sel_hi >> 2,
                op_sel_hi: op_sel_hi & 3,
                src0,
                src1,
                src2,
                neg,
                ..Default::default()
            };
            push2(out, words.pack());
        }
        Fmt::Ds => {
            if !(0..=0xffff).contains(&inst.imm) {
                return Err(bad(format!("{}: LDS offset {} out of range", name(), inst.imm)));
            }
            let offset = inst.imm as u32;
            let addr = vgpr(nth(uses, 0)?)?;
            let (vdst, data0) = if defs.is_empty() {
                (0, vgpr(nth(uses, 1)?)?)
            } else if form == Form::DsData {
                (vgpr(&defs[0])?, vgpr(nth(uses, 1)?)?)
            } else {
                (vgpr(&defs[0])?, 0)
            };
            let data1 = if form == Form::Ds2addr { vgpr(nth(uses, 2)?)? } else { 0 };
            push2(out, t::EncDs { op: code, offset, addr, data0, data1, vdst, ..Default::default() }.pack());
        }
        Fmt::Global if form == Form::Scratch => {
            let (vdst, data) = match defs.first() {
                Some(d) => (vgpr(d)?, 0),
                None => (0, vgpr(nth(uses, 0)?)?),
            };
            let offset = inst.imm;
            if !(-4096..4096).contains(&offset) {
                return Err(bad(format!("scratch offset {offset} out of range")));
            }
            let words = t::EncScratch { op: code, offset: offset as u32, vdst, data, saddr: t::SRC_SADDR_OFF, ..Default::default() };
            push2(out, words.pack());
        }
        Fmt::Global => {
            let saddr_mode = mods & M_SADDR != 0;
            let addr = vgpr(nth(uses, 0)?)?;
            let saddr = if saddr_mode { sdst(nth(uses, 1)?)? } else { t::SRC_SADDR_OFF };
            let next = if saddr_mode { 2 } else { 1 };
            let (vdst, data) = if defs.is_empty() {
                (0, vgpr(nth(uses, next)?)?)
            } else if flags & F_VMEM_STORE != 0 {
                // A returning atomic: both the old value and the operand.
                // Only sc0 makes it return the old value.
                if mods & M_GLC == 0 {
                    return Err(bad("a returning atomic without sc0"));
                }
                (vgpr(&defs[0])?, vgpr(nth(uses, next)?)?)
            } else {
                (vgpr(&defs[0])?, 0)
            };
            let offset = inst.imm;
            if !(-4096..4096).contains(&offset) {
                return Err(bad(format!("global offset {offset} out of range")));
            }
            let sc0 = u32::from(mods & M_GLC != 0);
            let words = t::EncGlobal { op: code, offset: offset as u32, sc0, vdst, saddr, data, addr, ..Default::default() };
            push2(out, words.pack());
        }
        Fmt::Mubuf => {
            let (vdata, rest) = if defs.is_empty() { (vgpr(nth(uses, 0)?)?, &uses[1..]) } else { (vgpr(&defs[0])?, uses) };
            let vaddr = vgpr(nth(rest, 0)?)?;
            let srsrc = match *nth(rest, 1)? {
                Opnd::P(Phys::S(r), 4) if r % 4 == 0 => sgpr(r, 4)? / 4,
                ref o => return Err(bad(format!("{o:?} is not an aligned SGPR quad"))),
            };
            let (soffset, lit) = src(nth(rest, 2)?, Lit::Plain)?;
            if lit.is_some() {
                return Err(bad("buffer soffset literal"));
            }
            let offset = inst.imm as u32;
            if offset >= 4096 {
                return Err(bad(format!("buffer offset {offset} out of range")));
            }
            push2(out, t::EncMubuf { op: code, offset, offen: 1, soffset, srsrc, vdata, vaddr, ..Default::default() }.pack());
        }
        Fmt::Vopd | Fmt::Pseudo => return Err(bad(format!("{} has no gfx950 encoding", op.name()))),
    }
    Ok(())
}

/// The encoder's words for one instruction.
pub fn encode_words(inst: &MInst) -> Result<Vec<u32>, CompileError> {
    let mut out = Vec::with_capacity(3);
    encode(inst, &mut out)?;
    Ok(out)
}

/// Encodes all blocks: the words, the word offset of each block (and the
/// end), and each branch's word offset and target block.
fn encode_blocks(f: &MFunc) -> Result<(Vec<u32>, Vec<u32>, Vec<(usize, usize)>), CompileError> {
    let mut words = Vec::with_capacity(f.blocks.iter().map(|b| 2 * b.insts.len()).sum());
    let mut block_word = Vec::with_capacity(f.blocks.len() + 1);
    let mut branches = Vec::new();
    for block in &f.blocks {
        block_word.push(words.len() as u32);
        for inst in &block.insts {
            if inst.is_branch() {
                branches.push((words.len(), inst.imm as usize));
            }
            encode(inst, &mut words)?;
        }
    }
    block_word.push(words.len() as u32);
    Ok((words, block_word, branches))
}

fn branch_delta(block_word: &[u32], at: usize, target: usize) -> Result<i64, CompileError> {
    let delta = block_word[target] as i64 - (at as i64 + 1);
    if !(-32768..32768).contains(&delta) {
        return Err(bad("branch out of range"));
    }
    Ok(delta)
}

/// Encodes all blocks, resolving branch targets to word offsets.
pub fn assemble(f: &MFunc) -> Result<Vec<u8>, CompileError> {
    let (mut words, block_word, branches) = encode_blocks(f)?;
    for (at, target) in branches {
        let delta = branch_delta(&block_word, at, target)?;
        words[at] = words[at] & 0xffff_0000 | (delta as u32 & 0xffff);
    }
    Ok(words.iter().flat_map(|w| w.to_le_bytes()).collect())
}

/// Text listing with gfx950 mnemonics, byte offsets and encodings.
pub fn listing(f: &MFunc) -> Result<String, CompileError> {
    let (_, block_word, _) = encode_blocks(f)?;
    let mut at = 0usize;
    let mut text = String::new();
    for (b, block) in f.blocks.iter().enumerate() {
        let _ = writeln!(text, "bb{b}:");
        for inst in &block.insts {
            let mut inst = inst.clone();
            if inst.is_branch() {
                inst.imm = branch_delta(&block_word, at, inst.imm as usize)? & 0xffff;
            }
            let words = encode_words(&inst)?;
            let gfx11 = inst_text(f, &inst);
            let shown = match (t::mnemonic(inst.op), gfx11.split_once(' ')) {
                (Some(name), Some((_, rest))) => format!("{name} {rest}"),
                (Some(name), None) => name.to_string(),
                (None, _) => gfx11,
            };
            let hex: Vec<String> = words.iter().map(|w| format!("{w:08X}")).collect();
            let _ = writeln!(text, "  {shown:<60} // {:06X}: {}", at * 4, hex.join(" "));
            at += words.len();
        }
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(r: u16, n: u8) -> Opnd {
        Opnd::P(Phys::V(r), n)
    }

    fn s(r: u16, n: u8) -> Opnd {
        Opnd::P(Phys::S(r), n)
    }

    fn words(inst: MInst) -> Result<Vec<u32>, String> {
        encode_words(&inst).map_err(|e| e.message)
    }

    /// Encodings llvm-mc -mcpu=gfx950 -show-encoding gives (g950a's
    /// prototype report and the S4 cross-check).
    #[test]
    fn encodings_match_llvm_mc() {
        let vcc = Opnd::P(Phys::VccLo, 2);
        let cases: Vec<(MInst, Vec<u32>)> = vec![
            // s_mov_b32 s3, s5 / s_endpgm / s_nop 0
            (MInst::new(Opc::SMovB32, [s(3, 1)], [s(5, 1)]), vec![0xbe83_0005]),
            (MInst::new(Opc::SEndpgm, [], []), vec![0xbf81_0000]),
            (MInst::new(Opc::SNop, [], []), vec![0xbf80_0000]),
            // s_load_dwordx2 s[2:3], s[0:1], 0x0
            (MInst::new(Opc::SLoadB64, [s(2, 2)], [s(0, 2)]), vec![0xc006_0080, 0x0000_0000]),
            // v_mov_b32_e32 v1, 0x3fb8aa3b
            (MInst::new(Opc::VMovB32, [v(1, 1)], [Opnd::Imm(0x3fb8_aa3b)]), vec![0x7e02_02ff, 0x3fb8_aa3b]),
            // v_add_u32_e32 v1, v9, v2 (the gfx9 no-carry add, 0x34)
            (MInst::new(Opc::VAddNcU32, [v(1, 1)], [v(9, 1), v(2, 1)]), vec![0x6802_0509]),
            // v_cndmask_b32_e64 v1, 0, v3, s[4:5]
            (MInst::new(Opc::VCndmaskB32, [v(1, 1)], [Opnd::Imm(0), v(3, 1), s(4, 2)]), vec![0xd100_0001, 0x0012_0680]),
            // v_add_co_u32_e64 v1, s[6:7], v2, v3
            (MInst::new(Opc::VAddCoU32, [v(1, 1), s(6, 2)], [v(2, 1), v(3, 1)]), vec![0xd119_0601, 0x0002_0702]),
            // v_cmp_gt_f32_e64 vcc, v1, 2 (the compare's shared code 0x14)
            (MInst::new(Opc::VCmp(0x14), [vcc], [v(1, 1), Opnd::Imm(2)]), vec![0xd044_006a, 0x0001_0501]),
            // v_fma_f32 v1, -v2, v3, v4
            (MInst::new(Opc::VFmaF32, [v(1, 1)], [v(2, 1), v(3, 1), v(4, 1)]).with_mods(M_NEG0), vec![0xd1cb_0001, 0x2412_0702]),
            // v_mbcnt_lo_u32_b32 v1, -1, 0
            (MInst::new(Opc::VMbcntLoU32B32, [v(1, 1)], [Opnd::Imm(u32::MAX), Opnd::Imm(0)]), vec![0xd28c_0001, 0x0001_00c1]),
            // ds_read_b64 v[4:5], v2 offset:1024
            (MInst::new(Opc::DsLoadB64, [v(4, 2)], [v(2, 1)]).with_imm(1024), vec![0xd8ec_0400, 0x0400_0002]),
            // global_load_dword v1, v[2:3], off offset:-64
            (MInst::new(Opc::GLoadB32, [v(1, 1)], [v(2, 2)]).with_imm(-64), vec![0xdc50_9fc0, 0x017f_0002]),
            // global_load_dword v1, v2, s[4:5] offset:16
            (MInst::new(Opc::GLoadB32, [v(1, 1)], [v(2, 1), s(4, 2)]).with_imm(16).with_mods(M_SADDR), vec![0xdc50_8010, 0x0104_0002]),
            // scratch_load_dword v1, off, off offset:16
            (MInst::new(Opc::ScratchLoadB32, [v(1, 1)], []).with_imm(16), vec![0xdc50_4010, 0x017f_0000]),
            // buffer_store_dword v1, v2, s[8:11], 0 offen offset:12
            (MInst::new(Opc::BufStoreB32, [], [v(1, 1), v(2, 1), s(8, 4), Opnd::Imm(0)]).with_imm(12), vec![0xe070_100c, 0x8002_0102]),
            // v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
            (MInst::new(Opc::VMovB32Dpp, [v(1, 1)], [v(2, 1)]).with_imm(dpp16_ctrl(0xb1, true) as i64), vec![0x7e02_02fa, 0xff08_b102]),
        ];
        for (inst, want) in cases {
            assert_eq!(words(inst.clone()), Ok(want), "{inst:?}");
        }
    }

    #[test]
    fn what_gfx9_cannot_encode_is_refused() {
        let refused = [
            // A literal in a VOP3 encoding.
            MInst::new(Opc::VFmaF32, [v(1, 1)], [v(2, 1), Opnd::Imm(0x1234_5678), v(4, 1)]),
            MInst::new(Opc::VCndmaskB32, [v(1, 1)], [Opnd::Imm(0x4280_0000), v(3, 1), s(4, 2)]),
            // No null register; s102..s105 are not SGPRs.
            MInst::new(Opc::SMovB32, [Opnd::P(Phys::Null, 1)], [s(5, 1)]),
            MInst::new(Opc::SMovB32, [s(102, 1)], [s(5, 1)]),
            MInst::new(Opc::SMovB64, [s(101, 2)], [s(0, 2)]),
            // A modifier the opcode does not take (op_sel on a 32-bit VOP3).
            MInst::new(Opc::VAdd3U32, [v(1, 1)], [v(2, 1), v(3, 1), v(4, 1)]).with_mods(M_OPSEL_SRC0),
            // Opcodes gfx950 lacks, and a pseudo.
            MInst::new(Opc::VMaxminI32, [v(1, 1)], [v(2, 1), v(3, 1), v(4, 1)]),
            MInst::new(Opc::SDelayAlu, [], []),
            MInst::new(Opc::Copy, [v(1, 1)], [v(2, 1)]),
            // row_xmask is a gfx10+ DPP control.
            MInst::new(Opc::VMovB32Dpp, [v(1, 1)], [v(2, 1)]).with_imm(dpp16_ctrl(0x161, true) as i64),
        ];
        for inst in refused {
            assert!(words(inst.clone()).is_err(), "{inst:?}");
        }
    }

    #[test]
    fn flags_are_the_shared_flags() {
        for &op in &t::ALL {
            let info = t::info(op).expect("a row");
            assert_eq!(info.flags, op.flags(), "{}", op.name());
        }
        for &(code, _, name) in &t::VCMP {
            assert_eq!(t::info(Opc::VCmp(code)).map(|i| i.flags), Some(Opc::VCmp(code).flags()), "{name}");
        }
    }
}
