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
/// keep tuples at even registers. AGPRs (`Phys::V(256 + k)`) go only where
/// a form says so (`agpr`).
fn aligned(r: u16, n: u8) -> Result<u32, CompileError> {
    if r >= AGPR0 {
        return Err(bad(format!("a{} where gfx950 reads or writes a VGPR", r - AGPR0)));
    }
    if n >= 2 && r % 2 != 0 {
        return Err(bad(format!("v{r} (x{n}): gfx950 VGPR tuples start at an even register")));
    }
    Ok(r as u32)
}

/// AGPR a<k> is `Phys::V(AGPR0 + k)`: the AGPR half of gfx950's unified
/// 512-entry register file, which gfx11 never reaches.
pub const AGPR0: u16 = 256;

/// An MFMA register operand (`mai`): its index in its bank and whether the
/// bank is the AGPRs'.
fn vgpr_or_agpr(o: &Opnd) -> Result<(u32, bool), CompileError> {
    match *o {
        Opnd::P(Phys::V(r), _) if r >= AGPR0 => Ok((agpr(o)?, true)),
        _ => Ok((vgpr(o)?, false)),
    }
}

/// The index k of AGPR a<k>, as the AGPR forms encode it; AGPR tuples
/// start at even registers too.
fn agpr(o: &Opnd) -> Result<u32, CompileError> {
    match *o {
        Opnd::P(Phys::V(r), n) if r >= AGPR0 && r - AGPR0 + n as u16 <= 256 => {
            let k = r - AGPR0;
            if n >= 2 && k % 2 != 0 {
                return Err(bad(format!("a{k} (x{n}): gfx950 AGPR tuples start at an even register")));
            }
            Ok(k as u32)
        }
        _ => Err(bad(format!("{o:?} is not an AGPR"))),
    }
}

/// A packed 32-bit source (`pk32`): a register pair at an even register,
/// whose lanes the instruction's lane selects pick, or one register both
/// lanes read (its selects must say so: op_sel clear, op_sel_hi cleared),
/// encoded at an odd register as the high half of the pair below it.
/// Constants are refused: which lane an inline constant feeds is not what
/// the compiler means by them. Returns the operand code and whether it is
/// the high half of its pair.
fn pk32_source(o: &Opnd, broadcast: bool) -> Result<(u32, bool), CompileError> {
    let one = o.dwords() == 1;
    if one && !broadcast {
        return Err(bad(format!("{o:?}: a one-dword packed 32-bit source is read by both lanes")));
    }
    match *o {
        Opnd::P(Phys::V(r), _) if r % 2 == 0 => Ok((t::SRC_VGPR + aligned(r, 2)?, false)),
        Opnd::P(Phys::V(r), 1) => Ok((t::SRC_VGPR + aligned(r - 1, 2)?, true)),
        Opnd::P(Phys::S(r), _) if r % 2 == 0 => Ok((sgpr(r, 2)?, false)),
        Opnd::P(Phys::S(r), 1) => Ok((sgpr(r - 1, 2)?, true)),
        Opnd::P(Phys::VccLo, 2) => Ok((t::REG_VCC_LO, false)),
        Opnd::P(Phys::ExecLo, 2) => Ok((t::REG_EXEC_LO, false)),
        _ => Err(bad(format!("{o:?} is not a register pair at an even register (a packed 32-bit source)"))),
    }
}

/// The encoded op_sel and op_sel_hi of a `pk32` instruction, and its source
/// codes: its lane selects (`imm`), with a one-dword source at an odd
/// register read as the high half of its pair by both lanes.
pub(super) fn pk32_encoded(inst: &MInst) -> Result<(u32, u32, [u32; 3]), CompileError> {
    let sel = inst.imm;
    if !(0..=0x77).contains(&sel) || sel & 0x08 != 0 {
        return Err(bad(format!("{}: lane selects {sel:#x}", t::mnemonic(inst.op).unwrap_or("?"))));
    }
    let (mut op_sel, mut op_sel_hi) = (sel as u32 & 7, !(sel as u32 >> 4) & 7);
    let mut codes = [0u32; 3];
    for (k, (c, o)) in codes.iter_mut().zip(&inst.uses).enumerate() {
        let broadcast = op_sel >> k & 1 == 0 && op_sel_hi >> k & 1 == 0;
        let (code, high) = pk32_source(o, broadcast)?;
        *c = code;
        if high {
            op_sel |= 1 << k;
            op_sel_hi |= 1 << k;
        }
    }
    Ok((op_sel, op_sel_hi, codes))
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

/// Source 2 of v_cndmask_b32 (the lane mask) and of v_addc_co_u32 /
/// v_subb_co_u32 (the carry-in) is a 64-bit lane mask: an even SGPR pair,
/// VCC or EXEC. The VOP3 source field also encodes inline constants,
/// literals, VGPRs, single SGPRs, M0 and SCC, which the hardware does not
/// read as a lane mask: such a select picks garbage. gfx950 encodes these
/// opcodes as VOP3 only (`noshort`, VOP3B), never as the VOP2 forms that
/// read VCC implicitly, so the mask is always this explicit operand.
fn lane_mask_source(inst: &MInst) -> Result<(), CompileError> {
    if !matches!(inst.op, Opc::VCndmaskB32 | Opc::VAddCoCiU32 | Opc::VSubCoCiU32) {
        return Ok(());
    }
    match inst.uses.get(2) {
        Some(Opnd::P(Phys::S(r), 2)) if r % 2 == 0 => Ok(()),
        Some(Opnd::P(Phys::VccLo | Phys::ExecLo, 2)) => Ok(()),
        o => Err(bad(format!(
            "{} reads its lane mask from an even SGPR pair, VCC or EXEC, not {o:?}",
            t::mnemonic(inst.op).unwrap_or("?")
        ))),
    }
}

/// Encodes one instruction into `out`.
pub fn encode(inst: &MInst, out: &mut Vec<u32>) -> Result<(), CompileError> {
    lane_mask_source(inst)?;
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
        Fmt::Vop1 if form == Form::Swap => {
            // Both registers are read and written: the uses are the defs.
            if uses != defs {
                return Err(bad(format!("{}: the registers it swaps are its defs and its uses", name())));
            }
            let vdst = vgpr(nth(defs, 0)?)?;
            let src0 = t::SRC_VGPR + vgpr(nth(defs, 1)?)?;
            out.push(t::EncVop1 { op: code, vdst, src0 }.pack());
        }
        Fmt::Vop1 if form == Form::AccMov => {
            let vdst = agpr(nth(defs, 0)?)?;
            let src0 = t::SRC_VGPR + agpr(nth(uses, 0)?)?;
            out.push(t::EncVop1 { op: code, vdst, src0 }.pack());
        }
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
        Fmt::Vop3p if form == Form::Pk32 => {
            let vdst = vgpr(nth(defs, 0)?)?;
            if nth(defs, 0)?.dwords() != 2 {
                return Err(bad(format!("{}: the destination is a register pair", name())));
            }
            // Lane selects: op_sel (source k's low lane reads its high half)
            // and the op_sel_hi bits cleared (its high lane reads its low half).
            let (op_sel, op_sel_hi, codes) = pk32_encoded(inst)?;
            let [src0, src1, src2] = codes;
            let neg = mods & (M_NEG0 | M_NEG1 | M_NEG2);
            let words = t::EncVop3p {
                op: code,
                vdst,
                neg_hi: neg,
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
        Fmt::Vop3p if form == Form::Mai => {
            // D and C share one bank (acc_cd); A and B each name theirs (acc).
            let (vdst, acc_cd) = vgpr_or_agpr(nth(defs, 0)?)?;
            let (a, acc_a) = vgpr_or_agpr(nth(uses, 0)?)?;
            let (b, acc_b) = vgpr_or_agpr(nth(uses, 1)?)?;
            // An f64 MFMA's C constant is f64 (imm its high dword, as in f64
            // forms), and it holds neg where blgp would be; the f8f6f4 form's
            // cbsz and blgp name the A and B formats, which fix their sizes.
            let f64 = matches!(t::lat(op), t::Lat::MfmaDgemm4 | t::Lat::MfmaDgemm16);
            let src2 = match *nth(uses, 2)? {
                Opnd::Imm(v) => t::inline_int(v)
                    .or_else(|| if f64 { t::inline_f64(v) } else { t::inline_f32(v) })
                    .ok_or_else(|| bad(format!("{}: C is a register or an inline constant", name())))?,
                ref c => {
                    let (k, acc_c) = vgpr_or_agpr(c)?;
                    if acc_c != acc_cd {
                        return Err(bad(format!("{}: D and C are in one bank (VGPRs or AGPRs)", name())));
                    }
                    t::SRC_VGPR + k
                }
            };
            let ctl = inst.imm;
            if !(0..1 << 10).contains(&ctl) {
                return Err(bad(format!("{}: controls {ctl:#x} past cbsz, abid and blgp", name())));
            }
            let ctl = ctl as u32;
            if f64 && ctl >> 7 != 0 {
                return Err(bad(format!("{}: blgp on an f64 MFMA (those bits are neg there)", name())));
            }
            if op == Opc::Gfx9(t::Gfx9Op::VMfma16x16x128F8f6f4) {
                let dwords = |format: u32| [8u8, 8, 6, 6, 4].get(format as usize).copied();
                let (a, b) = (nth(uses, 0)?.dwords(), nth(uses, 1)?.dwords());
                if ctl >> 3 & 15 != 0 || dwords(ctl & 7) != Some(a) || dwords(ctl >> 7 & 7) != Some(b) {
                    return Err(bad(format!(
                        "{}: A and B formats (cbsz {}, blgp {}) and sizes ({a}, {b} dwords) disagree, or abid is set",
                        name(),
                        ctl & 7,
                        ctl >> 7 & 7
                    )));
                }
            }
            let words = t::EncVop3pMai {
                vdst,
                cbsz: ctl & 7,
                abid: ctl >> 3 & 15,
                acc_cd: u32::from(acc_cd),
                op: code,
                src0: t::SRC_VGPR + a,
                src1: t::SRC_VGPR + b,
                src2,
                acc: u32::from(acc_a) | u32::from(acc_b) << 1,
                blgp: ctl >> 7 & 7,
            };
            push2(out, words.pack());
        }
        Fmt::Vop3p if matches!(form, Form::AccRead | Form::AccWrite) => {
            // The AGPR is vdst (a write) or src0 (a read, as 256 + index);
            // op_sel_hi stays at its default, as llvm-mc writes it.
            let (vdst, src0) = if form == Form::AccRead {
                (vgpr(nth(defs, 0)?)?, t::SRC_VGPR + agpr(nth(uses, 0)?)?)
            } else {
                let (src0, lit) = src(nth(uses, 0)?, valu_lits(form, flags)[0])?;
                if lit.is_some() {
                    return Err(bad(format!("{}: a literal in a VOP3P encoding (gfx950 has none)", name())));
                }
                (agpr(nth(defs, 0)?)?, src0)
            };
            push2(out, t::EncVop3p { op: code, vdst, op_sel_hi2: 1, op_sel_hi: 3, src0, ..Default::default() }.pack());
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
        Fmt::Mubuf if form == Form::Nodata => {
            if !defs.is_empty() || !uses.is_empty() || !(0..=3).contains(&inst.imm) {
                return Err(bad(format!("{}: no operands, and a cache policy of sc0 and sc1 only", name())));
            }
            let (sc0, sc1) = (inst.imm as u32 & 1, inst.imm as u32 >> 1);
            push2(out, t::EncMubuf { op: code, sc0, sc1, ..Default::default() }.pack());
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

    /// The lane mask of v_cndmask_b32 and the carry-in of v_addc_co_u32 /
    /// v_subb_co_u32: what llvm-mc -mcpu=gfx950 accepts there encodes with
    /// its words; every other operand the source field could hold is refused.
    #[test]
    fn lane_masks_are_sgpr_pairs_vcc_or_exec() {
        let (vcc, exec) = (Opnd::P(Phys::VccLo, 2), Opnd::P(Phys::ExecLo, 2));
        let cnd = |m: Opnd| MInst::new(Opc::VCndmaskB32, [v(1, 1)], [v(2, 1), v(3, 1), m]);
        let addc = |m: Opnd| MInst::new(Opc::VAddCoCiU32, [v(1, 1), s(6, 2)], [v(2, 1), v(3, 1), m]);
        let subb = |m: Opnd| MInst::new(Opc::VSubCoCiU32, [v(1, 1), s(6, 2)], [v(2, 1), v(3, 1), m]);
        let encoded: Vec<(MInst, Vec<u32>)> = vec![
            (cnd(s(4, 2)), vec![0xd100_0001, 0x0012_0702]),
            (cnd(vcc), vec![0xd100_0001, 0x01aa_0702]),
            (cnd(exec), vec![0xd100_0001, 0x01fa_0702]),
            (MInst::new(Opc::VCndmaskB32, [v(1, 1)], [Opnd::Imm(0), v(3, 1), s(100, 2)]), vec![0xd100_0001, 0x0192_0680]),
            (addc(s(4, 2)), vec![0xd11c_0601, 0x0012_0702]),
            (addc(vcc), vec![0xd11c_0601, 0x01aa_0702]),
            (addc(exec), vec![0xd11c_0601, 0x01fa_0702]),
            (subb(s(4, 2)), vec![0xd11d_0601, 0x0012_0702]),
            (subb(vcc), vec![0xd11d_0601, 0x01aa_0702]),
            (subb(exec), vec![0xd11d_0601, 0x01fa_0702]),
        ];
        for (inst, want) in encoded {
            assert_eq!(words(inst.clone()), Ok(want), "{inst:?}");
        }
        let refused = [
            // Inline constants (all lanes, no lane, an f32 constant) and a literal.
            Opnd::Imm(u32::MAX),
            Opnd::Imm(0),
            Opnd::Imm(0x3f80_0000),
            Opnd::Imm(0x1234_5678),
            // VGPRs, a single SGPR, a pair at an odd SGPR.
            v(4, 1),
            v(4, 2),
            s(4, 1),
            s(5, 2),
            // Half of VCC or EXEC, M0, SCC.
            Opnd::P(Phys::VccLo, 1),
            Opnd::P(Phys::ExecLo, 1),
            Opnd::P(Phys::M0, 1),
            Opnd::P(Phys::Scc, 1),
        ];
        for mask in refused {
            for inst in [cnd(mask), addc(mask), subb(mask)] {
                let err = words(inst.clone()).expect_err("a lane mask that is no SGPR pair, VCC or EXEC");
                assert!(err.contains("lane mask"), "{inst:?}: {err}");
            }
        }
        // No VOP2 form, whose lane mask is an implicit VCC, is ever chosen.
        assert_eq!(t::info(Opc::VCndmaskB32).map(|i| (i.fmt, i.form)), Some((Fmt::Vop2, Form::Noshort)));
        assert_eq!(t::info(Opc::VAddCoCiU32).map(|i| i.fmt), Some(Fmt::Vop3b));
        assert_eq!(t::info(Opc::VSubCoCiU32).map(|i| i.fmt), Some(Fmt::Vop3b));
    }

    /// The gfx9 back end's own opcodes (`Opc::Gfx9`): llvm-mc -mcpu=gfx950
    /// -show-encoding words, and what their forms refuse.
    #[test]
    fn gfx9_only_rows_match_llvm_mc() {
        use t::Gfx9Op::*;
        let g = Opc::Gfx9;
        let cases: Vec<(MInst, Vec<u32>)> = vec![
            // v_pk_mul_f32 v[0:1], v[2:3], v[4:5]; op_sel_hi:[0,1] and [1,0];
            // an SGPR pair broadcast; neg_lo:[1,0] neg_hi:[1,0]
            (MInst::new(g(VPkMulF32), [v(0, 2)], [v(2, 2), v(4, 2)]), vec![0xd3b1_4000, 0x1802_0902]),
            (MInst::new(g(VPkMulF32), [v(0, 2)], [v(2, 2), v(4, 2)]).with_imm(0x10), vec![0xd3b1_4000, 0x1002_0902]),
            (MInst::new(g(VPkMulF32), [v(0, 2)], [v(2, 2), v(4, 2)]).with_imm(0x20), vec![0xd3b1_4000, 0x0802_0902]),
            (MInst::new(g(VPkMulF32), [v(0, 2)], [s(2, 2), v(4, 2)]).with_imm(0x10), vec![0xd3b1_4000, 0x1002_0802]),
            (MInst::new(g(VPkMulF32), [v(0, 2)], [v(2, 2), v(4, 2)]).with_mods(M_NEG0), vec![0xd3b1_4100, 0x3802_0902]),
            // A one-dword broadcast source: the low half of its pair at an even
            // register; at an odd one the high half of the pair below it, read
            // by both lanes (op_sel:[1,0] / [0,1,0] with op_sel_hi set).
            (MInst::new(g(VPkMulF32), [v(0, 2)], [v(2, 1), v(4, 2)]).with_imm(0x10), vec![0xd3b1_4000, 0x1002_0902]),
            (MInst::new(g(VPkMulF32), [v(0, 2)], [v(3, 1), v(4, 2)]).with_imm(0x10), vec![0xd3b1_4800, 0x1802_0902]),
            (MInst::new(g(VPkMulF32), [v(0, 2)], [s(5, 1), v(2, 2)]).with_imm(0x10), vec![0xd3b1_4800, 0x1802_0404]),
            (MInst::new(g(VPkFmaF32), [v(0, 2)], [v(2, 2), s(5, 1), v(6, 2)]).with_imm(0x20), vec![0xd3b0_5000, 0x1c18_0902]),
            // v_pk_add_f32 ... op_sel:[1,0]; v_pk_fma_f32 (and op_sel_hi:[0,1,1]); v_pk_mov_b32 op_sel:[0,1]
            (MInst::new(g(VPkAddF32), [v(0, 2)], [v(2, 2), v(4, 2)]).with_imm(0x01), vec![0xd3b2_4800, 0x1802_0902]),
            (MInst::new(g(VPkFmaF32), [v(0, 2)], [v(2, 2), v(4, 2), v(6, 2)]), vec![0xd3b0_4000, 0x1c1a_0902]),
            (MInst::new(g(VPkFmaF32), [v(0, 2)], [v(2, 2), v(4, 2), v(6, 2)]).with_imm(0x10), vec![0xd3b0_4000, 0x141a_0902]),
            (MInst::new(g(VPkMovB32), [v(0, 2)], [v(2, 2), v(4, 2)]).with_imm(0x02), vec![0xd3b3_5000, 0x1802_0902]),
            // v_lshl_add_u64 v[0:1], v[2:3], 1, s[4:5] / 2, v[4:5]
            (MInst::new(g(VLshlAddU64), [v(0, 2)], [v(2, 2), Opnd::Imm(1), s(4, 2)]), vec![0xd208_0000, 0x0011_0302]),
            (MInst::new(g(VLshlAddU64), [v(0, 2)], [v(2, 2), Opnd::Imm(2), v(4, 2)]), vec![0xd208_0000, 0x0411_0502]),
            // v_cvt_pk_f16_f32 v1, v0, v1; v_cvt_pk_bf16_f32 v1, 0, v1 and v1, -v0, v1
            (MInst::new(g(VCvtPkF16F32), [v(1, 1)], [v(0, 1), v(1, 1)]), vec![0xd267_0001, 0x0002_0300]),
            (MInst::new(g(VCvtPkBf16F32), [v(1, 1)], [Opnd::Imm(0), v(1, 1)]), vec![0xd268_0001, 0x0002_0280]),
            (MInst::new(g(VCvtPkBf16F32), [v(1, 1)], [v(0, 1), v(1, 1)]).with_mods(M_NEG0), vec![0xd268_0001, 0x2002_0300]),
            // v_permlane16_swap_b32 v0, v1 / v_permlane32_swap_b32 v0, v1
            (MInst::new(g(VPermlane16SwapB32), [v(0, 1), v(1, 1)], [v(0, 1), v(1, 1)]), vec![0x7e00_b301]),
            (MInst::new(g(VPermlane32SwapB32), [v(0, 1), v(1, 1)], [v(0, 1), v(1, 1)]), vec![0x7e00_b501]),
            // buffer_inv sc1 / sc0 sc1; buffer_wbl2 sc1
            (MInst::new(g(BufferInv), [], []).with_imm(2), vec![0xe0a4_8000, 0]),
            (MInst::new(g(BufferInv), [], []).with_imm(3), vec![0xe0a4_c000, 0]),
            (MInst::new(g(BufferWbl2), [], []).with_imm(2), vec![0xe0a0_8000, 0]),
            // v_mov_b64 v[0:1], v[2:3] / s[2:3]
            (MInst::new(g(VMovB64), [v(0, 2)], [v(2, 2)]), vec![0x7e00_7102]),
            (MInst::new(g(VMovB64), [v(0, 2)], [s(2, 2)]), vec![0x7e00_7002]),
            // v_accvgpr_read_b32 v6, a1 / v255, a255; v_accvgpr_write_b32 a3, 0 /
            // a3, v7 / a255, v255 / a3, -1 / a3, 1.0 / a3, s5; v_accvgpr_mov_b32
            // a2, a5 / a255, a0
            (MInst::new(g(VAccvgprReadB32), [v(6, 1)], [v(AGPR0 + 1, 1)]), vec![0xd3d8_4006, 0x1800_0101]),
            (MInst::new(g(VAccvgprReadB32), [v(255, 1)], [v(AGPR0 + 255, 1)]), vec![0xd3d8_40ff, 0x1800_01ff]),
            (MInst::new(g(VAccvgprWriteB32), [v(AGPR0 + 3, 1)], [Opnd::Imm(0)]), vec![0xd3d9_4003, 0x1800_0080]),
            (MInst::new(g(VAccvgprWriteB32), [v(AGPR0 + 3, 1)], [v(7, 1)]), vec![0xd3d9_4003, 0x1800_0107]),
            (MInst::new(g(VAccvgprWriteB32), [v(AGPR0 + 255, 1)], [v(255, 1)]), vec![0xd3d9_40ff, 0x1800_01ff]),
            (MInst::new(g(VAccvgprWriteB32), [v(AGPR0 + 3, 1)], [Opnd::Imm(u32::MAX)]), vec![0xd3d9_4003, 0x1800_00c1]),
            (MInst::new(g(VAccvgprWriteB32), [v(AGPR0 + 3, 1)], [Opnd::Imm(0x3f80_0000)]), vec![0xd3d9_4003, 0x1800_00f2]),
            (MInst::new(g(VAccvgprWriteB32), [v(AGPR0 + 3, 1)], [s(5, 1)]), vec![0xd3d9_4003, 0x1800_0005]),
            (MInst::new(g(VAccvgprMovB32), [v(AGPR0 + 2, 1)], [v(AGPR0 + 5, 1)]), vec![0x7e04_a505]),
            (MInst::new(g(VAccvgprMovB32), [v(AGPR0 + 255, 1)], [v(AGPR0, 1)]), vec![0x7ffe_a500]),
            // MFMAs: v_mfma_f32_16x16x32_bf16 v[0:3], v[4:7], v[8:11], v[0:3];
            // _16x16x32_f16 a[0:3], v[4:7], v[8:11], a[0:3] / a[0:3], a[4:7],
            // v[8:11], 0 / v[0:3], v[4:7], a[8:11], 1.0 / with cbsz:1 abid:2
            // blgp:3; _32x32x16_f16 v[0:15], v[16:19], v[20:23], v[0:15];
            // _16x16x4_f32 v[0:3], v4, v5, v[0:3]; v_mfma_f64_4x4x4_4b_f64
            // v[0:1], v[2:3], v[4:5], v[0:1]; _16x16x128_f8f6f4 v[0:3], v[4:9],
            // v[12:15], v[0:3] cbsz:2 blgp:4 (fp6 and fp4 operands)
            (MInst::new(g(VMfma16x16x32Bf16), [v(0, 4)], [v(4, 4), v(8, 4), v(0, 4)]), vec![0xd3b5_0000, 0x0402_1104]),
            (MInst::new(g(VMfma16x16x32F16), [v(AGPR0, 4)], [v(4, 4), v(8, 4), v(AGPR0, 4)]), vec![0xd3d4_8000, 0x0402_1104]),
            (MInst::new(g(VMfma16x16x32F16), [v(AGPR0, 4)], [v(AGPR0 + 4, 4), v(8, 4), Opnd::Imm(0)]), vec![0xd3d4_8000, 0x0a02_1104]),
            (MInst::new(g(VMfma16x16x32F16), [v(0, 4)], [v(4, 4), v(AGPR0 + 8, 4), Opnd::Imm(0x3f80_0000)]), vec![0xd3d4_0000, 0x13ca_1104]),
            (MInst::new(g(VMfma16x16x32F16), [v(0, 4)], [v(4, 4), v(8, 4), v(0, 4)]).with_imm(0x191), vec![0xd3d4_1100, 0x6402_1104]),
            (MInst::new(g(VMfma32x32x16F16), [v(0, 16)], [v(16, 4), v(20, 4), v(0, 16)]), vec![0xd3d5_0000, 0x0402_2910]),
            (MInst::new(g(VMfma16x16x4F32), [v(0, 4)], [v(4, 1), v(5, 1), v(0, 4)]), vec![0xd3c5_0000, 0x0402_0b04]),
            (MInst::new(g(VMfma4x4x4F64), [v(0, 2)], [v(2, 2), v(4, 2), v(0, 2)]), vec![0xd3ef_0000, 0x0402_0902]),
            (MInst::new(g(VMfma16x16x128F8f6f4), [v(0, 4)], [v(4, 6), v(12, 4), v(0, 4)]).with_imm(2 | 4 << 7), vec![0xd3ad_0200, 0x8402_1904]),
        ];
        for (inst, want) in cases {
            assert_eq!(words(inst.clone()), Ok(want), "{inst:?}");
        }
        let refused = [
            // An AGPR where a plain VALU instruction reads or writes a VGPR.
            MInst::new(Opc::VAddF32, [v(1, 1)], [v(AGPR0 + 2, 1), v(3, 1)]),
            MInst::new(Opc::VAddF32, [v(AGPR0 + 1, 1)], [v(2, 1), v(3, 1)]),
            // Packed 32-bit sources: a constant, an odd pair, two scalar pairs;
            // an odd or one-dword destination; lane selects past three sources.
            MInst::new(g(VPkMulF32), [v(0, 2)], [Opnd::Imm(0x3f80_0000), v(4, 2)]),
            MInst::new(g(VPkMulF32), [v(0, 2)], [v(3, 2), v(4, 2)]),
            MInst::new(g(VPkMulF32), [v(0, 2)], [s(2, 2), s(4, 2)]),
            MInst::new(g(VPkMulF32), [v(1, 2)], [v(2, 2), v(4, 2)]),
            MInst::new(g(VPkMulF32), [v(0, 1)], [v(2, 2), v(4, 2)]),
            MInst::new(g(VPkMulF32), [v(0, 2)], [v(2, 2), v(4, 2)]).with_imm(0x80),
            // A one-dword source both lanes do not read alike.
            MInst::new(g(VPkMulF32), [v(0, 2)], [v(2, 1), v(4, 2)]),
            // A swap whose uses are not its defs.
            MInst::new(g(VPermlane32SwapB32), [v(0, 1), v(1, 1)], [v(0, 1), v(2, 1)]),
            // Cache control with an operand or a policy past sc0 and sc1.
            MInst::new(g(BufferInv), [], [v(0, 1)]),
            MInst::new(g(BufferInv), [], []).with_imm(4),
            // AGPR moves: the AGPR and VGPR sides swapped, a literal (llvm-mc
            // refuses `v_accvgpr_write_b32 a3, 0x1234`), an AGPR past a255.
            MInst::new(g(VAccvgprReadB32), [v(AGPR0 + 1, 1)], [v(6, 1)]),
            MInst::new(g(VAccvgprWriteB32), [v(6, 1)], [v(7, 1)]),
            MInst::new(g(VAccvgprWriteB32), [v(AGPR0 + 3, 1)], [v(AGPR0 + 4, 1)]),
            MInst::new(g(VAccvgprWriteB32), [v(AGPR0 + 3, 1)], [Opnd::Imm(0x1234)]),
            MInst::new(g(VAccvgprMovB32), [v(AGPR0 + 2, 1)], [v(5, 1)]),
            MInst::new(g(VAccvgprReadB32), [v(6, 1)], [v(AGPR0 + 256, 1)]),
            // MFMAs: D and C in different banks, an SGPR source, an odd tuple,
            // a literal C, controls past blgp (llvm-mc refuses the first three).
            MInst::new(g(VMfma16x16x32F16), [v(0, 4)], [v(4, 4), v(8, 4), v(AGPR0, 4)]),
            MInst::new(g(VMfma16x16x32F16), [v(0, 4)], [s(4, 4), v(8, 4), v(0, 4)]),
            MInst::new(g(VMfma16x16x32F16), [v(1, 4)], [v(4, 4), v(8, 4), v(1, 4)]),
            MInst::new(g(VMfma16x16x32F16), [v(0, 4)], [v(4, 4), v(8, 4), Opnd::Imm(0x1234)]),
            MInst::new(g(VMfma16x16x32F16), [v(0, 4)], [v(4, 4), v(8, 4), v(0, 4)]).with_imm(1 << 10),
            // f64 MFMAs: blgp, an f32 constant as C (llvm-mc refuses both);
            // f8f6f4: abid, a B format whose size is not B's.
            MInst::new(g(VMfma4x4x4F64), [v(0, 2)], [v(2, 2), v(4, 2), v(0, 2)]).with_imm(3 << 7),
            MInst::new(g(VMfma4x4x4F64), [v(0, 2)], [v(2, 2), v(4, 2), Opnd::Imm(0x3e22_f983)]),
            MInst::new(g(VMfma16x16x128F8f6f4), [v(0, 4)], [v(4, 8), v(12, 8), v(0, 4)]).with_imm(2 << 3),
            MInst::new(g(VMfma16x16x128F8f6f4), [v(0, 4)], [v(4, 8), v(12, 8), v(0, 4)]).with_imm(3 << 7),
        ];
        for inst in refused {
            assert!(words(inst.clone()).is_err(), "{inst:?}");
        }
    }

    /// The gfx11 encoders never take a gfx950-only opcode: it is a pseudo there.
    #[test]
    fn gfx9_only_opcodes_have_no_gfx11_encoding() {
        for &op in &t::Gfx9Op::ALL {
            let inst = MInst::new(Opc::Gfx9(op), [v(0, 2)], [v(2, 2), v(4, 2)]);
            assert_eq!(Opc::Gfx9(op).fmt(), Fmt::Pseudo);
            let f = MFunc { blocks: vec![MBlock { insts: vec![inst], loop_depth: 0 }], ..MFunc::default() };
            assert!(crate::rir::asm::assemble(&f).is_err(), "{op:?}");
            assert_eq!(Opc::Gfx9(op).name(), op.mnemonic());
            assert_eq!(t::mnemonic(Opc::Gfx9(op)), Some(op.mnemonic()));
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
