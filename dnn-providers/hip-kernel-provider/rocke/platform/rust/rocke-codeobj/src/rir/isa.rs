// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Table-driven gfx11 encoder. Encoding layouts, opcodes, operand codes and
//! each opcode's encoding form come from the instruction description
//! (`gen_isa.rs`, generated from `isa/gfx11.isa`); this file chooses between
//! the forms a format allows. `asm.rs` calls it first and keeps its hand
//! encoder for the opcodes the description leaves to it
//! (`gen_isa::table_encoded`).

use super::gen_isa as t;
use super::gen_isa::Form;
use super::mir::*;
use crate::error::CompileError;

// Errors are built out of line: inlined formatting would enlarge every
// instruction's stack frame.
#[cold]
#[inline(never)]
fn bad(message: impl Into<String>) -> CompileError {
    CompileError::new("ROCKE_DIRECT_ENCODE", message)
}

#[cold]
#[inline(never)]
fn not_a(o: &Opnd, what: &str) -> CompileError {
    bad(format!("{o:?} is not {what}"))
}

#[cold]
#[inline(never)]
fn out_of_range(what: &str, offset: impl std::fmt::Display) -> CompileError {
    bad(format!("{what} offset {offset} out of range"))
}

#[cold]
#[inline(never)]
fn missing(i: usize) -> CompileError {
    bad(format!("missing operand {i}"))
}

#[cold]
#[inline(never)]
fn not_table_encoded(op: Opc, what: &str) -> CompileError {
    bad(format!("{} {what}", op.name()))
}

/// How a source reads an immediate that is not an inline integer (inline
/// integers are sign-extended to the source's width).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lit {
    /// A 32-bit value: an f32 inline constant or a literal.
    B32,
    /// A literal only: a 64-bit integer source (the literal is
    /// zero-extended), a pair of 16-bit values, or a source the f32
    /// constants do not apply to.
    Plain,
    /// The high dword of an f64 whose low dword is zero: an f64 inline
    /// constant or a literal.
    F64,
}

/// Operand code of a source, and its literal if it needs one.
fn src(o: &Opnd, lit: Lit) -> Result<(u32, Option<u32>), CompileError> {
    Ok(match *o {
        Opnd::P(Phys::V(r), _) => (t::SRC_VGPR + r as u32, None),
        Opnd::P(Phys::S(r), _) => (t::SRC_SGPR + r as u32, None),
        Opnd::P(Phys::VccLo, _) => (t::REG_VCC_LO, None),
        Opnd::P(Phys::Null, _) => (t::REG_NULL, None),
        Opnd::P(Phys::M0, _) => (t::REG_M0, None),
        Opnd::P(Phys::ExecLo, _) => (t::REG_EXEC_LO, None),
        Opnd::P(Phys::Scc, _) => (t::REG_SCC, None),
        Opnd::Imm(v) => match t::inline_int(v).or_else(|| match lit {
            Lit::B32 => t::inline_f32(v),
            Lit::F64 => t::inline_f64(v),
            Lit::Plain => None,
        }) {
            Some(code) => (code, None),
            None => (t::SRC_LITERAL, Some(v)),
        },
        Opnd::R(_) => return Err(bad("virtual register reached the encoder")),
    })
}

/// A scalar instruction's source, whose field has no VGPRs.
fn ssrc(o: &Opnd, lit: Lit) -> Result<(u32, Option<u32>), CompileError> {
    let (code, literal) = src(o, lit)?;
    if code >= t::SRC_VGPR {
        return Err(not_a(o, "a scalar source"));
    }
    Ok((code, literal))
}

/// How SALU source `k` reads an immediate: a 64-bit integer source takes no
/// f32 inline constant.
#[inline(always)]
fn salu_lit(form: Form, k: usize) -> Lit {
    match form {
        Form::B64 => Lit::Plain,
        Form::B64Shift if k == 0 => Lit::Plain,
        _ => Lit::B32,
    }
}

/// How VALU sources 0-2 read immediates.
#[inline(always)]
fn valu_lits(form: Form, flags: u32) -> [Lit; 3] {
    match form {
        Form::F64 => [Lit::F64; 3],
        Form::F64I32 => [Lit::F64, Lit::B32, Lit::B32],
        Form::Dot2 => [Lit::Plain, Lit::Plain, Lit::B32],
        Form::Writelane => [Lit::Plain; 3],
        _ if flags & (F_64BIT_SRC | F_16BIT_SRC) != 0 => [Lit::Plain; 3],
        _ => [Lit::B32; 3],
    }
}

fn sdst(o: &Opnd) -> Result<u32, CompileError> {
    Ok(match *o {
        Opnd::P(Phys::S(r), _) => t::SRC_SGPR + r as u32,
        Opnd::P(Phys::VccLo, _) => t::REG_VCC_LO,
        Opnd::P(Phys::Null, _) => t::REG_NULL,
        Opnd::P(Phys::M0, _) => t::REG_M0,
        Opnd::P(Phys::ExecLo, _) => t::REG_EXEC_LO,
        _ => return Err(not_a(o, "a scalar destination")),
    })
}

fn vgpr(o: &Opnd) -> Result<u32, CompileError> {
    match *o {
        Opnd::P(Phys::V(r), _) => Ok(r as u32),
        _ => Err(not_a(o, "a VGPR")),
    }
}

fn is_vgpr(o: &Opnd) -> bool {
    matches!(o, Opnd::P(Phys::V(_), _))
}

fn nth(opnds: &[Opnd], i: usize) -> Result<&Opnd, CompileError> {
    match opnds.get(i) {
        Some(o) => Ok(o),
        None => Err(missing(i)),
    }
}

/// Appends the one literal the sources share, if any.
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

#[inline(always)]
fn push2(out: &mut Vec<u32>, words: [u32; 2]) {
    out.push(words[0]);
    out.push(words[1]);
}

/// A VOP3 or VOP3SD instruction (`sdst` given).
fn vop3(
    out: &mut Vec<u32>,
    op: u32,
    dst: u32,
    sdst: Option<u32>,
    srcs: &[Opnd],
    mods: u32,
    reads: [Lit; 3],
) -> Result<(), CompileError> {
    if srcs.len() > 3 {
        return Err(bad("more than three VOP3 sources"));
    }
    let mut codes = [0u32; 3];
    let mut lits = [None; 3];
    for (i, s) in srcs.iter().enumerate() {
        (codes[i], lits[i]) = src(s, reads[i])?;
    }
    let [src0, src1, src2] = codes;
    let neg = mods & (M_NEG0 | M_NEG1 | M_NEG2);
    push2(out, match sdst {
        Some(sdst) => t::EncVop3sd { op, vdst: dst, sdst, src0, src1, src2, neg, ..Default::default() }.pack(),
        None => {
            let op_sel = (mods / M_OPSEL_SRC0) & 0xf;
            t::EncVop3 { op, vdst: dst, op_sel, src0, src1, src2, neg, ..Default::default() }.pack()
        }
    });
    literal(out, &lits)
}

/// Encodes `inst` into `out` if the description encodes its opcode; returns
/// whether it did.
#[inline]
pub fn encode(inst: &MInst, out: &mut Vec<u32>) -> Result<bool, CompileError> {
    let info = t::info(inst.op);
    if !info.table {
        return Ok(false);
    }
    encode_into(inst, info, out)?;
    Ok(true)
}

/// The table-driven encoder's words for one instruction.
pub fn encode_words(inst: &MInst) -> Result<Vec<u32>, CompileError> {
    let info = t::info(inst.op);
    if !info.table {
        return Err(not_table_encoded(inst.op, "is not table-encoded"));
    }
    let mut out = Vec::with_capacity(3);
    encode_into(inst, info, &mut out)?;
    Ok(out)
}

// Inlined into the encoding loop: a call per instruction costs a full
// register save.
#[inline(always)]
fn encode_into(inst: &MInst, info: t::Info, out: &mut Vec<u32>) -> Result<(), CompileError> {
    let (code, vop3_code, flags, form) = (info.code as u32, info.vop3 as u32, info.flags, info.form);
    let (defs, uses, mods) = (&inst.defs[..], &inst.uses[..], inst.mods);
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
            if form == Form::Vscnt && sdst != t::REG_NULL {
                return Err(bad("s_waitcnt_vscnt takes null"));
            }
            out.push(t::EncSopk { op: code, sdst, simm16: inst.imm as u32 & 0xffff }.pack());
        }
        Fmt::Sopp => out.push(t::EncSopp { op: code, simm16: inst.imm as u32 & 0xffff }.pack()),
        Fmt::Smem => {
            let sdata = sdst(nth(defs, 0)?)?;
            let sbase = sdst(nth(uses, 0)?)? >> 1;
            let offset = inst.imm as u32 & 0x1f_ffff;
            push2(out, t::EncSmem { op: code, sdata, sbase, offset, soffset: t::REG_NULL, ..Default::default() }.pack());
        }
        Fmt::Vop1 if defs.is_empty() => out.push(t::EncVop1 { op: code, ..Default::default() }.pack()),
        Fmt::Vop1 if matches!(form, Form::Dpp16 | Form::Dpp8) => {
            let vdst = vgpr(nth(defs, 0)?)?;
            let src0 = vgpr(nth(uses, 0)?)?;
            let control = inst.imm as u32 & 0xff_ffff;
            if form == Form::Dpp16 {
                out.push(t::EncVop1 { op: code, vdst, src0: t::SRC_DPP16 }.pack());
                out.push(t::EncDpp16 { src0, ctrl: control }.pack());
            } else {
                out.push(t::EncVop1 { op: code, vdst, src0: t::SRC_DPP8 }.pack());
                out.push(t::EncDpp8 { src0, sel: control }.pack());
            }
        }
        Fmt::Vop1 => {
            let d = nth(defs, 0)?;
            let dst = if is_vgpr(d) { vgpr(d)? } else { sdst(d)? };
            let s = nth(uses, 0)?;
            let (half_dst, half_src) = (flags & F_16BIT_DST != 0, flags & F_16BIT_SRC != 0);
            let reads = valu_lits(form, flags);
            let (mut src0, l0) = src(s, reads[0])?;
            // The 32-bit form reaches 16-bit halves of v0-v127 only, and
            // has no neg modifiers.
            let short = (!half_dst || dst < t::SRC_HALF_HI)
                && (!half_src || !is_vgpr(s) || src0 - t::SRC_VGPR < t::SRC_HALF_HI)
                && mods & (M_NEG0 | M_NEG1 | M_NEG2) == 0;
            if short {
                let mut vdst = dst;
                if half_dst && mods & M_OPSEL_DST != 0 {
                    vdst |= t::SRC_HALF_HI;
                }
                if half_src && mods & M_OPSEL_SRC0 != 0 && is_vgpr(s) {
                    src0 |= t::SRC_HALF_HI;
                }
                out.push(t::EncVop1 { op: code, vdst, src0 }.pack());
                literal(out, &[l0])?;
            } else {
                vop3(out, vop3_code, dst, None, &uses[..1], mods, reads)?;
            }
        }
        Fmt::Vop2 => {
            let vdst = vgpr(nth(defs, 0)?)?;
            let (s0, s1) = (nth(uses, 0)?, nth(uses, 1)?);
            let reads = valu_lits(form, flags);
            if uses.len() == 2 && is_vgpr(s1) && mods == 0 && form != Form::Noshort {
                let (src0, l0) = src(s0, reads[0])?;
                out.push(t::EncVop2 { op: code, vdst, vsrc1: vgpr(s1)?, src0 }.pack());
                literal(out, &[l0])?;
            } else {
                vop3(out, vop3_code, vdst, None, uses, mods, reads)?;
            }
        }
        Fmt::Vop3 if form == Form::Writelane => {
            let vdst = vgpr(nth(defs, 0)?)?;
            nth(uses, 1)?;
            vop3(out, vop3_code, vdst, None, &uses[..2], mods, valu_lits(form, flags))?;
        }
        Fmt::Vop3 => {
            // A fourth use is the destination's old value, not encoded.
            let vdst = vgpr(nth(defs, 0)?)?;
            vop3(out, vop3_code, vdst, None, &uses[..uses.len().min(3)], mods, valu_lits(form, flags))?;
        }
        Fmt::Vop3b => {
            let vdst = vgpr(nth(defs, 0)?)?;
            let sdst = sdst(nth(defs, 1)?)?;
            vop3(out, vop3_code, vdst, Some(sdst), uses, mods, valu_lits(form, flags))?;
        }
        Fmt::Vopc => {
            let sdst = sdst(nth(defs, 0)?)?;
            vop3(out, vop3_code, sdst, None, uses, mods, valu_lits(form, flags))?;
        }
        Fmt::Vop3p => {
            let vdst = vgpr(nth(defs, 0)?)?;
            // Packed sources read both halves (op_sel_hi all set); a mix
            // opcode marks its f16 sources there.
            let (op_sel, op_sel_hi) = if form == Form::Mix {
                ((mods / M_OPSEL_SRC0) & 7, (mods / M_MIX_F16_0) & 7)
            } else {
                (0, 7)
            };
            // A 16-bit pair source takes no f32 inline constant: the literal
            // that remains is refused.
            let reads = valu_lits(form, flags);
            let mut codes = [0u32; 3];
            for ((c, o), &read) in codes.iter_mut().zip(uses).zip(&reads) {
                let (code, lit) = src(o, read)?;
                if lit.is_some() {
                    return Err(bad("VOP3P literal"));
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
        Fmt::Vopd => {
            let (vdstx, vdsty) = (vgpr(nth(defs, 0)?)?, vgpr(nth(defs, 1)?)?);
            // The encoding stores vdsty[7:1]; bit 0 is the inverse of vdstx's.
            if (vdstx ^ vdsty) & 1 == 0 {
                return Err(bad("VOPD destinations of equal parity"));
            }
            if mods as usize > uses.len() {
                return Err(bad("VOPD split beyond its sources"));
            }
            let (xs, ys) = uses.split_at(mods as usize);
            let mut fields = [(0, 0); 2];
            let mut lits = [None; 2];
            for (k, srcs) in [xs, ys].into_iter().enumerate() {
                let (src0, lit) = src(nth(srcs, 0)?, Lit::B32)?;
                let vsrc1 = match srcs.get(1) {
                    Some(o) => vgpr(o)?,
                    None => 0,
                };
                fields[k] = (src0, vsrc1);
                lits[k] = lit;
            }
            let (opx, opy) = (inst.imm as u32 & 0xff, (inst.imm >> 8) as u32 & 0xff);
            let words = t::EncVopd {
                opx,
                opy,
                src0x: fields[0].0,
                vsrc1x: fields[0].1,
                vdstx,
                vdsty: vdsty >> 1,
                src0y: fields[1].0,
                vsrc1y: fields[1].1,
            };
            push2(out, words.pack());
            literal(out, &lits)?;
        }
        Fmt::Ds => {
            let offset = inst.imm as u32 & 0xffff;
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
                return Err(out_of_range("scratch", offset));
            }
            let offset = offset as u32;
            push2(out, t::EncScratch { op: code, offset, vdst, data, saddr: t::REG_NULL, ..Default::default() }.pack());
        }
        Fmt::Global => {
            let saddr_mode = mods & M_SADDR != 0;
            let addr = vgpr(nth(uses, 0)?)?;
            let saddr = if saddr_mode { sdst(nth(uses, 1)?)? } else { t::REG_NULL };
            let next = if saddr_mode { 2 } else { 1 };
            let (vdst, data) = if defs.is_empty() {
                (0, vgpr(nth(uses, next)?)?)
            } else if flags & F_VMEM_STORE != 0 {
                // A returning atomic: both the old value and the operand.
                // Only GLC makes it return the old value.
                if mods & M_GLC == 0 {
                    return Err(bad("a returning atomic without GLC"));
                }
                (vgpr(&defs[0])?, vgpr(nth(uses, next)?)?)
            } else {
                (vgpr(&defs[0])?, 0)
            };
            let offset = inst.imm;
            if !(-4096..4096).contains(&offset) {
                return Err(out_of_range("global", offset));
            }
            let (offset, glc) = (offset as u32, u32::from(mods & M_GLC != 0));
            push2(out, t::EncGlobal { op: code, offset, glc, vdst, saddr, data, addr, ..Default::default() }.pack());
        }
        Fmt::Mubuf if defs.is_empty() && uses.is_empty() => {
            push2(out, t::EncMubuf { op: code, ..Default::default() }.pack());
        }
        Fmt::Mubuf => {
            let (vdata, rest) = if defs.is_empty() {
                (vgpr(nth(uses, 0)?)?, &uses[1..])
            } else {
                (vgpr(&defs[0])?, uses)
            };
            let vaddr = vgpr(nth(rest, 0)?)?;
            let srsrc = sdst(nth(rest, 1)?)? / 4;
            let (soffset, lit) = src(nth(rest, 2)?, Lit::Plain)?;
            if lit.is_some() {
                return Err(bad("buffer soffset literal"));
            }
            let offset = inst.imm as u32;
            if offset >= 4096 {
                return Err(out_of_range("buffer", offset));
            }
            push2(out, t::EncMubuf { op: code, offset, soffset, offen: 1, srsrc, vdata, vaddr, ..Default::default() }.pack());
        }
        Fmt::Pseudo => return Err(not_table_encoded(inst.op, "is a pseudo and reached the encoder")),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FNV-1a 64 of the description with line endings as LF.
    fn digest(text: &str) -> u64 {
        text.bytes().filter(|&b| b != b'\r').fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
    }

    #[test]
    fn generated_tables_match_the_description() {
        assert_eq!(
            digest(include_str!("../../isa/gfx11.isa")),
            t::DESCRIPTION_DIGEST,
            "gen_isa.rs is stale: run `python platform/tools/isa_tables.py generate`"
        );
    }

    #[test]
    fn inline_constants_agree_with_isel() {
        let mut values: Vec<u32> = (0..=200u32).chain((0..=200u32).map(|v| v.wrapping_neg())).collect();
        values.extend(t::INLINE_F32.iter().map(|&(bits, _)| bits));
        values.extend(t::INLINE_F64.iter().map(|&(hi, _)| hi));
        values.extend([0x3f00_0001, 0x4000_0001, 0x8000_0000, 0x7fff_ffff, 0x3e22_f984, 0x3ff0_0001, 0x7ff0_0000]);
        for v in values {
            assert_eq!(super::super::isel::is_inline_int(v), t::inline_int(v).is_some(), "{v:#x}");
            assert_eq!(super::super::isel::is_inline_f32(v), t::inline_f32(v).is_some(), "{v:#x}");
            assert_eq!(super::super::isel::is_inline_f64(v), t::inline_f64(v).is_some(), "{v:#x}");
        }
        // An f64 code is the f32 code of the same value.
        for &(hi, code) in &t::INLINE_F64 {
            let f32_bits = t::INLINE_F32.iter().find(|&&(_, c)| c == code).map(|&(b, _)| b);
            let wide = f32_bits.map(|b| f64::from(f32::from_bits(b)).to_bits());
            assert_eq!(wide, Some((hi as u64) << 32), "{hi:#x}");
        }
    }

    /// Both encoders read an immediate by the type of its source, as
    /// llvm-mc does (the expected words are llvm-mc's): an f64 source takes
    /// the f64 inline constants and a literal high dword, a 64-bit integer
    /// or 16-bit pair source no f32 inline constant. A returning atomic needs
    /// GLC, s_waitcnt_vscnt a null register.
    #[test]
    fn immediates_follow_their_source_types() {
        let v = |r: u16, n: u8| Opnd::P(Phys::V(r), n);
        let s = |r: u16, n: u8| Opnd::P(Phys::S(r), n);
        let imm = Opnd::Imm;
        let both = |inst: MInst| -> Option<Vec<u32>> {
            let table = encode_words(&inst).ok();
            let hand = super::super::asm::encode_legacy_words(&inst).ok();
            assert_eq!(table, hand, "{inst:?}");
            table
        };
        let cases: Vec<(MInst, Option<Vec<u32>>)> = vec![
            // v_add_f64 v[0:1], 1.0 / -1 / 0x3f800000 (2^-7), v[2:3]
            (MInst::new(Opc::VAddF64, [v(0, 2)], [imm(0x3ff0_0000), v(2, 2)]), Some(vec![0xd727_0000, 0x0002_04f2])),
            (MInst::new(Opc::VAddF64, [v(0, 2)], [imm(0xffff_ffff), v(2, 2)]), Some(vec![0xd727_0000, 0x0002_04c1])),
            (
                MInst::new(Opc::VAddF64, [v(0, 2)], [imm(0x3f80_0000), v(2, 2)]),
                Some(vec![0xd727_0000, 0x0002_04ff, 0x3f80_0000]),
            ),
            // v_min_f64 v[0:1], +inf, v[2:3]; v_mul_f64 v[0:1], v[2:3], -4.0
            (
                MInst::new(Opc::VMinF64, [v(0, 2)], [imm(0x7ff0_0000), v(2, 2)]),
                Some(vec![0xd729_0000, 0x0002_04ff, 0x7ff0_0000]),
            ),
            (MInst::new(Opc::VMulF64, [v(0, 2)], [v(2, 2), imm(0xc010_0000)]), Some(vec![0xd728_0000, 0x0001_ef02])),
            // v_fma_f64 v[0:1], -v[2:3], v[4:5], 1.0
            (
                MInst::new(Opc::VFmaF64, [v(0, 2)], [v(2, 2), v(4, 2), imm(0x3ff0_0000)]).with_mods(M_NEG0),
                Some(vec![0xd614_0000, 0x23ca_0902]),
            ),
            // v_div_scale_f64 v[0:1], vcc_lo, 3.0, 3.0, v[2:3]: one shared literal
            (
                MInst::new(Opc::VDivScaleF64, [v(0, 2), Opnd::P(Phys::VccLo, 1)], [imm(0x4008_0000), imm(0x4008_0000), v(2, 2)]),
                Some(vec![0xd6fd_6a00, 0x0409_feff, 0x4008_0000]),
            ),
            // v_rcp_f64_e32 v[0:1], 1.0 / 3.0
            (MInst::new(Opc::VRcpF64, [v(0, 2)], [imm(0x3ff0_0000)]), Some(vec![0x7e00_5ef2])),
            (MInst::new(Opc::VRcpF64, [v(0, 2)], [imm(0x4008_0000)]), Some(vec![0x7e00_5eff, 0x4008_0000])),
            // v_ldexp_f64 v[0:1], v[2:3], 1023 / 0x3f800000: the exponent is a 32-bit source
            (
                MInst::new(Opc::VLdexpF64, [v(0, 2)], [v(2, 2), imm(1023)]),
                Some(vec![0xd72b_0000, 0x0001_ff02, 0x0000_03ff]),
            ),
            (MInst::new(Opc::VLdexpF64, [v(0, 2)], [v(2, 2), imm(0x3f80_0000)]), Some(vec![0xd72b_0000, 0x0001_e502])),
            // 64-bit SALU sources: 0x3f800000 is a literal; the shift amount is 32-bit
            (MInst::new(Opc::SMovB64, [s(0, 2)], [imm(0x3f80_0000)]), Some(vec![0xbe80_01ff, 0x3f80_0000])),
            (
                MInst::new(Opc::SLshlB64, [s(0, 2)], [imm(0x3f80_0000), imm(0x3f80_0000)]),
                Some(vec![0x8480_f2ff, 0x3f80_0000]),
            ),
            (MInst::new(Opc::SCmpEqU64, [], [imm(0x3f80_0000), s(2, 2)]), Some(vec![0xbf10_02ff, 0x3f80_0000])),
            // 16-bit pairs: llvm-mc keeps 0x3f800000 a literal, which VOP3P refuses here
            (MInst::new(Opc::VPkAddF16, [v(0, 1)], [imm(0x3f80_0000), v(1, 1)]), None),
            (MInst::new(Opc::VDot2F32F16, [v(0, 1)], [imm(0x3f80_0000), v(1, 1), v(2, 1)]), None),
            (MInst::new(Opc::VDot2F32Bf16, [v(0, 1)], [v(1, 1), imm(0x3f80_0000), v(2, 1)]), None),
            (MInst::new(Opc::VDot2F32F16, [v(0, 1)], [v(1, 1), v(2, 1), imm(0x3f80_0000)]), Some(vec![0xcc13_4000, 0x1bca_0501])),
            (MInst::new(Opc::VPkFmaF16, [v(0, 1)], [v(1, 1), v(2, 1), imm(1)]), Some(vec![0xcc0e_4000, 0x1a06_0501])),
            // global_atomic_add_u32 v2, v[2:3], v2, off glc
            (MInst::new(Opc::GAtomicAddU32Rtn, [v(2, 1)], [v(2, 2), v(2, 1)]).with_mods(M_GLC), Some(vec![0xdcd6_4000, 0x027c_0202])),
            (MInst::new(Opc::GAtomicAddU32Rtn, [v(2, 1)], [v(2, 2), v(2, 1)]), None),
            // s_waitcnt_vscnt null / s4, 0
            (MInst::new(Opc::SWaitcntVscnt, [Opnd::P(Phys::Null, 1)], []), Some(vec![0xbc7c_0000])),
            (MInst::new(Opc::SWaitcntVscnt, [s(4, 1)], []), None),
        ];
        for (inst, want) in cases {
            assert_eq!(both(inst.clone()), want, "{inst:?}");
        }
    }

    #[test]
    fn every_opcode_has_its_format_and_a_signature() {
        for &op in &t::ALL {
            let sig = t::signature(op);
            assert!(sig.defs.len() <= 2 && sig.uses.len() <= 4, "{}", t::variant(op));
            assert_eq!(t::vopd(op).is_some() && t::vopd_special(op).is_some(), false, "{}", t::variant(op));
        }
        for &(code, name) in &t::VCMP {
            assert_eq!(Opc::VCmp(code).name(), name);
        }
    }

    #[test]
    fn info_agrees_with_the_opcode_methods() {
        let vcmp = t::VCMP.iter().map(|&(code, _)| Opc::VCmp(code));
        for op in t::ALL.iter().copied().chain(vcmp) {
            let info = t::info(op);
            assert_eq!((info.code, info.flags, info.fmt), (op.code(), op.flags(), op.fmt()), "{}", op.name());
        }
    }

    /// A deterministic stream of choices.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self, n: usize) -> usize {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 33) % n as u64) as usize
        }
    }

    const IMMS: [u32; 15] = [
        0, 1, 64, 65, 0xffff_ffff, 0xffff_fff0, 0xffff_ffef, 0x3f80_0000, 0x3e22_f983, 0x1234_5678, 0xc080_0000, 0x7fc0_0000,
        0x3ff0_0000, 0xc010_0000, 0x7ff0_0000,
    ];

    fn sgpr(rng: &mut Rng, n: u8) -> Opnd {
        let align = if n >= 4 { 4 } else { n as u16 };
        let r = [0u16, 1, 2, 4, 9, 20, 64, 100][rng.next(8)] / align * align;
        Opnd::P(Phys::S(r), n)
    }

    fn vgpr_of(rng: &mut Rng, n: u8) -> Opnd {
        let r = [0u16, 1, 2, 7, 64, 100, 127, 128, 129, 200, 240][rng.next(11)];
        Opnd::P(Phys::V(r.min(256 - n as u16)), n)
    }

    fn special(rng: &mut Rng) -> Opnd {
        let p = [Phys::VccLo, Phys::Null, Phys::M0, Phys::ExecLo, Phys::Scc][rng.next(5)];
        Opnd::P(p, 1)
    }

    /// An operand of `class`, sometimes of another class too, so that the
    /// encoders' rejections are compared as well.
    fn operand(rng: &mut Rng, class: t::Class, dwords: u8) -> Opnd {
        use t::Class as C;
        let imm = || Opnd::Imm(0);
        let pick = rng.next(16);
        if pick == 0 {
            return special(rng);
        }
        match class {
            C::S | C::X => {
                if class == C::X {
                    Opnd::P(Phys::ExecLo, 1)
                } else if dwords == 1 && pick < 4 {
                    special(rng)
                } else {
                    sgpr(rng, dwords)
                }
            }
            C::V | C::Vh => vgpr_of(rng, dwords),
            C::Ss | C::Vs | C::Vi | C::Vsh => match pick % 4 {
                0 => Opnd::Imm(IMMS[rng.next(IMMS.len())]),
                1 if class != C::Ss => vgpr_of(rng, dwords),
                1 => imm(),
                2 => sgpr(rng, dwords),
                _ if class == C::Ss => sgpr(rng, dwords),
                _ => vgpr_of(rng, dwords),
            },
            C::A => vgpr_of(rng, dwords),
        }
    }

    fn imm_for(rng: &mut Rng, kind: t::Imm) -> i64 {
        use t::Imm as I;
        let offsets = [0i64, 4, 8, 4095, 4096, -8, -4096, -4097, 65535, 0x1f_ffff];
        match kind {
            I::None => [0, 0, 7][rng.next(3)],
            I::Simm16 | I::Waitcnt | I::Delay | I::Sendmsg | I::Nop | I::Clause | I::Swizzle | I::Label => {
                [0i64, 1, 0x91, 0x3f7, 0xffff, 0x1234][rng.next(6)]
            }
            I::Vopd => 0,
            I::Dpp16 => [0x00ff_0161i64, 0x00ff_096f, 0x00ff_08b1][rng.next(3)],
            I::Dpp8 => [0x00de_54c1i64, 0x00fa_c688][rng.next(2)],
            _ => offsets[rng.next(offsets.len())],
        }
    }

    fn mods_for(rng: &mut Rng, allowed: u32) -> u32 {
        match rng.next(6) {
            0 => 0,
            1 => allowed,
            2 => 1 << rng.next(13),
            _ => allowed & (rng.0 as u32),
        }
    }

    fn sample(rng: &mut Rng, op: Opc) -> MInst {
        let sig = t::signature(op);
        let mut defs = Defs::new();
        for o in sig.defs {
            defs.push(operand(rng, o.class, o.dwords));
        }
        let mods = mods_for(rng, sig.mods);
        // Through saddr a global access takes a 32-bit VGPR offset and an
        // SGPR base pair in place of its VGPR address pair.
        let saddr = op.fmt() == Fmt::Global && mods & M_SADDR != 0 && t::form(op) != Form::Scratch;
        let mut uses = Uses::new();
        for (i, o) in sig.uses.iter().enumerate() {
            if saddr && i == 0 {
                uses.push(vgpr_of(rng, 1));
                uses.push(sgpr(rng, 2));
            } else {
                uses.push(operand(rng, o.class, o.dwords));
            }
        }
        let mut inst = MInst::new(op, defs, uses).with_imm(imm_for(rng, sig.imm));
        inst.mods = mods;
        inst
    }

    fn vopd_sample(rng: &mut Rng) -> MInst {
        let codes = [0u8, 3, 4, 5, 6, 8, 10, 11, 16, 17, 18];
        let (cx, cy) = (codes[rng.next(8)], codes[rng.next(codes.len())]);
        let mut uses = Uses::new();
        let half = |rng: &mut Rng, uses: &mut Uses, code: u8| -> u32 {
            uses.push(operand(rng, t::Class::Vs, 1));
            if code != 8 {
                uses.push(vgpr_of(rng, 1));
            }
            if code == 8 { 1 } else { 2 }
        };
        let nx = half(rng, &mut uses, cx);
        half(rng, &mut uses, cy);
        let defs = [vgpr_of(rng, 1), vgpr_of(rng, 1)];
        let mut inst = MInst::new(Opc::VDual, defs, uses).with_imm(cx as i64 | (cy as i64) << 8);
        inst.mods = nx;
        inst
    }

    /// The table-driven encoder and asm.rs's hand encoder agree on every
    /// table-encoded opcode, on operands of the opcode's signature and on
    /// some outside it: the same words, or both reject.
    #[test]
    fn table_encoder_matches_the_hand_encoder() {
        let mut rng = Rng(0x5eed_1234_abcd_0001);
        let mut ops: Vec<Opc> = t::ALL.iter().copied().filter(|&op| t::table_encoded(op)).collect();
        ops.extend(t::VCMP.iter().map(|&(code, _)| Opc::VCmp(code)));
        let (mut agreed, mut encoded) = (0, 0);
        for &op in &ops {
            for _ in 0..600 {
                let inst = if op == Opc::VDual { vopd_sample(&mut rng) } else { sample(&mut rng, op) };
                let hand = std::panic::catch_unwind(|| super::super::asm::encode_legacy_words(&inst));
                let table = encode_words(&inst);
                match (hand, &table) {
                    (Ok(Ok(h)), Ok(w)) => {
                        assert_eq!(&h, w, "{inst:?}");
                        encoded += 1;
                    }
                    (Ok(Err(_)), Err(_)) | (Err(_), Err(_)) => {}
                    (h, w) => panic!("{inst:?}: hand {h:?}, table {w:?}"),
                }
                agreed += 1;
            }
        }
        assert!(encoded * 3 > agreed, "{encoded} of {agreed} samples encoded");
    }
}
