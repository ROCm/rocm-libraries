// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! gfx11 binary encoder for physical machine IR, plus a text listing on demand.

use std::fmt::Write;

use super::isel::is_inline_f32;
use super::mir::*;
use crate::error::CompileError;

fn bad(message: impl Into<String>) -> CompileError {
    CompileError::new("ROCKE_DIRECT_ENCODE", message)
}

/// 8-bit scalar source / 9-bit vector source operand code, plus a literal.
fn src_code(o: &Opnd, allow_float_inline: bool) -> Result<(u32, Option<u32>), CompileError> {
    Ok(match *o {
        Opnd::P(Phys::V(r), _) => (256 + r as u32, None),
        Opnd::P(Phys::S(r), _) => (r as u32, None),
        Opnd::P(Phys::VccLo, _) => (106, None),
        Opnd::P(Phys::Null, _) => (124, None),
        Opnd::P(Phys::M0, _) => (125, None),
        Opnd::P(Phys::ExecLo, _) => (126, None),
        Opnd::P(Phys::Scc, _) => (253, None),
        Opnd::Imm(v) => {
            let s = v as i32;
            if (0..=64).contains(&s) {
                (128 + s as u32, None)
            } else if (-16..0).contains(&s) {
                (192 + (-s) as u32, None)
            } else if allow_float_inline && is_inline_f32(v) {
                let code = match v {
                    0x3f00_0000 => 240,
                    0xbf00_0000 => 241,
                    0x3f80_0000 => 242,
                    0xbf80_0000 => 243,
                    0x4000_0000 => 244,
                    0xc000_0000 => 245,
                    0x4080_0000 => 246,
                    0xc080_0000 => 247,
                    _ => 248,
                };
                (code, None)
            } else {
                (255, Some(v))
            }
        }
        Opnd::R(_) => return Err(bad("virtual register reached the encoder")),
    })
}

fn sdst_code(o: &Opnd) -> Result<u32, CompileError> {
    Ok(match *o {
        Opnd::P(Phys::S(r), _) => r as u32,
        Opnd::P(Phys::VccLo, _) => 106,
        Opnd::P(Phys::Null, _) => 124,
        Opnd::P(Phys::M0, _) => 125,
        Opnd::P(Phys::ExecLo, _) => 126,
        _ => return Err(bad(format!("{o:?} is not a scalar destination"))),
    })
}

fn vgpr(o: &Opnd) -> Result<u32, CompileError> {
    match *o {
        Opnd::P(Phys::V(r), _) => Ok(r as u32),
        _ => Err(bad(format!("{o:?} is not a VGPR"))),
    }
}

fn is_vgpr(o: &Opnd) -> bool {
    matches!(o, Opnd::P(Phys::V(_), _))
}

struct Enc {
    words: Vec<u32>,
}

impl Enc {
    fn push(&mut self, w: u32) {
        self.words.push(w);
    }
}

fn push_literal(e: &mut Enc, lits: &[Option<u32>]) -> Result<(), CompileError> {
    let mut lit: Option<u32> = None;
    for l in lits.iter().flatten() {
        match lit {
            Some(x) if x != *l => return Err(bad("two different literals in one instruction")),
            _ => lit = Some(*l),
        }
    }
    if let Some(l) = lit {
        e.push(l);
    }
    Ok(())
}

fn vop3_words(
    e: &mut Enc,
    op10: u32,
    dst: u32,
    sdst: Option<u32>,
    srcs: &[Opnd],
    mods: u32,
    float_inline: bool,
) -> Result<(), CompileError> {
    let mut codes = [0u32; 3];
    let mut lits = [None; 3];
    for (i, s) in srcs.iter().enumerate() {
        let (c, l) = src_code(s, float_inline)?;
        codes[i] = c;
        lits[i] = l;
    }
    let neg = mods & 0x7;
    let op_sel = (mods >> 4) & 0xf;
    let mid = match sdst {
        Some(sd) => sd << 8,
        None => op_sel << 11,
    };
    e.push(0xD400_0000 | op10 << 16 | mid | dst);
    e.push(neg << 29 | codes[2] << 18 | codes[1] << 9 | codes[0]);
    push_literal(e, &lits)
}

fn encode(inst: &MInst, e: &mut Enc) -> Result<(), CompileError> {
    let op = inst.op;
    let code = op.code() as u32;
    let flags = op.flags();
    let float_inline = flags & (F_64BIT_SRC | F_16BIT_SRC) == 0;
    match op.fmt() {
        Fmt::Sop1 => {
            let sdst = sdst_code(&inst.defs[0])?;
            let (s0, l0) = src_code(&inst.uses[0], true)?;
            e.push(0xBE80_0000 | sdst << 16 | code << 8 | s0);
            push_literal(e, &[l0])?;
        }
        Fmt::Sop2 => {
            let sdst = sdst_code(&inst.defs[0])?;
            let (s0, l0) = src_code(&inst.uses[0], true)?;
            let (s1, l1) = src_code(&inst.uses[1], true)?;
            e.push(0x8000_0000 | code << 23 | sdst << 16 | s1 << 8 | s0);
            push_literal(e, &[l0, l1])?;
        }
        Fmt::Sopc => {
            let (s0, l0) = src_code(&inst.uses[0], true)?;
            let (s1, l1) = src_code(&inst.uses[1], true)?;
            e.push(0xBF00_0000 | code << 16 | s1 << 8 | s0);
            push_literal(e, &[l0, l1])?;
        }
        Fmt::Sopk => {
            let sdst = sdst_code(&inst.defs[0])?;
            e.push(0xB000_0000 | code << 23 | sdst << 16 | (inst.imm as u32 & 0xffff));
        }
        Fmt::Sopp => {
            e.push(0xBF80_0000 | code << 16 | (inst.imm as u32 & 0xffff));
        }
        Fmt::Smem => {
            let sdata = sdst_code(&inst.defs[0])?;
            let sbase = sdst_code(&inst.uses[0])?;
            e.push(0xF400_0000 | code << 18 | sdata << 6 | sbase >> 1);
            e.push(0xF800_0000 | (inst.imm as u32 & 0x1f_ffff));
        }
        Fmt::Vop1 if inst.defs.is_empty() => {
            e.push(0x7E00_0000 | code << 9);
        }
        Fmt::Vop1 => {
            let dst_is_s = !is_vgpr(&inst.defs[0]);
            let dst = if dst_is_s { sdst_code(&inst.defs[0])? } else { vgpr(&inst.defs[0])? };
            let src = &inst.uses[0];
            let hi_src = inst.mods & M_OPSEL_SRC0 != 0;
            let hi_dst = inst.mods & M_OPSEL_DST != 0;
            let half_dst = flags & F_16BIT_DST != 0;
            let half_src = flags & F_16BIT_SRC != 0;
            let (mut s0, l0) = src_code(src, float_inline)?;
            let short_ok = (!half_dst || dst < 128)
                && (!half_src || !is_vgpr(src) || s0 - 256 < 128)
                && inst.mods & 0x7 == 0;
            if short_ok {
                let mut vdst = dst;
                if half_dst && hi_dst {
                    vdst |= 0x80;
                }
                if half_src && hi_src && is_vgpr(src) {
                    s0 |= 0x80;
                }
                e.push(0x7E00_0000 | vdst << 17 | code << 9 | s0);
                push_literal(e, &[l0])?;
            } else {
                vop3_words(e, 0x180 + code, dst, None, &inst.uses[..1], inst.mods, float_inline)?;
            }
        }
        Fmt::Vop2 => {
            let dst = vgpr(&inst.defs[0])?;
            let s0 = &inst.uses[0];
            let s1 = &inst.uses[1];
            let short = inst.uses.len() == 2 && is_vgpr(s1) && inst.mods == 0 && op != Opc::VCndmaskB32;
            if short {
                let (c0, l0) = src_code(s0, float_inline)?;
                e.push(code << 25 | dst << 17 | vgpr(s1)? << 9 | c0);
                push_literal(e, &[l0])?;
            } else {
                vop3_words(e, 0x100 + code, dst, None, &inst.uses, inst.mods, float_inline)?;
            }
        }
        Fmt::Vop3 => {
            let dst = vgpr(&inst.defs[0])?;
            vop3_words(e, code, dst, None, &inst.uses, inst.mods, float_inline)?;
        }
        Fmt::Vop3b => {
            let dst = vgpr(&inst.defs[0])?;
            let sdst = sdst_code(&inst.defs[1])?;
            vop3_words(e, code, dst, Some(sdst), &inst.uses, inst.mods, float_inline)?;
        }
        Fmt::Vopc => {
            let sdst = sdst_code(&inst.defs[0])?;
            vop3_words(e, code, sdst, None, &inst.uses, inst.mods, float_inline)?;
        }
        Fmt::Vop3p => {
            let dst = vgpr(&inst.defs[0])?;
            // WMMA reads packed sources (op_sel_hi all set); v_fma_mix marks
            // its f16 sources there.
            let (op_sel, op_sel_hi) = if op == Opc::VFmaMixF32 {
                (inst.mods >> 4 & 7, inst.mods / M_MIX_F16_0 & 7)
            } else {
                (0, 7)
            };
            let mut src = [0u32; 3];
            for (s, o) in src.iter_mut().zip(&inst.uses) {
                let (c, literal) = src_code(o, true)?;
                if literal.is_some() {
                    return Err(bad("VOP3P literal"));
                }
                *s = c;
            }
            e.push(0xCC00_0000 | code << 16 | (op_sel_hi >> 2) << 14 | op_sel << 11 | dst);
            e.push((op_sel_hi & 3) << 27 | src[2] << 18 | src[1] << 9 | src[0]);
        }
        Fmt::Vopd => {
            let (dst_x, dst_y) = (vgpr(&inst.defs[0])?, vgpr(&inst.defs[1])?);
            // The encoding stores dst_y[7:1]; bit 0 is the inverse of dst_x's.
            if (dst_x ^ dst_y) & 1 == 0 {
                return Err(bad("VOPD destinations of equal parity"));
            }
            let (xs, ys) = inst.uses.split_at(inst.mods as usize);
            let mut fields = [(0, 0); 2];
            let mut lits = [None; 2];
            for (k, srcs) in [xs, ys].into_iter().enumerate() {
                let (src0, lit) = src_code(&srcs[0], true)?;
                let vsrc1 = match srcs.get(1) {
                    Some(o) => vgpr(o)?,
                    None => 0,
                };
                fields[k] = (src0, vsrc1);
                lits[k] = lit;
            }
            let (op_x, op_y) = (inst.imm as u32 & 0xf, (inst.imm >> 8) as u32 & 0x1f);
            e.push(0xC800_0000 | op_x << 22 | op_y << 17 | fields[0].1 << 9 | fields[0].0);
            e.push(dst_x << 24 | (dst_y >> 1) << 17 | fields[1].1 << 9 | fields[1].0);
            push_literal(e, &lits)?;
        }
        Fmt::Ds => {
            let offset = inst.imm as u32 & 0xffff;
            let addr = vgpr(&inst.uses[0])?;
            let (vdst, data0) = if inst.defs.is_empty() {
                (0, vgpr(&inst.uses[1])?)
            } else {
                (vgpr(&inst.defs[0])?, 0)
            };
            e.push(0xD800_0000 | code << 18 | offset);
            e.push(vdst << 24 | data0 << 8 | addr);
        }
        Fmt::Global => {
            let saddr_mode = inst.mods & M_SADDR != 0;
            let addr = vgpr(&inst.uses[0])?;
            let saddr = if saddr_mode { sdst_code(&inst.uses[1])? } else { 0x7c };
            let next = if saddr_mode { 2 } else { 1 };
            let (vdst, data) = if inst.defs.is_empty() {
                (0, vgpr(&inst.uses[next])?)
            } else {
                (vgpr(&inst.defs[0])?, 0)
            };
            let offset = inst.imm;
            if !(-4096..4096).contains(&offset) {
                return Err(bad(format!("global offset {offset} out of range")));
            }
            e.push(0xDC00_0000 | code << 18 | 2 << 16 | (offset as u32 & 0x1fff));
            e.push(vdst << 24 | saddr << 16 | data << 8 | addr);
        }
        Fmt::Mubuf => {
            let (vdata, rest) = if inst.defs.is_empty() {
                (vgpr(&inst.uses[0])?, &inst.uses[1..])
            } else {
                (vgpr(&inst.defs[0])?, &inst.uses[..])
            };
            let vaddr = vgpr(&rest[0])?;
            let srsrc = sdst_code(&rest[1])?;
            let (soffset, lit) = src_code(&rest[2], false)?;
            if lit.is_some() {
                return Err(bad("buffer soffset literal"));
            }
            let offset = inst.imm as u32;
            if offset >= 4096 {
                return Err(bad(format!("buffer offset {offset} out of range")));
            }
            e.push(0xE000_0000 | code << 18 | offset);
            e.push(soffset << 24 | 0x40_0000 | (srsrc / 4) << 16 | vdata << 8 | vaddr);
        }
        Fmt::Pseudo => return Err(bad(format!("pseudo {} reached the encoder", op.name()))),
    }
    Ok(())
}

/// Encodes all blocks; returns the words, the word offset of each block (plus
/// the end), and each branch's word offset and target block. Branches are one
/// word whose low 16 bits still hold the target block.
fn encode_blocks(f: &MFunc) -> Result<(Vec<u32>, Vec<u32>, Vec<(usize, usize)>), CompileError> {
    let mut e = Enc { words: Vec::with_capacity(f.blocks.iter().map(|b| 2 * b.insts.len()).sum()) };
    let mut block_word = Vec::with_capacity(f.blocks.len() + 1);
    let mut branches = Vec::new();
    for block in &f.blocks {
        block_word.push(e.words.len() as u32);
        for inst in &block.insts {
            if inst.is_branch() {
                branches.push((e.words.len(), inst.imm as usize));
            }
            encode(inst, &mut e)?;
        }
    }
    block_word.push(e.words.len() as u32);
    Ok((e.words, block_word, branches))
}

/// Word offset from the branch at word `at` to block `target`.
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
    let mut code = Vec::with_capacity(4 * words.len());
    for w in words {
        code.extend_from_slice(&w.to_le_bytes());
    }
    Ok(code)
}

/// Text listing of the encoded blocks: each instruction with its resolved
/// branch offset, byte offset and encoding.
pub fn listing(f: &MFunc) -> Result<String, CompileError> {
    let (_, block_word, _) = encode_blocks(f)?;
    let mut words = 0usize;
    let mut listing = String::new();
    for (b, block) in f.blocks.iter().enumerate() {
        let _ = writeln!(listing, "bb{b}:");
        for inst in &block.insts {
            let mut inst = inst.clone();
            if inst.is_branch() {
                inst.imm = branch_delta(&block_word, words, inst.imm as usize)? & 0xffff;
            }
            let mut e = Enc { words: Vec::new() };
            encode(&inst, &mut e)?;
            let hex: Vec<String> = e.words.iter().map(|w| format!("{w:08X}")).collect();
            let _ = writeln!(listing, "  {:<60} // {:06X}: {}", inst_text(f, &inst), words * 4, hex.join(" "));
            words += e.words.len();
        }
    }
    Ok(listing)
}
