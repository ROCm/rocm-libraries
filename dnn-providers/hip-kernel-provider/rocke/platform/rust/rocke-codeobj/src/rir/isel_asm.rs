// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! gfx950's general `tile.inline_asm`: the template assembled into machine
//! instructions on virtual registers, so register allocation, `s_waitcnt`
//! insertion and the hazard pass see every instruction (LLVM keeps the block
//! opaque; the template's own waits stay as explicit, fencing waits).
//!
//! The probes' bounded polls (P4 counter / pair, P5 host round) and the CCL
//! state machines and `tag_add` are the templates this covers: SOP1 / SOP2 /
//! SOPC / SOPP, VOP1 / VOP2 / VOP3 / VOPC (`_e64`), GLOBAL loads and stores
//! through `saddr` or a 64-bit address (`off`) with `sc0` / `sc1` / `nt` and
//! `offset:`, `s_waitcnt`, `s_memrealtime`, and local labels
//! (`${:private}name${:uid}_part:`) with the branches to them; a label that a
//! later branch goes back to opens a loop (its blocks one loop level deeper).
//!
//! Operands follow the constraint string: outputs (`=v`, `=&v`, `=s`, `=&s`)
//! get fresh registers of their result's size; inputs (`v`, `s`) are the
//! operand values moved into that file; a digit ties an input to that output
//! (copied in first); `~{memory}` fences the block for the scheduler. Early
//! clobbers need nothing: the instructions' own reads and writes give the
//! allocator the live ranges. Anything else in a template is refused.

use super::*;
use crate::target::gfx9::gen_isa as t;

/// One constraint of the string, outputs first.
#[derive(Clone, Copy, Debug)]
enum Constraint {
    Out(char),
    /// An output in fixed VGPRs `{v[a:b]}` / `{vN}`: the first register and the count.
    OutFixed(u16, u8),
    In(char),
    Tied(usize),
}

/// `{v[a:b]}` or `{vN}`: the first VGPR and how many.
fn fixed_vgprs(text: &str) -> Option<(u16, u8)> {
    let inner = text.strip_prefix("{v")?.strip_suffix('}')?;
    if let Some(range) = inner.strip_prefix('[').and_then(|r| r.strip_suffix(']')) {
        let (a, b) = range.split_once(':')?;
        let (a, b): (u16, u16) = (a.parse().ok()?, b.parse().ok()?);
        return (b >= a && b - a < 16).then_some((a, (b - a + 1) as u8));
    }
    Some((inner.parse().ok()?, 1))
}

fn parse_constraints(text: &str) -> Result<(Vec<Constraint>, Vec<String>), CompileError> {
    let (mut out, mut clobbers) = (Vec::new(), Vec::new());
    for part in text.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        if let Some(c) = part.strip_prefix("~{").and_then(|c| c.strip_suffix('}')) {
            clobbers.push(c.to_string());
        } else if let Some(rest) = part.strip_prefix('=') {
            match rest.trim_start_matches('&') {
                c @ ("v" | "s") => out.push(Constraint::Out(c.chars().next().unwrap())),
                other => match fixed_vgprs(other) {
                    Some((first, n)) => out.push(Constraint::OutFixed(first, n)),
                    None => return Err(unsupported(format!("inline asm output constraint {other:?}"))),
                },
            }
        } else if let Ok(k) = part.parse::<usize>() {
            out.push(Constraint::Tied(k));
        } else {
            match part {
                c @ ("v" | "s") => out.push(Constraint::In(c.chars().next().unwrap())),
                other => return Err(unsupported(format!("inline asm constraint {other:?}"))),
            }
        }
    }
    Ok((out, clobbers))
}

/// A label reference with LLVM's `${:private}` and `${:uid}` dropped.
fn label_name(text: &str) -> String {
    text.replace("${:private}", "").replace("${:uid}", "")
}

/// A template line: a label definition or an instruction with its operand
/// texts and trailing modifier words.
enum Line {
    Label(String),
    Inst { mnemonic: String, operands: Vec<String>, mods: Vec<String> },
}

fn parse_template(template: &str) -> Vec<Line> {
    let mut out = Vec::new();
    for raw in template.lines() {
        let line = raw.split(';').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if let Some(label) = line.strip_suffix(':') {
            out.push(Line::Label(label_name(label.trim())));
            continue;
        }
        let (mnemonic, rest) = match line.split_once(char::is_whitespace) {
            Some((m, r)) => (m.to_string(), r.trim()),
            None => (line.to_string(), ""),
        };
        let mut operands: Vec<String> = Vec::new();
        let mut mods: Vec<String> = Vec::new();
        if !rest.is_empty() {
            let items: Vec<&str> = rest.split(',').map(str::trim).collect();
            let last = items.len() - 1;
            for (i, item) in items.iter().enumerate() {
                if i < last {
                    operands.push(item.to_string());
                } else {
                    let mut words = item.split_whitespace();
                    if let Some(first) = words.next() {
                        operands.push(first.to_string());
                    }
                    mods.extend(words.map(str::to_string));
                }
            }
        }
        out.push(Line::Inst { mnemonic, operands, mods });
    }
    out
}

/// `s_waitcnt` counters as gfx9's immediate (vmcnt 3:0 and 15:14, expcnt 6:4,
/// lgkmcnt 11:8; a counter not named waits for nothing).
fn waitcnt_imm(words: &[String]) -> Result<i64, CompileError> {
    let (mut vm, mut exp, mut lgkm) = (63i64, 7i64, 15i64);
    for w in words {
        let parse = |name: &str| -> Option<i64> { w.strip_prefix(name)?.strip_prefix('(')?.strip_suffix(')')?.parse().ok() };
        if let Some(n) = parse("vmcnt") {
            vm = n;
        } else if let Some(n) = parse("expcnt") {
            exp = n;
        } else if let Some(n) = parse("lgkmcnt") {
            lgkm = n;
        } else {
            return Err(unsupported(format!("inline asm s_waitcnt operand {w:?}")));
        }
    }
    Ok((vm & 15) | (vm >> 4 & 3) << 14 | (exp & 7) << 4 | (lgkm & 15) << 8)
}

/// The table opcode of a mnemonic (`_e32` / `_e64` dropped) taking `n`
/// encoded operands, the plain form first (no DPP / SDWA / tied-old rows).
fn opcode_of(mnemonic: &str, n: usize) -> Option<Opc> {
    let base = mnemonic.trim_end_matches("_e32").trim_end_matches("_e64");
    if let Some(&(code, _, _)) = t::VCMP.iter().find(|(_, _, name)| *name == base) {
        return Some(Opc::VCmp(code));
    }
    // Only the formats whose syntax is registers and integers in signature order: SALU and VALU (VOP3P,
    // DPP, SDWA, memory and the rest have their own syntax; the special cases above take what is needed).
    let plain = |op: Opc| {
        t::info(op).is_some_and(|i| {
            i.table
                && matches!(i.form, t::Form::Plain | t::Form::Noshort | t::Form::B64 | t::Form::B64Shift)
                && matches!(i.fmt, Fmt::Sop1 | Fmt::Sop2 | Fmt::Sopc | Fmt::Vop1 | Fmt::Vop2 | Fmt::Vop3 | Fmt::Vop3b | Fmt::Vopc)
        })
    };
    let arity = |op: Opc| {
        t::signature(op).map(|s| s.defs.len() + s.uses.iter().filter(|o| o.role != t::Role::Old).count())
    };
    let mut fallback = None;
    for &op in t::ALL.iter() {
        if t::mnemonic(op) != Some(base) || !plain(op) {
            continue;
        }
        if arity(op) == Some(n) {
            return Some(op);
        }
        fallback.get_or_insert(op);
    }
    fallback
}

impl Isel<'_> {
    /// `tile.inline_asm` (gfx950) other than the identity pins and the LDS
    /// read the callers take first: see the module comment.
    pub(super) fn inline_asm(&mut self, op: &Op) -> Result<(), CompileError> {
        let template = op.str_attr("template").unwrap_or("").to_string();
        let (constraints, clobbers) = parse_constraints(op.str_attr("constraints").unwrap_or(""))?;
        if let Some(c) = clobbers.iter().find(|c| !matches!(c.as_str(), "memory" | "scc" | "vcc")) {
            return Err(unsupported(format!("inline asm clobber {c:?}")));
        }
        let fence = clobbers.iter().any(|c| c == "memory");
        // Operand registers, numbered as the template's `$N`.
        let mut regs: Vec<Opnd> = Vec::with_capacity(constraints.len());
        let mut outs: Vec<(VRef, ValueId)> = Vec::new();
        let mut inputs = op.operands.iter().copied();
        let outputs: Vec<Constraint> = constraints.iter().copied().filter(|c| matches!(c, Constraint::Out(_) | Constraint::OutFixed(..))).collect();
        if outputs.len() != op.results.len() {
            return Err(unsupported(format!("inline asm with {} outputs and {} results", outputs.len(), op.results.len())));
        }
        let dwords_of = |ty: &Type| -> u8 { ty.dwords().max(1) as u8 };
        // Fixed outputs: (first VGPR, count, the vreg precolored there) for the template's raw `vN` names.
        let mut fixed: Vec<(u16, u8, VRef)> = Vec::new();
        for (k, c) in outputs.iter().enumerate() {
            let res = op.results[k];
            let n = dwords_of(self.k.ty(res));
            let reg = match *c {
                Constraint::Out('v') => self.vgpr(n),
                Constraint::Out(_) => self.sreg(n),
                Constraint::OutFixed(first, count) => {
                    if count != n {
                        return Err(unsupported(format!("inline asm fixed output of {count} VGPRs for a {n}-dword result")));
                    }
                    let reg = self.vgpr(n);
                    self.f.vregs[reg.reg as usize].fixed = Some(first);
                    fixed.push((first, count, reg));
                    reg
                }
                _ => unreachable!(),
            };
            regs.push(r(reg));
            outs.push((reg, res));
        }
        for c in constraints.iter().filter(|c| !matches!(c, Constraint::Out(_) | Constraint::OutFixed(..))) {
            let v = inputs.next().ok_or_else(|| unsupported("inline asm with fewer operands than input constraints"))?;
            let ty = self.k.ty(v).clone();
            let wide = dwords_of(&ty);
            let reg = match *c {
                Constraint::Tied(k) => {
                    let (dst, _) = *outs.get(k).ok_or_else(|| unsupported(format!("inline asm tied to output {k}")))?;
                    let src = self.asm_value(v, wide, if self.f.vregs[dst.reg as usize].class == Class::V { 'v' } else { 's' })?;
                    self.copy(dst, src);
                    r(dst)
                }
                Constraint::In(class) => self.asm_value(v, wide, class)?,
                Constraint::Out(_) | Constraint::OutFixed(..) => unreachable!(),
            };
            regs.push(reg);
        }
        if inputs.next().is_some() {
            return Err(unsupported("inline asm with more operands than constraints"));
        }
        let lines = parse_template(&template);
        // Loop nesting per line: a label some later branch goes back to.
        let mut nest = vec![0u32; lines.len() + 1];
        for (j, line) in lines.iter().enumerate() {
            if let Line::Inst { mnemonic, operands, .. } = line {
                if mnemonic.starts_with("s_branch") || mnemonic.starts_with("s_cbranch") {
                    let target = label_name(operands.first().map(String::as_str).unwrap_or(""));
                    if let Some(i) = lines.iter().position(|l| matches!(l, Line::Label(n) if *n == target)) {
                        if i < j {
                            for d in &mut nest[i..=j] {
                                *d += 1;
                            }
                        }
                    }
                }
            }
        }
        let base_depth = self.depth;
        if fence {
            self.emit(MInst::new(Opc::SchedBarrier, [], []));
        }
        let mut labels: FxHashMap<String, usize> = FxHashMap::default();
        let mut fixups: Vec<(usize, usize, String)> = Vec::new();
        let mut ended = false;
        for (j, line) in lines.iter().enumerate() {
            match line {
                Line::Label(name) => {
                    self.depth = base_depth + nest[j];
                    let b = self.new_block();
                    if labels.insert(name.clone(), b).is_some() {
                        return Err(unsupported(format!("inline asm label {name:?} twice")));
                    }
                    ended = false;
                }
                Line::Inst { mnemonic, operands, mods } => {
                    if ended {
                        self.depth = base_depth + nest[j];
                        self.new_block();
                        ended = false;
                    }
                    let branch = self.asm_inst(mnemonic, operands, mods, &regs, &fixed)?;
                    if let Some(target) = branch {
                        // The branch is the block's last instruction; its target is set below.
                        let block = self.cur;
                        fixups.push((block, self.f.blocks[block].insts.len() - 1, target));
                        ended = true;
                    } else {
                        // The template's order is kept: no instruction moves across another of the block.
                        self.emit(MInst::new(Opc::SchedBarrier, [], []));
                    }
                }
            }
        }
        self.depth = base_depth;
        if ended {
            self.new_block();
        }
        for (block, at, target) in fixups {
            let b = *labels.get(&target).ok_or_else(|| unsupported(format!("inline asm branch to unknown label {target:?}")))?;
            self.f.blocks[block].insts[at].imm = b as i64;
        }
        if fence {
            self.emit(MInst::new(Opc::SchedBarrier, [], []));
        }
        for (reg, res) in outs {
            let ty = self.k.ty(res).clone();
            let val = match (&ty, reg.n) {
                (_, 1) => Val::W(r(reg)),
                (Type::Scalar(Scalar::I64 | Scalar::F64), 2) | (Type::Ptr(_), 2) => Val::D(r(sub(reg, 0, 1)), r(sub(reg, 1, 1))),
                _ => Val::T(reg),
            };
            self.set(res, val);
        }
        Ok(())
    }

    /// An input operand of `wide` dwords in register file `class`.
    fn asm_value(&mut self, v: ValueId, wide: u8, class: char) -> Result<Opnd, CompileError> {
        // An `s` operand of a value this analysis cannot prove uniform: LLVM accepted the program only if
        // its divergence analysis found it uniform (else "illegal VGPR to SGPR copy"), so the value is the
        // same in every lane and the first lane's copy is exact (`v_readfirstlane`).
        if wide == 1 {
            let x = self.w(v)?;
            return Ok(if class == 'v' { self.to_v(x) } else { self.to_s(x) });
        }
        if let Ok((lo, hi)) = self.d(v) {
            if wide == 2 {
                return Ok(r(if class == 'v' { self.pair_v(lo, hi) } else { self.pair_s(lo, hi) }));
            }
        }
        let t = self.tuple(v)?;
        let file = self.f.vregs[t.reg as usize].class;
        if class == 'v' && file != Class::V {
            let dst = self.vgpr(t.n);
            self.copy(dst, r(t));
            return Ok(r(dst));
        }
        if class == 's' && file == Class::V {
            let dst = self.sreg(t.n);
            for k in 0..t.n {
                self.emit(MInst::new(Opc::VReadfirstlaneB32, [r(sub(dst, k, 1))], [r(sub(t, k, 1))]));
            }
            return Ok(r(dst));
        }
        Ok(r(t))
    }

    /// One template instruction; returns the label it branches to.
    fn asm_inst(&mut self, mnemonic: &str, operands: &[String], mods: &[String], regs: &[Opnd], fixed: &[(u16, u8, VRef)]) -> Result<Option<String>, CompileError> {
        use crate::target::gfx9::cpol::{M_NT, M_SC1};
        let opnd = |s: &Self, text: &str| -> Result<Opnd, CompileError> {
            if let Some(k) = text.strip_prefix('$') {
                let k: usize = k.parse().map_err(|_| unsupported(format!("inline asm operand {text:?}")))?;
                return regs.get(k).copied().ok_or_else(|| unsupported(format!("inline asm operand ${k} past the constraints")));
            }
            match text {
                "exec" => return Ok(s.exec()),
                "vcc" => return Ok(s.vcc()),
                _ => {}
            }
            // A raw VGPR name: only inside a fixed output (`{v[a:b]}`), as that output's dwords.
            if let Some(name) = text.strip_prefix('v') {
                let (a, n) = match name.strip_prefix('[').and_then(|r| r.strip_suffix(']')).and_then(|r| r.split_once(':')) {
                    Some((a, b)) => match (a.parse::<u16>(), b.parse::<u16>()) {
                        (Ok(a), Ok(b)) if b >= a => (a, (b - a + 1) as u8),
                        _ => return Err(unsupported(format!("inline asm operand {text:?}"))),
                    },
                    None => match name.parse::<u16>() {
                        Ok(a) => (a, 1),
                        Err(_) => return Err(unsupported(format!("inline asm operand {text:?}"))),
                    },
                };
                let within = fixed.iter().find(|&&(first, count, _)| a >= first && a + n as u16 <= first + count as u16);
                return match within {
                    Some(&(first, _, reg)) => Ok(Opnd::R(VRef { off: reg.off + (a - first) as u8, n, ..reg })),
                    None => Err(unsupported(format!("inline asm register {text} outside the fixed outputs"))),
                };
            }
            let v = if let Some(h) = text.strip_prefix("0x") {
                i64::from_str_radix(h, 16).ok()
            } else {
                text.parse::<i64>().ok()
            };
            v.map(|v| Opnd::Imm(v as u32)).ok_or_else(|| unsupported(format!("inline asm operand {text:?}")))
        };
        match mnemonic {
            "s_waitcnt" => {
                let mut words: Vec<String> = operands.to_vec();
                words.extend(mods.iter().cloned());
                self.emit(MInst::new(Opc::SWaitcntExplicit, [], []).with_imm(waitcnt_imm(&words)?));
                return Ok(None);
            }
            "s_branch" | "s_cbranch_scc0" | "s_cbranch_scc1" | "s_cbranch_execz" | "s_cbranch_execnz" => {
                let opc = match mnemonic {
                    "s_branch" => Opc::SBranch,
                    "s_cbranch_scc0" => Opc::SCbranchScc0,
                    "s_cbranch_scc1" => Opc::SCbranchScc1,
                    "s_cbranch_execz" => Opc::SCbranchExecz,
                    _ => Opc::SCbranchExecnz,
                };
                let target = label_name(operands.first().map(String::as_str).unwrap_or(""));
                self.emit(MInst::new(opc, [], []));
                return Ok(Some(target));
            }
            "s_nop" | "s_sleep" => {
                let n = operands.first().and_then(|o| o.parse::<i64>().ok()).unwrap_or(0);
                let opc = if mnemonic == "s_nop" { Opc::SNop } else { Opc::Gfx9(t::Gfx9Op::SSleep) };
                self.emit(MInst::new(opc, [], []).with_imm(n));
                return Ok(None);
            }
            "s_memrealtime" => {
                let d = opnd(self, &operands[0])?;
                self.emit(MInst::new(Opc::Gfx9(t::Gfx9Op::SMemrealtime), [d], []));
                return Ok(None);
            }
            _ => {}
        }
        if mnemonic == "s_dcache_inv" {
            self.emit(MInst::new(Opc::Gfx9(t::Gfx9Op::SDcacheInv), [], []));
            return Ok(None);
        }
        if let Some(opc) = match mnemonic {
            "s_load_dword" => Some(Opc::SLoadB32),
            "s_load_dwordx2" => Some(Opc::SLoadB64),
            "s_load_dwordx4" => Some(Opc::SLoadB128),
            "s_buffer_load_dword" => Some(Opc::Gfx9(t::Gfx9Op::SBufLoadB32)),
            _ => None,
        } {
            // sdata, sbase (pair or descriptor), byte offset [glc]
            let [d, base, off] = operands else {
                return Err(unsupported(format!("inline asm {mnemonic} with {} operands", operands.len())));
            };
            let offset = opnd(self, off)?;
            let Opnd::Imm(offset) = offset else {
                return Err(unsupported(format!("inline asm {mnemonic} with a register offset")));
            };
            let mut m = 0;
            for w in mods {
                match w.as_str() {
                    "glc" | "sc0" => m |= M_GLC,
                    other => return Err(unsupported(format!("inline asm {mnemonic} modifier {other:?}"))),
                }
            }
            let (d, base) = (opnd(self, d)?, opnd(self, base)?);
            self.emit(MInst::new(opc, [d], [base]).with_imm(offset as i32 as i64).with_mods(m));
            return Ok(None);
        }
        if mnemonic == "buffer_inv" {
            let mut cpol = 0i64;
            for w in mods.iter().chain(operands.iter()) {
                match w.as_str() {
                    "sc0" => cpol |= 1,
                    "sc1" => cpol |= 2,
                    other => return Err(unsupported(format!("inline asm buffer_inv modifier {other:?}"))),
                }
            }
            self.emit(MInst::new(Opc::Gfx9(t::Gfx9Op::BufferInv), [], []).with_imm(cpol));
            return Ok(None);
        }
        if let Some(width) = mnemonic.strip_prefix("global_atomic_add") {
            // global_atomic_add[_x2] [vdst,] vaddr, vdata, saddr|off [sc0 = returns] [sc1] [offset:]
            let wide = match width {
                "" => false,
                "_x2" => true,
                _ => return Err(unsupported(format!("inline asm {mnemonic}"))),
            };
            let (mut m, mut offset, mut returns) = (0u32, 0i64, false);
            for w in mods {
                match w.as_str() {
                    "sc0" | "glc" => returns = true,
                    "sc1" => m |= M_SC1,
                    w if w.starts_with("offset:") => offset = w["offset:".len()..].parse().map_err(|_| unsupported(format!("inline asm {w}")))?,
                    other => return Err(unsupported(format!("inline asm atomic modifier {other:?}"))),
                }
            }
            let want = if returns { 4 } else { 3 };
            if operands.len() != want {
                return Err(unsupported(format!("inline asm {mnemonic} with {} operands", operands.len())));
            }
            let vals: Vec<Opnd> = operands.iter().map(|o| opnd(self, o)).collect::<Result<_, _>>()?;
            let (dst, rest) = if returns { (Some(vals[0]), &vals[1..]) } else { (None, &vals[..]) };
            let mut uses = Uses::new();
            uses.push(rest[0]);
            if operands[want - 1] != "off" {
                m |= M_SADDR;
                uses.push(rest[2]);
            }
            uses.push(rest[1]);
            let opc = match (wide, returns) {
                (false, false) => Opc::GAtomicAddU32,
                (false, true) => Opc::GAtomicAddU32Rtn,
                (true, false) => Opc::Gfx9(t::Gfx9Op::GAtomicAddX2),
                (true, true) => Opc::Gfx9(t::Gfx9Op::GAtomicAddX2Rtn),
            };
            let inst = match dst {
                Some(d) => MInst::new(opc, [d], uses).with_mods(m | M_GLC),
                None => MInst::new(opc, [], uses).with_mods(m),
            };
            self.emit(inst.with_imm(offset));
            return Ok(None);
        }
        if let Some(width) = mnemonic.strip_prefix("global_load_dword").or_else(|| mnemonic.strip_prefix("global_store_dword")) {
            let store = mnemonic.starts_with("global_store");
            let opc = match (store, width) {
                (false, "") => Opc::GLoadB32,
                (false, "x2") => Opc::GLoadB64,
                (false, "x4") => Opc::GLoadB128,
                (true, "") => Opc::GStoreB32,
                (true, "x2") => Opc::GStoreB64,
                (true, "x4") => Opc::GStoreB128,
                _ => return Err(unsupported(format!("inline asm {mnemonic}"))),
            };
            let (mut m, mut offset) = (0u32, 0i64);
            for w in mods {
                match w.as_str() {
                    "sc0" | "glc" => m |= M_GLC,
                    "sc1" => m |= M_SC1,
                    "nt" => m |= M_NT,
                    w if w.starts_with("offset:") => offset = w["offset:".len()..].parse().map_err(|_| unsupported(format!("inline asm {w}")))?,
                    other => return Err(unsupported(format!("inline asm global modifier {other:?}"))),
                }
            }
            // vdst / vaddr, vaddr / vdata, then saddr or `off`.
            let [a0, a1, a2] = operands else {
                return Err(unsupported(format!("inline asm {mnemonic} with {} operands", operands.len())));
            };
            let (x0, x1) = (opnd(self, a0)?, opnd(self, a1)?);
            let addr: Uses = if a2 == "off" {
                [if store { x0 } else { x1 }].into()
            } else {
                m |= M_SADDR;
                [if store { x0 } else { x1 }, opnd(self, a2)?].into()
            };
            let inst = if store {
                let mut uses = addr;
                uses.push(x1);
                MInst::new(opc, [], uses)
            } else {
                MInst::new(opc, [x0], addr)
            };
            self.emit(inst.with_imm(offset).with_mods(m));
            return Ok(None);
        }
        let opc = opcode_of(mnemonic, operands.len()).ok_or_else(|| unsupported(format!("inline asm instruction {mnemonic}")))?;
        if !mods.is_empty() {
            return Err(unsupported(format!("inline asm {mnemonic} modifiers {mods:?}")));
        }
        let sig = t::signature(opc).ok_or_else(|| unsupported(format!("inline asm {mnemonic} without a signature")))?;
        let nd = sig.defs.len();
        let vals: Vec<Opnd> = operands.iter().map(|o| opnd(self, o)).collect::<Result<_, _>>()?;
        if vals.len() < nd {
            return Err(unsupported(format!("inline asm {mnemonic}: {} operands", vals.len())));
        }
        let mut defs = Defs::new();
        for &d in &vals[..nd] {
            defs.push(d);
        }
        let mut uses = Uses::new();
        for &u in &vals[nd..] {
            uses.push(u);
        }
        self.emit(MInst { op: opc, defs, uses, imm: 0, mods: 0, mem: 0, scope: 0 });
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn templates_parse_into_labels_instructions_and_modifiers() {
        let lines = parse_template("s_mov_b32 $2, 0\n${:private}p4c${:uid}_top:\nglobal_load_dword $0, $3, $4 sc1\ns_waitcnt vmcnt(0)\ns_branch ${:private}p4c${:uid}_top\n; done");
        assert_eq!(lines.len(), 5);
        assert!(matches!(&lines[1], Line::Label(n) if n == "p4c_top"));
        match &lines[2] {
            Line::Inst { mnemonic, operands, mods } => {
                assert_eq!((mnemonic.as_str(), operands.clone(), mods.clone()), ("global_load_dword", vec!["$0".to_string(), "$3".into(), "$4".into()], vec!["sc1".to_string()]));
            }
            _ => panic!("an instruction"),
        }
        assert_eq!(waitcnt_imm(&["vmcnt(0)".to_string()]).unwrap(), 0xf70);
        assert_eq!(waitcnt_imm(&["lgkmcnt(0)".to_string()]).unwrap(), 0xc07f);
        let (c, clob) = parse_constraints("=&v,=&s,=&s,v,s,v,s,~{vcc},~{scc}").unwrap();
        assert_eq!(c.len(), 7);
        assert_eq!(clob, ["vcc", "scc"]);
        assert!(matches!(parse_constraints("=v,0").unwrap().0[1], Constraint::Tied(0)));
        assert_eq!(opcode_of("v_cmp_lt_u32_e64", 3).map(|o| matches!(o, Opc::VCmp(_))), Some(true));
        assert_eq!(opcode_of("s_addc_u32", 3), Some(Opc::SAddcU32));
        assert_eq!(opcode_of("v_readfirstlane_b32", 2), Some(Opc::VReadfirstlaneB32));
    }
}
