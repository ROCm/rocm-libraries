// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Offline cross-check input: `cargo run --release --example isa_xcheck > variants.tsv`.
//! For every table-encoded opcode, a systematic set of operand, modifier and
//! immediate variants of its signature (`gen_isa::signature`), one per line:
//! `index, variant, compare code, defs, uses, imm, mods, table words, hand words`
//! (tab-separated; words as hex or `ERR:<message>`). Operands are `v<r>x<n>`,
//! `s<r>x<n>`, a special register name or `i<hex>`.
//! `platform/tools/isa_xcheck.py` writes them in LLVM syntax, assembles them
//! with llvm-mc and compares the encodings.

use rocke_codeobj_ffi::rir::gen_isa::{self as t, Class, Imm};
use rocke_codeobj_ffi::rir::mir::*;
use rocke_codeobj_ffi::rir::{asm, isa};

/// Inline in some source: integers, f32 constants, and the high dwords of
/// f64 1.0 and -0.5 (inline in f64 sources only).
const INLINE: [u32; 9] = [0, 64, 0xffff_fff0, 0x3f80_0000, 0x3f00_0000, 0xc080_0000, 0x3e22_f983, 0x3ff0_0000, 0xbfe0_0000];
/// Never inline; 0x7ff00000 is the high dword of f64 +inf.
const LITERAL: [u32; 5] = [0x1234_5678, 65, 0xffff_ffef, 0x4049_0fdb, 0x7ff0_0000];

fn v(r: u16, n: u8) -> Opnd {
    Opnd::P(Phys::V(r), n)
}

fn s(r: u16, n: u8) -> Opnd {
    Opnd::P(Phys::S(r), n)
}

fn sp(p: Phys) -> Opnd {
    Opnd::P(p, 1)
}

/// The first choice for an operand of `class` and the variants to try.
fn choices(class: Class, dwords: u8, def: bool) -> Vec<Opnd> {
    let n = dwords;
    let wide = n > 1;
    let high = |r: u16| v(r.min(256 - n as u16) / n as u16 * n as u16, n);
    let sgpr = |r: u16| {
        let a = if n >= 4 { 4 } else { n as u16 };
        s(r / a * a, n)
    };
    let mut out = Vec::new();
    match class {
        Class::V | Class::Vh => out.extend([v(2, n), high(130), high(255)]),
        Class::S if def => {
            // The highest tuple of the SGPR file (s0-s105) at its alignment.
            out.extend([sgpr(4), sgpr(106 - n.max(2) as u16)]);
            if !wide {
                out.extend([sp(Phys::VccLo), sp(Phys::Null), sp(Phys::M0), sp(Phys::ExecLo)]);
            }
        }
        Class::S => out.extend([sgpr(4), sgpr(100)]),
        Class::X => out.push(sp(Phys::ExecLo)),
        Class::Ss | Class::Vs | Class::Vi | Class::Vsh => {
            if class == Class::Ss {
                out.push(sgpr(6));
            } else {
                out.extend([v(3, n), high(131)]);
                out.push(sgpr(6));
            }
            if !wide {
                out.extend([sp(Phys::VccLo), sp(Phys::M0), sp(Phys::ExecLo)]);
            }
            out.extend(INLINE.iter().map(|&k| Opnd::Imm(k)));
            out.extend(LITERAL.iter().map(|&k| Opnd::Imm(k)));
        }
        Class::A => out.push(v(2, n)),
    }
    out
}

fn imms(kind: Imm) -> Vec<i64> {
    match kind {
        Imm::None => vec![0],
        Imm::Simm16 | Imm::Waitcnt | Imm::Sendmsg | Imm::Clause | Imm::Nop => vec![0, 3, 0x3f7, 0x1234],
        Imm::Delay => vec![0x91, 0x1, 0x105],
        Imm::Label => vec![5],
        Imm::Swizzle => vec![0x41f, 0x841f],
        Imm::Soff21 => vec![0, 0x10, 0xfffff, -16],
        Imm::Off16 => vec![0, 16, 65535],
        Imm::Off2x8 => vec![0x0201, 0xff00],
        Imm::Off13s => vec![0, 8, 4095, -8, -4096],
        Imm::Off12 => vec![0, 2, 4095],
        Imm::Vopd => vec![0],
        Imm::Dpp16 => vec![0x00ff_0161, 0x00ff_096f, 0x00ff_08b1, 0x00ff_00e4],
        Imm::Dpp8 => vec![0x00de_54c1, 0x00fa_c688],
    }
}

/// The accepted modifier bits that name a source the instruction has: neg
/// and op_sel bits of absent sources are not instructions.
fn present(op: Opc, allowed: u32) -> u32 {
    let sig = t::signature(op);
    let n = sig.uses.iter().filter(|o| o.role != t::Role::Old).count().min(3) as u32;
    let sources = (1 << n) - 1;
    allowed & !(M_NEG0 | M_NEG1 | M_NEG2 | M_OPSEL_SRC0 | M_OPSEL_SRC1 | M_OPSEL_SRC2)
        | allowed & sources
        | allowed & sources * M_OPSEL_SRC0
}

/// Modifier variants: none, each accepted bit alone, all accepted bits.
fn mods(allowed: u32) -> Vec<u32> {
    let mut out = vec![0];
    for b in 0..16 {
        if allowed & 1 << b != 0 {
            out.push(1 << b);
        }
    }
    if allowed.count_ones() > 1 {
        out.push(allowed);
    }
    out
}

fn token(o: &Opnd) -> String {
    match *o {
        Opnd::P(Phys::V(r), n) => format!("v{r}x{n}"),
        Opnd::P(Phys::S(r), n) => format!("s{r}x{n}"),
        Opnd::P(Phys::VccLo, _) => "vcc_lo".into(),
        Opnd::P(Phys::ExecLo, _) => "exec_lo".into(),
        Opnd::P(Phys::M0, _) => "m0".into(),
        Opnd::P(Phys::Null, _) => "null".into(),
        Opnd::P(Phys::Scc, _) => "scc".into(),
        Opnd::Imm(k) => format!("i{k:x}"),
        Opnd::R(_) => "?".into(),
    }
}

fn words(r: Result<Vec<u32>, rocke_codeobj_ffi::CompileError>) -> String {
    match r {
        Ok(w) => w.iter().map(|x| format!("{x:08x}")).collect::<Vec<_>>().join(","),
        Err(e) => format!("ERR:{}", e.message.replace('\t', " ")),
    }
}

/// Instructions of `op`: the first choice everywhere, then each operand
/// slot, modifier and immediate varied alone. Through saddr a global access
/// takes a 32-bit offset and an SGPR base pair in place of its address pair.
fn variants(op: Opc) -> Vec<MInst> {
    let sig = t::signature(op);
    let defs: Vec<Vec<Opnd>> = sig.defs.iter().map(|o| choices(o.class, o.dwords, true)).collect();
    let uses: Vec<Vec<Opnd>> = sig.uses.iter().map(|o| choices(o.class, o.dwords, false)).collect();
    let pick = |di: Option<(usize, usize)>, ui: Option<(usize, usize)>| -> (Defs, Uses) {
        let mut d = Defs::new();
        for (k, c) in defs.iter().enumerate() {
            d.push(if di.is_some_and(|(s, _)| s == k) { c[di.unwrap().1] } else { c[0] });
        }
        let mut u = Uses::new();
        for (k, c) in uses.iter().enumerate() {
            u.push(if ui.is_some_and(|(s, _)| s == k) { c[ui.unwrap().1] } else { c[0] });
        }
        (d, u)
    };
    let base_imm = imms(sig.imm)[0];
    let mut out = Vec::new();
    let mut push = |d: Defs, u: Uses, imm: i64, m: u32| {
        let mut inst = MInst::new(op, d, u).with_imm(imm);
        inst.mods = m;
        if op.fmt() == Fmt::Global && m & M_SADDR != 0 && t::form(op) != t::Form::Scratch {
            let mut u2 = Uses::new();
            for (k, &o) in inst.uses.iter().enumerate() {
                if k == 0 {
                    u2.push(v(5, 1));
                    u2.push(s(8, 2));
                } else {
                    u2.push(o);
                }
            }
            inst.uses = u2;
        }
        out.push(inst);
    };
    let (d0, u0) = pick(None, None);
    push(d0, u0, base_imm, 0);
    for (k, c) in defs.iter().enumerate() {
        for i in 1..c.len() {
            let (d, u) = pick(Some((k, i)), None);
            push(d, u, base_imm, 0);
        }
    }
    for (k, c) in uses.iter().enumerate() {
        for i in 1..c.len() {
            let (d, u) = pick(None, Some((k, i)));
            push(d, u, base_imm, 0);
        }
    }
    for &m in mods(present(op, sig.mods)).iter().skip(1) {
        let (d, u) = pick(None, None);
        push(d, u, base_imm, m);
    }
    for &imm in imms(sig.imm).iter().skip(1) {
        let (d, u) = pick(None, None);
        push(d, u, imm, 0);
    }
    out
}

/// VOPD pairs: every X and Y component the compiler names, on operands the
/// bank rules allow, then source variants of one pair.
fn vopd_variants() -> Vec<MInst> {
    let named: Vec<u8> = (0..32).filter(|&c| t::vopd_name(c) != "v_dual_?").collect();
    let mut out = Vec::new();
    let half = |code: u8, dst: u16, src0: Opnd, vsrc1: u16, uses: &mut Uses| -> (Opnd, u32) {
        uses.push(src0);
        if code == 8 {
            return (v(dst, 1), 1);
        }
        uses.push(v(vsrc1, 1));
        (v(dst, 1), 2)
    };
    let mut pair = |x: u8, y: u8, xs: Opnd, ys: Opnd| {
        let mut uses = Uses::new();
        let (dx, nx) = half(x, 0, xs, 2, &mut uses);
        let (dy, _) = half(y, 3, ys, 7, &mut uses);
        let mut inst = MInst::new(Opc::VDual, [dx, dy], uses).with_imm(x as i64 | (y as i64) << 8);
        inst.mods = nx;
        out.push(inst);
    };
    for &x in named.iter().filter(|&&c| t::vopd_component(c).is_some_and(|(_, y_only)| !y_only)) {
        for &y in &named {
            pair(x, y, v(1, 1), v(6, 1));
        }
    }
    for (xs, ys) in [
        (s(5, 1), v(6, 1)),
        (Opnd::Imm(0x3f80_0000), Opnd::Imm(64)),
        (Opnd::Imm(0x1234_5678), Opnd::Imm(0x1234_5678)),
        (Opnd::Imm(0x1234_5678), v(6, 1)),
        (v(1, 1), Opnd::Imm(0x4049_0fdb)),
        (sp(Phys::VccLo), v(6, 1)),
    ] {
        pair(3, 4, xs, ys);
        pair(4, 16, xs, ys);
    }
    out
}

fn main() {
    let mut index = 0usize;
    let mut emit = |inst: &MInst, code: Option<u8>| {
        let table = words(isa::encode_words(inst));
        let hand = words(asm::encode_legacy_words(inst));
        let defs: Vec<String> = inst.defs.iter().map(token).collect();
        let uses: Vec<String> = inst.uses.iter().map(token).collect();
        let code = code.map_or("-".to_string(), |c| format!("{c:#04x}"));
        println!(
            "{index}\t{}\t{code}\t{}\t{}\t{}\t{:#x}\t{table}\t{hand}",
            t::variant(inst.op),
            defs.join(","),
            uses.join(","),
            inst.imm,
            inst.mods
        );
        index += 1;
    };
    for &op in t::ALL.iter().filter(|&&op| t::table_encoded(op)) {
        if op == Opc::VDual {
            for inst in vopd_variants() {
                emit(&inst, None);
            }
            continue;
        }
        for inst in variants(op) {
            emit(&inst, None);
        }
    }
    for &(code, _) in &t::VCMP {
        for inst in variants(Opc::VCmp(code)) {
            emit(&inst, Some(code));
        }
    }
}
