// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Machine IR for gfx11: virtual registers, gfx11 opcodes with their
//! encoding facts, and basic blocks.

use std::fmt::Write;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Class {
    S,
    V,
}

#[derive(Clone, Copy, Debug)]
pub struct VRegInfo {
    pub class: Class,
    pub size: u8,
    pub align: u8,
    /// Precolored physical register (hardware-initialized inputs).
    pub fixed: Option<u16>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VRef {
    pub reg: u32,
    pub off: u8,
    pub n: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Phys {
    S(u16),
    V(u16),
    VccLo,
    ExecLo,
    M0,
    Null,
    Scc,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Opnd {
    R(VRef),
    P(Phys, u8),
    /// A constant. In a 64-bit source an inline integer (-16..=64) is
    /// sign-extended; any other value is the high dword of an f64 source
    /// (low dword zero) and is zero-extended in a 64-bit integer source, as
    /// the hardware reads the literal.
    Imm(u32),
}

impl Opnd {
    pub fn vref(&self) -> Option<VRef> {
        match self {
            Opnd::R(v) => Some(*v),
            _ => None,
        }
    }

    pub fn dwords(&self) -> u8 {
        match self {
            Opnd::R(v) => v.n,
            Opnd::P(_, n) => *n,
            Opnd::Imm(_) => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fmt {
    Sop1,
    Sop2,
    Sopc,
    Sopk,
    Sopp,
    Smem,
    Vop1,
    Vop2,
    /// Native VOP3 (no VOP1/VOP2 short form).
    Vop3,
    /// VOP3 with a scalar destination (carry, div_scale flag) in bits 14:8.
    Vop3b,
    /// Compare, always encoded as VOP3 with an SGPR mask destination.
    Vopc,
    Vop3p,
    /// Two VALU operations issued together (`v_dual_<x> :: v_dual_<y>`).
    Vopd,
    Ds,
    Global,
    Mubuf,
    Pseudo,
}

pub const F_DEF_SCC: u32 = 1 << 0;
pub const F_USE_SCC: u32 = 1 << 1;
pub const F_USE_VCC: u32 = 1 << 2;
pub const F_DEF_EXEC: u32 = 1 << 3;
pub const F_USE_EXEC: u32 = 1 << 4;
pub const F_VMEM_LOAD: u32 = 1 << 5;
pub const F_VMEM_STORE: u32 = 1 << 6;
pub const F_LDS: u32 = 1 << 7;
pub const F_SMEM: u32 = 1 << 8;
pub const F_BRANCH: u32 = 1 << 9;
pub const F_TRANS: u32 = 1 << 10;
pub const F_WMMA: u32 = 1 << 11;
/// The destination is also read (true16 half writes, fmac, d16 loads).
pub const F_TIED: u32 = 1 << 12;
pub const F_64BIT_SRC: u32 = 1 << 13;
pub const F_16BIT_DST: u32 = 1 << 14;
pub const F_16BIT_SRC: u32 = 1 << 15;
pub const F_END: u32 = 1 << 16;
/// The destination is written before the sources are fully read.
pub const F_EARLY: u32 = 1 << 17;
/// A memory fence or barrier: no instruction is scheduled across it.
pub const F_FENCE: u32 = 1 << 18;

pub const M_NEG0: u32 = 1 << 0;
pub const M_NEG1: u32 = 1 << 1;
pub const M_NEG2: u32 = 1 << 2;
/// VOP3 op_sel bits: source 0/1/2 high half, destination high half.
pub const M_OPSEL_SRC0: u32 = 1 << 4;
pub const M_OPSEL_SRC1: u32 = 1 << 5;
pub const M_OPSEL_SRC2: u32 = 1 << 6;
pub const M_OPSEL_DST: u32 = 1 << 7;
/// Global/buffer access through saddr (SGPR base + VGPR 32-bit offset).
pub const M_SADDR: u32 = 1 << 8;
/// `v_fma_mix_f32` source 0/1/2 is f16 (VOP3P op_sel_hi); M_OPSEL_SRCn then
/// picks its high half.
pub const M_MIX_F16_0: u32 = 1 << 9;
/// Global/buffer GLC bit: an atomic returns the value before the operation.
pub const M_GLC: u32 = 1 << 12;
/// DPP16 control fields of `v_mov_b32_dpp`, as bits 31:8 of its second word.
pub fn dpp16_ctrl(dpp_ctrl: u32, bound_ctrl: bool) -> u32 {
    (dpp_ctrl & 0x1ff) | (bound_ctrl as u32) << 11 | 0xf << 16 | 0xf << 20
}

macro_rules! opcodes {
    ($($name:ident = ($fmt:ident, $code:expr, $mn:expr, $flags:expr)),* $(,)?) => {
        /// `Gfx9` holds the gfx9 back end's own opcodes (rows of `isa/gfx950.isa`
        /// with `opc=gfx9`): they have no gfx11 format or opcode (`Fmt::Pseudo`,
        /// which the gfx11 encoders refuse), only their flags and mnemonic.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum Opc { $($name),*, VCmp(u8), Gfx9(crate::target::gfx9::gen_isa::Gfx9Op) }

        impl Opc {
            #[inline(always)]
            pub fn fmt(self) -> Fmt {
                match self { $(Opc::$name => Fmt::$fmt),*, Opc::VCmp(_) => Fmt::Vopc, Opc::Gfx9(_) => Fmt::Pseudo }
            }
            #[inline(always)]
            pub fn code(self) -> u16 {
                match self { $(Opc::$name => $code),*, Opc::VCmp(op) => op as u16, Opc::Gfx9(_) => 0 }
            }
            pub fn name(self) -> String {
                match self { $(Opc::$name => $mn.to_string()),*, Opc::VCmp(op) => vcmp_name(op), Opc::Gfx9(op) => op.mnemonic().to_string() }
            }
            #[inline(always)]
            pub fn flags(self) -> u32 {
                match self {
                    $(Opc::$name => $flags),*,
                    Opc::VCmp(op) => F_USE_EXEC | if (op & 0xf0) == 0x50 { F_64BIT_SRC } else { 0 },
                    Opc::Gfx9(op) => op.flags(),
                }
            }
        }
    };
}

const VALU: u32 = F_USE_EXEC;

// The rows are generated from the gfx11 instruction description
// (`isa/gfx11.isa`) by `platform/tools/isa_tables.py generate`.
include!("gen_opcodes.rs");

/// `v_cmp_class_f32`: whether src0's class is in the mask src1.
pub const VCMP_CLASS_F32: u8 = 0x7e;

pub fn vcmp_name(op: u8) -> String {
    if op == VCMP_CLASS_F32 {
        return "v_cmp_class_f32".into();
    }
    let (ty, base) = match op & 0xf8 {
        0x10 | 0x18 => ("f32", 0x10),
        0x40 => ("i32", 0x40),
        0x48 => ("u32", 0x48),
        0x50 => ("i64", 0x50),
        0x58 => ("u64", 0x58),
        _ => ("?", op),
    };
    let pred = if ty == "f32" {
        ["f", "lt", "eq", "le", "gt", "lg", "ge", "o", "u", "nge", "nlg", "ngt", "nle", "neq", "nlt", "t"]
            .get((op - base) as usize)
            .copied()
            .unwrap_or("?")
    } else {
        ["f", "lt", "eq", "le", "gt", "ne", "ge", "t"]
            .get((op - base) as usize)
            .copied()
            .unwrap_or("?")
    };
    format!("v_cmp_{pred}_{ty}")
}

/// Mnemonic of a VOPD component opcode.
pub fn vopd_name(code: u8) -> &'static str {
    super::gen_isa::vopd_name(code)
}

/// Up to `N` operands, stored inline.
#[derive(Clone, Copy)]
pub struct Opnds<const N: usize> {
    len: u8,
    items: [Opnd; N],
}

impl<const N: usize> Opnds<N> {
    pub fn new() -> Self {
        Opnds { len: 0, items: [Opnd::Imm(0); N] }
    }

    /// Appends `o`; panics when all `N` slots are taken.
    pub fn push(&mut self, o: Opnd) {
        self.items[self.len as usize] = o;
        self.len += 1;
    }
}

impl<const N: usize> Default for Opnds<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> std::ops::Deref for Opnds<N> {
    type Target = [Opnd];
    fn deref(&self) -> &[Opnd] {
        &self.items[..self.len as usize]
    }
}

impl<const N: usize> std::ops::DerefMut for Opnds<N> {
    fn deref_mut(&mut self) -> &mut [Opnd] {
        &mut self.items[..self.len as usize]
    }
}

impl<'a, const N: usize> IntoIterator for &'a Opnds<N> {
    type Item = &'a Opnd;
    type IntoIter = std::slice::Iter<'a, Opnd>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, const N: usize> IntoIterator for &'a mut Opnds<N> {
    type Item = &'a mut Opnd;
    type IntoIter = std::slice::IterMut<'a, Opnd>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<const N: usize> From<&[Opnd]> for Opnds<N> {
    fn from(opnds: &[Opnd]) -> Self {
        let mut out = Self::new();
        for &o in opnds {
            out.push(o);
        }
        out
    }
}

impl<const K: usize, const N: usize> From<[Opnd; K]> for Opnds<N> {
    fn from(opnds: [Opnd; K]) -> Self {
        Self::from(&opnds[..])
    }
}

impl<const N: usize> PartialEq for Opnds<N> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<const N: usize> Eq for Opnds<N> {}

impl<const N: usize> std::fmt::Debug for Opnds<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

pub type Defs = Opnds<2>;
pub type Uses = Opnds<4>;

#[derive(Clone, Debug)]
pub struct MInst {
    pub op: Opc,
    pub defs: Defs,
    pub uses: Uses,
    /// Memory offset, SOPP immediate, or branch target block.
    pub imm: i64,
    pub mods: u32,
    /// The global memory a VMEM access may touch: `alias::ANY`, or the tag
    /// of the `noalias` kernel argument its address is based on.
    pub mem: u16,
}

impl MInst {
    pub fn new(op: Opc, defs: impl Into<Defs>, uses: impl Into<Uses>) -> Self {
        Self {
            op,
            defs: defs.into(),
            uses: uses.into(),
            imm: 0,
            mods: 0,
            mem: super::alias::ANY,
        }
    }

    pub fn with_mem(mut self, mem: u16) -> Self {
        self.mem = mem;
        self
    }

    pub fn with_imm(mut self, imm: i64) -> Self {
        self.imm = imm;
        self
    }

    pub fn with_mods(mut self, mods: u32) -> Self {
        self.mods |= mods;
        self
    }

    pub fn is_branch(&self) -> bool {
        self.op.flags() & F_BRANCH != 0
    }
}

#[derive(Clone, Debug, Default)]
pub struct MBlock {
    pub insts: Vec<MInst>,
    pub loop_depth: u32,
}

#[derive(Clone, Debug, Default)]
pub struct MFunc {
    pub blocks: Vec<MBlock>,
    pub vregs: Vec<VRegInfo>,
    /// The target the function is compiled for, from instruction selection
    /// on (gfx1151 by default).
    pub target: crate::target::Target,
    /// VGPRs the allocation may use when fewer than the target's: set only
    /// by `rir::compile_any_target_limited`, which tests spilling with it.
    pub vgpr_limit: Option<u32>,
}

impl MFunc {
    pub fn successors(&self, block: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let insts = &self.blocks[block].insts;
        let mut falls_through = true;
        if let Some(last) = insts.last() {
            if last.op == Opc::SEndpgm {
                falls_through = false;
            }
            if last.op == Opc::SBranch {
                falls_through = false;
            }
        }
        for inst in insts.iter().rev().take(2) {
            if inst.is_branch() {
                out.push(inst.imm as usize);
            }
        }
        if falls_through && block + 1 < self.blocks.len() {
            out.push(block + 1);
        }
        out.sort_unstable();
        out.dedup();
        out
    }
}

pub fn phys_name(phys: Phys, n: u8) -> String {
    match phys {
        Phys::S(i) if n == 1 => format!("s{i}"),
        Phys::S(i) => format!("s[{}:{}]", i, i + n as u16 - 1),
        Phys::V(i) if n == 1 => format!("v{i}"),
        Phys::V(i) => format!("v[{}:{}]", i, i + n as u16 - 1),
        Phys::VccLo if n == 2 => "vcc".into(),
        Phys::ExecLo if n == 2 => "exec".into(),
        Phys::VccLo => "vcc_lo".into(),
        Phys::ExecLo => "exec_lo".into(),
        Phys::M0 => "m0".into(),
        Phys::Null => "null".into(),
        Phys::Scc => "scc".into(),
    }
}

pub fn opnd_name(func: &MFunc, opnd: &Opnd) -> String {
    match opnd {
        Opnd::R(v) => {
            let info = func.vregs[v.reg as usize];
            let prefix = if info.class == Class::S { "%s" } else { "%v" };
            if v.off == 0 && v.n == info.size {
                format!("{prefix}{}", v.reg)
            } else {
                format!("{prefix}{}[{}:{}]", v.reg, v.off, v.off + v.n - 1)
            }
        }
        Opnd::P(p, n) => phys_name(*p, *n),
        Opnd::Imm(v) => {
            let signed = *v as i32;
            if (-16..=64).contains(&signed) {
                signed.to_string()
            } else {
                format!("0x{v:x}")
            }
        }
    }
}

pub fn inst_text(func: &MFunc, inst: &MInst) -> String {
    if inst.op == Opc::VDual {
        let names = |defs: &[Opnd], uses: &[Opnd]| -> Vec<String> {
            defs.iter().chain(uses).map(|o| opnd_name(func, o)).collect()
        };
        let nx = inst.mods as usize;
        let x = names(&inst.defs[..1], &inst.uses[..nx]).join(", ");
        let y = names(&inst.defs[1..], &inst.uses[nx..]).join(", ");
        let (cx, cy) = (inst.imm as u8, (inst.imm >> 8) as u8);
        return format!("{} {x} :: {} {y}", vopd_name(cx), vopd_name(cy));
    }
    let mut text = inst.op.name();
    let mut parts: Vec<String> = inst.defs.iter().map(|o| opnd_name(func, o)).collect();
    parts.extend(inst.uses.iter().map(|o| opnd_name(func, o)));
    if !parts.is_empty() {
        text.push(' ');
        text.push_str(&parts.join(", "));
    }
    if inst.imm != 0 || inst.is_branch() {
        let _ = write!(text, " imm:{}", inst.imm);
    }
    if inst.mods != 0 {
        let _ = write!(text, " mods:0x{:x}", inst.mods);
    }
    text
}

pub fn dump(func: &MFunc) -> String {
    let mut out = String::new();
    for (index, block) in func.blocks.iter().enumerate() {
        let _ = writeln!(out, "bb{index} (depth {}) -> {:?}", block.loop_depth, func.successors(index));
        for inst in &block.insts {
            let _ = writeln!(out, "    {}", inst_text(func, inst));
        }
    }
    out
}
