// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Instruction selection: optimized AST to gfx11 machine IR on virtual
//! registers. Uniform values use SALU and SGPRs; divergent ones VALU and
//! VGPRs. Structured control flow becomes EXEC-masked regions or scalar
//! branches.

use super::ast::{Kernel, Op, OpKind, Region, Scalar, Type, ValueId};
use super::fx::{FxHashMap, FxHashSet};
use super::mir::*;
use super::opt::Facts;
use crate::error::CompileError;
use crate::target::{gfx9::isel_hooks, Target, TargetInfo};

#[path = "burs.rs"]
mod burs;
#[path = "gen_burs.rs"]
#[rustfmt::skip]
mod gen_burs;

fn unsupported(message: impl Into<String>) -> CompileError {
    CompileError::new("ROCKE_DIRECT_UNSUPPORTED", message)
}

/// IEEE binary16 bits of `x`, rounding to nearest even.
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    let (m, shift) = if e <= 0 {
        if e < -10 {
            return sign;
        }
        (mant | 0x80_0000, (14 - e) as u32)
    } else {
        (mant, 13)
    };
    let mut h = (m >> shift) as u16;
    let rem = m & ((1 << shift) - 1);
    let half = 1 << (shift - 1);
    if rem > half || (rem == half && h & 1 != 0) {
        h += 1;
    }
    if e <= 0 {
        sign | h
    } else {
        sign | ((e as u16) << 10) + h
    }
}

/// bfloat16 bits of `x` as LLVM's `fptrunc float to bfloat` gives them on
/// gfx11: round to nearest even; a NaN keeps its sign and payload and is quieted.
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits | 0x40_0000) >> 16) as u16;
    }
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

pub fn is_inline_int(v: u32) -> bool {
    (-16..=64).contains(&(v as i32))
}

pub fn is_inline_f32(v: u32) -> bool {
    matches!(
        v,
        0x3f00_0000 | 0xbf00_0000 | 0x3f80_0000 | 0xbf80_0000 | 0x4000_0000 | 0xc000_0000 | 0x4080_0000 | 0xc080_0000 | 0x3e22_f983
    )
}

/// Whether `hi` is the high dword of an f64 inline constant (±0.5, ±1, ±2,
/// ±4, whose low dwords are zero).
pub fn is_inline_f64(hi: u32) -> bool {
    matches!(
        hi,
        0x3fe0_0000 | 0xbfe0_0000 | 0x3ff0_0000 | 0xbff0_0000 | 0x4000_0000 | 0xc000_0000 | 0x4010_0000 | 0xc010_0000
    )
}

#[derive(Clone, Copy, Debug)]
struct SCmp {
    op: Opc,
    a: Opnd,
    b: Opnd,
}

#[derive(Clone, Copy, Debug)]
enum Val {
    /// 32-bit scalar: immediate or register.
    W(Opnd),
    /// 64-bit scalar as low and high halves.
    D(Opnd, Opnd),
    /// Register tuple: vectors and buffer descriptors.
    T(VRef),
    /// f16 in one half of a 32-bit operand.
    H(Opnd, bool),
    /// Uniform compare, materialized at each use.
    C(SCmp),
    /// Lane mask (divergent bool; uniform bools as 0 or -1).
    M(Opnd),
    /// LDS allocation base offset.
    L(u32),
    /// i8 in byte `k` of a 32-bit operand; the other bytes hold anything.
    B(Opnd, u8),
}

pub struct IselResult {
    pub func: MFunc,
    pub kernarg_size: u32,
    pub lds_size: u32,
    pub wg_ids: [bool; 3],
}

struct Isel<'a> {
    /// The BURS prototype's plan (`ROCKE_DIRECT_DEBUG=burs`), else `None`.
    burs: Option<burs::Plan>,
    k: &'a Kernel,
    facts: &'a Facts,
    uniform: Vec<bool>,
    f: MFunc,
    cur: usize,
    vals: Vec<Option<Val>>,
    lds: &'a FxHashMap<ValueId, u32>,
    depth: u32,
    single_wave: bool,
    /// The facts of the target selected for: wave size, lane-mask width.
    t: &'static TargetInfo,
    kernarg_ptr: VRef,
    tid: VRef,
    bid: [Option<VRef>; 3],
    zero_v: Option<VRef>,
    uses: Vec<u32>,
    gathers: FxHashSet<ValueId>,
    chains: Chains,
    deferred: FxHashMap<ValueId, Deferred>,
    /// `analysis::canonical_f32`, computed at the first `arith.fmax`.
    canonical: Vec<bool>,
    /// f64 values produced by f64 arithmetic, hence never signaling NaNs.
    canon64: FxHashSet<ValueId>,
    /// SGPR holding `v_permlanex16_b32`'s low lane select, set at the entry.
    lane_sel: Option<VRef>,
    /// `vector.splat` results and the scalar they repeat: element-wise ops
    /// read the scalar, and the copies die unless something reads the tuple.
    splats: FxHashMap<ValueId, Opnd>,
    /// `arith.fneg` results every use of which is an f32 FMA operand: they
    /// become source negation modifiers.
    fneg_fma: FxHashMap<ValueId, ValueId>,
    /// i32 `arith.or`s that pack two f16 bit patterns, and their halves.
    packs: FxHashMap<ValueId, [PackPart; 2]>,
    /// Vote predicates `select(c, k1, k2)` of constants: c and whether k1 / k2 are nonzero.
    vote_sel: FxHashMap<ValueId, (ValueId, bool, bool)>,
    /// f16 `vector.trunc_f32_to` results of f32 vector products, and the factors.
    trunc_mul: FxHashMap<ValueId, (ValueId, ValueId)>,
    /// `alias::value_tags`, by which VMEM instructions get `MInst::mem`;
    /// empty (every access `alias::ANY`) with `ROCKE_DIRECT_DEBUG=noalias`.
    mem_tags: Vec<u16>,
}

/// One half of an f16 pair packed into an i32: an f16 (or i16) value, the
/// f16 conversion of an f32, or that of an f32 product.
#[derive(Clone, Copy)]
enum PackPart {
    Half(ValueId),
    Cvt(ValueId),
    Mix(ValueId, ValueId),
}

/// A 16-bit gather load waiting for the insert that receives it.
enum Deferred {
    /// Global address operands, modifiers, offset, memory tag.
    Global(Uses, u32, i64, u16),
    /// LDS address and offset.
    Lds(Opnd, i64),
    /// Buffer operands (voffset, rsrc, soffset), offset, memory tag.
    Buffer(Uses, i64, u16),
}

fn r(v: VRef) -> Opnd {
    Opnd::R(v)
}

fn sub(v: VRef, off: u8, n: u8) -> VRef {
    VRef {
        reg: v.reg,
        off: v.off + off,
        n,
    }
}

const NULL: Opnd = Opnd::P(Phys::Null, 1);

/// `v_permlanex16_b32` lane selects that make every lane read lane `L ^ 16`.
const PERMLANEX16_SEL: (u32, u32) = (0x7654_3210, 0xfedc_ba98);

impl<'a> Isel<'a> {
    // ----- infrastructure -----

    fn vreg(&mut self, class: Class, size: u8, align: u8) -> VRef {
        let reg = self.f.vregs.len() as u32;
        self.f.vregs.push(VRegInfo {
            class,
            size,
            align,
            fixed: None,
        });
        VRef { reg, off: 0, n: size }
    }

    fn sreg(&mut self, size: u8) -> VRef {
        let align = match size {
            1 => 1,
            2 => 2,
            _ => 4,
        };
        self.vreg(Class::S, size, align)
    }

    /// An SGPR lane mask: a compare result, a carry, a saved EXEC.
    fn mreg(&mut self) -> VRef {
        self.sreg(self.t.lane_mask_dwords)
    }

    /// EXEC and VCC as lane masks.
    fn exec(&self) -> Opnd {
        Opnd::P(Phys::ExecLo, self.t.lane_mask_dwords)
    }

    fn vcc(&self) -> Opnd {
        Opnd::P(Phys::VccLo, self.t.lane_mask_dwords)
    }

    /// The wave32 lane-mask opcode `op` at the target's lane-mask width.
    fn lm(&self, op: Opc) -> Opc {
        if self.t.lane_mask_dwords == 2 {
            isel_hooks::wide(op)
        } else {
            op
        }
    }

    /// Lane masks `a op b` in a new lane mask.
    fn m_bin(&mut self, op: Opc, a: Opnd, b: Opnd) -> Opnd {
        let dst = self.mreg();
        self.salu(self.lm(op), [r(dst)], [a, b]);
        r(dst)
    }

    /// Refuses `what` on gfx950, which has no lowering of it yet.
    fn gfx11_only(&self, what: &str) -> Result<(), CompileError> {
        match self.t.target {
            Target::Gfx950 => Err(unsupported(format!("{what} on gfx950"))),
            Target::Gfx1151 => Ok(()),
        }
    }

    fn vgpr(&mut self, size: u8) -> VRef {
        self.vreg(Class::V, size, 1)
    }

    fn emit(&mut self, inst: MInst) {
        self.f.blocks[self.cur].insts.push(inst);
    }

    fn new_block(&mut self) -> usize {
        self.f.blocks.push(MBlock {
            insts: Vec::new(),
            loop_depth: self.depth,
        });
        self.cur = self.f.blocks.len() - 1;
        self.cur
    }

    fn class_of(&self, o: Opnd) -> Option<Class> {
        match o {
            Opnd::R(v) => Some(self.f.vregs[v.reg as usize].class),
            Opnd::P(Phys::V(_), _) => Some(Class::V),
            Opnd::P(_, _) => Some(Class::S),
            Opnd::Imm(_) => None,
        }
    }

    fn is_v(&self, o: Opnd) -> bool {
        self.class_of(o) == Some(Class::V)
    }

    fn set(&mut self, v: ValueId, val: Val) {
        self.vals[v.0 as usize] = Some(val);
    }

    fn get(&self, v: ValueId) -> Result<Val, CompileError> {
        self.vals[v.0 as usize].ok_or_else(|| unsupported(format!("value {} used before selection", self.k.name_of(v))))
    }

    fn uni(&self, v: ValueId) -> bool {
        self.uniform[v.0 as usize]
    }

    fn copy(&mut self, dst: VRef, src: Opnd) {
        self.emit(MInst::new(Opc::Copy, [r(dst)], [src]));
    }

    /// Moves an operand into a fresh VGPR unless it already is one.
    fn to_v(&mut self, o: Opnd) -> Opnd {
        if self.is_v(o) {
            return o;
        }
        let dst = self.vgpr(1);
        self.copy(dst, o);
        r(dst)
    }

    /// The first active lane's value of `o`, or `o` itself when it is uniform.
    fn first_lane(&mut self, o: Opnd) -> Opnd {
        if !self.is_v(o) {
            return o;
        }
        let dst = self.sreg(1);
        self.emit(MInst::new(Opc::VReadfirstlaneB32, [r(dst)], [o]));
        r(dst)
    }

    /// Moves an operand into a fresh SGPR unless it already is one.
    fn to_s(&mut self, o: Opnd) -> Opnd {
        match self.class_of(o) {
            Some(Class::S) => o,
            Some(Class::V) => {
                let dst = self.sreg(1);
                self.emit(MInst::new(Opc::VReadfirstlaneB32, [r(dst)], [o]));
                r(dst)
            }
            None => {
                let dst = self.sreg(1);
                self.copy(dst, o);
                r(dst)
            }
        }
    }

    // ----- instruction builders with operand legalization -----

    fn valu(&mut self, op: Opc, defs: impl Into<Defs>, srcs: impl Into<Uses>, mods: u32) {
        let srcs = self.legalize_valu(op, srcs.into());
        self.emit(MInst::new(op, defs, srcs).with_mods(mods));
    }

    fn legalize_valu(&mut self, op: Opc, mut srcs: Uses) -> Uses {
        let flags = op.flags();
        let wide = flags & F_64BIT_SRC != 0;
        let half = flags & F_16BIT_SRC != 0;
        // An f64 source reads an immediate as its high dword, v_ldexp_f64's
        // exponent as a 32-bit value (`gen_isa::Form::F64`, `F64I32`).
        let form = if wide { super::gen_isa::form(op) } else { super::gen_isa::Form::Plain };
        let f64_src = |i: usize| match form {
            super::gen_isa::Form::F64 => Some(true),
            super::gen_isa::Form::F64I32 => Some(i == 0),
            _ => None,
        };
        let mut scalars = Uses::new();
        let mut literal: Option<u32> = None;
        for i in 0..srcs.len() {
            match srcs[i] {
                Opnd::Imm(v) if f64_src(i).is_some() => {
                    let is_f64 = f64_src(i) == Some(true);
                    let inline = is_inline_int(v) || if is_f64 { is_inline_f64(v) } else { is_inline_f32(v) };
                    if !inline {
                        if literal.is_none_or(|l| l == v) {
                            literal = Some(v);
                        } else if is_f64 {
                            srcs[i] = r(self.pair_v(Opnd::Imm(0), Opnd::Imm(v)));
                        } else {
                            srcs[i] = self.to_v(srcs[i]);
                        }
                    }
                }
                Opnd::Imm(v) => {
                    let inline = is_inline_int(v) || (!wide && is_inline_f32(v));
                    if half || (wide && !inline) {
                        srcs[i] = self.to_v(srcs[i]);
                    } else if !inline {
                        if literal.is_none_or(|l| l == v) {
                            literal = Some(v);
                        } else {
                            srcs[i] = self.to_v(srcs[i]);
                        }
                    }
                }
                o if self.class_of(o) == Some(Class::S) => {
                    if !scalars.contains(&o) {
                        scalars.push(o);
                    }
                }
                _ => {}
            }
        }
        let budget = 2 - literal.is_some() as usize;
        if scalars.len() > budget {
            let extra = Uses::from(&scalars[budget..]);
            for &o in extra.iter() {
                let n = o.dwords();
                let moved = if n == 1 {
                    self.to_v(o)
                } else {
                    let dst = self.vgpr(n);
                    self.copy(dst, o);
                    r(dst)
                };
                for s in srcs.iter_mut() {
                    if *s == o {
                        *s = moved;
                    }
                }
            }
        }
        srcs
    }

    fn salu(&mut self, op: Opc, defs: impl Into<Defs>, srcs: impl Into<Uses>) {
        let mut srcs: Uses = srcs.into();
        let wide = matches!(
            op,
            Opc::SLshlB64
                | Opc::SLshrB64
                | Opc::SAshrI64
                | Opc::SAndB64
                | Opc::SOrB64
                | Opc::SXorB64
                | Opc::SCselectB64
                | Opc::SCmpEqU64
                | Opc::SCmpLgU64
                | Opc::SMovB64
                | Opc::SAndNot1B64
                | Opc::SOrNot1B64
                | Opc::SNotB64
                | Opc::SAndSaveexecB64
        );
        let mut literal: Option<u32> = None;
        for (i, s) in srcs.iter_mut().enumerate() {
            if let Opnd::Imm(v) = *s {
                if is_inline_int(v) || (!wide && is_inline_f32(v)) {
                    continue;
                }
                let shift_amount = i == 1 && matches!(op, Opc::SLshlB64 | Opc::SLshrB64 | Opc::SAshrI64);
                if (wide && !shift_amount) || literal.is_some_and(|l| l != v) {
                    let n = if wide && !shift_amount { 2 } else { 1 };
                    let dst = self.sreg(n);
                    self.f.blocks[self.cur].insts.push(MInst::new(Opc::Copy, [r(sub(dst, 0, 1))], [Opnd::Imm(v)]));
                    if n == 2 {
                        self.f.blocks[self.cur].insts.push(MInst::new(Opc::Copy, [r(sub(dst, 1, 1))], [Opnd::Imm(0)]));
                    }
                    *s = r(dst);
                } else {
                    literal = Some(v);
                }
            }
        }
        self.emit(MInst::new(op, defs, srcs));
    }

    // ----- value accessors -----

    fn w(&mut self, v: ValueId) -> Result<Opnd, CompileError> {
        match self.get(v)? {
            Val::W(o) => Ok(o),
            Val::H(o, false) => Ok(o),
            Val::H(o, true) => {
                let dst = self.vgpr(1);
                self.valu(Opc::VLshrrevB32, [r(dst)], [Opnd::Imm(16), o], 0);
                Ok(r(dst))
            }
            Val::D(lo, _) => Ok(lo),
            Val::T(t) if t.n == 1 => Ok(r(t)),
            Val::B(o, 0) => Ok(o),
            Val::B(o, k) => Ok(match o {
                Opnd::Imm(c) => Opnd::Imm(c >> (8 * k)),
                _ if self.is_v(o) => self.v_bin(Opc::VLshrrevB32, Opnd::Imm(8 * k as u32), o),
                _ => self.s_bin(Opc::SLshrB32, o, Opnd::Imm(8 * k as u32)),
            }),
            Val::C(_) | Val::M(_) => {
                // An i1 used as a 32-bit value: 0 or 1.
                let mask = self.mask(v)?;
                Ok(self.bool_to_int(mask, self.uni(v)))
            }
            other => Err(unsupported(format!("value {} ({other:?}) is not a 32-bit scalar", self.k.name_of(v)))),
        }
    }

    fn d(&mut self, v: ValueId) -> Result<(Opnd, Opnd), CompileError> {
        match self.get(v)? {
            Val::D(lo, hi) => Ok((lo, hi)),
            Val::T(t) if t.n == 2 => Ok((r(sub(t, 0, 1)), r(sub(t, 1, 1)))),
            other => Err(unsupported(format!("value {} ({other:?}) is not 64-bit", self.k.name_of(v)))),
        }
    }

    fn tuple(&mut self, v: ValueId) -> Result<VRef, CompileError> {
        match self.get(v)? {
            Val::T(t) => Ok(t),
            Val::W(o) => {
                if let Opnd::R(vr) = o {
                    if self.is_v(o) {
                        return Ok(vr);
                    }
                }
                let dst = self.vgpr(1);
                self.copy(dst, o);
                Ok(dst)
            }
            Val::H(o, hi) => {
                let lo = if hi { self.w(v)? } else { o };
                if let Opnd::R(vr) = lo {
                    if self.is_v(lo) {
                        return Ok(vr);
                    }
                }
                let dst = self.vgpr(1);
                self.copy(dst, lo);
                Ok(dst)
            }
            Val::D(lo, hi) => {
                let pair = self.pair_v(lo, hi);
                Ok(pair)
            }
            Val::B(..) => {
                let lo = self.w(v)?;
                if let Opnd::R(vr) = lo {
                    if self.is_v(lo) {
                        return Ok(vr);
                    }
                }
                let dst = self.vgpr(1);
                self.copy(dst, lo);
                Ok(dst)
            }
            other => Err(unsupported(format!("value {} ({other:?}) is not a vector", self.k.name_of(v)))),
        }
    }

    /// The halves as one contiguous register pair of `class`.
    fn pair(&mut self, lo: Opnd, hi: Opnd, class: Class) -> VRef {
        if let (Opnd::R(a), Opnd::R(b)) = (lo, hi) {
            let info = self.f.vregs[a.reg as usize];
            if a.reg == b.reg && b.off == a.off + 1 && info.class == class && (class == Class::V || a.off % 2 == 0) {
                return VRef { reg: a.reg, off: a.off, n: 2 };
            }
        }
        let dst = if class == Class::S { self.sreg(2) } else { self.vgpr(2) };
        self.copy(sub(dst, 0, 1), lo);
        self.copy(sub(dst, 1, 1), hi);
        dst
    }

    fn pair_v(&mut self, lo: Opnd, hi: Opnd) -> VRef {
        self.pair(lo, hi, Class::V)
    }

    fn pair_s(&mut self, lo: Opnd, hi: Opnd) -> VRef {
        self.pair(lo, hi, Class::S)
    }

    fn wide_operand(&mut self, lo: Opnd, hi: Opnd) -> Opnd {
        if let (Opnd::Imm(l), Opnd::Imm(h)) = (lo, hi) {
            let value = ((h as u64) << 32 | l as u64) as i64;
            if (-16..=64).contains(&value) {
                return Opnd::Imm(value as u32);
            }
        }
        let class = if self.is_v(lo) || self.is_v(hi) { Class::V } else { Class::S };
        r(self.pair(lo, hi, class))
    }

    /// A source of an opcode of form `f64` / `f64_i32`, whose immediates
    /// `legalize_valu` keeps: a constant with a zero low dword is its high
    /// dword (`Opnd::Imm`) unless that reads as an inline integer.
    fn f64_operand(&mut self, lo: Opnd, hi: Opnd) -> Opnd {
        match (lo, hi) {
            (Opnd::Imm(0), Opnd::Imm(h)) if !is_inline_int(h) => Opnd::Imm(h),
            _ => self.wide_operand(lo, hi),
        }
    }

    // ----- booleans -----

    fn scc(&mut self, v: ValueId) -> Result<(), CompileError> {
        match self.get(v)? {
            Val::C(c) => self.salu(c.op, [], [c.a, c.b]),
            Val::M(m) => self.salu(self.lm(Opc::SCmpLgU32), [], [m, Opnd::Imm(0)]),
            Val::W(o) => self.salu(Opc::SCmpLgU32, [], [o, Opnd::Imm(0)]),
            other => return Err(unsupported(format!("{other:?} is not a condition"))),
        }
        Ok(())
    }

    fn mask(&mut self, v: ValueId) -> Result<Opnd, CompileError> {
        match self.get(v)? {
            Val::M(m) => Ok(m),
            Val::C(c) => {
                self.salu(c.op, [], [c.a, c.b]);
                let dst = self.mreg();
                self.salu(self.lm(Opc::SCselectB32), [r(dst)], [Opnd::Imm(u32::MAX), Opnd::Imm(0)]);
                Ok(r(dst))
            }
            Val::W(Opnd::Imm(c)) => Ok(Opnd::Imm(if c & 1 != 0 { u32::MAX } else { 0 })),
            other => Err(unsupported(format!("{other:?} is not a boolean"))),
        }
    }

    fn bool_to_int(&mut self, mask: Opnd, uniform: bool) -> Opnd {
        if let Opnd::Imm(m) = mask {
            return Opnd::Imm((m != 0) as u32);
        }
        if uniform {
            let dst = self.sreg(1);
            self.salu(self.lm(Opc::SCmpLgU32), [], [mask, Opnd::Imm(0)]);
            self.salu(Opc::SCselectB32, [r(dst)], [Opnd::Imm(1), Opnd::Imm(0)]);
            r(dst)
        } else {
            let dst = self.vgpr(1);
            self.valu(Opc::VCndmaskB32, [r(dst)], [Opnd::Imm(0), Opnd::Imm(1), mask], 0);
            r(dst)
        }
    }

    // ----- integer helpers -----

    fn range_lt(&self, v: ValueId, bound: i64) -> bool {
        self.facts.range_of(v).is_some_and(|r| r.lo >= 0 && r.hi < bound)
    }

    fn s_bin(&mut self, op: Opc, a: Opnd, b: Opnd) -> Opnd {
        let dst = self.sreg(1);
        self.salu(op, [r(dst)], [a, b]);
        r(dst)
    }

    fn v_bin(&mut self, op: Opc, a: Opnd, b: Opnd) -> Opnd {
        let dst = self.vgpr(1);
        self.valu(op, [r(dst)], [a, b], 0);
        r(dst)
    }

    fn v3(&mut self, op: Opc, a: Opnd, b: Opnd, c: Opnd, mods: u32) -> Opnd {
        let dst = self.vgpr(1);
        self.valu(op, [r(dst)], [a, b, c], mods);
        r(dst)
    }

    /// A VALU op with a register-pair result.
    fn v64(&mut self, op: Opc, srcs: impl Into<Uses>, mods: u32) -> VRef {
        let dst = self.vgpr(2);
        self.valu(op, [r(dst)], srcs, mods);
        dst
    }

    /// 32-bit integer binary op on operands (uniform -> SALU).
    fn int_bin(&mut self, kind: OpKind, a: Opnd, b: Opnd, uniform: bool, small: bool) -> Result<Opnd, CompileError> {
        if uniform {
            let op = match kind {
                OpKind::Add => Opc::SAddI32,
                OpKind::Sub => Opc::SSubI32,
                OpKind::Mul => Opc::SMulI32,
                OpKind::And => Opc::SAndB32,
                OpKind::Or => Opc::SOrB32,
                OpKind::Xor => Opc::SXorB32,
                OpKind::Shl => Opc::SLshlB32,
                OpKind::Lshr => Opc::SLshrB32,
                OpKind::Ashr => Opc::SAshrI32,
                OpKind::Smin => Opc::SMinI32,
                OpKind::Smax => Opc::SMaxI32,
                OpKind::UmulHi => Opc::SMulHiU32,
                OpKind::SmulHi => Opc::SMulHiI32,
                other => return Err(unsupported(format!("uniform {other:?}"))),
            };
            return Ok(self.s_bin(op, a, b));
        }
        // VOP2 wants the VGPR in src1; reversed shift forms take (amount, value).
        let (op, x, y) = match kind {
            OpKind::Add => (Opc::VAddNcU32, a, b),
            OpKind::Sub => {
                if !self.is_v(b) && self.is_v(a) {
                    (Opc::VSubrevNcU32, b, a)
                } else {
                    (Opc::VSubNcU32, a, b)
                }
            }
            OpKind::Mul if small => (Opc::VMulU32U24, a, b),
            OpKind::Mul => (Opc::VMulLoU32, a, b),
            OpKind::And => (Opc::VAndB32, a, b),
            OpKind::Or => (Opc::VOrB32, a, b),
            OpKind::Xor => (Opc::VXorB32, a, b),
            OpKind::Shl => (Opc::VLshlrevB32, b, a),
            OpKind::Lshr => (Opc::VLshrrevB32, b, a),
            OpKind::Ashr => (Opc::VAshrrevI32, b, a),
            OpKind::Smin => (Opc::VMinI32, a, b),
            OpKind::Smax => (Opc::VMaxI32, a, b),
            OpKind::UmulHi if small => (Opc::VMulHiU32U24, a, b),
            OpKind::UmulHi => (Opc::VMulHiU32, a, b),
            OpKind::SmulHi => (Opc::VMulHiI32, a, b),
            other => return Err(unsupported(format!("divergent {other:?}"))),
        };
        let commutative = matches!(
            op,
            Opc::VAddNcU32 | Opc::VMulU32U24 | Opc::VAndB32 | Opc::VOrB32 | Opc::VXorB32 | Opc::VMinI32 | Opc::VMaxI32
        );
        let (x, y) = if commutative && !self.is_v(y) && self.is_v(x) { (y, x) } else { (x, y) };
        Ok(self.v_bin(op, x, y))
    }

    fn add64(&mut self, a: (Opnd, Opnd), b: (Opnd, Opnd), uniform: bool, subtract: bool) -> (Opnd, Opnd) {
        if b == (Opnd::Imm(0), Opnd::Imm(0)) {
            return a;
        }
        if uniform {
            let dst = self.sreg(2);
            let (lo_op, hi_op) = if subtract { (Opc::SSubU32, Opc::SSubbU32) } else { (Opc::SAddU32, Opc::SAddcU32) };
            self.salu(lo_op, [r(sub(dst, 0, 1))], [a.0, b.0]);
            self.salu(hi_op, [r(sub(dst, 1, 1))], [a.1, b.1]);
            return (r(sub(dst, 0, 1)), r(sub(dst, 1, 1)));
        }
        let dst = self.vgpr(2);
        let carry = self.mreg();
        let (lo_op, hi_op) = if subtract { (Opc::VSubCoU32, Opc::VSubCoCiU32) } else { (Opc::VAddCoU32, Opc::VAddCoCiU32) };
        self.valu(lo_op, [r(sub(dst, 0, 1)), r(carry)], [a.0, b.0], 0);
        self.valu(hi_op, [r(sub(dst, 1, 1)), NULL], [a.1, b.1, r(carry)], 0);
        (r(sub(dst, 0, 1)), r(sub(dst, 1, 1)))
    }

    fn mul64(&mut self, a: (Opnd, Opnd), b: (Opnd, Opnd), uniform: bool) -> (Opnd, Opnd) {
        let zero = Opnd::Imm(0);
        if uniform {
            let dst = self.sreg(2);
            self.salu(Opc::SMulI32, [r(sub(dst, 0, 1))], [a.0, b.0]);
            let hi = r(sub(dst, 1, 1));
            if a.1 == zero && b.1 == zero {
                self.salu(Opc::SMulHiU32, [hi], [a.0, b.0]);
            } else {
                let mut acc = self.s_bin(Opc::SMulHiU32, a.0, b.0);
                if b.1 != zero {
                    let c = self.s_bin(Opc::SMulI32, a.0, b.1);
                    acc = self.s_bin(Opc::SAddI32, acc, c);
                }
                if a.1 != zero {
                    let c = self.s_bin(Opc::SMulI32, a.1, b.0);
                    acc = self.s_bin(Opc::SAddI32, acc, c);
                }
                self.copy(sub(dst, 1, 1), acc);
            }
            return (r(sub(dst, 0, 1)), hi);
        }
        let dst = self.vgpr(2);
        self.valu(Opc::VMadU64U32, [r(dst), NULL], [a.0, b.0, Opnd::Imm(0)], 0);
        if a.1 == zero && b.1 == zero {
            return (r(sub(dst, 0, 1)), r(sub(dst, 1, 1)));
        }
        let mut terms = vec![r(sub(dst, 1, 1))];
        if b.1 != zero {
            terms.push(self.v_bin(Opc::VMulLoU32, a.0, b.1));
        }
        if a.1 != zero {
            terms.push(self.v_bin(Opc::VMulLoU32, a.1, b.0));
        }
        let hi = if terms.len() == 3 {
            self.v3(Opc::VAdd3U32, terms[0], terms[1], terms[2], 0)
        } else {
            self.v_bin(Opc::VAddNcU32, terms[1], terms[0])
        };
        (r(sub(dst, 0, 1)), hi)
    }

    /// `x << k` for a 64-bit value and constant `k`.
    fn shl64(&mut self, v: ValueId, x: (Opnd, Opnd), k: u32, uniform: bool, result: ValueId) -> (Opnd, Opnd) {
        if k == 0 {
            return x;
        }
        if k >= 32 {
            let lo = Opnd::Imm(0);
            let hi = if k == 32 {
                x.0
            } else if uniform {
                self.s_bin(Opc::SLshlB32, x.0, Opnd::Imm(k - 32))
            } else {
                self.v_bin(Opc::VLshlrevB32, Opnd::Imm(k - 32), x.0)
            };
            return (lo, hi);
        }
        let hi_zero = x.1 == Opnd::Imm(0);
        // Result known below 2^32: the high word is zero.
        let fits = self.facts.range_of(result).is_some_and(|r| r.lo >= 0 && r.hi < (1i64 << 32));
        let _ = v;
        if hi_zero && fits {
            let lo = if uniform {
                self.s_bin(Opc::SLshlB32, x.0, Opnd::Imm(k))
            } else {
                self.v_bin(Opc::VLshlrevB32, Opnd::Imm(k), x.0)
            };
            return (lo, Opnd::Imm(0));
        }
        if uniform {
            let src = self.pair_s(x.0, x.1);
            let dst = self.sreg(2);
            self.salu(Opc::SLshlB64, [r(dst)], [r(src), Opnd::Imm(k)]);
            return (r(sub(dst, 0, 1)), r(sub(dst, 1, 1)));
        }
        if hi_zero {
            let lo = self.v_bin(Opc::VLshlrevB32, Opnd::Imm(k), x.0);
            let hi = self.v_bin(Opc::VLshrrevB32, Opnd::Imm(32 - k), x.0);
            return (lo, hi);
        }
        let src = self.wide_operand(x.0, x.1);
        let dst = self.vgpr(2);
        self.valu(Opc::VLshlrevB64, [r(dst)], [Opnd::Imm(k), src], 0);
        (r(sub(dst, 0, 1)), r(sub(dst, 1, 1)))
    }

    /// Signed 32-bit division by a runtime value (LLVM's expansion).
    fn sdiv32(&mut self, x: Opnd, y: Opnd, x_nonneg: bool, y_nonneg: bool) -> Opnd {
        let (ax, sx) = if x_nonneg {
            (x, None)
        } else {
            let s = self.v_bin(Opc::VAshrrevI32, Opnd::Imm(31), x);
            let t = self.v_bin(Opc::VAddNcU32, x, s);
            (self.v_bin(Opc::VXorB32, t, s), Some(s))
        };
        let (ay, sy) = if y_nonneg {
            (y, None)
        } else {
            let s = self.v_bin(Opc::VAshrrevI32, Opnd::Imm(31), y);
            let t = self.v_bin(Opc::VAddNcU32, y, s);
            (self.v_bin(Opc::VXorB32, t, s), Some(s))
        };
        let q = self.udiv32(ax, ay);
        let sign = match (sx, sy) {
            (None, None) => return q,
            (Some(s), None) | (None, Some(s)) => s,
            (Some(a), Some(b)) => self.v_bin(Opc::VXorB32, a, b),
        };
        let t = self.v_bin(Opc::VXorB32, q, sign);
        self.v_bin(Opc::VSubNcU32, t, sign)
    }

    fn udiv32(&mut self, n: Opnd, d: Opnd) -> Opnd {
        let fd = self.vgpr(1);
        self.valu(Opc::VCvtF32U32, [r(fd)], [d], 0);
        let rc = self.vgpr(1);
        self.valu(Opc::VRcpIflagF32, [r(rc)], [r(fd)], 0);
        let scaled = self.v_bin(Opc::VMulF32, Opnd::Imm(0x4f7f_fffe), r(rc));
        let z0 = self.vgpr(1);
        self.valu(Opc::VCvtU32F32, [r(z0)], [scaled], 0);
        let negd = self.v_bin(Opc::VSubNcU32, Opnd::Imm(0), d);
        let t = self.v_bin(Opc::VMulLoU32, negd, r(z0));
        let zz = self.v_bin(Opc::VMulHiU32, r(z0), t);
        let z = self.v_bin(Opc::VAddNcU32, r(z0), zz);
        let mut q = self.v_bin(Opc::VMulHiU32, n, z);
        let qd = self.v_bin(Opc::VMulLoU32, q, d);
        let mut rem = self.v_bin(Opc::VSubNcU32, n, qd);
        for step in 0..2 {
            let ge = self.mreg();
            self.valu(Opc::VCmp(0x4e), [r(ge)], [rem, d], 0);
            let q1 = self.v_bin(Opc::VAddNcU32, Opnd::Imm(1), q);
            q = self.v3(Opc::VCndmaskB32, q, q1, r(ge), 0);
            if step == 0 {
                let r1 = self.v_bin(Opc::VSubNcU32, rem, d);
                rem = self.v3(Opc::VCndmaskB32, rem, r1, r(ge), 0);
            }
        }
        q
    }

    // ----- lowering -----

    fn region(&mut self, region: &Region) -> Result<(), CompileError> {
        let mut i = 0;
        while i < region.ops.len() {
            let run = region.ops[i..].iter().take_while(|op| op.kind == OpKind::KernArg).count();
            if run > 1 {
                self.kernargs(&region.ops[i..i + run]);
                i += run;
            } else {
                self.op(&region.ops[i])?;
                i += 1;
            }
        }
        Ok(())
    }

    /// Loads a run of kernel arguments with as few scalar loads as possible:
    /// each load covers up to 16 dwords starting at an even dword, so 64-bit
    /// arguments stay aligned pairs, and at least half of what it reads is
    /// used. Loads never read past the kernarg segment.
    fn kernargs(&mut self, ops: &[Op]) {
        let mut args: Vec<(u32, u32, ValueId)> = ops
            .iter()
            .map(|op| {
                let v = op.results[0];
                (op.int_attr("offset").unwrap_or(0) as u32, self.k.ty(v).dwords(), v)
            })
            .collect();
        args.sort_by_key(|a| a.0);
        let limit = super::opt::kernarg_layout(self.k).1.next_multiple_of(4);
        let mut i = 0;
        while i < args.len() {
            let start = args[i].0 & !7;
            let mut take = (i + 1, (args[i].0 + 4 * args[i].1 - start).div_ceil(4).next_power_of_two());
            let mut used = args[i].1;
            for j in i + 1..args.len() {
                used += args[j].1;
                let dwords = (args[j].0 + 4 * args[j].1 - start).div_ceil(4).next_power_of_two();
                if dwords > 16 {
                    break;
                }
                if 2 * used >= dwords && start + 4 * dwords <= limit {
                    take = (j + 1, dwords);
                }
            }
            let (end, dwords) = take;
            let opc = match dwords {
                1 => Opc::SLoadB32,
                2 => Opc::SLoadB64,
                4 => Opc::SLoadB128,
                8 => Opc::SLoadB256,
                _ => Opc::SLoadB512,
            };
            let dst = self.sreg(dwords as u8);
            self.emit(MInst::new(opc, [r(dst)], [r(self.kernarg_ptr)]).with_imm(start as i64));
            for &(offset, n, v) in &args[i..end] {
                let at = ((offset - start) / 4) as u8;
                let val = if n == 1 {
                    Val::W(r(sub(dst, at, 1)))
                } else {
                    Val::D(r(sub(dst, at, 1)), r(sub(dst, at + 1, 1)))
                };
                self.set(v, val);
            }
            i = end;
        }
    }

    fn op(&mut self, op: &Op) -> Result<(), CompileError> {
        use OpKind::*;
        let res = op.results.first().copied();
        if res.is_some_and(|v| self.chains.skip.contains(&v)) {
            return Ok(());
        }
        if self.burs.is_some() && burs::select(self, op)? {
            return Ok(());
        }
        // i16 is implemented only as the 16-bit view of a scalar f16.
        let i16_vec = |v: &ValueId| matches!(self.k.ty(*v), Type::Vec(Scalar::I16, _));
        if op.kind != VecBitcast && (op.results.iter().any(i16_vec) || op.operands.iter().any(i16_vec)) {
            return Err(unsupported(format!("{:?} of i16 vectors", op.kind)));
        }
        match op.kind {
            Constant => {
                let v = res.expect("constant has a result");
                let konst = self.facts.konst.get(v.0 as usize).copied().flatten();
                let val = match (self.k.ty(v), konst) {
                    (Type::Scalar(Scalar::I64 | Scalar::F64), Some(super::opt::Konst::Int(x))) => {
                        Val::D(Opnd::Imm(x as u64 as u32), Opnd::Imm((x as u64 >> 32) as u32))
                    }
                    (Type::Scalar(Scalar::I1), Some(super::opt::Konst::Int(x))) => {
                        Val::M(Opnd::Imm(if x & 1 != 0 { u32::MAX } else { 0 }))
                    }
                    (_, Some(super::opt::Konst::Int(x))) => Val::W(Opnd::Imm(x as u32)),
                    (_, Some(super::opt::Konst::F32(bits))) => Val::W(Opnd::Imm(bits)),
                    _ => return Err(unsupported("constant without a folded value")),
                };
                self.set(v, val);
            }
            KernArg => {
                let v = res.expect("kernarg result");
                let offset = op.int_attr("offset").unwrap_or(0);
                let dwords = self.k.ty(v).dwords() as u8;
                let dst = self.sreg(dwords);
                let opc = if dwords == 1 { Opc::SLoadB32 } else { Opc::SLoadB64 };
                self.emit(MInst::new(opc, [r(dst)], [r(self.kernarg_ptr)]).with_imm(offset));
                let val = if dwords == 1 { Val::W(r(dst)) } else { Val::D(r(sub(dst, 0, 1)), r(sub(dst, 1, 1))) };
                self.set(v, val);
            }
            ThreadId => {
                if !matches!(op.str_attr("axis"), None | Some("x")) {
                    return Err(unsupported("only gpu.thread_id x is supported"));
                }
                self.set(res.unwrap(), Val::W(r(self.tid)));
            }
            BlockId => {
                let axis = match op.str_attr("axis") {
                    None | Some("x") => 0,
                    Some("y") => 1,
                    _ => 2,
                };
                let reg = self.bid[axis].expect("block id axis enabled");
                self.set(res.unwrap(), Val::W(r(reg)));
            }
            SmemAlloc => {
                let v = res.unwrap();
                let offset = *self.lds.get(&v).unwrap_or(&0);
                self.set(v, Val::L(offset));
            }
            Or if self.packs.contains_key(&res.unwrap()) => self.pack_op(op)?,
            Add | Sub | Mul | And | Or | Xor | Shl | Lshr | Ashr | Smin | Smax | UmulHi | SmulHi => self.int_op(op)?,
            Div => self.div_op(op)?,
            Mod if *self.k.ty(op.results[0]) == Type::Scalar(Scalar::I64) => self.div64_op(op)?,
            Mod => return Err(unsupported("arith.mod survived the optimizer")),
            Cmp => self.cmp_op(op)?,
            Fcmp if matches!(*self.k.ty(op.operands[0]), Type::Scalar(Scalar::F64 | Scalar::BF16)) => {
                return Err(unsupported(format!("fcmp of {:?}", self.k.ty(op.operands[0]))));
            }
            Fcmp => {
                let pred = match op.str_attr("pred").unwrap_or("olt") {
                    "oeq" => 0x12,
                    "olt" => 0x11,
                    "ole" => 0x13,
                    "ogt" => 0x14,
                    "oge" => 0x16,
                    "one" => 0x15,
                    "une" => 0x1d,
                    other => return Err(unsupported(format!("fcmp predicate {other}"))),
                };
                let a = self.w(op.operands[0])?;
                let b = self.w(op.operands[1])?;
                let dst = self.mreg();
                self.valu(Opc::VCmp(pred), [r(dst)], [a, b], 0);
                self.set(res.unwrap(), Val::M(r(dst)));
            }
            Select => self.select_op(op)?,
            Zext | Sext | Trunc | Bitcast => self.convert_op(op)?,
            GlobalPtrAdd => {
                let v = res.unwrap();
                let ptr = self.d(op.operands[0])?;
                let off_ty = self.k.ty(op.operands[1]).clone();
                // An i32 byte offset is unsigned, as in the LLVM lowering.
                let off = if off_ty == Type::Scalar(Scalar::I64) {
                    self.d(op.operands[1])?
                } else {
                    (self.w(op.operands[1])?, Opnd::Imm(0))
                };
                let (lo, hi) = self.add64(ptr, off, self.uni(v), false);
                self.set(v, Val::D(lo, hi));
            }
            Fadd | Fsub | Fmul | Fmax | Fmin | Fma if *self.k.ty(res.unwrap()) == Type::Scalar(Scalar::F64) => self.f64_op(op)?,
            Fadd | Fsub | Fmul | Fma if matches!(self.k.ty(res.unwrap()), Type::Vec(..)) => self.vec_arith(op)?,
            Fadd | Fsub | Fmul | Fmax | Fma if *self.k.ty(res.unwrap()) != Type::Scalar(Scalar::F32) => {
                return Err(unsupported(format!("{:?} of {:?}", op.kind, self.k.ty(res.unwrap()))));
            }
            Fadd | Fsub | Fmul | Fmax => {
                let (a, b) = if op.kind == Fmax {
                    (self.max_source(op.operands[0])?, self.max_source(op.operands[1])?)
                } else {
                    (self.w(op.operands[0])?, self.w(op.operands[1])?)
                };
                let opc = match op.kind {
                    Fadd => Opc::VAddF32,
                    Fsub => Opc::VSubF32,
                    Fmul => Opc::VMulF32,
                    _ => Opc::VMaxF32,
                };
                let (a, b) = if opc != Opc::VSubF32 && !self.is_v(b) && self.is_v(a) { (b, a) } else { (a, b) };
                let opc = if opc == Opc::VSubF32 && !self.is_v(b) && self.is_v(a) { Opc::VSubrevF32 } else { opc };
                let (a, b) = if opc == Opc::VSubrevF32 { (b, a) } else { (a, b) };
                let out = self.v_bin(opc, a, b);
                self.set(res.unwrap(), Val::W(out));
            }
            Fma => {
                let mut mods = 0;
                let mut srcs = [Opnd::Imm(0); 3];
                for (i, s) in srcs.iter_mut().enumerate() {
                    let v = op.operands[i];
                    *s = match self.fneg_fma.get(&v) {
                        Some(&x) => {
                            mods |= M_NEG0 << i;
                            self.w(x)?
                        }
                        None => self.w(v)?,
                    };
                }
                let out = self.v3(Opc::VFmaF32, srcs[0], srcs[1], srcs[2], mods);
                self.set(res.unwrap(), Val::W(out));
            }
            CastToF32 if self.k.ty(op.operands[0]).elem() == Scalar::BF16 => {
                let (src, hi) = self.half(op.operands[0])?;
                let out = self.bf16_to_f32(src, hi);
                self.set(res.unwrap(), Val::W(out));
            }
            CastF32To | TruncF32ToBf16 if self.k.ty(res.unwrap()).elem() == Scalar::BF16 => {
                let x = self.w(op.operands[0])?;
                let (out, hi) = self.f32_to_bf16(x);
                self.set(res.unwrap(), Val::H(out, hi));
            }
            CastToF32 => {
                let (src, hi) = match self.get(op.operands[0])? {
                    Val::H(o, hi) => (o, hi),
                    Val::W(o) => (o, false),
                    other => return Err(unsupported(format!("cast_to_f32 of {other:?}"))),
                };
                let src = self.to_v(src);
                let dst = self.vgpr(1);
                self.valu(Opc::VCvtF32F16, [r(dst)], [src], if hi { M_OPSEL_SRC0 } else { 0 });
                self.set(res.unwrap(), Val::W(r(dst)));
            }
            CastF32To => {
                if op.str_attr("target") != Some("f16") {
                    return Err(unsupported("cast_f32_to only supports f16"));
                }
                let src = self.w(op.operands[0])?;
                let dst = self.vgpr(1);
                self.valu(Opc::VCvtF16F32, [r(dst)], [src], 0);
                self.set(res.unwrap(), Val::H(r(dst), false));
            }
            Exp2 => {
                let x = self.w(op.operands[0])?;
                let m = self.mreg();
                self.valu(Opc::VCmp(0x14), [r(m)], [Opnd::Imm(0xc2fc_0000), x], 0);
                let t = self.v3(Opc::VCndmaskB32, Opnd::Imm(0), Opnd::Imm(0x4280_0000), r(m), 0);
                let x2 = self.v_bin(Opc::VAddF32, x, t);
                let s = self.v3(Opc::VCndmaskB32, Opnd::Imm(0), Opnd::Imm(0xffff_ffc0), r(m), 0);
                let e = self.vgpr(1);
                self.valu(Opc::VExpF32, [r(e)], [x2], 0);
                let out = self.v_bin(Opc::VLdexpF32, r(e), s);
                self.set(res.unwrap(), Val::W(out));
            }
            Log2 => {
                let x = self.w(op.operands[0])?;
                let m = self.mreg();
                self.valu(Opc::VCmp(0x14), [r(m)], [Opnd::Imm(0x0080_0000), x], 0);
                let s = self.v3(Opc::VCndmaskB32, Opnd::Imm(0), Opnd::Imm(32), r(m), 0);
                let o = self.v3(Opc::VCndmaskB32, Opnd::Imm(0), Opnd::Imm(0x4200_0000), r(m), 0);
                let xs = self.v_bin(Opc::VLdexpF32, x, s);
                let l = self.vgpr(1);
                self.valu(Opc::VLogF32, [r(l)], [xs], 0);
                let out = self.v_bin(Opc::VSubF32, r(l), o);
                self.set(res.unwrap(), Val::W(out));
            }
            Rcp => {
                let x = self.w(op.operands[0])?;
                let x = self.to_v(x);
                let one = Opnd::Imm(0x3f80_0000);
                let den = self.vgpr(1);
                self.valu(Opc::VDivScaleF32, [r(den), NULL], [x, x, one], 0);
                let num = self.vgpr(1);
                let flag = self.mreg();
                self.valu(Opc::VDivScaleF32, [r(num), r(flag)], [one, x, one], 0);
                let rc = self.vgpr(1);
                self.valu(Opc::VRcpF32, [r(rc)], [r(den)], 0);
                let t = self.v3(Opc::VFmaF32, r(den), r(rc), one, M_NEG0);
                let r2 = self.v3(Opc::VFmaF32, t, r(rc), r(rc), 0);
                let q = self.v_bin(Opc::VMulF32, r(num), r2);
                let e = self.v3(Opc::VFmaF32, r(den), q, r(num), M_NEG0);
                let q2 = self.v3(Opc::VFmaF32, e, r2, q, 0);
                let e2 = self.v3(Opc::VFmaF32, r(den), q2, r(num), M_NEG0);
                self.emit(MInst::new(self.lm(Opc::SMovB32), [self.vcc()], [r(flag)]));
                let fm = self.v3(Opc::VDivFmasF32, e2, r2, q2, 0);
                let out = self.v3(Opc::VDivFixupF32, fm, x, one, 0);
                self.set(res.unwrap(), Val::W(out));
            }
            Extract => {
                let v = res.unwrap();
                let index = op.int_attr("index").unwrap_or(0) as u8;
                let src = self.get(op.operands[0])?;
                let elem = self.k.ty(op.operands[0]).elem();
                let val = match (src, elem) {
                    (Val::T(t), e) if e.is_half() => Val::H(r(sub(t, index / 2, 1)), index % 2 == 1),
                    (Val::T(t), e) if e.is_byte() => Val::B(r(sub(t, index / 4, 1)), index % 4),
                    (Val::T(t), Scalar::F64 | Scalar::I64) => Val::D(r(sub(t, 2 * index, 1)), r(sub(t, 2 * index + 1, 1))),
                    // A pair seen as two 32-bit elements (an f64 bitcast to vec<i32x2>).
                    (Val::D(lo, hi), Scalar::I32 | Scalar::F32) if index < 2 => Val::W(if index == 0 { lo } else { hi }),
                    (Val::T(t), _) => Val::W(r(sub(t, index, 1))),
                    (other, _) => {
                        if index == 0 {
                            other
                        } else {
                            return Err(unsupported("extract from a non-tuple"));
                        }
                    }
                };
                self.set(v, val);
            }
            Insert => {
                let v = res.unwrap();
                let index = op.int_attr("index").unwrap_or(0) as u8;
                if self.k.ty(v).elem().is_half() {
                    let dwords = self.k.ty(v).dwords() as u8;
                    if let Some(&(smem, addr, first, align)) = self.chains.lds.get(&v) {
                        let dst = self.vgpr(dwords);
                        let (a, offset) = self.lds_addr(smem, addr, first)?;
                        let mut done = 0u8;
                        while done < dwords {
                            let left = dwords - done;
                            let (opc, n) = match () {
                                _ if align >= 16 && left >= 4 => (Opc::DsLoadB128, 4),
                                _ if align >= 8 && left >= 2 => (Opc::DsLoadB64, 2),
                                _ => (Opc::DsLoadB32, 1),
                            };
                            self.emit(MInst::new(opc, [r(sub(dst, done, n))], [a]).with_imm(offset + 4 * done as i64));
                            done += n;
                        }
                        self.set(v, Val::T(dst));
                        return Ok(());
                    }
                    let src = if self.chains.undef_src.contains(&v) {
                        let t = self.vgpr(dwords);
                        self.emit(MInst::new(Opc::ImplicitDef, [r(t)], []));
                        t
                    } else {
                        self.tuple(op.operands[0])?
                    };
                    let d = index / 2;
                    if let Some(load) = self.deferred.remove(&op.operands[1]) {
                        let (mut uses, mods, offset, d16, mem) = match load {
                            Deferred::Global(uses, mods, offset, mem) => (uses, mods, offset, [Opc::GLoadD16B16, Opc::GLoadD16HiB16], mem),
                            Deferred::Buffer(uses, offset, mem) => (uses, 0, offset, [Opc::BufLoadD16B16, Opc::BufLoadD16HiB16], mem),
                            Deferred::Lds(..) => return Err(unsupported("LDS gather into a vector insert")),
                        };
                        let dst = self.vgpr(src.n);
                        for i in 0..src.n {
                            self.copy(sub(dst, i, 1), r(sub(src, i, 1)));
                        }
                        let part = r(sub(dst, d, 1));
                        let opc = d16[index as usize % 2];
                        uses.push(part);
                        self.emit(MInst::new(opc, [part], uses).with_imm(offset).with_mods(mods).with_mem(mem));
                        self.set(v, Val::T(dst));
                        return Ok(());
                    }
                    let elem = self.half(op.operands[1])?;
                    let dst = self.vgpr(src.n);
                    for i in 0..src.n {
                        if i != d {
                            self.copy(sub(dst, i, 1), r(sub(src, i, 1)));
                        }
                    }
                    let old = r(sub(src, d, 1));
                    if index % 2 == 0 {
                        self.pack_halves(sub(dst, d, 1), elem, (old, true));
                    } else {
                        self.pack_halves(sub(dst, d, 1), (old, false), elem);
                    }
                    self.set(v, Val::T(dst));
                    return Ok(());
                }
                let src = self.tuple(op.operands[0])?;
                let elem = self.w(op.operands[1])?;
                let dst = self.vgpr(src.n);
                for i in 0..src.n {
                    if i != index {
                        self.copy(sub(dst, i, 1), r(sub(src, i, 1)));
                    }
                }
                self.copy(sub(dst, index, 1), elem);
                self.set(v, Val::T(dst));
            }
            Pack if matches!(self.k.ty(res.unwrap()).elem(), Scalar::F64 | Scalar::I64) => {
                let v = res.unwrap();
                let dst = self.vgpr(2 * op.operands.len() as u8);
                for (i, &e) in op.operands.iter().enumerate() {
                    let (lo, hi) = self.d(e)?;
                    self.copy(sub(dst, 2 * i as u8, 1), lo);
                    self.copy(sub(dst, 2 * i as u8 + 1, 1), hi);
                }
                self.set(v, Val::T(dst));
            }
            Pack if self.k.ty(res.unwrap()).elem().is_byte() => {
                let v = res.unwrap();
                let n = op.operands.len();
                let dst = self.vgpr(n.div_ceil(4) as u8);
                for d in 0..n.div_ceil(4) {
                    let mut bytes = Vec::new();
                    for &e in &op.operands[4 * d..n.min(4 * d + 4)] {
                        bytes.push(self.byte(e)?);
                    }
                    self.pack_bytes(sub(dst, d as u8, 1), &bytes);
                }
                self.set(v, Val::T(dst));
            }
            Pack => {
                let v = res.unwrap();
                let ty = self.k.ty(v).clone();
                if ty.elem().is_half() {
                    let n = op.operands.len();
                    let dwords = n.div_ceil(2) as u8;
                    let dst = self.vgpr(dwords);
                    for i in 0..dwords as usize {
                        let lo = self.half(op.operands[2 * i])?;
                        let hi = if 2 * i + 1 < n { self.half(op.operands[2 * i + 1])? } else { (Opnd::Imm(0), false) };
                        self.pack_halves(sub(dst, i as u8, 1), lo, hi);
                    }
                    self.set(v, Val::T(dst));
                } else {
                    let dst = self.vgpr(op.operands.len() as u8);
                    for (i, &e) in op.operands.iter().enumerate() {
                        let o = self.w(e)?;
                        self.copy(sub(dst, i as u8, 1), o);
                    }
                    self.set(v, Val::T(dst));
                }
            }
            Concat => {
                let v = res.unwrap();
                if self.gathers.contains(&op.operands[0]) {
                    let (Some(Deferred::Lds(lo, lo_offset)), Some(Deferred::Lds(hi, hi_offset))) =
                        (self.deferred.remove(&op.operands[0]), self.deferred.remove(&op.operands[1]))
                    else {
                        return Err(unsupported("LDS gather pair without both loads"));
                    };
                    let dst = self.vgpr(1);
                    self.emit(MInst::new(Opc::DsLoadU16, [r(dst)], [lo]).with_imm(lo_offset));
                    self.emit(MInst::new(Opc::DsLoadU16D16Hi, [r(dst)], [hi, r(dst)]).with_imm(hi_offset));
                    self.set(v, Val::T(dst));
                    return Ok(());
                }
                let first_len = match self.k.ty(op.operands[0]) {
                    Type::Vec(_, n) => *n,
                    _ => 1,
                };
                if self.k.ty(v).elem().is_half() && first_len % 2 == 1 {
                    let mut halves = self.f16_elems(op.operands[0])?;
                    halves.extend(self.f16_elems(op.operands[1])?);
                    let dwords = halves.len().div_ceil(2) as u8;
                    let dst = self.vgpr(dwords);
                    for i in 0..dwords as usize {
                        let hi = halves.get(2 * i + 1).copied().unwrap_or((Opnd::Imm(0), false));
                        self.pack_halves(sub(dst, i as u8, 1), halves[2 * i], hi);
                    }
                    self.set(v, Val::T(dst));
                    return Ok(());
                }
                let a = self.tuple(op.operands[0])?;
                let b = self.tuple(op.operands[1])?;
                let dst = self.vgpr(a.n + b.n);
                self.copy(sub(dst, 0, a.n), r(a));
                self.copy(sub(dst, a.n, b.n), r(b));
                self.set(v, Val::T(dst));
            }
            ConstantVec => {
                let v = res.unwrap();
                let ty = self.k.ty(v).clone();
                let fill = op.float_attr("fill").unwrap_or(0.0);
                let bits = match ty.elem() {
                    Scalar::F32 => (fill as f32).to_bits(),
                    Scalar::I32 => fill as i64 as u32,
                    Scalar::F16 => {
                        let h = f32_to_f16_bits(fill as f32) as u32;
                        h | h << 16
                    }
                    Scalar::BF16 => {
                        let h = f32_to_bf16_bits(fill as f32) as u32;
                        h | h << 16
                    }
                    Scalar::I8 => (fill as i64 as u8 as u32) * 0x0101_0101,
                    _ => return Err(unsupported("constant_vec element type")),
                };
                let n = ty.dwords() as u8;
                let dst = self.vgpr(n);
                for i in 0..n {
                    self.copy(sub(dst, i, 1), Opnd::Imm(bits));
                }
                self.set(v, Val::T(dst));
            }
            TruncF32To if self.k.ty(res.unwrap()).elem() == Scalar::BF16 => {
                let v = res.unwrap();
                let src = self.tuple(op.operands[0])?;
                let n = src.n;
                let dst = self.vgpr(n.div_ceil(2));
                for i in 0..n.div_ceil(2) {
                    let lo = self.f32_to_bf16(r(sub(src, 2 * i, 1)));
                    let hi = if 2 * i + 1 < n { self.f32_to_bf16(r(sub(src, 2 * i + 1, 1))) } else { (Opnd::Imm(0), false) };
                    self.pack_halves(sub(dst, i, 1), lo, hi);
                }
                self.set(v, Val::T(dst));
            }
            TruncF32To if self.trunc_mul.contains_key(&res.unwrap()) => self.trunc_mix(op)?,
            TruncF32To => {
                let v = res.unwrap();
                let src = self.tuple(op.operands[0])?;
                let n = src.n;
                let dwords = n.div_ceil(2);
                let dst = self.vgpr(dwords);
                for i in 0..dwords {
                    let lo = r(sub(src, 2 * i, 1));
                    let hi = if 2 * i + 1 < n { r(sub(src, 2 * i + 1, 1)) } else { Opnd::Imm(0) };
                    self.emit(MInst::new(Opc::PackF16, [r(sub(dst, i, 1))], [lo, hi]));
                }
                self.set(v, Val::T(dst));
            }
            GLoad => self.gload(op)?,
            GStore => self.gstore(op)?,
            GAtomicAddF32 => {
                let (value, address) = op.operands.split_last().expect("atomic has a value");
                let data = self.w(*value)?;
                let data = self.to_v(data);
                let (vaddr, mods) = self.global_addr(address)?;
                let offset = op.int_attr("offset").unwrap_or(0);
                let mut uses = vaddr;
                uses.push(data);
                let mem = super::alias::op_tag(&self.mem_tags, op);
                self.emit(MInst::new(Opc::GAtomicAddF32, [], uses).with_imm(offset).with_mods(mods).with_mem(mem));
            }
            LLoad => self.lload(op)?,
            LStore => self.lstore(op)?,
            BufferRsrc => {
                let v = res.unwrap();
                if !self.uni(v) {
                    return Err(unsupported("divergent buffer resource"));
                }
                let (lo, hi) = self.d(op.operands[0])?;
                let size = match self.get(op.operands[1])? {
                    Val::D(lo, _) => lo,
                    _ => self.w(op.operands[1])?,
                };
                let dst = self.sreg(4);
                self.copy(sub(dst, 0, 1), lo);
                self.salu(Opc::SAndB32, [r(sub(dst, 1, 1))], [hi, Opnd::Imm(0xffff)]);
                self.copy(sub(dst, 2, 1), size);
                self.copy(sub(dst, 3, 1), Opnd::Imm(0x3101_4000));
                self.set(v, Val::T(dst));
            }
            BufferLoadVec => {
                let v = res.unwrap();
                let rsrc = self.tuple(op.operands[0])?;
                let voff = self.w(op.operands[1])?;
                let voff = self.to_v(voff);
                let soff = self.soffset(op.operands[2])?;
                let dwords = op.int_attr("dwords").unwrap_or(4) as u8;
                let opc = match dwords {
                    1 => Opc::BufLoadB32,
                    2 => Opc::BufLoadB64,
                    4 => Opc::BufLoadB128,
                    other => return Err(unsupported(format!("buffer load of {other} dwords"))),
                };
                let dst = self.vgpr(dwords);
                let offset = op.int_attr("offset").unwrap_or(0);
                let mem = super::alias::op_tag(&self.mem_tags, op);
                self.emit(MInst::new(opc, [r(dst)], [voff, r(rsrc), soff]).with_imm(offset).with_mem(mem));
                self.set(v, Val::T(dst));
            }
            BufferStoreVecF16 => {
                let rsrc = self.tuple(op.operands[0])?;
                let voff = self.w(op.operands[1])?;
                let voff = self.to_v(voff);
                let soff = self.soffset(op.operands[2])?;
                let data = self.tuple(op.operands[3])?;
                let opc = match data.n {
                    1 => Opc::BufStoreB32,
                    2 => Opc::BufStoreB64,
                    4 => Opc::BufStoreB128,
                    other => return Err(unsupported(format!("buffer store of {other} dwords"))),
                };
                let offset = op.int_attr("offset").unwrap_or(0);
                let mem = super::alias::op_tag(&self.mem_tags, op);
                self.emit(MInst::new(opc, [], [r(data), voff, r(rsrc), soff]).with_imm(offset).with_mem(mem));
            }
            DsSwizzleXor => {
                let mask = op.int_attr("xor_mask").unwrap_or(0);
                let x = self.w(op.operands[0])?;
                let x = self.to_v(x);
                let dst = self.vgpr(1);
                self.emit(MInst::new(Opc::DsSwizzleB32, [r(dst)], [x]).with_imm((mask << 10) | 0x1f));
                self.set(res.unwrap(), Val::W(r(dst)));
            }
            BufferLoadScalar => self.buffer_load(op)?,
            BufferStoreScalar => self.buffer_store(op)?,
            Mma if op.str_attr("op_id") == Some("wmma_f32_16x16x16_bf16") => {
                self.wmma(op, Opc::VWmmaF32Bf16, Opc::VWmmaF32Bf16Early, 0)?;
            }
            // Signed A and B (neg_lo:[1,1,0]), no clamp, as the LLVM lowering
            // calls the intrinsic.
            Mma if op.str_attr("op_id") == Some("wmma_i32_16x16x16_iu8") => {
                self.wmma(op, Opc::VWmmaI32Iu8, Opc::VWmmaI32Iu8Early, M_NEG0 | M_NEG1)?;
            }
            Mma if op.str_attr("op_id") == Some("wmma_i32_16x16x16_iu4") => {
                self.wmma(op, Opc::VWmmaI32Iu4, Opc::VWmmaI32Iu4Early, M_NEG0 | M_NEG1)?;
            }
            CvtF32ToI8Sat => {
                let x = self.w(op.operands[0])?;
                let rounded = self.vgpr(1);
                self.valu(Opc::VRndneF32, [r(rounded)], [x], 0);
                let whole = self.vgpr(1);
                self.valu(Opc::VCvtI32F32, [r(whole)], [r(rounded)], 0);
                let clamped = self.v3(Opc::VMed3I32, r(whole), Opnd::Imm(-128i32 as u32), Opnd::Imm(127), 0);
                self.set(res.unwrap(), Val::B(clamped, 0));
            }
            Mma => {
                if op.str_attr("op_id") != Some("wmma_f32_16x16x16_f16") {
                    return Err(unsupported(format!("tile.mma {:?}", op.str_attr("op_id"))));
                }
                let a = self.tuple(op.operands[0])?;
                let b = self.tuple(op.operands[1])?;
                let c = self.tuple(op.operands[2])?;
                let d = self.vgpr(8);
                // Accumulate in place when this is C's last use, so the copy
                // coalesces away; otherwise write a fresh D, which must not
                // overlap any source.
                if self.uses[op.operands[2].0 as usize] == 1 {
                    self.copy(d, r(c));
                    self.emit(MInst::new(Opc::VWmmaF32F16, [r(d)], [r(a), r(b), r(d)]));
                } else {
                    self.emit(MInst::new(Opc::VWmmaF32F16Early, [r(d)], [r(a), r(b), r(c)]));
                }
                self.set(res.unwrap(), Val::T(d));
            }
            // `tile.sync` waits for every counter, `sync_lds_only` for LDS
            // only and `s_barrier_bare` for nothing; a "workgroup" fence adds
            // the release and acquire fences around the barrier, as the LLVM
            // lowering writes them. One wave needs neither.
            Sync | SyncLdsOnly | BarrierBare => {
                if !self.single_wave {
                    let fenced = op.str_attr("fence") == Some("workgroup");
                    if fenced {
                        self.memory_fence("workgroup", false)?;
                    }
                    let barrier = match op.kind {
                        Sync => Opc::SBarrier,
                        SyncLdsOnly => Opc::SBarrierLds,
                        _ => Opc::SBarrierBare,
                    };
                    self.emit(MInst::new(barrier, [], []));
                    if fenced {
                        self.memory_fence("workgroup", true)?;
                    }
                }
            }
            ReadFirstLane => {
                let v = res.unwrap();
                if matches!(self.k.ty(op.operands[0]), Type::Scalar(Scalar::I64) | Type::Ptr(_)) {
                    let (lo, hi) = self.d(op.operands[0])?;
                    let lo = if self.is_v(lo) { self.to_s(lo) } else { lo };
                    let hi = if self.is_v(hi) { self.to_s(hi) } else { hi };
                    self.set(v, Val::D(lo, hi));
                    return Ok(());
                }
                let x = self.w(op.operands[0])?;
                let out = if self.is_v(x) {
                    let dst = self.sreg(1);
                    self.emit(MInst::new(Opc::VReadfirstlaneB32, [r(dst)], [x]));
                    r(dst)
                } else {
                    x
                };
                self.set(v, Val::W(out));
            }
            InlineAsmIdentity => {
                if op.str_attr("template") != Some("") || op.str_attr("constraints") != Some("=v,0") {
                    return Err(unsupported("tile.inline_asm other than the empty \"=v,0\" barrier"));
                }
                // The empty barrier: the value in VGPRs, unchanged.
                let val = match self.get(op.operands[0])? {
                    Val::D(lo, hi) => {
                        let p = self.pair_v(lo, hi);
                        Val::D(r(sub(p, 0, 1)), r(sub(p, 1, 1)))
                    }
                    Val::H(o, hi) => Val::H(self.to_v(o), hi),
                    Val::B(o, k) => Val::B(self.to_v(o), k),
                    _ => {
                        let x = self.w(op.operands[0])?;
                        Val::W(self.to_v(x))
                    }
                };
                self.set(res.unwrap(), val);
            }
            Fneg | Fabs => self.sign_op(op)?,
            Fmin | Fmax3 | Fmin3 => self.min_max_op(op)?,
            Fdiv if *self.k.ty(res.unwrap()) == Type::Scalar(Scalar::F64) => self.fdiv64_op(op)?,
            Fdiv => {
                if *self.k.ty(res.unwrap()) != Type::Scalar(Scalar::F32) {
                    return Err(unsupported(format!("arith.fdiv of {:?}", self.k.ty(res.unwrap()))));
                }
                let a = self.w(op.operands[0])?;
                let b = self.w(op.operands[1])?;
                let out = self.fdiv32(a, b);
                self.set(res.unwrap(), Val::W(out));
            }
            TruncF32ToF16 => {
                let src = self.w(op.operands[0])?;
                let dst = self.vgpr(1);
                self.valu(Opc::VCvtF16F32, [r(dst)], [src], 0);
                self.set(res.unwrap(), Val::H(r(dst), false));
            }
            SetPrio => {
                let level = op.int_attr("level").ok_or_else(|| unsupported("tile.s_setprio without level"))?;
                self.emit(MInst::new(Opc::SSetprio, [], []).with_imm(level & 0xffff));
            }
            WaitCnt => {
                self.gfx11_only("tile.wait_cnt")?;
                // gfx11 fields: expcnt 2:0, lgkmcnt 9:4, vmcnt 15:10; -1 is no wait.
                let count = |key: &str, max: i64| match op.int_attr(key).unwrap_or(-1) {
                    c if c < 0 => max,
                    c => c.min(max),
                };
                let imm = count("expcnt", 7) | count("lgkmcnt", 63) << 4 | count("vmcnt", 63) << 10;
                self.emit(MInst::new(Opc::SWaitcntExplicit, [], []).with_imm(imm));
            }
            // LLVM schedules nothing across a sched_barrier of mask 0 and lets
            // the IGroupLP mutation interpret the other masks and the group
            // barriers, which this scheduler does not model.
            SchedBarrier => {
                if op.int_attr("mask").unwrap_or(0) == 0 {
                    self.emit(MInst::new(Opc::SchedBarrier, [], []));
                }
            }
            SchedGroupBarrier => {}
            LAtomicAdd => self.lds_atomic_add(op)?,
            DppXor | QuadPerm | MovDpp8 => {
                let (opc, ctrl) = match op.kind {
                    DppXor => match op.int_attr("xor_mask") {
                        Some(mask @ 1..=15) if self.t.target == Target::Gfx950 => match isel_hooks::xor_shuffle(mask as u32) {
                            isel_hooks::XorShuffle::QuadPerm(ctrl) => (Opc::VMovB32Dpp, dpp16_ctrl(ctrl, true)),
                            isel_hooks::XorShuffle::Swizzle(offset) => (Opc::DsSwizzleB32, offset),
                        },
                        Some(mask @ 1..=15) => (Opc::VMovB32Dpp, dpp16_ctrl(0x160 | mask as u32, true)),
                        _ => return Err(unsupported("tile.dpp_xor xor_mask outside 1..15")),
                    },
                    QuadPerm => match op.int_attr("ctrl") {
                        Some(ctrl @ 0..=255) => (Opc::VMovB32Dpp, dpp16_ctrl(ctrl as u32, true)),
                        _ => return Err(unsupported("tile.quad_perm ctrl outside 0..255")),
                    },
                    _ => match op.int_attr("sel") {
                        Some(sel) => {
                            self.gfx11_only("tile.mov_dpp8")?;
                            (Opc::VMovB32Dpp8, sel as u32 & 0xff_ffff)
                        }
                        None => return Err(unsupported("tile.mov_dpp8 without sel")),
                    },
                };
                if !matches!(self.k.ty(op.operands[0]), Type::Scalar(Scalar::I32 | Scalar::F32)) {
                    return Err(unsupported(format!("{:?} of {:?}", op.kind, self.k.ty(op.operands[0]))));
                }
                let x = self.w(op.operands[0])?;
                let x = self.to_v(x);
                let dst = self.vgpr(1);
                self.emit(MInst::new(opc, [r(dst)], [x]).with_imm(ctrl as i64));
                self.set(res.unwrap(), Val::W(r(dst)));
            }
            Exp2Fast | RcpFast | Rsqrt | SitofpF32 | Rint => {
                let opc = match op.kind {
                    Exp2Fast => Opc::VExpF32,
                    RcpFast => Opc::VRcpF32,
                    Rsqrt => Opc::VRsqF32,
                    Rint => Opc::VRndneF32,
                    _ => Opc::VCvtF32I32,
                };
                let x = self.w(op.operands[0])?;
                let dst = self.vgpr(1);
                self.valu(opc, [r(dst)], [x], 0);
                self.set(res.unwrap(), Val::W(r(dst)));
            }
            Tanh => return Err(unsupported("math.tanh survived the optimizer")),
            Fdot2 => self.fdot2_op(op)?,
            Not => self.not_op(op)?,
            GlobalPtrToI64 => {
                let (lo, hi) = self.d(op.operands[0])?;
                self.set(res.unwrap(), Val::D(lo, hi));
            }
            VecAdd | VecSub | VecMul | VecFma => self.vec_arith(op)?,
            VecSplat => self.splat_op(op)?,
            VecBitcast => {
                let t = self.same_class_tuple(op.operands[0])?;
                self.set(res.unwrap(), Val::T(t));
            }
            VecUndef => {
                let t = self.vgpr(self.k.ty(res.unwrap()).dwords() as u8);
                self.emit(MInst::new(Opc::ImplicitDef, [r(t)], []));
                self.set(res.unwrap(), Val::T(t));
            }
            VecSelect => {
                if *self.k.ty(op.operands[0]) != Type::Scalar(Scalar::I1) {
                    return Err(unsupported("vector.select with a per-element mask"));
                }
                self.select_op(op)?;
            }
            VecSum => self.vec_sum(op)?,
            PermB32 => {
                let a = self.w(op.operands[0])?;
                let b = self.w(op.operands[1])?;
                let sel = self.w(op.operands[2])?;
                let out = self.v3(Opc::VPermB32, a, b, sel, 0);
                self.set(res.unwrap(), Val::W(out));
            }
            Permlanex16 | Permlane16 => {
                self.gfx11_only("tile.permlane16 / tile.permlanex16")?;
                self.permlane_op(op)?
            }
            // A wave32 has no other half to swap with: the source itself.
            Permlane64 => {
                self.gfx11_only("tile.permlane64")?;
                let x = self.get(op.operands[0])?;
                self.set(res.unwrap(), x);
            }
            DsBpermute => {
                let addr = self.w(op.operands[0])?;
                let addr = self.to_v(addr);
                let data = self.w(op.operands[1])?;
                let data = self.to_v(data);
                let dst = self.vgpr(1);
                self.emit(MInst::new(Opc::DsBpermuteB32, [r(dst)], [addr, data]));
                self.set(res.unwrap(), Val::W(r(dst)));
            }
            WaveAny | WaveAll => self.vote_op(op)?,
            LaneId => {
                let dst = self.vgpr(1);
                self.valu(Opc::VMbcntLoU32B32, [r(dst)], [Opnd::Imm(u32::MAX), Opnd::Imm(0)], 0);
                if self.t.wave == 64 {
                    let lo = dst;
                    let dst = self.vgpr(1);
                    self.valu(Opc::VMbcntHiU32B32, [r(dst)], [Opnd::Imm(u32::MAX), r(lo)], 0);
                    self.set(res.unwrap(), Val::W(r(dst)));
                } else {
                    self.set(res.unwrap(), Val::W(r(dst)));
                }
            }
            GAtomicAdd => self.atomic_add_op(op)?,
            Fence => {
                let scope = op.str_attr("scope").unwrap_or("agent");
                let ordering = op.str_attr("ordering").unwrap_or("acq_rel");
                let acquire = match ordering {
                    "release" => false,
                    "acquire" | "acq_rel" | "seq_cst" => true,
                    other => return Err(unsupported(format!("memref.fence ordering {other}"))),
                };
                self.memory_fence(scope, acquire)?;
            }
            GlobalAtomicAdd => return Err(unsupported("memref.global_atomic_add survived legalization")),
            For => self.for_op(op)?,
            If | IfElse => self.if_op(op)?,
            Yield => {}
            Return => self.emit(MInst::new(Opc::SEndpgm, [], [])),
            other => return Err(unsupported(format!("op {other:?} has no direct lowering"))),
        }
        Ok(())
    }

    /// `v` as a source of `v_max_f32`. In IEEE mode the instruction returns a
    /// quiet NaN for a signaling NaN source, where `llvm.maxnum`, the
    /// reference lowering, returns the other operand; so a source that may
    /// be signaling is quieted first, as LLVM does: with `v_max_f32 x, x, x`
    /// or, for a constant, at compile time.
    fn max_source(&mut self, v: ValueId) -> Result<Opnd, CompileError> {
        let o = self.w(v)?;
        if self.canonical.is_empty() {
            self.canonical = super::analysis::canonical_f32(self.k, self.facts);
        }
        match o {
            _ if self.canonical[v.0 as usize] => Ok(o),
            Opnd::Imm(bits) if super::analysis::is_snan_f32(bits) => Ok(Opnd::Imm(0x7fc0_0000)),
            Opnd::Imm(_) => Ok(o),
            _ => {
                let dst = self.vgpr(1);
                self.valu(Opc::VMaxF32, [r(dst)], [o, o], 0);
                Ok(r(dst))
            }
        }
    }

    /// `arith.fneg` / `arith.fabs`: the sign bit flipped or cleared, of an
    /// f32 or f16 scalar or vector, or of an f64 (its high half).
    fn sign_op(&mut self, op: &Op) -> Result<(), CompileError> {
        let v = op.results[0];
        if self.fneg_fma.contains_key(&v) {
            return Ok(());
        }
        let neg = op.kind == OpKind::Fneg;
        let apply = |s: &mut Self, x: Opnd, bits: u32| match x {
            Opnd::Imm(b) => Opnd::Imm(if neg { b ^ bits } else { b & !bits }),
            _ if neg => s.v_bin(Opc::VXorB32, Opnd::Imm(bits), x),
            _ => s.v_bin(Opc::VAndB32, Opnd::Imm(!bits), x),
        };
        let val = match self.k.ty(v).clone() {
            Type::Scalar(Scalar::F32) => {
                let x = self.w(op.operands[0])?;
                Val::W(apply(self, x, 0x8000_0000))
            }
            Type::Scalar(Scalar::F16) => {
                let (x, hi) = self.half(op.operands[0])?;
                Val::H(apply(self, x, if hi { 0x8000_0000 } else { 0x8000 }), hi)
            }
            Type::Scalar(Scalar::F64) => {
                let (lo, hi) = self.d(op.operands[0])?;
                if self.canon64.contains(&op.operands[0]) {
                    self.canon64.insert(v);
                }
                Val::D(lo, apply(self, hi, 0x8000_0000))
            }
            Type::Vec(elem @ (Scalar::F32 | Scalar::F16), _) => {
                let t = self.tuple(op.operands[0])?;
                let bits = if elem == Scalar::F32 { 0x8000_0000 } else { 0x8000_8000 };
                let dst = self.vgpr(t.n);
                for i in 0..t.n {
                    let o = apply(self, r(sub(t, i, 1)), bits);
                    self.copy(sub(dst, i, 1), o);
                }
                Val::T(dst)
            }
            other => return Err(unsupported(format!("{:?} of {other:?}", op.kind))),
        };
        self.set(v, val);
        Ok(())
    }

    /// `arith.fmin`, `arith.fmax3`, `arith.fmin3` on f32, with the sources
    /// quieted like `arith.fmax`'s.
    fn min_max_op(&mut self, op: &Op) -> Result<(), CompileError> {
        let v = op.results[0];
        if *self.k.ty(v) != Type::Scalar(Scalar::F32) {
            return Err(unsupported(format!("{:?} of {:?}", op.kind, self.k.ty(v))));
        }
        let mut srcs = Vec::with_capacity(3);
        for &o in &op.operands {
            srcs.push(self.max_source(o)?);
        }
        let out = match op.kind {
            OpKind::Fmin => {
                let (a, b) = if !self.is_v(srcs[1]) && self.is_v(srcs[0]) { (srcs[1], srcs[0]) } else { (srcs[0], srcs[1]) };
                self.v_bin(Opc::VMinF32, a, b)
            }
            OpKind::Fmax3 => self.v3(Opc::VMax3F32, srcs[0], srcs[1], srcs[2], 0),
            _ => self.v3(Opc::VMin3F32, srcs[0], srcs[1], srcs[2], 0),
        };
        self.set(v, Val::W(out));
        Ok(())
    }

    /// IEEE f32 `a / b` with f32 denormals on: LLVM's sequence, as `math.rcp`
    /// (`1 / x`) uses with `a = 1`.
    fn fdiv32(&mut self, a: Opnd, b: Opnd) -> Opnd {
        let b = self.to_v(b);
        let one = Opnd::Imm(0x3f80_0000);
        let den = self.vgpr(1);
        self.valu(Opc::VDivScaleF32, [r(den), NULL], [b, b, a], 0);
        let num = self.vgpr(1);
        let flag = self.mreg();
        self.valu(Opc::VDivScaleF32, [r(num), r(flag)], [a, b, a], 0);
        let rc = self.vgpr(1);
        self.valu(Opc::VRcpF32, [r(rc)], [r(den)], 0);
        let t = self.v3(Opc::VFmaF32, r(den), r(rc), one, M_NEG0);
        let r2 = self.v3(Opc::VFmaF32, t, r(rc), r(rc), 0);
        let q = self.v_bin(Opc::VMulF32, r(num), r2);
        let e = self.v3(Opc::VFmaF32, r(den), q, r(num), M_NEG0);
        let q2 = self.v3(Opc::VFmaF32, e, r2, q, 0);
        let e2 = self.v3(Opc::VFmaF32, r(den), q2, r(num), M_NEG0);
        self.emit(MInst::new(self.lm(Opc::SMovB32), [self.vcc()], [r(flag)]));
        let fm = self.v3(Opc::VDivFmasF32, e2, r2, q2, 0);
        self.v3(Opc::VDivFixupF32, fm, b, a, 0)
    }

    /// IEEE f64 `a / b`, as LLVM lowers it. A constant divisor with an exact
    /// normal reciprocal `r = ±2^k` is a multiply by `r`, which LLVM then
    /// simplifies: `a`, `-a` and `a + a` for `r` = 1, -1 and 2, `v_mul_f64` for
    /// the other inline constants, `v_ldexp_f64 ±a, k` otherwise. Any other
    /// divisor takes div_scale, rcp, two Newton steps, div_fmas and div_fixup.
    fn fdiv64_op(&mut self, op: &Op) -> Result<(), CompileError> {
        let v = op.results[0];
        let (alo, ahi) = self.d(op.operands[0])?;
        let (blo, bhi) = self.d(op.operands[1])?;
        if let (Opnd::Imm(l), Opnd::Imm(h)) = (blo, bhi) {
            let bits = (h as u64) << 32 | l as u64;
            let exp = (bits >> 52 & 0x7ff) as i32;
            if bits & 0x000f_ffff_ffff_ffff == 0 && (1..=2045).contains(&exp) {
                let (k, neg) = (1023 - exp, bits >> 63 != 0);
                if k == 0 {
                    let hi = match ahi {
                        Opnd::Imm(x) if neg => Opnd::Imm(x ^ 0x8000_0000),
                        _ if neg => self.v_bin(Opc::VXorB32, Opnd::Imm(0x8000_0000), ahi),
                        _ => ahi,
                    };
                    if self.canon64.contains(&op.operands[0]) {
                        self.canon64.insert(v);
                    }
                    self.set(v, Val::D(alo, hi));
                    return Ok(());
                }
                let a = self.f64_operand(alo, ahi);
                let out = match (k, neg) {
                    (1, false) => self.v64(Opc::VAddF64, [a, a], 0),
                    (-1..=2, _) => {
                        let rbits = ((1023 + k) as u64) << 52 | (neg as u64) << 63;
                        let rc = self.f64_operand(Opnd::Imm(0), Opnd::Imm((rbits >> 32) as u32));
                        self.v64(Opc::VMulF64, [a, rc], 0)
                    }
                    _ => self.v64(Opc::VLdexpF64, [a, Opnd::Imm(k as u32)], if neg { M_NEG0 } else { 0 }),
                };
                self.canon64.insert(v);
                self.set(v, Val::D(r(sub(out, 0, 1)), r(sub(out, 1, 1))));
                return Ok(());
            }
        }
        let a = self.f64_operand(alo, ahi);
        let b = self.f64_operand(blo, bhi);
        let one = self.f64_operand(Opnd::Imm(0), Opnd::Imm(0x3ff0_0000));
        let den = self.vgpr(2);
        self.valu(Opc::VDivScaleF64, [r(den), NULL], [b, b, a], 0);
        let num = self.vgpr(2);
        let flag = self.mreg();
        self.valu(Opc::VDivScaleF64, [r(num), r(flag)], [a, b, a], 0);
        let rc = self.v64(Opc::VRcpF64, [r(den)], 0);
        let e0 = self.v64(Opc::VFmaF64, [r(den), r(rc), one], M_NEG0);
        let r1 = self.v64(Opc::VFmaF64, [r(rc), r(e0), r(rc)], 0);
        let e1 = self.v64(Opc::VFmaF64, [r(den), r(r1), one], M_NEG0);
        let r2 = self.v64(Opc::VFmaF64, [r(r1), r(e1), r(r1)], 0);
        let q = self.v64(Opc::VMulF64, [r(num), r(r2)], 0);
        let e2 = self.v64(Opc::VFmaF64, [r(den), r(q), r(num)], M_NEG0);
        self.emit(MInst::new(self.lm(Opc::SMovB32), [self.vcc()], [r(flag)]));
        let fm = self.v64(Opc::VDivFmasF64, [r(e2), r(r2), r(q)], 0);
        let out = self.v64(Opc::VDivFixupF64, [r(fm), b, a], 0);
        self.canon64.insert(v);
        self.set(v, Val::D(r(sub(out, 0, 1)), r(sub(out, 1, 1))));
        Ok(())
    }

    /// `arith.fdot2(a, b: f16x2 or bf16x2, c: f32)`, clamp off:
    /// `v_dot2_f32_f16` / `v_dot2_f32_bf16`.
    fn fdot2_op(&mut self, op: &Op) -> Result<(), CompileError> {
        // gfx950's DOT hazards are the MFMA-family rules (design 5.2, G2).
        self.gfx11_only("arith.fdot2")?;
        let opc = match self.k.ty(op.operands[0]).elem() {
            Scalar::F16 => Opc::VDot2F32F16,
            Scalar::BF16 => Opc::VDot2F32Bf16,
            _ => return Err(unsupported("arith.fdot2 of pairs other than f16 or bf16")),
        };
        let a = self.tuple(op.operands[0])?;
        let b = self.tuple(op.operands[1])?;
        let c = self.w(op.operands[2])?;
        // VOP3P takes no literal.
        let c = match c {
            Opnd::Imm(x) if !(is_inline_int(x) || is_inline_f32(x)) => self.to_v(c),
            _ => c,
        };
        let dst = self.vgpr(1);
        self.valu(opc, [r(dst)], [r(a), r(b), c], 0);
        self.set(op.results[0], Val::W(r(dst)));
        Ok(())
    }

    /// `arith.not`: all bits flipped; a uniform compare flips its predicate.
    fn not_op(&mut self, op: &Op) -> Result<(), CompileError> {
        let v = op.results[0];
        let x = op.operands[0];
        let val = match self.k.ty(v).clone() {
            Type::Scalar(Scalar::I1) => match self.get(x)? {
                Val::C(c) if inverse_scmp(c.op).is_some() => {
                    Val::C(SCmp { op: inverse_scmp(c.op).expect("checked"), a: c.a, b: c.b })
                }
                _ => match self.mask(x)? {
                    Opnd::Imm(m) => Val::M(Opnd::Imm(!m)),
                    m => {
                        let dst = self.mreg();
                        self.salu(self.lm(Opc::SNotB32), [r(dst)], [m]);
                        Val::M(r(dst))
                    }
                },
            },
            Type::Scalar(Scalar::I32) => {
                let o = self.w(x)?;
                Val::W(self.not32(o, self.uni(v)))
            }
            Type::Scalar(Scalar::I64) => {
                let (lo, hi) = self.d(x)?;
                let uniform = self.uni(v);
                Val::D(self.not32(lo, uniform), self.not32(hi, uniform))
            }
            other => return Err(unsupported(format!("arith.not of {other:?}"))),
        };
        self.set(v, val);
        Ok(())
    }

    fn not32(&mut self, o: Opnd, uniform: bool) -> Opnd {
        if let Opnd::Imm(x) = o {
            return Opnd::Imm(!x);
        }
        let dst = if uniform { self.sreg(1) } else { self.vgpr(1) };
        if uniform {
            self.salu(Opc::SNotB32, [r(dst)], [o]);
        } else {
            self.valu(Opc::VNotB32, [r(dst)], [o], 0);
        }
        r(dst)
    }

    /// Element-wise `vector.add/sub/mul/fma`, and `arith.fadd/fsub/fmul/fma`
    /// on vectors: one VALU op per f32 or i32 element, one packed op per pair
    /// of f16 elements. A `vector.splat` operand is read as its scalar.
    fn vec_arith(&mut self, op: &Op) -> Result<(), CompileError> {
        #[derive(Clone, Copy)]
        enum Src {
            Tuple(VRef),
            Scalar(Opnd),
        }
        let v = op.results[0];
        let Type::Vec(elem, n) = *self.k.ty(v) else {
            return Err(unsupported(format!("{:?} of a scalar", op.kind)));
        };
        let kind = match op.kind {
            OpKind::VecAdd | OpKind::Fadd => OpKind::Fadd,
            OpKind::VecSub | OpKind::Fsub => OpKind::Fsub,
            OpKind::VecMul | OpKind::Fmul => OpKind::Fmul,
            _ => OpKind::Fma,
        };
        let mut srcs = Vec::with_capacity(3);
        for &o in &op.operands {
            srcs.push(match self.splats.get(&o) {
                Some(&s) if elem.bytes() == 4 => Src::Scalar(s),
                _ if matches!(self.k.ty(o), Type::Vec(..)) => Src::Tuple(self.tuple(o)?),
                _ if elem.bytes() == 4 => Src::Scalar(self.w(o)?),
                _ => return Err(unsupported(format!("{:?} of an f16 vector and a scalar", op.kind))),
            });
        }
        let float = elem.is_float();
        let (dwords, opc) = match (elem, kind) {
            (Scalar::F32 | Scalar::I32, _) => (n as u8, None),
            (Scalar::F16, OpKind::Fadd) => (n.div_ceil(2) as u8, Some(Opc::VPkAddF16)),
            (Scalar::F16, OpKind::Fmul) => (n.div_ceil(2) as u8, Some(Opc::VPkMulF16)),
            (Scalar::F16, OpKind::Fma) => (n.div_ceil(2) as u8, Some(Opc::VPkFmaF16)),
            _ => return Err(unsupported(format!("{kind:?} of vectors of {elem:?}"))),
        };
        let dst = self.vgpr(dwords);
        for i in 0..dwords {
            let x: Vec<Opnd> = srcs
                .iter()
                .map(|s| match *s {
                    Src::Tuple(t) => r(sub(t, i, 1)),
                    Src::Scalar(o) => o,
                })
                .collect();
            let d = [r(sub(dst, i, 1))];
            if let Some(opc) = opc {
                self.valu(opc, d, &x[..], 0);
                continue;
            }
            // VOP2 wants the VGPR in src1.
            let swap = !self.is_v(x[1]) && self.is_v(x[0]);
            match (kind, float) {
                (OpKind::Fma, true) => self.valu(Opc::VFmaF32, d, [x[0], x[1], x[2]], 0),
                (OpKind::Fma, false) => return Err(unsupported("integer vector.fma")),
                (OpKind::Fsub, _) => {
                    let (fwd, rev) = if float { (Opc::VSubF32, Opc::VSubrevF32) } else { (Opc::VSubNcU32, Opc::VSubrevNcU32) };
                    if swap {
                        self.valu(rev, d, [x[1], x[0]], 0);
                    } else {
                        self.valu(fwd, d, [x[0], x[1]], 0);
                    }
                }
                _ => {
                    let opc = match (kind, float) {
                        (OpKind::Fadd, true) => Opc::VAddF32,
                        (OpKind::Fmul, true) => Opc::VMulF32,
                        (OpKind::Fadd, false) => Opc::VAddNcU32,
                        _ => Opc::VMulLoU32,
                    };
                    let (a, b) = if swap { (x[1], x[0]) } else { (x[0], x[1]) };
                    self.valu(opc, d, [a, b], 0);
                }
            }
        }
        self.set(v, Val::T(dst));
        Ok(())
    }

    /// `vector.splat`: the scalar copied into every element.
    fn splat_op(&mut self, op: &Op) -> Result<(), CompileError> {
        let v = op.results[0];
        let Type::Vec(elem, n) = *self.k.ty(v) else {
            return Err(unsupported("vector.splat to a scalar"));
        };
        match elem.bytes() {
            4 => {
                let o = self.w(op.operands[0])?;
                let dst = self.vgpr(n as u8);
                for i in 0..n as u8 {
                    self.copy(sub(dst, i, 1), o);
                }
                self.splats.insert(v, o);
                self.set(v, Val::T(dst));
            }
            2 if elem == Scalar::F16 => {
                let h = self.half(op.operands[0])?;
                let dwords = n.div_ceil(2) as u8;
                let dst = self.vgpr(dwords);
                self.pack_halves(sub(dst, 0, 1), h, h);
                for i in 1..dwords {
                    self.copy(sub(dst, i, 1), r(sub(dst, 0, 1)));
                }
                self.set(v, Val::T(dst));
            }
            _ => return Err(unsupported(format!("vector.splat of {elem:?}"))),
        }
        Ok(())
    }

    /// `v`'s registers as one tuple in the class `v` lives in, for a
    /// reinterpretation that keeps uniform values in SGPRs.
    fn same_class_tuple(&mut self, v: ValueId) -> Result<VRef, CompileError> {
        match self.get(v)? {
            Val::T(t) | Val::W(Opnd::R(t)) | Val::H(Opnd::R(t), false) => Ok(t),
            Val::D(lo, hi) => {
                let class = if self.is_v(lo) || self.is_v(hi) || !self.uni(v) { Class::V } else { Class::S };
                Ok(self.pair(lo, hi, class))
            }
            Val::W(o @ Opnd::Imm(_)) | Val::H(o @ Opnd::Imm(_), false) => {
                let dst = if self.uni(v) { self.sreg(1) } else { self.vgpr(1) };
                self.copy(dst, o);
                Ok(dst)
            }
            _ => self.tuple(v),
        }
    }

    /// `vector.sum` of f32 elements: `((0 + e0) + e1) + ...`, in the order
    /// and from the +0 start the LLVM lowering uses.
    fn vec_sum(&mut self, op: &Op) -> Result<(), CompileError> {
        let v = op.results[0];
        let Type::Vec(Scalar::F32, n) = *self.k.ty(op.operands[0]) else {
            return Err(unsupported(format!("vector.sum of {:?}", self.k.ty(op.operands[0]))));
        };
        let t = self.tuple(op.operands[0])?;
        let mut acc = Opnd::Imm(0);
        for i in 0..n as u8 {
            acc = self.v_bin(Opc::VAddF32, acc, r(sub(t, i, 1)));
        }
        self.set(v, Val::W(acc));
        Ok(())
    }

    /// A `v_permlane*` lane select: an SGPR or an inline constant.
    fn lane_select(&mut self, v: ValueId) -> Result<Opnd, CompileError> {
        let o = self.w(v)?;
        Ok(match o {
            Opnd::Imm(x) if is_inline_int(x) => o,
            _ => self.to_s(o),
        })
    }

    /// `tile.permlanex16` (every lane reads lane `L ^ 16`) and
    /// `tile.permlane16(old, src, sel_lo, sel_hi) {fi, bound_ctrl}`.
    fn permlane_op(&mut self, op: &Op) -> Result<(), CompileError> {
        let v = op.results[0];
        let dst = self.vgpr(1);
        if op.kind == OpKind::Permlanex16 {
            let x = self.w(op.operands[0])?;
            let x = self.to_v(x);
            let sel = match self.lane_sel {
                Some(s) => s,
                None => {
                    let s = self.sreg(1);
                    self.lane_sel = Some(s);
                    s
                }
            };
            // bound_ctrl: a lane whose partner is inactive reads 0, so the
            // destination's old value is never used.
            let uses = [x, r(sel), Opnd::Imm(PERMLANEX16_SEL.1)];
            self.emit(MInst::new(Opc::VPermlanex16B32, [r(dst)], uses).with_mods(M_OPSEL_SRC1));
        } else {
            let old = self.w(op.operands[0])?;
            let src = self.w(op.operands[1])?;
            let src = self.to_v(src);
            let lo = self.lane_select(op.operands[2])?;
            let hi = self.lane_select(op.operands[3])?;
            let mut mods = 0;
            if op.bool_attr("fi") == Some(true) {
                mods |= M_OPSEL_SRC0;
            }
            if op.bool_attr("bound_ctrl") == Some(true) {
                mods |= M_OPSEL_SRC1;
            }
            // Lanes reading an inactive lane keep `old`: the destination is
            // also read (the fourth use, not encoded).
            self.copy(dst, old);
            self.emit(MInst::new(Opc::VPermlane16B32, [r(dst)], [src, lo, hi, r(dst)]).with_mods(mods));
        }
        self.set(v, Val::W(r(dst)));
        Ok(())
    }

    /// A packed f16 pair (`find_f16_packs`) built in the two halves of one
    /// register: two conversions as `PACK_F16`, two products as
    /// `v_fma_mixlo/mixhi_f16` (as LLVM selects them), else the halves
    /// joined with `v_perm_b32`.
    fn pack_op(&mut self, op: &Op) -> Result<(), CompileError> {
        let v = op.results[0];
        let parts = self.packs[&v];
        let dst = self.vgpr(1);
        match parts {
            [PackPart::Cvt(a), PackPart::Cvt(b)] => {
                let (a, b) = (self.w(a)?, self.w(b)?);
                self.emit(MInst::new(Opc::PackF16, [r(dst)], [a, b]));
            }
            [PackPart::Mix(a, b), PackPart::Mix(c, d)] => {
                let (a, b) = (self.vop3p_src(a)?, self.vop3p_src(b)?);
                self.valu(Opc::VFmaMixloF16, [r(dst)], [a, b, Opnd::Imm(0)], 0);
                let (c, d) = (self.vop3p_src(c)?, self.vop3p_src(d)?);
                self.valu(Opc::VFmaMixhiF16, [r(dst)], [c, d, Opnd::Imm(0), r(dst)], 0);
            }
            _ => {
                let lo = self.pack_half(parts[0])?;
                let hi = self.pack_half(parts[1])?;
                self.pack_halves(dst, lo, hi);
            }
        }
        self.set(v, Val::W(r(dst)));
        Ok(())
    }

    fn pack_half(&mut self, part: PackPart) -> Result<(Opnd, bool), CompileError> {
        Ok(match part {
            PackPart::Half(h) => self.half(h)?,
            PackPart::Cvt(f) => {
                let x = self.w(f)?;
                let t = self.vgpr(1);
                self.valu(Opc::VCvtF16F32, [r(t)], [x], 0);
                (r(t), false)
            }
            PackPart::Mix(a, b) => {
                let (a, b) = (self.vop3p_src(a)?, self.vop3p_src(b)?);
                let t = self.vgpr(1);
                self.valu(Opc::VFmaMixloF16, [r(t)], [a, b, Opnd::Imm(0)], 0);
                (r(t), false)
            }
        })
    }

    /// `v` as a VOP3P source, which takes no literal.
    fn vop3p_src(&mut self, v: ValueId) -> Result<Opnd, CompileError> {
        let o = self.w(v)?;
        Ok(self.vop3p_opnd(o))
    }

    fn vop3p_opnd(&mut self, o: Opnd) -> Opnd {
        match o {
            Opnd::Imm(x) if !(is_inline_int(x) || is_inline_f32(x)) => self.to_v(o),
            _ => o,
        }
    }

    /// `vector.trunc_f32_to` f16 of an f32 vector product (`find_trunc_products`):
    /// every element rounded once, by `v_fma_mixlo/mixhi_f16(a, b, 0)` as LLVM
    /// selects it.
    fn trunc_mix(&mut self, op: &Op) -> Result<(), CompileError> {
        let v = op.results[0];
        let (a, b) = self.trunc_mul[&v];
        let Type::Vec(_, n) = *self.k.ty(a) else { unreachable!("find_trunc_products takes vectors") };
        let n = n as u8;
        let mut elems = Vec::with_capacity(2);
        for x in [a, b] {
            elems.push(match self.splats.get(&x) {
                Some(&o) => vec![o; n as usize],
                None => {
                    let t = self.tuple(x)?;
                    (0..n).map(|i| r(sub(t, i, 1))).collect()
                }
            });
        }
        let dst = self.vgpr(n / 2);
        for i in 0..n {
            let d = sub(dst, i / 2, 1);
            let (p, q) = (self.vop3p_opnd(elems[0][i as usize]), self.vop3p_opnd(elems[1][i as usize]));
            if i % 2 == 0 {
                self.valu(Opc::VFmaMixloF16, [r(d)], [p, q, Opnd::Imm(0)], 0);
            } else {
                self.valu(Opc::VFmaMixhiF16, [r(d)], [p, q, Opnd::Imm(0), r(d)], 0);
            }
        }
        self.set(v, Val::T(dst));
        Ok(())
    }

    /// `tile.wave_any(p)` / `tile.wave_all(p)`: whether some / every lane of
    /// the wave has `p != 0`; all of them must be active for `wave_all`, as
    /// in the LLVM lowering (ballot == full wave mask).
    fn vote_op(&mut self, op: &Op) -> Result<(), CompileError> {
        let p = op.operands[0];
        let lanes = if let Some(&(c, t, f)) = self.vote_sel.get(&p) {
            // p = select(c, t, f): the lanes where p != 0 are c's or its complement's.
            match (t, f) {
                (true, true) => self.exec(),
                (false, false) => Opnd::Imm(0),
                (true, false) => {
                    let m = self.mask(c)?;
                    self.m_bin(Opc::SAndB32, m, self.exec())
                }
                (false, true) => {
                    let m = self.mask(c)?;
                    self.m_bin(Opc::SAndNot1B32, self.exec(), m)
                }
            }
        } else if *self.k.ty(p) == Type::Scalar(Scalar::I1) {
            let m = self.mask(p)?;
            self.m_bin(Opc::SAndB32, m, self.exec())
        } else {
            let x = self.w(p)?;
            let m = self.mreg();
            self.valu(Opc::VCmp(0x4d), [r(m)], [Opnd::Imm(0), x], 0);
            r(m)
        };
        let (test, against) = if op.kind == OpKind::WaveAny {
            (Opc::SCmpLgU32, Opnd::Imm(0))
        } else {
            (Opc::SCmpEqU32, Opnd::Imm(u32::MAX))
        };
        self.salu(self.lm(test), [], [lanes, against]);
        let dst = self.sreg(1);
        self.salu(Opc::SCselectB32, [r(dst)], [Opnd::Imm(1), Opnd::Imm(0)]);
        self.set(op.results[0], Val::W(r(dst)));
        Ok(())
    }

    /// Legalized `memref.global_atomic_add` of an i32 or f32: the returning
    /// form (GLC) only when the old value is used, and fences around it for
    /// the orderings above monotonic.
    fn atomic_add_op(&mut self, op: &Op) -> Result<(), CompileError> {
        let v = op.results[0];
        let (&value, address) = op.operands.split_last().expect("atomic has a value");
        let (plain, returning) = match self.k.ty(value).elem() {
            Scalar::I32 => (Opc::GAtomicAddU32, Opc::GAtomicAddU32Rtn),
            Scalar::F32 => (Opc::GAtomicAddF32, Opc::GAtomicAddF32Rtn),
            other => return Err(unsupported(format!("memref.global_atomic_add of {other:?}"))),
        };
        let (release, acquire) = match op.str_attr("ordering").unwrap_or("monotonic") {
            "monotonic" => (false, false),
            "release" => (true, false),
            "acquire" => (false, true),
            "acq_rel" | "seq_cst" => (true, true),
            other => return Err(unsupported(format!("memref.global_atomic_add ordering {other}"))),
        };
        let data = self.w(value)?;
        let data = self.to_v(data);
        let (mut uses, mods) = self.global_addr(address)?;
        uses.push(data);
        let offset = op.int_attr("offset").unwrap_or(0);
        let mem = super::alias::op_tag(&self.mem_tags, op);
        if release {
            self.memory_fence("system", false)?;
        }
        if self.uses[v.0 as usize] > 0 {
            let dst = self.vgpr(1);
            self.emit(MInst::new(returning, [r(dst)], uses).with_imm(offset).with_mods(mods | M_GLC).with_mem(mem));
            self.set(v, Val::W(r(dst)));
        } else {
            self.emit(MInst::new(plain, [], uses).with_imm(offset).with_mods(mods).with_mem(mem));
        }
        if acquire {
            self.memory_fence("system", true)?;
        }
        Ok(())
    }

    /// A gfx11 memory fence at `scope`: every earlier load and store completes
    /// (`insert_waitcnt` drains the loads before the store wait); an acquire
    /// then invalidates the caches below the scope (L0 is per CU, and a
    /// workgroup spans both CUs of a WGP).
    fn memory_fence(&mut self, scope: &str, acquire: bool) -> Result<(), CompileError> {
        self.gfx11_only("memory fences")?;
        let l1 = match scope {
            "agent" | "system" => true,
            "workgroup" => false,
            other => return Err(unsupported(format!("memory fence scope {other}"))),
        };
        self.emit(MInst::new(Opc::SWaitcntVscnt, [NULL], []));
        if acquire {
            if l1 {
                self.emit(MInst::new(Opc::BufferGl1Inv, [], []));
            }
            self.emit(MInst::new(Opc::BufferGl0Inv, [], []));
        }
        Ok(())
    }

    fn half(&mut self, v: ValueId) -> Result<(Opnd, bool), CompileError> {
        match self.get(v)? {
            Val::H(o, hi) => Ok((o, hi)),
            Val::W(o) => Ok((o, false)),
            Val::T(t) if t.n == 1 => Ok((r(t), false)),
            other => Err(unsupported(format!("{other:?} is not an f16 value"))),
        }
    }

    /// Every element of an f16 scalar or vector as (dword, high half).
    fn f16_elems(&mut self, v: ValueId) -> Result<Vec<(Opnd, bool)>, CompileError> {
        let n = match self.k.ty(v) {
            Type::Vec(_, n) => *n as u8,
            _ => 1,
        };
        if n == 1 {
            return Ok(vec![self.half(v)?]);
        }
        let t = self.tuple(v)?;
        Ok((0..n).map(|i| (r(sub(t, i / 2, 1)), i % 2 == 1)).collect())
    }

    /// `dst = {lo half, hi half}` with v_perm_b32.
    fn pack_halves(&mut self, dst: VRef, lo: (Opnd, bool), hi: (Opnd, bool)) {
        if let (Opnd::R(a), Opnd::R(b)) = (lo.0, hi.0) {
            if a == b && !lo.1 && hi.1 {
                self.copy(dst, lo.0);
                return;
            }
        }
        let lo_src = self.to_v(lo.0);
        let hi_src = self.to_v(hi.0);
        let lo_byte = if lo.1 { 2 } else { 0 };
        let hi_byte = if hi.1 { 6 } else { 4 };
        let sel = lo_byte | (lo_byte + 1) << 8 | hi_byte << 16 | (hi_byte + 1) << 24;
        self.valu(Opc::VPermB32, [r(dst)], [hi_src, lo_src, Opnd::Imm(sel)], 0);
    }

    /// f64 add, subtract (an add of the negated second source), multiply,
    /// maxnum, minnum and fma on register pairs. As for f32, a maxnum or
    /// minnum source that may be a signaling NaN is quieted first
    /// (`v_max_f64 x, x`), as LLVM does; f64 arithmetic results are already quiet.
    fn f64_op(&mut self, op: &Op) -> Result<(), CompileError> {
        let v = op.results[0];
        let mut srcs = Uses::new();
        for &x in &op.operands {
            let (lo, hi) = self.d(x)?;
            let mut s = self.f64_operand(lo, hi);
            if matches!(op.kind, OpKind::Fmax | OpKind::Fmin) && !self.canon64.contains(&x) {
                s = match (lo, hi) {
                    (Opnd::Imm(l), Opnd::Imm(h)) => {
                        let bits = (h as u64) << 32 | l as u64;
                        let snan = bits & 0x7ff8_0000_0000_0000 == 0x7ff0_0000_0000_0000 && bits & 0x000f_ffff_ffff_ffff != 0;
                        let q = if snan { bits | 0x0008_0000_0000_0000 } else { bits };
                        self.f64_operand(Opnd::Imm(q as u32), Opnd::Imm((q >> 32) as u32))
                    }
                    _ => {
                        let dst = self.vgpr(2);
                        self.valu(Opc::VMaxF64, [r(dst)], [s, s], 0);
                        r(dst)
                    }
                };
            }
            srcs.push(s);
        }
        let (opc, mods) = match op.kind {
            OpKind::Fadd => (Opc::VAddF64, 0),
            OpKind::Fsub => (Opc::VAddF64, M_NEG1),
            OpKind::Fmul => (Opc::VMulF64, 0),
            OpKind::Fmax => (Opc::VMaxF64, 0),
            OpKind::Fmin => (Opc::VMinF64, 0),
            _ => (Opc::VFmaF64, 0),
        };
        let dst = self.vgpr(2);
        self.valu(opc, [r(dst)], srcs, mods);
        self.canon64.insert(v);
        self.set(v, Val::D(r(sub(dst, 0, 1)), r(sub(dst, 1, 1))));
        Ok(())
    }

    /// The 16-bit integer in one half of `o`, zero- or sign-extended to 32
    /// bits; the other half may hold anything.
    fn widen16(&mut self, o: Opnd, hi: bool, signed: bool, uniform: bool) -> Opnd {
        if let Opnd::Imm(c) = o {
            let h = if hi { c >> 16 } else { c & 0xffff };
            return Opnd::Imm(if signed { h as u16 as i16 as i32 as u32 } else { h });
        }
        if uniform && !self.is_v(o) {
            return match (hi, signed) {
                (false, false) => self.s_bin(Opc::SAndB32, o, Opnd::Imm(0xffff)),
                (true, false) => self.s_bin(Opc::SLshrB32, o, Opnd::Imm(16)),
                (true, true) => self.s_bin(Opc::SAshrI32, o, Opnd::Imm(16)),
                (false, true) => {
                    let up = self.s_bin(Opc::SLshlB32, o, Opnd::Imm(16));
                    self.s_bin(Opc::SAshrI32, up, Opnd::Imm(16))
                }
            };
        }
        match (hi, signed) {
            (false, false) => self.v_bin(Opc::VAndB32, Opnd::Imm(0xffff), o),
            (true, false) => self.v_bin(Opc::VLshrrevB32, Opnd::Imm(16), o),
            (true, true) => self.v_bin(Opc::VAshrrevI32, Opnd::Imm(16), o),
            (false, true) => self.v3(Opc::VBfeI32, o, Opnd::Imm(0), Opnd::Imm(16), 0),
        }
    }

    /// An i8 as (operand, byte index).
    fn byte(&mut self, v: ValueId) -> Result<(Opnd, u8), CompileError> {
        match self.get(v)? {
            Val::B(o, k) => Ok((o, k)),
            Val::W(o) => Ok((o, 0)),
            Val::T(t) if t.n == 1 => Ok((r(t), 0)),
            other => Err(unsupported(format!("{other:?} is not an i8 value"))),
        }
    }

    /// Bit 0 of `x` as a boolean: a uniform compare of an SGPR, else a lane mask.
    fn low_bit_mask(&mut self, x: Opnd, uniform: bool) -> Val {
        if let Opnd::Imm(c) = x {
            return Val::M(Opnd::Imm(if c & 1 != 0 { u32::MAX } else { 0 }));
        }
        if uniform && !self.is_v(x) {
            let bit = self.s_bin(Opc::SAndB32, x, Opnd::Imm(1));
            return Val::C(SCmp { op: Opc::SCmpLgU32, a: bit, b: Opnd::Imm(0) });
        }
        let bit = self.v_bin(Opc::VAndB32, Opnd::Imm(1), x);
        let dst = self.mreg();
        self.valu(Opc::VCmp(0x4d), [r(dst)], [bit, Opnd::Imm(0)], 0);
        Val::M(r(dst))
    }

    /// The data operand and opcode storing byte `k` of `o`: `plain` stores
    /// bits 7:0, `hi` (the d16_hi form) bits 23:16.
    fn byte_store(&mut self, o: Opnd, k: u8, plain: Opc, hi: Opc) -> (Opnd, Opc) {
        let o = self.to_v(o);
        match k {
            0 => (o, plain),
            2 => (o, hi),
            1 => (self.v_bin(Opc::VLshrrevB32, Opnd::Imm(8), o), plain),
            _ => (self.v_bin(Opc::VLshrrevB32, Opnd::Imm(8), o), hi),
        }
    }

    /// Byte `k` of `o` zero- or sign-extended to 32 bits.
    fn widen8(&mut self, o: Opnd, k: u8, signed: bool, uniform: bool) -> Opnd {
        let shift = 8 * k as u32;
        if let Opnd::Imm(c) = o {
            let b = (c >> shift) & 0xff;
            return Opnd::Imm(if signed { b as u8 as i8 as i32 as u32 } else { b });
        }
        if uniform && !self.is_v(o) {
            return match (signed, k) {
                (false, 0) => self.s_bin(Opc::SAndB32, o, Opnd::Imm(0xff)),
                (false, 3) => self.s_bin(Opc::SLshrB32, o, Opnd::Imm(24)),
                (false, _) => self.s_bin(Opc::SBfeU32, o, Opnd::Imm(8 << 16 | shift)),
                (true, _) => {
                    let up = if k == 3 { o } else { self.s_bin(Opc::SLshlB32, o, Opnd::Imm(24 - shift)) };
                    self.s_bin(Opc::SAshrI32, up, Opnd::Imm(24))
                }
            };
        }
        match (signed, k) {
            (false, 0) => self.v_bin(Opc::VAndB32, Opnd::Imm(0xff), o),
            (false, 3) => self.v_bin(Opc::VLshrrevB32, Opnd::Imm(24), o),
            (true, 3) => self.v_bin(Opc::VAshrrevI32, Opnd::Imm(24), o),
            (false, _) => self.v3(Opc::VBfeU32, o, Opnd::Imm(shift), Opnd::Imm(8), 0),
            (true, _) => self.v3(Opc::VBfeI32, o, Opnd::Imm(shift), Opnd::Imm(8), 0),
        }
    }

    /// `dst = bytes[0] | bytes[1] << 8 | ...` (missing bytes are zero), each
    /// byte an (operand, byte index): one v_perm_b32 when the bytes come from
    /// at most two registers, else two pairs and their merge.
    fn pack_bytes(&mut self, dst: VRef, bytes: &[(Opnd, u8)]) {
        let bytes: Vec<(Opnd, u8)> = bytes.iter().map(|&(o, k)| (self.to_v(o), k)).collect();
        let mut srcs: Vec<Opnd> = Vec::new();
        for &(o, _) in &bytes {
            if !srcs.contains(&o) {
                srcs.push(o);
            }
        }
        if srcs.len() > 2 {
            let (lo, hi) = (self.vgpr(1), self.vgpr(1));
            self.pack_bytes(lo, &bytes[..2]);
            self.pack_bytes(hi, &bytes[2..]);
            self.valu(Opc::VPermB32, [r(dst)], [r(hi), r(lo), Opnd::Imm(0x0504_0100)], 0);
            return;
        }
        if bytes.len() == 4 && srcs.len() == 1 && bytes.iter().enumerate().all(|(i, &(_, k))| k as usize == i) {
            self.copy(dst, srcs[0]);
            return;
        }
        // v_perm_b32 numbers src1's bytes 0-3 and src0's bytes 4-7; 0x0c is zero.
        let (hi_src, lo_src) = (srcs[0], *srcs.last().expect("at least one byte"));
        let mut sel = 0x0c0c_0c0cu32;
        for (i, &(o, k)) in bytes.iter().enumerate() {
            let at = if o == lo_src { k as u32 } else { 4 + k as u32 };
            sel = sel & !(0xff << (8 * i)) | at << (8 * i);
        }
        self.valu(Opc::VPermB32, [r(dst)], [hi_src, lo_src, Opnd::Imm(sel)], 0);
    }

    /// `fpext bfloat to float` of the bf16 in one half of `src`: its bits
    /// become the f32's high half; a signaling NaN stays signaling.
    fn bf16_to_f32(&mut self, src: Opnd, hi: bool) -> Opnd {
        if hi {
            self.v_bin(Opc::VAndB32, Opnd::Imm(0xffff_0000), src)
        } else {
            self.v_bin(Opc::VLshlrevB32, Opnd::Imm(16), src)
        }
    }

    /// `fptrunc float to bfloat` as LLVM selects it for gfx11, which has no
    /// bf16 convert: `x + 0x7fff + bit 16 of x` rounds to nearest even, and a
    /// NaN becomes `x | 0x400000`. The bf16 is the high half of the result.
    fn f32_to_bf16(&mut self, x: Opnd) -> (Opnd, bool) {
        if let Opnd::Imm(bits) = x {
            return (Opnd::Imm(f32_to_bf16_bits(f32::from_bits(bits)) as u32), false);
        }
        let x = self.to_v(x);
        let lsb = self.v3(Opc::VBfeU32, x, Opnd::Imm(16), Opnd::Imm(1), 0);
        let quiet = self.v_bin(Opc::VOrB32, Opnd::Imm(0x40_0000), x);
        let nan = self.mreg();
        self.valu(Opc::VCmp(0x18), [r(nan)], [x, x], 0);
        let rounded = self.v3(Opc::VAdd3U32, lsb, x, Opnd::Imm(0x7fff), 0);
        (self.v3(Opc::VCndmaskB32, rounded, quiet, r(nan), 0), true)
    }

    /// `tile.mma` as one WMMA: `accumulate` writes D over C when this is C's
    /// last use (the copy coalesces away), `early` a fresh D otherwise.
    fn wmma(&mut self, op: &Op, accumulate: Opc, early: Opc, mods: u32) -> Result<(), CompileError> {
        self.gfx11_only("tile.mma")?;
        let a = self.tuple(op.operands[0])?;
        let b = self.tuple(op.operands[1])?;
        let c = self.tuple(op.operands[2])?;
        let d = self.vgpr(8);
        if self.uses[op.operands[2].0 as usize] == 1 {
            self.copy(d, r(c));
            self.emit(MInst::new(accumulate, [r(d)], [r(a), r(b), r(d)]).with_mods(mods));
        } else {
            self.emit(MInst::new(early, [r(d)], [r(a), r(b), r(c)]).with_mods(mods));
        }
        self.set(op.results[0], Val::T(d));
        Ok(())
    }

    fn soffset(&mut self, v: ValueId) -> Result<Opnd, CompileError> {
        let o = self.w(v)?;
        Ok(match o {
            Opnd::Imm(x) if is_inline_int(x) => o,
            _ => self.to_s(o),
        })
    }

    /// `(voffset, rsrc, soffset)` of a buffer access.
    fn buffer_operands(&mut self, op: &Op) -> Result<Uses, CompileError> {
        let rsrc = self.tuple(op.operands[0])?;
        let voff = self.w(op.operands[1])?;
        let voff = self.to_v(voff);
        let soff = self.soffset(op.operands[2])?;
        Ok([voff, r(rsrc), soff].into())
    }

    /// One element through a bounds-checked buffer: a 16-bit element with
    /// `buffer_load_u16` (or a d16 load into the half a vector insert takes
    /// it to), a 32-bit one with `buffer_load_b32`. An access past the
    /// buffer reads zero element by element, as the reference's i16/i32
    /// intrinsics do.
    fn buffer_load(&mut self, op: &Op) -> Result<(), CompileError> {
        let v = op.results[0];
        let uses = self.buffer_operands(op)?;
        let offset = op.int_attr("offset").unwrap_or(0);
        let mem = super::alias::op_tag(&self.mem_tags, op);
        let (opc, val): (Opc, fn(VRef) -> Val) = match self.k.ty(v) {
            Type::Scalar(s) if s.bytes() == 2 => {
                if self.gathers.contains(&v) {
                    self.deferred.insert(v, Deferred::Buffer(uses, offset, mem));
                    return Ok(());
                }
                (Opc::BufLoadU16, |d| Val::H(r(d), false))
            }
            Type::Scalar(s) if s.bytes() == 4 => (Opc::BufLoadB32, |d| Val::W(r(d))),
            other => return Err(unsupported(format!("scalar buffer load of {other:?}"))),
        };
        let dst = self.vgpr(1);
        self.emit(MInst::new(opc, [r(dst)], uses).with_imm(offset).with_mem(mem));
        self.set(v, val(dst));
        Ok(())
    }

    fn buffer_store(&mut self, op: &Op) -> Result<(), CompileError> {
        let value = op.operands[3];
        let ty = self.k.ty(value).clone();
        let bytes = match &ty {
            Type::Vec(e, n) => e.bytes() * n,
            other => other.elem().bytes(),
        };
        let (opc, data) = match bytes {
            2 => {
                let (o, hi) = self.half(value)?;
                (if hi { Opc::BufStoreD16HiB16 } else { Opc::BufStoreB16 }, o)
            }
            4 => (Opc::BufStoreB32, self.w(value)?),
            _ => return Err(unsupported(format!("scalar buffer store of {ty:?}"))),
        };
        let data = self.to_v(data);
        let at = self.buffer_operands(op)?;
        let mem = super::alias::op_tag(&self.mem_tags, op);
        self.emit(MInst::new(opc, [], [data, at[0], at[1], at[2]]).with_imm(op.int_attr("offset").unwrap_or(0)).with_mem(mem));
        Ok(())
    }

    /// `ds_add_u32` / `ds_add_f32`, the returning form when the old value is used.
    fn lds_atomic_add(&mut self, op: &Op) -> Result<(), CompileError> {
        let v = op.results[0];
        let float = match self.k.ty(v) {
            Type::Scalar(Scalar::I32) => false,
            Type::Scalar(Scalar::F32) => true,
            other => return Err(unsupported(format!("LDS atomic add of {other:?}"))),
        };
        let (addr, offset) = self.lds_addr(op.operands[0], op.operands[1], op.int_attr("offset").unwrap_or(0))?;
        let data = self.w(op.operands[2])?;
        let data = self.to_v(data);
        if self.uses[v.0 as usize] == 0 {
            let opc = if float { Opc::DsAddF32 } else { Opc::DsAddU32 };
            self.emit(MInst::new(opc, [], [addr, data]).with_imm(offset));
        } else {
            let opc = if float { Opc::DsAddRtnF32 } else { Opc::DsAddRtnU32 };
            let dst = self.vgpr(1);
            self.emit(MInst::new(opc, [r(dst)], [addr, data]).with_imm(offset));
            self.set(v, Val::W(r(dst)));
        }
        Ok(())
    }

    fn sign_hi(&mut self, lo: Opnd, v: ValueId) -> Opnd {
        if self.facts.nonneg(v) {
            return Opnd::Imm(0);
        }
        match lo {
            Opnd::Imm(x) => Opnd::Imm(((x as i32) >> 31) as u32),
            _ if self.is_v(lo) => self.v_bin(Opc::VAshrrevI32, Opnd::Imm(31), lo),
            _ => self.s_bin(Opc::SAshrI32, lo, Opnd::Imm(31)),
        }
    }

    fn int_op(&mut self, op: &Op) -> Result<(), CompileError> {
        let v = op.results[0];
        let ty = self.k.ty(v).clone();
        let uniform = self.uni(v);
        if let Some(&(lo, hi)) = self.chains.packs.get(&v) {
            let (lo, hi) = (self.half(lo)?, self.half(hi)?);
            let dst = self.vgpr(1);
            self.pack_halves(dst, lo, hi);
            self.set(v, Val::W(r(dst)));
            return Ok(());
        }
        if matches!(ty, Type::Scalar(s) if s == Scalar::I16 || s.is_byte()) {
            return Err(unsupported(format!("{ty:?} {:?}", op.kind)));
        }
        match ty {
            Type::Scalar(Scalar::I1) if self.chains.finite.contains_key(&v) => {
                let x = self.w(self.chains.finite[&v])?;
                let dst = self.mreg();
                // Negative and positive normals, denormals and zeros.
                self.valu(Opc::VCmp(VCMP_CLASS_F32), [r(dst)], [x, Opnd::Imm(0x1f8)], 0);
                self.set(v, Val::M(r(dst)));
            }
            Type::Scalar(Scalar::I1) => {
                let a = self.mask(op.operands[0])?;
                let b = self.mask(op.operands[1])?;
                let opc = match op.kind {
                    OpKind::And => Opc::SAndB32,
                    OpKind::Or => Opc::SOrB32,
                    OpKind::Xor => Opc::SXorB32,
                    other => return Err(unsupported(format!("i1 {other:?}"))),
                };
                let out = self.m_bin(opc, a, b);
                self.set(v, Val::M(out));
            }
            Type::Scalar(Scalar::I64) => {
                let a = self.d(op.operands[0])?;
                let out = match op.kind {
                    OpKind::Add => {
                        let b = self.d(op.operands[1])?;
                        self.add64(a, b, uniform, false)
                    }
                    OpKind::Sub => {
                        let b = self.d(op.operands[1])?;
                        self.add64(a, b, uniform, true)
                    }
                    OpKind::Mul => {
                        let b = self.d(op.operands[1])?;
                        self.mul64(a, b, uniform)
                    }
                    OpKind::Shl => {
                        let k = self
                            .facts
                            .int(op.operands[1])
                            .ok_or_else(|| unsupported("64-bit shift by a runtime amount"))?;
                        self.shl64(op.operands[0], a, (k & 63) as u32, uniform, v)
                    }
                    OpKind::And | OpKind::Or | OpKind::Xor => {
                        let b = self.d(op.operands[1])?;
                        let lo = self.int_bin(op.kind, a.0, b.0, uniform, false)?;
                        let hi = self.int_bin(op.kind, a.1, b.1, uniform, false)?;
                        (lo, hi)
                    }
                    other => return Err(unsupported(format!("i64 {other:?}"))),
                };
                self.set(v, Val::D(out.0, out.1));
            }
            _ => {
                let a = self.w(op.operands[0])?;
                let b = self.w(op.operands[1])?;
                let small = self.range_lt(op.operands[0], 1 << 24) && self.range_lt(op.operands[1], 1 << 24);
                let out = self.int_bin(op.kind, a, b, uniform, small)?;
                self.set(v, Val::W(out));
            }
        }
        Ok(())
    }

    fn div_op(&mut self, op: &Op) -> Result<(), CompileError> {
        let v = op.results[0];
        if *self.k.ty(v) == Type::Scalar(Scalar::I64) {
            return self.div64_op(op);
        }
        if *self.k.ty(v) != Type::Scalar(Scalar::I32) {
            return Err(unsupported("non-i32 division"));
        }
        let x = self.w(op.operands[0])?;
        let y = self.w(op.operands[1])?;
        let q = self.sdiv32(x, y, self.facts.nonneg(op.operands[0]), self.facts.nonneg(op.operands[1]));
        let out = if self.uni(v) { self.to_s(q) } else { q };
        self.set(v, Val::W(out));
        Ok(())
    }

    /// 64-bit signed `arith.div` / `arith.mod`, as LLVM lowers them: one
    /// 32-bit division when both operands fit in 32 bits, shifts or a magic
    /// multiply for a constant divisor, and otherwise the absolute values
    /// through `AMDGPUTargetLowering::LowerUDIVREM64` (a float reciprocal
    /// estimate, two Newton steps, up to two corrections) with the signs
    /// applied after. Uniform operands keep the integer work on the SALU and
    /// read the estimate back from the VALU, which has the only reciprocal.
    fn div64_op(&mut self, op: &Op) -> Result<(), CompileError> {
        let v = op.results[0];
        let (x, y) = (op.operands[0], op.operands[1]);
        let rem = op.kind == OpKind::Mod;
        let uniform = self.uni(v);
        let (mut a, mut b) = (self.d(x)?, self.d(y)?);
        if uniform {
            a = (self.first_lane(a.0), self.first_lane(a.1));
            b = (self.first_lane(b.0), self.first_lane(b.1));
        }
        // i32::MIN / -1 does not fit in 32 bits; everything else in range does.
        let range = |v: ValueId| self.facts.range_of(v).filter(|r| r.hi <= i32::MAX as i64);
        if range(x).is_some_and(|r| r.lo > i32::MIN as i64) && range(y).is_some_and(|r| r.lo >= i32::MIN as i64) {
            let q = self.sdiv32(a.0, b.0, self.facts.nonneg(x), self.facts.nonneg(y));
            let out = if rem {
                let p = self.v_bin(Opc::VMulLoU32, q, b.0);
                self.v_bin(Opc::VSubNcU32, a.0, p)
            } else {
                q
            };
            let out = if uniform { self.to_s(out) } else { out };
            let hi = self.sign_hi(out, v);
            self.set(v, Val::D(out, hi));
            return Ok(());
        }
        let nonneg = self.facts.nonneg(x);
        if let Some(out) = self.facts.int(y).and_then(|d| self.div64_by_const(a, d, nonneg, uniform, rem)) {
            self.set(v, Val::D(out.0, out.1));
            return Ok(());
        }
        let (ua, sa) = self.abs64(a, nonneg, uniform);
        let (ub, sb) = self.abs64(b, self.facts.nonneg(y), uniform);
        let mag = self.udivrem64(ua, ub, uniform, rem);
        // The remainder takes the dividend's sign, the quotient the product's.
        let sign = match (sa, sb, rem) {
            (s, _, true) => s,
            (Some(p), Some(q), false) => Some(if uniform { self.s_bin(Opc::SXorB32, p, q) } else { self.v_bin(Opc::VXorB32, p, q) }),
            (s, t, false) => s.or(t),
        };
        let out = match sign {
            Some(s) => self.negate_if64(mag, s, uniform),
            None => mag,
        };
        self.set(v, Val::D(out.0, out.1));
        Ok(())
    }

    /// `(|x|, sign word)` of a 64-bit value; no sign word when `x >= 0` is known.
    fn abs64(&mut self, x: (Opnd, Opnd), nonneg: bool, uniform: bool) -> ((Opnd, Opnd), Option<Opnd>) {
        if nonneg {
            return (x, None);
        }
        let s = self.sign_word(x.1, uniform);
        (self.negate_if64(x, s, uniform), Some(s))
    }

    /// `(x ^ s) - s` per 64 bits for a sign word `s` (0 or -1): `-x` when `s` is -1.
    fn negate_if64(&mut self, x: (Opnd, Opnd), s: Opnd, uniform: bool) -> (Opnd, Opnd) {
        let lo = self.int_bin(OpKind::Xor, x.0, s, uniform, false).expect("xor has a lowering");
        let hi = self.int_bin(OpKind::Xor, x.1, s, uniform, false).expect("xor has a lowering");
        self.add64((lo, hi), (s, s), uniform, true)
    }

    /// 0 or -1: the sign of the 32-bit `hi`.
    fn sign_word(&mut self, hi: Opnd, uniform: bool) -> Opnd {
        match hi {
            Opnd::Imm(c) => Opnd::Imm(((c as i32) >> 31) as u32),
            _ if uniform => self.s_bin(Opc::SAshrI32, hi, Opnd::Imm(31)),
            _ => self.v_bin(Opc::VAshrrevI32, Opnd::Imm(31), hi),
        }
    }

    /// Unsigned `n / d` (or `n % d` with `rem`) of 64-bit values, as
    /// `AMDGPUTargetLowering::LowerUDIVREM64` expands it (Rodeheffer, "Software
    /// Integer Division"): the estimate is a lower bound within two of the
    /// quotient, and each correction step subtracts `d` once more.
    fn udivrem64(&mut self, n: (Opnd, Opnd), d: (Opnd, Opnd), uniform: bool, rem: bool) -> (Opnd, Opnd) {
        let fl = self.v1(Opc::VCvtF32U32, d.0);
        let fh = self.v1(Opc::VCvtF32U32, d.1);
        let mad1 = self.v3(Opc::VFmaF32, fh, Opnd::Imm(0x4f80_0000), fl, 0);
        let rcp = self.v1(Opc::VRcpF32, mad1);
        let mul1 = self.v_bin(Opc::VMulF32, Opnd::Imm(0x5f7f_fffc), rcp);
        let mul2 = self.v_bin(Opc::VMulF32, Opnd::Imm(0x2f80_0000), mul1);
        let trunc = self.v1(Opc::VTruncF32, mul2);
        let mad2 = self.v3(Opc::VFmaF32, trunc, Opnd::Imm(0xcf80_0000), mul1, 0);
        let mut est = (self.v1(Opc::VCvtU32F32, mad2), self.v1(Opc::VCvtU32F32, trunc));
        if uniform {
            est = (self.first_lane(est.0), self.first_lane(est.1));
        }
        let zero = (Opnd::Imm(0), Opnd::Imm(0));
        let neg_d = self.add64(zero, d, uniform, true);
        let lo1 = self.mul64(neg_d, est, uniform);
        let hi1 = self.mulhu64(est, lo1, uniform);
        let add1 = self.add64(est, hi1, uniform, false);
        let lo2 = self.mul64(neg_d, add1, uniform);
        let hi2 = self.mulhu64(add1, lo2, uniform);
        let add2 = self.add64(add1, hi2, uniform, false);
        let q0 = self.mulhu64(n, add2, uniform);
        let p = self.mul64(d, q0, uniform);
        let r1 = self.add64(n, p, uniform, true);
        let (r2, r1_below) = self.sub64_borrow(r1, d, uniform);
        let (r3, r2_below) = self.sub64_borrow(r2, d, uniform);
        if rem {
            let inner = self.select64(r2_below, r2, r3, uniform);
            return self.select64(r1_below, r1, inner, uniform);
        }
        let one = (Opnd::Imm(1), Opnd::Imm(0));
        let q1 = self.add64(q0, one, uniform, false);
        let q2 = self.add64(q1, one, uniform, false);
        let inner = self.select64(r2_below, q1, q2, uniform);
        self.select64(r1_below, q0, inner, uniform)
    }

    /// High 64 bits of the 128-bit product of two 64-bit values.
    fn mulhu64(&mut self, a: (Opnd, Opnd), b: (Opnd, Opnd), uniform: bool) -> (Opnd, Opnd) {
        let zero = Opnd::Imm(0);
        // t = hi(a0 b0) + a0 b1, u = lo(t) + a1 b0, result = a1 b1 + hi(t) + hi(u).
        if uniform {
            let hi00 = self.s_bin(Opc::SMulHiU32, a.0, b.0);
            let p01 = (self.s_bin(Opc::SMulI32, a.0, b.1), self.s_bin(Opc::SMulHiU32, a.0, b.1));
            let p10 = (self.s_bin(Opc::SMulI32, a.1, b.0), self.s_bin(Opc::SMulHiU32, a.1, b.0));
            let p11 = (self.s_bin(Opc::SMulI32, a.1, b.1), self.s_bin(Opc::SMulHiU32, a.1, b.1));
            let t = self.add64((hi00, zero), p01, true, false);
            let u = self.add64((t.0, zero), p10, true, false);
            let w = self.add64(p11, (t.1, zero), true, false);
            return self.add64(w, (u.1, zero), true, false);
        }
        let hi00 = self.v_bin(Opc::VMulHiU32, a.0, b.0);
        let t = self.mad64(a.0, b.1, (hi00, zero));
        let u = self.mad64(a.1, b.0, (t.0, zero));
        let w = self.mad64(a.1, b.1, (t.1, zero));
        self.add64(w, (u.1, zero), false, false)
    }

    /// `a * b + c` with 32-bit `a`, `b` and 64-bit `c` (`v_mad_u64_u32`).
    fn mad64(&mut self, a: Opnd, b: Opnd, c: (Opnd, Opnd)) -> (Opnd, Opnd) {
        let c = self.wide_operand(c.0, c.1);
        let dst = self.vgpr(2);
        self.valu(Opc::VMadU64U32, [r(dst), NULL], [a, b, c], 0);
        (r(sub(dst, 0, 1)), r(sub(dst, 1, 1)))
    }

    /// `a - b` and whether it borrowed (`a < b` unsigned): 0 or 1 in an SGPR
    /// for uniform operands, a lane mask otherwise.
    fn sub64_borrow(&mut self, a: (Opnd, Opnd), b: (Opnd, Opnd), uniform: bool) -> ((Opnd, Opnd), Opnd) {
        let borrow = if uniform { self.sreg(1) } else { self.mreg() };
        if uniform {
            let dst = self.sreg(2);
            self.salu(Opc::SSubU32, [r(sub(dst, 0, 1))], [a.0, b.0]);
            self.salu(Opc::SSubbU32, [r(sub(dst, 1, 1))], [a.1, b.1]);
            self.salu(Opc::SCselectB32, [r(borrow)], [Opnd::Imm(1), Opnd::Imm(0)]);
            return ((r(sub(dst, 0, 1)), r(sub(dst, 1, 1))), r(borrow));
        }
        let dst = self.vgpr(2);
        let carry = self.mreg();
        self.valu(Opc::VSubCoU32, [r(sub(dst, 0, 1)), r(carry)], [a.0, b.0], 0);
        self.valu(Opc::VSubCoCiU32, [r(sub(dst, 1, 1)), r(borrow)], [a.1, b.1, r(carry)], 0);
        ((r(sub(dst, 0, 1)), r(sub(dst, 1, 1))), r(borrow))
    }

    /// `cond ? t : f` for 64-bit values, `cond` as `sub64_borrow` gives it.
    fn select64(&mut self, cond: Opnd, t: (Opnd, Opnd), f: (Opnd, Opnd), uniform: bool) -> (Opnd, Opnd) {
        if uniform {
            let (x, y) = (self.wide_operand(t.0, t.1), self.wide_operand(f.0, f.1));
            self.salu(Opc::SCmpLgU32, [], [cond, Opnd::Imm(0)]);
            let dst = self.sreg(2);
            self.salu(Opc::SCselectB64, [r(dst)], [x, y]);
            return (r(sub(dst, 0, 1)), r(sub(dst, 1, 1)));
        }
        (self.v3(Opc::VCndmaskB32, f.0, t.0, cond, 0), self.v3(Opc::VCndmaskB32, f.1, t.1, cond, 0))
    }

    /// `x >> k` for a 64-bit value and a constant `k < 64`, arithmetic or logical.
    fn shr64(&mut self, x: (Opnd, Opnd), k: u32, arith: bool, uniform: bool) -> (Opnd, Opnd) {
        if k == 0 {
            return x;
        }
        if k >= 32 {
            let lo = match (k, uniform, arith) {
                (32, _, _) => x.1,
                (_, true, true) => self.s_bin(Opc::SAshrI32, x.1, Opnd::Imm(k - 32)),
                (_, true, false) => self.s_bin(Opc::SLshrB32, x.1, Opnd::Imm(k - 32)),
                (_, false, true) => self.v_bin(Opc::VAshrrevI32, Opnd::Imm(k - 32), x.1),
                (_, false, false) => self.v_bin(Opc::VLshrrevB32, Opnd::Imm(k - 32), x.1),
            };
            let hi = if arith { self.sign_word(x.1, uniform) } else { Opnd::Imm(0) };
            return (lo, hi);
        }
        let dst;
        if uniform {
            let src = self.pair_s(x.0, x.1);
            dst = self.sreg(2);
            self.salu(if arith { Opc::SAshrI64 } else { Opc::SLshrB64 }, [r(dst)], [r(src), Opnd::Imm(k)]);
        } else {
            let src = self.wide_operand(x.0, x.1);
            dst = self.vgpr(2);
            self.valu(if arith { Opc::VAshrrevI64 } else { Opc::VLshrrevB64 }, [r(dst)], [Opnd::Imm(k), src], 0);
        }
        (r(sub(dst, 0, 1)), r(sub(dst, 1, 1)))
    }

    /// `x / d` or `x % d` (`x` 64-bit, `d` a constant) as shifts or a multiply
    /// by a magic number (Hacker's Delight 10-1); `None` leaves `d` to the
    /// general expansion.
    fn div64_by_const(&mut self, a: (Opnd, Opnd), d: i64, nonneg: bool, uniform: bool, rem: bool) -> Option<(Opnd, Opnd)> {
        let zero = (Opnd::Imm(0), Opnd::Imm(0));
        let split = |c: u64| (Opnd::Imm(c as u32), Opnd::Imm((c >> 32) as u32));
        if d == 1 || d == -1 {
            return Some(match (rem, d) {
                (true, _) => zero,
                (false, 1) => a,
                _ => self.add64(zero, a, uniform, true),
            });
        }
        if d < 2 {
            return None;
        }
        let and64 = |s: &mut Self, x: (Opnd, Opnd), m: u64| {
            let lo = s.int_bin(OpKind::And, x.0, Opnd::Imm(m as u32), uniform, false).expect("and has a lowering");
            let hi = s.int_bin(OpKind::And, x.1, Opnd::Imm((m >> 32) as u32), uniform, false).expect("and has a lowering");
            (lo, hi)
        };
        let q = if (d as u64).is_power_of_two() {
            let k = d.trailing_zeros();
            if nonneg && rem {
                return Some(and64(self, a, d as u64 - 1));
            }
            if nonneg {
                self.shr64(a, k, false, uniform)
            } else {
                // Round toward zero: a negative dividend gets d - 1 added first.
                let s = self.sign_word(a.1, uniform);
                let bias = and64(self, (s, s), d as u64 - 1);
                let t = self.add64(a, bias, uniform, false);
                self.shr64(t, k, true, uniform)
            }
        } else {
            let (m, shift) = signed_magic64(d);
            let mut q = self.mulhu64(a, split(m), uniform);
            if !nonneg {
                // mulhs(x, M), plus x when M reads as negative, is this.
                let s = self.sign_word(a.1, uniform);
                let fix = and64(self, (s, s), m);
                q = self.add64(q, fix, uniform, true);
            }
            q = self.shr64(q, shift, true, uniform);
            if !nonneg {
                // Round toward zero: add one to a negative quotient.
                let up = match q.1 {
                    Opnd::Imm(c) => Opnd::Imm(c >> 31),
                    hi if uniform => self.s_bin(Opc::SLshrB32, hi, Opnd::Imm(31)),
                    hi => self.v_bin(Opc::VLshrrevB32, Opnd::Imm(31), hi),
                };
                q = self.add64(q, (up, Opnd::Imm(0)), uniform, false);
            }
            q
        };
        if !rem {
            return Some(q);
        }
        let p = self.mul64(q, split(d as u64), uniform);
        Some(self.add64(a, p, uniform, true))
    }

    fn v1(&mut self, op: Opc, a: Opnd) -> Opnd {
        let dst = self.vgpr(1);
        self.valu(op, [r(dst)], [a], 0);
        r(dst)
    }

    fn cmp_op(&mut self, op: &Op) -> Result<(), CompileError> {
        let v = op.results[0];
        let pred = op.str_attr("pred").unwrap_or("lt");
        let ty = self.k.ty(op.operands[0]).clone();
        if matches!(ty, Type::Scalar(s) if s == Scalar::I16 || s.is_byte()) {
            return Err(unsupported(format!("{ty:?} compare")));
        }
        let uniform = self.uni(v);
        let wide = ty == Type::Scalar(Scalar::I64);
        if uniform && !wide {
            let opc = match pred {
                "eq" => Opc::SCmpEqI32,
                "ne" => Opc::SCmpLgI32,
                "lt" => Opc::SCmpLtI32,
                "le" => Opc::SCmpLeI32,
                "gt" => Opc::SCmpGtI32,
                "ge" => Opc::SCmpGeI32,
                "ult" => Opc::SCmpLtU32,
                other => return Err(unsupported(format!("cmp predicate {other}"))),
            };
            let a = self.w(op.operands[0])?;
            let b = self.w(op.operands[1])?;
            self.set(v, Val::C(SCmp { op: opc, a, b }));
            return Ok(());
        }
        if uniform && wide && matches!(pred, "eq" | "ne") {
            let (al, ah) = self.d(op.operands[0])?;
            let (bl, bh) = self.d(op.operands[1])?;
            let a = self.wide_operand(al, ah);
            let b = self.wide_operand(bl, bh);
            let opc = if pred == "eq" { Opc::SCmpEqU64 } else { Opc::SCmpLgU64 };
            self.set(v, Val::C(SCmp { op: opc, a, b }));
            return Ok(());
        }
        // The sign of a 64-bit value is the sign of its high word.
        if uniform && wide && matches!(pred, "lt" | "ge") && self.facts.int(op.operands[1]) == Some(0) {
            let (_, ah) = self.d(op.operands[0])?;
            if !self.is_v(ah) {
                let opc = if pred == "lt" { Opc::SCmpLtI32 } else { Opc::SCmpGeI32 };
                self.set(v, Val::C(SCmp { op: opc, a: ah, b: Opnd::Imm(0) }));
                return Ok(());
            }
        }
        let code = match (pred, wide) {
            ("lt", false) => 0x41,
            ("eq", false) => 0x42,
            ("le", false) => 0x43,
            ("gt", false) => 0x44,
            ("ne", false) => 0x45,
            ("ge", false) => 0x46,
            ("ult", false) => 0x49,
            ("lt", true) => 0x51,
            ("eq", true) => 0x52,
            ("le", true) => 0x53,
            ("gt", true) => 0x54,
            ("ne", true) => 0x55,
            ("ge", true) => 0x56,
            ("ult", true) => 0x59,
            (other, _) => return Err(unsupported(format!("cmp predicate {other}"))),
        };
        let (a, b) = if wide {
            let (al, ah) = self.d(op.operands[0])?;
            let (bl, bh) = self.d(op.operands[1])?;
            (self.wide_operand(al, ah), self.wide_operand(bl, bh))
        } else {
            (self.w(op.operands[0])?, self.w(op.operands[1])?)
        };
        let dst = self.mreg();
        self.valu(Opc::VCmp(code), [r(dst)], [a, b], 0);
        self.set(v, Val::M(r(dst)));
        Ok(())
    }

    fn select_op(&mut self, op: &Op) -> Result<(), CompileError> {
        let v = op.results[0];
        let ty = self.k.ty(v).clone();
        let (c, a, b) = (op.operands[0], op.operands[1], op.operands[2]);
        if ty == Type::Scalar(Scalar::I1) {
            let mc = self.mask(c)?;
            let ma = self.mask(a)?;
            let mb = self.mask(b)?;
            let t = self.m_bin(Opc::SAndB32, mc, ma);
            let f = self.m_bin(Opc::SAndNot1B32, mb, mc);
            let out = self.m_bin(Opc::SOrB32, t, f);
            self.set(v, Val::M(out));
            return Ok(());
        }
        let uniform = self.uni(v);
        let pieces: Vec<(Opnd, Opnd)>;
        let shape: u8;
        match self.get(a)? {
            Val::D(..) => {
                let (al, ah) = self.d(a)?;
                let (bl, bh) = self.d(b)?;
                pieces = vec![(al, bl), (ah, bh)];
                shape = 2;
            }
            Val::T(_) => {
                let ta = self.tuple(a)?;
                let tb = self.tuple(b)?;
                pieces = (0..ta.n).map(|i| (r(sub(ta, i, 1)), r(sub(tb, i, 1)))).collect();
                shape = 3;
            }
            Val::H(..) => {
                let (oa, ha) = self.half(a)?;
                let (ob, hb) = self.half(b)?;
                if ha == hb {
                    pieces = vec![(oa, ob)];
                    shape = if ha { 5 } else { 4 };
                } else {
                    let la = self.w(a)?;
                    let lb = self.w(b)?;
                    pieces = vec![(la, lb)];
                    shape = 4;
                }
            }
            _ => {
                pieces = vec![(self.w(a)?, self.w(b)?)];
                shape = 1;
            }
        }
        let mut outs = Vec::new();
        if uniform && shape == 2 && pieces.iter().all(|&(x, y)| !self.is_v(x) && !self.is_v(y)) {
            let x = self.wide_operand(pieces[0].0, pieces[1].0);
            let y = self.wide_operand(pieces[0].1, pieces[1].1);
            self.scc(c)?;
            let dst = self.sreg(2);
            self.salu(Opc::SCselectB64, [r(dst)], [x, y]);
            self.set(v, Val::D(r(sub(dst, 0, 1)), r(sub(dst, 1, 1))));
            return Ok(());
        }
        if uniform {
            self.scc(c)?;
            for &(x, y) in &pieces {
                outs.push(self.s_bin(Opc::SCselectB32, x, y));
            }
        } else {
            let m = self.mask(c)?;
            for &(x, y) in &pieces {
                outs.push(self.select_lanes(y, x, m));
            }
        }
        let val = match shape {
            1 => Val::W(outs[0]),
            2 => {
                let lo = outs[0];
                let hi = outs[1];
                Val::D(lo, hi)
            }
            3 => {
                let dst = self.vgpr(outs.len() as u8);
                for (i, o) in outs.iter().enumerate() {
                    self.copy(sub(dst, i as u8, 1), *o);
                }
                Val::T(dst)
            }
            4 => Val::H(outs[0], false),
            _ => Val::H(outs[0], true),
        };
        self.set(v, val);
        Ok(())
    }

    /// `m ? t : f` in each lane. v_cndmask_b32 reads its lane mask from a
    /// scalar register only: an inline constant there is no mask, and the
    /// hardware selects garbage. A condition known for every lane (`mask`
    /// gives it as 0 or all ones) takes the operand it selects.
    fn select_lanes(&mut self, f: Opnd, t: Opnd, m: Opnd) -> Opnd {
        match m {
            Opnd::Imm(0) => f,
            Opnd::Imm(u32::MAX) => t,
            _ => self.v3(Opc::VCndmaskB32, f, t, m, 0),
        }
    }

    fn convert_op(&mut self, op: &Op) -> Result<(), CompileError> {
        let v = op.results[0];
        let src = op.operands[0];
        let from = self.k.ty(src).clone();
        let to = self.k.ty(v).clone();
        let val = match op.kind {
            OpKind::Zext | OpKind::Sext if from == Type::Scalar(Scalar::I1) => {
                let m = self.mask(src)?;
                let lo = self.bool_to_int(m, self.uni(v));
                let lo = if op.kind == OpKind::Sext {
                    if self.uni(v) {
                        self.s_bin(Opc::SSubI32, Opnd::Imm(0), lo)
                    } else {
                        self.v_bin(Opc::VSubNcU32, Opnd::Imm(0), lo)
                    }
                } else {
                    lo
                };
                if to == Type::Scalar(Scalar::I64) {
                    let hi = if op.kind == OpKind::Sext { lo } else { Opnd::Imm(0) };
                    Val::D(lo, hi)
                } else {
                    Val::W(lo)
                }
            }
            OpKind::Zext | OpKind::Sext if from == Type::Scalar(Scalar::I16) => {
                let (o, hi) = self.half(src)?;
                let signed = op.kind == OpKind::Sext;
                let lo = self.widen16(o, hi, signed, self.uni(v));
                if to == Type::Scalar(Scalar::I64) {
                    let hi_word = if signed { self.sign_hi(lo, v) } else { Opnd::Imm(0) };
                    Val::D(lo, hi_word)
                } else {
                    Val::W(lo)
                }
            }
            OpKind::Trunc if to == Type::Scalar(Scalar::I16) => match self.get(src)? {
                Val::D(lo, _) => Val::H(lo, false),
                _ => Val::H(self.w(src)?, false),
            },
            OpKind::Zext | OpKind::Sext if matches!(from, Type::Scalar(s) if s.is_byte()) => {
                let (o, k) = self.byte(src)?;
                let signed = op.kind == OpKind::Sext;
                let lo = self.widen8(o, k, signed, self.uni(v));
                if to == Type::Scalar(Scalar::I64) {
                    let hi_word = if signed { self.sign_hi(lo, v) } else { Opnd::Imm(0) };
                    Val::D(lo, hi_word)
                } else {
                    Val::W(lo)
                }
            }
            OpKind::Trunc if to == Type::Scalar(Scalar::I1) => {
                let x = match self.get(src)? {
                    Val::D(lo, _) => lo,
                    _ => self.w(src)?,
                };
                self.low_bit_mask(x, self.uni(v))
            }
            OpKind::Trunc if to == Type::Scalar(Scalar::I8) => match self.get(src)? {
                Val::D(lo, _) => Val::B(lo, 0),
                Val::H(o, hi) => Val::B(o, if hi { 2 } else { 0 }),
                _ => Val::B(self.w(src)?, 0),
            },
            OpKind::Zext => {
                let lo = self.w(src)?;
                Val::D(lo, Opnd::Imm(0))
            }
            OpKind::Sext => {
                let lo = self.w(src)?;
                let hi = self.sign_hi(lo, src);
                Val::D(lo, hi)
            }
            OpKind::Trunc => {
                let (lo, _) = self.d(src)?;
                Val::W(lo)
            }
            _ => {
                // bitcast between same-width scalars
                match self.get(src)? {
                    Val::H(o, hi) => Val::H(o, hi),
                    Val::W(o) => Val::W(o),
                    other => other,
                }
            }
        };
        self.set(v, val);
        Ok(())
    }

    /// Address operands for a global access: a VGPR pair, or (zero VGPR
    /// offset, SGPR base) in saddr form.
    fn vmem_addr(&mut self, lo: Opnd, hi: Opnd) -> (Uses, u32) {
        if !self.is_v(lo) && !self.is_v(hi) {
            let base = self.pair_s(lo, hi);
            let zero = match self.zero_v {
                Some(z) => z,
                None => {
                    let z = self.vgpr(1);
                    self.zero_v = Some(z);
                    z
                }
            };
            return ([r(zero), r(base)].into(), M_SADDR);
        }
        let pair = self.pair_v(lo, hi);
        ([r(pair)].into(), 0)
    }

    /// Address operands of a legalized global access: `[addr]` or
    /// `[sbase, voffset]` (see `opt::Legalizer::global_address`).
    fn global_addr(&mut self, operands: &[ValueId]) -> Result<(Uses, u32), CompileError> {
        let (lo, hi) = self.d(operands[0])?;
        let [_, voffset] = operands else {
            return Ok(self.vmem_addr(lo, hi));
        };
        let off = self.w(*voffset)?;
        if self.is_v(lo) || self.is_v(hi) {
            let (lo, hi) = self.add64((lo, hi), (off, Opnd::Imm(0)), false, false);
            return Ok(self.vmem_addr(lo, hi));
        }
        let base = self.pair_s(lo, hi);
        let off = self.to_v(off);
        Ok(([off, r(base)].into(), M_SADDR))
    }

    fn gload(&mut self, op: &Op) -> Result<(), CompileError> {
        let v = op.results[0];
        let ty = self.k.ty(v).clone();
        let mut offset = op.int_attr("offset").unwrap_or(0);
        let align = op.int_attr("align").unwrap_or(1);
        let bytes = match &ty {
            Type::Vec(e, n) => e.bytes() * n,
            other => other.elem().bytes(),
        };
        if self.uni(v) {
            let (lo, hi) = self.d(op.operands[0])?;
            let dwords = bytes.div_ceil(4) as u8;
            let (opc, n) = match dwords {
                1 => (Opc::SLoadB32, 1),
                2 => (Opc::SLoadB64, 2),
                3 | 4 => (Opc::SLoadB128, 4),
                5..=8 => (Opc::SLoadB256, 8),
                _ => (Opc::SLoadB512, 16),
            };
            let mut base = (lo, hi);
            // The `[sbase, voffset]` form (see `global_addr`): the offset is
            // unsigned.
            if let [_, voffset] = op.operands[..] {
                let off = self.w(voffset)?;
                let off = if self.is_v(off) { self.to_s(off) } else { off };
                base = self.add64(base, (off, Opnd::Imm(0)), true, false);
            }
            if !(-(1 << 20)..(1 << 20)).contains(&offset) {
                base = self.add64(base, (Opnd::Imm(offset as u32), Opnd::Imm((offset >> 32) as u32)), true, false);
                offset = 0;
            }
            let sbase = self.pair_s(base.0, base.1);
            let dst = self.sreg(n);
            self.emit(MInst::new(opc, [r(dst)], [r(sbase)]).with_imm(offset));
            let val = match ty {
                Type::Scalar(Scalar::I64) => Val::D(r(sub(dst, 0, 1)), r(sub(dst, 1, 1))),
                Type::Scalar(_) => Val::W(r(sub(dst, 0, 1))),
                _ => Val::T(sub(dst, 0, dwords)),
            };
            self.set(v, val);
            return Ok(());
        }
        let (addr, mods) = self.global_addr(&op.operands)?;
        let mem = super::alias::op_tag(&self.mem_tags, op);
        if self.gathers.contains(&v) {
            self.deferred.insert(v, Deferred::Global(addr, mods, offset, mem));
            return Ok(());
        }
        let dwords = bytes.div_ceil(4) as u8;
        let dst = self.vgpr(dwords.max(1));
        if bytes == 1 {
            self.emit(MInst::new(Opc::GLoadU8, [r(dst)], addr).with_imm(offset).with_mods(mods).with_mem(mem));
            if ty == Type::Scalar(Scalar::I1) {
                // An i1 in memory is a byte whose low bit is the value.
                let mask = self.low_bit_mask(r(dst), false);
                self.set(v, mask);
            } else {
                self.set(v, Val::B(r(dst), 0));
            }
            return Ok(());
        }
        // HSA code objects run with unaligned global access enabled, so
        // dword-multiple accesses need no alignment (LLVM selects them the
        // same way); only the odd f16 tail goes out as a 16-bit access.
        let full = (bytes / 4) as u8;
        if bytes % 2 != 0 || (bytes % 4 == 2 && bytes != 2 && align < 2) {
            return Err(unsupported(format!("global load of {bytes} bytes at align {align}")));
        }
        let mut done = 0u8;
        while done < full {
            let (opc, n) = match full - done {
                1 => (Opc::GLoadB32, 1),
                2 => (Opc::GLoadB64, 2),
                3 => (Opc::GLoadB96, 3),
                _ => (Opc::GLoadB128, 4),
            };
            self.emit(
                MInst::new(opc, [r(sub(dst, done, n))], addr.clone())
                    .with_imm(offset + 4 * done as i64)
                    .with_mods(mods)
                    .with_mem(mem),
            );
            done += n;
        }
        if bytes % 4 == 2 {
            self.emit(
                MInst::new(Opc::GLoadU16, [r(sub(dst, full, 1))], addr)
                    .with_imm(offset + 4 * full as i64)
                    .with_mods(mods)
                    .with_mem(mem),
            );
        }
        let val = match ty {
            Type::Scalar(s) if s.is_half() => Val::H(r(dst), false),
            Type::Scalar(Scalar::I64 | Scalar::F64) | Type::Ptr(_) => Val::D(r(sub(dst, 0, 1)), r(sub(dst, 1, 1))),
            Type::Scalar(_) => Val::W(r(dst)),
            _ => Val::T(dst),
        };
        self.set(v, val);
        Ok(())
    }

    fn gstore(&mut self, op: &Op) -> Result<(), CompileError> {
        let (&value, address) = op.operands.split_last().expect("store has a value");
        let ty = self.k.ty(value).clone();
        let offset = op.int_attr("offset").unwrap_or(0);
        let align = op.int_attr("align").unwrap_or(1);
        let (addr, mods) = self.global_addr(address)?;
        let mem = super::alias::op_tag(&self.mem_tags, op);
        let bytes = match &ty {
            Type::Vec(e, n) => e.bytes() * n,
            other => other.elem().bytes(),
        };
        if bytes == 1 {
            let (o, k) = if ty == Type::Scalar(Scalar::I1) {
                let mask = self.mask(value)?;
                (self.bool_to_int(mask, false), 0)
            } else {
                self.byte(value)?
            };
            let (data, opc) = self.byte_store(o, k, Opc::GStoreB8, Opc::GStoreD16HiB8);
            let mut uses = addr;
            uses.push(data);
            self.emit(MInst::new(opc, [], uses).with_imm(offset).with_mods(mods).with_mem(mem));
            return Ok(());
        }
        if bytes == 2 {
            let (o, hi_half) = self.half(value)?;
            let o = self.to_v(o);
            let opc = if hi_half { Opc::GStoreD16HiB16 } else { Opc::GStoreB16 };
            let mut uses = addr;
            uses.push(o);
            self.emit(MInst::new(opc, [], uses).with_imm(offset).with_mods(mods).with_mem(mem));
            return Ok(());
        }
        let data = match ty {
            Type::Scalar(Scalar::I64 | Scalar::F64) | Type::Ptr(_) => {
                let (l, h) = self.d(value)?;
                self.pair_v(l, h)
            }
            Type::Scalar(_) => {
                let o = self.w(value)?;
                match self.to_v(o) {
                    Opnd::R(v) => v,
                    _ => unreachable!("to_v returns a register"),
                }
            }
            _ => self.tuple(value)?,
        };
        // Dword-multiple stores need no alignment either; see `gload`.
        if bytes % 2 != 0 || (bytes % 4 == 2 && align < 2) {
            return Err(unsupported(format!("global store of {bytes} bytes at align {align}")));
        }
        let full = (bytes / 4) as u8;
        let mut done = 0u8;
        while done < full {
            let (opc, n) = match full - done {
                1 => (Opc::GStoreB32, 1),
                2 => (Opc::GStoreB64, 2),
                3 => (Opc::GStoreB96, 3),
                _ => (Opc::GStoreB128, 4),
            };
            let mut uses = addr.clone();
            uses.push(r(sub(data, done, n)));
            self.emit(MInst::new(opc, [], uses).with_imm(offset + 4 * done as i64).with_mods(mods).with_mem(mem));
            done += n;
        }
        if bytes % 4 == 2 {
            let mut uses = addr;
            uses.push(r(sub(data, full, 1)));
            self.emit(MInst::new(Opc::GStoreB16, [], uses).with_imm(offset + 4 * full as i64).with_mods(mods).with_mem(mem));
        }
        Ok(())
    }

    fn lds_addr(&mut self, smem: ValueId, addr: ValueId, offset: i64) -> Result<(Opnd, i64), CompileError> {
        let base = match self.get(smem)? {
            Val::L(b) => b as i64,
            _ => return Err(unsupported("LDS access through a non-allocation")),
        };
        let a = self.w(addr)?;
        let total = base + offset;
        if (0..65536).contains(&total) {
            return Ok((self.to_v(a), total));
        }
        let va = self.to_v(a);
        let sum = self.v_bin(Opc::VAddNcU32, Opnd::Imm(total as u32), va);
        Ok((sum, 0))
    }

    fn lload(&mut self, op: &Op) -> Result<(), CompileError> {
        let v = op.results[0];
        if let Some(&quad) = self.chains.quads.get(&v) {
            return self.lload_quad(op, quad);
        }
        let ty = self.k.ty(v).clone();
        let (addr, offset) = self.lds_addr(op.operands[0], op.operands[1], op.int_attr("offset").unwrap_or(0))?;
        if self.gathers.contains(&v) {
            self.deferred.insert(v, Deferred::Lds(addr, offset));
            return Ok(());
        }
        let align = op.int_attr("align").unwrap_or(4);
        let bytes = match &ty {
            Type::Vec(e, n) => e.bytes() * n,
            other => other.elem().bytes(),
        };
        let dwords = bytes.div_ceil(4) as u8;
        let dst = self.vgpr(dwords);
        if bytes == 1 {
            self.emit(MInst::new(Opc::DsLoadU8, [r(dst)], [addr]).with_imm(offset));
            self.set(v, Val::B(r(dst), 0));
            return Ok(());
        }
        if bytes == 2 {
            self.emit(MInst::new(Opc::DsLoadU16, [r(dst)], [addr]).with_imm(offset));
        } else {
            let unit: u8 = match (dwords, align) {
                (4, a) if a >= 16 => 4,
                (2, a) | (4, a) if a >= 8 => 2,
                _ if align >= 4 => 1,
                _ => return Err(unsupported(format!("LDS load of {bytes} bytes at align {align}"))),
            };
            let mut done = 0u8;
            while done < dwords {
                let n = unit.min(dwords - done);
                let opc = match n {
                    4 => Opc::DsLoadB128,
                    2 => Opc::DsLoadB64,
                    _ => Opc::DsLoadB32,
                };
                self.emit(MInst::new(opc, [r(sub(dst, done, n))], [addr]).with_imm(offset + 4 * done as i64));
                done += n;
            }
        }
        let val = match ty {
            Type::Scalar(s) if s.is_half() => Val::H(r(dst), false),
            Type::Scalar(Scalar::I64 | Scalar::F64) => Val::D(r(sub(dst, 0, 1)), r(sub(dst, 1, 1))),
            Type::Scalar(_) => Val::W(r(dst)),
            _ => Val::T(dst),
        };
        self.set(v, val);
        Ok(())
    }

    /// The four loads of `find_dword_quads` that start at `op`, as one
    /// `ds_load_b128` whose dwords they read.
    fn lload_quad(&mut self, op: &Op, quad: [ValueId; 4]) -> Result<(), CompileError> {
        let (addr, offset) = self.lds_addr(op.operands[0], op.operands[1], op.int_attr("offset").unwrap_or(0))?;
        let dst = self.vgpr(4);
        self.emit(MInst::new(Opc::DsLoadB128, [r(dst)], [addr]).with_imm(offset));
        for (k, v) in quad.into_iter().enumerate() {
            let part = sub(dst, k as u8, 1);
            self.set(v, if matches!(self.k.ty(v), Type::Vec(..)) { Val::T(part) } else { Val::W(r(part)) });
        }
        Ok(())
    }

    fn lstore(&mut self, op: &Op) -> Result<(), CompileError> {
        let value = op.operands[2];
        let ty = self.k.ty(value).clone();
        let (addr, offset) = self.lds_addr(op.operands[0], op.operands[1], op.int_attr("offset").unwrap_or(0))?;
        let align = op.int_attr("align").unwrap_or(4);
        let bytes = match &ty {
            Type::Vec(e, n) => e.bytes() * n,
            other => other.elem().bytes(),
        };
        if bytes == 1 {
            let (o, k) = self.byte(value)?;
            let (data, opc) = self.byte_store(o, k, Opc::DsStoreB8, Opc::DsStoreB8D16Hi);
            self.emit(MInst::new(opc, [], [addr, data]).with_imm(offset));
            return Ok(());
        }
        if bytes == 2 {
            let (o, hi) = self.half(value)?;
            let o = self.to_v(o);
            let opc = if hi { Opc::DsStoreB16D16Hi } else { Opc::DsStoreB16 };
            self.emit(MInst::new(opc, [], [addr, o]).with_imm(offset));
            return Ok(());
        }
        let data = match ty {
            Type::Scalar(Scalar::I64 | Scalar::F64) => {
                let (l, h) = self.d(value)?;
                self.pair_v(l, h)
            }
            Type::Scalar(_) => {
                let o = self.w(value)?;
                match self.to_v(o) {
                    Opnd::R(v) => v,
                    _ => unreachable!("to_v returns a register"),
                }
            }
            _ => self.tuple(value)?,
        };
        let dwords = data.n;
        let unit: u8 = match (dwords, align) {
            (4, a) if a >= 16 => 4,
            (2, a) | (4, a) if a >= 8 => 2,
            _ if align >= 4 => 1,
            _ => return Err(unsupported(format!("LDS store of {bytes} bytes at align {align}"))),
        };
        let mut done = 0u8;
        while done < dwords {
            let n = unit.min(dwords - done);
            let opc = match n {
                4 => Opc::DsStoreB128,
                2 => Opc::DsStoreB64,
                _ => Opc::DsStoreB32,
            };
            self.emit(MInst::new(opc, [], [addr, r(sub(data, done, n))]).with_imm(offset + 4 * done as i64));
            done += n;
        }
        Ok(())
    }

    /// Register holding a loop-carried value of `ty` and its typed view.
    fn carried(&mut self, ty: &Type, uniform: bool) -> (VRef, Val) {
        match ty {
            Type::Scalar(Scalar::I1) => {
                let reg = self.mreg();
                (reg, Val::M(r(reg)))
            }
            Type::Scalar(Scalar::I64 | Scalar::F64) | Type::Ptr(_) => {
                let reg = if uniform { self.sreg(2) } else { self.vgpr(2) };
                (reg, Val::D(r(sub(reg, 0, 1)), r(sub(reg, 1, 1))))
            }
            Type::Scalar(s) if s.is_half() => {
                let reg = self.vgpr(1);
                (reg, Val::H(r(reg), false))
            }
            Type::Scalar(_) => {
                let reg = if uniform { self.sreg(1) } else { self.vgpr(1) };
                (reg, Val::W(r(reg)))
            }
            _ => {
                let reg = self.vgpr(ty.dwords() as u8);
                (reg, Val::T(reg))
            }
        }
    }

    /// Copies `v`'s value into the loop-carried register `dst`.
    fn copy_into(&mut self, dst: VRef, v: ValueId) -> Result<(), CompileError> {
        let ty = self.k.ty(v).clone();
        match ty {
            Type::Scalar(Scalar::I1) => {
                let m = self.mask(v)?;
                self.copy(dst, m);
            }
            Type::Scalar(Scalar::I64 | Scalar::F64) | Type::Ptr(_) => {
                let (lo, hi) = self.d(v)?;
                self.copy(sub(dst, 0, 1), lo);
                self.copy(sub(dst, 1, 1), hi);
            }
            Type::Scalar(_) => {
                let o = self.w(v)?;
                self.copy(dst, o);
            }
            _ => {
                let t = self.tuple(v)?;
                self.copy(dst, r(t));
            }
        }
        Ok(())
    }

    fn for_op(&mut self, op: &Op) -> Result<(), CompileError> {
        let (lb_v, ub_v, step_v) = (op.operands[0], op.operands[1], op.operands[2]);
        if !(self.uni(lb_v) && self.uni(ub_v) && self.uni(step_v)) {
            return self.divergent_for(op);
        }
        let lb = self.w(lb_v)?;
        let ub = self.w(ub_v)?;
        let step = self.w(step_v)?;
        let ub = match ub {
            Opnd::Imm(x) if !is_inline_int(x) => self.to_s(ub),
            _ => ub,
        };
        let iv = self.sreg(1);
        self.copy(iv, lb);
        self.set(op.block_args[0], Val::W(r(iv)));
        let mut carried = Vec::new();
        for (i, &arg) in op.block_args.iter().enumerate().skip(1) {
            let ty = self.k.ty(arg).clone();
            let (reg, view) = self.carried(&ty, self.uni(arg));
            self.copy_into(reg, op.operands[2 + i])?;
            self.set(arg, view);
            carried.push(reg);
        }
        let always_runs = match (self.facts.range_of(lb_v), self.facts.range_of(ub_v)) {
            (Some(l), Some(u)) => l.hi < u.lo,
            _ => false,
        };
        let skip_branch = if always_runs {
            None
        } else {
            self.salu(Opc::SCmpLtI32, [], [r(iv), ub]);
            self.emit(MInst::new(Opc::SCbranchScc0, [], []));
            Some((self.cur, self.f.blocks[self.cur].insts.len() - 1))
        };
        self.depth += 1;
        let body_block = self.new_block();
        let body = &op.regions[0];
        self.region(body)?;
        let yields = body.ops.last().filter(|y| y.kind == OpKind::Yield).map(|y| y.operands.clone()).unwrap_or_default();
        // Parallel copy of the yielded values into the carried registers.
        let reads_carried = yields.iter().any(|&y| {
            let reg = match self.vals[y.0 as usize] {
                Some(Val::W(Opnd::R(v))) | Some(Val::T(v)) | Some(Val::M(Opnd::R(v))) => Some(v.reg),
                Some(Val::D(Opnd::R(v), _)) | Some(Val::H(Opnd::R(v), _)) | Some(Val::B(Opnd::R(v), _)) => Some(v.reg),
                _ => None,
            };
            reg.is_some_and(|reg| carried.iter().any(|c| c.reg == reg))
        });
        if reads_carried {
            let mut temps = Vec::new();
            for (i, &y) in yields.iter().enumerate() {
                let ty = self.k.ty(y).clone();
                let (tmp, _) = self.carried(&ty, self.f.vregs[carried[i].reg as usize].class == Class::S);
                self.copy_into(tmp, y)?;
                temps.push(tmp);
            }
            for (i, tmp) in temps.into_iter().enumerate() {
                self.copy(carried[i], r(tmp));
            }
        } else {
            for (i, &y) in yields.iter().enumerate() {
                self.copy_into(carried[i], y)?;
            }
        }
        self.salu(Opc::SAddI32, [r(iv)], [r(iv), step]);
        self.salu(Opc::SCmpLtI32, [], [r(iv), ub]);
        self.emit(MInst::new(Opc::SCbranchScc1, [], []).with_imm(body_block as i64));
        self.depth -= 1;
        let exit = self.new_block();
        if let Some((block, index)) = skip_branch {
            self.f.blocks[block].insts[index].imm = exit as i64;
        }
        for (i, &res) in op.results.iter().enumerate() {
            let ty = self.k.ty(res).clone();
            let view = match ty {
                Type::Scalar(Scalar::I1) => Val::M(r(carried[i])),
                Type::Scalar(Scalar::I64 | Scalar::F64) | Type::Ptr(_) => Val::D(r(sub(carried[i], 0, 1)), r(sub(carried[i], 1, 1))),
                Type::Scalar(s) if s.is_half() => Val::H(r(carried[i]), false),
                Type::Scalar(_) => Val::W(r(carried[i])),
                _ => Val::T(carried[i]),
            };
            self.set(res, view);
        }
        Ok(())
    }

    /// `scf.for` whose bounds differ between lanes, as an EXEC-masked loop:
    /// each trip first drops the lanes whose induction variable reached
    /// their bound, and the loop ends when none is left; EXEC is restored
    /// after it. The induction variable stays in an SGPR when the start and
    /// the step are uniform, since every lane still running is on the same
    /// trip. Carried values are VGPRs (see `analysis::classify`), which keep
    /// a lane's value once it stops. A loop of at most one trip (the bound
    /// known to be at most `lb + step`) is a masked `if`.
    fn divergent_for(&mut self, op: &Op) -> Result<(), CompileError> {
        let (lb_v, ub_v, step_v) = (op.operands[0], op.operands[1], op.operands[2]);
        if let Some(kind) = unsafe_under_partial_exec(&op.regions[0]) {
            return Err(unsupported(format!("{kind:?} inside an scf.for with divergent bounds")));
        }
        if op.block_args[1..].iter().any(|&a| *self.k.ty(a) == Type::Scalar(Scalar::I1)) {
            return Err(unsupported("i1 carried through an scf.for with divergent bounds"));
        }
        let iv_uniform = self.uni(op.block_args[0]);
        let lb = self.w(lb_v)?;
        let ub = self.w(ub_v)?;
        let step = self.w(step_v)?;
        let iv = if iv_uniform { self.sreg(1) } else { self.vgpr(1) };
        self.copy(iv, lb);
        self.set(op.block_args[0], Val::W(r(iv)));
        let mut carried = Vec::new();
        for (i, &arg) in op.block_args.iter().enumerate().skip(1) {
            let ty = self.k.ty(arg).clone();
            let (reg, view) = self.carried(&ty, false);
            self.copy_into(reg, op.operands[2 + i])?;
            self.set(arg, view);
            carried.push(reg);
        }
        let one_trip = match (self.facts.int(lb_v), self.facts.int(step_v), self.facts.range_of(ub_v)) {
            (Some(l), Some(s), Some(u)) => s > 0 && u.hi <= l + s,
            _ => false,
        };
        let saved = self.mreg();
        self.emit(MInst::new(self.lm(Opc::SMovB32), [r(saved)], [self.exec()]));
        self.depth += 1;
        let head = self.new_block();
        let running = self.mreg();
        self.valu(Opc::VCmp(0x41), [r(running)], [r(iv), ub], 0);
        self.salu(self.lm(Opc::SAndB32), [self.exec()], [self.exec(), r(running)]);
        self.emit(MInst::new(Opc::SCbranchExecz, [], []));
        let leave = (self.cur, self.f.blocks[self.cur].insts.len() - 1);
        self.new_block();
        let body = &op.regions[0];
        self.region(body)?;
        let yields = body.ops.last().map(|y| y.operands.clone()).unwrap_or_default();
        let reads_carried = yields.iter().any(|&y| {
            let reg = match self.vals[y.0 as usize] {
                Some(Val::W(Opnd::R(v))) | Some(Val::T(v)) => Some(v.reg),
                Some(Val::D(Opnd::R(v), _)) | Some(Val::H(Opnd::R(v), _)) | Some(Val::B(Opnd::R(v), _)) => Some(v.reg),
                _ => None,
            };
            reg.is_some_and(|reg| carried.iter().any(|c| c.reg == reg))
        });
        if reads_carried {
            let mut temps = Vec::new();
            for &y in &yields {
                let ty = self.k.ty(y).clone();
                let (tmp, _) = self.carried(&ty, false);
                self.copy_into(tmp, y)?;
                temps.push(tmp);
            }
            for (i, tmp) in temps.into_iter().enumerate() {
                self.copy(carried[i], r(tmp));
            }
        } else {
            for (i, &y) in yields.iter().enumerate() {
                self.copy_into(carried[i], y)?;
            }
        }
        if !one_trip {
            if iv_uniform {
                self.salu(Opc::SAddI32, [r(iv)], [r(iv), step]);
            } else {
                self.valu(Opc::VAddNcU32, [r(iv)], [step, r(iv)], 0);
            }
            self.emit(MInst::new(Opc::SBranch, [], []).with_imm(head as i64));
        }
        self.depth -= 1;
        let exit = self.new_block();
        self.f.blocks[leave.0].insts[leave.1].imm = exit as i64;
        self.emit(MInst::new(self.lm(Opc::SMovB32), [self.exec()], [r(saved)]));
        for (i, &res) in op.results.iter().enumerate() {
            let view = match self.k.ty(res).clone() {
                Type::Scalar(Scalar::I64 | Scalar::F64) | Type::Ptr(_) => Val::D(r(sub(carried[i], 0, 1)), r(sub(carried[i], 1, 1))),
                Type::Scalar(s) if s.is_half() => Val::H(r(carried[i]), false),
                Type::Scalar(_) => Val::W(r(carried[i])),
                _ => Val::T(carried[i]),
            };
            self.set(res, view);
        }
        Ok(())
    }

    fn if_op(&mut self, op: &Op) -> Result<(), CompileError> {
        let cond = op.operands[0];
        let has_else = op.kind == OpKind::IfElse;
        if self.uni(cond) {
            self.scc(cond)?;
            self.emit(MInst::new(Opc::SCbranchScc0, [], []));
            let first = (self.cur, self.f.blocks[self.cur].insts.len() - 1);
            self.new_block();
            self.region(&op.regions[0])?;
            if has_else {
                self.emit(MInst::new(Opc::SBranch, [], []));
                let to_end = (self.cur, self.f.blocks[self.cur].insts.len() - 1);
                let else_block = self.new_block();
                self.f.blocks[first.0].insts[first.1].imm = else_block as i64;
                self.region(&op.regions[1])?;
                let end = self.new_block();
                self.f.blocks[to_end.0].insts[to_end.1].imm = end as i64;
            } else {
                let end = self.new_block();
                self.f.blocks[first.0].insts[first.1].imm = end as i64;
            }
            return Ok(());
        }
        let mask = self.mask(cond)?;
        let saved = self.mreg();
        let exec = self.exec();
        self.emit(MInst::new(self.lm(Opc::SAndSaveexecB32), [r(saved), exec], [mask, exec]));
        self.emit(MInst::new(Opc::SCbranchExecz, [], []));
        let first = (self.cur, self.f.blocks[self.cur].insts.len() - 1);
        self.new_block();
        self.region(&op.regions[0])?;
        if has_else {
            let else_block = self.new_block();
            self.f.blocks[first.0].insts[first.1].imm = else_block as i64;
            self.salu(self.lm(Opc::SAndNot1B32), [exec], [r(saved), mask]);
            self.emit(MInst::new(Opc::SCbranchExecz, [], []));
            let second = (self.cur, self.f.blocks[self.cur].insts.len() - 1);
            self.new_block();
            self.region(&op.regions[1])?;
            let end = self.new_block();
            self.f.blocks[second.0].insts[second.1].imm = end as i64;
            self.emit(MInst::new(self.lm(Opc::SMovB32), [exec], [r(saved)]));
        } else {
            let end = self.new_block();
            self.f.blocks[first.0].insts[first.1].imm = end as i64;
            self.salu(self.lm(Opc::SOrB32), [exec], [exec, r(saved)]);
        }
        Ok(())
    }
}

/// The SOPC compare with the opposite outcome.
fn inverse_scmp(op: Opc) -> Option<Opc> {
    use Opc::*;
    Some(match op {
        SCmpEqI32 => SCmpLgI32,
        SCmpLgI32 => SCmpEqI32,
        SCmpGtI32 => SCmpLeI32,
        SCmpLeI32 => SCmpGtI32,
        SCmpGeI32 => SCmpLtI32,
        SCmpLtI32 => SCmpGeI32,
        SCmpEqU32 => SCmpLgU32,
        SCmpLgU32 => SCmpEqU32,
        SCmpGtU32 => SCmpLeU32,
        SCmpLeU32 => SCmpGtU32,
        SCmpGeU32 => SCmpLtU32,
        SCmpLtU32 => SCmpGeU32,
        SCmpEqU64 => SCmpLgU64,
        SCmpLgU64 => SCmpEqU64,
        _ => return None,
    })
}

/// An op in `region` that is wrong under a partial EXEC mask: a WMMA writes
/// every lane whatever EXEC holds, and a barrier or an early return would
/// leave the lanes that stopped (and other waves) behind.
fn unsafe_under_partial_exec(region: &Region) -> Option<OpKind> {
    region.ops.iter().find_map(|op| match op.kind {
        OpKind::Mma | OpKind::Sync | OpKind::BarrierBare | OpKind::SyncLdsOnly | OpKind::Return => Some(op.kind),
        _ => op.regions.iter().find_map(unsafe_under_partial_exec),
    })
}

/// Hacker's Delight magic number `M` and shift `s` for signed 64-bit
/// division by `d >= 2`: with `q = mulhs(x, M) + (M < 0 ? x : 0) >> s`,
/// `x / d` is `q` plus one when `q` is negative.
fn signed_magic64(d: i64) -> (u64, u32) {
    let two63: u64 = 1 << 63;
    let ad = d as u64;
    let anc = two63 - 1 - two63 % ad;
    let mut p = 63u32;
    let (mut q1, mut r1) = (two63 / anc, two63 % anc);
    let (mut q2, mut r2) = (two63 / ad, two63 % ad);
    loop {
        p += 1;
        q1 = q1.wrapping_mul(2);
        r1 = r1.wrapping_mul(2);
        if r1 >= anc {
            q1 = q1.wrapping_add(1);
            r1 = r1.wrapping_sub(anc);
        }
        q2 = q2.wrapping_mul(2);
        r2 = r2.wrapping_mul(2);
        if r2 >= ad {
            q2 = q2.wrapping_add(1);
            r2 = r2.wrapping_sub(ad);
        }
        let delta = ad - r2;
        if !(q1 < delta || (q1 == delta && r1 == 0)) {
            break;
        }
    }
    (q2.wrapping_add(1), p - 64)
}

fn count_uses(region: &Region, uses: &mut [u32]) {
    for op in &region.ops {
        for v in &op.operands {
            uses[v.0 as usize] += 1;
        }
        for inner in &op.regions {
            count_uses(inner, uses);
        }
    }
}

fn has_side_effects(kind: OpKind) -> bool {
    use OpKind::*;
    matches!(
        kind,
        For | If
            | IfElse
            | Yield
            | Return
            | GlobalStore
            | GlobalStoreVec
            | GlobalAtomicAddF32
            | SmemStore
            | SmemStoreVec
            | BufferStoreVecF16
            | Sync
            | InlineAsmIdentity
            | GStore
            | GAtomicAddF32
            | LStore
            | SetPrio
            | WaitCnt
            | SchedBarrier
            | BarrierBare
            | SyncLdsOnly
            | LdsAtomicAdd
            | LAtomicAdd
            | BufferStoreScalar
            | GlobalAtomicAdd
            | GAtomicAdd
            | Fence
    )
}

/// 16-bit loads that become d16 loads straight into the half that receives
/// them, as LLVM selects them: divergent f16 global loads whose only use is
/// a later `vector.insert`, and pairs of f16 LDS loads whose only use is a
/// `vector.concat` into an f16x2, each with no side effect in between.
fn find_gathers(k: &Kernel, region: &Region, uniform: &[bool], uses: &[u32], out: &mut FxHashSet<ValueId>) {
    let mut lds_halves: FxHashMap<ValueId, usize> = FxHashMap::default();
    let mut last_effect = None;
    for (i, op) in region.ops.iter().enumerate() {
        for inner in &op.regions {
            find_gathers(k, inner, uniform, uses, out);
        }
        if has_side_effects(op.kind) {
            last_effect = Some(i);
        }
        if op.kind == OpKind::LLoad
            && matches!(k.ty(op.results[0]), Type::Vec(e, 1) if e.is_half())
            && uses[op.results[0].0 as usize] == 1
        {
            lds_halves.insert(op.results[0], i);
        }
        if op.kind == OpKind::Concat && matches!(k.ty(op.results[0]), Type::Vec(e, 2) if e.is_half()) {
            let (a, b) = (op.operands[0], op.operands[1]);
            if let (Some(&pa), Some(&pb)) = (lds_halves.get(&a), lds_halves.get(&b)) {
                if a != b && last_effect.is_none_or(|e| e < pa.min(pb)) {
                    out.insert(a);
                    out.insert(b);
                }
            }
        }
        if !matches!(op.kind, OpKind::GLoad | OpKind::BufferLoadScalar) {
            continue;
        }
        let v = op.results[0];
        if uniform[v.0 as usize] || !matches!(k.ty(v), Type::Scalar(e) if e.is_half()) || uses[v.0 as usize] != 1 {
            continue;
        }
        for later in &region.ops[i + 1..] {
            if later.kind == OpKind::Insert && later.operands[1] == v {
                if later.operands[0] != v && k.ty(later.results[0]).elem().is_half() {
                    out.insert(v);
                }
                break;
            }
            if has_side_effects(later.kind) || later.operands.contains(&v) {
                break;
            }
        }
    }
}

/// f16 insert chains that rewrite every element of their vector.
#[derive(Default)]
struct Chains {
    /// Head inserts whose source is dead because the chain rewrites it all.
    undef_src: FxHashSet<ValueId>,
    /// Loads, extracts and inserts folded into a merged LDS load.
    skip: FxHashSet<ValueId>,
    /// Chain tail -> (smem, address, byte offset of element 0, alignment).
    lds: FxHashMap<ValueId, (ValueId, ValueId, i64, i64)>,
    /// `-inf < x && x < +inf` -> x, whose compares are in `skip`.
    finite: FxHashMap<ValueId, ValueId>,
    /// i32 `or(zext(lo), shl(zext(hi), 16))` of two 16-bit values -> (lo, hi);
    /// the zexts and the shift are in `skip`.
    packs: FxHashMap<ValueId, (ValueId, ValueId)>,
    /// First of four dword LDS loads made one (`find_dword_quads`) -> the four
    /// results; the other three are in `skip`.
    quads: FxHashMap<ValueId, [ValueId; 4]>,
}

struct ChainCx<'a> {
    k: &'a Kernel,
    facts: &'a Facts,
    lds: &'a FxHashMap<ValueId, u32>,
    uses: &'a [u32],
    defs: &'a FxHashMap<ValueId, &'a Op>,
}

fn collect_defs<'a>(region: &'a Region, defs: &mut FxHashMap<ValueId, &'a Op>) {
    for op in &region.ops {
        for &r in &op.results {
            defs.insert(r, op);
        }
        for inner in &op.regions {
            collect_defs(inner, defs);
        }
    }
}

/// Largest power of two (capped at 64 KiB) known to divide `v`.
fn known_align(cx: &ChainCx, v: ValueId, depth: u32) -> i64 {
    const MAX: i64 = 1 << 16;
    if let Some(c) = cx.facts.int(v) {
        return if c == 0 { MAX } else { (c & c.wrapping_neg()).abs().min(MAX) };
    }
    let Some(op) = cx.defs.get(&v) else { return 1 };
    if depth > 8 {
        return 1;
    }
    let a = |i: usize| known_align(cx, op.operands[i], depth + 1);
    match op.kind {
        OpKind::Mul => (a(0) * a(1)).min(MAX),
        OpKind::Shl => match cx.facts.int(op.operands[1]) {
            Some(s) if (0..16).contains(&s) => (a(0) << s).min(MAX),
            _ => 1,
        },
        OpKind::Add | OpKind::Sub => a(0).min(a(1)),
        _ => 1,
    }
}

fn find_chains(cx: &ChainCx, region: &Region, out: &mut Chains) {
    for op in &region.ops {
        for inner in &op.regions {
            find_chains(cx, inner, out);
        }
    }
    let mut pos = FxHashMap::default();
    for (i, op) in region.ops.iter().enumerate() {
        for &r in &op.results {
            pos.insert(r, i);
        }
    }
    let f16_insert = |op: &Op| op.kind == OpKind::Insert && cx.k.ty(op.results[0]).elem().is_half();
    let linked: FxHashSet<ValueId> = region
        .ops
        .iter()
        .filter(|op| {
            f16_insert(op)
                && cx.uses[op.operands[0].0 as usize] == 1
                && pos.get(&op.operands[0]).is_some_and(|&j| f16_insert(&region.ops[j]))
        })
        .map(|op| op.operands[0])
        .collect();
    for (i, op) in region.ops.iter().enumerate() {
        if !f16_insert(op) || linked.contains(&op.results[0]) {
            continue;
        }
        let Type::Vec(_, lanes) = *cx.k.ty(op.results[0]) else { continue };
        let lanes = lanes as i64;
        let mut chain = vec![i];
        let mut src = op.operands[0];
        while linked.contains(&src) {
            let j = pos[&src];
            chain.push(j);
            src = region.ops[j].operands[0];
        }
        chain.reverse();
        let mut covered = vec![false; lanes as usize];
        for &j in &chain {
            if let Some(index) = region.ops[j].int_attr("index").filter(|x| (0..lanes).contains(x)) {
                covered[index as usize] = true;
            }
        }
        if covered.iter().all(|&c| c) && !lds_chain(cx, region, &pos, &chain, out) {
            out.undef_src.insert(region.ops[chain[0]].results[0]);
        }
    }
}

/// Folds a chain whose element `t` is a single-use f16 LDS load at byte
/// offset `first + 2t` from one address into one vector load, when the
/// address is at least dword aligned and nothing in between has a side effect.
fn lds_chain(cx: &ChainCx, region: &Region, pos: &FxHashMap<ValueId, usize>, chain: &[usize], out: &mut Chains) -> bool {
    let mut folded = Vec::new();
    let mut key: Option<(ValueId, ValueId, i64)> = None;
    let mut first_pos = chain[0];
    for (t, &j) in chain.iter().enumerate() {
        let insert = &region.ops[j];
        if insert.int_attr("index") != Some(t as i64) {
            return false;
        }
        let mut e = insert.operands[1];
        let Some(&p) = pos.get(&e) else { return false };
        let mut def = &region.ops[p];
        if def.kind == OpKind::Extract {
            if def.int_attr("index").unwrap_or(0) != 0 || cx.uses[e.0 as usize] != 1 {
                return false;
            }
            folded.push(e);
            e = def.operands[0];
            let Some(&q) = pos.get(&e) else { return false };
            def = &region.ops[q];
        }
        let bytes = match cx.k.ty(e) {
            Type::Vec(s, n) => s.bytes() * n,
            other => other.elem().bytes(),
        };
        if def.kind != OpKind::LLoad || !cx.k.ty(e).elem().is_half() || bytes != 2 || cx.uses[e.0 as usize] != 1 {
            return false;
        }
        let offset = def.int_attr("offset").unwrap_or(0);
        match key {
            None => key = Some((def.operands[0], def.operands[1], offset)),
            Some((smem, addr, first)) => {
                if def.operands[0] != smem || def.operands[1] != addr || offset != first + 2 * t as i64 {
                    return false;
                }
            }
        }
        first_pos = first_pos.min(pos[&e]);
        folded.push(e);
    }
    let Some((smem, addr, first)) = key else { return false };
    let tail = *chain.last().expect("chain is non-empty");
    if region.ops[first_pos..=tail].iter().any(|op| has_side_effects(op.kind)) {
        return false;
    }
    let Some(&base) = cx.lds.get(&smem) else { return false };
    let start = base as i64 + first;
    let align = known_align(cx, addr, 0).min(if start == 0 { 1 << 16 } else { start & start.wrapping_neg() });
    if align < 4 {
        return false;
    }
    folded.extend(chain[..chain.len() - 1].iter().map(|&j| region.ops[j].results[0]));
    out.skip.extend(folded);
    out.lds.insert(region.ops[tail].results[0], (smem, addr, first, align));
    true
}

/// Four dword LDS loads (f32 or i32, scalar or one-element vectors) of one
/// address at byte offsets `o` to `o + 12`, in program order with nothing
/// in between that has a side effect, the address and `o` 16-byte aligned:
/// one `ds_load_b128` at the first (the scalar reads of a block reduction).
fn find_dword_quads(cx: &ChainCx, region: &Region, gathers: &FxHashSet<ValueId>, out: &mut Chains) {
    for op in &region.ops {
        for inner in &op.regions {
            find_dword_quads(cx, inner, gathers, out);
        }
    }
    let dword = |v: ValueId| {
        matches!(cx.k.ty(v), Type::Scalar(Scalar::F32 | Scalar::I32) | Type::Vec(Scalar::F32 | Scalar::I32, 1))
    };
    let mut effects = vec![0u32; region.ops.len() + 1];
    let mut at: FxHashMap<(ValueId, ValueId, i64), Option<usize>> = FxHashMap::default();
    for (i, op) in region.ops.iter().enumerate() {
        effects[i + 1] = effects[i] + has_side_effects(op.kind) as u32;
        if op.kind == OpKind::LLoad && dword(op.results[0]) && !gathers.contains(&op.results[0]) {
            let key = (op.operands[0], op.operands[1], op.int_attr("offset").unwrap_or(0));
            at.entry(key).and_modify(|p| *p = None).or_insert(Some(i));
        }
    }
    let mut taken = FxHashSet::default();
    for (i, op) in region.ops.iter().enumerate() {
        if op.kind != OpKind::LLoad || taken.contains(&i) {
            continue;
        }
        let (smem, addr, first) = (op.operands[0], op.operands[1], op.int_attr("offset").unwrap_or(0));
        if at.get(&(smem, addr, first)) != Some(&Some(i)) {
            continue;
        }
        let Some(&base) = cx.lds.get(&smem) else { continue };
        if (base as i64 + first) % 16 != 0 || known_align(cx, addr, 0) % 16 != 0 {
            continue;
        }
        let rest: Option<Vec<usize>> = (1..4).map(|k| at.get(&(smem, addr, first + 4 * k)).copied().flatten()).collect();
        let Some(rest) = rest else { continue };
        let last = rest[2];
        let ordered = i < rest[0] && rest[0] < rest[1] && rest[1] < last;
        if !ordered || effects[last + 1] != effects[i] || rest.iter().any(|p| taken.contains(p)) {
            continue;
        }
        let quad = [i, rest[0], rest[1], last].map(|p| region.ops[p].results[0]);
        taken.extend([i, rest[0], rest[1], last]);
        out.skip.extend(&quad[1..]);
        out.quads.insert(quad[0], quad);
    }
}

/// Finiteness tests `fcmp ogt x, -inf && fcmp olt x, +inf` whose compares
/// have no other use: one `v_cmp_class_f32` of the finite classes.
fn find_finite_tests(cx: &ChainCx, region: &Region, out: &mut Chains) {
    for op in &region.ops {
        for inner in &op.regions {
            find_finite_tests(cx, inner, out);
        }
        if op.kind != OpKind::And || *cx.k.ty(op.results[0]) != Type::Scalar(Scalar::I1) {
            continue;
        }
        let bound = |v: ValueId| -> Option<(ValueId, bool)> {
            let def = cx.defs.get(&v)?;
            if def.kind != OpKind::Fcmp || cx.uses[v.0 as usize] != 1 {
                return None;
            }
            let Some(super::opt::Konst::F32(limit)) = cx.facts.konst.get(def.operands[1].0 as usize).copied().flatten()
            else {
                return None;
            };
            match (def.str_attr("pred")?, limit) {
                ("ogt", 0xff80_0000) => Some((def.operands[0], false)),
                ("olt", 0x7f80_0000) => Some((def.operands[0], true)),
                _ => None,
            }
        };
        if let (Some((x, upper)), Some((y, other))) = (bound(op.operands[0]), bound(op.operands[1])) {
            if x == y && upper != other {
                out.finite.insert(op.results[0], x);
                out.skip.extend([op.operands[0], op.operands[1]]);
            }
        }
    }
}

/// `or(zext(a), shl(zext(b), 16))` of two 16-bit values whose zexts and
/// shift have no other use: one v_perm_b32 of the two halves, as LLVM
/// selects it, instead of masking, shifting and merging.
fn find_half_packs(cx: &ChainCx, region: &Region, uniform: &[bool], out: &mut Chains) {
    for op in &region.ops {
        for inner in &op.regions {
            find_half_packs(cx, inner, uniform, out);
        }
        if op.kind != OpKind::Or || *cx.k.ty(op.results[0]) != Type::Scalar(Scalar::I32) || uniform[op.results[0].0 as usize] {
            continue;
        }
        let zext16 = |v: ValueId| -> Option<ValueId> {
            let def = cx.defs.get(&v).filter(|d| d.kind == OpKind::Zext)?;
            let half = *cx.k.ty(def.operands[0]) == Type::Scalar(Scalar::I16);
            (half && cx.uses[v.0 as usize] == 1).then(|| def.operands[0])
        };
        let shifted = |v: ValueId| -> Option<(ValueId, ValueId)> {
            let def = cx.defs.get(&v)?;
            if def.kind != OpKind::Shl || cx.uses[v.0 as usize] != 1 || cx.facts.int(def.operands[1]) != Some(16) {
                return None;
            }
            Some((def.operands[0], zext16(def.operands[0])?))
        };
        let (a, b) = (op.operands[0], op.operands[1]);
        for (x, y) in [(a, b), (b, a)] {
            if let (Some(lo), Some((z, hi))) = (zext16(x), shifted(y)) {
                out.packs.insert(op.results[0], (lo, hi));
                out.skip.extend([x, y, z]);
                break;
            }
        }
    }
}

/// f32 `arith.fneg` results whose every use is an operand of an f32
/// `arith.fma`, with the negated value: the FMAs take it with a negation
/// modifier, as LLVM folds them, and the negation itself is never emitted.
fn find_fneg_fma(k: &Kernel, region: &Region, negs: &mut FxHashMap<ValueId, ValueId>, fma_uses: &mut FxHashMap<ValueId, u32>) {
    for op in &region.ops {
        if op.kind == OpKind::Fneg && *k.ty(op.results[0]) == Type::Scalar(Scalar::F32) {
            negs.insert(op.results[0], op.operands[0]);
        }
        if op.kind == OpKind::Fma && *k.ty(op.results[0]) == Type::Scalar(Scalar::F32) {
            for v in &op.operands {
                if negs.contains_key(v) {
                    *fma_uses.entry(*v).or_default() += 1;
                }
            }
        }
        for inner in &op.regions {
            find_fneg_fma(k, inner, negs, fma_uses);
        }
    }
}

/// `or(zext(bitcast_i16(lo)), shl(zext(bitcast_i16(hi)), 16))`, two f16 bit
/// patterns packed into an i32, whose intermediates have no other use: the
/// pair is built in the two halves of one register, from the f32 values
/// (or f32 products) when single-use f16 conversions produce the halves.
/// Every op that folds in goes to `skip`.
fn find_f16_packs(cx: &ChainCx, region: &Region, packs: &mut FxHashMap<ValueId, [PackPart; 2]>, skip: &mut FxHashSet<ValueId>) {
    let single = |v: ValueId| cx.uses[v.0 as usize] == 1;
    let def = |v: ValueId| cx.defs.get(&v).copied();
    // The 16-bit value a single-use `zext(i16)` widens; the folded ops go to `chain`.
    let zext16 = |v: ValueId, chain: &mut Vec<ValueId>| -> Option<ValueId> {
        let z = def(v).filter(|d| d.kind == OpKind::Zext && single(v))?;
        let s = z.operands[0];
        if *cx.k.ty(s) != Type::Scalar(Scalar::I16) {
            return None;
        }
        chain.push(v);
        match def(s) {
            Some(b) if b.kind == OpKind::Bitcast && single(s) && *cx.k.ty(b.operands[0]) == Type::Scalar(Scalar::F16) => {
                chain.push(s);
                Some(b.operands[0])
            }
            _ => Some(s),
        }
    };
    for op in &region.ops {
        for inner in &op.regions {
            find_f16_packs(cx, inner, packs, skip);
        }
        if op.kind != OpKind::Or || *cx.k.ty(op.results[0]) != Type::Scalar(Scalar::I32) {
            continue;
        }
        let (x, y) = (op.operands[0], op.operands[1]);
        let mut found = None;
        for (lo, hi) in [(x, y), (y, x)] {
            let mut chain = Vec::new();
            let Some(l) = zext16(lo, &mut chain) else { continue };
            let Some(sh) = def(hi).filter(|d| d.kind == OpKind::Shl && single(hi) && cx.facts.int(d.operands[1]) == Some(16))
            else {
                continue;
            };
            let Some(h) = zext16(sh.operands[0], &mut chain) else { continue };
            chain.push(hi);
            found = Some((l, h, chain));
            break;
        }
        let Some((l, h, chain)) = found else { continue };
        let mut part = |v: ValueId| -> PackPart {
            let cast = def(v).filter(|d| d.kind == OpKind::CastF32To && d.str_attr("target") == Some("f16") && single(v));
            let Some(c) = cast else { return PackPart::Half(v) };
            skip.insert(v);
            let f = c.operands[0];
            match def(f).filter(|d| d.kind == OpKind::Fmul && *cx.k.ty(f) == Type::Scalar(Scalar::F32) && single(f)) {
                Some(m) => {
                    skip.insert(f);
                    PackPart::Mix(m.operands[0], m.operands[1])
                }
                None => PackPart::Cvt(f),
            }
        };
        let parts = [part(l), part(h)];
        skip.extend(chain);
        packs.insert(op.results[0], parts);
    }
}

/// f16 `vector.trunc_f32_to` of a single-use product of even-length f32
/// vectors: the product folds into the conversion (`trunc_mix`).
fn find_trunc_products(cx: &ChainCx, region: &Region, found: &mut FxHashMap<ValueId, (ValueId, ValueId)>, skip: &mut FxHashSet<ValueId>) {
    for op in &region.ops {
        for inner in &op.regions {
            find_trunc_products(cx, inner, found, skip);
        }
        if op.kind != OpKind::TruncF32To || op.str_attr("target") != Some("f16") {
            continue;
        }
        let x = op.operands[0];
        let Some(m) = cx.defs.get(&x).filter(|d| matches!(d.kind, OpKind::Fmul | OpKind::VecMul) && cx.uses[x.0 as usize] == 1) else {
            continue;
        };
        if matches!(*cx.k.ty(x), Type::Vec(Scalar::F32, n) if n % 2 == 0)
            && m.operands.iter().all(|&o| *cx.k.ty(o) == *cx.k.ty(x))
        {
            found.insert(op.results[0], (m.operands[0], m.operands[1]));
            skip.insert(x);
        }
    }
}

/// `tile.wave_any` / `tile.wave_all` of a single-use `select(c, k1, k2)` of
/// constants: the vote reads c's lane mask and the select is never emitted.
fn find_vote_selects(cx: &ChainCx, region: &Region, votes: &mut FxHashMap<ValueId, (ValueId, bool, bool)>, skip: &mut FxHashSet<ValueId>) {
    for op in &region.ops {
        for inner in &op.regions {
            find_vote_selects(cx, inner, votes, skip);
        }
        if !matches!(op.kind, OpKind::WaveAny | OpKind::WaveAll) {
            continue;
        }
        let p = op.operands[0];
        let Some(s) = cx.defs.get(&p).filter(|d| d.kind == OpKind::Select && cx.uses[p.0 as usize] == 1) else {
            continue;
        };
        let (Some(a), Some(b)) = (cx.facts.int(s.operands[1]), cx.facts.int(s.operands[2])) else { continue };
        if *cx.k.ty(s.operands[0]) == Type::Scalar(Scalar::I1) {
            votes.insert(p, (s.operands[0], a != 0, b != 0));
            skip.insert(p);
        }
    }
}

/// Two `ds_store_b32` to one address register, consecutive among the block's
/// LDS accesses, become one `ds_store_2addr_b32` at the second, as LLVM pairs
/// them. Each dword offset takes 8 bits, so the address is rebased by a
/// multiple of 1 KiB when needed. The first store moves down only past
/// instructions that neither touch LDS, write EXEC, nor redefine its operands.
fn pair_lds_stores(f: &mut MFunc) {
    let overlaps = |a: &Opnd, b: &Opnd| match (a, b) {
        (Opnd::R(x), Opnd::R(y)) => x.reg == y.reg && x.off < y.off + y.n && y.off < x.off + x.n,
        _ => false,
    };
    for b in 0..f.blocks.len() {
        let insts = std::mem::take(&mut f.blocks[b].insts);
        let mut out: Vec<Option<MInst>> = Vec::with_capacity(insts.len());
        // The last LDS access, when it is a single-dword store that may still pair.
        let mut open: Option<usize> = None;
        let mut rebased: FxHashMap<(VRef, u32), VRef> = FxHashMap::default();
        for inst in insts {
            let flags = inst.op.flags();
            let single = inst.op == Opc::DsStoreB32 && matches!(inst.uses[0], Opnd::R(_)) && inst.imm & 3 == 0;
            if let (true, Some(p)) = (single, open) {
                let first = out[p].as_ref().expect("open store is live");
                let (o0, o1) = (first.imm as u32, inst.imm as u32);
                let base = o0.min(o1) & !1023;
                if first.uses[0] == inst.uses[0] && o0 != o1 && (o0.max(o1) - base) / 4 <= 255 {
                    let first = out[p].take().expect("open store is live");
                    let mut addr = first.uses[0];
                    if base > 0 {
                        let Opnd::R(a) = addr else { unreachable!("checked above") };
                        let new = *rebased.entry((a, base)).or_insert_with(|| {
                            let reg = f.vregs.len() as u32;
                            f.vregs.push(VRegInfo { class: Class::V, size: 1, align: 1, fixed: None });
                            let v = VRef { reg, off: 0, n: 1 };
                            out.push(Some(MInst::new(Opc::VAddNcU32, [Opnd::R(v)], [Opnd::Imm(base), addr])));
                            v
                        });
                        addr = Opnd::R(new);
                    }
                    let offsets = ((o0 - base) / 4 | (o1 - base) / 4 << 8) as i64;
                    out.push(Some(
                        MInst::new(Opc::DsStore2addrB32, [], [addr, first.uses[1], inst.uses[1]]).with_imm(offsets),
                    ));
                    open = None;
                    continue;
                }
            }
            if let Some(p) = open {
                let first = out[p].as_ref().expect("open store is live");
                let clobbers = inst.defs.iter().any(|d| first.uses.iter().any(|u| overlaps(d, u)));
                if clobbers || flags & (F_LDS | F_DEF_EXEC | F_BRANCH | F_END | F_FENCE) != 0 || inst.op == Opc::SBarrier {
                    open = None;
                }
            }
            if !rebased.is_empty() {
                rebased.retain(|(a, _), _| !inst.defs.iter().any(|d| overlaps(d, &Opnd::R(*a))));
            }
            if single {
                open = Some(out.len());
            }
            out.push(Some(inst));
        }
        f.blocks[b].insts = out.into_iter().flatten().collect();
    }
}

fn find_block_ids(region: &Region, used: &mut [bool; 3]) {
    for op in &region.ops {
        if op.kind == OpKind::BlockId {
            match op.str_attr("axis") {
                None | Some("x") => used[0] = true,
                Some("y") => used[1] = true,
                _ => used[2] = true,
            }
        }
        for inner in &op.regions {
            find_block_ids(inner, used);
        }
    }
}

pub fn select(
    kernel: &Kernel,
    facts: &Facts,
    uniform: Vec<bool>,
    lds: &FxHashMap<ValueId, u32>,
    lds_size: u32,
) -> Result<IselResult, CompileError> {
    select_for(Target::Gfx1151, kernel, facts, uniform, lds, lds_size)
}

/// Selection for `target`; the machine function carries it on.
pub fn select_for(
    target: Target,
    kernel: &Kernel,
    facts: &Facts,
    uniform: Vec<bool>,
    lds: &FxHashMap<ValueId, u32>,
    lds_size: u32,
) -> Result<IselResult, CompileError> {
    let mut wg_ids = [false; 3];
    find_block_ids(&kernel.body, &mut wg_ids);
    // Workgroup id x is always delivered; it anchors the SGPR layout.
    wg_ids[0] = true;
    let mut uses = vec![0u32; kernel.values.len()];
    count_uses(&kernel.body, &mut uses);
    let mut gathers = FxHashSet::default();
    find_gathers(kernel, &kernel.body, &uniform, &uses, &mut gathers);
    let mut defs = FxHashMap::default();
    collect_defs(&kernel.body, &mut defs);
    let mut chains = Chains::default();
    let cx = ChainCx { k: kernel, facts, lds, uses: &uses, defs: &defs };
    find_chains(&cx, &kernel.body, &mut chains);
    if target == Target::Gfx1151 {
        find_dword_quads(&cx, &kernel.body, &gathers, &mut chains);
    }
    find_finite_tests(&cx, &kernel.body, &mut chains);
    find_half_packs(&cx, &kernel.body, &uniform, &mut chains);
    let mut packs = FxHashMap::default();
    find_f16_packs(&cx, &kernel.body, &mut packs, &mut chains.skip);
    let mut vote_sel = FxHashMap::default();
    find_vote_selects(&cx, &kernel.body, &mut vote_sel, &mut chains.skip);
    let mut trunc_mul = FxHashMap::default();
    find_trunc_products(&cx, &kernel.body, &mut trunc_mul, &mut chains.skip);
    let (mut negs, mut fma_uses) = (FxHashMap::default(), FxHashMap::default());
    find_fneg_fma(kernel, &kernel.body, &mut negs, &mut fma_uses);
    let fneg_fma = negs.into_iter().filter(|(v, _)| fma_uses.get(v) == Some(&uses[v.0 as usize])).collect();
    let mut isel = Isel {
        burs: None,
        k: kernel,
        facts,
        uniform,
        f: MFunc { target, ..MFunc::default() },
        cur: 0,
        vals: vec![None; kernel.values.len()],
        lds,
        depth: 0,
        single_wave: kernel.max_workgroup_size <= target.info().wave,
        t: target.info(),
        kernarg_ptr: VRef { reg: 0, off: 0, n: 2 },
        tid: VRef { reg: 0, off: 0, n: 1 },
        bid: [None; 3],
        zero_v: None,
        uses,
        gathers,
        chains,
        deferred: FxHashMap::default(),
        canonical: Vec::new(),
        canon64: FxHashSet::default(),
        lane_sel: None,
        splats: FxHashMap::default(),
        fneg_fma,
        packs,
        vote_sel,
        trunc_mul,
        mem_tags: if super::debug_flag("noalias") { Vec::new() } else { super::alias::value_tags(kernel) },
    };
    if super::debug_flag("burs") {
        isel.burs = Some(burs::plan(&isel));
    }
    isel.f.blocks.push(MBlock::default());
    let kernarg = isel.vreg(Class::S, 2, 2);
    isel.f.vregs[kernarg.reg as usize].fixed = Some(0);
    isel.kernarg_ptr = kernarg;
    let mut next_sgpr = 2u16;
    for axis in 0..3 {
        if wg_ids[axis] {
            let fixed = isel.vreg(Class::S, 1, 1);
            isel.f.vregs[fixed.reg as usize].fixed = Some(next_sgpr);
            next_sgpr += 1;
            let copy = isel.sreg(1);
            isel.copy(copy, r(fixed));
            isel.bid[axis] = Some(copy);
        }
    }
    let tid_fixed = isel.vreg(Class::V, 1, 1);
    isel.f.vregs[tid_fixed.reg as usize].fixed = Some(0);
    let tid = isel.vgpr(1);
    isel.copy(tid, r(tid_fixed));
    isel.tid = tid;
    isel.region(&kernel.body)?;
    let last = isel.f.blocks.last().and_then(|b| b.insts.last()).map(|i| i.op);
    if last != Some(Opc::SEndpgm) {
        isel.emit(MInst::new(Opc::SEndpgm, [], []));
    }
    pair_lds_stores(&mut isel.f);
    if let Some(z) = isel.zero_v {
        isel.f.blocks[0].insts.insert(0, MInst::new(Opc::Copy, [r(z)], [Opnd::Imm(0)]));
    }
    if let Some(s) = isel.lane_sel {
        isel.f.blocks[0].insts.insert(0, MInst::new(Opc::Copy, [r(s)], [Opnd::Imm(PERMLANEX16_SEL.0)]));
    }
    let (_, kernarg_size) = super::opt::kernarg_layout(kernel);
    Ok(IselResult {
        func: isel.f,
        kernarg_size,
        lds_size,
        wg_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn insts(text: &str) -> Vec<MInst> {
        let compiled = super::super::compile(text).unwrap_or_else(|e| panic!("{e}"));
        compiled.func.blocks.into_iter().flat_map(|b| b.insts).collect()
    }

    fn kernel(params: &str, body: &str) -> String {
        format!(
            "ckdsl.ir v1\nkernel @probe {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n{params}  }}\n  \
             region @entry {{\n    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n{body}    cf.return (  )\n  }}\n}}\n"
        )
    }

    #[test]
    fn fmax_quiets_only_sources_that_may_be_signaling() {
        let params = "    %X : ptr<f32,global>\n    %Y : ptr<f32,global>\n    %s : f32\n";
        let body = r#"    %a : f32 = memref.global_load_typed ( %X, %tid ) { align = i:4, elem_type = s:"f32" }
    %c64 : i32 = arith.constant (  ) { ity = s:"i32", value = i:64 }
    %t2 : i32 = arith.add ( %tid, %c64 )
    %b : f32 = memref.global_load_typed ( %X, %t2 ) { align = i:4, elem_type = s:"f32" }
    %mul : f32 = arith.fmul ( %a, %b )
    %bc : i32 = arith.bitcast ( %mul ) { target = s:"i32" }
    %sw : i32 = tile.ds_swizzle_xor ( %bc ) { xor_mask = i:1 }
    %swf : f32 = arith.bitcast ( %sw ) { target = s:"f32" }
    %snan : i32 = arith.constant (  ) { ity = s:"i32", value = i:2141192192 }
    %snanf : f32 = arith.bitcast ( %snan ) { target = s:"f32" }
    %m1 : f32 = arith.fmax ( %a, %b )
    %m2 : f32 = arith.fmax ( %mul, %swf )
    %m3 : f32 = arith.fmax ( %s, %mul )
    %m4 : f32 = arith.fmax ( %snanf, %mul )
    %m5 : f32 = arith.fmax ( %a, %a )
    %r1 : f32 = arith.fadd ( %m1, %m2 )
    %r2 : f32 = arith.fadd ( %m3, %m4 )
    %r3 : f32 = arith.fadd ( %r1, %r2 )
    %r4 : f32 = arith.fadd ( %r3, %m5 )
    memref.global_store_typed ( %Y, %tid, %r4 ) { align = i:4, elem_type = s:"f32" }
"#;
        let insts = insts(&kernel(params, body));
        // Sources of every f32 max, including the halves of VOPD pairs and
        // the two maxes of a v_max3_f32 (the peephole pass folds a max into
        // the max reading it): max(max(s0, s1), s2), whose outer max reads
        // the inner result and s2.
        let mut maxes: Vec<&[Opnd]> = Vec::new();
        for i in &insts {
            if i.op == Opc::VMaxF32 {
                maxes.push(&i.uses);
            } else if i.op == Opc::VMax3F32 {
                maxes.push(&i.uses[..2]);
                maxes.push(&i.uses[2..]);
            } else if i.op == Opc::VDual {
                let (x, y) = i.uses.split_at(i.mods as usize);
                for (code, srcs) in [(i.imm & 0xff, x), (i.imm >> 8 & 0xff, y)] {
                    if vopd_name(code as u8) == "v_dual_max_f32" {
                        maxes.push(srcs);
                    }
                }
            }
        }
        let quiets = maxes.iter().filter(|s| s.len() == 2 && s[0] == s[1]).count();
        // %a and %b for m1, the argument %s for m3; m2's sources are
        // arithmetic, m4's signaling constant folds, m5 is %a itself.
        assert_eq!((maxes.len(), quiets), (7, 3));
        assert!(maxes.iter().any(|s| s.contains(&Opnd::Imm(0x7fc0_0000))));
        assert!(!maxes.iter().any(|s| s.contains(&Opnd::Imm(0x7fa0_0000))));
    }

    fn count(insts: &[MInst], op: Opc) -> usize {
        insts.iter().filter(|i| i.op == op).count()
    }

    #[test]
    fn uniform_i64_clamp_selects_register_pairs() {
        let params = "    %Y : ptr<f32,global>\n    %a : i64\n    %b : i64\n";
        let body = r#"    %c0 : i64 = arith.constant (  ) { ity = s:"i64", value = i:0 }
    %lt : i1 = arith.cmp ( %a, %b ) { pred = s:"lt" }
    %mn : i64 = arith.select ( %lt, %a, %b )
    %neg : i1 = arith.cmp ( %mn, %c0 ) { pred = s:"lt" }
    %cl : i64 = arith.select ( %neg, %c0, %mn )
    %t : i32 = arith.trunc ( %cl )
    %f : f32 = arith.constant (  ) { ity = s:"f32", value = f:2.0 }
    memref.global_store_typed ( %Y, %t, %f ) { align = i:4, elem_type = s:"f32" }
"#;
        let insts = insts(&kernel(params, body));
        // a < b needs the VALU; the sign test reads the high word.
        assert_eq!(count(&insts, Opc::VCmp(0x51)), 1);
        assert_eq!(count(&insts, Opc::SCmpLtI32), 1);
        assert_eq!((count(&insts, Opc::SCselectB64), count(&insts, Opc::SCselectB32)), (2, 0));
    }

    #[test]
    fn finiteness_test_is_one_class_compare() {
        let params = "    %X : ptr<f32,global>\n    %Y : ptr<f32,global>\n";
        let body = r#"    %x : f32 = memref.global_load_typed ( %X, %tid ) { align = i:4, elem_type = s:"f32" }
    %c64 : i32 = arith.constant (  ) { ity = s:"i32", value = i:64 }
    %t2 : i32 = arith.add ( %tid, %c64 )
    %y : f32 = memref.global_load_typed ( %X, %t2 ) { align = i:4, elem_type = s:"f32" }
    %ninf : f32 = arith.constant (  ) { ity = s:"f32", value = f:-inf }
    %pinf : f32 = arith.constant (  ) { ity = s:"f32", value = f:inf }
    %zero : f32 = arith.constant (  ) { ity = s:"f32", value = f:0.0 }
    %gt : i1 = arith.fcmp ( %x, %ninf ) { pred = s:"ogt" }
    %lt : i1 = arith.fcmp ( %x, %pinf ) { pred = s:"olt" }
    %fin : i1 = arith.and ( %lt, %gt )
    %r : f32 = arith.select ( %fin, %x, %zero )
    %gty : i1 = arith.fcmp ( %y, %ninf ) { pred = s:"ogt" }
    %lty : i1 = arith.fcmp ( %y, %pinf ) { pred = s:"olt" }
    %finy : i1 = arith.and ( %gty, %lty )
    %r2 : f32 = arith.select ( %finy, %y, %zero )
    %r3 : f32 = arith.select ( %gty, %r2, %r )
    memref.global_store_typed ( %Y, %tid, %r3 ) { align = i:4, elem_type = s:"f32" }
"#;
        let insts = insts(&kernel(params, body));
        let class: Vec<&MInst> = insts.iter().filter(|i| i.op == Opc::VCmp(VCMP_CLASS_F32)).collect();
        assert_eq!(class.len(), 1);
        assert_eq!(class[0].uses[1], Opnd::Imm(0x1f8));
        // %gty has a second use, so y's test keeps its two compares.
        assert_eq!((count(&insts, Opc::VCmp(0x14)), count(&insts, Opc::VCmp(0x11))), (1, 1));
    }
}

/// Element types beyond f16/f32 (census W02, W09, W17, W18, W19).
#[cfg(test)]
mod types_tests {
    use super::*;

    fn insts(text: &str) -> Vec<MInst> {
        let compiled = super::super::compile(text).unwrap_or_else(|e| panic!("{e}"));
        compiled.func.blocks.into_iter().flat_map(|b| b.insts).collect()
    }

    fn kernel(params: &str, body: &str) -> String {
        format!(
            "ckdsl.ir v1\nkernel @probe {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n{params}  }}\n  \
             region @entry {{\n    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n{body}    cf.return (  )\n  }}\n}}\n"
        )
    }

    fn count(insts: &[MInst], op: Opc) -> usize {
        insts.iter().filter(|i| i.op == op).count()
    }

    /// Mnemonic and sources of every operation, the halves of VOPD pairs
    /// included (`v_dual_lshlrev_b32` counts as `v_lshlrev_b32`).
    fn ops(insts: &[MInst]) -> Vec<(String, Vec<Opnd>)> {
        let mut out = Vec::new();
        for i in insts {
            if i.op == Opc::VDual {
                let (x, y) = i.uses.split_at(i.mods as usize);
                for (code, srcs) in [(i.imm & 0xff, x), (i.imm >> 8 & 0xff, y)] {
                    out.push((vopd_name(code as u8).replace("v_dual_", "v_"), srcs.to_vec()));
                }
            } else {
                out.push((i.op.name(), i.uses.to_vec()));
            }
        }
        out
    }

    #[test]
    fn bf16_rounding_matches_llvm_fptrunc() {
        // Ties go to even, carries reach the exponent, NaNs keep sign and payload.
        let cases: [(u32, u16); 12] = [
            (0x3f80_0000, 0x3f80),
            (0x3f80_8000, 0x3f80),
            (0x3f81_8000, 0x3f82),
            (0x3f80_8001, 0x3f81),
            (0x3f80_7fff, 0x3f80),
            (0x7f7f_ffff, 0x7f80),
            (0xff7f_8000, 0xff80),
            (0x7f80_0000, 0x7f80),
            (0x7f80_0001, 0x7fc0),
            (0xffa0_ffff, 0xffe0),
            (0x8000_8000, 0x8000),
            (0x0000_c000, 0x0001),
        ];
        for (bits, want) in cases {
            assert_eq!(f32_to_bf16_bits(f32::from_bits(bits)), want, "{bits:#x}");
        }
    }

    #[test]
    fn bf16_values_live_in_half_registers() {
        let params = "    %X : ptr<f32,global>\n    %B : ptr<bf16,global>\n    %Y : ptr<bf16,global>\n    %Z : ptr<f32,global>\n";
        let body = r#"    %x : f32 = memref.global_load_typed ( %X, %tid ) { align = i:4, elem_type = s:"f32" }
    %b1 : bf16 = arith.cast_f32_to ( %x ) { target = s:"bf16" }
    memref.global_store_typed ( %Y, %tid, %b1 ) { align = i:2, elem_type = s:"bf16" }
    %b : bf16 = memref.global_load_typed ( %B, %tid ) { align = i:2, elem_type = s:"bf16" }
    %f : f32 = arith.cast_to_f32 ( %b )
    memref.global_store_typed ( %Z, %tid, %f ) { align = i:4, elem_type = s:"f32" }
    %c8 : i32 = arith.constant (  ) { ity = s:"i32", value = i:8 }
    %row : i32 = arith.mul ( %tid, %c8 )
    %v : vec<bf16x8> = memref.global_load_vN ( %B, %row ) { align = i:16, elem_type = s:"bf16", vec = i:8 }
    %e3 : bf16 = vector.extract ( %v ) { index = i:3 }
    %f3 : f32 = arith.cast_to_f32 ( %e3 )
    memref.global_store_typed ( %Z, %row, %f3 ) { align = i:4, elem_type = s:"f32" }
"#;
        let insts = insts(&kernel(params, body));
        // The rounded bf16 is the high half: stored with the d16_hi form.
        assert_eq!(count(&insts, Opc::GStoreD16HiB16), 1);
        assert_eq!(count(&insts, Opc::VCmp(0x18)), 1);
        assert_eq!(count(&insts, Opc::VBfeU32), 1);
        assert!(insts.iter().any(|i| i.op == Opc::VAdd3U32 && i.uses[2] == Opnd::Imm(0x7fff)));
        // A loaded bf16 widens by a shift; an odd element by masking its dword.
        let ops = ops(&insts);
        assert!(ops.iter().any(|(n, u)| n == "v_lshlrev_b32" && u[0] == Opnd::Imm(16)));
        assert!(ops.iter().any(|(n, u)| n == "v_and_b32" && u.contains(&Opnd::Imm(0xffff_0000))));
        assert_eq!(count(&insts, Opc::VCvtF32F16) + count(&insts, Opc::VCvtF16F32), 0);
    }

    #[test]
    fn bf16_vectors_feed_wmma_and_buffer_ops() {
        let params = "    %X : ptr<bf16,global>\n    %Y : ptr<f32,global>\n    %W : ptr<bf16,global>\n    %n : i32\n";
        let body = r#"    %c16 : i32 = arith.constant (  ) { ity = s:"i32", value = i:16 }
    %c0 : i32 = arith.constant (  ) { ity = s:"i32", value = i:0 }
    %row : i32 = arith.mul ( %tid, %c16 )
    %a : vec<bf16x16> = memref.global_load_vN ( %X, %row ) { align = i:32, elem_type = s:"bf16", vec = i:16 }
    %z : vec<bf16x16> = arith.constant_vec (  ) { elem = s:"bf16", fill = f:0.0, vec = i:16 }
    %cz : vec<f32x8> = arith.constant_vec (  ) { elem = s:"f32", fill = f:0.0, vec = i:8 }
    %acc : vec<f32x8> = tile.mma ( %a, %a, %cz ) { op_id = s:"wmma_f32_16x16x16_bf16" }
    %acc2 : vec<f32x8> = tile.mma ( %a, %z, %acc ) { op_id = s:"wmma_f32_16x16x16_bf16" }
    %c8 : i32 = arith.constant (  ) { ity = s:"i32", value = i:8 }
    %r8 : i32 = arith.mul ( %tid, %c8 )
    memref.global_store_vN ( %Y, %r8, %acc2 ) { align = i:32, elem_type = s:"f32", vec = i:8 }
    %t : vec<bf16x8> = vector.trunc_f32_to ( %acc2 ) { target = s:"bf16" }
    memref.global_store_vN ( %W, %r8, %t ) { align = i:16, elem_type = s:"bf16", vec = i:8 }
    %rs : vec<i32x4> = tile.buffer_rsrc ( %W, %n )
    %c2 : i32 = arith.constant (  ) { ity = s:"i32", value = i:2 }
    %bo : i32 = arith.mul ( %tid, %c2 )
    %bl : bf16 = tile.buffer_load_bf16 ( %rs, %bo, %c0 )
    %blf : f32 = arith.cast_to_f32 ( %bl )
    %two : f32 = arith.constant (  ) { ity = s:"f32", value = f:2.0 }
    %dbl : f32 = arith.fmul ( %blf, %two )
    %h : bf16 = arith.trunc_f32_to_bf16 ( %dbl )
    tile.buffer_store_bf16 ( %rs, %bo, %c0, %h )
    %bv : vec<bf16x8> = tile.buffer_load_vN_bf16 ( %rs, %bo, %c0 ) { dwords = i:4 }
    tile.buffer_store_vN_bf16 ( %rs, %bo, %c0, %bv ) { dwords = i:4 }
"#;
        let insts = insts(&kernel(params, body));
        let wmma = count(&insts, Opc::VWmmaF32Bf16) + count(&insts, Opc::VWmmaF32Bf16Early);
        assert_eq!((wmma, count(&insts, Opc::VWmmaF32F16) + count(&insts, Opc::VWmmaF32F16Early)), (2, 0));
        // Eight roundings packed in pairs; then one more for the buffer value.
        assert_eq!(count(&insts, Opc::VCmp(0x18)), 9);
        assert_eq!(count(&insts, Opc::BufLoadU16), 1);
        assert_eq!(count(&insts, Opc::BufStoreD16HiB16), 1);
        assert_eq!((count(&insts, Opc::BufLoadB128), count(&insts, Opc::BufStoreB128)), (1, 1));
    }

    #[test]
    fn half_round_trip_through_f32_is_a_move() {
        // LLVM folds fptrunc(fpext(x)) to x; converting would quiet a
        // signaling NaN that a transpose must copy bit for bit.
        let params = "    %B : ptr<bf16,global>\n    %Y : ptr<bf16,global>\n    %H : ptr<f16,global>\n    %G : ptr<f16,global>\n";
        let body = r#"    %b : bf16 = memref.global_load_typed ( %B, %tid ) { align = i:2, elem_type = s:"bf16" }
    %f : f32 = arith.cast_to_f32 ( %b )
    %r : bf16 = arith.cast_f32_to ( %f ) { target = s:"bf16" }
    memref.global_store_typed ( %Y, %tid, %r ) { align = i:2, elem_type = s:"bf16" }
    %t : bf16 = arith.trunc_f32_to_bf16 ( %f )
    memref.global_store_typed ( %Y, %tid, %t ) { align = i:2, elem_type = s:"bf16" }
    %h : f16 = memref.global_load_typed ( %H, %tid ) { align = i:2, elem_type = s:"f16" }
    %hf : f32 = arith.cast_to_f32 ( %h )
    %hr : f16 = arith.cast_f32_to ( %hf ) { target = s:"f16" }
    memref.global_store_typed ( %G, %tid, %hr ) { align = i:2, elem_type = s:"f16" }
"#;
        let insts = insts(&kernel(params, body));
        let converts = count(&insts, Opc::VCmp(0x18)) + count(&insts, Opc::VCvtF32F16) + count(&insts, Opc::VCvtF16F32);
        assert_eq!(converts, 0);
        assert_eq!((count(&insts, Opc::GLoadU16), count(&insts, Opc::GStoreB16)), (2, 3));
    }

    #[test]
    fn i16_views_of_two_halves_pack_with_one_permute() {
        // or(zext(bitcast a), shl(zext(bitcast b), 16)) packs two 16-bit
        // values: one v_perm_b32, as LLVM selects it.
        let params = "    %X : ptr<f32,global>\n    %Y : ptr<i32,global>\n";
        let body = r#"    %c2 : i32 = arith.constant (  ) { ity = s:"i32", value = i:2 }
    %c1 : i32 = arith.constant (  ) { ity = s:"i32", value = i:1 }
    %c16 : i32 = arith.constant (  ) { ity = s:"i32", value = i:16 }
    %i0 : i32 = arith.mul ( %tid, %c2 )
    %i1 : i32 = arith.add ( %i0, %c1 )
    %x0 : f32 = memref.global_load_typed ( %X, %i0 ) { align = i:4, elem_type = s:"f32" }
    %x1 : f32 = memref.global_load_typed ( %X, %i1 ) { align = i:4, elem_type = s:"f32" }
    %b0 : bf16 = arith.cast_f32_to ( %x0 ) { target = s:"bf16" }
    %b1 : bf16 = arith.cast_f32_to ( %x1 ) { target = s:"bf16" }
    %h0 : i16 = arith.bitcast ( %b0 ) { target = s:"i16" }
    %h1 : i16 = arith.bitcast ( %b1 ) { target = s:"i16" }
    %z0 : i32 = arith.zext ( %h0 )
    %z1 : i32 = arith.zext ( %h1 )
    %s1 : i32 = arith.shl ( %z1, %c16 )
    %p : i32 = arith.or ( %z0, %s1 )
    memref.global_store_typed ( %Y, %tid, %p ) { align = i:4, elem_type = s:"i32" }
"#;
        let insts = insts(&kernel(params, body));
        let perms: Vec<&MInst> = insts.iter().filter(|i| i.op == Opc::VPermB32).collect();
        assert_eq!(perms.len(), 1);
        // Both rounded bf16 values are high halves: bytes 2-3 of each source.
        assert_eq!(perms[0].uses[2], Opnd::Imm(0x0706_0302));
        // (The v_or_b32 left are the NaN quieting of the two roundings.)
        assert_eq!(count(&insts, Opc::VLshlOrB32) + count(&insts, Opc::VLshrrevB32), 0);
        assert_eq!(count(&insts, Opc::VOrB32), 2);
    }

    /// A divergent i32 `or` with a constant operand, whose definition has no
    /// operands, is no half pack (once an index out of bounds).
    #[test]
    fn or_with_a_constant_is_no_half_pack() {
        let params = "    %Y : ptr<i32,global>\n";
        let body = r#"    %c16 : i32 = arith.constant (  ) { ity = s:"i32", value = i:128 }
    %o : i32 = arith.or ( %tid, %c16 )
    memref.global_store_typed ( %Y, %tid, %o ) { align = i:4, elem_type = s:"i32" }
"#;
        let code = insts(&kernel(params, body));
        assert_eq!(count(&code, Opc::VPermB32), 0);
    }

    #[test]
    fn i16_widens_and_narrows() {
        let params = "    %H : ptr<i16,global>\n    %Y : ptr<i32,global>\n    %Z : ptr<i16,global>\n    %F : ptr<f16,global>\n";
        let body = r#"    %h : i16 = memref.global_load_typed ( %H, %tid ) { align = i:2, elem_type = s:"i16" }
    %hz : i32 = arith.zext ( %h )
    %hs : i32 = arith.sext ( %h )
    %c3 : i32 = arith.constant (  ) { ity = s:"i32", value = i:3 }
    %c5 : i32 = arith.constant (  ) { ity = s:"i32", value = i:5 }
    %i1 : i32 = arith.mul ( %tid, %c3 )
    %i2 : i32 = arith.add ( %i1, %c5 )
    memref.global_store_typed ( %Y, %i1, %hz ) { align = i:4, elem_type = s:"i32" }
    memref.global_store_typed ( %Y, %i2, %hs ) { align = i:4, elem_type = s:"i32" }
    %big : i32 = arith.constant (  ) { ity = s:"i32", value = i:70000 }
    %sum : i32 = arith.add ( %hs, %big )
    %t : i16 = arith.trunc ( %sum )
    memref.global_store_typed ( %Z, %tid, %t ) { align = i:2, elem_type = s:"i16" }
    %f : f16 = arith.bitcast ( %h ) { target = s:"f16" }
    memref.global_store_typed ( %F, %tid, %f ) { align = i:2, elem_type = s:"f16" }
    %c7 : i32 = arith.constant (  ) { ity = s:"i32", value = i:7 }
    %c9 : i32 = arith.constant (  ) { ity = s:"i32", value = i:9 }
    %i7 : i32 = arith.add ( %i1, %c7 )
    %i9 : i32 = arith.add ( %i1, %c9 )
    %tc : i16 = arith.trunc ( %big )
    %tcz : i32 = arith.zext ( %tc )
    memref.global_store_typed ( %Y, %i7, %tcz ) { align = i:4, elem_type = s:"i32" }
    %ones : i32 = arith.constant (  ) { ity = s:"i32", value = i:131071 }
    %to : i16 = arith.trunc ( %ones )
    %toz : i32 = arith.zext ( %to )
    memref.global_store_typed ( %Y, %i9, %toz ) { align = i:4, elem_type = s:"i32" }
"#;
        let insts = insts(&kernel(params, body));
        let ops = ops(&insts);
        assert!(ops.iter().any(|(n, u)| n == "v_and_b32" && u.contains(&Opnd::Imm(0xffff))));
        assert!(ops.iter().any(|(n, u)| n == "v_bfe_i32" && u[1..] == [Opnd::Imm(0), Opnd::Imm(16)]));
        assert_eq!((count(&insts, Opc::GLoadU16), count(&insts, Opc::GStoreB16)), (1, 2));
        // Folded 16-bit constants wrap: zext(trunc(70000)) == 4464 and
        // zext(trunc(0x1ffff)) == 0xffff.
        let imms: Vec<Opnd> = insts.iter().flat_map(|i| i.uses.to_vec()).collect();
        assert!(imms.contains(&Opnd::Imm(4464)) && imms.contains(&Opnd::Imm(0xffff)));
        assert!(!imms.contains(&Opnd::Imm(131071)));
    }

    #[test]
    fn i8_bytes_extract_widen_pack_and_store() {
        let params = "    %S : ptr<i8,global>\n    %Y : ptr<i32,global>\n    %Q : ptr<i8,global>\n    %X : ptr<f32,global>\n";
        let body = r#"    %c16 : i32 = arith.constant (  ) { ity = s:"i32", value = i:16 }
    %c4 : i32 = arith.constant (  ) { ity = s:"i32", value = i:4 }
    %row : i32 = arith.mul ( %tid, %c16 )
    %v : vec<i8x16> = memref.global_load_vN ( %S, %row ) { align = i:16, elem_type = s:"i8", vec = i:16 }
    %e5 : i8 = vector.extract ( %v ) { index = i:5 }
    %e10 : i8 = vector.extract ( %v ) { index = i:10 }
    %e15 : i8 = vector.extract ( %v ) { index = i:15 }
    %s5 : i32 = arith.sext ( %e5 )
    %z15 : i32 = arith.zext ( %e15 )
    %sum : i32 = arith.add ( %s5, %z15 )
    memref.global_store_typed ( %Y, %tid, %sum ) { align = i:4, elem_type = s:"i32" }
    memref.global_store_typed ( %Q, %tid, %e10 ) { align = i:1, elem_type = s:"i8" }
    %b : i8 = memref.global_load_typed ( %S, %tid ) { align = i:1, elem_type = s:"i8" }
    %r4 : i32 = arith.mul ( %tid, %c4 )
    %x : f32 = memref.global_load_typed ( %X, %tid ) { align = i:4, elem_type = s:"f32" }
    %q : i8 = arith.cvt_f32_to_i8_sat ( %x )
    %p : vec<i8x4> = vector.pack ( %b, %e5, %e10, %q ) { elem = s:"i8", vec = i:4 }
    memref.global_store_vN ( %Q, %r4, %p ) { align = i:4, elem_type = s:"i8", vec = i:4 }
"#;
        let insts = insts(&kernel(params, body));
        // Byte 5 is byte 1 of dword 1: one signed field extract; byte 15 is
        // the top byte: one shift. Byte 10 (2 of its dword) stores d16_hi.
        assert!(insts.iter().any(|i| i.op == Opc::VBfeI32 && i.uses[1..] == [Opnd::Imm(8), Opnd::Imm(8)]));
        assert!(ops(&insts).iter().any(|(n, u)| n == "v_lshrrev_b32" && u[0] == Opnd::Imm(24)));
        assert_eq!((count(&insts, Opc::GStoreD16HiB8), count(&insts, Opc::GLoadU8)), (1, 1));
        // f32 -> i8 saturates as LLVM selects it: rndne, cvt, med3.
        assert_eq!((count(&insts, Opc::VRndneF32), count(&insts, Opc::VCvtI32F32), count(&insts, Opc::VMed3I32)), (1, 1, 1));
        // Four bytes from three registers: two pair permutes and their merge.
        assert_eq!(count(&insts, Opc::VPermB32), 3);
    }

    #[test]
    fn integer_wmma_marks_both_operands_signed() {
        let params = "    %A : ptr<i32,global>\n    %C : ptr<i32,global>\n";
        let body = r#"    %c4 : i32 = arith.constant (  ) { ity = s:"i32", value = i:4 }
    %c8 : i32 = arith.constant (  ) { ity = s:"i32", value = i:8 }
    %r4 : i32 = arith.mul ( %tid, %c4 )
    %r8 : i32 = arith.mul ( %tid, %c8 )
    %a : vec<i32x4> = memref.global_load_vN ( %A, %r4 ) { align = i:16, elem_type = s:"i32", vec = i:4 }
    %z : vec<i32x8> = arith.constant_vec (  ) { elem = s:"i32", fill = f:0.0, vec = i:8 }
    %d : vec<i32x8> = tile.mma ( %a, %a, %z ) { op_id = s:"wmma_i32_16x16x16_iu8" }
    memref.global_store_vN ( %C, %r8, %d ) { align = i:32, elem_type = s:"i32", vec = i:8 }
"#;
        let compiled = super::super::compile(&kernel(params, body)).unwrap_or_else(|e| panic!("{e}"));
        let listing = compiled.listing();
        let line = listing.lines().find(|l| l.contains("v_wmma_i32_16x16x16_iu8")).expect("one integer WMMA");
        let words: Vec<u32> = line.rsplit(": ").next().unwrap().split_whitespace().map(|w| u32::from_str_radix(w, 16).unwrap()).collect();
        // VOP3P opcode 0x44, op_sel_hi all set, neg_lo:[1,1,0] (signed A and B).
        assert_eq!((words[0] >> 16 & 0xff, words[0] >> 14 & 1), (0x44, 1));
        assert_eq!(words[1] >> 27, 0b01111);
    }

    #[test]
    fn byte_and_bool_memory_round_trips() {
        // fp8/bf8 are byte storage; an i1 in memory is a byte whose low bit
        // is the value; the empty asm barrier keeps an i64 whole.
        let params = "    %P : ptr<i1,global>\n    %F : ptr<fp8e4m3,global>\n    %G : ptr<bf8e5m2,global>\n    %W : ptr<i64,global>\n";
        let body = r#"    %b : i1 = memref.global_load_typed ( %P, %tid ) { align = i:1, elem_type = s:"i1" }
    %z : i32 = arith.zext ( %b )
    %a : i32 = tile.inline_asm ( %z ) { constraints = s:"=v,0", convergent = b:false, sideeffect = b:false, template = s:"" }
    %t : i1 = arith.trunc ( %a )
    memref.global_store_typed ( %P, %tid, %t ) { align = i:1, elem_type = s:"i1" }
    %f : fp8e4m3 = memref.global_load_typed ( %F, %tid ) { align = i:1, elem_type = s:"fp8e4m3" }
    %fi : i8 = arith.bitcast ( %f ) { target = s:"i8" }
    %fz : i32 = arith.zext ( %fi )
    %ft : i8 = arith.trunc ( %fz )
    %g : bf8e5m2 = arith.bitcast ( %ft ) { target = s:"bf8e5m2" }
    memref.global_store_typed ( %G, %tid, %g ) { align = i:1, elem_type = s:"bf8e5m2" }
    %w : i64 = memref.global_load_typed ( %W, %tid ) { align = i:8, elem_type = s:"i64" }
    %wa : i64 = tile.inline_asm ( %w ) { constraints = s:"=v,0", convergent = b:false, sideeffect = b:false, template = s:"" }
    memref.global_store_typed ( %W, %tid, %wa ) { align = i:8, elem_type = s:"i64" }
"#;
        let insts = insts(&kernel(params, body));
        assert_eq!((count(&insts, Opc::GLoadU8), count(&insts, Opc::GStoreB8)), (2, 2));
        assert_eq!(count(&insts, Opc::VCmp(0x4d)), 2);
        assert_eq!((count(&insts, Opc::GLoadB64), count(&insts, Opc::GStoreB64)), (1, 1));
    }

    #[test]
    fn f64_values_are_register_pairs() {
        let params = "    %X : ptr<f64,global>\n    %Y : ptr<f64,global>\n";
        let body = r#"    %c2 : i32 = arith.constant (  ) { ity = s:"i32", value = i:2 }
    %row : i32 = arith.mul ( %tid, %c2 )
    %v : vec<f64x2> = memref.global_load_vN ( %X, %row ) { align = i:16, elem_type = s:"f64", vec = i:2 }
    %a : f64 = vector.extract ( %v ) { index = i:0 }
    %b : f64 = vector.extract ( %v ) { index = i:1 }
    %k : f64 = arith.constant (  ) { ity = s:"f64", value = f:1.5 }
    %s : f64 = arith.fadd ( %a, %k )
    %d : f64 = arith.fsub ( %s, %b )
    %p : f64 = arith.fmul ( %d, %d )
    %m : f64 = arith.fmax ( %p, %a )
    %w : vec<i32x2> = arith.bitcast ( %m ) { target = s:"vec<i32x2>" }
    %hi : i32 = vector.extract ( %w ) { index = i:1 }
    %back : f64 = arith.bitcast ( %w ) { target = s:"f64" }
    memref.global_store_typed ( %Y, %tid, %back ) { align = i:8, elem_type = s:"f64" }
    %lds : smem<f64, [64]> = tile.smem_alloc (  )
    tile.smem_store ( %lds, %tid, %m ) { elem_type = s:"f64", rank = i:1 }
    tile.sync (  ) { fence = s:"compiler" }
    %l : vec<f64x1> = tile.smem_load_vN ( %lds, %tid ) { elem_type = s:"f64", rank = i:1, vec = i:1 }
    %le : f64 = vector.extract ( %l ) { index = i:0 }
    %q : vec<f64x2> = vector.pack ( %le, %a ) { elem = s:"f64", vec = i:2 }
    memref.global_store_vN ( %Y, %row, %q ) { align = i:16, elem_type = s:"f64", vec = i:2 }
    %hf : f32 = arith.bitcast ( %hi ) { target = s:"f32" }
"#;
        let insts = insts(&kernel(params, body));
        // fsub is an add of the negated second source.
        let adds: Vec<&MInst> = insts.iter().filter(|i| i.op == Opc::VAddF64).collect();
        assert_eq!(adds.len(), 2);
        assert!(adds.iter().any(|i| i.mods == M_NEG1));
        // %p is f64 arithmetic (quiet); the loaded %a is quieted for maxnum.
        let maxes: Vec<&MInst> = insts.iter().filter(|i| i.op == Opc::VMaxF64).collect();
        assert_eq!((maxes.len(), maxes.iter().filter(|i| i.uses[0] == i.uses[1]).count()), (2, 1));
        assert_eq!((count(&insts, Opc::VMulF64), count(&insts, Opc::DsStoreB64), count(&insts, Opc::DsLoadB64)), (1, 1, 1));
        assert_eq!((count(&insts, Opc::GStoreB64), count(&insts, Opc::GStoreB128)), (1, 1));
    }

    #[test]
    fn f64_division_minnum_and_sign() {
        // A divisor with an exact reciprocal folds as LLVM folds it; 3.0 takes
        // the full sequence.
        let params = "    %X : ptr<f64,global>\n    %Y : ptr<f64,global>\n";
        let body = r#"    %x : f64 = memref.global_load_typed ( %X, %tid ) { align = i:8, elem_type = s:"f64" }
    %c1 : i32 = arith.constant (  ) { ity = s:"i32", value = i:1 }
    %c8 : i32 = arith.constant (  ) { ity = s:"i32", value = i:8 }
    %i0 : i32 = arith.mul ( %tid, %c8 )
    %i1 : i32 = arith.add ( %i0, %c1 )
    %i2 : i32 = arith.add ( %i1, %c1 )
    %i3 : i32 = arith.add ( %i2, %c1 )
    %i4 : i32 = arith.add ( %i3, %c1 )
    %km : f64 = arith.constant (  ) { ity = s:"f64", value = f:-2048.0 }
    %kq : f64 = arith.constant (  ) { ity = s:"f64", value = f:0.25 }
    %kh : f64 = arith.constant (  ) { ity = s:"f64", value = f:0.5 }
    %k1 : f64 = arith.constant (  ) { ity = s:"f64", value = f:1.0 }
    %k3 : f64 = arith.constant (  ) { ity = s:"f64", value = f:3.0 }
    %d0 : f64 = arith.fdiv ( %x, %km )
    %d1 : f64 = arith.fdiv ( %x, %kq )
    %d2 : f64 = arith.fdiv ( %x, %kh )
    %d3 : f64 = arith.fdiv ( %x, %k1 )
    %d4 : f64 = arith.fdiv ( %x, %k3 )
    %a : f64 = arith.fabs ( %d4 )
    %n : f64 = arith.fneg ( %x )
    %m : f64 = arith.fmin ( %a, %n )
    memref.global_store_typed ( %Y, %i0, %d0 ) { align = i:8, elem_type = s:"f64" }
    memref.global_store_typed ( %Y, %i1, %d1 ) { align = i:8, elem_type = s:"f64" }
    memref.global_store_typed ( %Y, %i2, %d2 ) { align = i:8, elem_type = s:"f64" }
    memref.global_store_typed ( %Y, %i3, %d3 ) { align = i:8, elem_type = s:"f64" }
    memref.global_store_typed ( %Y, %i4, %m ) { align = i:8, elem_type = s:"f64" }
"#;
        let insts = insts(&kernel(params, body));
        // x / -2048 is v_ldexp_f64 -x, -11.
        let ldexp: Vec<&MInst> = insts.iter().filter(|i| i.op == Opc::VLdexpF64).collect();
        assert_eq!(ldexp.len(), 1);
        assert_eq!((ldexp[0].mods, ldexp[0].uses[1]), (M_NEG0, Opnd::Imm(-11i32 as u32)));
        // x / 0.25 is x * 4, x / 0.5 is x + x, x / 1 is x; the sequence has one multiply.
        assert_eq!(count(&insts, Opc::VMulF64), 2);
        let adds: Vec<&MInst> = insts.iter().filter(|i| i.op == Opc::VAddF64).collect();
        assert!(adds.len() == 1 && adds[0].uses[0] == adds[0].uses[1]);
        assert_eq!((count(&insts, Opc::VDivScaleF64), count(&insts, Opc::VRcpF64), count(&insts, Opc::VFmaF64)), (2, 1, 5));
        assert_eq!((count(&insts, Opc::VDivFmasF64), count(&insts, Opc::VDivFixupF64)), (1, 1));
        // fabs of a quotient is still quiet; the negated load is quieted for minnum.
        assert_eq!((count(&insts, Opc::VMinF64), count(&insts, Opc::VMaxF64)), (1, 1));
        let ops = ops(&insts);
        let named = |name: &str| ops.iter().filter(|(n, _)| n == name).count();
        assert_eq!((named("v_and_b32"), named("v_xor_b32")), (1, 1));
    }

    #[test]
    fn f64_constants_are_inline_constants_or_literals() {
        // A constant with a zero low dword is read as its high dword: ±0.5 to
        // ±4 inline, any other one literal, as COMGR selects them; a second
        // literal and a nonzero low dword take registers. A v_ldexp_f64
        // exponent beyond the inline integers is a literal.
        let params = "    %X : ptr<f64,global>\n    %Y : ptr<f64,global>\n";
        let body = r#"    %x : f64 = memref.global_load_typed ( %X, %tid ) { align = i:8, elem_type = s:"f64" }
    %c1 : i32 = arith.constant (  ) { ity = s:"i32", value = i:1 }
    %c8 : i32 = arith.constant (  ) { ity = s:"i32", value = i:8 }
    %i0 : i32 = arith.mul ( %tid, %c8 )
    %i1 : i32 = arith.add ( %i0, %c1 )
    %i2 : i32 = arith.add ( %i1, %c1 )
    %i3 : i32 = arith.add ( %i2, %c1 )
    %i4 : i32 = arith.add ( %i3, %c1 )
    %i5 : i32 = arith.add ( %i4, %c1 )
    %inf : f64 = arith.constant (  ) { ity = s:"f64", value = f:inf }
    %one : f64 = arith.constant (  ) { ity = s:"f64", value = f:1.0 }
    %k3 : f64 = arith.constant (  ) { ity = s:"f64", value = f:3.0 }
    %tiny : f64 = arith.constant (  ) { ity = s:"f64", value = f:2.2250738585072014e-308 }
    %pi : f64 = arith.constant (  ) { ity = s:"f64", value = f:3.141592653589793 }
    %m : f64 = arith.fmin ( %inf, %x )
    %p : f64 = arith.fmul ( %one, %x )
    %q : f64 = arith.fdiv ( %x, %k3 )
    %e : f64 = arith.fdiv ( %x, %tiny )
    %f : f64 = arith.fma ( %k3, %x, %inf )
    %g : f64 = arith.fadd ( %x, %pi )
    memref.global_store_typed ( %Y, %i0, %m ) { align = i:8, elem_type = s:"f64" }
    memref.global_store_typed ( %Y, %i1, %p ) { align = i:8, elem_type = s:"f64" }
    memref.global_store_typed ( %Y, %i2, %q ) { align = i:8, elem_type = s:"f64" }
    memref.global_store_typed ( %Y, %i3, %e ) { align = i:8, elem_type = s:"f64" }
    memref.global_store_typed ( %Y, %i4, %f ) { align = i:8, elem_type = s:"f64" }
    memref.global_store_typed ( %Y, %i5, %g ) { align = i:8, elem_type = s:"f64" }
"#;
        let insts = insts(&kernel(params, body));
        let with = |op: Opc, k: u32| insts.iter().filter(|i| i.op == op && i.uses.contains(&Opnd::Imm(k))).count();
        assert_eq!(with(Opc::VMinF64, 0x7ff0_0000), 1);
        assert_eq!(with(Opc::VMulF64, 0x3ff0_0000), 1);
        assert_eq!((with(Opc::VDivScaleF64, 0x4008_0000), with(Opc::VDivFixupF64, 0x4008_0000)), (2, 1));
        // The Newton steps' 1.0 is inline in each: nothing builds it.
        assert_eq!(with(Opc::VFmaF64, 0x3ff0_0000), 2);
        let f64_ops = [Opc::VMinF64, Opc::VMulF64, Opc::VDivScaleF64, Opc::VDivFixupF64, Opc::VFmaF64];
        assert!(!insts.iter().any(|i| !f64_ops.contains(&i.op) && i.uses.contains(&Opnd::Imm(0x3ff0_0000))));
        let ldexp: Vec<&MInst> = insts.iter().filter(|i| i.op == Opc::VLdexpF64).collect();
        assert_eq!((ldexp.len(), ldexp[0].uses[1]), (1, Opnd::Imm(1022)));
        // fma(3, x, inf): one literal, the other value in a register pair.
        let fma = insts.iter().find(|i| i.op == Opc::VFmaF64 && i.uses.contains(&Opnd::Imm(0x4008_0000))).expect("fma");
        assert!(!fma.uses.contains(&Opnd::Imm(0x7ff0_0000)) && fma.uses.iter().all(|o| o.dwords() == 2 || matches!(o, Opnd::Imm(_))));
        let add = insts.iter().find(|i| i.op == Opc::VAddF64).expect("x + pi");
        assert!(!add.uses.iter().any(|o| matches!(o, Opnd::Imm(_))));
    }

    #[test]
    fn fdot2_of_bf16_pairs() {
        let params = "    %A : ptr<bf16,global>\n    %D : ptr<f32,global>\n";
        let body = r#"    %c2 : i32 = arith.constant (  ) { ity = s:"i32", value = i:2 }
    %i : i32 = arith.mul ( %tid, %c2 )
    %v : vec<bf16x2> = memref.global_load_vN ( %A, %i ) { align = i:4, elem_type = s:"bf16", vec = i:2 }
    %c : f32 = memref.global_load_typed ( %D, %tid ) { align = i:4, elem_type = s:"f32" }
    %d : f32 = arith.fdot2 ( %v, %v, %c )
    memref.global_store_typed ( %D, %tid, %d ) { align = i:4, elem_type = s:"f32" }
"#;
        let compiled = super::super::compile(&kernel(params, body)).unwrap_or_else(|e| panic!("{e}"));
        let listing = compiled.listing();
        let line = listing.lines().find(|l| l.contains("v_dot2_f32_bf16")).expect("one bf16 dot2");
        let words: Vec<u32> = line.rsplit(": ").next().unwrap().split_whitespace().map(|w| u32::from_str_radix(w, 16).unwrap()).collect();
        // VOP3P opcode 0x1a, op_sel_hi all set and no negation, as v_dot2_f32_f16.
        assert_eq!((words[0] >> 16 & 0xff, words[0] >> 14 & 1, words[1] >> 27), (0x1a, 1, 0b11));
    }

    #[test]
    fn permlane64_is_its_source_in_wave32() {
        let params = "    %S : ptr<i32,global>\n    %D : ptr<i32,global>\n";
        let load = "    %x : i32 = memref.global_load_typed ( %S, %tid ) { align = i:4, elem_type = s:\"i32\" }\n";
        let swap = "    %p : i32 = tile.permlane64 ( %x )\n    memref.global_store_typed ( %D, %tid, %p ) { align = i:4, elem_type = s:\"i32\" }\n";
        let copy = "    memref.global_store_typed ( %D, %tid, %x ) { align = i:4, elem_type = s:\"i32\" }\n";
        let listing = |body: String| super::super::compile(&kernel(params, &body)).unwrap_or_else(|e| panic!("{e}")).listing();
        assert_eq!(listing(format!("{load}{swap}")), listing(format!("{load}{copy}")));
    }

    #[test]
    fn widened_bf16_is_quieted_before_fmax() {
        // `fpext bfloat` is a shift, which keeps a signaling NaN signaling, so
        // v_max_f32 needs the operand quieted as for any loaded value.
        let params = "    %B : ptr<bf16,global>\n    %Z : ptr<f32,global>\n";
        let body = r#"    %b : bf16 = memref.global_load_typed ( %B, %tid ) { align = i:2, elem_type = s:"bf16" }
    %f : f32 = arith.cast_to_f32 ( %b )
    %one : f32 = arith.constant (  ) { ity = s:"f32", value = f:1.0 }
    %m : f32 = arith.fmax ( %f, %one )
    memref.global_store_typed ( %Z, %tid, %m ) { align = i:4, elem_type = s:"f32" }
"#;
        let insts = insts(&kernel(params, body));
        let quiets = ops(&insts).iter().filter(|(n, u)| n == "v_max_f32" && u[0] == u[1]).count();
        assert_eq!(quiets, 1);
    }
}

/// Memory, conversion and control-flow ops (census W05, W06, W07, W11, W13, W14, W16).
#[cfg(test)]
mod convmem_tests {
    use super::*;

    fn compiled(text: &str) -> Result<Vec<MInst>, CompileError> {
        let compiled = super::super::compile(text)?;
        Ok(compiled.func.blocks.into_iter().flat_map(|b| b.insts).collect())
    }

    fn insts(text: &str) -> Vec<MInst> {
        compiled(text).unwrap_or_else(|e| panic!("{e}"))
    }

    fn kernel(params: &str, body: &str) -> String {
        format!(
            "ckdsl.ir v1\nkernel @probe {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n{params}  }}\n  \
             region @entry {{\n    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n{body}    cf.return (  )\n  }}\n}}\n"
        )
    }

    fn all(insts: &[MInst], op: Opc) -> Vec<&MInst> {
        insts.iter().filter(|i| i.op == op).collect()
    }

    fn count(insts: &[MInst], op: Opc) -> usize {
        all(insts, op).len()
    }

    #[test]
    fn scalar_buffer_accesses_pick_their_width() {
        let params = "    %X : ptr<f16,global>\n    %n : i32\n";
        let body = r#"    %rs : vec<i32x4> = tile.buffer_rsrc ( %X, %n )
    %c0 : i32 = arith.constant (  ) { ity = s:"i32", value = i:0 }
    %c2 : i32 = arith.constant (  ) { ity = s:"i32", value = i:2 }
    %c6 : i32 = arith.constant (  ) { ity = s:"i32", value = i:6 }
    %bo : i32 = arith.mul ( %tid, %c2 )
    %bo6 : i32 = arith.add ( %bo, %c6 )
    %h : f16 = tile.buffer_load_f16 ( %rs, %bo6, %c0 )
    %hf : f32 = arith.cast_to_f32 ( %h )
    %f : f32 = tile.buffer_load ( %rs, %bo, %c0 ) { elem_type = s:"f32" }
    %s : f32 = arith.fadd ( %hf, %f )
    %t : f16 = arith.trunc_f32_to_f16 ( %s )
    tile.buffer_store_f16 ( %rs, %bo, %c0, %t )
    tile.buffer_store_f32 ( %rs, %bo, %c0, %s )
    %z : vec<f16x2> = arith.constant_vec (  ) { elem = s:"f16", fill = f:0.0, vec = i:2 }
    %d0 : f16 = tile.buffer_load_f16_d16 ( %rs, %bo, %c0 )
    %v0 : vec<f16x2> = vector.insert ( %z, %d0 ) { index = i:0 }
    %d1 : f16 = tile.buffer_load_f16_d16 ( %rs, %bo6, %c0 )
    %v1 : vec<f16x2> = vector.insert ( %v0, %d1 ) { index = i:1 }
    %w : vec<f16x4> = tile.buffer_load_vN_f16 ( %rs, %bo, %c0 ) { dwords = i:2 }
    tile.buffer_store_vN_f16 ( %rs, %bo, %c0, %v1 ) { dwords = i:1 }
    tile.buffer_store_vN_f16 ( %rs, %bo, %c0, %w ) { dwords = i:2 }
"#;
        let insts = insts(&kernel(params, body));
        // The constant 6 of the voffset goes to the instruction offset.
        let u16s = all(&insts, Opc::BufLoadU16);
        assert_eq!((u16s.len(), u16s[0].imm), (1, 6));
        assert_eq!((count(&insts, Opc::BufLoadB32), count(&insts, Opc::BufStoreB16)), (1, 1));
        // The two halves feeding the inserts load straight into one dword.
        let (lo, hi) = (all(&insts, Opc::BufLoadD16B16), all(&insts, Opc::BufLoadD16HiB16));
        assert_eq!((lo.len(), hi.len(), hi[0].imm), (1, 1, 6));
        assert_eq!(lo[0].defs[0], hi[0].defs[0]);
        assert_eq!(count(&insts, Opc::BufStoreB32), 2);
        assert_eq!((count(&insts, Opc::BufLoadB64), count(&insts, Opc::BufStoreB64)), (1, 1));
        assert_eq!(count(&insts, Opc::VPermB32), 0);
    }

    #[test]
    fn scalar_conversions_are_single_instructions() {
        let params = "    %X : ptr<f32,global>\n    %I : ptr<i32,global>\n    %Y : ptr<f32,global>\n";
        let body = r#"    %x : f32 = memref.global_load_typed ( %X, %tid ) { align = i:4, elem_type = s:"f32" }
    %i : i32 = memref.global_load_typed ( %I, %tid ) { align = i:4, elem_type = s:"i32" }
    %r : f32 = arith.rint_f32 ( %x )
    %c : f32 = arith.sitofp_f32 ( %i )
    %s : f32 = arith.fadd ( %r, %c )
    %h : f16 = arith.trunc_f32_to_f16 ( %s )
    %b : f32 = arith.cast_to_f32 ( %h )
    memref.global_store_typed ( %Y, %tid, %b ) { align = i:4, elem_type = s:"f32" }
"#;
        let insts = insts(&kernel(params, body));
        assert_eq!((count(&insts, Opc::VRndneF32), count(&insts, Opc::VCvtF32I32), count(&insts, Opc::VCvtF16F32)), (1, 1, 1));
    }

    #[test]
    fn hints_and_partial_barriers_keep_their_waits() {
        let params = "    %X : ptr<f32,global>\n    %Y : ptr<f32,global>\n";
        let body = r#"    %lds : smem<f32, [64]> = tile.smem_alloc (  )
    %x : f32 = memref.global_load_typed ( %X, %tid ) { align = i:4, elem_type = s:"f32" }
    tile.s_setprio (  ) { level = i:1 }
    tile.smem_store_vN_f32 ( %lds, %tid, %x ) { elem_type = s:"f32", rank = i:1, vec = i:1 }
    tile.s_waitcnt (  ) { expcnt = i:-1, lgkmcnt = i:0, vmcnt = i:-1 }
    tile.sched_barrier (  ) { mask = i:0 }
    tile.sched_group_barrier (  ) { count = i:1, group = i:0, mask = i:8 }
    tile.s_barrier_bare (  )
    %c1 : i32 = arith.constant (  ) { ity = s:"i32", value = i:1 }
    %t1 : i32 = arith.xor ( %tid, %c1 )
    %y : vec<f32x1> = tile.smem_load_vN_f32 ( %lds, %t1 ) { elem_type = s:"f32", rank = i:1, vec = i:1 }
    %z : f32 = memref.global_load_typed ( %X, %t1 ) { align = i:4, elem_type = s:"f32" }
    tile.sync_lds_only (  )
    %e : f32 = vector.extract ( %y ) { index = i:0 }
    tile.s_setprio (  ) { level = i:0 }
    %s : f32 = arith.fadd ( %e, %z )
    memref.global_store_typed ( %Y, %tid, %s ) { align = i:4, elem_type = s:"f32" }
"#;
        let insts = insts(&kernel(params, body));
        let prio: Vec<i64> = all(&insts, Opc::SSetprio).iter().map(|i| i.imm).collect();
        assert_eq!(prio, [1, 0]);
        let explicit = all(&insts, Opc::SWaitcntExplicit);
        assert_eq!((explicit.len(), explicit[0].imm), (1, 0xfc07));
        assert_eq!(count(&insts, Opc::SchedBarrier), 0);
        let at = |op: Opc| (0..insts.len()).filter(|&i| insts[i].op == op).collect::<Vec<_>>();
        let (bare, lds) = (at(Opc::SBarrierBare), at(Opc::SBarrierLds));
        assert_eq!((bare.len(), lds.len(), count(&insts, Opc::SBarrier)), (1, 1, 0));
        // The bare barrier adds no wait of its own: the kernel's explicit
        // lgkmcnt(0) already covers the LDS store.
        assert_ne!(insts[bare[0] - 1].op, Opc::SWaitcnt);
        // sync_lds_only drains LDS but leaves the global load in flight.
        let before = &insts[lds[0] - 1];
        assert_eq!((before.op, before.imm >> 10 & 0x3f, before.imm >> 4 & 0x3f), (Opc::SWaitcnt, 0x3f, 0));
    }

    #[test]
    fn pin_sgpr_reads_a_lane_only_for_a_divergent_value() {
        let params = "    %Y : ptr<f32,global>\n    %n : i32\n";
        let body = r#"    %p : i32 = tile.pin_sgpr ( %n )
    %q : i32 = tile.pin_sgpr ( %tid )
    %s : i32 = arith.add ( %p, %q )
    %f : f32 = arith.sitofp_f32 ( %s )
    memref.global_store_typed ( %Y, %tid, %f ) { align = i:4, elem_type = s:"f32" }
"#;
        assert_eq!(count(&insts(&kernel(params, body)), Opc::VReadfirstlaneB32), 1);
    }

    #[test]
    fn lds_vectors_ranks_and_atomics() {
        let params = "    %Y : ptr<f16,global>\n    %I : ptr<i32,global>\n";
        let body = r#"    %lds : smem<f16, [4x16x16]> = tile.smem_alloc (  )
    %tile : smem<f16, [4x16]> = tile.smem_alloc (  )
    %cnt : smem<i32, [16]> = tile.smem_alloc (  )
    %c0 : i32 = arith.constant (  ) { ity = s:"i32", value = i:0 }
    %c1 : i32 = arith.constant (  ) { ity = s:"i32", value = i:1 }
    %c3 : i32 = arith.constant (  ) { ity = s:"i32", value = i:3 }
    %c4 : i32 = arith.constant (  ) { ity = s:"i32", value = i:4 }
    %c8 : i32 = arith.constant (  ) { ity = s:"i32", value = i:8 }
    %c15 : i32 = arith.constant (  ) { ity = s:"i32", value = i:15 }
    %col : i32 = arith.and ( %tid, %c15 )
    %row : i32 = arith.lshr ( %tid, %c4 )
    %q : i32 = arith.and ( %tid, %c3 )
    %col4 : i32 = arith.mul ( %q, %c4 )
    %v : vec<f16x8> = tile.smem_load_vN ( %lds, %row, %c0, %c0 ) { elem_type = s:"f16", rank = i:3, vec = i:8 }
    %a : vec<f16x4> = tile.smem_load_v4 ( %tile, %row, %col4 ) { elem_type = s:"f16" }
    %b : vec<f16x4> = tile.smem_load_v4 ( %tile, %row, %col ) { elem_type = s:"f16" }
    %old : i32 = tile.lds_atomic_add ( %cnt, %col, %c1 ) { elem_type = s:"i32", ordering = s:"monotonic", rank = i:1 }
    %dead : i32 = tile.lds_atomic_add ( %cnt, %col, %c1 ) { elem_type = s:"i32", ordering = s:"monotonic", rank = i:1 }
    %r8 : i32 = arith.mul ( %tid, %c8 )
    memref.global_store_vN ( %Y, %r8, %v ) { align = i:16, elem_type = s:"f16", vec = i:8 }
    memref.global_store_vN ( %Y, %r8, %a ) { align = i:8, elem_type = s:"f16", vec = i:4 }
    memref.global_store_vN ( %Y, %r8, %b ) { align = i:8, elem_type = s:"f16", vec = i:4 }
    memref.global_store_typed ( %I, %tid, %old ) { align = i:4, elem_type = s:"i32" }
"#;
        let insts = insts(&kernel(params, body));
        assert_eq!(count(&insts, Opc::DsLoadB128), 1);
        // The aligned smem_load_v4 is one 64-bit load, the other four u16 loads.
        assert_eq!((count(&insts, Opc::DsLoadB64), count(&insts, Opc::DsLoadU16)), (1, 4));
        assert_eq!((count(&insts, Opc::DsAddRtnU32), count(&insts, Opc::DsAddU32)), (1, 1));
        let bad = text_with(&kernel(params, body), "ordering = s:\"monotonic\"", "ordering = s:\"seq_cst\"");
        assert!(compiled(&bad).is_err());
    }

    fn text_with(text: &str, from: &str, to: &str) -> String {
        assert!(text.contains(from));
        text.replacen(from, to, 1)
    }

    #[test]
    fn dpp_moves_carry_their_control_words() {
        let params = "    %X : ptr<i32,global>\n    %Y : ptr<i32,global>\n";
        let body = r#"    %x : i32 = memref.global_load_typed ( %X, %tid ) { align = i:4, elem_type = s:"i32" }
    %a : i32 = tile.dpp_xor ( %x ) { xor_mask = i:1 }
    %b : i32 = tile.quad_perm ( %a ) { ctrl = i:177 }
    %c : i32 = tile.mov_dpp8 ( %b ) { sel = i:14570689 }
    memref.global_store_typed ( %Y, %tid, %c ) { align = i:4, elem_type = s:"i32" }
"#;
        let text = kernel(params, body);
        let insts = insts(&text);
        let dpp: Vec<i64> = all(&insts, Opc::VMovB32Dpp).iter().map(|i| i.imm).collect();
        assert_eq!(dpp, [dpp16_ctrl(0x161, true) as i64, dpp16_ctrl(177, true) as i64]);
        assert_eq!(all(&insts, Opc::VMovB32Dpp8)[0].imm, 0xde_54c1);
        assert!(compiled(&text_with(&text, "xor_mask = i:1", "xor_mask = i:16")).is_err());
        assert!(compiled(&text_with(&text, "ctrl = i:177", "ctrl = i:256")).is_err());
    }

    /// Encodings checked against `llvm-mc -arch=amdgcn -mcpu=gfx1151 -show-encoding`.
    #[test]
    fn new_encodings_match_the_llvm_assembler() {
        let v = |i: u16| Opnd::P(Phys::V(i), 1);
        let rsrc = Opnd::P(Phys::S(4), 4);
        let s8 = Opnd::P(Phys::S(8), 1);
        let cases: Vec<(MInst, &[u8])> = vec![
            (MInst::new(Opc::VMovB32Dpp, [v(1)], [v(2)]).with_imm(dpp16_ctrl(0x161, true) as i64), &[0xfa, 0x02, 0x02, 0x7e, 0x02, 0x61, 0x09, 0xff]),
            (MInst::new(Opc::VMovB32Dpp, [v(1)], [v(2)]).with_imm(dpp16_ctrl(0x16f, true) as i64), &[0xfa, 0x02, 0x02, 0x7e, 0x02, 0x6f, 0x09, 0xff]),
            (MInst::new(Opc::VMovB32Dpp, [v(1)], [v(2)]).with_imm(dpp16_ctrl(0xb1, true) as i64), &[0xfa, 0x02, 0x02, 0x7e, 0x02, 0xb1, 0x08, 0xff]),
            (MInst::new(Opc::VMovB32Dpp, [v(200)], [v(201)]).with_imm(dpp16_ctrl(0x167, true) as i64), &[0xfa, 0x02, 0x90, 0x7f, 0xc9, 0x67, 0x09, 0xff]),
            (MInst::new(Opc::VMovB32Dpp8, [v(1)], [v(2)]).with_imm(0xde_54c1), &[0xe9, 0x02, 0x02, 0x7e, 0x02, 0xc1, 0x54, 0xde]),
            (MInst::new(Opc::VRndneF32, [v(1)], [v(2)]), &[0x02, 0x47, 0x02, 0x7e]),
            (MInst::new(Opc::VTruncF32, [v(1)], [v(2)]), &[0x02, 0x43, 0x02, 0x7e]),
            (MInst::new(Opc::SSetprio, [], []).with_imm(1), &[0x01, 0x00, 0xb5, 0xbf]),
            (MInst::new(Opc::SWaitcntExplicit, [], []).with_imm(0x3f7), &[0xf7, 0x03, 0x89, 0xbf]),
            (MInst::new(Opc::DsAddU32, [], [v(1), v(2)]).with_imm(8), &[0x08, 0x00, 0x00, 0xd8, 0x01, 0x02, 0x00, 0x00]),
            (MInst::new(Opc::DsAddRtnU32, [v(3)], [v(1), v(2)]).with_imm(8), &[0x08, 0x00, 0x80, 0xd8, 0x01, 0x02, 0x00, 0x03]),
            (MInst::new(Opc::DsAddF32, [], [v(1), v(2)]).with_imm(8), &[0x08, 0x00, 0x54, 0xd8, 0x01, 0x02, 0x00, 0x00]),
            (MInst::new(Opc::DsAddRtnF32, [v(3)], [v(1), v(2)]).with_imm(8), &[0x08, 0x00, 0xe4, 0xd9, 0x01, 0x02, 0x00, 0x03]),
            (MInst::new(Opc::BufLoadU16, [v(1)], [v(2), rsrc, s8]).with_imm(12), &[0x0c, 0x00, 0x48, 0xe0, 0x02, 0x01, 0x41, 0x08]),
            (MInst::new(Opc::BufLoadD16B16, [v(1)], [v(2), rsrc, s8, v(1)]).with_imm(12), &[0x0c, 0x00, 0x80, 0xe0, 0x02, 0x01, 0x41, 0x08]),
            (MInst::new(Opc::BufLoadD16HiB16, [v(1)], [v(2), rsrc, s8, v(1)]).with_imm(12), &[0x0c, 0x00, 0x8c, 0xe0, 0x02, 0x01, 0x41, 0x08]),
            (MInst::new(Opc::BufStoreD16HiB16, [], [v(1), v(2), rsrc, s8]).with_imm(12), &[0x0c, 0x00, 0x94, 0xe0, 0x02, 0x01, 0x41, 0x08]),
        ];
        for (inst, want) in cases {
            let text = inst_text(&MFunc::default(), &inst);
            let f = MFunc { blocks: vec![MBlock { insts: vec![inst], loop_depth: 0 }], vregs: Vec::new(), ..Default::default() };
            assert_eq!(super::super::asm::assemble(&f).expect("encodes"), want, "{text}");
        }
    }

    fn i64_kernel(op: &str, divisor: &str, divergent: bool) -> String {
        let params = "    %X : ptr<i64,global>\n    %Y : ptr<i64,global>\n    %a : i64\n    %b : i64\n";
        let x = if divergent {
            "    %x : i64 = memref.global_load_typed ( %X, %tid ) { align = i:8, elem_type = s:\"i64\" }\n"
        } else {
            "    %x : i64 = arith.add ( %a, %a )\n"
        };
        let body = format!(
            "{x}    %k : i64 = arith.constant (  ) {{ ity = s:\"i64\", value = i:{divisor} }}\n    \
             %d : i64 = arith.{op} ( %x, {} )\n    \
             memref.global_store_typed ( %Y, %tid, %d ) {{ align = i:8, elem_type = s:\"i64\" }}\n",
            if divisor == "b" { "%b" } else { "%k" }
        );
        kernel(params, &body.replace("value = i:b", "value = i:0"))
    }

    #[test]
    fn i64_division_by_a_runtime_value() {
        // Uniform: the estimate comes back from the VALU, the rest is SALU.
        let uniform = insts(&i64_kernel("div", "b", false));
        assert_eq!((count(&uniform, Opc::VRcpF32), count(&uniform, Opc::VReadfirstlaneB32)), (1, 2));
        assert!(count(&uniform, Opc::SMulHiU32) >= 8);
        assert_eq!(count(&uniform, Opc::VMadU64U32), 0);
        // Divergent: all VALU, with the 32x32->64 multiply-adds.
        for op in ["div", "mod"] {
            let divergent = insts(&i64_kernel(op, "b", true));
            assert_eq!((count(&divergent, Opc::VRcpF32), count(&divergent, Opc::VReadfirstlaneB32)), (1, 0));
            assert!(count(&divergent, Opc::VMadU64U32) >= 6, "{op}");
            assert_eq!(count(&divergent, Opc::SMulHiU32), 0);
        }
    }

    #[test]
    fn i64_division_by_a_constant_needs_no_reciprocal() {
        for (op, k) in [("div", "7"), ("mod", "7"), ("div", "8"), ("mod", "8"), ("div", "-1"), ("div", "1000000007")] {
            for divergent in [false, true] {
                let insts = insts(&i64_kernel(op, k, divergent));
                assert_eq!(count(&insts, Opc::VRcpF32), 0, "{op} {k}");
            }
        }
        // A negative divisor other than -1 takes the general expansion.
        assert_eq!(count(&insts(&i64_kernel("div", "-7", true)), Opc::VRcpF32), 1);
    }

    #[test]
    fn signed_magic64_divides_like_rust() {
        let mulhu = |a: u64, b: u64| ((a as u128 * b as u128) >> 64) as u64;
        let divisors = [2i64, 3, 5, 6, 7, 10, 12, 25, 125, 641, 1_000_000_007, (1 << 33) + 1, i64::MAX, i64::MAX / 3];
        let mut xs = vec![0i64, 1, -1, 2, -2, 7, -7, 100, -100, i64::MAX, i64::MIN, i64::MIN + 1, i64::MAX - 1];
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..2000 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            xs.push((seed ^ (seed >> 29)) as i64 >> (seed % 60));
        }
        for &d in &divisors {
            let (m, s) = signed_magic64(d);
            for &x in &xs {
                // As div64_by_const computes it.
                let neg = if x < 0 { m } else { 0 };
                let q = (mulhu(x as u64, m).wrapping_sub(neg) as i64) >> s;
                let q = q.wrapping_add((q as u64 >> 63) as i64);
                assert_eq!(q, x / d, "{x} / {d}");
            }
        }
    }

    fn loop_kernel(lb: &str, ub: &str, carried: &str, body: &str) -> String {
        let params = "    %X : ptr<i32,global>\n    %Y : ptr<f32,global>\n";
        let text = format!(
            "    %n : i32 = memref.global_load_typed ( %X, %tid ) {{ align = i:4, elem_type = s:\"i32\" }}\n    \
             %c0 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:0 }}\n    \
             %c1 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:1 }}\n    \
             %two : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:2 }}\n    \
             %z : f32 = arith.constant (  ) {{ ity = s:\"f32\", value = f:0.0 }}\n    \
             %lt : i1 = arith.cmp ( %n, %two ) {{ pred = s:\"lt\" }}\n    \
             %sel : i32 = arith.select ( %lt, %c0, %c1 )\n    \
             %r : f32 = scf.for ( {lb}, {ub}, %c1, {carried} ) {{ iter_args = l:[ {{ name = s:\"%acc\", type = s:\"f32\" }} ], \
             iv = s:\"%i\", iv_type = s:\"i32\", num_iter_args = i:1, unroll = b:false }}\n      region @body {{\n{body}      }}\n    \
             memref.global_store_typed ( %Y, %tid, %r ) {{ align = i:4, elem_type = s:\"f32\" }}\n"
        );
        kernel(params, &text)
    }

    const ACCUMULATE: &str = "        %fi : f32 = arith.sitofp_f32 ( %i )\n        %a2 : f32 = arith.fadd ( %acc, %fi )\n        scf.yield ( %a2 ) { num = i:1 }\n";

    #[test]
    fn loops_with_per_lane_bounds_mask_exec() {
        let exec_writes = |insts: &[MInst]| {
            insts.iter().filter(|i| i.defs.contains(&Opnd::P(Phys::ExecLo, 1))).map(|i| i.op).collect::<Vec<_>>()
        };
        // Per-lane bound, uniform start and step: the induction variable is an SGPR.
        let insts = insts(&loop_kernel("%c0", "%n", "%z", ACCUMULATE));
        assert_eq!(exec_writes(&insts), [Opc::SAndB32, Opc::SMovB32]);
        assert_eq!((count(&insts, Opc::SCbranchExecz), count(&insts, Opc::SBranch)), (1, 1));
        assert!(count(&insts, Opc::SAddI32) >= 1);
        // A per-lane start makes it a VGPR, stepped by a VALU add (perhaps paired).
        let insts = self::insts(&loop_kernel("%n", "%c1", "%z", ACCUMULATE));
        let adds_one = |i: &MInst| match i.op {
            Opc::VAddNcU32 => i.uses.contains(&Opnd::Imm(1)),
            Opc::VDual => {
                let (x, y) = i.uses.split_at(i.mods as usize);
                [(i.imm & 0xff, x), (i.imm >> 8 & 0xff, y)]
                    .iter()
                    .any(|&(code, srcs)| vopd_name(code as u8) == "v_dual_add_nc_u32" && srcs.contains(&Opnd::Imm(1)))
            }
            _ => false,
        };
        assert!(insts.iter().any(adds_one));
        // At most one trip (bound 0 or 1 from 0 by 1): a masked if, no back edge.
        let insts = self::insts(&loop_kernel("%c0", "%sel", "%z", ACCUMULATE));
        assert_eq!((count(&insts, Opc::SCbranchExecz), count(&insts, Opc::SBranch)), (1, 0));
    }

    #[test]
    fn per_lane_loops_refuse_what_needs_every_lane() {
        let sync = "        tile.sync (  ) { fence = s:\"compiler\" }\n".to_string() + ACCUMULATE;
        let err = compiled(&loop_kernel("%c0", "%n", "%z", &sync)).err().expect("refused");
        assert!(err.message.contains("Sync"), "{}", err.message);
        // The same body under a uniform bound is fine.
        compiled(&loop_kernel("%c0", "%c1", "%z", &sync)).unwrap_or_else(|e| panic!("{e}"));
        // An early return in a per-lane loop would end lanes that already left it.
        let ret = "        cf.return (  )\n".to_string() + ACCUMULATE;
        let err = compiled(&loop_kernel("%c0", "%n", "%z", &ret)).err().expect("refused");
        assert_eq!(err.kind, "ROCKE_DIRECT_UNSUPPORTED", "{}", err.message);
    }

    /// Operand and result types, attributes and ranges the LLVM lowering
    /// relies on, for the kinds this work added or widened.
    #[test]
    fn new_and_widened_ops_are_validated() {
        let params = "    %X : ptr<f32,global>\n    %Y : ptr<f32,global>\n    %n : i32\n";
        let pre = "    %f : f32 = memref.global_load_typed ( %X, %tid ) { align = i:4, elem_type = s:\"f32\" }\n    \
                   %c0 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }\n    \
                   %rs : vec<i32x4> = tile.buffer_rsrc ( %X, %n )\n    \
                   %l1 : smem<f16, [64]> = tile.smem_alloc (  )\n    %l2 : smem<f16, [8x8]> = tile.smem_alloc (  )\n    \
                   %ci : smem<i32, [64]> = tile.smem_alloc (  )\n    %cl : smem<i64, [64]> = tile.smem_alloc (  )\n";
        let store = "    memref.global_store_typed ( %Y, %tid, %f ) { align = i:4, elem_type = s:\"f32\" }\n";
        let cases = [
            ("    %h : f16 = arith.trunc_f32_to_f16 ( %tid )\n", "MALFORMED"),
            ("    %h : f16 = arith.cast_f32_to ( %f ) { target = s:\"f16\" }\n    %r : f16 = arith.rint_f32 ( %h )\n", "UNSUPPORTED"),
            ("    tile.s_setprio (  )\n", "MALFORMED"),
            ("    tile.s_setprio (  ) { level = i:70000 }\n", "MALFORMED"),
            ("    tile.s_waitcnt (  ) { vmcnt = s:\"0\" }\n", "MALFORMED"),
            ("    tile.sched_group_barrier (  ) { mask = i:8 }\n", "MALFORMED"),
            ("    tile.sched_barrier (  ) { mask = i:8589934592 }\n", "MALFORMED"),
            ("    tile.s_barrier_bare (  ) { fence = s:\"agent\" }\n", "INVALID"),
            ("    tile.sync_lds_only (  ) { fence = s:\"system\" }\n", "INVALID"),
            ("    tile.sync (  ) { fence = s:\"device\" }\n", "INVALID"),
            ("    %v : vec<f16x4> = tile.smem_load_v4 ( %l1, %tid, %tid ) { elem_type = s:\"f16\" }\n", "MALFORMED"),
            ("    %v : vec<bf16x4> = tile.smem_load_v4 ( %l2, %tid, %tid ) { elem_type = s:\"f16\" }\n", "MALFORMED"),
            ("    %w : i64 = arith.zext ( %tid )\n    %o : i64 = tile.lds_atomic_add ( %cl, %tid, %w ) { rank = i:1 }\n", "UNSUPPORTED"),
            ("    %o : i32 = tile.lds_atomic_add ( %ci, %tid, %tid ) { ordering = s:\"seq_cst\" }\n", "UNSUPPORTED"),
            ("    %o : i32 = tile.lds_atomic_add ( %ci, %tid, %tid ) { ordering = s:\"sometimes\" }\n", "INVALID"),
            ("    %o : i32 = tile.lds_atomic_add ( %ci, %tid, %tid, %tid )\n", "MALFORMED"),
            ("    %d : f32 = tile.dpp_xor ( %f ) { xor_mask = i:1 }\n", "MALFORMED"),
            ("    %d : i32 = tile.dpp_xor ( %tid ) { xor_mask = i:16 }\n", "INVALID"),
            ("    %d : i32 = tile.quad_perm ( %tid ) { ctrl = i:256 }\n", "INVALID"),
            ("    %d : f32 = tile.mov_dpp8 ( %f )\n", "INVALID"),
            ("    %b : f32 = tile.buffer_load ( %rs, %tid, %c0 ) { elem_type = s:\"f16\" }\n", "MALFORMED"),
            ("    %b : f16 = tile.buffer_load ( %rs, %tid, %c0 ) { elem_type = s:\"f64\" }\n", "INVALID"),
            ("    %b : i64 = tile.buffer_load_f16 ( %rs, %tid, %c0 )\n", "MALFORMED"),
            ("    tile.buffer_store_f16 ( %rs, %tid, %c0, %tid )\n", "MALFORMED"),
            ("    tile.buffer_store_f32 ( %rs, %tid, %tid, %f )\n", "UNSUPPORTED"),
            ("    %m : f32 = arith.mul ( %f, %f )\n", "MALFORMED"),
            ("    %p : f32 = tile.pin_sgpr ( %f )\n", "INVALID"),
        ];
        for (body, kind) in cases {
            let err = compiled(&kernel(params, &format!("{pre}{body}{store}"))).err().unwrap_or_else(|| panic!("accepted:\n{body}"));
            assert_eq!(err.kind, format!("ROCKE_DIRECT_{kind}"), "{}\n{body}", err.message);
        }
        let unroll = loop_kernel("%c0", "%n", "%z", ACCUMULATE).replace("unroll = b:false", "unroll = i:1");
        assert_eq!(compiled(&unroll).err().map(|e| e.kind), Some("ROCKE_DIRECT_MALFORMED"));
    }

    /// A "workgroup" fence on a barrier adds the gfx11 release before it
    /// and the acquire after it, whichever counters the barrier drains.
    #[test]
    fn workgroup_fences_wrap_barriers() {
        let params = "    %X : ptr<f32,global>\n";
        for (op, barrier) in [("tile.sync", Opc::SBarrier), ("tile.sync_lds_only", Opc::SBarrierLds), ("tile.s_barrier_bare", Opc::SBarrierBare)] {
            for fence in ["", " { fence = s:\"compiler\" }", " { fence = s:\"workgroup\" }"] {
                let insts = insts(&kernel(params, &format!("    {op} (  ){fence}\n")));
                let at = (0..insts.len()).find(|&i| insts[i].op == barrier).expect("a barrier");
                let fenced = fence.contains("workgroup");
                assert_eq!(insts[..at].iter().any(|i| i.op == Opc::SWaitcntVscnt), fenced, "{op}{fence}");
                assert_eq!(insts[at..].iter().any(|i| i.op == Opc::BufferGl0Inv), fenced, "{op}{fence}");
            }
        }
    }

    /// Two LDS stores off one address pair into `ds_store_2addr_b32`, but
    /// never across a barrier of any kind.
    #[test]
    fn lds_store_pairs_stop_at_fences() {
        let params = "    %X : ptr<f32,global>\n";
        let body = |between: &str| {
            format!(
                "    %l : smem<f32, [256]> = tile.smem_alloc (  )\n    %f : f32 = memref.global_load_typed ( %X, %tid ) {{ align = i:4, elem_type = s:\"f32\" }}\n    \
                 %c4 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:4 }}\n    %c1 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:1 }}\n    \
                 %i : i32 = arith.mul ( %tid, %c4 )\n    %j : i32 = arith.add ( %i, %c1 )\n    \
                 tile.smem_store ( %l, %i, %f ) {{ align = i:4 }}\n{between}    tile.smem_store ( %l, %j, %f ) {{ align = i:4 }}\n"
            )
        };
        assert_eq!(count(&insts(&kernel(params, &body(""))), Opc::DsStore2addrB32), 1);
        for between in ["tile.sync_lds_only (  )", "tile.s_barrier_bare (  )", "tile.sync (  )", "tile.s_waitcnt (  ) { lgkmcnt = i:0 }"] {
            let insts = insts(&kernel(params, &body(&format!("    {between}\n"))));
            assert_eq!(count(&insts, Opc::DsStore2addrB32), 0, "{between}");
        }
    }

    /// A loop bound that turns out divergent only once a loop-carried value
    /// around it does: the inner loop masks EXEC instead of branching on a
    /// scalar compare.
    #[test]
    fn late_divergence_reaches_loop_bounds() {
        let params = "    %Y : ptr<f32,global>\n";
        let body = "    %c0 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }\n    \
                    %c1 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:1 }\n    \
                    %c3 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:3 }\n    \
                    %z : f32 = arith.constant (  ) { ity = s:\"f32\", value = f:0.0 }\n    \
                    %m : i32, %r : f32 = scf.for ( %c0, %c3, %c1, %c1, %z ) { iter_args = l:[ { name = s:\"%k\", type = s:\"i32\" }, { name = s:\"%acc\", type = s:\"f32\" } ], iv = s:\"%j\", iv_type = s:\"i32\", num_iter_args = i:2, unroll = b:false }\n      \
                    region @body {\n        \
                    %s : f32 = scf.for ( %c0, %k, %c1, %acc ) { iter_args = l:[ { name = s:\"%b\", type = s:\"f32\" } ], iv = s:\"%i\", iv_type = s:\"i32\", num_iter_args = i:1, unroll = b:false }\n          \
                    region @body {\n            %fi : f32 = arith.sitofp_f32 ( %i )\n            %b2 : f32 = arith.fadd ( %b, %fi )\n            scf.yield ( %b2 ) { num = i:1 }\n          }\n        \
                    %k2 : i32 = arith.add ( %k, %tid )\n        scf.yield ( %k2, %s ) { num = i:2 }\n      }\n    \
                    memref.global_store_typed ( %Y, %tid, %r ) { align = i:4, elem_type = s:\"f32\" }\n";
        let text = kernel(params, body);
        let k = super::super::parse::parse(&text).expect("parses");
        let uniform = super::super::analysis::classify(&k);
        let outer = k.body.ops.iter().find(|op| op.kind == OpKind::For).expect("outer loop");
        let inner = outer.regions[0].ops.iter().find(|op| op.kind == OpKind::For).expect("inner loop");
        let u = |v: ValueId| uniform[v.0 as usize];
        assert!(!u(outer.block_args[1]) && !u(inner.operands[1]));
        assert!(u(inner.block_args[0]) && !u(inner.block_args[1]) && !u(inner.results[0]));
        let insts = insts(&text);
        let narrows = insts.iter().filter(|i| i.op == Opc::SAndB32 && i.defs.contains(&Opnd::P(Phys::ExecLo, 1))).count();
        assert_eq!(narrows, 1);
    }

    /// Global loads keep their place relative to explicit waits and
    /// scheduling hints.
    #[test]
    fn load_order_stops_at_waits_and_hints() {
        let text = |between: &str| {
            kernel(
                "    %X : ptr<f32,global>\n    %Y : ptr<f32,global>\n    %n : i32\n",
                &format!(
                    "    %a : f32 = memref.global_load_typed ( %X, %tid ) {{ align = i:4, elem_type = s:\"f32\" }}\n    \
                     %b : f32 = memref.global_load_typed ( %X, %n ) {{ align = i:4, elem_type = s:\"f32\" }}\n{between}    \
                     %c : f32 = arith.fadd ( %b, %b )\n    memref.global_store_typed ( %Y, %n, %c ) {{ align = i:4, elem_type = s:\"f32\" }}\n    \
                     %d : f32 = arith.fadd ( %a, %c )\n    memref.global_store_typed ( %Y, %tid, %d ) {{ align = i:4, elem_type = s:\"f32\" }}\n"
                ),
            )
        };
        let order = |text: &str| {
            let mut k = super::super::parse::parse(text).expect("parses");
            super::super::loadorder::order_global_loads(&mut k.body);
            k.body.ops.iter().map(|op| op.kind).collect::<Vec<_>>()
        };
        let moved = order(&text(""));
        assert!(moved.iter().position(|&k| k == OpKind::Fadd) < moved.iter().rposition(|&k| k == OpKind::GlobalLoad));
        for between in ["tile.s_waitcnt (  ) { vmcnt = i:0 }", "tile.sched_barrier (  ) { mask = i:0 }", "tile.s_setprio (  ) { level = i:1 }", "tile.s_barrier_bare (  )"] {
            let kept = order(&text(&format!("    {between}\n")));
            let hint = kept.iter().position(|&k| !matches!(k, OpKind::ThreadId | OpKind::GlobalLoad)).expect("the hint");
            assert!(kept[..hint].iter().filter(|&&k| k == OpKind::GlobalLoad).count() == 2, "{between}: {kept:?}");
        }
    }
}

/// Lane selects whose condition is known for every lane.
#[cfg(test)]
mod lane_mask_tests {
    use super::*;

    fn kernel(body: &str) -> String {
        format!(
            "ckdsl.ir v1\nkernel @probe {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n    %X : ptr<i8,global>\n    \
             %Y : ptr<i8,global>\n  }}\n  region @entry {{\n    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n{body}    \
             cf.return (  )\n  }}\n}}\n"
        )
    }

    /// Guards the optimizer proves for every thread of the workgroup (a thread
    /// id against a bound past it, as `deep_fused_conv_pool` stages its
    /// operands) leave vector selects of a divergent load on a constant
    /// condition. Each is the operand it selects: v_cndmask_b32 cannot take a
    /// constant lane mask.
    #[test]
    fn vector_select_on_a_known_condition_is_its_operand() {
        let body = r#"    %c16 : i32 = arith.constant (  ) { ity = s:"i32", value = i:16 }
    %c100 : i32 = arith.constant (  ) { ity = s:"i32", value = i:100 }
    %c1024 : i32 = arith.constant (  ) { ity = s:"i32", value = i:1024 }
    %row : i32 = arith.mul ( %tid, %c16 )
    %far : i32 = arith.add ( %row, %c1024 )
    %in : i1 = arith.cmp ( %tid, %c100 ) { pred = s:"lt" }
    %out : i1 = arith.cmp ( %tid, %c100 ) { pred = s:"ge" }
    %v : vec<i8x16> = memref.global_load_vN ( %X, %row ) { align = i:16, elem_type = s:"i8", vec = i:16 }
    %z : vec<i8x16> = arith.constant_vec (  ) { elem = s:"i8", fill = f:0.0, vec = i:16 }
    %taken : vec<i8x16> = vector.select ( %in, %v, %z )
    %kept : vec<i8x16> = vector.select ( %out, %z, %v )
    memref.global_store_vN ( %Y, %row, %taken ) { align = i:16, elem_type = s:"i8", vec = i:16 }
    memref.global_store_vN ( %Y, %far, %kept ) { align = i:16, elem_type = s:"i8", vec = i:16 }
"#;
        let compiled = super::super::compile(&kernel(body)).unwrap_or_else(|e| panic!("{e}"));
        let insts: Vec<&MInst> = compiled.func.blocks.iter().flat_map(|b| &b.insts).collect();
        assert!(!insts.iter().any(|i| i.op == Opc::VCndmaskB32), "{}", compiled.listing());
        assert_eq!(insts.iter().filter(|i| i.op == Opc::GStoreB128).count(), 2);
    }

    /// The encoder refuses a lane mask or carry-in that is not a scalar
    /// register, whatever pass put it there.
    #[test]
    fn lane_masks_come_from_scalar_registers() {
        let v = |r: u16| Opnd::P(Phys::V(r), 1);
        let encode = |op: Opc, defs: &[Opnd], m: Opnd| {
            let inst = MInst::new(op, defs, [v(1), v(2), m]);
            let f = MFunc { blocks: vec![MBlock { insts: vec![inst], loop_depth: 0 }], ..Default::default() };
            super::super::asm::assemble(&f).map(|_| ()).map_err(|e| e.kind)
        };
        let carry = [v(0), Opnd::P(Phys::Null, 1)];
        for (op, defs) in [(Opc::VCndmaskB32, &[v(0)][..]), (Opc::VAddCoCiU32, &carry[..]), (Opc::VSubCoCiU32, &carry[..])] {
            assert_eq!(encode(op, defs, Opnd::P(Phys::S(3), 1)), Ok(()));
            assert_eq!(encode(op, defs, Opnd::P(Phys::VccLo, 1)), Ok(()));
            for m in [Opnd::Imm(u32::MAX), Opnd::Imm(0), v(3)] {
                assert_eq!(encode(op, defs, m), Err("ROCKE_DIRECT_ENCODE"), "{} with {m:?}", op.name());
            }
        }
    }
}

/// Dword LDS loads merged into ds_load_b128.
#[cfg(test)]
mod dword_quad_tests {
    use super::*;

    /// A block reduction: every thread stores one f32 to `%red[tid]`, then
    /// threads below 2 sum `count` consecutive values from `%red[tid * 128 +
    /// first]`, with `between` after the first load.
    fn reduction(first: i64, count: i64, between: &str) -> String {
        let mut body = String::from(
            "    %red : smem<f32, [1x256]> = tile.smem_alloc (  )\n    \
             %c0 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }\n    \
             %c2 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:2 }\n    \
             %c128 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:128 }\n    \
             %x : f32 = memref.global_load_typed ( %F, %tid ) { align = i:4, elem_type = s:\"f32\" }\n    \
             tile.smem_store_vN_f32 ( %red, %c0, %tid, %x ) { elem_type = s:\"f32\", rank = i:2, vec = i:1 }\n    \
             tile.sync (  )\n    %lt : i1 = arith.cmp ( %tid, %c2 ) { pred = s:\"lt\" }\n    scf.if ( %lt )\n      \
             region @then {\n        %row : i32 = arith.mul ( %tid, %c128 )\n",
        );
        let mut sum = String::new();
        for j in 0..count {
            body += &format!(
                "        %k{j} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{} }}\n        \
                 %i{j} : i32 = arith.add ( %row, %k{j} )\n        \
                 %l{j} : vec<f32x1> = tile.smem_load_vN_f32 ( %red, %c0, %i{j} ) {{ elem_type = s:\"f32\", rank = i:2, vec = i:1 }}\n        \
                 %e{j} : f32 = vector.extract ( %l{j} ) {{ index = i:0 }}\n",
                first + j
            );
            if j == 0 {
                body += between;
                sum = "%e0".into();
            } else {
                body += &format!("        %s{j} : f32 = arith.fadd ( {sum}, %e{j} )\n");
                sum = format!("%s{j}");
            }
        }
        body += &format!(
            "        memref.global_store_typed ( %F, %tid, {sum} ) {{ align = i:4, elem_type = s:\"f32\" }}\n      }}\n"
        );
        format!(
            "ckdsl.ir v1\nkernel @reduce {{\n  attrs {{ max_workgroup_size = i:256 }}\n  params {{\n    \
             %F : ptr<f32,global>\n  }}\n  region @entry {{\n    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n\
             {body}    cf.return (  )\n  }}\n}}\n"
        )
    }

    fn lds_loads(text: &str, target: Target) -> (usize, usize) {
        let compiled = super::super::compile_any_target(text, target).unwrap_or_else(|e| panic!("{e}"));
        let insts: Vec<&MInst> = compiled.func.blocks.iter().flat_map(|b| &b.insts).collect();
        let count = |op: Opc| insts.iter().filter(|i| i.op == op).count();
        (count(Opc::DsLoadB128), count(Opc::DsLoadB32))
    }

    /// The reduction's reads at 16-byte aligned offsets take one
    /// `ds_load_b128` per four dwords (the LDS swizzle keeps them aligned);
    /// a partial quad, a misaligned start, a store between and gfx950 keep
    /// their `ds_load_b32`s.
    #[test]
    fn aligned_dword_loads_of_one_address_merge_in_fours() {
        assert_eq!(lds_loads(&reduction(0, 8, ""), Target::Gfx1151), (2, 0));
        assert_eq!(lds_loads(&reduction(0, 3, ""), Target::Gfx1151), (0, 3));
        assert_eq!(lds_loads(&reduction(1, 4, ""), Target::Gfx1151), (0, 4));
        let store = "        tile.smem_store_vN_f32 ( %red, %c0, %row, %x ) { elem_type = s:\"f32\", rank = i:2, vec = i:1 }\n";
        assert_eq!(lds_loads(&reduction(0, 4, store), Target::Gfx1151), (0, 4));
        assert_eq!(lds_loads(&reduction(0, 8, ""), Target::Gfx950).0, 0);
    }
}
