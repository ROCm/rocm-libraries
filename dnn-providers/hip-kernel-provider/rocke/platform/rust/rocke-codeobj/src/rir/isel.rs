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

pub fn is_inline_int(v: u32) -> bool {
    (-16..=64).contains(&(v as i32))
}

pub fn is_inline_f32(v: u32) -> bool {
    matches!(
        v,
        0x3f00_0000 | 0xbf00_0000 | 0x3f80_0000 | 0xbf80_0000 | 0x4000_0000 | 0xc000_0000 | 0x4080_0000 | 0xc080_0000 | 0x3e22_f983
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
}

pub struct IselResult {
    pub func: MFunc,
    pub kernarg_size: u32,
    pub lds_size: u32,
    pub wg_ids: [bool; 3],
}

struct Isel<'a> {
    k: &'a Kernel,
    facts: &'a Facts,
    uniform: Vec<bool>,
    f: MFunc,
    cur: usize,
    vals: Vec<Option<Val>>,
    lds: &'a FxHashMap<ValueId, u32>,
    depth: u32,
    single_wave: bool,
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
}

/// A 16-bit gather load waiting for the insert that receives it.
enum Deferred {
    /// Global address operands, modifiers, offset.
    Global(Uses, u32, i64),
    /// LDS address and offset.
    Lds(Opnd, i64),
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

const EXEC: Opnd = Opnd::P(Phys::ExecLo, 1);
const VCC: Opnd = Opnd::P(Phys::VccLo, 1);
const NULL: Opnd = Opnd::P(Phys::Null, 1);

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
        let mut scalars = Uses::new();
        let mut literal: Option<u32> = None;
        for i in 0..srcs.len() {
            match srcs[i] {
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

    // ----- booleans -----

    fn scc(&mut self, v: ValueId) -> Result<(), CompileError> {
        match self.get(v)? {
            Val::C(c) => self.salu(c.op, [], [c.a, c.b]),
            Val::M(m) => self.salu(Opc::SCmpLgU32, [], [m, Opnd::Imm(0)]),
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
                let dst = self.sreg(1);
                self.salu(Opc::SCselectB32, [r(dst)], [Opnd::Imm(u32::MAX), Opnd::Imm(0)]);
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
            self.salu(Opc::SCmpLgU32, [], [mask, Opnd::Imm(0)]);
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
        let carry = self.sreg(1);
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
            let ge = self.sreg(1);
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
        match op.kind {
            Constant => {
                let v = res.expect("constant has a result");
                let konst = self.facts.konst.get(v.0 as usize).copied().flatten();
                let val = match (self.k.ty(v), konst) {
                    (Type::Scalar(Scalar::I64), Some(super::opt::Konst::Int(x))) => {
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
                if op.str_attr("axis") != Some("x") {
                    return Err(unsupported("only gpu.thread_id x is supported"));
                }
                self.set(res.unwrap(), Val::W(r(self.tid)));
            }
            BlockId => {
                let axis = match op.str_attr("axis") {
                    Some("x") => 0,
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
            Add | Sub | Mul | And | Or | Xor | Shl | Lshr | Ashr | Smin | Smax | UmulHi | SmulHi => self.int_op(op)?,
            Div => self.div_op(op)?,
            Mod => return Err(unsupported("arith.mod survived the optimizer")),
            Cmp => self.cmp_op(op)?,
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
                let dst = self.sreg(1);
                self.valu(Opc::VCmp(pred), [r(dst)], [a, b], 0);
                self.set(res.unwrap(), Val::M(r(dst)));
            }
            Select => self.select_op(op)?,
            Zext | Sext | Trunc | Bitcast => self.convert_op(op)?,
            GlobalPtrAdd => {
                let v = res.unwrap();
                let ptr = self.d(op.operands[0])?;
                let off_ty = self.k.ty(op.operands[1]).clone();
                let off = if off_ty == Type::Scalar(Scalar::I64) {
                    self.d(op.operands[1])?
                } else {
                    let lo = self.w(op.operands[1])?;
                    let hi = self.sign_hi(lo, op.operands[1]);
                    (lo, hi)
                };
                let (lo, hi) = self.add64(ptr, off, self.uni(v), false);
                self.set(v, Val::D(lo, hi));
            }
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
                let a = self.w(op.operands[0])?;
                let b = self.w(op.operands[1])?;
                let c = self.w(op.operands[2])?;
                let out = self.v3(Opc::VFmaF32, a, b, c, 0);
                self.set(res.unwrap(), Val::W(out));
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
                let m = self.sreg(1);
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
                let m = self.sreg(1);
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
                let flag = self.sreg(1);
                self.valu(Opc::VDivScaleF32, [r(num), r(flag)], [one, x, one], 0);
                let rc = self.vgpr(1);
                self.valu(Opc::VRcpF32, [r(rc)], [r(den)], 0);
                let t = self.v3(Opc::VFmaF32, r(den), r(rc), one, M_NEG0);
                let r2 = self.v3(Opc::VFmaF32, t, r(rc), r(rc), 0);
                let q = self.v_bin(Opc::VMulF32, r(num), r2);
                let e = self.v3(Opc::VFmaF32, r(den), q, r(num), M_NEG0);
                let q2 = self.v3(Opc::VFmaF32, e, r2, q, 0);
                let e2 = self.v3(Opc::VFmaF32, r(den), q2, r(num), M_NEG0);
                self.emit(MInst::new(Opc::SMovB32, [VCC], [r(flag)]));
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
                    (Val::T(t), Scalar::F16) => Val::H(r(sub(t, index / 2, 1)), index % 2 == 1),
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
                if self.k.ty(v).elem() == Scalar::F16 {
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
                        let Deferred::Global(mut uses, mods, offset) = load else {
                            return Err(unsupported("LDS gather into a vector insert"));
                        };
                        let dst = self.vgpr(src.n);
                        for i in 0..src.n {
                            self.copy(sub(dst, i, 1), r(sub(src, i, 1)));
                        }
                        let part = r(sub(dst, d, 1));
                        let opc = if index % 2 == 0 { Opc::GLoadD16B16 } else { Opc::GLoadD16HiB16 };
                        uses.push(part);
                        self.emit(MInst::new(opc, [part], uses).with_imm(offset).with_mods(mods));
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
            Pack => {
                let v = res.unwrap();
                let ty = self.k.ty(v).clone();
                if ty.elem() == Scalar::F16 {
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
                if self.k.ty(v).elem() == Scalar::F16 && first_len % 2 == 1 {
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
                    _ => return Err(unsupported("constant_vec element type")),
                };
                let n = ty.dwords() as u8;
                let dst = self.vgpr(n);
                for i in 0..n {
                    self.copy(sub(dst, i, 1), Opnd::Imm(bits));
                }
                self.set(v, Val::T(dst));
            }
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
                self.emit(MInst::new(Opc::GAtomicAddF32, [], uses).with_imm(offset).with_mods(mods));
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
                self.emit(MInst::new(opc, [r(dst)], [voff, r(rsrc), soff]).with_imm(offset));
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
                self.emit(MInst::new(opc, [], [r(data), voff, r(rsrc), soff]).with_imm(offset));
            }
            DsSwizzleXor => {
                let mask = op.int_attr("xor_mask").unwrap_or(0);
                let x = self.w(op.operands[0])?;
                let x = self.to_v(x);
                let dst = self.vgpr(1);
                self.emit(MInst::new(Opc::DsSwizzleB32, [r(dst)], [x]).with_imm((mask << 10) | 0x1f));
                self.set(res.unwrap(), Val::W(r(dst)));
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
            Sync => {
                if !self.single_wave {
                    self.emit(MInst::new(Opc::SBarrier, [], []));
                }
            }
            ReadFirstLane => {
                let v = res.unwrap();
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
                let x = self.w(op.operands[0])?;
                let out = self.to_v(x);
                self.set(res.unwrap(), Val::W(out));
            }
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

    fn soffset(&mut self, v: ValueId) -> Result<Opnd, CompileError> {
        let o = self.w(v)?;
        Ok(match o {
            Opnd::Imm(x) if is_inline_int(x) => o,
            _ => self.to_s(o),
        })
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
        match ty {
            Type::Scalar(Scalar::I1) if self.chains.finite.contains_key(&v) => {
                let x = self.w(self.chains.finite[&v])?;
                let dst = self.sreg(1);
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
                let out = self.s_bin(opc, a, b);
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
                        self.shl64(op.operands[0], a, k as u32, uniform, v)
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

    fn cmp_op(&mut self, op: &Op) -> Result<(), CompileError> {
        let v = op.results[0];
        let pred = op.str_attr("pred").unwrap_or("lt");
        let ty = self.k.ty(op.operands[0]).clone();
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
        let dst = self.sreg(1);
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
            let t = self.s_bin(Opc::SAndB32, mc, ma);
            let f = self.s_bin(Opc::SAndNot1B32, mb, mc);
            let out = self.s_bin(Opc::SOrB32, t, f);
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
                outs.push(self.v3(Opc::VCndmaskB32, y, x, m, 0));
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
        if self.gathers.contains(&v) {
            self.deferred.insert(v, Deferred::Global(addr, mods, offset));
            return Ok(());
        }
        let dwords = bytes.div_ceil(4) as u8;
        let dst = self.vgpr(dwords.max(1));
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
                    .with_mods(mods),
            );
            done += n;
        }
        if bytes % 4 == 2 {
            self.emit(
                MInst::new(Opc::GLoadU16, [r(sub(dst, full, 1))], addr)
                    .with_imm(offset + 4 * full as i64)
                    .with_mods(mods),
            );
        }
        let val = match ty {
            Type::Scalar(Scalar::F16) => Val::H(r(dst), false),
            Type::Scalar(Scalar::I64) | Type::Ptr(_) => Val::D(r(sub(dst, 0, 1)), r(sub(dst, 1, 1))),
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
        let bytes = match &ty {
            Type::Vec(e, n) => e.bytes() * n,
            other => other.elem().bytes(),
        };
        if bytes == 2 {
            let (o, hi_half) = self.half(value)?;
            let o = self.to_v(o);
            let opc = if hi_half { Opc::GStoreD16HiB16 } else { Opc::GStoreB16 };
            let mut uses = addr;
            uses.push(o);
            self.emit(MInst::new(opc, [], uses).with_imm(offset).with_mods(mods));
            return Ok(());
        }
        let data = match ty {
            Type::Scalar(Scalar::I64) | Type::Ptr(_) => {
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
            self.emit(MInst::new(opc, [], uses).with_imm(offset + 4 * done as i64).with_mods(mods));
            done += n;
        }
        if bytes % 4 == 2 {
            let mut uses = addr;
            uses.push(r(sub(data, full, 1)));
            self.emit(MInst::new(Opc::GStoreB16, [], uses).with_imm(offset + 4 * full as i64).with_mods(mods));
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
            Type::Scalar(Scalar::F16) => Val::H(r(dst), false),
            Type::Scalar(_) => Val::W(r(dst)),
            _ => Val::T(dst),
        };
        self.set(v, val);
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
        if bytes == 2 {
            let (o, hi) = self.half(value)?;
            let o = self.to_v(o);
            let opc = if hi { Opc::DsStoreB16D16Hi } else { Opc::DsStoreB16 };
            self.emit(MInst::new(opc, [], [addr, o]).with_imm(offset));
            return Ok(());
        }
        let data = match ty {
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
                let reg = self.sreg(1);
                (reg, Val::M(r(reg)))
            }
            Type::Scalar(Scalar::I64) | Type::Ptr(_) => {
                let reg = if uniform { self.sreg(2) } else { self.vgpr(2) };
                (reg, Val::D(r(sub(reg, 0, 1)), r(sub(reg, 1, 1))))
            }
            Type::Scalar(Scalar::F16) => {
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
            Type::Scalar(Scalar::I64) | Type::Ptr(_) => {
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
            return Err(unsupported("scf.for with divergent bounds"));
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
        let yields = body.ops.last().map(|y| y.operands.clone()).unwrap_or_default();
        // Parallel copy of the yielded values into the carried registers.
        let reads_carried = yields.iter().any(|&y| {
            let reg = match self.vals[y.0 as usize] {
                Some(Val::W(Opnd::R(v))) | Some(Val::T(v)) | Some(Val::M(Opnd::R(v))) => Some(v.reg),
                Some(Val::D(Opnd::R(v), _)) | Some(Val::H(Opnd::R(v), _)) => Some(v.reg),
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
                Type::Scalar(Scalar::I64) | Type::Ptr(_) => Val::D(r(sub(carried[i], 0, 1)), r(sub(carried[i], 1, 1))),
                Type::Scalar(Scalar::F16) => Val::H(r(carried[i]), false),
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
        let saved = self.sreg(1);
        self.emit(MInst::new(Opc::SAndSaveexecB32, [r(saved), EXEC], [mask, EXEC]));
        self.emit(MInst::new(Opc::SCbranchExecz, [], []));
        let first = (self.cur, self.f.blocks[self.cur].insts.len() - 1);
        self.new_block();
        self.region(&op.regions[0])?;
        if has_else {
            let else_block = self.new_block();
            self.f.blocks[first.0].insts[first.1].imm = else_block as i64;
            self.salu(Opc::SAndNot1B32, [EXEC], [r(saved), mask]);
            self.emit(MInst::new(Opc::SCbranchExecz, [], []));
            let second = (self.cur, self.f.blocks[self.cur].insts.len() - 1);
            self.new_block();
            self.region(&op.regions[1])?;
            let end = self.new_block();
            self.f.blocks[second.0].insts[second.1].imm = end as i64;
            self.emit(MInst::new(Opc::SMovB32, [EXEC], [r(saved)]));
        } else {
            let end = self.new_block();
            self.f.blocks[first.0].insts[first.1].imm = end as i64;
            self.salu(Opc::SOrB32, [EXEC], [EXEC, r(saved)]);
        }
        Ok(())
    }
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
            && *k.ty(op.results[0]) == Type::Vec(Scalar::F16, 1)
            && uses[op.results[0].0 as usize] == 1
        {
            lds_halves.insert(op.results[0], i);
        }
        if op.kind == OpKind::Concat && *k.ty(op.results[0]) == Type::Vec(Scalar::F16, 2) {
            let (a, b) = (op.operands[0], op.operands[1]);
            if let (Some(&pa), Some(&pb)) = (lds_halves.get(&a), lds_halves.get(&b)) {
                if a != b && last_effect.is_none_or(|e| e < pa.min(pb)) {
                    out.insert(a);
                    out.insert(b);
                }
            }
        }
        if op.kind != OpKind::GLoad {
            continue;
        }
        let v = op.results[0];
        if uniform[v.0 as usize] || *k.ty(v) != Type::Scalar(Scalar::F16) || uses[v.0 as usize] != 1 {
            continue;
        }
        for later in &region.ops[i + 1..] {
            if later.kind == OpKind::Insert && later.operands[1] == v {
                if later.operands[0] != v && k.ty(later.results[0]).elem() == Scalar::F16 {
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
    let f16_insert = |op: &Op| op.kind == OpKind::Insert && cx.k.ty(op.results[0]).elem() == Scalar::F16;
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
        if def.kind != OpKind::LLoad || cx.k.ty(e).elem() != Scalar::F16 || bytes != 2 || cx.uses[e.0 as usize] != 1 {
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

fn find_block_ids(region: &Region, used: &mut [bool; 3]) {
    for op in &region.ops {
        if op.kind == OpKind::BlockId {
            match op.str_attr("axis") {
                Some("x") => used[0] = true,
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
    find_finite_tests(&cx, &kernel.body, &mut chains);
    let mut isel = Isel {
        k: kernel,
        facts,
        uniform,
        f: MFunc::default(),
        cur: 0,
        vals: vec![None; kernel.values.len()],
        lds,
        depth: 0,
        single_wave: kernel.max_workgroup_size <= 32,
        kernarg_ptr: VRef { reg: 0, off: 0, n: 2 },
        tid: VRef { reg: 0, off: 0, n: 1 },
        bid: [None; 3],
        zero_v: None,
        uses,
        gathers,
        chains,
        deferred: FxHashMap::default(),
        canonical: Vec::new(),
    };
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
    if let Some(z) = isel.zero_v {
        isel.f.blocks[0].insts.insert(0, MInst::new(Opc::Copy, [r(z)], [Opnd::Imm(0)]));
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
        // Sources of every f32 max, including the halves of VOPD pairs.
        let mut maxes: Vec<&[Opnd]> = Vec::new();
        for i in &insts {
            if i.op == Opc::VMaxF32 {
                maxes.push(&i.uses);
            } else if i.op == Opc::VDual {
                let (x, y) = i.uses.split_at(i.mods as usize);
                for (code, srcs) in [(i.imm & 0xff, x), (i.imm >> 8 & 0xff, y)] {
                    if vopd_name(code as u8) == "v_dual_max_f32" {
                        maxes.push(srcs);
                    }
                }
            }
        }
        let quiets = maxes.iter().filter(|s| s[0] == s[1]).count();
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
