// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Middle end: folding, CSE, value ranges, strength reduction, loop-invariant
//! code motion, address legalization, and kernel-argument load placement.

use super::ast::{Attr, Kernel, Op, OpKind, Operands, Region, Results, Scalar, Text, Type, ValueId, DROPPED};
use super::fx::{FxHashMap, ScopedTable};
use crate::error::CompileError;

const I32_MIN: i64 = i32::MIN as i64;
const I32_MAX: i64 = i32::MAX as i64;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Konst {
    Int(i64),
    F32(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Range {
    pub lo: i64,
    pub hi: i64,
}

impl Range {
    pub fn new(lo: i64, hi: i64) -> Self {
        Self { lo, hi }
    }

    pub fn nonneg(self) -> bool {
        self.lo >= 0
    }
}

fn full_range(ty: &Type) -> Option<Range> {
    match ty {
        Type::Scalar(Scalar::I32) => Some(Range::new(I32_MIN, I32_MAX)),
        Type::Scalar(Scalar::I64) => Some(Range::new(i64::MIN, i64::MAX)),
        Type::Scalar(Scalar::I1) => Some(Range::new(0, 1)),
        _ => None,
    }
}

fn clamp_range(ty: &Type, lo: i128, hi: i128) -> Option<Range> {
    let full = full_range(ty)?;
    if lo > hi {
        return Some(full);
    }
    let lo = lo.max(full.lo as i128) as i64;
    let hi = hi.min(full.hi as i128) as i64;
    if lo > hi {
        return Some(full);
    }
    Some(Range::new(lo, hi))
}

/// Facts the middle end proves about values; isel reuses them.
#[derive(Default, Clone)]
pub struct Facts {
    pub konst: Vec<Option<Konst>>,
    pub range: Vec<Option<Range>>,
}

impl Facts {
    fn grow(&mut self, count: usize) {
        if self.konst.len() < count {
            self.konst.resize(count, None);
            self.range.resize(count, None);
        }
    }

    /// Moves the facts to the numbering of `Kernel::compact_values`.
    fn renumber(&mut self, map: &[u32]) {
        let mut kept = 0;
        for (old, &new) in map.iter().enumerate().take(self.konst.len()) {
            if new != DROPPED {
                self.konst[new as usize] = self.konst[old];
                self.range[new as usize] = self.range[old];
                kept += 1;
            }
        }
        self.konst.truncate(kept);
        self.range.truncate(kept);
    }

    pub fn int(&self, value: ValueId) -> Option<i64> {
        match self.konst.get(value.0 as usize).copied().flatten() {
            Some(Konst::Int(v)) => Some(v),
            _ => None,
        }
    }

    pub fn range_of(&self, value: ValueId) -> Option<Range> {
        self.range.get(value.0 as usize).copied().flatten()
    }

    pub fn nonneg(&self, value: ValueId) -> bool {
        self.range_of(value).is_some_and(Range::nonneg)
    }
}

/// Appends an injective encoding of `ty`.
fn push_type(words: &mut Vec<u32>, ty: &Type) {
    match ty {
        Type::Scalar(s) => words.extend([0, *s as u32]),
        Type::Ptr(s) => words.extend([1, *s as u32]),
        Type::Vec(s, n) => words.extend([2, *s as u32, *n]),
        Type::Smem(s, shape) => {
            words.extend([3, *s as u32, shape.len() as u32]);
            words.extend_from_slice(shape);
        }
    }
}

fn push_str(words: &mut Vec<u32>, text: &str) {
    words.push(text.len() as u32);
    for chunk in text.as_bytes().chunks(4) {
        let mut word = [0u8; 4];
        word[..chunk.len()].copy_from_slice(chunk);
        words.push(u32::from_le_bytes(word));
    }
}

/// Appends an encoding of `attr` under which two attributes are equal
/// exactly when their `Debug` texts are: floats compare by bits, all NaNs alike.
fn push_attr(words: &mut Vec<u32>, attr: &Attr) {
    match attr {
        Attr::Bool(b) => words.extend([0, *b as u32]),
        Attr::Int(v) => words.extend([1, *v as u32, (*v >> 32) as u32]),
        Attr::Float(f) => {
            let bits = if f.is_nan() { f64::NAN.to_bits() } else { f.to_bits() };
            words.extend([2, bits as u32, (bits >> 32) as u32]);
        }
        Attr::Str(s) => {
            words.push(3);
            push_str(words, s);
        }
        Attr::List(items) => {
            words.extend([4, items.len() as u32]);
            for item in items {
                push_attr(words, item);
            }
        }
        Attr::Map(entries) => {
            words.extend([5, entries.len() as u32]);
            push_attrs(words, entries);
        }
    }
}

fn push_attrs(words: &mut Vec<u32>, attrs: &[(Text, Attr)]) {
    words.push(attrs.len() as u32);
    for (key, value) in attrs {
        push_str(words, key);
        push_attr(words, value);
    }
}

/// Defining op of a pure value: its kind, first two operands, whether it
/// carries a `wrap` attribute (no signed no-overflow guarantee), and whether
/// all its operands are the same value.
#[derive(Clone, Copy)]
struct Def {
    kind: OpKind,
    operands: [ValueId; 2],
    count: u8,
    wrap: bool,
    splat: bool,
}

impl Def {
    fn new(op: &Op) -> Def {
        let at = |i: usize| op.operands.get(i).copied().unwrap_or(ValueId(u32::MAX));
        Def {
            kind: op.kind,
            operands: [at(0), at(1)],
            count: op.operands.len().min(2) as u8,
            wrap: op.attr("wrap").is_some(),
            splat: op.operands.iter().all(|&v| v == at(0)),
        }
    }

    /// Operand `i < 2`; panics like indexing the op's operands would.
    fn operand(&self, i: usize) -> ValueId {
        self.operands[..self.count as usize][i]
    }

    /// `(a, b)` when this is an nsw `a + b`.
    fn nsw_add(&self) -> Option<(ValueId, ValueId)> {
        (self.kind == OpKind::Add && !self.wrap).then(|| (self.operand(0), self.operand(1)))
    }
}

fn is_int(ty: &Type) -> bool {
    matches!(ty, Type::Scalar(Scalar::I32 | Scalar::I64 | Scalar::I1))
}

fn wrap(ty: &Type, value: i128) -> i64 {
    match ty {
        Type::Scalar(Scalar::I32) => value as i32 as i64,
        Type::Scalar(Scalar::I1) => (value & 1) as i64,
        _ => value as i64,
    }
}

fn mk_op(
    kind: OpKind,
    results: impl Into<Results>,
    operands: impl Into<Operands>,
    attrs: impl Into<Vec<(Text, Attr)>>,
) -> Op {
    Op {
        kind,
        results: results.into(),
        operands: operands.into(),
        attrs: attrs.into(),
        regions: Vec::new(),
        block_args: Vec::new(),
    }
}

/// `wrap` attribute value for adds the optimizer synthesizes: plain modular
/// arithmetic with no overflow assumption.
const WRAP_MODULAR: &str = "modular";

/// Attribute of a shift the optimizer made from an nsw multiply by a power
/// of two: like the multiply, it does not overflow.
const SHL_NSW: &str = "nsw";

fn modular() -> Vec<(Text, Attr)> {
    vec![("wrap".into(), Attr::Str(WRAP_MODULAR.into()))]
}

/// Tables and buffers that successive simplification sweeps reuse.
#[derive(Default)]
pub struct Workspace {
    /// Facts proved by the last sweep.
    pub facts: Facts,
    /// Sweeps keep every LDS load.
    pub keep_lds_loads: bool,
    repl: Vec<Option<ValueId>>,
    cse: ScopedTable<ValueId>,
    key: Vec<u32>,
    konst_values: FxHashMap<(Type, u64), ValueId>,
    defs: Vec<Option<Def>>,
    lds_writes: Vec<LdsExtent>,
    lds_since: Vec<u32>,
    /// Emptied op vectors to build regions in.
    spare: Vec<Vec<Op>>,
}

/// An LDS allocation and a byte range `[lo, hi)` of it.
type LdsExtent = (ValueId, i64, i64);

/// An empty vector with room for `len` ops: the smallest of `spare` that
/// fits, else a new one.
fn take_buffer(spare: &mut Vec<Vec<Op>>, len: usize) -> Vec<Op> {
    let fit = (0..spare.len()).filter(|&i| spare[i].capacity() >= len).min_by_key(|&i| spare[i].capacity());
    match fit {
        Some(i) => spare.swap_remove(i),
        None => Vec::with_capacity(len),
    }
}

struct Simplifier<'k> {
    kernel: &'k mut Kernel,
    facts: Facts,
    repl: Vec<Option<ValueId>>,
    /// Kind, result type, operands and attributes of each available pure op,
    /// and of each available LDS load followed by its `lds_epoch`.
    cse: ScopedTable<ValueId>,
    key: Vec<u32>,
    konst_values: FxHashMap<(Type, u64), ValueId>,
    changed: bool,
    defs: Vec<Option<Def>>,
    nosimplify: bool,
    lds_cse: bool,
    /// Advances at every op that may change what a later LDS load reads,
    /// except LDS stores with a known extent, which go to `lds_writes`.
    lds_epoch: u32,
    lds_writes: Vec<LdsExtent>,
    /// For each available LDS load, the length of `lds_writes` when it was
    /// made available.
    lds_since: Vec<u32>,
    spare: Vec<Vec<Op>>,
    /// Region nesting: 1 while visiting the kernel body itself.
    depth: u32,
    /// Canonical constants emitted at depth 1 and deeper, in order. They all
    /// go to the head of the kernel, the former first.
    top_constants: Vec<Op>,
    nested_constants: Vec<Op>,
    /// Whether the sweep did more than give the kernel's leading, distinct
    /// constants new values in the same order.
    rewrote: bool,
    /// Whether a non-constant op has been visited at depth 1.
    past_constants: bool,
}

impl<'k> Simplifier<'k> {
    fn resolve(&self, mut value: ValueId) -> ValueId {
        while let Some(next) = self.repl.get(value.0 as usize).copied().flatten() {
            value = next;
        }
        value
    }

    fn ty(&self, value: ValueId) -> Type {
        self.kernel.ty(value).clone()
    }

    fn sync_len(&mut self) {
        let count = self.kernel.values.len();
        self.facts.grow(count);
        if self.repl.len() < count {
            self.repl.resize(count, None);
            self.defs.resize(count, None);
            self.lds_since.resize(count, 0);
        }
    }

    fn replace(&mut self, from: ValueId, to: ValueId) {
        self.sync_len();
        self.repl[from.0 as usize] = Some(to);
        self.changed = true;
    }

    /// A constant value, materialized once at the top of the kernel.
    fn constant(&mut self, ty: Type, konst: Konst) -> ValueId {
        let bits = match konst {
            Konst::Int(v) => v as u64,
            Konst::F32(b) => b as u64,
        };
        let key = (ty.clone(), bits);
        if let Some(&value) = self.konst_values.get(&key) {
            self.rewrote = true;
            return value;
        }
        let value = self.kernel.new_value("%k", ty.clone());
        self.sync_len();
        self.facts.konst[value.0 as usize] = Some(konst);
        if let Konst::Int(v) = konst {
            self.facts.range[value.0 as usize] = Some(Range::new(v, v));
        }
        let (ity, attr) = match (&ty, konst) {
            (Type::Scalar(Scalar::I64), Konst::Int(v)) => ("i64", Attr::Int(v)),
            (Type::Scalar(Scalar::I1), Konst::Int(v)) => ("i1", Attr::Int(v)),
            (_, Konst::Int(v)) => ("i32", Attr::Int(v)),
            (_, Konst::F32(b)) => ("f32", Attr::Float(f32::from_bits(b) as f64)),
        };
        let op = mk_op(
            OpKind::Constant,
            [value],
            [],
            [("ity".into(), Attr::Str(ity.into())), ("value".into(), attr)],
        );
        if self.depth <= 1 {
            self.top_constants.push(op);
        } else {
            self.nested_constants.push(op);
        }
        self.konst_values.insert(key, value);
        value
    }

    fn int_const(&mut self, ty: &Type, value: i64) -> ValueId {
        self.rewrote = true;
        let value = wrap(ty, value as i128);
        self.constant(ty.clone(), Konst::Int(value))
    }

    fn emit(
        &mut self,
        kind: OpKind,
        ty: Type,
        operands: impl Into<Operands>,
        attrs: impl Into<Vec<(Text, Attr)>>,
        out: &mut Vec<Op>,
    ) -> ValueId {
        self.rewrote = true;
        let result = self.kernel.new_value("%t", ty);
        self.sync_len();
        let op = mk_op(kind, [result], operands, attrs);
        self.visit(op, out)
            .expect("synthesized ops are always valid");
        self.resolve(result)
    }

    fn range(&self, value: ValueId) -> Option<Range> {
        self.facts.range_of(value)
    }

    fn int(&self, value: ValueId) -> Option<i64> {
        self.facts.int(value)
    }

    /// The key buffer holding `op`'s kind, result type, operands and
    /// attributes; hand it back in `self.key`.
    fn op_key(&mut self, op: &Op) -> Vec<u32> {
        let mut key = std::mem::take(&mut self.key);
        key.clear();
        key.push(op.kind as u32);
        push_type(&mut key, self.kernel.ty(op.results[0]));
        key.push(op.operands.len() as u32);
        key.extend(op.operands.iter().map(|v| v.0));
        push_attrs(&mut key, &op.attrs);
        key
    }

    /// The result of an available op with `op`'s key; when there is none,
    /// `op`'s result becomes the available one.
    fn available(&mut self, op: &Op) -> Option<ValueId> {
        let key = self.op_key(op);
        let existing = self.cse.get(&key);
        if existing.is_none() {
            self.cse.insert(&key, op.results[0]);
        }
        self.key = key;
        existing
    }

    /// An earlier load of the same LDS bytes, in the same type, that no LDS
    /// write that may overlap them, barrier or asm has followed; when there
    /// is none, `op`'s result becomes the available load. Under the workgroup
    /// memory model another wave cannot write those bytes in between: that
    /// would be a data race. The IR has no atomic or volatile LDS loads.
    fn available_lds_load(&mut self, op: &Op) -> Option<ValueId> {
        let extent = self.lds_extent(op);
        let mut key = self.op_key(op);
        key.push(self.lds_epoch);
        let unwritten = |s: &Self, load: ValueId| {
            let since = s.lds_since[load.0 as usize] as usize;
            s.lds_writes[since..].iter().all(|&(smem, lo, hi)| {
                extent.is_some_and(|(alloc, start, end)| alloc == smem && (end <= lo || hi <= start))
            })
        };
        let existing = self.cse.get(&key).filter(|&load| unwritten(self, load));
        if existing.is_none() {
            let result = op.results[0];
            self.cse.insert(&key, result);
            self.lds_since[result.0 as usize] = self.lds_writes.len() as u32;
        }
        self.key = key;
        existing
    }

    /// The allocation and byte range a legalized LDS access may touch, from
    /// the range of its address.
    fn lds_extent(&self, op: &Op) -> Option<LdsExtent> {
        let ty = match op.kind {
            OpKind::LLoad => self.kernel.ty(op.results[0]),
            OpKind::LStore => self.kernel.ty(op.operands[2]),
            _ => return None,
        };
        let bytes = match ty {
            Type::Vec(elem, n) => elem.bytes() as i64 * *n as i64,
            other => other.elem().bytes() as i64,
        };
        let address = self.range(op.operands[1])?;
        let offset = op.int_attr("offset").unwrap_or(0);
        (address.lo >= 0 && offset >= 0).then(|| (op.operands[0], address.lo + offset, address.hi + offset + bytes))
    }

    fn clobber_lds(&mut self) {
        self.lds_epoch += 1;
        self.lds_writes.clear();
    }

    fn region(&mut self, region: Region) -> Result<Region, CompileError> {
        let mark = self.cse.mark();
        self.depth += 1;
        let mut ops = region.ops;
        let mut out = take_buffer(&mut self.spare, ops.len());
        for op in ops.drain(..) {
            self.visit(op, &mut out)?;
        }
        self.spare.push(ops);
        self.depth -= 1;
        self.cse.truncate(mark);
        Ok(Region { ops: out })
    }

    fn visit(&mut self, mut op: Op, out: &mut Vec<Op>) -> Result<(), CompileError> {
        self.sync_len();
        for operand in op.operands.iter_mut() {
            *operand = self.resolve(*operand);
        }
        if self.depth == 1 && op.kind != OpKind::Constant {
            self.past_constants = true;
        }
        match op.kind {
            OpKind::Constant => {
                if self.depth != 1 || self.past_constants {
                    self.rewrote = true;
                }
                let result = op.results[0];
                let ty = self.ty(result);
                let konst = match (op.attr("value"), &ty) {
                    (Some(Attr::Int(v)), Type::Scalar(Scalar::F32)) => Konst::F32((*v as f32).to_bits()),
                    (Some(Attr::Float(v)), Type::Scalar(Scalar::F32)) => Konst::F32((*v as f32).to_bits()),
                    (Some(Attr::Int(v)), _) if is_int(&ty) => Konst::Int(wrap(&ty, *v as i128)),
                    (Some(Attr::Float(v)), _) if is_int(&ty) => Konst::Int(wrap(&ty, *v as i128)),
                    _ => {
                        return Err(CompileError::new(
                            "ROCKE_DIRECT_UNSUPPORTED",
                            format!("unsupported constant {:?} of type {ty:?}", op.attrs),
                        ))
                    }
                };
                let canonical = self.constant(ty, konst);
                self.replace(result, canonical);
                Ok(())
            }
            OpKind::For => {
                let mut body = op.regions.pop().expect("scf.for has a body");
                let iv = op.block_args[0];
                if let (Some(lb), Some(ub)) = (self.range(op.operands[0]), self.range(op.operands[1])) {
                    let step = self.int(op.operands[2]).unwrap_or(0);
                    if step > 0 {
                        self.facts.range[iv.0 as usize] = Some(Range::new(lb.lo, (ub.hi - 1).max(lb.lo)));
                    }
                }
                // Through the back edge, the body's LDS writes precede it.
                if self.lds_cse && clobbers_lds(&body) {
                    self.clobber_lds();
                }
                body = self.region(body)?;
                op.regions.push(body);
                out.push(op);
                Ok(())
            }
            OpKind::If | OpKind::IfElse => {
                let cond = op.operands[0];
                if let Some(c) = self.int(cond) {
                    // Statically known branch: splice the taken region in place.
                    let taken = if c != 0 {
                        Some(op.regions.swap_remove(0))
                    } else if op.kind == OpKind::IfElse {
                        Some(op.regions.pop().expect("if_else has two regions"))
                    } else {
                        None
                    };
                    self.changed = true;
                    self.rewrote = true;
                    if let Some(region) = taken {
                        for inner in region.ops {
                            self.visit(inner, out)?;
                        }
                    }
                    return Ok(());
                }
                let regions = std::mem::take(&mut op.regions);
                for region in regions {
                    let simplified = self.region(region)?;
                    op.regions.push(simplified);
                }
                out.push(op);
                Ok(())
            }
            kind if kind.is_pure() && op.results.len() == 1 => {
                let result = op.results[0];
                if is_commutative(kind) && op.operands.len() == 2 {
                    let (a, b) = (op.operands[0], op.operands[1]);
                    let a_const = self.facts.konst[a.0 as usize].is_some();
                    let b_const = self.facts.konst[b.0 as usize].is_some();
                    if (a_const && !b_const) || (a_const == b_const && a > b) {
                        op.operands.swap(0, 1);
                        self.rewrote = true;
                    }
                }
                if let Some(value) = self.simplify(&op, out)? {
                    self.rewrote = true;
                    if let Some(known) = self.compute_range(&op) {
                        let slot = &mut self.facts.range[value.0 as usize];
                        *slot = Some(match *slot {
                            Some(r) => Range::new(r.lo.max(known.lo), r.hi.min(known.hi)),
                            None => known,
                        });
                    }
                    self.replace(result, value);
                    return Ok(());
                }
                // Under nosimplify nothing is merged, so the table is not kept.
                if !self.nosimplify {
                    if let Some(existing) = self.available(&op) {
                        self.rewrote = true;
                        self.replace(result, existing);
                        return Ok(());
                    }
                }
                let range = self.compute_range(&op);
                self.facts.range[result.0 as usize] = range.or_else(|| full_range(self.kernel.ty(result)));
                self.defs[result.0 as usize] = Some(Def::new(&op));
                out.push(op);
                Ok(())
            }
            OpKind::SmemLoadVec | OpKind::LLoad if self.lds_cse => {
                if let Some(existing) = self.available_lds_load(&op) {
                    self.rewrote = true;
                    self.replace(op.results[0], existing);
                    return Ok(());
                }
                out.push(op);
                Ok(())
            }
            _ => {
                if self.lds_cse && may_clobber_lds(op.kind) {
                    match self.lds_extent(&op) {
                        Some(extent) if op.kind == OpKind::LStore => self.lds_writes.push(extent),
                        _ => self.clobber_lds(),
                    }
                }
                for region in std::mem::take(&mut op.regions) {
                    let simplified = self.region(region)?;
                    op.regions.push(simplified);
                }
                for &result in &op.results {
                    let ty = self.ty(result);
                    self.facts.range[result.0 as usize] = full_range(&ty);
                }
                out.push(op);
                Ok(())
            }
        }
    }

    fn compute_range(&self, op: &Op) -> Option<Range> {
        let ty = self.ty(op.results[0]);
        let r = |i: usize| self.range(op.operands[i]);
        let wide = |range: Range| (range.lo as i128, range.hi as i128);
        match op.kind {
            OpKind::ThreadId => Some(Range::new(0, self.kernel.max_workgroup_size as i64 - 1)),
            OpKind::BlockId => Some(Range::new(0, I32_MAX)),
            OpKind::Add => {
                let (a, b) = (wide(r(0)?), wide(r(1)?));
                if op.attr("wrap").is_some() {
                    // No signed-overflow guarantee: usable only when it provably cannot wrap.
                    let full = full_range(&ty)?;
                    if a.0 + b.0 >= full.lo as i128 && a.1 + b.1 <= full.hi as i128 {
                        return clamp_range(&ty, a.0 + b.0, a.1 + b.1);
                    }
                    return None;
                }
                clamp_range(&ty, a.0 + b.0, a.1 + b.1)
            }
            OpKind::Sub => {
                let (a, b) = (wide(r(0)?), wide(r(1)?));
                clamp_range(&ty, a.0 - b.1, a.1 - b.0)
            }
            OpKind::Mul => {
                let (a, b) = (wide(r(0)?), wide(r(1)?));
                let products = [a.0 * b.0, a.0 * b.1, a.1 * b.0, a.1 * b.1];
                clamp_range(&ty, *products.iter().min()?, *products.iter().max()?)
            }
            OpKind::Div => {
                let a = r(0)?;
                let d = self.int(op.operands[1])?;
                if d <= 0 {
                    return None;
                }
                Some(Range::new(a.lo / d, a.hi / d))
            }
            OpKind::Mod => {
                let a = r(0)?;
                let d = self.int(op.operands[1]).filter(|d| *d > 0).or_else(|| {
                    let rd = r(1)?;
                    (rd.lo > 0).then_some(rd.hi)
                })?;
                if a.lo >= 0 {
                    Some(Range::new(0, a.hi.min(d - 1)))
                } else {
                    Some(Range::new(-(d - 1), d - 1))
                }
            }
            OpKind::And => {
                if ty == Type::Scalar(Scalar::I1) {
                    return Some(Range::new(0, 1));
                }
                let (a, b) = (r(0)?, r(1)?);
                match (a.nonneg(), b.nonneg()) {
                    (true, true) => Some(Range::new(0, a.hi.min(b.hi))),
                    (true, false) => Some(Range::new(0, a.hi)),
                    (false, true) => Some(Range::new(0, b.hi)),
                    _ => None,
                }
            }
            OpKind::Lshr => {
                let k = self.int(op.operands[1])?;
                if ty != Type::Scalar(Scalar::I32) || !(0..32).contains(&k) {
                    return None;
                }
                let a = r(0)?;
                if a.nonneg() {
                    Some(Range::new(a.lo >> k, a.hi >> k))
                } else if k > 0 {
                    Some(Range::new(0, (u32::MAX >> k) as i64))
                } else {
                    None
                }
            }
            OpKind::Shl => {
                let a = wide(r(0)?);
                let k = self.int(op.operands[1])?;
                let bits = if ty == Type::Scalar(Scalar::I64) { 64 } else { 32 };
                if !(0..bits).contains(&k) {
                    return None;
                }
                let (lo, hi) = (a.0 << k, a.1 << k);
                if op.attr(SHL_NSW).is_some() {
                    return clamp_range(&ty, lo, hi);
                }
                // A plain shl is modular: only exact when nothing shifts out.
                let full = full_range(&ty)?;
                if lo >= full.lo as i128 && hi <= full.hi as i128 {
                    Some(Range::new(lo as i64, hi as i64))
                } else {
                    None
                }
            }
            OpKind::Ashr => {
                let a = r(0)?;
                let k = self.int(op.operands[1])?;
                if !(0..64).contains(&k) {
                    return None;
                }
                Some(Range::new(a.lo >> k, a.hi >> k))
            }
            OpKind::UmulHi => {
                let unsigned_hi = |i: usize| -> Option<u64> {
                    match self.int(op.operands[i]) {
                        Some(v) => Some(v as u32 as u64),
                        None => r(i).filter(|range| range.nonneg()).map(|range| range.hi as u64),
                    }
                };
                let (a, b) = (unsigned_hi(0)?, unsigned_hi(1)?);
                let hi = ((a as u128 * b as u128) >> 32) as i64;
                (hi <= I32_MAX).then(|| Range::new(0, hi))
            }
            // One bounded operand bounds the result on its side.
            OpKind::Smin => {
                let full = full_range(&ty)?;
                let (a, b) = (r(0).unwrap_or(full), r(1).unwrap_or(full));
                Some(Range::new(a.lo.min(b.lo), a.hi.min(b.hi)))
            }
            OpKind::Smax => {
                let full = full_range(&ty)?;
                let (a, b) = (r(0).unwrap_or(full), r(1).unwrap_or(full));
                Some(Range::new(a.lo.max(b.lo), a.hi.max(b.hi)))
            }
            OpKind::Select => {
                let (a, b) = (r(1)?, r(2)?);
                Some(Range::new(a.lo.min(b.lo), a.hi.max(b.hi)))
            }
            OpKind::Zext => {
                let a = r(0)?;
                if a.nonneg() {
                    Some(a)
                } else if self.ty(op.operands[0]) == Type::Scalar(Scalar::I1) {
                    Some(Range::new(0, 1))
                } else {
                    Some(Range::new(0, u32::MAX as i64))
                }
            }
            OpKind::Sext | OpKind::ReadFirstLane => r(0),
            OpKind::Trunc => {
                let a = r(0)?;
                if a.lo >= I32_MIN && a.hi <= I32_MAX {
                    Some(a)
                } else {
                    None
                }
            }
            OpKind::Cmp | OpKind::Fcmp => Some(Range::new(0, 1)),
            _ => None,
        }
    }

    /// Returns a replacement value for `op`'s result, emitting any new ops.
    fn simplify(&mut self, op: &Op, out: &mut Vec<Op>) -> Result<Option<ValueId>, CompileError> {
        let result = op.results[0];
        let ty = self.ty(result);
        let operands = &op.operands;
        let c = |s: &Self, i: usize| s.int(operands[i]);
        if self.nosimplify {
            if op.kind == OpKind::Mod && ty == Type::Scalar(Scalar::I32) {
                let (a, b) = (operands[0], operands[1]);
                let q = self.emit(OpKind::Div, ty.clone(), [a, b], [], out);
                let prod = self.emit(OpKind::Mul, ty.clone(), [q, b], [], out);
                return Ok(Some(self.emit(OpKind::Sub, ty, [a, prod], [], out)));
            }
            return Ok(None);
        }
        match op.kind {
            OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::And | OpKind::Or | OpKind::Xor | OpKind::Smin
            | OpKind::Smax | OpKind::Shl | OpKind::Lshr | OpKind::Ashr | OpKind::Div | OpKind::Mod
            | OpKind::UmulHi | OpKind::SmulHi
                if is_int(&ty) =>
            {
                let (a, b) = (operands[0], operands[1]);
                if let (Some(x), Some(y)) = (c(self, 0), c(self, 1)) {
                    if let Some(v) = fold_binary(op.kind, &ty, x, y) {
                        return Ok(Some(self.int_const(&ty, v)));
                    }
                }
                let ca = c(self, 0);
                let cb = c(self, 1);
                match op.kind {
                    OpKind::Add => {
                        if cb == Some(0) {
                            return Ok(Some(a));
                        }
                        if ca == Some(0) {
                            return Ok(Some(b));
                        }
                        if ca.is_some() && cb.is_none() {
                            return Ok(Some(self.emit(OpKind::Add, ty, [b, a], op.attrs.clone(), out)));
                        }
                        let plain_nsw = op.attr("wrap").is_none();
                        if let (Some(k), true) = (cb, plain_nsw) {
                            // (x + k1) + k2 -> x + (k1 + k2): exact when both adds are nsw.
                            if let Some((inner, k1)) = self.add_const_parts(a) {
                                if self.fits(&ty, k1 as i128 + k as i128) {
                                    let sum = self.int_const(&ty, k1 + k);
                                    return Ok(Some(self.emit(OpKind::Add, ty, [inner, sum], [], out)));
                                }
                            }
                        }
                        if plain_nsw && cb.is_none() {
                            // Move constants outward, (x + k) + y -> (x + y) + k, when
                            // the ranges prove x + y cannot overflow.
                            for (with_const, other) in [(a, b), (b, a)] {
                                if let Some((inner, k)) = self.add_const_parts(with_const) {
                                    if self.add_is_exact(&ty, inner, other) {
                                        let sum = self.emit(OpKind::Add, ty.clone(), [inner, other], [], out);
                                        let kv = self.int_const(&ty, k);
                                        return Ok(Some(self.emit(OpKind::Add, ty, [sum, kv], [], out)));
                                    }
                                }
                            }
                        }
                    }
                    OpKind::Sub => {
                        if cb == Some(0) {
                            return Ok(Some(a));
                        }
                        if let Some(k) = cb {
                            if k != I32_MIN && k != i64::MIN {
                                let neg = self.int_const(&ty, -k);
                                return Ok(Some(self.emit(OpKind::Add, ty, [a, neg], [], out)));
                            }
                        }
                        if a == b {
                            return Ok(Some(self.int_const(&ty, 0)));
                        }
                        // (x + y) - x -> y and (x + y) - y -> x
                        if let Some((x, y)) = self.def_of(a).and_then(|d| d.nsw_add()) {
                            if x == b {
                                return Ok(Some(y));
                            }
                            if y == b {
                                return Ok(Some(x));
                            }
                        }
                    }
                    OpKind::Mul => {
                        if ca.is_some() && cb.is_none() {
                            return Ok(Some(self.emit(OpKind::Mul, ty, [b, a], op.attrs.clone(), out)));
                        }
                        let nsw = op.attr("wrap").is_none();
                        match cb {
                            Some(0) => return Ok(Some(self.int_const(&ty, 0))),
                            Some(1) => return Ok(Some(a)),
                            Some(k) if k > 0 => {
                                // (x + k1) * k -> x*k + k1*k when x*k provably fits.
                                if let Some((inner, k1)) = self.add_const_parts(a).filter(|_| nsw) {
                                    let exact = self.range(inner).is_some_and(|r| {
                                        self.fits(&ty, r.lo as i128 * k as i128) && self.fits(&ty, r.hi as i128 * k as i128)
                                    }) && self.fits(&ty, k1 as i128 * k as i128);
                                    if exact {
                                        let kv = self.int_const(&ty, k);
                                        let prod = self.emit(OpKind::Mul, ty.clone(), [inner, kv], [], out);
                                        let off = self.int_const(&ty, k1 * k);
                                        return Ok(Some(self.emit(OpKind::Add, ty, [prod, off], [], out)));
                                    }
                                }
                                if (k as u64).is_power_of_two() {
                                    let sh = self.int_const(&ty, k.trailing_zeros() as i64);
                                    // An nsw multiply keeps its range as a shift.
                                    let attrs = if nsw { vec![(SHL_NSW.into(), Attr::Int(1))] } else { vec![] };
                                    return Ok(Some(self.emit(OpKind::Shl, ty, [a, sh], attrs, out)));
                                }
                            }
                            _ => {}
                        }
                    }
                    OpKind::Shl if cb == Some(0) => return Ok(Some(a)),
                    OpKind::Lshr | OpKind::Ashr if cb == Some(0) => return Ok(Some(a)),
                    OpKind::And => {
                        if a == b {
                            return Ok(Some(a));
                        }
                        if ty == Type::Scalar(Scalar::I1) {
                            match (ca, cb) {
                                (Some(0), _) | (_, Some(0)) => return Ok(Some(self.int_const(&ty, 0))),
                                (Some(_), _) => return Ok(Some(b)),
                                (_, Some(_)) => return Ok(Some(a)),
                                _ => {}
                            }
                        } else {
                            if cb == Some(-1) {
                                return Ok(Some(a));
                            }
                            if let (Some(mask), Some(ra)) = (cb, self.range(a)) {
                                // Mask covering the whole known range is a no-op.
                                if mask >= 0 && ra.nonneg() && ((mask + 1) as u64).is_power_of_two() && ra.hi <= mask {
                                    return Ok(Some(a));
                                }
                            }
                        }
                    }
                    OpKind::Or => {
                        if a == b {
                            return Ok(Some(a));
                        }
                        if ty == Type::Scalar(Scalar::I1) {
                            match (ca, cb) {
                                (Some(1), _) | (_, Some(1)) => return Ok(Some(self.int_const(&ty, 1))),
                                (Some(0), _) => return Ok(Some(b)),
                                (_, Some(0)) => return Ok(Some(a)),
                                _ => {}
                            }
                        } else if cb == Some(0) {
                            return Ok(Some(a));
                        }
                    }
                    OpKind::Smin | OpKind::Smax => {
                        if let (Some(ra), Some(rb)) = (self.range(a), self.range(b)) {
                            let a_wins = if op.kind == OpKind::Smin { ra.hi <= rb.lo } else { ra.lo >= rb.hi };
                            let b_wins = if op.kind == OpKind::Smin { rb.hi <= ra.lo } else { rb.lo >= ra.hi };
                            if a_wins {
                                return Ok(Some(a));
                            }
                            if b_wins {
                                return Ok(Some(b));
                            }
                        }
                    }
                    OpKind::Div => {
                        if let Some(d) = cb {
                            return self.div_by_const(a, d, &ty, out);
                        }
                    }
                    OpKind::Mod => {
                        if let Some(d) = cb {
                            if let Some((inner, _)) = self.multiple_offset(a, d) {
                                return Ok(Some(self.emit(OpKind::Mod, ty, [inner, b], [], out)));
                            }
                            if d > 0 && self.range(a).is_some_and(|r| r.nonneg() && r.hi < d) {
                                return Ok(Some(a));
                            }
                            if d == 1 {
                                return Ok(Some(self.int_const(&ty, 0)));
                            }
                            if d > 1 && (d as u64).is_power_of_two() && self.facts.nonneg(a) {
                                let mask = self.int_const(&ty, d - 1);
                                return Ok(Some(self.emit(OpKind::And, ty, [a, mask], [], out)));
                            }
                        }
                        if ty == Type::Scalar(Scalar::I32) && cb != Some(0) {
                            // x % d == x - (x / d) * d; shares the quotient with any x / d.
                            let q = self.emit(OpKind::Div, ty.clone(), [a, b], [], out);
                            let prod = self.emit(OpKind::Mul, ty.clone(), [q, b], [], out);
                            return Ok(Some(self.emit(OpKind::Sub, ty, [a, prod], [], out)));
                        }
                    }
                    _ => {}
                }
                Ok(None)
            }
            OpKind::Cmp => {
                let (a, b) = (operands[0], operands[1]);
                let pred = op.str_attr("pred").unwrap_or("lt");
                let bool_ty = Type::Scalar(Scalar::I1);
                if let (Some(ra), Some(rb)) = (self.range(a), self.range(b)) {
                    let known = match pred {
                        "lt" => decide(ra.hi < rb.lo, ra.lo >= rb.hi),
                        "le" => decide(ra.hi <= rb.lo, ra.lo > rb.hi),
                        "gt" => decide(ra.lo > rb.hi, ra.hi <= rb.lo),
                        "ge" => decide(ra.lo >= rb.hi, ra.hi < rb.lo),
                        "eq" => decide(ra.lo == ra.hi && rb.lo == rb.hi && ra.lo == rb.lo, ra.hi < rb.lo || rb.hi < ra.lo),
                        "ne" => decide(ra.hi < rb.lo || rb.hi < ra.lo, ra.lo == ra.hi && rb.lo == rb.hi && ra.lo == rb.lo),
                        _ => None,
                    };
                    if let Some(v) = known {
                        return Ok(Some(self.int_const(&bool_ty, v as i64)));
                    }
                }
                Ok(None)
            }
            OpKind::Select => {
                let (cond, a, b) = (operands[0], operands[1], operands[2]);
                if a == b {
                    return Ok(Some(a));
                }
                match self.int(cond) {
                    Some(0) => Ok(Some(b)),
                    Some(_) => Ok(Some(a)),
                    None => Ok(None),
                }
            }
            OpKind::Zext | OpKind::Sext => {
                let src = operands[0];
                if let Some(v) = self.int(src) {
                    let src_ty = self.ty(src);
                    let value = match (op.kind, &src_ty) {
                        (OpKind::Zext, Type::Scalar(Scalar::I32)) => v as u32 as i64,
                        (OpKind::Zext, Type::Scalar(Scalar::I1)) => v & 1,
                        (OpKind::Sext, Type::Scalar(Scalar::I1)) => -(v & 1),
                        _ => v,
                    };
                    return Ok(Some(self.int_const(&ty, value)));
                }
                if self.ty(src) == ty {
                    return Ok(Some(src));
                }
                if op.kind == OpKind::Sext && self.facts.nonneg(src) {
                    return Ok(Some(self.emit(OpKind::Zext, ty, [src], [], out)));
                }
                Ok(None)
            }
            OpKind::Trunc => {
                let src = operands[0];
                if let Some(v) = self.int(src) {
                    return Ok(Some(self.int_const(&ty, v)));
                }
                if let Some(def) = self.def_of(src) {
                    if matches!(def.kind, OpKind::Zext | OpKind::Sext) && *self.kernel.ty(def.operand(0)) == ty {
                        return Ok(Some(def.operand(0)));
                    }
                }
                Ok(None)
            }
            // As LLVM folds maxnum(x, x): x when it is a number, a NaN otherwise.
            OpKind::Fmax if operands[0] == operands[1] => Ok(Some(operands[0])),
            OpKind::Extract => {
                let index = op.int_attr("index").unwrap_or(0);
                let pack = self.def_of(operands[0]).filter(|d| d.kind == OpKind::Pack);
                Ok(pack.and_then(|d| match index {
                    _ if d.splat => Some(d.operand(0)),
                    0 | 1 if index < d.count as i64 => Some(d.operand(index as usize)),
                    _ => None,
                }))
            }
            OpKind::Fmul => {
                let one = 1.0f32.to_bits();
                for (x, y) in [(0, 1), (1, 0)] {
                    if self.facts.konst[operands[y].0 as usize] == Some(Konst::F32(one)) {
                        return Ok(Some(operands[x]));
                    }
                }
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    fn fits(&self, ty: &Type, value: i128) -> bool {
        full_range(ty).is_some_and(|full| value >= full.lo as i128 && value <= full.hi as i128)
    }

    fn add_is_exact(&self, ty: &Type, a: ValueId, b: ValueId) -> bool {
        match (self.range(a), self.range(b)) {
            (Some(ra), Some(rb)) => {
                self.fits(ty, ra.lo as i128 + rb.lo as i128) && self.fits(ty, ra.hi as i128 + rb.hi as i128)
            }
            _ => false,
        }
    }

    /// `(x, k)` when `value` is `x + k` with an nsw add and constant `k`.
    fn add_const_parts(&self, value: ValueId) -> Option<(ValueId, i64)> {
        let (x, y) = self.def_of(value)?.nsw_add()?;
        Some((x, self.int(y)?))
    }

    /// `(x, k)` when `value` is `x + k` with `k` a multiple of `d > 0` and
    /// both `x` and `x + k` non-negative, so that `(x + k) / d == x / d + k / d`
    /// and `(x + k) % d == x % d` under truncating division.
    fn multiple_offset(&self, value: ValueId, d: i64) -> Option<(ValueId, i64)> {
        let (inner, k) = self.add_const_parts(value)?;
        (d > 0 && k % d == 0 && self.facts.nonneg(value) && self.facts.nonneg(inner)).then_some((inner, k))
    }

    fn def_of(&self, value: ValueId) -> Option<Def> {
        self.defs.get(value.0 as usize).copied().flatten()
    }

    fn div_by_const(&mut self, x: ValueId, d: i64, ty: &Type, out: &mut Vec<Op>) -> Result<Option<ValueId>, CompileError> {
        if *ty != Type::Scalar(Scalar::I32) || d == 0 {
            return Ok(None);
        }
        if d == 1 {
            return Ok(Some(x));
        }
        if d < 0 {
            return Ok(None);
        }
        if let Some((inner, k)) = self.multiple_offset(x, d) {
            let dv = self.int_const(ty, d);
            let q = self.emit(OpKind::Div, ty.clone(), [inner, dv], [], out);
            let kq = self.int_const(ty, k / d);
            return Ok(Some(self.emit(OpKind::Add, ty.clone(), [q, kq], [], out)));
        }
        let range = self.range(x);
        if let Some(r) = range.filter(|r| r.nonneg()) {
            if r.hi < d {
                return Ok(Some(self.int_const(ty, 0)));
            }
            if (d as u64).is_power_of_two() {
                let sh = self.int_const(ty, d.trailing_zeros() as i64);
                return Ok(Some(self.emit(OpKind::Lshr, ty.clone(), [x, sh], [], out)));
            }
            if let Some((magic, shift)) = unsigned_magic(d as u64, r.hi as u64) {
                let m = self.int_const(ty, magic as i64);
                let hi = self.emit(OpKind::UmulHi, ty.clone(), [x, m], [], out);
                if shift == 0 {
                    return Ok(Some(hi));
                }
                let sh = self.int_const(ty, shift as i64);
                return Ok(Some(self.emit(OpKind::Lshr, ty.clone(), [hi, sh], [], out)));
            }
            return Ok(None);
        }
        // Signed dividend of unknown sign.
        if (d as u64).is_power_of_two() {
            let k = d.trailing_zeros() as i64;
            let c31 = self.int_const(ty, 31);
            let sign = self.emit(OpKind::Ashr, ty.clone(), [x, c31], [], out);
            let ck = self.int_const(ty, 32 - k);
            let bias = self.emit(OpKind::Lshr, ty.clone(), [sign, ck], [], out);
            let biased = self.emit(OpKind::Add, ty.clone(), [x, bias], modular(), out);
            let kk = self.int_const(ty, k);
            return Ok(Some(self.emit(OpKind::Ashr, ty.clone(), [biased, kk], [], out)));
        }
        let (magic, shift) = signed_magic(d);
        let m = self.int_const(ty, magic as i64);
        let mut q = self.emit(OpKind::SmulHi, ty.clone(), [x, m], [], out);
        if magic < 0 {
            q = self.emit(OpKind::Add, ty.clone(), [q, x], modular(), out);
        }
        if shift > 0 {
            let sh = self.int_const(ty, shift as i64);
            q = self.emit(OpKind::Ashr, ty.clone(), [q, sh], [], out);
        }
        // Round toward zero: add one when the quotient estimate is negative.
        let c31 = self.int_const(ty, 31);
        let sign = self.emit(OpKind::Lshr, ty.clone(), [q, c31], [], out);
        Ok(Some(self.emit(OpKind::Add, ty.clone(), [q, sign], modular(), out)))
    }
}

/// Whether an op of `kind` may change what a later LDS load reads: an LDS
/// write, a barrier (past it, other waves' writes are visible), inline
/// assembly, or a kind not known to leave LDS alone. Swizzles do not access
/// LDS memory.
fn may_clobber_lds(kind: OpKind) -> bool {
    !(kind.is_pure()
        || matches!(
            kind,
            OpKind::Yield
                | OpKind::Return
                | OpKind::SmemAlloc
                | OpKind::SmemLoadVec
                | OpKind::LLoad
                | OpKind::DsSwizzleXor
                | OpKind::GlobalLoad
                | OpKind::GlobalLoadVec
                | OpKind::GLoad
                | OpKind::BufferLoadVec
                | OpKind::GlobalStore
                | OpKind::GlobalStoreVec
                | OpKind::GStore
                | OpKind::GlobalAtomicAddF32
                | OpKind::GAtomicAddF32
                | OpKind::BufferStoreVecF16
        ))
}

fn clobbers_lds(region: &Region) -> bool {
    region.ops.iter().any(|op| match op.kind {
        OpKind::For | OpKind::If | OpKind::IfElse => op.regions.iter().any(clobbers_lds),
        kind => may_clobber_lds(kind),
    })
}

fn is_commutative(kind: OpKind) -> bool {
    matches!(
        kind,
        OpKind::Add
            | OpKind::Mul
            | OpKind::And
            | OpKind::Or
            | OpKind::Xor
            | OpKind::Smin
            | OpKind::Smax
            | OpKind::UmulHi
            | OpKind::SmulHi
            | OpKind::Fadd
            | OpKind::Fmul
            | OpKind::Fmax
    )
}

fn decide(is_true: bool, is_false: bool) -> Option<bool> {
    if is_true {
        Some(true)
    } else if is_false {
        Some(false)
    } else {
        None
    }
}

fn fold_binary(kind: OpKind, ty: &Type, x: i64, y: i64) -> Option<i64> {
    let bits = match ty {
        Type::Scalar(Scalar::I64) => 64,
        Type::Scalar(Scalar::I1) => 1,
        _ => 32,
    };
    let (xw, yw) = (x as i128, y as i128);
    let unsigned = |v: i64| -> u64 {
        if bits == 64 {
            v as u64
        } else {
            (v as u64) & ((1u64 << bits) - 1)
        }
    };
    let value: i128 = match kind {
        OpKind::Add => xw + yw,
        OpKind::Sub => xw - yw,
        OpKind::Mul => xw * yw,
        OpKind::And => (x & y) as i128,
        OpKind::Or => (x | y) as i128,
        OpKind::Xor => (x ^ y) as i128,
        OpKind::Smin => xw.min(yw),
        OpKind::Smax => xw.max(yw),
        OpKind::Shl if (0..bits).contains(&y) => xw << y,
        OpKind::Lshr if (0..bits).contains(&y) => (unsigned(x) >> y) as i128,
        OpKind::Ashr if (0..bits).contains(&y) => xw >> y,
        OpKind::Div if y != 0 => xw / yw,
        OpKind::Mod if y != 0 => xw % yw,
        OpKind::UmulHi if bits == 32 => ((unsigned(x) * unsigned(y)) >> 32) as i128,
        OpKind::SmulHi if bits == 32 => (xw * yw) >> 32,
        _ => return None,
    };
    Some(wrap(ty, value))
}

/// Magic multiplier for unsigned `x / d` with `0 <= x <= x_max < 2^32`:
/// `x / d == umulhi(x, magic) >> shift`.
fn unsigned_magic(d: u64, x_max: u64) -> Option<(u32, u32)> {
    for shift in 0..32u32 {
        let pow = 1u128 << (32 + shift);
        let magic = pow.div_ceil(d as u128);
        if magic >= 1u128 << 32 {
            break;
        }
        let err = magic * d as u128 - pow;
        if err * (x_max as u128) < pow {
            return Some((magic as u32, shift));
        }
    }
    None
}

/// Hacker's Delight signed magic number for `d >= 2`.
fn signed_magic(d: i64) -> (i32, u32) {
    let d = d as i32;
    let two31: u32 = 0x8000_0000;
    let ad = d.unsigned_abs();
    let t = two31 + ((d as u32) >> 31);
    let anc = t - 1 - t % ad;
    let mut p = 31u32;
    let mut q1 = two31 / anc;
    let mut r1 = two31 - q1 * anc;
    let mut q2 = two31 / ad;
    let mut r2 = two31 - q2 * ad;
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
    let magic = q2.wrapping_add(1) as i32;
    (magic, p - 32)
}

impl<'k> Simplifier<'k> {
    fn new(kernel: &'k mut Kernel, ws: &mut Workspace) -> Self {
        let mut facts = std::mem::take(&mut ws.facts);
        facts.konst.clear();
        facts.range.clear();
        let mut repl = std::mem::take(&mut ws.repl);
        repl.clear();
        let mut defs = std::mem::take(&mut ws.defs);
        defs.clear();
        let mut konst_values = std::mem::take(&mut ws.konst_values);
        konst_values.clear();
        let mut lds_writes = std::mem::take(&mut ws.lds_writes);
        lds_writes.clear();
        let nosimplify = super::debug_flag("nosimplify");
        let mut s = Self {
            kernel,
            facts,
            repl,
            cse: std::mem::take(&mut ws.cse),
            key: std::mem::take(&mut ws.key),
            konst_values,
            changed: false,
            defs,
            nosimplify,
            lds_cse: !nosimplify && !ws.keep_lds_loads && !super::debug_flag("noldscse"),
            lds_epoch: 0,
            lds_writes,
            lds_since: std::mem::take(&mut ws.lds_since),
            spare: std::mem::take(&mut ws.spare),
            depth: 0,
            top_constants: Vec::new(),
            nested_constants: Vec::new(),
            rewrote: false,
            past_constants: false,
        };
        s.sync_len();
        for param in &s.kernel.params {
            s.facts.range[param.value.0 as usize] = full_range(&param.ty);
        }
        s
    }

    fn finish(self, ws: &mut Workspace) {
        ws.facts = self.facts;
        ws.repl = self.repl;
        ws.cse = self.cse;
        ws.key = self.key;
        ws.konst_values = self.konst_values;
        ws.defs = self.defs;
        ws.lds_writes = self.lds_writes;
        ws.lds_since = self.lds_since;
        ws.spare = self.spare;
    }
}

pub struct SimplifyResult {
    pub facts: Facts,
    pub changed: bool,
}

pub struct Sweep {
    /// Some value was replaced; constants always are.
    pub changed: bool,
    /// The kernel changed beyond its leading constants getting new values in
    /// the same order. A sweep that does not rewrite leaves a kernel that the
    /// passes treat exactly as before: they compare values only for equality
    /// and relative order.
    pub rewrote: bool,
}

/// One forward simplification sweep with the tables of `ws`, leaving the
/// facts it proved in `ws.facts`. Constants are collected at the head of the
/// kernel so every use is dominated.
pub fn sweep(kernel: &mut Kernel, ws: &mut Workspace) -> Result<Sweep, CompileError> {
    let mut simplifier = Simplifier::new(kernel, ws);
    let body = std::mem::take(&mut simplifier.kernel.body);
    let mut rest = simplifier.region(body)?.ops;
    let len = simplifier.top_constants.len() + simplifier.nested_constants.len() + rest.len();
    let mut ops = take_buffer(&mut simplifier.spare, len);
    ops.append(&mut simplifier.top_constants);
    ops.append(&mut simplifier.nested_constants);
    ops.append(&mut rest);
    simplifier.spare.push(rest);
    simplifier.kernel.body = Region { ops };
    let result = Sweep { changed: simplifier.changed, rewrote: simplifier.rewrote };
    let count = simplifier.kernel.values.len();
    simplifier.facts.grow(count);
    simplifier.finish(ws);
    Ok(result)
}

/// One forward simplification sweep.
pub fn simplify(kernel: &mut Kernel) -> Result<SimplifyResult, CompileError> {
    let mut ws = Workspace::default();
    let changed = sweep(kernel, &mut ws)?.changed;
    Ok(SimplifyResult { facts: ws.facts, changed })
}

// ---------------------------------------------------------------------------
// Loop-invariant code motion
// ---------------------------------------------------------------------------

fn collect_defs(region: &Region, out: &mut Vec<ValueId>) {
    for op in &region.ops {
        out.extend(op.results.iter().copied());
        out.extend(op.block_args.iter().copied());
        for inner in &op.regions {
            collect_defs(inner, out);
        }
    }
}

fn hoistable(kind: OpKind) -> bool {
    kind.is_pure() && !matches!(kind, OpKind::Constant | OpKind::ConstantVec | OpKind::Yield)
}

/// Hoists loop-invariant pure ops out of `scf.for` bodies, innermost first.
pub fn licm(kernel: &mut Kernel) -> bool {
    if super::debug_flag("nolicm") {
        return false;
    }
    let mut variant = vec![false; kernel.values.len()];
    licm_region(&mut kernel.body, &mut variant, &mut Vec::new())
}

/// `variant` is all false on entry and on return.
fn licm_region(region: &mut Region, variant: &mut [bool], defs: &mut Vec<ValueId>) -> bool {
    let mut changed = false;
    let mut i = 0;
    while i < region.ops.len() {
        for inner in region.ops[i].regions.iter_mut() {
            changed |= licm_region(inner, variant, defs);
        }
        if region.ops[i].kind == OpKind::For {
            let op = &mut region.ops[i];
            defs.clear();
            defs.extend(op.block_args.iter().copied());
            collect_defs(&op.regions[0], defs);
            for v in defs.iter() {
                variant[v.0 as usize] = true;
            }
            let hoisted: Vec<Op> = op.regions[0]
                .ops
                .extract_if(.., |inner| {
                    let invariant = hoistable(inner.kind) && inner.operands.iter().all(|v| !variant[v.0 as usize]);
                    if invariant {
                        for result in &inner.results {
                            variant[result.0 as usize] = false;
                        }
                    }
                    invariant
                })
                .collect();
            for v in defs.iter() {
                variant[v.0 as usize] = false;
            }
            if !hoisted.is_empty() {
                changed = true;
                let n = hoisted.len();
                region.ops.splice(i..i, hoisted);
                i += n;
            }
        }
        i += 1;
    }
    changed
}

// ---------------------------------------------------------------------------
// Dead code elimination
// ---------------------------------------------------------------------------

fn count_uses(region: &Region, uses: &mut Vec<u32>) {
    for op in &region.ops {
        for operand in &op.operands {
            uses[operand.0 as usize] += 1;
        }
        for inner in &op.regions {
            count_uses(inner, uses);
        }
    }
}

fn has_side_effects(region: &Region) -> bool {
    region.ops.iter().any(|op| match op.kind {
        OpKind::For | OpKind::If | OpKind::IfElse => op.regions.iter().any(has_side_effects),
        OpKind::Yield => false,
        kind => !kind.is_removable(),
    })
}

fn release_uses(op: &Op, uses: &mut Vec<u32>) {
    for operand in &op.operands {
        uses[operand.0 as usize] -= 1;
    }
    for inner in &op.regions {
        for nested in &inner.ops {
            release_uses(nested, uses);
        }
    }
}

/// Removes, walking backwards, ops whose results are all unused.
fn dce_region(region: &mut Region, uses: &mut Vec<u32>) -> bool {
    let mut changed = false;
    let mut dead = vec![false; region.ops.len()];
    for i in (0..region.ops.len()).rev() {
        let op = &mut region.ops[i];
        let dead_results = op.results.iter().all(|r| uses[r.0 as usize] == 0);
        let removable = match op.kind {
            OpKind::For | OpKind::If | OpKind::IfElse => dead_results && !op.regions.iter().any(has_side_effects),
            OpKind::Yield | OpKind::Return => false,
            kind => dead_results && kind.is_removable() && !op.results.is_empty(),
        };
        if removable {
            release_uses(op, uses);
            dead[i] = true;
            changed = true;
            continue;
        }
        for inner in op.regions.iter_mut() {
            changed |= dce_region(inner, uses);
        }
    }
    if dead.contains(&true) {
        let mut i = 0;
        region.ops.retain(|_| {
            i += 1;
            !dead[i - 1]
        });
    }
    changed
}

pub fn dce(kernel: &mut Kernel) -> bool {
    let mut changed = false;
    let mut uses = vec![0u32; kernel.values.len()];
    loop {
        uses.fill(0);
        count_uses(&kernel.body, &mut uses);
        if !dce_region(&mut kernel.body, &mut uses) {
            return changed;
        }
        changed = true;
    }
}

// ---------------------------------------------------------------------------
// Address legalization
// ---------------------------------------------------------------------------

const GLOBAL_OFFSET_MIN: i64 = -4096;
const GLOBAL_OFFSET_MAX: i64 = 4095;

/// A term of a linear integer form: a value, or the product of two values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Atom {
    Val(ValueId),
    Prod(ValueId, ValueId),
}

impl Atom {
    fn prod(a: ValueId, b: ValueId) -> Atom {
        Atom::Prod(a.min(b), a.max(b))
    }
}

/// A linear combination of atoms: sorted by atom, each atom once, no zero
/// coefficient.
type Form = Vec<(Atom, i64)>;

/// Turns a list of `(atom, coefficient)` terms into a `Form`.
fn normalize_form(terms: &mut Form) {
    terms.sort_unstable_by_key(|&(atom, _)| atom);
    let mut kept = 0;
    for i in 0..terms.len() {
        if kept > 0 && terms[kept - 1].0 == terms[i].0 {
            terms[kept - 1].1 = terms[kept - 1].1.wrapping_add(terms[i].1);
        } else {
            terms[kept] = terms[i];
            kept += 1;
        }
    }
    terms.truncate(kept);
    terms.retain(|&(_, c)| c != 0);
}

/// A scalar base address `ptr + (sum << shift)` already built in scope,
/// with `sum` as a linear form over atoms plus a constant.
struct ScalarBase {
    ptr: ValueId,
    shift: i64,
    form: Form,
    konst: i64,
    value: ValueId,
}

struct Legalizer<'k> {
    kernel: &'k mut Kernel,
    facts: &'k Facts,
    defs: Vec<Option<Def>>,
    uniform: Vec<bool>,
    bases: Vec<ScalarBase>,
    /// Emptied term lists for reuse.
    spare: Vec<Vec<ValueId>>,
}

type Terms = (Vec<ValueId>, Vec<ValueId>, i64);

impl<'k> Legalizer<'k> {
    fn def(&self, value: ValueId) -> Option<Def> {
        self.defs.get(value.0 as usize).copied().flatten()
    }

    fn nsw_add(&self, value: ValueId) -> Option<(ValueId, ValueId)> {
        self.def(value)?.nsw_add()
    }

    fn uni(&self, value: ValueId) -> bool {
        self.uniform.get(value.0 as usize).copied().unwrap_or(false)
    }

    fn mark(&mut self, value: ValueId, uniform: bool) {
        let i = value.0 as usize;
        if self.uniform.len() <= i {
            self.uniform.resize(i + 1, false);
        }
        self.uniform[i] = uniform;
    }

    fn new_op(&mut self, kind: OpKind, ty: Type, operands: impl Into<Operands>, out: &mut Vec<Op>) -> ValueId {
        let operands = operands.into();
        let uniform = operands.iter().all(|&v| self.uni(v));
        let result = self.kernel.new_value("%a", ty);
        self.mark(result, uniform);
        out.push(mk_op(kind, [result], operands, []));
        result
    }

    fn konst(&mut self, ty: Type, value: i64, out: &mut Vec<Op>) -> ValueId {
        let result = self.kernel.new_value("%ak", ty.clone());
        self.mark(result, true);
        let ity = if ty == Type::Scalar(Scalar::I64) { "i64" } else { "i32" };
        out.push(mk_op(
            OpKind::Constant,
            [result],
            [],
            [("ity".into(), Attr::Str(ity.into())), ("value".into(), Attr::Int(value))],
        ));
        result
    }

    /// `index = base + k` with an nsw add; `base` is None for a constant index.
    fn split(&self, index: ValueId) -> (Option<ValueId>, i64) {
        if let Some(k) = self.facts.int(index) {
            return (None, k);
        }
        if let Some((x, y)) = self.nsw_add(index) {
            if let Some(k) = self.facts.int(y) {
                return (Some(x), k);
            }
        }
        (Some(index), 0)
    }

    /// `terms` of `index` in lists to hand back with `recycle`.
    fn split_terms(&mut self, index: ValueId) -> Terms {
        let mut parts = (self.spare.pop().unwrap_or_default(), self.spare.pop().unwrap_or_default(), 0);
        self.terms(index, 0, &mut parts);
        parts
    }

    fn recycle(&mut self, (mut uniform, mut divergent, _): Terms) {
        uniform.clear();
        divergent.clear();
        self.spare.push(uniform);
        self.spare.push(divergent);
    }

    /// Splits an nsw add tree into uniform subtrees, divergent leaves and a
    /// constant; sign extension distributes over each such add.
    fn terms(&self, v: ValueId, depth: u32, parts: &mut Terms) {
        if let Some(k) = self.facts.int(v) {
            parts.2 += k;
        } else if self.uni(v) {
            parts.0.push(v);
        } else if let Some((x, y)) = (depth < 8).then(|| self.nsw_add(v)).flatten() {
            self.terms(x, depth + 1, parts);
            self.terms(y, depth + 1, parts);
        } else {
            parts.1.push(v);
        }
    }

    /// Adds `scale * v` to `form` as terms over atoms, exact over the
    /// integers for nsw adds and multiplies.
    fn linear(&self, v: ValueId, depth: u32, scale: i64, form: &mut (Form, i64)) {
        if let Some(k) = self.facts.int(v) {
            form.1 += scale * k;
            return;
        }
        let def = if depth < 8 { self.def(v) } else { None };
        match def {
            Some(d) if d.kind == OpKind::Add && !d.wrap => {
                self.linear(d.operand(0), depth + 1, scale, form);
                self.linear(d.operand(1), depth + 1, scale, form);
            }
            Some(d) if d.kind == OpKind::Mul && !d.wrap => {
                let (a, b) = (d.operand(0), d.operand(1));
                if let Some(k) = self.facts.int(b) {
                    return self.linear(a, depth + 1, scale * k, form);
                }
                for (sum, other) in [(a, b), (b, a)] {
                    if let Some((x, y)) = self.nsw_add(sum) {
                        if let Some(k) = self.facts.int(y) {
                            form.0.push((Atom::prod(x, other), scale));
                            form.0.push((Atom::Val(other), scale * k));
                            return;
                        }
                    }
                }
                form.0.push((Atom::prod(a, b), scale));
            }
            _ => form.0.push((Atom::Val(v), scale)),
        }
    }

    /// The uniform value `v` and power `p` when `to - from == 2^p * v`.
    fn single_step(&self, from: &[(Atom, i64)], to: &[(Atom, i64)]) -> Option<(ValueId, i64)> {
        let mut step = None;
        let (mut i, mut j) = (0, 0);
        while i < from.len() || j < to.len() {
            let (key, d) = match (from.get(i), to.get(j)) {
                (Some(&(a, x)), Some(&(b, y))) if a == b => {
                    i += 1;
                    j += 1;
                    (a, y - x)
                }
                (Some(&(a, x)), next) if next.is_none_or(|&(b, _)| a < b) => {
                    i += 1;
                    (a, 0 - x)
                }
                (_, Some(&(b, y))) => {
                    j += 1;
                    (b, y)
                }
                (_, None) => unreachable!("one list has a next atom"),
            };
            if d == 0 {
                continue;
            }
            match (key, step) {
                (Atom::Val(v), None) if self.uni(v) && d > 0 && d <= 1 << 16 && (d as u64).is_power_of_two() => {
                    step = Some((v, d.trailing_zeros() as i64));
                }
                _ => return None,
            }
        }
        step
    }

    /// `ptr + (sum(terms) << shift)`, reusing a base already in scope that
    /// differs from it by nothing or by one power-of-two multiple of a
    /// uniform value; rows of a strided tile then cost one 64-bit add each.
    fn scalar_base(&mut self, ptr: ValueId, terms: &[ValueId], shift: i64, out: &mut Vec<Op>) -> ValueId {
        if terms.is_empty() {
            return ptr;
        }
        let mut form = (Form::new(), 0i64);
        for &term in terms {
            self.linear(term, 0, 1, &mut form);
        }
        normalize_form(&mut form.0);
        let ptr_ty = self.kernel.ty(ptr).clone();
        let mut found = None;
        for b in self.bases.iter().rev() {
            if b.ptr != ptr || b.shift != shift || b.konst != form.1 {
                continue;
            }
            if b.form == form.0 {
                return b.value;
            }
            if found.is_none() {
                found = self.single_step(&b.form, &form.0).map(|step| (b.value, step));
            }
        }
        let base = if let Some((prev, (v, p))) = found {
            let delta = self.scaled64(v, shift + p, out);
            self.new_op(OpKind::GlobalPtrAdd, ptr_ty, [prev, delta], out)
        } else {
            let mut base = ptr;
            for &term in terms {
                let scaled = self.scaled64(term, shift, out);
                base = self.new_op(OpKind::GlobalPtrAdd, ptr_ty.clone(), [base, scaled], out);
            }
            base
        };
        self.bases.push(ScalarBase { ptr, shift, form: form.0, konst: form.1, value: base });
        base
    }

    /// `sext(v) << shift` as i64.
    fn scaled64(&mut self, v: ValueId, shift: i64, out: &mut Vec<Op>) -> ValueId {
        let i64t = Type::Scalar(Scalar::I64);
        let wide = if *self.kernel.ty(v) == i64t { v } else { self.new_op(OpKind::Sext, i64t.clone(), [v], out) };
        if shift == 0 {
            return wide;
        }
        let s = self.konst(i64t.clone(), shift, out);
        self.new_op(OpKind::Shl, i64t, [wide, s], out)
    }

    /// Address operands and immediate offset for element `index` of a global
    /// pointer. The uniform part of the index folds into a scalar base; a
    /// single divergent term that is a non-negative i32 becomes the 32-bit
    /// VGPR offset of the SGPR-base form, `[sbase, voffset]`, which the
    /// hardware adds unsigned, and so does a sum of such terms whose scaled
    /// range fits. Anything else is one 64-bit address, `[addr]`.
    fn global_address(&mut self, ptr: ValueId, index: ValueId, bytes: u32, out: &mut Vec<Op>) -> (Operands, i64) {
        let index_ty = self.kernel.ty(index).clone();
        let mut parts = self.split_terms(index);
        let shift = bytes.trailing_zeros() as i64;
        let lanes = self.uni(ptr) && index_ty == Type::Scalar(Scalar::I32);
        let single = match parts.1.as_slice() {
            [] if self.uni(ptr) => Some(None),
            [d] if lanes => Some(Some(*d)),
            terms if lanes && self.offset_sum_fits(terms, shift) => {
                let mut sum = terms[0];
                for &term in &terms[1..] {
                    sum = self.new_op(OpKind::Add, index_ty.clone(), [sum, term], out);
                }
                Some(Some(sum))
            }
            _ => None,
        };
        let Some(single) = single else {
            self.recycle(parts);
            let (addr, offset) = self.flat_address(ptr, index, bytes, out);
            return ([addr].into(), offset);
        };
        let k = parts.2;
        let mut offset = k * bytes as i64;
        if !(GLOBAL_OFFSET_MIN..=GLOBAL_OFFSET_MAX).contains(&offset) {
            let konst = self.konst(index_ty.clone(), k, out);
            parts.0.push(konst);
            offset = 0;
        }
        let ptr_ty = self.kernel.ty(ptr).clone();
        let base = self.scalar_base(ptr, &parts.0, shift, out);
        let summed = parts.1.len() > 1;
        self.recycle(parts);
        let Some(d) = single else {
            return ([base].into(), offset);
        };
        let fits = summed || self.offset_sum_fits(&[d], shift);
        if fits {
            let voffset = if shift == 0 {
                d
            } else {
                let s = self.konst(index_ty.clone(), shift, out);
                self.new_op(OpKind::Shl, index_ty, [d, s], out)
            };
            return ([base, voffset].into(), offset);
        }
        let scaled = self.scaled64(d, shift, out);
        let addr = self.new_op(OpKind::GlobalPtrAdd, ptr_ty, [base, scaled], out);
        ([addr].into(), offset)
    }

    /// Whether the sum of `terms`, each a non-negative i32, is exact in i32
    /// and, shifted left by `shift`, an unsigned 32-bit offset.
    fn offset_sum_fits(&self, terms: &[ValueId], shift: i64) -> bool {
        let mut hi = 0i64;
        for &term in terms {
            match self.facts.range_of(term) {
                Some(r) if r.nonneg() => hi += r.hi,
                _ => return false,
            }
        }
        hi <= I32_MAX && (shift <= 1 || hi < (1i64 << 32) >> shift)
    }

    /// `(addr, offset)` for element `index` of a global pointer, as one
    /// 64-bit address.
    fn flat_address(&mut self, ptr: ValueId, index: ValueId, bytes: u32, out: &mut Vec<Op>) -> (ValueId, i64) {
        let index_ty = self.kernel.ty(index).clone();
        let (base, k) = self.split(index);
        // The constant of the whole add tree folds into the offset, so
        // accesses a constant apart share an address. Summing the other
        // terms in 32 bits is exact when none is negative and k >= 0.
        let parts = self.split_terms(index);
        let (base, k) = if parts.2 > 0
            && index_ty == Type::Scalar(Scalar::I32)
            && parts.0.iter().chain(&parts.1).all(|&t| self.facts.nonneg(t))
        {
            let mut sum = None;
            for &t in parts.0.iter().chain(&parts.1) {
                sum = Some(match sum {
                    None => t,
                    Some(s) => self.new_op(OpKind::Add, index_ty.clone(), [s, t], out),
                });
            }
            (sum, parts.2)
        } else {
            (base, k)
        };
        self.recycle(parts);
        let folded = k * bytes as i64;
        let (base, offset) = if (GLOBAL_OFFSET_MIN..=GLOBAL_OFFSET_MAX).contains(&folded) {
            (base, folded)
        } else {
            (Some(index), 0)
        };
        let Some(base) = base else {
            return (ptr, offset);
        };
        let wide = if index_ty == Type::Scalar(Scalar::I64) {
            base
        } else {
            self.new_op(OpKind::Sext, Type::Scalar(Scalar::I64), [base], out)
        };
        let shift = self.konst(Type::Scalar(Scalar::I64), bytes.trailing_zeros() as i64, out);
        let byte_offset = self.new_op(OpKind::Shl, Type::Scalar(Scalar::I64), [wide, shift], out);
        let ptr_ty = self.kernel.ty(ptr).clone();
        let addr = self.new_op(OpKind::GlobalPtrAdd, ptr_ty, [ptr, byte_offset], out);
        (addr, offset)
    }

    /// `(addr, offset)` for an LDS element at `indices` of an `smem` allocation.
    fn lds_address(&mut self, smem: ValueId, indices: &[ValueId], out: &mut Vec<Op>) -> Result<(ValueId, i64), CompileError> {
        let Type::Smem(elem, shape) = self.kernel.ty(smem).clone() else {
            return Err(CompileError::new("ROCKE_DIRECT_UNSUPPORTED", "LDS access through a non-smem value"));
        };
        if indices.len() != shape.len() || !(1..=2).contains(&indices.len()) {
            return Err(CompileError::new("ROCKE_DIRECT_UNSUPPORTED", "LDS access of rank other than 1 or 2"));
        }
        let bytes = elem.bytes() as i64;
        let i32t = Type::Scalar(Scalar::I32);
        let mut constant = 0i64;
        let mut linear: Option<ValueId> = None;
        for (dim, &index) in indices.iter().enumerate() {
            let stride: i64 = shape[dim + 1..].iter().map(|&d| d as i64).product();
            let (base, k) = self.split(index);
            constant += k * stride;
            if let Some(base) = base {
                let scaled = if stride == 1 {
                    base
                } else {
                    let s = self.konst(i32t.clone(), stride, out);
                    self.new_op(OpKind::Mul, i32t.clone(), [base, s], out)
                };
                linear = Some(match linear {
                    None => scaled,
                    Some(prev) => self.new_op(OpKind::Add, i32t.clone(), [prev, scaled], out),
                });
            }
        }
        let addr = match linear {
            None => self.konst(i32t.clone(), 0, out),
            Some(index) if bytes == 1 => index,
            Some(index) => {
                let s = self.konst(i32t.clone(), bytes.trailing_zeros() as i64, out);
                self.new_op(OpKind::Shl, i32t.clone(), [index, s], out)
            }
        };
        Ok((addr, constant * bytes))
    }

    fn region(&mut self, region: &mut Region) -> Result<(), CompileError> {
        let scope = self.bases.len();
        let mut out = Vec::with_capacity(region.ops.len());
        for mut op in std::mem::take(&mut region.ops) {
            for inner in op.regions.iter_mut() {
                self.region(inner)?;
            }
            if op.kind.is_pure() && op.results.len() == 1 {
                let i = op.results[0].0 as usize;
                if self.defs.len() <= i {
                    self.defs.resize(i + 1, None);
                }
                self.defs[i] = Some(Def::new(&op));
            }
            let align = op.int_attr("align");
            let vec = op.int_attr("vec").unwrap_or(1);
            match op.kind {
                OpKind::GlobalLoad | OpKind::GlobalLoadVec => {
                    let bytes = self.kernel.ty(op.results[0]).elem().bytes();
                    let (addr, offset) = self.global_address(op.operands[0], op.operands[1], bytes, &mut out);
                    // Defaults mirror the LLVM lowering.
                    let default = if op.kind == OpKind::GlobalLoadVec { vec * 2 } else { 1 };
                    let mut attrs = std::mem::take(&mut op.attrs);
                    attrs.clear();
                    attrs.push(("offset".into(), Attr::Int(offset)));
                    attrs.push(("align".into(), Attr::Int(align.unwrap_or(default))));
                    out.push(mk_op(OpKind::GLoad, op.results, addr, attrs));
                }
                OpKind::GlobalStore | OpKind::GlobalStoreVec | OpKind::GlobalAtomicAddF32 => {
                    let value = op.operands[2];
                    let bytes = self.kernel.ty(value).elem().bytes();
                    let (addr, offset) = self.global_address(op.operands[0], op.operands[1], bytes, &mut out);
                    let default = match op.kind {
                        OpKind::GlobalStoreVec => vec * bytes as i64,
                        OpKind::GlobalAtomicAddF32 => 4,
                        _ => 1,
                    };
                    let mut attrs = std::mem::take(&mut op.attrs);
                    attrs.clear();
                    attrs.push(("offset".into(), Attr::Int(offset)));
                    attrs.push(("align".into(), Attr::Int(align.unwrap_or(default))));
                    let kind = if op.kind == OpKind::GlobalAtomicAddF32 { OpKind::GAtomicAddF32 } else { OpKind::GStore };
                    let mut operands = addr;
                    operands.push(value);
                    out.push(mk_op(kind, [], operands, attrs));
                }
                OpKind::SmemLoadVec => {
                    let smem = op.operands[0];
                    let (addr, offset) = self.lds_address(smem, &op.operands[1..], &mut out)?;
                    let bytes = self.kernel.ty(op.results[0]).elem().bytes() as i64;
                    let natural = if matches!(vec, 3 | 6 | 12) { 12 / vec } else { vec * bytes };
                    let mut attrs = std::mem::take(&mut op.attrs);
                    attrs.clear();
                    attrs.push(("offset".into(), Attr::Int(offset)));
                    attrs.push(("align".into(), Attr::Int(natural)));
                    out.push(mk_op(OpKind::LLoad, op.results, [smem, addr], attrs));
                }
                OpKind::SmemStore | OpKind::SmemStoreVec => {
                    let smem = op.operands[0];
                    let value = *op.operands.last().expect("store has a value");
                    let count = op.operands.len();
                    let (addr, offset) = self.lds_address(smem, &op.operands[1..count - 1], &mut out)?;
                    let bytes = self.kernel.ty(value).elem().bytes() as i64;
                    let default = if op.kind == OpKind::SmemStoreVec { vec * bytes } else { bytes };
                    let mut attrs = std::mem::take(&mut op.attrs);
                    attrs.clear();
                    attrs.push(("offset".into(), Attr::Int(offset)));
                    attrs.push(("align".into(), Attr::Int(align.unwrap_or(default))));
                    out.push(mk_op(OpKind::LStore, [], [smem, addr, value], attrs));
                }
                OpKind::BufferLoadVec | OpKind::BufferStoreVecF16 => {
                    let voffset = op.operands[1];
                    let (base, k) = self.split(voffset);
                    let mut offset = 0;
                    if let Some(base) = base {
                        if (0..4096).contains(&k) && k != 0 && self.facts.nonneg(base) {
                            op.operands[1] = base;
                            offset = k;
                        }
                    }
                    op.set_attr("offset", Attr::Int(offset));
                    out.push(op);
                }
                _ => out.push(op),
            }
        }
        self.bases.truncate(scope);
        region.ops = out;
        Ok(())
    }
}

pub fn legalize_addresses(kernel: &mut Kernel, facts: &Facts) -> Result<(), CompileError> {
    let uniform = super::analysis::classify(kernel);
    let mut body = std::mem::take(&mut kernel.body);
    let mut legalizer = Legalizer {
        kernel,
        facts,
        defs: Vec::new(),
        uniform,
        bases: Vec::new(),
        spare: Vec::new(),
    };
    legalizer.region(&mut body)?;
    kernel.body = body;
    Ok(())
}

// ---------------------------------------------------------------------------
// Kernel argument placement
// ---------------------------------------------------------------------------

/// Byte offset of every kernel argument (natural alignment, no hidden args)
/// and the total kernarg segment size.
pub fn kernarg_layout(kernel: &Kernel) -> (Vec<u32>, u32) {
    let mut offset = 0u32;
    let mut offsets = Vec::with_capacity(kernel.params.len());
    for param in &kernel.params {
        let size = param.ty.dwords() * 4;
        offset = offset.next_multiple_of(size);
        offsets.push(offset);
        offset += size;
    }
    (offsets, offset)
}

fn mark_param_uses(region: &Region, param_of: &[Option<u32>], used: &mut [bool]) {
    for op in &region.ops {
        for operand in &op.operands {
            if let Some(&Some(param)) = param_of.get(operand.0 as usize) {
                used[param as usize] = true;
            }
        }
        for inner in &op.regions {
            mark_param_uses(inner, param_of, used);
        }
    }
}

fn substitute(region: &mut Region, map: &[Option<ValueId>]) {
    for op in region.ops.iter_mut() {
        if op.kind != OpKind::KernArg {
            for operand in op.operands.iter_mut() {
                if let Some(&Some(to)) = map.get(operand.0 as usize) {
                    *operand = to;
                }
            }
        }
        for inner in op.regions.iter_mut() {
            substitute(inner, map);
        }
    }
}

/// Inserts one `KernArg` load per used parameter at the kernel entry, in
/// offset order, so that isel can merge neighbours into wide scalar loads
/// that issue back to back and share one wait.
pub fn place_kernargs(kernel: &mut Kernel) {
    let mut param_of = vec![None; kernel.values.len()];
    for (index, param) in kernel.params.iter().enumerate() {
        param_of[param.value.0 as usize] = Some(index as u32);
    }
    let mut used = vec![false; kernel.params.len()];
    mark_param_uses(&kernel.body, &param_of, &mut used);
    let (offsets, _) = kernarg_layout(kernel);
    let mut loads = Vec::new();
    let mut map = vec![None; kernel.values.len()];
    for param_index in 0..kernel.params.len() {
        if !used[param_index] {
            continue;
        }
        let param_value = kernel.params[param_index].value;
        let ty = kernel.params[param_index].ty.clone();
        let name = format!("%arg.{}", kernel.params[param_index].name);
        let value = kernel.new_value(&name, ty);
        map[param_value.0 as usize] = Some(value);
        let op = mk_op(
            OpKind::KernArg,
            [value],
            [],
            [
                ("offset".into(), Attr::Int(offsets[param_index] as i64)),
                ("param".into(), Attr::Int(param_index as i64)),
            ],
        );
        loads.push(op);
    }
    loads.sort_by_key(|op| op.int_attr("offset").unwrap_or(0));
    kernel.body.ops.splice(0..0, loads);
    substitute(&mut kernel.body, &map);
}

/// Renumbers the kernel's values densely, carrying the facts and the LDS
/// offsets along. Passes compare values only for equality and relative
/// order, so this changes nothing but the size of the tables they index.
pub fn compact(kernel: &mut Kernel, ws: &mut Workspace, lds: &mut FxHashMap<ValueId, u32>) {
    let map = kernel.compact_values();
    ws.facts.renumber(&map);
    ws.konst_values.clear();
    *lds = lds
        .iter()
        .filter(|(v, _)| map[v.0 as usize] != DROPPED)
        .map(|(v, &offset)| (ValueId(map[v.0 as usize]), offset))
        .collect();
}

/// The full middle-end pipeline. Returns facts valid for the final IR, whose
/// values are renumbered; `lds` follows the renumbering.
pub fn optimize(kernel: &mut Kernel, lds: &mut FxHashMap<ValueId, u32>) -> Result<Facts, CompileError> {
    optimize_with(kernel, lds, Workspace::default())
}

/// `optimize` without merging LDS loads, which lengthens live ranges.
pub fn optimize_keeping_lds_loads(kernel: &mut Kernel, lds: &mut FxHashMap<ValueId, u32>) -> Result<Facts, CompileError> {
    optimize_with(kernel, lds, Workspace { keep_lds_loads: true, ..Workspace::default() })
}

fn optimize_with(kernel: &mut Kernel, lds: &mut FxHashMap<ValueId, u32>, mut ws: Workspace) -> Result<Facts, CompileError> {
    // LDS loads merge once addresses are legalized: equal addresses are then
    // equal values, and stores have extents.
    let keep_lds_loads = std::mem::replace(&mut ws.keep_lds_loads, true);
    sweep(kernel, &mut ws)?;
    compact(kernel, &mut ws, lds);
    // Once a round only renumbers constants, every later round and the next
    // sweep would only renumber them again, and the facts in `ws` hold.
    let mut settled = false;
    for _ in 0..4 {
        let hoisted = licm(kernel);
        let swept = sweep(kernel, &mut ws)?;
        let removed = dce(kernel);
        if !(hoisted || swept.rewrote || removed) {
            settled = true;
            break;
        }
        if !(hoisted || swept.changed || removed) {
            break;
        }
    }
    if !settled {
        sweep(kernel, &mut ws)?;
    }
    legalize_addresses(kernel, &ws.facts)?;
    ws.keep_lds_loads = keep_lds_loads;
    sweep(kernel, &mut ws)?;
    compact(kernel, &mut ws, lds);
    licm(kernel);
    sweep(kernel, &mut ws)?;
    dce(kernel);
    place_kernargs(kernel);
    sweep(kernel, &mut ws)?;
    dce(kernel);
    compact(kernel, &mut ws, lds);
    Ok(ws.facts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn optimized(text: &str) -> Kernel {
        let mut kernel = super::super::parse::parse(text).unwrap_or_else(|e| panic!("{e}"));
        let (mut lds, _) = super::super::analysis::lds_layout(&kernel);
        optimize(&mut kernel, &mut lds).unwrap_or_else(|e| panic!("{e}"));
        kernel
    }

    fn names_of(kernel: &Kernel, region: &Region, kind: OpKind, out: &mut Vec<String>) {
        for op in &region.ops {
            if op.kind == kind {
                out.push(kernel.name_of(op.results[0]).to_string());
            }
            for inner in &op.regions {
                names_of(kernel, inner, kind, out);
            }
        }
    }

    const LDS_PROBE: &str = r#"ckdsl.ir v1
kernel @lds_probe {
  attrs { max_workgroup_size = i:64 }
  params {
    %Y : ptr<f16,global>
    %Z : ptr<f32,global>
    %n : i32
  }
  region @entry {
    %tid : i32 = gpu.thread_id (  ) { axis = s:"x" }
    %lds : smem<f16, [1024]> = tile.smem_alloc (  )
    %c0 : i32 = arith.constant (  ) { ity = s:"i32", value = i:0 }
    %c1 : i32 = arith.constant (  ) { ity = s:"i32", value = i:1 }
    %c8 : i32 = arith.constant (  ) { ity = s:"i32", value = i:8 }
    %c512 : i32 = arith.constant (  ) { ity = s:"i32", value = i:512 }
    %row : i32 = arith.mul ( %tid, %c8 )
    %high : i32 = arith.add ( %row, %c512 )
    %g : vec<f16x8> = memref.global_load_vN ( %Y, %row ) { align = i:16, elem_type = s:"f16", vec = i:8 }
    tile.smem_store_vN ( %lds, %row, %g ) { align = i:16, elem_type = s:"f16", rank = i:1, vec = i:8 }
    tile.sync (  ) { fence = s:"compiler" }
    %a1 : vec<f16x8> = tile.smem_load_vN ( %lds, %row ) { elem_type = s:"f16", rank = i:1, vec = i:8 }
    %a2 : vec<f16x8> = tile.smem_load_vN ( %lds, %row ) { elem_type = s:"f16", rank = i:1, vec = i:8 }
    %sw : i32 = tile.ds_swizzle_xor ( %tid ) { xor_mask = i:1 }
    tile.smem_store_vN ( %lds, %high, %g ) { align = i:16, elem_type = s:"f16", rank = i:1, vec = i:8 }
    %a3 : vec<f16x8> = tile.smem_load_vN ( %lds, %row ) { elem_type = s:"f16", rank = i:1, vec = i:8 }
    tile.smem_store_vN ( %lds, %row, %g ) { align = i:16, elem_type = s:"f16", rank = i:1, vec = i:8 }
    %a4 : vec<f16x8> = tile.smem_load_vN ( %lds, %row ) { elem_type = s:"f16", rank = i:1, vec = i:8 }
    %a5 : vec<f16x8> = tile.smem_load_vN ( %lds, %row ) { elem_type = s:"f16", rank = i:1, vec = i:8 }
    tile.sync (  ) { fence = s:"compiler" }
    %a6 : vec<f16x8> = tile.smem_load_vN ( %lds, %row ) { elem_type = s:"f16", rank = i:1, vec = i:8 }
    %one : f32 = arith.constant (  ) { ity = s:"f32", value = f:1.0 }
    %q : f32 = tile.inline_asm ( %one ) { constraints = s:"=v,0", convergent = b:false, sideeffect = b:false, template = s:"" }
    %a7 : vec<f16x8> = tile.smem_load_vN ( %lds, %row ) { elem_type = s:"f16", rank = i:1, vec = i:8 }
    scf.for ( %c0, %n, %c1 ) { iv = s:"%i", iv_type = s:"i32", num_iter_args = i:0, unroll = b:false }
      region @body {
        %b1 : vec<f16x8> = tile.smem_load_vN ( %lds, %row ) { elem_type = s:"f16", rank = i:1, vec = i:8 }
        memref.global_store_vN ( %Y, %row, %b1 ) { align = i:16, elem_type = s:"f16", vec = i:8 }
        scf.yield (  ) { num = i:0 }
      }
    scf.for ( %c0, %n, %c1 ) { iv = s:"%j", iv_type = s:"i32", num_iter_args = i:0, unroll = b:false }
      region @body {
        %c1l : vec<f16x8> = tile.smem_load_vN ( %lds, %row ) { elem_type = s:"f16", rank = i:1, vec = i:8 }
        memref.global_store_vN ( %Y, %row, %c1l ) { align = i:16, elem_type = s:"f16", vec = i:8 }
        tile.smem_store_vN ( %lds, %high, %g ) { align = i:16, elem_type = s:"f16", rank = i:1, vec = i:8 }
        scf.yield (  ) { num = i:0 }
      }
    %d1 : vec<f16x8> = tile.smem_load_vN ( %lds, %row ) { elem_type = s:"f16", rank = i:1, vec = i:8 }
    %lt : i1 = arith.cmp ( %tid, %n ) { pred = s:"lt" }
    scf.if ( %lt )
      region @then {
        %e1 : vec<f16x8> = tile.smem_load_vN ( %lds, %row ) { elem_type = s:"f16", rank = i:1, vec = i:8 }
        memref.global_store_vN ( %Y, %row, %e1 ) { align = i:16, elem_type = s:"f16", vec = i:8 }
      }
    %e2 : vec<f16x8> = tile.smem_load_vN ( %lds, %row ) { elem_type = s:"f16", rank = i:1, vec = i:8 }
    %w4 : vec<f16x4> = tile.smem_load_vN ( %lds, %row ) { elem_type = s:"f16", rank = i:1, vec = i:4 }
    memref.global_store_vN ( %Y, %row, %a1 ) { align = i:16, elem_type = s:"f16", vec = i:8 }
    memref.global_store_vN ( %Y, %row, %a2 ) { align = i:16, elem_type = s:"f16", vec = i:8 }
    memref.global_store_vN ( %Y, %row, %a3 ) { align = i:16, elem_type = s:"f16", vec = i:8 }
    memref.global_store_vN ( %Y, %row, %a4 ) { align = i:16, elem_type = s:"f16", vec = i:8 }
    memref.global_store_vN ( %Y, %row, %a5 ) { align = i:16, elem_type = s:"f16", vec = i:8 }
    memref.global_store_vN ( %Y, %row, %a6 ) { align = i:16, elem_type = s:"f16", vec = i:8 }
    memref.global_store_vN ( %Y, %row, %a7 ) { align = i:16, elem_type = s:"f16", vec = i:8 }
    memref.global_store_vN ( %Y, %row, %d1 ) { align = i:16, elem_type = s:"f16", vec = i:8 }
    memref.global_store_vN ( %Y, %row, %e2 ) { align = i:16, elem_type = s:"f16", vec = i:8 }
    memref.global_store_vN ( %Y, %row, %w4 ) { align = i:8, elem_type = s:"f16", vec = i:4 }
    memref.global_store_typed ( %Z, %sw, %q ) { align = i:4, elem_type = s:"f32" }
    cf.return (  )
  }
}
"#;

    #[test]
    fn lds_loads_merge_only_without_an_intervening_clobber() {
        let kernel = optimized(LDS_PROBE);
        let mut loads = Vec::new();
        names_of(&kernel, &kernel.body, OpKind::LLoad, &mut loads);
        // a2 repeats a1; a3 follows only a store to other bytes (and a
        // swizzle); a5 repeats a4. The first loop's body writes no LDS, so
        // b1 reuses a7; the second loop's store reaches c1 through the back
        // edge and d1 after the loop. e1 and e2 reuse d1, which dominates
        // them; w4 has another width.
        assert_eq!(loads, ["%a1", "%a4", "%a6", "%a7", "%c1l", "%d1", "%w4"]);
    }

    #[test]
    fn lds_loads_stay_unmerged_when_merging_runs_out_of_registers() {
        let mut text = String::from(
            "ckdsl.ir v1\nkernel @pressure_probe {\n  attrs { max_workgroup_size = i:64 }\n  params {\n    \
             %Y : ptr<f16,global>\n  }\n  region @entry {\n    %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n    \
             %lds : smem<f16, [2048]> = tile.smem_alloc (  )\n    \
             %c8 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:8 }\n    \
             %row : i32 = arith.mul ( %tid, %c8 )\n",
        );
        let rows = 70;
        for i in 0..rows {
            text += &format!(
                "    %k{i} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{} }}\n    \
                 %at{i} : i32 = arith.add ( %row, %k{i} )\n",
                8 * i
            );
        }
        // Each load is read twice; merged, all of them stay live in between.
        for pass in 0..2 {
            for i in 0..rows {
                text += &format!(
                    "    %l{pass}_{i} : vec<f16x8> = tile.smem_load_vN ( %lds, %at{i} ) \
                     {{ elem_type = s:\"f16\", rank = i:1, vec = i:8 }}\n    \
                     memref.global_store_vN ( %Y, %at{i}, %l{pass}_{i} ) {{ align = i:16, elem_type = s:\"f16\", vec = i:8 }}\n"
                );
            }
        }
        text += "    cf.return (  )\n  }\n}\n";
        let kernel = optimized(&text);
        let mut loads = Vec::new();
        names_of(&kernel, &kernel.body, OpKind::LLoad, &mut loads);
        assert_eq!(loads.len(), rows);
        let compiled = super::super::compile(&text).unwrap_or_else(|e| panic!("{e}"));
        let ds_loads = compiled.func.blocks.iter().flat_map(|b| &b.insts).filter(|i| i.op == super::super::mir::Opc::DsLoadB128);
        assert_eq!(ds_loads.count(), 2 * rows);
    }

    #[test]
    fn readfirstlane_keeps_the_range_of_its_operand() {
        let text = r#"ckdsl.ir v1
kernel @lane_probe {
  attrs { max_workgroup_size = i:64 }
  params {
    %Y : ptr<f32,global>
  }
  region @entry {
    %tid : i32 = gpu.thread_id (  ) { axis = s:"x" }
    %c2 : i32 = arith.constant (  ) { ity = s:"i32", value = i:2 }
    %c5 : i32 = arith.constant (  ) { ity = s:"i32", value = i:5 }
    %wave : i32 = arith.lshr ( %tid, %c5 )
    %w : i32 = tile.readfirstlane ( %wave )
    %lt : i1 = arith.cmp ( %w, %c2 ) { pred = s:"lt" }
    scf.if ( %lt )
      region @then {
        %f : f32 = arith.constant (  ) { ity = s:"f32", value = f:1.0 }
        memref.global_store_typed ( %Y, %w, %f ) { align = i:4, elem_type = s:"f32" }
      }
    cf.return (  )
  }
}
"#;
        let kernel = optimized(text);
        assert!(kernel.body.ops.iter().all(|op| op.kind != OpKind::If), "w < 2 always holds");
    }

    fn global_stores(region: &Region, out: &mut Vec<usize>) {
        for op in &region.ops {
            if op.kind == OpKind::GStore {
                out.push(op.operands.len());
            }
            for inner in &op.regions {
                global_stores(inner, out);
            }
        }
    }

    #[test]
    fn splat_quads_merge_and_lane_sums_use_the_sgpr_base() {
        let text = r#"ckdsl.ir v1
kernel @fill_probe {
  attrs { max_workgroup_size = i:64 }
  params {
    %Y : ptr<f32,global>
  }
  region @entry {
    %tid : i32 = gpu.thread_id (  ) { axis = s:"x" }
    %c3 : i32 = arith.constant (  ) { ity = s:"i32", value = i:3 }
    %c5 : i32 = arith.constant (  ) { ity = s:"i32", value = i:5 }
    %c6 : i32 = arith.constant (  ) { ity = s:"i32", value = i:6 }
    %c7 : i32 = arith.constant (  ) { ity = s:"i32", value = i:7 }
    %col : i32 = arith.and ( %tid, %c7 )
    %colx : i32 = arith.shl ( %col, %c3 )
    %row : i32 = arith.lshr ( %tid, %c3 )
    %rowx : i32 = arith.shl ( %row, %c6 )
    %idx : i32 = arith.add ( %colx, %rowx )
    %back : i32 = arith.sub ( %c5, %tid )
    %low : i32 = arith.add ( %colx, %back )
    %k : f32 = arith.constant (  ) { ity = s:"f32", value = f:0.0 }
    %v8 : vec<f32x8> = vector.pack ( %k, %k, %k, %k, %k, %k, %k, %k ) { elem = s:"f32", vec = i:8 }
    %e0 : f32 = vector.extract ( %v8 ) { index = i:0 }
    %e1 : f32 = vector.extract ( %v8 ) { index = i:1 }
    %e2 : f32 = vector.extract ( %v8 ) { index = i:2 }
    %e3 : f32 = vector.extract ( %v8 ) { index = i:3 }
    %e4 : f32 = vector.extract ( %v8 ) { index = i:4 }
    %e5 : f32 = vector.extract ( %v8 ) { index = i:5 }
    %e6 : f32 = vector.extract ( %v8 ) { index = i:6 }
    %e7 : f32 = vector.extract ( %v8 ) { index = i:7 }
    %lo : vec<f32x4> = vector.pack ( %e0, %e1, %e2, %e3 ) { elem = s:"f32", vec = i:4 }
    %hi : vec<f32x4> = vector.pack ( %e4, %e5, %e6, %e7 ) { elem = s:"f32", vec = i:4 }
    memref.global_store_vN ( %Y, %idx, %lo ) { align = i:16, elem_type = s:"f32", vec = i:4 }
    memref.global_store_vN ( %Y, %low, %hi ) { align = i:16, elem_type = s:"f32", vec = i:4 }
    cf.return (  )
  }
}
"#;
        let kernel = optimized(text);
        let mut packs = Vec::new();
        names_of(&kernel, &kernel.body, OpKind::Pack, &mut packs);
        assert_eq!(packs, ["%lo"]);
        // [sbase, voffset, value] for the sum of two lane terms; one 64-bit
        // address when a lane term may be negative.
        let mut stores = Vec::new();
        global_stores(&kernel.body, &mut stores);
        assert_eq!(stores, [3, 2]);
    }
}