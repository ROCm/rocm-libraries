// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Checks a parsed kernel before any pass runs. Every op must have the
//! operand and result counts, types and attributes its lowering relies on, so
//! that no later pass indexes past an operand list, a vector or a register
//! tuple; and limits the reference lowering enforces for gfx1151 are enforced
//! here too. Errors are `ROCKE_DIRECT_MALFORMED` for IR that is not well
//! formed, `ROCKE_DIRECT_INVALID` for IR the LLVM lowering or COMGR refuses
//! for gfx1151, and `ROCKE_DIRECT_UNSUPPORTED` for limits of this compiler.

use super::ast::{Attr, Kernel, Op, OpKind, Region, Scalar, Type, ValueId};
use crate::error::CompileError;

/// Deepest region nesting accepted, counting the kernel body. Instruction
/// selection recurses once per level; a 1 MiB thread stack takes about 2.4
/// times this many levels of loops. Kernels nest a few levels, and long
/// before this depth every level's saved EXEC mask or loop counter has used
/// up the SGPRs.
pub const MAX_NESTING: usize = 128;

/// Largest vector, in bytes: the widest scalar load (`s_load_b512`).
pub const MAX_VECTOR_BYTES: u32 = 64;

/// LDS a gfx1151 workgroup can allocate.
pub const MAX_LDS_BYTES: u32 = 64 << 10;

/// Largest flat workgroup size.
pub const MAX_WORKGROUP_SIZE: u32 = 1024;

fn malformed(message: impl Into<String>) -> CompileError {
    CompileError::new("ROCKE_DIRECT_MALFORMED", message)
}

fn invalid(message: impl Into<String>) -> CompileError {
    CompileError::new("ROCKE_DIRECT_INVALID", message)
}

fn unsupported(message: impl Into<String>) -> CompileError {
    CompileError::new("ROCKE_DIRECT_UNSUPPORTED", message)
}

/// Whether `name` is an LLVM identifier that needs no quotes, as the
/// lowering prints it.
fn plain_symbol(name: &str) -> bool {
    let ok = |c: char| c.is_ascii_alphanumeric() || matches!(c, '-' | '$' | '.' | '_');
    name.chars().next().is_some_and(|c| ok(c) && !c.is_ascii_digit()) && name.chars().all(ok)
}

fn check_type(ty: &Type) -> Result<(), String> {
    match ty {
        Type::Vec(elem, n) => {
            if *elem == Scalar::I1 {
                return Err("vectors of i1".into());
            }
            let bytes = elem.bytes().checked_mul(*n).unwrap_or(u32::MAX);
            if *n == 0 || bytes > MAX_VECTOR_BYTES {
                return Err(format!("vectors of {n} elements of {elem:?} (at most {MAX_VECTOR_BYTES} bytes)"));
            }
        }
        Type::Smem(elem, shape) => {
            if *elem == Scalar::I1 {
                return Err("LDS allocations of i1".into());
            }
            let elems = shape.iter().try_fold(1u64, |acc, &d| acc.checked_mul(d as u64));
            let bytes = elems.and_then(|e| e.checked_mul(elem.bytes() as u64));
            if !bytes.is_some_and(|b| b <= u32::MAX as u64) {
                return Err(format!("an LDS allocation of shape {shape:?}"));
            }
        }
        Type::Scalar(_) | Type::Ptr(_) => {}
    }
    Ok(())
}

/// The kernel-level and per-op checks; see the module documentation.
pub fn kernel(kernel: &Kernel) -> Result<(), CompileError> {
    if !plain_symbol(&kernel.name) {
        return Err(malformed(format!("kernel name {:?} is not a plain symbol", kernel.name)));
    }
    if !(1..=MAX_WORKGROUP_SIZE).contains(&kernel.max_workgroup_size) {
        return Err(invalid(format!(
            "max_workgroup_size {} is outside 1..={MAX_WORKGROUP_SIZE}",
            kernel.max_workgroup_size
        )));
    }
    let checker = Checker { kernel };
    // Regions to check, in any order: the ops, whether it is a loop body,
    // the number of loops around it, and its nesting depth.
    let mut stack: Vec<(&[Op], bool, usize, usize)> = vec![(&kernel.body.ops, false, 0, 1)];
    while let Some((ops, loop_body, loops, depth)) = stack.pop() {
        for (i, op) in ops.iter().enumerate() {
            let context = |e: CompileError| {
                let name = op.results.first().map_or("", |&v| kernel.values.get(v.0 as usize).map_or("", |d| d.name(&kernel.names)));
                CompileError::new(e.kind, format!("{:?} {name}: {}", op.kind, e.message))
            };
            checker.op(op).map_err(context)?;
            // A loop body's trailing yield was checked with its loop; any
            // other yields nothing, inside some loop (as the LLVM lowering
            // requires).
            let terminates_body = loop_body && i + 1 == ops.len();
            if op.kind == OpKind::Yield && !terminates_body && !(op.operands.is_empty() && loops > 0) {
                return Err(context(malformed("scf.yield that does not end a loop body")));
            }
            if op.regions.is_empty() {
                continue;
            }
            if depth >= MAX_NESTING {
                return Err(unsupported(format!("regions nest deeper than {MAX_NESTING} levels")));
            }
            let is_loop = op.kind == OpKind::For;
            let regions: &[Region] = &op.regions;
            stack.extend(regions.iter().map(|r| (&r.ops[..], is_loop, loops + is_loop as usize, depth + 1)));
        }
    }
    Ok(())
}

/// The LDS pool `analysis::lds_layout` packed: what the LLVM lowering
/// allocates, and gfx1151 refuses above 64 KiB.
pub fn lds_pool(bytes: u32) -> Result<(), CompileError> {
    if bytes > MAX_LDS_BYTES {
        return Err(invalid(format!("the kernel needs {bytes} bytes of LDS; a workgroup has at most {MAX_LDS_BYTES}")));
    }
    Ok(())
}

/// Checks of the optimized kernel against `analysis::classify`. `cf.return`
/// ends the whole wave, which is right only where every active lane
/// returns: not inside an `scf.if` whose condition is divergent, nor in a
/// loop with per-lane bounds, which lanes leave at different trips. A buffer
/// access takes its resource and `soffset` from SGPRs, where the LLVM
/// backend would loop over the distinct values of divergent ones.
pub fn uniformity(kernel: &Kernel, uniform: &[bool]) -> Result<(), CompileError> {
    let uni = |v: &ValueId| uniform.get(v.0 as usize) == Some(&true);
    let mut stack = vec![(kernel.body.ops.iter(), false)];
    while let Some((ops, divergent)) = stack.last_mut() {
        let divergent = *divergent;
        let Some(op) = ops.next() else {
            stack.pop();
            continue;
        };
        match op.kind {
            OpKind::Return if divergent => {
                return Err(unsupported("cf.return under a divergent scf.if condition or loop bound"));
            }
            OpKind::BufferLoadVec | OpKind::BufferStoreVecF16 | OpKind::BufferLoadScalar | OpKind::BufferStoreScalar
                if !(op.operands.first().is_some_and(uni) && op.operands.get(2).is_some_and(uni)) =>
            {
                return Err(unsupported(format!("{:?} with a divergent resource or soffset", op.kind)));
            }
            OpKind::If | OpKind::IfElse => {
                let cond = op.operands.first().is_some_and(uni);
                stack.extend(op.regions.iter().map(|r| (r.ops.iter(), divergent || !cond)));
            }
            OpKind::For => {
                let bounds = op.operands.get(..3).is_some_and(|b| b.iter().all(uni));
                stack.extend(op.regions.iter().map(|r| (r.ops.iter(), divergent || !bounds)));
            }
            _ => stack.extend(op.regions.iter().map(|r| (r.ops.iter(), divergent))),
        }
    }
    Ok(())
}

struct Checker<'k> {
    kernel: &'k Kernel,
}

static PLACEHOLDER: Type = Type::Scalar(Scalar::I1);

/// The attributes of a memory access the checks read, from one pass.
#[derive(Default)]
struct MemoryAttrs<'a> {
    align: Option<&'a Attr>,
    vec: Option<&'a Attr>,
    nontemporal: Option<&'a Attr>,
}

impl<'a> MemoryAttrs<'a> {
    fn of(op: &'a Op) -> Self {
        let mut out = Self::default();
        // The first of equal keys, as `Op::attr` finds.
        for (key, value) in op.attrs.iter().rev() {
            match &**key {
                "align" => out.align = Some(value),
                "vec" => out.vec = Some(value),
                "nontemporal" => out.nontemporal = Some(value),
                _ => {}
            }
        }
        out
    }
}

fn int_value(key: &str, attr: Option<&Attr>) -> Result<Option<i64>, CompileError> {
    match attr {
        None => Ok(None),
        Some(Attr::Int(v)) => Ok(Some(*v)),
        Some(other) => Err(malformed(format!("attribute {key} is {other:?}, not an integer"))),
    }
}

impl Checker<'_> {
    fn ty(&self, v: ValueId) -> Result<&Type, CompileError> {
        self.kernel.values.get(v.0 as usize).map(|d| &d.ty).ok_or_else(|| malformed("an undefined value"))
    }

    fn int(&self, op: &Op, key: &str) -> Result<Option<i64>, CompileError> {
        match op.attr(key) {
            None => Ok(None),
            Some(Attr::Int(v)) => Ok(Some(*v)),
            Some(other) => Err(malformed(format!("attribute {key} is {other:?}, not an integer"))),
        }
    }

    fn str<'a>(&self, op: &'a Op, key: &str) -> Result<Option<&'a str>, CompileError> {
        match op.attr(key) {
            None => Ok(None),
            Some(Attr::Str(s)) => Ok(Some(s)),
            Some(other) => Err(malformed(format!("attribute {key} is {other:?}, not a string"))),
        }
    }

    /// The types of the values `op` defines; parameters have scalar or
    /// pointer types (the parser checks).
    fn defined_types(&self, op: &Op) -> Result<(), CompileError> {
        for &v in op.results.iter().chain(&op.block_args) {
            let ty = self.ty(v)?;
            if matches!(ty, Type::Vec(..) | Type::Smem(..)) {
                check_type(ty).map_err(|what| unsupported(format!("value {} has type {ty:?}: {what}", self.kernel.name_of(v))))?;
            }
        }
        Ok(())
    }

    fn op(&self, op: &Op) -> Result<(), CompileError> {
        use OpKind::*;
        self.defined_types(op)?;
        let (r, o) = (op.results.len(), op.operands.len());
        let arity = |results: usize, operands: usize| -> Result<(), CompileError> {
            if r != results || o != operands {
                return Err(malformed(format!(
                    "{r} result(s) and {o} operand(s); expected {results} and {operands}"
                )));
            }
            Ok(())
        };
        let mut inline = [&PLACEHOLDER; 4];
        let mut spilled = Vec::new();
        let types: &[&Type] = if o <= inline.len() {
            for (slot, &v) in inline.iter_mut().zip(op.operands.iter()) {
                *slot = self.ty(v)?;
            }
            &inline[..o]
        } else {
            for &v in op.operands.iter() {
                spilled.push(self.ty(v)?);
            }
            &spilled
        };
        let res = match op.results.first() {
            Some(&v) => Some(self.ty(v)?),
            None => None,
        };
        // fp8 and bf8 are i8 to the LLVM lowering.
        let int = |t: &Type| matches!(t, Type::Scalar(s) if int_bits(*s) > 0);
        let float = |t: &Type| t.elem().is_float() && matches!(t, Type::Scalar(_) | Type::Vec(..));
        let float_scalar = |t: &Type| matches!(t, Type::Scalar(s) if s.is_float());
        let index = |t: &Type| matches!(t, Type::Scalar(Scalar::I32 | Scalar::I64));
        let i32t = Type::Scalar(Scalar::I32);
        let f32t = Type::Scalar(Scalar::F32);
        let rsrc = Type::Vec(Scalar::I32, 4);
        let mismatch = || malformed(format!("operand types {types:?} and result type {res:?} do not fit"));
        // `n` operands, each of the result's type.
        let same = |n: usize| -> Result<&Type, CompileError> {
            arity(1, n)?;
            let t = res.expect("one result");
            if types.iter().any(|u| *u != t) {
                return Err(mismatch());
            }
            Ok(t)
        };
        // Instruction selection lowers these only for f32.
        let only_f32 = |t: &Type| -> Result<(), CompileError> {
            match t {
                _ if *t == f32t => Ok(()),
                _ if float(t) => Err(unsupported(format!("{:?} of {t:?}", op.kind))),
                _ => Err(mismatch()),
            }
        };
        let mut memory = MemoryAttrs::default();
        if matches!(
            op.kind,
            GlobalLoad | GlobalLoadVec | GlobalStore | GlobalStoreVec | SmemLoadVec | SmemStore | SmemStoreVec
        ) {
            memory = MemoryAttrs::of(op);
            if let Some(align) = int_value("align", memory.align)? {
                if align <= 0 || align & (align - 1) != 0 {
                    return Err(malformed(format!("align {align} is not a positive power of two")));
                }
            }
        }
        match op.kind {
            For => {
                let n = op.block_args.len().saturating_sub(1);
                if o != 3 + n || r != n || op.regions.len() != 1 {
                    return Err(malformed("scf.for operand, iteration argument and result counts disagree"));
                }
                if types[..3].iter().any(|t| **t != i32t) {
                    return Err(malformed("scf.for bounds and step must be i32"));
                }
                let yields = match op.regions[0].ops.last() {
                    Some(last) if last.kind == OpKind::Yield => &last.operands[..],
                    _ if n == 0 => &[][..],
                    _ => return Err(malformed(format!("scf.for with {n} iteration argument(s) has no scf.yield"))),
                };
                if yields.len() != n {
                    return Err(malformed(format!("scf.for with {n} iteration argument(s) yields {}", yields.len())));
                }
                for i in 0..n {
                    let arg = self.ty(op.block_args[i + 1])?;
                    if types[3 + i] != arg || self.ty(op.results[i])? != arg || self.ty(yields[i])? != arg {
                        return Err(malformed(format!("scf.for iteration argument {i} changes type")));
                    }
                }
                for key in ["unroll", "elide_trailing_barrier"] {
                    if !matches!(op.attr(key), None | Some(Attr::Bool(_))) {
                        return Err(malformed(format!("{key} is not a bool")));
                    }
                }
            }
            If | IfElse => {
                arity(0, 1)?;
                if *types[0] != Type::Scalar(Scalar::I1) {
                    return Err(malformed("scf.if condition must be i1"));
                }
            }
            Yield => {
                if r != 0 {
                    return Err(malformed("scf.yield has no results"));
                }
            }
            Return => arity(0, 0)?,
            // The fences the LLVM lowering knows.
            Sync | SyncLdsOnly | BarrierBare => {
                arity(0, 0)?;
                if !matches!(self.str(op, "fence")?, None | Some("compiler" | "workgroup")) {
                    return Err(invalid(format!("barrier fence {:?}", op.attr("fence"))));
                }
            }
            Constant | ConstantVec | SmemAlloc => arity(1, 0)?,
            Add | Sub | Mul | Div | Mod | And | Or | Xor | Shl | Lshr | Ashr | Smin | Smax | UmulHi | SmulHi => {
                arity(1, 2)?;
                let t = res.expect("one result");
                let logic = matches!(op.kind, And | Or | Xor) && *t == Type::Scalar(Scalar::I1);
                let fits = |u: &Type| u == t || (logic && *u == i32t);
                if !int(t) || !types.iter().all(|u| fits(u)) {
                    return Err(mismatch());
                }
            }
            Cmp => {
                arity(1, 2)?;
                if *res.expect("one result") != Type::Scalar(Scalar::I1) || types[0] != types[1] || !int(types[0]) {
                    return Err(mismatch());
                }
            }
            Fcmp => {
                arity(1, 2)?;
                if *res.expect("one result") != Type::Scalar(Scalar::I1) || types[0] != types[1] || !float_scalar(types[0]) {
                    return Err(mismatch());
                }
                // Selection compares 32-bit words as f32.
                only_f32(types[0])?;
            }
            Select => {
                arity(1, 3)?;
                let t = res.expect("one result");
                if *types[0] != Type::Scalar(Scalar::I1) || types[1] != t || types[2] != t || matches!(t, Type::Smem(..)) {
                    return Err(mismatch());
                }
            }
            Zext | Sext | Trunc => {
                arity(1, 1)?;
                let (from, to) = (types[0], res.expect("one result"));
                let bits = |t: &Type| match t {
                    Type::Scalar(s) => int_bits(*s),
                    _ => 0,
                };
                // As LLVM's casts: the width changes.
                let ok = if op.kind == Trunc {
                    bits(to) > 0 && bits(from) > bits(to)
                } else {
                    bits(from) > 0 && bits(from) < bits(to)
                };
                if !ok {
                    return Err(mismatch());
                }
            }
            Bitcast | VecBitcast => {
                arity(1, 1)?;
                let from = bit_size(types[0]);
                if from.is_none() || from != bit_size(res.expect("one result")) {
                    return Err(mismatch());
                }
            }
            Fadd | Fsub | Fmul | Fmax | Fma => {
                if !float(same(if op.kind == Fma { 3 } else { 2 })?) {
                    return Err(mismatch());
                }
            }
            // Selection reads the operand as an f32 word.
            Exp2 | Log2 | Rcp | Exp2Fast | RcpFast | Rsqrt | Tanh | Rint => only_f32(same(1)?)?,
            CastToF32 => {
                arity(1, 1)?;
                if !matches!(types[0], Type::Scalar(Scalar::F16 | Scalar::BF16)) || *res.expect("one result") != f32t {
                    return Err(mismatch());
                }
            }
            CastF32To => {
                arity(1, 1)?;
                if *types[0] != f32t || !matches!(res.expect("one result"), Type::Scalar(Scalar::F16 | Scalar::BF16)) {
                    return Err(mismatch());
                }
            }
            Extract | Insert => {
                arity(1, if op.kind == Extract { 1 } else { 2 })?;
                let Type::Vec(elem, n) = *types[0] else { return Err(mismatch()) };
                let t = res.expect("one result");
                let fits = if op.kind == Extract { *t == Type::Scalar(elem) } else { t == types[0] && *types[1] == Type::Scalar(elem) };
                if !fits {
                    return Err(mismatch());
                }
                match self.int(op, "index")? {
                    Some(i) if (0..n as i64).contains(&i) => {}
                    Some(i) => return Err(malformed(format!("index {i} is outside a {n}-element vector"))),
                    None => return Err(malformed("no index attribute")),
                }
            }
            Pack => {
                if r != 1 || o == 0 {
                    return Err(malformed("vector.pack needs one result and an operand per element"));
                }
                let Type::Vec(elem, n) = *res.expect("one result") else { return Err(mismatch()) };
                if n as usize != o || types.iter().any(|u| **u != Type::Scalar(elem)) {
                    return Err(mismatch());
                }
            }
            Concat => {
                arity(1, 2)?;
                match (types[0], types[1], res.expect("one result")) {
                    (Type::Vec(a, m), Type::Vec(b, n), Type::Vec(c, k)) if a == b && b == c && m.checked_add(*n) == Some(*k) => {}
                    _ => return Err(mismatch()),
                }
            }
            TruncF32To => {
                arity(1, 1)?;
                match (types[0], res.expect("one result")) {
                    (Type::Vec(Scalar::F32, m), Type::Vec(Scalar::F16 | Scalar::BF16, n)) if m == n => {}
                    _ => return Err(mismatch()),
                }
            }
            ThreadId | BlockId => {
                arity(1, 0)?;
                if *res.expect("one result") != i32t {
                    return Err(mismatch());
                }
                if self.str(op, "axis")?.is_some_and(|a| !matches!(a, "x" | "y" | "z")) {
                    return Err(malformed("axis must be x, y or z"));
                }
            }
            GlobalPtrAdd => {
                arity(1, 2)?;
                if !matches!(types[0], Type::Ptr(_)) || res != Some(types[0]) || !index(types[1]) {
                    return Err(mismatch());
                }
            }
            GlobalLoad | GlobalLoadVec => {
                arity(1, 2)?;
                let t = res.expect("one result");
                if !matches!(types[0], Type::Ptr(_)) || !index(types[1]) || !matches!(t, Type::Scalar(_) | Type::Vec(..)) {
                    return Err(mismatch());
                }
                if op.kind == GlobalLoadVec {
                    vec_attr(memory.vec, t, true)?;
                    nontemporal(memory.nontemporal)?;
                }
            }
            GlobalStore | GlobalStoreVec | GlobalAtomicAddF32 => {
                arity(0, 3)?;
                let value = types[2];
                if !matches!(types[0], Type::Ptr(_)) || !index(types[1]) || !matches!(value, Type::Scalar(_) | Type::Vec(..)) {
                    return Err(mismatch());
                }
                if op.kind == GlobalAtomicAddF32 && *value != Type::Scalar(Scalar::F32) {
                    return Err(mismatch());
                }
                if op.kind == GlobalStoreVec {
                    vec_attr(memory.vec, value, true)?;
                    nontemporal(memory.nontemporal)?;
                }
            }
            SmemLoadVec | SmemStore | SmemStoreVec => {
                let loads = op.kind == SmemLoadVec;
                if r != loads as usize || o < 2 + !loads as usize {
                    return Err(malformed("LDS access with the wrong operand or result count"));
                }
                let Type::Smem(_, shape) = types[0] else { return Err(mismatch()) };
                let indices = &types[1..if loads { o } else { o - 1 }];
                if indices.len() != shape.len() || indices.iter().any(|t| **t != i32t) {
                    return Err(malformed(format!("{} indices of an LDS allocation of rank {}", indices.len(), shape.len())));
                }
                let value = if loads { res.expect("one result") } else { types[o - 1] };
                if !matches!(value, Type::Scalar(_) | Type::Vec(..)) {
                    return Err(mismatch());
                }
                if op.kind != SmemStore {
                    vec_attr(memory.vec, value, false)?;
                }
            }
            BufferRsrc => {
                arity(1, 2)?;
                if !matches!(types[0], Type::Ptr(_)) || !index(types[1]) || *res.expect("one result") != Type::Vec(Scalar::I32, 4) {
                    return Err(mismatch());
                }
            }
            BufferLoadVec | BufferStoreVecF16 => {
                let loads = op.kind == BufferLoadVec;
                arity(loads as usize, if loads { 3 } else { 4 })?;
                if *types[0] != Type::Vec(Scalar::I32, 4) || *types[1] != i32t || *types[2] != i32t {
                    return Err(mismatch());
                }
                let data = if loads { res.expect("one result") } else { types[3] };
                let dwords = match data {
                    Type::Vec(Scalar::F16 | Scalar::BF16, _) => data.dwords(),
                    Type::Vec(_, _) if loads => data.dwords(),
                    _ => return Err(mismatch()),
                };
                let declared = self.int(op, "dwords")?;
                if !matches!(dwords, 1 | 2 | 4) || declared.is_some_and(|d| d != dwords as i64) || (loads && declared.is_none() && dwords != 4) {
                    return Err(malformed(format!("a buffer access of {dwords} dwords declared as {declared:?}")));
                }
            }
            DsSwizzleXor => {
                arity(1, 1)?;
                if *types[0] != i32t || *res.expect("one result") != i32t {
                    return Err(mismatch());
                }
                match self.int(op, "xor_mask")? {
                    Some(mask) if (0..32).contains(&mask) => {}
                    other => return Err(malformed(format!("xor_mask {other:?} is not in 0..32"))),
                }
            }
            Mma => {
                arity(1, 3)?;
                // C and D as the gfx11 WMMA intrinsics take them; A and B in
                // their register type or, as the LLVM lowering reinterprets
                // them, the same bits of the instruction's element type
                // (`vec<i8x16>` for iu8; i4 has no values).
                let (f16, bf16, f32x8, i32x8) = (Type::Vec(Scalar::F16, 16), Type::Vec(Scalar::BF16, 16), Type::Vec(Scalar::F32, 8), Type::Vec(Scalar::I32, 8));
                let (i32x4, i8x16, i32x2) = (Type::Vec(Scalar::I32, 4), Type::Vec(Scalar::I8, 16), Type::Vec(Scalar::I32, 2));
                let shape: Option<(&[&Type], &Type)> = match self.str(op, "op_id")? {
                    Some("wmma_f32_16x16x16_f16") => Some((&[&f16], &f32x8)),
                    Some("wmma_f32_16x16x16_bf16") => Some((&[&bf16], &f32x8)),
                    Some("wmma_i32_16x16x16_iu8") => Some((&[&i32x4, &i8x16], &i32x8)),
                    Some("wmma_i32_16x16x16_iu4") => Some((&[&i32x2], &i32x8)),
                    _ => None,
                };
                if let Some((ab, cd)) = shape {
                    if !ab.contains(&types[0]) || !ab.contains(&types[1]) || types[2] != cd || res != Some(cd) {
                        return Err(mismatch());
                    }
                }
            }
            ReadFirstLane => {
                arity(1, 1)?;
                if res != Some(types[0]) {
                    return Err(mismatch());
                }
                if !matches!(types[0], Type::Scalar(Scalar::I32 | Scalar::I64)) {
                    return Err(invalid(format!("the LLVM lowering reads the first lane of an i32 or i64, not {:?}", types[0])));
                }
            }
            InlineAsmIdentity => {
                arity(1, 1)?;
                if res != Some(types[0]) || !matches!(types[0], Type::Scalar(s) if *s != Scalar::I1) && !matches!(types[0], Type::Ptr(_)) {
                    return Err(mismatch());
                }
            }
            KernArg | GLoad | GStore | GAtomicAddF32 | GAtomicAdd | LLoad | LStore | LAtomicAdd => {
                return Err(malformed("an internal op in the input"));
            }
            Fneg | Fdiv => {
                if !float(same(if op.kind == Fneg { 1 } else { 2 })?) {
                    return Err(mismatch());
                }
            }
            // The LLVM lowering takes these for scalars only.
            Fabs | Fmin | Fmax3 | Fmin3 => {
                let operands = match op.kind {
                    Fabs => 1,
                    Fmin => 2,
                    _ => 3,
                };
                if !float_scalar(same(operands)?) {
                    return Err(mismatch());
                }
            }
            Fdot2 => {
                arity(1, 3)?;
                let pair = matches!(types[0], Type::Vec(Scalar::F16 | Scalar::BF16, 2));
                if !pair || types[1] != types[0] || *types[2] != f32t || *res.expect("one result") != f32t {
                    return Err(mismatch());
                }
            }
            SitofpF32 => {
                arity(1, 1)?;
                if *types[0] != i32t || *res.expect("one result") != f32t {
                    return Err(mismatch());
                }
            }
            Not => {
                if !int(same(1)?) {
                    return Err(mismatch());
                }
            }
            GlobalPtrToI64 => {
                arity(1, 1)?;
                if !matches!(types[0], Type::Ptr(_)) || *res.expect("one result") != Type::Scalar(Scalar::I64) {
                    return Err(mismatch());
                }
            }
            // Both operands of the result's vector type, as LLVM's vector
            // arithmetic takes them.
            VecAdd | VecSub | VecMul | VecFma => {
                let ok = match same(if op.kind == VecFma { 3 } else { 2 })? {
                    Type::Vec(elem, _) => op.kind != VecFma || matches!(elem, Scalar::F16 | Scalar::BF16 | Scalar::F32),
                    _ => false,
                };
                if !ok {
                    return Err(mismatch());
                }
            }
            VecSplat => {
                arity(1, 1)?;
                match res.expect("one result") {
                    Type::Vec(elem, _) if *types[0] == Type::Scalar(*elem) => {}
                    _ => return Err(mismatch()),
                }
            }
            VecUndef => {
                arity(1, 0)?;
                if !matches!(res.expect("one result"), Type::Vec(..)) {
                    return Err(mismatch());
                }
            }
            VecSelect => {
                arity(1, 3)?;
                let t = res.expect("one result");
                let Type::Vec(_, n) = *t else { return Err(mismatch()) };
                let mask = *types[0] == Type::Scalar(Scalar::I1) || *types[0] == Type::Vec(Scalar::I1, n);
                if !mask || types[1] != t || types[2] != t {
                    return Err(mismatch());
                }
            }
            VecSum => {
                arity(1, 1)?;
                match types[0] {
                    Type::Vec(elem, _) if elem.is_float() && *res.expect("one result") == Type::Scalar(*elem) => {}
                    _ => return Err(mismatch()),
                }
            }
            // i32 in and out, as the LLVM intrinsics take them.
            PermB32 | Permlanex16 | Permlane16 | Permlane64 | DsBpermute | WaveAny | WaveAll => {
                let operands = match op.kind {
                    PermB32 => 3,
                    Permlane16 => 4,
                    DsBpermute => 2,
                    _ => 1,
                };
                arity(1, operands)?;
                if types.iter().any(|t| **t != i32t) || *res.expect("one result") != i32t {
                    return Err(mismatch());
                }
                // The lowering reads these as truth values, selection as bools.
                for key in ["fi", "bound_ctrl"] {
                    if !matches!(op.attr(key), None | Some(Attr::Bool(_))) {
                        return Err(malformed(format!("{key} is not a bool")));
                    }
                }
            }
            LaneId => {
                arity(1, 0)?;
                if *res.expect("one result") != i32t {
                    return Err(mismatch());
                }
            }
            GlobalAtomicAdd => {
                arity(1, 3)?;
                let value = types[2];
                if !matches!(types[0], Type::Ptr(_)) || !index(types[1]) || !matches!(value, Type::Scalar(_)) || res != Some(value) {
                    return Err(mismatch());
                }
                self.str(op, "ordering")?;
            }
            // The LLVM lowering requires both, and these values.
            Fence => {
                arity(0, 0)?;
                if !matches!(self.str(op, "scope")?, Some("workgroup" | "agent" | "system")) {
                    return Err(invalid(format!("memref.fence scope {:?}", op.attr("scope"))));
                }
                if !matches!(self.str(op, "ordering")?, Some("acquire" | "release" | "acq_rel" | "seq_cst")) {
                    return Err(invalid(format!("memref.fence ordering {:?}", op.attr("ordering"))));
                }
            }
            TruncF32ToBf16 | TruncF32ToF16 | CvtF32ToI8Sat => {
                arity(1, 1)?;
                let to = match op.kind {
                    TruncF32ToBf16 => Scalar::BF16,
                    TruncF32ToF16 => Scalar::F16,
                    _ => Scalar::I8,
                };
                if *types[0] != f32t || *res.expect("one result") != Type::Scalar(to) {
                    return Err(mismatch());
                }
            }
            // `tile.buffer_load` (typed by `elem_type`) and its `_f16`,
            // `_f16_d16` and `_bf16` forms; `tile.buffer_store_f16`, `_f32`
            // and `_bf16`. `uniformity` checks the resource and soffset.
            BufferLoadScalar | BufferStoreScalar => {
                let loads = op.kind == BufferLoadScalar;
                arity(loads as usize, if loads { 3 } else { 4 })?;
                if *types[0] != rsrc || *types[1] != i32t || *types[2] != i32t {
                    return Err(mismatch());
                }
                let data = if loads { res.expect("one result") } else { types[3] };
                let elem = match *data {
                    Type::Scalar(e @ (Scalar::F16 | Scalar::BF16 | Scalar::F32)) => e,
                    Type::Scalar(Scalar::I32) if loads => Scalar::I32,
                    _ => return Err(mismatch()),
                };
                let named = match self.str(op, "elem_type")? {
                    None => None,
                    Some("f16") => Some(Scalar::F16),
                    Some("bf16") => Some(Scalar::BF16),
                    Some("f32") => Some(Scalar::F32),
                    Some("i32") => Some(Scalar::I32),
                    Some(other) => return Err(invalid(format!("buffer element type {other:?}"))),
                };
                if named.is_some_and(|n| n != elem) {
                    return Err(mismatch());
                }
            }
            SetPrio => {
                arity(0, 0)?;
                match self.int(op, "level")? {
                    Some(level) if (i16::MIN as i64..=u16::MAX as i64).contains(&level) => {}
                    other => return Err(malformed(format!("level {other:?} is not an i16"))),
                }
            }
            // A negative count waits for nothing and a large one clamps, as
            // the lowering encodes them.
            WaitCnt => {
                arity(0, 0)?;
                for key in ["vmcnt", "lgkmcnt", "expcnt"] {
                    self.int(op, key)?;
                }
            }
            // i32 arguments of the LLVM intrinsics; a group barrier needs its
            // mask and count.
            SchedBarrier | SchedGroupBarrier => {
                arity(0, 0)?;
                for key in ["mask", "count", "group"] {
                    match self.int(op, key)? {
                        Some(v) if (i32::MIN as i64..=u32::MAX as i64).contains(&v) => {}
                        Some(v) => return Err(malformed(format!("{key} {v} is not an i32"))),
                        None if op.kind == SchedGroupBarrier && key != "group" => {
                            return Err(malformed(format!("no {key} attribute")));
                        }
                        None => {}
                    }
                }
            }
            // The lowering indexes a rank-2 allocation and reads four halves.
            SmemLoadV4 => {
                arity(1, 3)?;
                let Type::Smem(_, shape) = types[0] else { return Err(mismatch()) };
                if shape.len() != 2 || *types[1] != i32t || *types[2] != i32t || *res.expect("one result") != Type::Vec(Scalar::F16, 4) {
                    return Err(mismatch());
                }
            }
            LdsAtomicAdd => {
                let Some(Type::Smem(_, shape)) = types.first().copied() else { return Err(mismatch()) };
                arity(1, 2 + shape.len())?;
                let value = types[o - 1];
                if types[1..o - 1].iter().any(|t| **t != i32t) || res != Some(value) {
                    return Err(mismatch());
                }
                if !matches!(value, Type::Scalar(Scalar::I32 | Scalar::F32)) {
                    return Err(unsupported(format!("an LDS atomic add of {value:?}")));
                }
                match self.str(op, "ordering")? {
                    None | Some("monotonic") => {}
                    Some("acquire" | "release" | "acq_rel" | "seq_cst") => {
                        return Err(unsupported("an LDS atomic add ordered other than monotonic"));
                    }
                    Some(other) => return Err(invalid(format!("atomic ordering {other:?}"))),
                }
            }
            // DPP moves of i32 words (`mov_dpp8` also f32), with the control
            // words the lowering accepts.
            DppXor | QuadPerm | MovDpp8 => {
                arity(1, 1)?;
                let t = types[0];
                if !(*t == i32t || (op.kind == MovDpp8 && *t == f32t)) || res != Some(t) {
                    return Err(mismatch());
                }
                let (key, range) = match op.kind {
                    DppXor => ("xor_mask", 1..=15),
                    QuadPerm => ("ctrl", 0..=255),
                    _ => ("sel", i64::MIN..=i64::MAX),
                };
                match self.int(op, key)? {
                    Some(v) if range.contains(&v) => {}
                    other => return Err(invalid(format!("{key} {other:?} is outside {range:?}"))),
                }
            }
        }
        Ok(())
    }

}

/// Bits of `s` as the LLVM lowering's integer type (fp8 and bf8 are `i8`),
/// 0 for other scalars.
fn int_bits(s: Scalar) -> u32 {
    match s {
        Scalar::I1 => 1,
        Scalar::I8 | Scalar::FP8E4M3 | Scalar::BF8E5M2 => 8,
        Scalar::I16 => 16,
        Scalar::I32 => 32,
        Scalar::I64 => 64,
        _ => 0,
    }
}

/// Bits of a scalar or vector of other than `i1`: what a bitcast keeps.
fn bit_size(ty: &Type) -> Option<u32> {
    match ty {
        Type::Scalar(s) if *s != Scalar::I1 => Some(s.bytes() * 8),
        Type::Vec(e, n) if *e != Scalar::I1 => e.bytes().checked_mul(8 * n),
        _ => None,
    }
}

/// `vec`, required or not, must be the number of elements of `ty`.
fn vec_attr(vec: Option<&Attr>, ty: &Type, required: bool) -> Result<(), CompileError> {
    let n = match ty {
        Type::Vec(_, n) => *n as i64,
        _ => 1,
    };
    match int_value("vec", vec)? {
        Some(v) if v == n => Ok(()),
        None if !required => Ok(()),
        other => Err(malformed(format!("vec {other:?} for a value of type {ty:?}"))),
    }
}

/// The LLVM lowering marks a `nontemporal` access only for the streaming
/// targets gfx942 and gfx950.
fn nontemporal(attr: Option<&Attr>) -> Result<(), CompileError> {
    match attr {
        None | Some(Attr::Bool(false)) => Ok(()),
        Some(Attr::Bool(true)) => Err(invalid("a nontemporal (temporal_hint STREAMING) access needs gfx942 or gfx950")),
        Some(other) => Err(malformed(format!("nontemporal {other:?} is not a bool"))),
    }
}

#[cfg(test)]
mod tests {
    use super::super::{analysis, compile, opt, parse};

    fn kernel_with(attrs: &str, params: &str, body: &str) -> String {
        format!(
            "ckdsl.ir v1\nkernel @probe {{\n  attrs {{ {attrs} }}\n  params {{\n    %Y : ptr<f32,global>\n{params}  }}\n  \
             region @entry {{\n    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n{body}    cf.return (  )\n  }}\n}}\n"
        )
    }

    fn kernel(body: &str) -> String {
        kernel_with("max_workgroup_size = i:64", "    %n : i32\n", body)
    }

    fn rejection(text: &str) -> (&'static str, String) {
        match compile(text) {
            Ok(_) => panic!("accepted:\n{text}"),
            Err(e) => (e.kind, e.message),
        }
    }

    /// Each of these used to panic, overflow the stack or compile to code
    /// that does not do what the IR says.
    #[test]
    fn malformed_ops_are_rejected() {
        let store = "    memref.global_store_typed ( %Y, %tid, %e ) { align = i:4, elem_type = s:\"f32\" }\n";
        let cases = [
            ("    %v : vec<f32x4> = arith.constant_vec (  ) { fill = f:2.0 }\n    %e : f32 = vector.extract ( %v ) { index = i:7 }\n", "MALFORMED"),
            ("    %v : vec<f32x4> = arith.constant_vec (  ) { fill = f:2.0 }\n    %e : f32 = vector.extract ( %v ) { index = i:300 }\n", "MALFORMED"),
            ("    %v : vec<f32x4> = arith.constant_vec (  ) { fill = f:2.0 }\n    %e : f32 = vector.extract ( %v )\n", "MALFORMED"),
            ("    %v : vec<f32x512> = arith.constant_vec (  ) { fill = f:2.0 }\n    %e : f32 = vector.extract ( %v ) { index = i:511 }\n", "UNSUPPORTED"),
            ("    %v : vec<f32x0> = arith.constant_vec (  ) { fill = f:2.0 }\n    %e : f32 = vector.extract ( %v ) { index = i:0 }\n", "UNSUPPORTED"),
            ("    %c : f32 = arith.constant (  ) { ity = s:\"f32\", value = f:1.0 }\n    %v : vec<f32x4> = arith.constant_vec (  ) { fill = f:2.0 }\n    \
              %w : vec<f32x4> = vector.insert ( %v, %c ) { index = i:4 }\n    %e : f32 = vector.extract ( %w ) { index = i:0 }\n", "MALFORMED"),
            ("    arith.add ( %tid, %n )\n    %e : f32 = arith.bitcast ( %n )\n", "MALFORMED"),
            ("    arith.constant (  ) { ity = s:\"i32\", value = i:3 }\n    %e : f32 = arith.bitcast ( %n )\n", "MALFORMED"),
            ("    %k : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:5 }\n    %f : f32 = arith.constant (  ) { ity = s:\"f32\", value = f:0.0 }\n    \
              %a : i32 = arith.add ( %k, %f )\n    %e : f32 = arith.bitcast ( %a )\n", "MALFORMED"),
            ("    %w : i64 = arith.zext ( %tid )\n    %a : i32 = arith.add ( %w, %n )\n    %e : f32 = arith.bitcast ( %a )\n", "MALFORMED"),
            ("    %w : i64 = arith.zext ( %tid )\n    %e : f32 = arith.bitcast ( %w )\n", "MALFORMED"),
            ("    %c : i1 = arith.cmp ( %tid, %n ) { pred = s:\"lt\" }\n    %a : vec<f32x4> = arith.constant_vec (  ) { fill = f:1.0 }\n    \
              %b : vec<f32x8> = arith.constant_vec (  ) { fill = f:1.0 }\n    %s : vec<f32x4> = arith.select ( %c, %a, %b )\n    \
              %e : f32 = vector.extract ( %s ) { index = i:3 }\n", "MALFORMED"),
            ("    %a : f32 = arith.bitcast ( %tid )\n    %p : vec<f32x4> = vector.pack ( %a, %a ) { elem = s:\"f32\", vec = i:4 }\n    \
              %e : f32 = vector.extract ( %p ) { index = i:3 }\n", "MALFORMED"),
            ("    %a : vec<f16x16> = arith.constant_vec (  ) { fill = f:1.0 }\n    %c : vec<f32x4> = arith.constant_vec (  ) { fill = f:0.0 }\n    \
              %d : vec<f32x4> = tile.mma ( %a, %a, %c ) { op_id = s:\"wmma_f32_16x16x16_f16\" }\n    %e : f32 = vector.extract ( %d ) { index = i:0 }\n", "MALFORMED"),
            ("    %s : i32 = tile.ds_swizzle_xor ( %tid ) { xor_mask = i:40 }\n    %e : f32 = arith.bitcast ( %s )\n", "MALFORMED"),
            ("    %f : f32 = arith.bitcast ( %tid )\n    %q : ptr<f32,global> = tile.global_ptr_add ( %Y, %f )\n    %e : f32 = arith.bitcast ( %tid )\n", "MALFORMED"),
            ("    %c0 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }\n    %c1 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:1 }\n    \
              %r : i32 = scf.for ( %c0, %n, %c1, %c0 ) { iter_args = l:[ { name = s:\"%x\", type = s:\"i32\" } ], iv = s:\"%i\", iv_type = s:\"i32\", num_iter_args = i:1 }\n      \
              region @body {\n        %y : i32 = arith.add ( %x, %i )\n      }\n    %e : f32 = arith.bitcast ( %r )\n", "MALFORMED"),
            ("    %c : i1 = arith.cmp ( %tid, %n ) { pred = s:\"lt\" }\n    scf.if ( %c )\n      region @then {\n        scf.yield (  )\n      }\n    \
              %e : f32 = arith.bitcast ( %tid )\n", "MALFORMED"),
            ("    %x : vec<i32x4> = tile.buffer_rsrc ( %Y, %n )\n    %l : vec<f16x8> = tile.buffer_load_vN ( %x, %tid, %n ) { dwords = i:2 }\n    \
              %e : f32 = arith.bitcast ( %tid )\n", "MALFORMED"),
        ];
        for (body, kind) in cases {
            let text = kernel(&format!("{body}{store}"));
            let (got, message) = rejection(&text);
            assert_eq!(got, format!("ROCKE_DIRECT_{kind}"), "{message}\n{body}");
        }
    }

    /// The op kinds stage3 added: operand counts, types and attributes their
    /// lowerings rely on, and what the LLVM lowering refuses.
    #[test]
    fn malformed_new_ops_are_rejected() {
        let store = "    memref.global_store_typed ( %Y, %tid, %e ) { align = i:4, elem_type = s:\"f32\" }\n";
        let half = "    %x : f32 = arith.bitcast ( %tid )\n    %h : f16 = arith.cast_f32_to ( %x ) { target = s:\"f16\" }\n";
        let cases: Vec<(String, &str)> = vec![
            // A scalar operand of vector arithmetic.
            ("    %v : vec<f32x4> = arith.constant_vec (  ) { fill = f:1.0 }\n    %s : f32 = arith.bitcast ( %tid )\n    \
              %a : vec<f32x4> = vector.add ( %v, %s )\n    %e : f32 = vector.extract ( %a ) { index = i:0 }\n".into(), "MALFORMED"),
            ("    %a : vec<f16x4> = arith.constant_vec (  ) { fill = f:1.0 }\n    %z : f32 = arith.bitcast ( %tid )\n    \
              %e : f32 = arith.fdot2 ( %a, %a, %z )\n".into(), "MALFORMED"),
            ("    %p : i32 = tile.permlane16 ( %tid, %tid, %n, %n ) { fi = i:1 }\n    %e : f32 = arith.bitcast ( %p )\n".into(), "MALFORMED"),
            ("    %f : f32 = arith.bitcast ( %tid )\n    %p : f32 = tile.ds_bpermute ( %tid, %f )\n    %e : f32 = arith.bitcast ( %p )\n".into(), "MALFORMED"),
            ("    memref.fence (  ) { ordering = s:\"acquire\" }\n    %e : f32 = arith.bitcast ( %tid )\n".into(), "INVALID"),
            ("    memref.fence (  ) { scope = s:\"device\", ordering = s:\"acquire\" }\n    %e : f32 = arith.bitcast ( %tid )\n".into(), "INVALID"),
            ("    %f : f32 = arith.bitcast ( %tid )\n    %e : f32 = tile.readfirstlane ( %f )\n".into(), "INVALID"),
            (format!("{half}    %c : i1 = arith.fcmp ( %h, %h ) {{ pred = s:\"olt\" }}\n    %s : i32 = arith.zext ( %c )\n    %e : f32 = arith.bitcast ( %s )\n"), "UNSUPPORTED"),
            (format!("{half}    %r : f16 = math.rsqrt ( %h )\n    %e : f32 = arith.cast_to_f32 ( %r )\n"), "UNSUPPORTED"),
            (format!("{half}    %r : f16 = math.rcp ( %h )\n    %e : f32 = arith.cast_to_f32 ( %r )\n"), "UNSUPPORTED"),
            ("    %z : i32 = arith.zext ( %tid )\n    %e : f32 = arith.bitcast ( %z )\n".into(), "MALFORMED"),
            ("    %x : f32 = arith.bitcast ( %tid )\n    memref.global_atomic_add ( %Y, %tid, %x )\n    %e : f32 = arith.bitcast ( %tid )\n".into(), "MALFORMED"),
            ("    %r : vec<i32x4> = tile.buffer_rsrc ( %Y, %n )\n    tile.buffer_store_f32 ( %r, %tid, %n, %tid )\n    %e : f32 = arith.bitcast ( %tid )\n".into(), "MALFORMED"),
            ("    %v : vec<f32x4> = arith.constant_vec (  ) { fill = f:1.0 }\n    %b : vec<f16x4> = vector.bitcast ( %v )\n    \
              %g : f16 = vector.extract ( %b ) { index = i:0 }\n    %e : f32 = arith.cast_to_f32 ( %g )\n".into(), "MALFORMED"),
            ("    %a : vec<f16x16> = arith.constant_vec (  ) { fill = f:1.0 }\n    %c : vec<f32x8> = arith.constant_vec (  ) { fill = f:0.0 }\n    \
              %d : vec<f32x8> = tile.mma ( %a, %a, %c ) { op_id = s:\"wmma_f32_16x16x16_bf16\" }\n    %e : f32 = vector.extract ( %d ) { index = i:0 }\n".into(), "MALFORMED"),
            ("    %v : vec<f32x4> = arith.constant_vec (  ) { fill = f:1.0 }\n    %a : vec<f32x4> = arith.fabs ( %v )\n    \
              %e : f32 = vector.extract ( %a ) { index = i:0 }\n".into(), "MALFORMED"),
        ];
        for (body, kind) in cases {
            let text = kernel(&format!("{body}{store}"));
            let (got, message) = rejection(&text);
            assert_eq!(got, format!("ROCKE_DIRECT_{kind}"), "{message}\n{body}");
        }
    }

    /// Well-formed uses of the new op kinds compile.
    #[test]
    fn new_ops_validate() {
        let body = "    %x : f32 = arith.bitcast ( %tid )\n    %h : f16 = arith.cast_f32_to ( %x ) { target = s:\"f16\" }\n    \
                    %p : vec<f16x2> = vector.pack ( %h, %h ) { elem = s:\"f16\", vec = i:2 }\n    %d : f32 = arith.fdot2 ( %p, %p, %x )\n    \
                    %m : f32 = arith.fmin ( %d, %x )\n    %r : f32 = math.rsqrt ( %m )\n    %g : f32 = arith.fneg ( %r )\n    \
                    %s : vec<f32x4> = vector.splat ( %g )\n    %a : vec<f32x4> = vector.add ( %s, %s )\n    %e0 : f32 = vector.extract ( %a ) { index = i:1 }\n    \
                    %l : i32 = tile.lane_id (  )\n    %w : i32 = tile.wave_any ( %l )\n    %q : i32 = tile.permlanex16 ( %w )\n    \
                    %u : i32 = tile.readfirstlane ( %q )\n    %uf : f32 = arith.bitcast ( %u )\n    %e : f32 = arith.fadd ( %e0, %uf )\n    \
                    memref.fence (  ) { scope = s:\"agent\", ordering = s:\"acq_rel\" }\n    \
                    %old : f32 = memref.global_atomic_add ( %Y, %tid, %e )\n    memref.global_store_typed ( %Y, %n, %old ) { align = i:4 }\n";
        if let Err(e) = compile(&kernel(body)) {
            panic!("{}: {}", e.kind, e.message);
        }
    }

    /// Inputs the LLVM lowering or COMGR refuses for gfx1151.
    #[test]
    fn limits_of_the_reference_are_enforced() {
        let use_lds = "    %s : smem<f32, [262144]> = tile.smem_alloc (  )\n    %f : f32 = arith.bitcast ( %tid )\n    \
                       tile.smem_store ( %s, %tid, %f ) { align = i:4 }\n    tile.sync (  )\n    \
                       %g : vec<f32x1> = tile.smem_load_vN ( %s, %tid ) { vec = i:1 }\n    %e : f32 = vector.extract ( %g ) { index = i:0 }\n    \
                       memref.global_store_typed ( %Y, %tid, %e ) { align = i:4 }\n";
        assert_eq!(rejection(&kernel(use_lds)).0, "ROCKE_DIRECT_INVALID");
        let plain = "    %e : f32 = arith.bitcast ( %tid )\n    memref.global_store_typed ( %Y, %tid, %e ) { align = i:4 }\n";
        for size in ["0", "1025", "100000", "-1", "4294967360"] {
            let text = kernel_with(&format!("max_workgroup_size = i:{size}"), "", plain);
            assert_ne!(compile(&text).err().map(|e| e.kind), None, "max_workgroup_size {size}");
        }
        assert!(compile(&kernel_with("max_workgroup_size = i:1024", "", plain)).is_ok());
        let streaming = "    %v : vec<f32x4> = arith.constant_vec (  ) { fill = f:1.0 }\n    \
                         memref.global_store_vN ( %Y, %tid, %v ) { align = i:16, nontemporal = b:true, vec = i:4 }\n";
        assert_eq!(rejection(&kernel(streaming)).0, "ROCKE_DIRECT_INVALID");
        // An LDS allocation nothing uses takes no space, as in the LLVM lowering.
        let unused = "    %s : smem<f32, [262144]> = tile.smem_alloc (  )\n";
        assert!(compile(&kernel(&format!("{unused}{plain}"))).is_ok());
    }

    /// A return under a divergent condition, and a divergent buffer
    /// `soffset` (once read from the first lane only).
    #[test]
    fn divergence_the_hardware_cannot_take_is_rejected() {
        let body = |cond: &str| {
            format!(
                "    %c : i1 = arith.cmp ( {cond}, %n ) {{ pred = s:\"lt\" }}\n    scf.if ( %c )\n      region @then {{\n        cf.return (  )\n      }}\n    \
                 %e : f32 = arith.bitcast ( %tid )\n    memref.global_store_typed ( %Y, %tid, %e ) {{ align = i:4 }}\n"
            )
        };
        assert_eq!(rejection(&kernel(&body("%tid"))).0, "ROCKE_DIRECT_UNSUPPORTED");
        let uniform = body("%k").replace("    %c :", "    %k : i32 = gpu.block_id (  ) { axis = s:\"x\" }\n    %c :");
        assert!(compile(&kernel(&uniform)).is_ok());
        let buffer = |soffset: &str| {
            format!(
                "    %x : vec<i32x4> = tile.buffer_rsrc ( %Y, %n )\n    %l : vec<f32x4> = tile.buffer_load_vN ( %x, %tid, {soffset} ) {{ dwords = i:4 }}\n    \
                 %e : f32 = vector.extract ( %l ) {{ index = i:0 }}\n    memref.global_store_typed ( %Y, %tid, %e ) {{ align = i:4 }}\n"
            )
        };
        assert_eq!(rejection(&kernel(&buffer("%tid"))).0, "ROCKE_DIRECT_UNSUPPORTED");
        assert!(compile(&kernel(&buffer("%n"))).is_ok());
    }

    /// `depth` nested `scf.if` regions, or `scf.for` loops when `loops`.
    fn nested(depth: usize, loops: bool) -> String {
        let mut text = String::from(
            "ckdsl.ir v1\nkernel @deep {\n  attrs { max_workgroup_size = i:64 }\n  params {\n    %Y : ptr<f32,global>\n    %n : i32\n  }\n  \
             region @entry {\n    %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n    %c : i1 = arith.cmp ( %tid, %n ) { pred = s:\"lt\" }\n    \
             %c0 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }\n    %c1 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:1 }\n",
        );
        for i in 0..depth {
            if loops {
                text += &format!("scf.for ( %c0, %n, %c1 ) {{ iv = s:\"%i{i}\", iv_type = s:\"i32\" }}\nregion @body {{\n");
            } else {
                text += "scf.if ( %c )\nregion @then {\n";
            }
        }
        text += "%f : f32 = arith.bitcast ( %tid )\nmemref.global_store_typed ( %Y, %tid, %f ) { align = i:4 }\n";
        for _ in 0..depth {
            text += if loops { "scf.yield (  )\n}\n" } else { "}\n" };
        }
        text + "cf.return (  )\n}\n}\n"
    }

    /// The middle end and the analyses run without recursion: 100,000
    /// levels of nesting on a 1 MiB thread stack. The whole compile refuses
    /// such a kernel cleanly, at the nesting limit.
    #[test]
    fn deep_nesting_on_a_small_stack() {
        let run = || {
            for loops in [false, true] {
                let text = nested(100_000, loops);
                let mut kernel = parse::parse_with_threads(&text, 1).expect("parses");
                let (mut lds, _) = analysis::lds_layout(&kernel);
                let facts = opt::optimize(&mut kernel, &mut lds).expect("optimizes");
                let uniform = analysis::classify(&kernel);
                let canonical = analysis::canonical_f32(&kernel, &facts);
                assert_eq!(uniform.len(), canonical.len());
                drop(kernel);
                let error = compile(&text).err().expect("refused");
                assert_eq!(error.kind, "ROCKE_DIRECT_UNSUPPORTED", "{}", error.message);
                // The deepest accepted nesting goes through the whole compile.
                let deepest = compile(&nested(super::MAX_NESTING - 1, loops));
                assert!(deepest.as_ref().map_or_else(|e| e.kind == "ROCKE_DIRECT_REGISTERS", |_| true));
                assert!(compile(&nested(super::MAX_NESTING, loops)).is_err_and(|e| e.kind == "ROCKE_DIRECT_UNSUPPORTED"));
            }
        };
        std::thread::Builder::new().stack_size(1 << 20).spawn(run).expect("spawns").join().expect("no overflow");
    }

    /// Attribute maps and lists parse recursively, to a bounded depth.
    #[test]
    fn deep_attributes_on_a_small_stack() {
        let with = |depth: usize, list: bool| {
            let (open, close) = if list { ("l:[ ", " ]") } else { ("{ a = ", " }") };
            let attr = open.repeat(depth) + "i:1" + &close.repeat(depth);
            kernel(&format!(
                "    %c : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:7, x = {attr} }}\n    %e : f32 = arith.bitcast ( %c )\n    \
                 memref.global_store_typed ( %Y, %tid, %e ) {{ align = i:4 }}\n"
            ))
        };
        let run = move || {
            for list in [false, true] {
                assert!(compile(&with(1000, list)).is_ok());
                assert_eq!(compile(&with(100_000, list)).err().map(|e| e.kind), Some("ROCKE_DIRECT_PARSE"));
            }
        };
        std::thread::Builder::new().stack_size(1 << 20).spawn(run).expect("spawns").join().expect("no overflow");
    }
}
