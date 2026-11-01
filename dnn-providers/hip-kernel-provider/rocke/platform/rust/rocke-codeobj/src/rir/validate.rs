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
use crate::target::TargetInfo;

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
/// allocates, and the target refuses above its workgroup limit (gfx1151:
/// 64 KiB).
pub fn lds_pool(bytes: u32, t: &TargetInfo) -> Result<(), CompileError> {
    let most = t.lds_bytes_per_group;
    if bytes > most {
        return Err(invalid(format!("the kernel needs {bytes} bytes of LDS; a workgroup has at most {most}")));
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
            OpKind::BufferLoadLds if !(op.operands.first().is_some_and(uni) && op.operands.get(3).is_some_and(uni)) => {
                return Err(unsupported(format!("{:?} with a divergent resource or soffset", op.kind)));
            }
            OpKind::BufferLoadCp if !(op.operands.first().is_some_and(uni) && op.operands.get(2).is_some_and(uni)) => {
                return Err(unsupported(format!("{:?} with a divergent resource or soffset", op.kind)));
            }
            OpKind::BufferStoreCp if !(op.operands.get(1).is_some_and(uni) && op.operands.get(3).is_some_and(uni)) => {
                return Err(unsupported(format!("{:?} with a divergent resource or soffset", op.kind)));
            }
            // Resources at operands 1, 3, ..; the tag and the L2-touch / atomic bases are SGPR operands.
            OpKind::TaggedPoll if !op.operands.iter().skip(1).step_by(2).all(uni) => {
                return Err(unsupported("tile.tagged_poll with a divergent resource"));
            }
            OpKind::L2TouchLoad | OpKind::AtomicAddScoped if !op.operands.last().is_some_and(uni) => {
                return Err(unsupported(format!("{:?} with a divergent base address", op.kind)));
            }
            // The saddr base of a global access with cache bits, and s_ff1's source, are SGPR operands.
            OpKind::GlobalStoreCp if op.operands.len() == 3 && !op.operands.last().is_some_and(uni) => {
                return Err(unsupported("tile.global_store_cp with a divergent base address"));
            }
            OpKind::GlobalLoadCp if op.operands.len() == 2 && !op.operands.last().is_some_and(uni) => {
                return Err(unsupported("tile.global_load_cp with a divergent base address"));
            }
            OpKind::SFf1 if !op.operands.first().is_some_and(uni) => {
                return Err(unsupported("tile.s_ff1 of a divergent value"));
            }
            OpKind::InlineAsmIdentity if op.asm_pin() == Some('s') && !op.operands.first().is_some_and(uni) => {
                return Err(unsupported("an \"=s,0\" pin of a divergent value"));
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
            Exp2 | Log2 | Rcp | Exp2Fast | RcpFast | Rsqrt | Tanh | Rint | Exp => only_f32(same(1)?)?,
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
                // `tile.buffer_store_vN_f32` stores its dwords as the 16-bit forms do.
                let dwords = match data {
                    Type::Vec(Scalar::F16 | Scalar::BF16 | Scalar::F32, _) => data.dwords(),
                    Type::Vec(_, _) if loads => data.dwords(),
                    _ => return Err(mismatch()),
                };
                let declared = self.int(op, "dwords")?;
                if !matches!(dwords, 1 | 2 | 4) || declared.is_some_and(|d| d != dwords as i64) || (loads && declared.is_none() && dwords != 4) {
                    return Err(malformed(format!("a buffer access of {dwords} dwords declared as {declared:?}")));
                }
            }
            // LDS addresses are i64, as the LLVM lowering's `ptrtoint` makes them.
            SmemAddrOf => {
                arity(1, 1)?;
                if !matches!(types[0], Type::Smem(..)) || *res.expect("one result") != Type::Scalar(Scalar::I64) {
                    return Err(mismatch());
                }
            }
            SmemPtrAdd => {
                let i64t = Type::Scalar(Scalar::I64);
                if *same(2)? != i64t {
                    return Err(mismatch());
                }
            }
            // `raw.ptr.buffer.load.lds` of 4 or 16 bytes per lane; `aux` is its
            // cache policy (sc0 1, nt 2, sc1 16). `uniformity` checks the
            // resource and soffset.
            BufferLoadLds => {
                arity(0, 4)?;
                if *types[0] != rsrc || *types[1] != Type::Scalar(Scalar::I64) || *types[2] != i32t || *types[3] != i32t {
                    return Err(mismatch());
                }
                match self.int(op, "dwords")? {
                    Some(1 | 4) => {}
                    other => return Err(unsupported(format!("an LDS DMA load of {other:?} dwords per lane"))),
                }
                match self.int(op, "aux")? {
                    None => {}
                    Some(aux) if aux & !(1 | 2 | 16) == 0 => {}
                    Some(aux) => return Err(unsupported(format!("LDS DMA cache policy {aux:#x}"))),
                }
            }
            // `global.load.lds` from `ptr + byte_offset` (an i8 GEP).
            GlobalLoadLds => {
                arity(0, 3)?;
                if !matches!(types[0], Type::Ptr(_)) || *types[1] != i32t || *types[2] != Type::Scalar(Scalar::I64) {
                    return Err(mismatch());
                }
                match self.int(op, "size_bytes")? {
                    Some(4 | 16) => {}
                    other => return Err(unsupported(format!("an LDS DMA load of {other:?} bytes per lane"))),
                }
                match self.int(op, "aux")? {
                    None => {}
                    Some(aux) if aux & !(1 | 2 | 16) == 0 => {}
                    Some(aux) => return Err(unsupported(format!("LDS DMA cache policy {aux:#x}"))),
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
                // gfx950's block-scaled atoms also take the A and B scales.
                let scaled = self.str(op, "op_id")?.is_some_and(crate::target::gfx9::mfma::is_scaled);
                arity(1, if scaled { 5 } else { 3 })?;
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
            // An LDS read written as inline asm: an i32 address, an integer result of its size.
            InlineAsmIdentity if crate::target::gfx9::lds_dma::asm_lds_read(op).is_some() => {
                arity(1, 1)?;
                let dwords = crate::target::gfx9::lds_dma::asm_lds_read(op).map_or(0, |(d, _)| d as u32);
                let t = res.expect("one result");
                if *types[0] != i32t || !matches!(t, Type::Scalar(Scalar::I32) | Type::Vec(Scalar::I32, _)) || t.dwords() != dwords {
                    return Err(mismatch());
                }
            }
            // A general template (gfx950 assembles it, `rir::isel_asm`; gfx1151
            // lowers only identity pins): one result per output constraint and
            // one operand per input constraint; what the template says is
            // isel's to accept or refuse.
            InlineAsmIdentity if op.asm_pin().is_none() => {
                let text = op.str_attr("constraints").unwrap_or("");
                let parts: Vec<&str> = text.split(',').map(str::trim).filter(|p| !p.is_empty() && !p.starts_with("~{")).collect();
                let outputs = parts.iter().filter(|p| p.starts_with('=')).count();
                if outputs != op.results.len() || parts.len() - outputs != op.operands.len() {
                    return Err(mismatch());
                }
            }
            InlineAsmIdentity => {
                arity(1, 1)?;
                if res != Some(types[0]) || !matches!(types[0], Type::Scalar(s) if *s != Scalar::I1) && !matches!(types[0], Type::Ptr(_)) {
                    return Err(mismatch());
                }
                // An SGPR pin holds one dword (`device_ops.fresh_s`).
                if op.asm_pin() == Some('s') && !matches!(types[0], Type::Scalar(Scalar::I32 | Scalar::F32)) {
                    return Err(unsupported(format!("an \"=s,0\" pin of {:?}", types[0])));
                }
            }
            KernArg | GLoad | GStore | GAtomicAddF32 | GAtomicAdd | LLoad | LStore | LAtomicAdd | LdsReadTr => {
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
            VecAnd => {
                if !matches!(same(2)?, Type::Vec(Scalar::I32, _)) {
                    return Err(mismatch());
                }
            }
            // An i32 flag of a global array; the reference lowering requires
            // the scope, and the store's ordering, from these values.
            FlagWaitEq | FlagStore => {
                arity(0, 3)?;
                let i32t = Type::Scalar(Scalar::I32);
                if !matches!(types[0], Type::Ptr(Scalar::I32)) || *types[1] != i32t || *types[2] != i32t {
                    return Err(mismatch());
                }
                if !matches!(self.str(op, "scope")?, Some("workgroup" | "agent" | "system")) {
                    return Err(invalid(format!("{:?} scope {:?}", op.kind, op.attr("scope"))));
                }
                if op.kind == FlagStore && !matches!(self.str(op, "ordering")?, Some("monotonic" | "release" | "seq_cst")) {
                    return Err(invalid(format!("memref.global_flag_store ordering {:?}", op.attr("ordering"))));
                }
            }
            WaitAsyncCnt => {
                arity(0, 0)?;
                match self.int(op, "n")? {
                    None => {}
                    Some(n) if (0..=u16::MAX as i64).contains(&n) => {}
                    Some(n) => return Err(malformed(format!("n {n} is not a u16"))),
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
                let packed = *value == Type::Vec(Scalar::F16, 2);
                if !matches!(types[0], Type::Ptr(_)) || !index(types[1]) || !(matches!(value, Type::Scalar(_)) || packed) || res != Some(value) {
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
            // LLVM defines strategies 0 to 3 and has no meaning for others.
            IglpOpt => {
                arity(0, 0)?;
                match self.int(op, "level")? {
                    Some(0..=3) => {}
                    other => return Err(malformed(format!("level {other:?} names no strategy"))),
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
            Permlane32Swap => {
                arity(2, 2)?;
                let results = op.results.iter().map(|&v| self.ty(v)).collect::<Result<Vec<_>, _>>()?;
                if *types[0] != i32t || *types[1] != i32t || results.iter().any(|t| **t != i32t) {
                    return Err(mismatch());
                }
            }
            // `llvm.amdgcn.readlane` of an i32 or f32.
            ReadLane => {
                arity(1, 2)?;
                if !matches!(types[0], Type::Scalar(Scalar::I32 | Scalar::F32)) || *types[1] != i32t || res != Some(types[0]) {
                    return Err(mismatch());
                }
            }
            CvtFp8ToF32 | CvtBf8ToF32 | CvtPkF32Fp8 | CvtPkF32Bf8 | CvtF32ToFp8 | CvtF32ToBf8 | CvtPkFp8F32 | CvtPkBf8F32 => {
                arity(1, 1)?;
                let e = if matches!(op.kind, CvtFp8ToF32 | CvtPkF32Fp8 | CvtF32ToFp8 | CvtPkFp8F32) { Scalar::FP8E4M3 } else { Scalar::BF8E5M2 };
                let (from, to) = match op.kind {
                    CvtFp8ToF32 | CvtBf8ToF32 => (Type::Scalar(e), f32t.clone()),
                    CvtPkF32Fp8 | CvtPkF32Bf8 => (Type::Vec(e, 4), Type::Vec(Scalar::F32, 4)),
                    CvtF32ToFp8 | CvtF32ToBf8 => (f32t.clone(), Type::Scalar(e)),
                    _ => (Type::Vec(Scalar::F32, 4), Type::Vec(e, 4)),
                };
                if *types[0] != from || *res.expect("one result") != to {
                    return Err(mismatch());
                }
            }
            // The LLVM lowering indexes the allocation (a GEP of its element
            // type) and reads 64 bits of 16- or 8-bit elements there.
            DsReadTr16 | DsReadTr8 => {
                let Some(Type::Smem(elem, shape)) = types.first().copied() else { return Err(mismatch()) };
                arity(1, 1 + shape.len())?;
                let bytes = if op.kind == DsReadTr16 { 2 } else { 1 };
                let lanes_ok = match res.expect("one result") {
                    Type::Vec(e, 4) if op.kind == DsReadTr16 => e.bytes() == 2,
                    Type::Vec(e, 8) if op.kind == DsReadTr8 => e.bytes() == 1 && *e != Scalar::I1,
                    _ => false,
                };
                if types[1..].iter().any(|t| **t != i32t) || elem.bytes() != bytes || !lanes_ok {
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
            IntToPtr => {
                arity(1, 1)?;
                if *types[0] != Type::Scalar(Scalar::I64) || !matches!(res.expect("one result"), Type::Ptr(_)) {
                    return Err(mismatch());
                }
            }
            // `aux`: sc0 1, nt 2, sc1 16 (`rocke.core.ir.CPOL_*`); 1, 2 or 4 dwords of i32.
            BufferLoadCp | BufferStoreCp => {
                let loads = op.kind == BufferLoadCp;
                arity(loads as usize, if loads { 3 } else { 4 })?;
                let at = if loads { 0 } else { 1 };
                if *types[at] != rsrc || *types[at + 1] != i32t || *types[at + 2] != i32t {
                    return Err(mismatch());
                }
                let data = if loads { res.expect("one result") } else { types[0] };
                let dwords = match data {
                    Type::Scalar(Scalar::I32) => 1,
                    Type::Vec(Scalar::I32, n @ (2 | 4)) => *n,
                    _ => return Err(mismatch()),
                };
                if self.int(op, "dwords")? != Some(dwords as i64) {
                    return Err(malformed(format!("a {dwords}-dword cache-policy buffer access declared {:?}", op.attr("dwords"))));
                }
                cache_policy(self.int(op, "aux")?)?;
            }
            TaggedPoll => {
                let Some(Attr::List(widths)) = op.attr("dwords") else {
                    return Err(malformed("tile.tagged_poll without a dwords list"));
                };
                let n = widths.len();
                if n == 0 || r != n || o != 1 + 2 * n || *types[0] != i32t {
                    return Err(malformed("tile.tagged_poll needs a tag and a (resource, offset) and result per spec"));
                }
                for (k, w) in widths.iter().enumerate() {
                    let Attr::Int(w @ (2 | 4)) = *w else {
                        return Err(malformed(format!("tile.tagged_poll spec {k} of {w:?} dwords")));
                    };
                    if *types[1 + 2 * k] != rsrc || *types[2 + 2 * k] != i32t || *self.ty(op.results[k])? != Type::Vec(Scalar::I32, w as u32) {
                        return Err(mismatch());
                    }
                }
                cache_policy(self.int(op, "aux")?)?;
                if !matches!(self.str(op, "pause")?, Some("" | "s_nop 0" | "s_sleep 1")) {
                    return Err(unsupported(format!("tile.tagged_poll pause {:?}", op.attr("pause"))));
                }
            }
            CvtScalePkBf16Fp4 => {
                arity(1, 2)?;
                if *types[0] != i32t || *types[1] != f32t || *res.expect("one result") != Type::Vec(Scalar::BF16, 2) {
                    return Err(mismatch());
                }
                if !matches!(self.int(op, "byte")?, Some(0..=3)) {
                    return Err(malformed(format!("byte {:?} is not 0..3", op.attr("byte"))));
                }
            }
            Permlane16Swap => {
                arity(2, 2)?;
                let results = op.results.iter().map(|&v| self.ty(v)).collect::<Result<Vec<_>, _>>()?;
                if *types[0] != i32t || *types[1] != i32t || results.iter().any(|t| **t != i32t) {
                    return Err(mismatch());
                }
            }
            SGetreg => {
                arity(1, 0)?;
                if *res.expect("one result") != i32t || !matches!(self.str(op, "hwreg")?, Some("XCC_ID" | "HW_ID")) {
                    return Err(malformed(format!("tile.s_getreg of {:?}", op.attr("hwreg"))));
                }
            }
            SSleep => {
                arity(0, 0)?;
                if !matches!(self.int(op, "units")?, Some(1..=0xffff)) {
                    return Err(malformed(format!("tile.s_sleep units {:?}", op.attr("units"))));
                }
            }
            MemRealtime => {
                arity(1, 0)?;
                if *res.expect("one result") != Type::Scalar(Scalar::I64) {
                    return Err(mismatch());
                }
            }
            AtomicAddScoped => {
                let returns = match op.attr("returns") {
                    Some(Attr::Bool(b)) => *b,
                    _ => return Err(malformed("tile.global_atomic_add_scoped without returns")),
                };
                arity(returns as usize, 3)?;
                let value = types[1];
                if *types[0] != i32t || !matches!(value, Type::Scalar(Scalar::I32 | Scalar::I64)) || *types[2] != Type::Scalar(Scalar::I64) {
                    return Err(mismatch());
                }
                if returns && res != Some(value) {
                    return Err(mismatch());
                }
                if !matches!(self.str(op, "scope")?, Some("workgroup" | "agent" | "system")) {
                    return Err(invalid(format!("atomic scope {:?}", op.attr("scope"))));
                }
            }
            L2TouchLoad => {
                arity(1, 2)?;
                if *types[0] != i32t || *types[1] != Type::Scalar(Scalar::I64) || *res.expect("one result") != Type::Vec(Scalar::I32, 4) {
                    return Err(mismatch());
                }
            }
            L2TouchDrain => {
                if r != 0 || o == 0 || types.iter().any(|t| **t != Type::Vec(Scalar::I32, 4)) {
                    return Err(mismatch());
                }
            }
            CvtScalePkFp8F32 | CvtScalePkBf8F32 => {
                arity(1, 4)?;
                if *types[0] != i32t || types[1..].iter().any(|t| **t != f32t) || *res.expect("one result") != i32t {
                    return Err(mismatch());
                }
            }
            MfmaScaleMx => {
                arity(1, 5)?;
                let wide = match self.str(op, "shape")? {
                    Some("16x16x128") => false,
                    Some("32x32x64") => true,
                    other => return Err(invalid(format!("tile.mfma_scale_mx shape {other:?}"))),
                };
                let dwords = |f: Option<i64>| match f {
                    Some(0 | 1) => Some(8),
                    Some(2 | 3) => Some(6),
                    Some(4) => Some(4),
                    _ => None,
                };
                let (na, nb) = match (dwords(self.int(op, "fmt_a")?), dwords(self.int(op, "fmt_b")?)) {
                    (Some(a), Some(b)) => (a, b),
                    _ => return Err(invalid(format!("tile.mfma_scale_mx formats {:?}, {:?}", op.attr("fmt_a"), op.attr("fmt_b")))),
                };
                for key in ["opsel_a", "opsel_b"] {
                    if !matches!(self.int(op, key)?, None | Some(0..=3)) {
                        return Err(invalid(format!("tile.mfma_scale_mx {key} {:?}", op.attr(key))));
                    }
                }
                let cd = Type::Vec(Scalar::F32, if wide { 16 } else { 4 });
                if *types[0] != Type::Vec(Scalar::I32, na)
                    || *types[1] != Type::Vec(Scalar::I32, nb)
                    || *types[2] != cd
                    || *types[3] != i32t
                    || *types[4] != i32t
                    || res != Some(&cd)
                {
                    return Err(mismatch());
                }
            }
            Maximum3 => {
                only_f32(same(3)?)?;
                if !matches!(self.int(op, "abs")?, None | Some(0..=7)) {
                    return Err(invalid(format!("arith.maximum3_f32 abs {:?}", op.attr("abs"))));
                }
            }
            MovDpp => {
                arity(1, 1)?;
                if *types[0] != i32t || *res.expect("one result") != i32t {
                    return Err(mismatch());
                }
                let shift = match (self.int(op, "row_shl")?, self.int(op, "row_shr")?) {
                    (Some(k), None) | (None, Some(k)) => k,
                    _ => return Err(invalid("tile.mov_dpp needs one of row_shl, row_shr")),
                };
                if !(1..=15).contains(&shift) {
                    return Err(invalid(format!("tile.mov_dpp shift {shift}")));
                }
            }
            CvtF32ToF64 => {
                arity(1, 1)?;
                if *types[0] != f32t || *res.expect("one result") != Type::Scalar(Scalar::F64) {
                    return Err(mismatch());
                }
            }
            CvtF64ToF32 => {
                arity(1, 1)?;
                if *types[0] != Type::Scalar(Scalar::F64) || *res.expect("one result") != f32t {
                    return Err(mismatch());
                }
            }
            Sqrt => {
                let t = same(1)?;
                if *t != Type::Scalar(Scalar::F64) {
                    return Err(if float(t) { unsupported(format!("math.sqrt of {t:?}")) } else { mismatch() });
                }
            }
            GlobalStoreCp => {
                if r != 0 || !(2..=3).contains(&o) {
                    return Err(mismatch());
                }
                let data_ok = matches!(types[0], Type::Scalar(Scalar::I32)) || matches!(types[0], Type::Vec(Scalar::I32, 2 | 4));
                let addr_ok = if o == 3 { *types[1] == i32t && *types[2] == Type::Scalar(Scalar::I64) } else { *types[1] == Type::Scalar(Scalar::I64) };
                if !data_ok || !addr_ok {
                    return Err(mismatch());
                }
                if !matches!(self.int(op, "aux")?, None | Some(0..=19)) || !matches!(self.int(op, "offset")?, None | Some(-4096..=4095)) {
                    return Err(invalid(format!("tile.global_store_cp aux {:?} offset {:?}", op.attr("aux"), op.attr("offset"))));
                }
            }
            GlobalLoadCp => {
                if r != 1 || !(1..=2).contains(&o) {
                    return Err(mismatch());
                }
                let addr_ok = if o == 2 { *types[0] == i32t && *types[1] == Type::Scalar(Scalar::I64) } else { *types[0] == Type::Scalar(Scalar::I64) };
                let t = res.expect("one result");
                let words = match t {
                    Type::Scalar(Scalar::I32 | Scalar::F32) => 1,
                    Type::Scalar(Scalar::I64) | Type::Vec(Scalar::I32, 2) => 2,
                    Type::Vec(Scalar::I32, 4) => 4,
                    _ => 0,
                };
                if !addr_ok || words == 0 || self.int(op, "dwords")?.is_some_and(|d| d != words) {
                    return Err(mismatch());
                }
                if !matches!(self.int(op, "aux")?, None | Some(0..=19)) || !matches!(self.int(op, "offset")?, None | Some(-4096..=4095)) {
                    return Err(invalid(format!("tile.global_load_cp aux {:?} offset {:?}", op.attr("aux"), op.attr("offset"))));
                }
            }
            Bcnt => {
                arity(1, 1)?;
                if *types[0] != i32t || *res.expect("one result") != i32t {
                    return Err(mismatch());
                }
            }
            SFf1 => {
                arity(1, 1)?;
                if !matches!(types[0], Type::Scalar(Scalar::I32 | Scalar::I64)) || *res.expect("one result") != i32t {
                    return Err(mismatch());
                }
            }
            MemTime => {
                arity(1, 0)?;
                if *res.expect("one result") != Type::Scalar(Scalar::I64) {
                    return Err(mismatch());
                }
            }
            SNop => {
                arity(0, 0)?;
                if !matches!(self.int(op, "n")?, Some(0..=15)) {
                    return Err(invalid(format!("tile.s_nop n {:?}", op.attr("n"))));
                }
            }
            CompilerFence => arity(0, 0)?,
            DsOrRaw | DsWriteRaw => {
                arity(0, 2)?;
                if *types[0] != i32t || *types[1] != i32t {
                    return Err(mismatch());
                }
            }
            DsReadRaw => {
                arity(1, 1)?;
                if *types[0] != i32t || *res.expect("one result") != i32t {
                    return Err(mismatch());
                }
            }
            DppOp => {
                let max = match self.str(op, "op")? {
                    Some("mov") => false,
                    Some("max_u32") => true,
                    other => return Err(invalid(format!("tile.dpp_op op {other:?}"))),
                };
                arity(1, if max { 3 } else { 2 })?;
                if types.iter().any(|t| **t != i32t) || *res.expect("one result") != i32t {
                    return Err(mismatch());
                }
                match self.int(op, "ctrl")? {
                    Some(c) if crate::target::gfx9::q38::dpp_ctrl_ok(c) => {}
                    other => return Err(invalid(format!("tile.dpp_op ctrl {other:?}"))),
                }
                for key in ["row_mask", "bank_mask"] {
                    if !matches!(self.int(op, key)?, None | Some(0..=15)) {
                        return Err(invalid(format!("tile.dpp_op {key} {:?}", op.attr(key))));
                    }
                }
            }
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

/// A cache policy of sc0 (1), nt (2) and sc1 (16), as `rocke.core.ir` checks it.
fn cache_policy(aux: Option<i64>) -> Result<(), CompileError> {
    match aux {
        Some(a) if a & !(1 | 2 | 16) == 0 => Ok(()),
        other => Err(malformed(format!("cache policy {other:?} beyond sc0, nt and sc1"))),
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

    /// Well-formed rocKE IR this compiler does not lower is refused as
    /// `ROCKE_DIRECT_UNSUPPORTED` (`compile_kernel` then compiles with COMGR),
    /// on gfx950 and gfx1151 alike: the ops, types, parameter and kernel
    /// attributes and inline asm the Qwen3.8 builds use. Text that is not
    /// rocKE IR stays a parse error.
    #[test]
    fn rocke_ir_the_direct_backend_does_not_lower_is_unsupported() {
        use crate::target::Target;
        let store = "    %e : f32 = arith.bitcast ( %tid )\n    memref.global_store_typed ( %Y, %tid, %e ) { align = i:4, elem_type = s:\"f32\" }\n";
        let body = |ops: &str| kernel(&format!("{ops}{store}"));
        let cases: Vec<(String, &str)> = vec![
            // Ops rocKE defines that the direct backend has no lowering for.
            (body("    %x : f32 = math.sinh ( %n )\n"), "UNSUPPORTED"),
            (body("    %v : vec<f32x4> = tile.mfma_unknown_mx ( %n, %n, %n ) { m = i:16 }\n"), "UNSUPPORTED"),
            (body("    %x : vec<i32x2> = arith.cvt_scalef32_pk_fp6_f32 ( %n, %n )\n"), "UNSUPPORTED"),
            // Not a rocKE IR dialect, or not an identifier: not rocKE IR.
            (body("    %x : f32 = foo.exp ( %n )\n"), "PARSE"),
            (body("    %x : f32 = math.e-xp ( %n )\n"), "PARSE"),
            // An SGPR pin of a lane's value.
            (kernel("    %a : i32 = tile.inline_asm ( %tid ) { constraints = s:\"=s,0\", template = s:\"; fresh $0\" }\n    \
                     %e : f32 = arith.bitcast ( %a )\n    memref.global_store_typed ( %Y, %tid, %e ) { align = i:4, elem_type = s:\"f32\" }\n"), "UNSUPPORTED"),
            // A pointer other than a global one, and an element type the backend has not.
            (body("    %w : i64 = arith.zext ( %tid )\n    %q : ptr<i32,constant> = tile.global_ptr_add ( %Y, %w )\n"), "UNSUPPORTED"),
            (kernel_with("max_workgroup_size = i:64", "    %s : ptr<fp4e2m1,global>\n", store), "UNSUPPORTED"),
            (kernel_with("max_workgroup_size = i:64", "    %s : ptr<vec<i32x8>,global>\n", store), "UNSUPPORTED"),
            (kernel_with("max_workgroup_size = i:64", "    %s : ptr<vec<i32x>,global>\n", store), "PARSE"),
            (kernel_with("max_workgroup_size = i:64", "    %s : ptr<i3-2,global>\n", store), "PARSE"),
            // An unknown kernel attribute, an AGPR range, an argument type.
            (kernel_with("max_workgroup_size = i:64, lds_budget = i:4096", "    %n : i32\n", store), "UNSUPPORTED"),
            (kernel_with("max_workgroup_size = i:64, agpr_alloc = l:[ i:0, i:64 ]", "    %n : i32\n", store), "UNSUPPORTED"),
            (kernel_with("max_workgroup_size = i:64", "    %n : f16\n", store), "UNSUPPORTED"),
            // A kernel symbol the ELF writer does not emit (a `-1` knob value in the name).
            (kernel(store).replace("kernel @probe {", "kernel @probe_base+1 {"), "UNSUPPORTED"),
            (kernel(store).replace("kernel @probe {", &format!("kernel @{} {{", "k".repeat(1025))), "UNSUPPORTED"),
            (kernel(store).replace("kernel @probe {", "kernel @ {"), "PARSE"),
        ];
        for (text, kind) in &cases {
            for target in [Target::Gfx950, Target::Gfx1151] {
                let error = super::super::compile_any_target(text, target).err().unwrap_or_else(|| panic!("accepted:\n{text}"));
                assert_eq!(error.kind, format!("ROCKE_DIRECT_{kind}"), "{target:?}: {}\n{text}", error.message);
            }
        }
        // General inline asm (no result, one, two; a tied pin with an instruction): gfx950 assembles the
        // template (`rir::isel_asm`), gfx1151 lowers only the identity pins.
        let used = |ops: &str| {
            kernel(&format!(
                "{ops}    %e : f32 = arith.bitcast ( %a )
    memref.global_store_typed ( %Y, %tid, %e ) {{ align = i:4, elem_type = s:\"f32\" }}
"
            ))
        };
        let asm: Vec<String> = vec![
            body("    tile.inline_asm ( %n ) { constraints = s:\"v\", sideeffect = b:true, template = s:\"s_sleep 1\" }
"),
            used("    %a : i32 = tile.inline_asm ( %n ) { constraints = s:\"=s,s\", template = s:\"s_mov_b32 $0, $1\" }
"),
            used("    %a : i32, %b : i32 = tile.inline_asm ( %n ) { constraints = s:\"=v,=v,v\", template = s:\"v_mov_b32 $0, $2\" }
"),
            used("    %a : i32 = tile.inline_asm ( %n ) { constraints = s:\"=v,0\", template = s:\"v_mov_b32 $0, $0 ; fresh\" }
"),
        ];
        for text in &asm {
            let error = super::super::compile_any_target(text, Target::Gfx1151).err().unwrap_or_else(|| panic!("gfx1151 accepted:
{text}"));
            assert_eq!(error.kind, "ROCKE_DIRECT_UNSUPPORTED", "{}
{text}", error.message);
            super::super::compile_any_target(text, Target::Gfx950).unwrap_or_else(|e| panic!("gfx950: {e}
{text}"));
        }
        // `inreg = b:false` states nothing and the empty pin still lowers.
        let pin = "    %a : i32 = tile.inline_asm ( %tid ) { constraints = s:\"=v,0\", convergent = b:false, sideeffect = b:false, template = s:\"\" }\n    \
                   %e : f32 = arith.bitcast ( %a )\n    memref.global_store_typed ( %Y, %tid, %e ) { align = i:4, elem_type = s:\"f32\" }\n";
        assert!(compile(&kernel_with("max_workgroup_size = i:64", "    %n : i32 { inreg = b:false }\n", pin)).is_ok());
        // COMGR's fresh-process policy carries nothing for this compiler.
        assert!(compile(&kernel_with("comgr_fresh_process = b:true, max_workgroup_size = i:64", "    %n : i32\n", pin)).is_ok());
    }

    /// The Qwen3.8 monokernel's persistent-kernel and MX ops lower on gfx950
    /// and are refused (UNSUPPORTED, COMGR then compiles) on gfx1151; so is
    /// kernarg preload. Malformed forms of them are MALFORMED on both. The
    /// comment-only `fresh` pins (`device_ops.fresh`, `"=v,0"` / `"=s,0"`)
    /// lower on both.
    #[test]
    fn qwen38_device_ops_lower_on_gfx950_only() {
        use crate::target::Target;
        let store = "    %e : f32 = arith.bitcast ( %r )\n    memref.global_store_typed ( %Y, %tid, %e ) { align = i:4, elem_type = s:\"f32\" }\n";
        let body = |ops: &str| kernel(&format!("{ops}{store}"));
        let rsrc = "    %c64 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:64 }\n    \
                    %rs : vec<i32x4> = tile.buffer_rsrc ( %Y, %c64 )\n    %c0 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }\n";
        let lowered: Vec<String> = vec![
            body(&format!("{rsrc}    %v : vec<i32x4> = tile.buffer_load_cp ( %rs, %tid, %c0 ) {{ aux = i:17, dwords = i:4 }}\n    \
                           %r : i32 = vector.extract ( %v ) {{ index = i:2 }}\n")),
            body(&format!("{rsrc}    tile.buffer_store_cp ( %tid, %rs, %tid, %c0 ) {{ aux = i:18, dwords = i:1 }}\n    %r : i32 = arith.add ( %tid, %n )\n")),
            body(&format!("{rsrc}    %p0 : vec<i32x2>, %p1 : vec<i32x4> = tile.tagged_poll ( %n, %rs, %tid, %rs, %c0 ) \
                           {{ aux = i:16, dwords = l:[ i:2, i:4 ], pause = s:\"s_sleep 1\" }}\n    %r : i32 = vector.extract ( %p1 ) {{ index = i:2 }}\n")),
            body("    %w : i64 = arith.zext ( %n )\n    %q : ptr<i32,global> = tile.int_to_ptr ( %w )\n    \
                  %r : i32 = memref.global_load_typed ( %q, %tid ) { align = i:4, elem_type = s:\"i32\" }\n"),
            body("    %h : f32 = arith.bitcast ( %n )\n    %d : vec<bf16x2> = arith.cvt_scalef32_pk_bf16_fp4 ( %tid, %h ) { byte = i:3 }\n    \
                  %r : i32 = vector.bitcast ( %d )\n"),
            body("    %r : i32, %r2 : i32 = tile.permlane16_swap ( %tid, %n )\n"),
            body("    %r : i32 = tile.s_getreg (  ) { hwreg = s:\"XCC_ID\" }\n    tile.s_sleep (  ) { units = i:200 }\n"),
            body("    %t : i64 = tile.memrealtime (  )\n    %r : i32 = arith.trunc ( %t )\n"),
            body("    %w : i64 = arith.zext ( %n )\n    %o : i64 = tile.global_atomic_add_scoped ( %tid, %w, %w ) { returns = b:true, scope = s:\"system\" }\n    \
                  %r : i32 = arith.trunc ( %o )\n"),
            body("    %w : i64 = arith.zext ( %n )\n    %v : vec<i32x4> = tile.l2_touch_load ( %tid, %w )\n    tile.l2_touch_drain ( %v )\n    \
                  %r : i32 = arith.add ( %tid, %n )\n"),
            kernel_with("max_workgroup_size = i:64", "    %n : i32 { inreg = b:true }\n", "    %r : i32 = arith.add ( %tid, %n )\n    \
                        %e : f32 = arith.bitcast ( %r )\n    memref.global_store_typed ( %Y, %tid, %e ) { align = i:4, elem_type = s:\"f32\" }\n"),
        ];
        for text in &lowered {
            if let Err(e) = super::super::compile_any_target(text, Target::Gfx950) {
                panic!("gfx950: {} {}\n{text}", e.kind, e.message);
            }
            let error = super::super::compile_any_target(text, Target::Gfx1151).err().unwrap_or_else(|| panic!("gfx1151 accepted:\n{text}"));
            assert_eq!(error.kind, "ROCKE_DIRECT_UNSUPPORTED", "gfx1151: {}\n{text}", error.message);
        }
        let malformed = [
            body(&format!("{rsrc}    %p : vec<i32x2> = tile.tagged_poll ( %n, %rs, %tid ) {{ aux = i:16, dwords = l:[ i:3 ], pause = s:\"s_nop 0\" }}\n    %r : i32 = arith.add ( %tid, %n )\n")),
            body(&format!("{rsrc}    %v : vec<i32x4> = tile.buffer_load_cp ( %rs, %tid, %c0 ) {{ aux = i:4, dwords = i:4 }}\n    %r : i32 = vector.extract ( %v ) {{ index = i:2 }}\n")),
            body("    %w : i64 = arith.zext ( %tid )\n    %r : i32 = tile.int_to_ptr ( %w )\n"),
            body("    %r : i32 = tile.s_getreg (  ) { hwreg = s:\"MODE\" }\n"),
        ];
        for text in &malformed {
            for target in [Target::Gfx950, Target::Gfx1151] {
                let error = super::super::compile_any_target(text, target).err().unwrap_or_else(|| panic!("accepted:\n{text}"));
                assert_eq!(error.kind, "ROCKE_DIRECT_MALFORMED", "{target:?}: {}\n{text}", error.message);
            }
        }
        // An SGPR pin of a lane's value is refused by the uniformity check (before selection).
        let divergent = kernel("    %r : i32 = tile.inline_asm ( %tid ) { constraints = s:\"=s,0\", template = s:\"; fresh $0\" }
                                    %e : f32 = arith.bitcast ( %r )
    memref.global_store_typed ( %Y, %tid, %e ) { align = i:4, elem_type = s:\"f32\" }
");
        let error = super::super::compile_any_target(&divergent, Target::Gfx950).err().expect("refused");
        assert!(error.message.contains("pin of a divergent value"), "{}", error.message);
        for (pin, arg) in [("=v,0", "%tid"), ("=s,0", "%n")] {
            let text = body(&format!("    %r : i32 = tile.inline_asm ( {arg} ) {{ constraints = s:\"{pin}\", convergent = b:false, sideeffect = b:true, template = s:\"; fresh $0\" }}\n"));
            for target in [Target::Gfx950, Target::Gfx1151] {
                if let Err(e) = super::super::compile_any_target(&text, target) {
                    panic!("{target:?} {pin}: {} {}", e.kind, e.message);
                }
            }
        }
    }
}
