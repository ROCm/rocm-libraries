// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! LDS layout (a port of the Python lowering's packer), uniformity, and
//! f32 canonicality. Every walk here nests through explicit stacks, so the
//! depth of region nesting is bounded by memory, not by the thread stack.

use super::ast::{Kernel, Op, OpKind, Region, Scalar, Type, ValueId};
use super::fx::{FxHashMap, FxHashSet};
use super::opt::{Facts, Konst};

/// Calls `f` on every op of `region` and of all regions nested in it, in
/// preorder (program order).
pub fn preorder<'a>(region: &'a Region, mut f: impl FnMut(&'a Op)) {
    let mut stack = vec![region.ops.iter()];
    while let Some(mut ops) = stack.pop() {
        while let Some(op) = ops.next() {
            f(op);
            if !op.regions.is_empty() {
                stack.push(ops);
                stack.extend(op.regions.iter().rev().map(|r| r.ops.iter()));
                break;
            }
        }
    }
}

/// For every `scf.for`, keyed by its preorder number: the preorder number
/// of the last op inside it.
fn loop_ends(region: &Region) -> FxHashMap<usize, usize> {
    enum Walk<'a> {
        Ops(std::slice::Iter<'a, Op>),
        Close(usize),
    }
    let mut ends = FxHashMap::default();
    let mut count = 0;
    let mut stack = vec![Walk::Ops(region.ops.iter())];
    while let Some(frame) = stack.pop() {
        let mut ops = match frame {
            Walk::Close(idx) => {
                ends.insert(idx, count - 1);
                continue;
            }
            Walk::Ops(ops) => ops,
        };
        while let Some(op) = ops.next() {
            count += 1;
            if !op.regions.is_empty() {
                stack.push(Walk::Ops(ops));
                if op.kind == OpKind::For {
                    stack.push(Walk::Close(count - 1));
                }
                stack.extend(op.regions.iter().rev().map(|r| Walk::Ops(r.ops.iter())));
                break;
            }
        }
    }
    ends
}

fn smem_liveness(
    body: &Region,
    allocs: &FxHashSet<ValueId>,
    intervals: &mut FxHashMap<ValueId, (usize, usize)>,
    used: &mut FxHashSet<ValueId>,
) {
    let ends = loop_ends(body);
    let mut counter = 0;
    // Each open region with the last preorder index of the loops around it.
    let mut stack: Vec<(std::slice::Iter<Op>, Option<usize>)> = vec![(body.ops.iter(), None)];
    while let Some((mut ops, loop_end)) = stack.pop() {
        while let Some(op) = ops.next() {
            let idx = counter;
            counter += 1;
            if op.kind == OpKind::SmemAlloc {
                if let Some(&result) = op.results.first() {
                    intervals.entry(result).or_insert((idx, idx));
                }
            }
            for operand in &op.operands {
                if allocs.contains(operand) {
                    used.insert(*operand);
                    let (first, last) = intervals.get(operand).copied().unwrap_or((idx, idx));
                    let new_last = loop_end.unwrap_or(idx);
                    intervals.insert(*operand, (first.min(idx), last.max(new_last)));
                }
            }
            if !op.regions.is_empty() {
                let inner_end = if op.kind == OpKind::For {
                    let own_last = ends.get(&idx).copied().unwrap_or(idx);
                    Some(loop_end.map_or(own_last, |end| own_last.max(end)))
                } else {
                    loop_end
                };
                stack.push((ops, loop_end));
                stack.extend(op.regions.iter().rev().map(|r| (r.ops.iter(), inner_end)));
                break;
            }
        }
    }
}

fn collect_allocs(region: &Region, out: &mut Vec<(ValueId, bool)>) {
    preorder(region, |op| {
        if op.kind == OpKind::SmemAlloc {
            if let Some(&result) = op.results.first() {
                out.push((result, op.bool_attr("exclusive") == Some(true)));
            }
        }
    });
}

/// Byte offset per allocation and the 16-byte-rounded pool size. Must run on
/// the unoptimized kernel so preorder numbering matches the Python lowering.
pub fn lds_layout(kernel: &Kernel) -> (FxHashMap<ValueId, u32>, u32) {
    let mut allocs = Vec::new();
    collect_allocs(&kernel.body, &mut allocs);
    let mut offsets = FxHashMap::default();
    if allocs.is_empty() {
        return (offsets, 0);
    }
    let alloc_set: FxHashSet<ValueId> = allocs.iter().map(|(v, _)| *v).collect();
    let mut intervals = FxHashMap::default();
    let mut used = FxHashSet::default();
    smem_liveness(&kernel.body, &alloc_set, &mut intervals, &mut used);
    let mut live: Vec<(ValueId, bool)> = Vec::new();
    for &(value, exclusive) in &allocs {
        if used.contains(&value) {
            live.push((value, exclusive));
        } else {
            offsets.insert(value, 0);
        }
    }
    live.sort_by_key(|(value, _)| intervals.get(value).map_or(0, |iv| iv.0));
    if live.is_empty() {
        return (offsets, 0);
    }
    const EXCL_LAST_SEQ: usize = 1 << 30;
    let mut slots: Vec<(usize, usize, usize)> = Vec::new();
    let no_reuse = super::debug_flag("noldsreuse");
    for (value, exclusive) in live {
        let exclusive = exclusive || no_reuse;
        let Type::Smem(elem, shape) = kernel.ty(value) else {
            continue;
        };
        let seg = elem.bytes() as usize * shape.iter().map(|&d| d as usize).product::<usize>();
        let align = 4usize;
        let (first, last) = intervals.get(&value).copied().unwrap_or((0, 0));
        let mut best: Option<(usize, usize)> = None;
        if !exclusive {
            for (i, &(s_off, _s_size, s_last)) in slots.iter().enumerate() {
                if s_last >= first {
                    continue;
                }
                let aligned = (s_off + align - 1) & !(align - 1);
                let end = aligned + seg;
                let conflict = slots
                    .iter()
                    .enumerate()
                    .any(|(j, &(o2, sz2, l2))| j != i && l2 >= first && aligned < o2 + sz2 && o2 < end);
                if conflict {
                    continue;
                }
                if best.is_none_or(|(_, b)| aligned < b) {
                    best = Some((i, aligned));
                }
            }
        }
        if let Some((i, aligned)) = best {
            let (s_off, s_size, _) = slots[i];
            offsets.insert(value, aligned as u32);
            slots[i] = (s_off, s_size.max(aligned - s_off + seg), last);
        } else {
            let end = slots.iter().map(|&(o, s, _)| o + s).max().unwrap_or(0);
            let aligned = (end + align - 1) & !(align - 1);
            offsets.insert(value, aligned as u32);
            slots.push((aligned, seg, if exclusive { EXCL_LAST_SEQ } else { last }));
        }
    }
    let pool = slots.iter().map(|&(o, s, _)| o + s).max().unwrap_or(0);
    (offsets, ((pool + 15) & !15) as u32)
}

/// Per-value uniformity: true means the value lives in SGPRs (or is an
/// immediate), false means VGPRs (or a lane mask for i1).
///
/// The greatest fixpoint: a loop-carried value is uniform when its initial
/// value and every yield are, assuming the loop-carried values are
/// (induction on the trip). Values in regions of ops other than `scf.for`
/// and `scf.if` are divergent.
pub fn classify(kernel: &Kernel) -> Vec<bool> {
    let mut uniform = vec![false; kernel.values.len()];
    for param in &kernel.params {
        uniform[param.value.0 as usize] = true;
    }
    let mut body_writes = FxHashMap::default();
    let u = |uniform: &Vec<bool>, v: &ValueId| uniform.get(v.0 as usize).copied().unwrap_or(false);
    // Program order, assuming every loop-carried value keeps the uniformity
    // of its initial value; `carried` lists each with its yield.
    let mut carried: Vec<(ValueId, Option<ValueId>)> = Vec::new();
    enum Exit<'a> {
        Root,
        Loop(&'a Op),
        Branch(&'a Op, usize),
    }
    struct Frame<'a> {
        ops: std::slice::Iter<'a, Op>,
        written: bool,
        exit: Exit<'a>,
    }
    let mut stack = vec![Frame { ops: kernel.body.ops.iter(), written: false, exit: Exit::Root }];
    while let Some(top) = stack.last_mut() {
        // Plain ops in a tight loop; control flow below.
        let mut control = None;
        for op in top.ops.by_ref() {
            if matches!(op.kind, OpKind::For | OpKind::If | OpKind::IfElse) {
                control = Some(op);
                break;
            }
            let value = match uniform_rule(kernel, op, top.written) {
                Rule::Always(value) => value,
                Rule::Inputs(inputs) => inputs.iter().all(|v| u(&uniform, v)),
            };
            for &result in &op.results {
                set(&mut uniform, result, value);
            }
            if op.kind.writes_global() {
                top.written = true;
            }
        }
        let Some(op) = control else {
            let frame = stack.pop().expect("a frame");
            match frame.exit {
                Exit::Root => {}
                Exit::Loop(op) => {
                    for (i, &result) in op.results.iter().enumerate() {
                        let arg = op.block_args.get(i + 1).is_some_and(|a| u(&uniform, a));
                        set(&mut uniform, result, arg);
                    }
                }
                Exit::Branch(op, k) => {
                    let parent = stack.last_mut().expect("a branch has a parent");
                    parent.written |= frame.written;
                    if let Some(next) = op.regions.get(k + 1) {
                        let written = parent.written;
                        stack.push(Frame { ops: next.ops.iter(), written, exit: Exit::Branch(op, k + 1) });
                    }
                }
            }
            continue;
        };
        let written = top.written;
        match op.kind {
            OpKind::For => {
                let Some(body) = op.regions.first() else { continue };
                let writes = loop_writes(op, &mut body_writes);
                let yields = loop_yields(op);
                // Lanes leave a loop with per-lane bounds after different
                // trips, so what it carries differs between lanes; the
                // induction variable steps in lockstep from a uniform start.
                let bounds = op.operands.get(..3).unwrap_or(&[]);
                let lanes_differ = bounds.len() < 3 || !bounds.iter().all(|v| u(&uniform, v));
                for (i, &arg) in op.block_args.iter().enumerate() {
                    let init = if i == 0 {
                        bounds.iter().step_by(2).all(|v| u(&uniform, v))
                    } else {
                        !lanes_differ && op.operands.get(2 + i).is_some_and(|v| u(&uniform, v))
                    };
                    set(&mut uniform, arg, init);
                    if i > 0 {
                        carried.push((arg, yields.get(i - 1).copied()));
                    }
                }
                top.written = written || writes;
                stack.push(Frame { ops: body.ops.iter(), written: written || writes, exit: Exit::Loop(op) });
            }
            _ => {
                if let Some(first) = op.regions.first() {
                    stack.push(Frame { ops: first.ops.iter(), written, exit: Exit::Branch(op, 0) });
                }
            }
        }
    }
    let seeds: Vec<ValueId> = carried
        .iter()
        .filter(|(arg, yielded)| u(&uniform, arg) && !yielded.is_some_and(|y| u(&uniform, &y)))
        .map(|&(arg, _)| arg)
        .collect();
    if seeds.is_empty() {
        return uniform;
    }
    // Some loop-carried value is divergent after all: clear it and
    // everything that depends on it.
    let mut edges = Vec::new();
    walk_loops_and_branches(&kernel.body, |op| match op.kind {
        OpKind::For => {
            carried_edges(op, &mut edges);
            bound_edges(op, &mut edges);
        }
        OpKind::If | OpKind::IfElse => {}
        _ => {
            if let Rule::Inputs(inputs) = uniform_rule(kernel, op, false) {
                for &result in &op.results {
                    edges.extend(inputs.iter().map(|&v| (v, result)));
                }
            }
        }
    });
    propagate_false(&mut uniform, &edges, seeds);
    uniform
}

fn set(values: &mut [bool], v: ValueId, value: bool) {
    if let Some(slot) = values.get_mut(v.0 as usize) {
        *slot = value;
    }
}

/// How a property of an op's results follows from its operands.
enum Rule<'a> {
    Always(bool),
    /// Holds when it holds for all of these.
    Inputs(&'a [ValueId]),
}

fn uniform_rule<'a>(kernel: &Kernel, op: &'a Op, written: bool) -> Rule<'a> {
    match op.kind {
        OpKind::Constant | OpKind::KernArg | OpKind::BlockId | OpKind::ReadFirstLane | OpKind::SmemAlloc => {
            Rule::Always(true)
        }
        // An LDS address is the allocation's offset, the same in every lane.
        OpKind::SmemAddrOf => Rule::Always(true),
        OpKind::SmemPtrAdd => Rule::Inputs(&op.operands),
        // Wave votes: one value for the whole wave.
        OpKind::WaveAny | OpKind::WaveAll => Rule::Always(true),
        OpKind::ReadLane => Rule::Always(true),
        OpKind::Not | OpKind::GlobalPtrToI64 => Rule::Inputs(&op.operands),
        OpKind::VecBitcast => Rule::Inputs(op.operands.get(..1).unwrap_or(&[])),
        OpKind::Add
        | OpKind::Sub
        | OpKind::Mul
        | OpKind::Div
        | OpKind::Mod
        | OpKind::And
        | OpKind::Or
        | OpKind::Xor
        | OpKind::Shl
        | OpKind::Lshr
        | OpKind::Ashr
        | OpKind::Smin
        | OpKind::Smax
        | OpKind::UmulHi
        | OpKind::SmulHi
        | OpKind::Cmp
        | OpKind::Select
        | OpKind::Zext
        | OpKind::Sext
        | OpKind::Trunc
        | OpKind::Bitcast
        | OpKind::GlobalPtrAdd
        | OpKind::BufferRsrc => Rule::Inputs(&op.operands),
        OpKind::Extract => Rule::Inputs(op.operands.get(..1).unwrap_or(&[])),
        OpKind::GLoad if smem_eligible_statically(kernel, op, written) => Rule::Inputs(&op.operands),
        _ => Rule::Always(false),
    }
}

/// The values an `scf.for` body yields: the operands of its trailing
/// `scf.yield` (a loop without iteration arguments may have none).
fn loop_yields(op: &Op) -> &[ValueId] {
    match op.regions.first().and_then(|b| b.ops.last()) {
        Some(last) if last.kind == OpKind::Yield => &last.operands,
        _ => &[],
    }
}

/// Edges (input, loop-carried value) and (loop-carried value, loop result)
/// of an `scf.for`.
fn carried_edges(op: &Op, edges: &mut Vec<(ValueId, ValueId)>) {
    let yields = loop_yields(op);
    for (i, &arg) in op.block_args.iter().enumerate().skip(1) {
        if let Some(&init) = op.operands.get(2 + i) {
            edges.push((init, arg));
        }
        if let Some(&yielded) = yields.get(i - 1) {
            edges.push((yielded, arg));
        }
    }
    for (i, &result) in op.results.iter().enumerate() {
        if let Some(&arg) = op.block_args.get(i + 1) {
            edges.push((arg, result));
        }
    }
}

/// Uniformity edges from the bounds of an `scf.for`: the induction variable
/// follows the start and step, the loop-carried values all three bounds.
fn bound_edges(op: &Op, edges: &mut Vec<(ValueId, ValueId)>) {
    let bounds = op.operands.get(..3).unwrap_or(&[]);
    if let Some(&iv) = op.block_args.first() {
        edges.extend(bounds.iter().step_by(2).map(|&b| (b, iv)));
    }
    for &arg in op.block_args.iter().skip(1) {
        edges.extend(bounds.iter().map(|&b| (b, arg)));
    }
}

/// Calls `f` on every op in the regions the uniformity and canonicality
/// analyses enter: the kernel body, `scf.for` bodies and `scf.if` regions.
fn walk_loops_and_branches<'a>(region: &'a Region, mut f: impl FnMut(&'a Op)) {
    let mut stack = vec![region.ops.iter()];
    while let Some(top) = stack.last_mut() {
        let Some(op) = top.next() else {
            stack.pop();
            continue;
        };
        f(op);
        match op.kind {
            OpKind::For => stack.extend(op.regions.first().map(|r| r.ops.iter())),
            OpKind::If | OpKind::IfElse => stack.extend(op.regions.iter().rev().map(|r| r.ops.iter())),
            _ => {}
        }
    }
}

/// Clears `holds` for the `seeds` and, along `edges` (input, dependent), for
/// everything that depends on a value it was cleared for.
fn propagate_false(holds: &mut [bool], edges: &[(ValueId, ValueId)], seeds: Vec<ValueId>) {
    let n = holds.len();
    let mut start = vec![0u32; n + 1];
    for &(from, _) in edges {
        if (from.0 as usize) < n {
            start[from.0 as usize + 1] += 1;
        }
    }
    for i in 0..n {
        start[i + 1] += start[i];
    }
    let mut fill = start.clone();
    let mut targets = vec![ValueId(0); start[n] as usize];
    for &(from, to) in edges {
        if (from.0 as usize) < n {
            targets[fill[from.0 as usize] as usize] = to;
            fill[from.0 as usize] += 1;
        }
    }
    let mut work = Vec::new();
    for seed in seeds {
        if holds.get(seed.0 as usize) == Some(&true) {
            holds[seed.0 as usize] = false;
            work.push(seed);
        }
    }
    while let Some(v) = work.pop() {
        let (lo, hi) = (start[v.0 as usize] as usize, start[v.0 as usize + 1] as usize);
        for &to in &targets[lo..hi] {
            if holds.get(to.0 as usize) == Some(&true) {
                holds[to.0 as usize] = false;
                work.push(to);
            }
        }
    }
}

/// Whether the body of the `scf.for` `op` may write global memory. The scan
/// stops at the first write and records the answer, in `known` (keyed by
/// address), for every loop it went into: those it left write nothing,
/// those still open contain the write. Loops are visited outside in, so no
/// op is scanned twice.
fn loop_writes(op: &Op, known: &mut FxHashMap<usize, bool>) -> bool {
    let key = |op: &Op| op as *const Op as usize;
    if let Some(&writes) = known.get(&key(op)) {
        return writes;
    }
    let mut stack: Vec<(std::slice::Iter<Op>, Option<usize>)> =
        op.regions.first().map(|b| (b.ops.iter(), Some(key(op)))).into_iter().collect();
    while let Some((ops, _)) = stack.last_mut() {
        let Some(inner) = ops.next() else {
            if let Some((_, Some(done))) = stack.pop() {
                known.insert(done, false);
            }
            continue;
        };
        if inner.kind.writes_global() {
            for (_, open) in &stack {
                if let Some(open) = open {
                    known.insert(*open, true);
                }
            }
            return true;
        }
        for (i, r) in inner.regions.iter().enumerate() {
            let owner = (inner.kind == OpKind::For && i == 0).then(|| key(inner));
            stack.push((r.ops.iter(), owner));
        }
    }
    false
}

/// Whether a load may use the scalar cache: uniform address, dword-sized
/// elements with dword alignment, and no global write that could precede it.
pub fn smem_eligible(kernel: &Kernel, op: &Op, uniform: &[bool], written: bool) -> bool {
    smem_eligible_statically(kernel, op, written)
        && op.operands.iter().all(|v| uniform.get(v.0 as usize).copied().unwrap_or(false))
}

/// `smem_eligible` but for the uniformity of the address.
fn smem_eligible_statically(kernel: &Kernel, op: &Op, written: bool) -> bool {
    if op.kind != OpKind::GLoad || written {
        return false;
    }
    let Some(&result) = op.results.first() else { return false };
    // gfx950 (wave64) adds a negative offset into the scalar base.
    matches!(kernel.ty(result).elem(), Scalar::I32 | Scalar::F32 | Scalar::I64)
        && op.int_attr("align").unwrap_or(1) >= 4
        && (op.int_attr("offset").unwrap_or(0) >= 0 || kernel.wave == 64)
}

pub fn is_snan_f32(bits: u32) -> bool {
    bits & 0x7fc0_0000 == 0x7f80_0000 && bits & 0x003f_ffff != 0
}

fn word_elements(ty: &Type) -> bool {
    matches!(ty, Type::Scalar(Scalar::F32 | Scalar::I32) | Type::Vec(Scalar::F32 | Scalar::I32, _))
}

/// Per value: whether no 32-bit element of it, read as an f32, can be a
/// signaling NaN. FP arithmetic quiets NaNs and bit-exact moves (bitcasts,
/// swizzles, lane reads, selects, vector assembly, loop-carried values) keep
/// them; loads, kernel arguments and integer arithmetic may hold any pattern.
///
/// Iteration arguments start from their initial values and stay canonical
/// while every yield is, assuming the arguments are (induction on the trip):
/// the greatest fixpoint, as for `classify`.
pub fn canonical_f32(kernel: &Kernel, facts: &Facts) -> Vec<bool> {
    let mut canonical = vec![false; kernel.values.len()];
    let word = |v: ValueId| v.0 < kernel.values.len() as u32 && word_elements(kernel.ty(v));
    let c = |canonical: &Vec<bool>, v: &ValueId| canonical.get(v.0 as usize).copied().unwrap_or(false);
    // Program order, assuming every iteration argument stays canonical when
    // its initial value is; `carried` lists each with its yield.
    let mut carried: Vec<(ValueId, Option<ValueId>)> = Vec::new();
    let mut stack: Vec<(std::slice::Iter<Op>, Option<&Op>)> = vec![(kernel.body.ops.iter(), None)];
    while let Some((ops, owner)) = stack.last_mut() {
        // Plain ops in a tight loop; control flow below.
        let mut control = None;
        for op in ops.by_ref() {
            if matches!(op.kind, OpKind::For | OpKind::If | OpKind::IfElse) {
                control = Some(op);
                break;
            }
            let value = match canonical_rule(kernel, facts, op) {
                Rule::Always(value) => value,
                Rule::Inputs(inputs) => inputs.iter().all(|v| c(&canonical, v)),
            };
            for &result in &op.results {
                set(&mut canonical, result, value && word(result));
            }
        }
        let Some(op) = control else {
            if let Some(op) = *owner {
                for (i, &result) in op.results.iter().enumerate() {
                    let arg = op.block_args.get(i + 1).is_some_and(|a| c(&canonical, a));
                    set(&mut canonical, result, arg);
                }
            }
            stack.pop();
            continue;
        };
        match op.kind {
            OpKind::For => {
                let Some(body) = op.regions.first() else { continue };
                let yields = loop_yields(op);
                for (i, &arg) in op.block_args.iter().enumerate().skip(1) {
                    let init = op.operands.get(2 + i).is_some_and(|v| c(&canonical, v)) && word(arg);
                    set(&mut canonical, arg, init);
                    carried.push((arg, yields.get(i - 1).copied()));
                }
                stack.push((body.ops.iter(), Some(op)));
            }
            _ => stack.extend(op.regions.iter().rev().map(|r| (r.ops.iter(), None))),
        }
    }
    let seeds: Vec<ValueId> = carried
        .iter()
        .filter(|(arg, yielded)| c(&canonical, arg) && !yielded.is_some_and(|y| c(&canonical, &y)))
        .map(|&(arg, _)| arg)
        .collect();
    if seeds.is_empty() {
        return canonical;
    }
    let mut edges = Vec::new();
    walk_loops_and_branches(&kernel.body, |op| match op.kind {
        OpKind::For => carried_edges(op, &mut edges),
        OpKind::If | OpKind::IfElse => {}
        _ => {
            if let Rule::Inputs(inputs) = canonical_rule(kernel, facts, op) {
                for &result in &op.results {
                    edges.extend(inputs.iter().map(|&v| (v, result)));
                }
            }
        }
    });
    propagate_false(&mut canonical, &edges, seeds);
    canonical
}

fn canonical_rule<'a>(kernel: &Kernel, facts: &Facts, op: &'a Op) -> Rule<'a> {
    let first = |n: usize| op.operands.get(..n).unwrap_or(&[]);
    match op.kind {
        // An inline-asm LDS read (gfx950) gives what memory holds.
        OpKind::InlineAsmIdentity if op.str_attr("template").is_some_and(|t| !t.is_empty()) => Rule::Always(false),
        OpKind::Constant => {
            let konst = op.results.first().and_then(|r| facts.konst.get(r.0 as usize).copied().flatten());
            Rule::Always(match konst {
                Some(Konst::F32(bits)) => !is_snan_f32(bits),
                Some(Konst::Int(v)) => !is_snan_f32(v as u32),
                None => false,
            })
        }
        OpKind::ConstantVec => {
            let fill = op.float_attr("fill").unwrap_or(0.0);
            Rule::Always(match op.results.first().map(|&r| kernel.ty(r).elem()) {
                Some(Scalar::F32) => !is_snan_f32((fill as f32).to_bits()),
                Some(Scalar::I32) => !is_snan_f32(fill as i64 as u32),
                _ => false,
            })
        }
        OpKind::Fadd
        | OpKind::Fsub
        | OpKind::Fmul
        | OpKind::Fma
        | OpKind::Fmax
        | OpKind::Exp2
        | OpKind::Log2
        | OpKind::Rcp
        | OpKind::Mma
        | OpKind::Fmax3
        | OpKind::Fmin
        | OpKind::Fmin3
        | OpKind::Fdiv
        | OpKind::Fdot2
        | OpKind::Exp2Fast
        | OpKind::RcpFast
        | OpKind::Rsqrt
        | OpKind::Tanh
        | OpKind::SitofpF32 => Rule::Always(true),
        // A bf16 widens by a shift, which keeps a signaling NaN.
        OpKind::CastToF32 => Rule::Always(op.operands.first().is_some_and(|&v| kernel.ty(v).elem() != Scalar::BF16)),
        OpKind::VecAdd | OpKind::VecSub | OpKind::VecMul | OpKind::VecFma => {
            Rule::Always(op.results.first().is_some_and(|&r| kernel.ty(r).elem().is_float()))
        }
        OpKind::Bitcast
        | OpKind::Extract
        | OpKind::DsSwizzleXor
        | OpKind::ReadFirstLane
        | OpKind::InlineAsmIdentity
        | OpKind::Fneg
        | OpKind::Fabs
        | OpKind::VecBitcast
        | OpKind::VecSplat
        | OpKind::Permlanex16
        | OpKind::Permlane64 => match first(1) {
            [] => Rule::Always(false),
            inputs => Rule::Inputs(inputs),
        },
        // A lane whose source lane is inactive keeps `old`, operand 0.
        OpKind::Permlane16 => match first(2) {
            inputs @ [_, _] => Rule::Inputs(inputs),
            _ => Rule::Always(false),
        },
        OpKind::DsBpermute => match op.operands.get(1..2) {
            Some(inputs) => Rule::Inputs(inputs),
            None => Rule::Always(false),
        },
        OpKind::Select => match op.operands.get(1..3) {
            Some(inputs) => Rule::Inputs(inputs),
            None => Rule::Always(false),
        },
        OpKind::Insert | OpKind::Pack | OpKind::Concat => Rule::Inputs(&op.operands),
        _ => Rule::Always(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROBE: &str = r#"ckdsl.ir v1
kernel @canonical_probe {
  attrs { max_workgroup_size = i:64 }
  params {
    %X : ptr<f32,global>
    %Y : ptr<f32,global>
    %s : f32
    %n : i32
  }
  region @entry {
    %tid : i32 = gpu.thread_id (  ) { axis = s:"x" }
    %ld : f32 = memref.global_load_typed ( %X, %tid ) { align = i:4, elem_type = s:"f32" }
    %mul : f32 = arith.fmul ( %ld, %s )
    %bc : i32 = arith.bitcast ( %mul ) { target = s:"i32" }
    %sw : i32 = tile.ds_swizzle_xor ( %bc ) { xor_mask = i:1 }
    %swf : f32 = arith.bitcast ( %sw ) { target = s:"f32" }
    %raw : i32 = arith.bitcast ( %ld ) { target = s:"i32" }
    %rsw : i32 = tile.ds_swizzle_xor ( %raw ) { xor_mask = i:1 }
    %rswf : f32 = arith.bitcast ( %rsw ) { target = s:"f32" }
    %snan : i32 = arith.constant (  ) { ity = s:"i32", value = i:2141192192 }
    %snanf : f32 = arith.bitcast ( %snan ) { target = s:"f32" }
    %one : i32 = arith.constant (  ) { ity = s:"i32", value = i:1065353216 }
    %onef : f32 = arith.bitcast ( %one ) { target = s:"f32" }
    %ninf : f32 = arith.constant (  ) { ity = s:"f32", value = f:-inf }
    %c0 : i32 = arith.constant (  ) { ity = s:"i32", value = i:0 }
    %c1 : i32 = arith.constant (  ) { ity = s:"i32", value = i:1 }
    %lt : i1 = arith.cmp ( %tid, %n ) { pred = s:"lt" }
    %pick : f32 = arith.select ( %lt, %mul, %swf )
    %mixed : f32 = arith.select ( %lt, %mul, %ld )
    %rx : f32, %rw : f32, %rz : f32 = scf.for ( %c0, %n, %c1, %ninf, %ninf, %ninf ) { iter_args = l:[ { name = s:"%x", type = s:"f32" }, { name = s:"%w", type = s:"f32" }, { name = s:"%z", type = s:"f32" } ], iv = s:"%i", iv_type = s:"i32", num_iter_args = i:3, unroll = b:false }
      region @body {
        %x2 : f32 = arith.select ( %lt, %x, %w )
        %z2 : f32 = arith.fmax ( %z, %mul )
        scf.yield ( %x2, %ld, %z2 ) { num = i:3 }
      }
    memref.global_store_typed ( %Y, %tid, %pick ) { align = i:4, elem_type = s:"f32" }
    memref.global_store_typed ( %Y, %tid, %mixed ) { align = i:4, elem_type = s:"f32" }
    memref.global_store_typed ( %Y, %tid, %rswf ) { align = i:4, elem_type = s:"f32" }
    memref.global_store_typed ( %Y, %tid, %snanf ) { align = i:4, elem_type = s:"f32" }
    memref.global_store_typed ( %Y, %tid, %onef ) { align = i:4, elem_type = s:"f32" }
    memref.global_store_typed ( %Y, %tid, %rx ) { align = i:4, elem_type = s:"f32" }
    memref.global_store_typed ( %Y, %tid, %rw ) { align = i:4, elem_type = s:"f32" }
    memref.global_store_typed ( %Y, %tid, %rz ) { align = i:4, elem_type = s:"f32" }
    memref.global_store_typed ( %Y, %tid, %s ) { align = i:4, elem_type = s:"f32" }
    cf.return (  )
  }
}
"#;

    #[test]
    fn canonicality_follows_provenance() {
        let mut kernel = super::super::parse::parse(PROBE).unwrap();
        let (mut lds, _) = lds_layout(&kernel);
        let facts = super::super::opt::optimize(&mut kernel, &mut lds).unwrap();
        let canonical = canonical_f32(&kernel, &facts);
        let of = |name: &str| {
            let v = (0..kernel.values.len() as u32)
                .map(ValueId)
                .find(|&v| kernel.name_of(v) == name)
                .unwrap_or_else(|| panic!("{name} was optimized away"));
            canonical[v.0 as usize]
        };
        for name in ["%mul", "%bc", "%sw", "%swf", "%pick", "%onef", "%z", "%z2", "%rz"] {
            assert!(of(name), "{name} should be canonical");
        }
        for name in ["%ld", "%raw", "%rsw", "%rswf", "%snanf", "%mixed", "%x", "%w", "%rx", "%rw", "%arg.s"] {
            assert!(!of(name), "{name} may be a signaling NaN");
        }
        assert!(is_snan_f32(0x7fa0_0000) && is_snan_f32(0xff80_0001));
        assert!(!is_snan_f32(0x7fc0_0000) && !is_snan_f32(0x7f80_0000) && !is_snan_f32(0x0000_0001));
    }

    /// Ten loop-carried values each take the next one's value and the last
    /// a lane's: all ten are divergent, though only one more is found per
    /// pass over the body. Stopping after eight passes, as the analysis
    /// once did, left two of them uniform.
    #[test]
    fn loop_carried_divergence_reaches_every_value() {
        let n = 10;
        let names: Vec<String> = (1..=n).map(|i| format!("%x{i}")).collect();
        let iter_args: Vec<String> = names.iter().map(|x| format!("{{ name = s:\"{x}\", type = s:\"i32\" }}")).collect();
        let results: Vec<String> = (1..=n).map(|i| format!("%r{i} : i32")).collect();
        let yields: Vec<String> = names[1..].iter().cloned().chain(["%tid".to_string()]).collect();
        let text = format!(
            "ckdsl.ir v1\nkernel @chain {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n    %n : i32\n  }}\n  region @entry {{\n    \
             %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n    %c0 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:0 }}\n    \
             %c1 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:1 }}\n    \
             {} = scf.for ( %c0, %n, %c1, {} ) {{ iter_args = l:[ {} ], iv = s:\"%i\", iv_type = s:\"i32\", num_iter_args = i:{n} }}\n      \
             region @body {{\n        scf.yield ( {} )\n      }}\n    cf.return (  )\n  }}\n}}\n",
            results.join(", "),
            vec!["%c0"; n].join(", "),
            iter_args.join(", "),
            yields.join(", "),
        );
        let kernel = super::super::parse::parse(&text).unwrap();
        let uniform = classify(&kernel);
        let divergent: Vec<String> = (0..kernel.values.len() as u32)
            .map(ValueId)
            .filter(|&v| !uniform[v.0 as usize])
            .map(|v| kernel.name_of(v).to_string())
            .collect();
        for name in names.iter().chain(&results.iter().map(|r| r.split(' ').next().unwrap().to_string()).collect::<Vec<_>>()) {
            assert!(divergent.contains(name), "{name} is divergent: {divergent:?}");
        }
        assert!(uniform[kernel.params[0].value.0 as usize]);
    }

    /// `tile.permlane16(old, src, ..)`: a lane whose source lane is inactive
    /// keeps `old`, the others read `src`; the result is as canonical as both.
    #[test]
    fn permlane16_is_as_canonical_as_old_and_src() {
        let text = "ckdsl.ir v1\nkernel @perm {\n  attrs { max_workgroup_size = i:64 }\n  params {\n    %X : ptr<f32,global>\n    %Y : ptr<i32,global>\n  }\n  \
                    region @entry {\n    %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n    \
                    %c0 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }\n    %one : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:1065353216 }\n    \
                    %ld : f32 = memref.global_load_typed ( %X, %tid ) { align = i:4 }\n    %raw : i32 = arith.bitcast ( %ld )\n    \
                    %p : i32 = tile.permlane16 ( %one, %raw, %c0, %c0 )\n    %q : i32 = tile.permlane16 ( %one, %one, %c0, %c0 )\n    \
                    memref.global_store_typed ( %Y, %tid, %p ) { align = i:4 }\n    memref.global_store_typed ( %Y, %c0, %q ) { align = i:4 }\n    \
                    cf.return (  )\n  }\n}\n";
        let mut kernel = super::super::parse::parse(text).unwrap();
        let (mut lds, _) = lds_layout(&kernel);
        let facts = super::super::opt::optimize(&mut kernel, &mut lds).unwrap();
        let canonical = canonical_f32(&kernel, &facts);
        let mut perms = Vec::new();
        preorder(&kernel.body, |op| {
            if op.kind == OpKind::Permlane16 {
                perms.push((facts.int(op.operands[1]).is_some(), canonical[op.results[0].0 as usize]));
            }
        });
        perms.sort();
        assert_eq!(perms, [(false, false), (true, true)]);
    }
}
