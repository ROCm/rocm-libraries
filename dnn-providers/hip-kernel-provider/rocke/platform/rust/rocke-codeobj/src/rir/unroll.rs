// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Full unrolling of `scf.for` with `unroll = true`, as the LLVM lowering
//! (`_lower_unrolled_for`) does it: only when the three bounds are
//! `arith.constant` results, with `(ub - lb) // step` iterations (floor
//! division), and with a `tile.sync` right before the yield dropped from
//! every iteration but the last unless `elide_trailing_barrier = false`.
//! Other loops stay loops, whatever their hint, unless an [`Auto`] policy
//! admits them (`unroll_optimized`). Every walk uses an explicit stack, so
//! nesting depth costs no native stack.

use super::ast::{Attr, Kernel, Op, OpKind, Region, Scalar, Type, ValueId};
use super::fx::{FxHashMap, FxHashSet};
use super::opt::{self, Facts, Workspace};
use crate::error::CompileError;

/// Most ops unrolling may add to a kernel; beyond it the kernel is refused
/// rather than built (the reference lowering would emit them all).
pub const MAX_UNROLLED_OPS: usize = 1 << 20;

/// Which of the loops the IR leaves rolled are unrolled fully anyway, as
/// LLVM's full unroller unrolls small constant-trip loops at -O3 on its
/// own: innermost `scf.for` loops whose three bounds are `arith.constant`
/// results, with 1 to `max_trips` iterations, at most `max_ops` ops in all
/// copies of the body, and no load that reads a value carried from the
/// iteration before (other than one stepped by a loop-invariant amount), so
/// that the copies' loads can all be in flight at once. Copies of all such
/// loops add at most `max_total_ops` ops to a kernel; the loops past that
/// stay rolled. Every barrier stays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Auto {
    pub max_trips: i64,
    pub max_ops: usize,
    pub max_total_ops: usize,
}

impl Auto {
    /// The policy `compile` tries. 32 iterations is as many as the AMDGPU
    /// target lets LLVM's full unroller analyze in a small innermost loop;
    /// 512 ops lies between that unroller's AMDGPU threshold of 300 and the
    /// four times as much it allows copies that simplify; 16384 ops of
    /// copies keep the largest unrolled loop body measured within the reach
    /// of a 16-bit branch offset.
    pub const DEFAULT: Auto = Auto { max_trips: 32, max_ops: 512, max_total_ops: 1 << 14 };
}

/// Unrolls every eligible loop, innermost first. Cloned LDS allocations
/// get the offset of the allocation they copy.
pub fn unroll_loops(kernel: &mut Kernel, lds: &mut FxHashMap<ValueId, u32>) -> Result<(), CompileError> {
    unroll_loops_with(kernel, lds, None).map(|_| ())
}

/// `unroll_loops`, also unrolling the loops `auto` admits. Returns how many
/// of those it unrolled; with none, the kernel is as `unroll_loops` leaves it.
pub fn unroll_loops_with(
    kernel: &mut Kernel,
    lds: &mut FxHashMap<ValueId, u32>,
    auto: Option<Auto>,
) -> Result<usize, CompileError> {
    unroll_walk(kernel, lds, auto).map(|(n, _)| n)
}

/// `unroll_loops_with`, also returning the induction variables of the loops
/// whose bodies the copies of loops `auto` admitted went into.
fn unroll_walk(
    kernel: &mut Kernel,
    lds: &mut FxHashMap<ValueId, u32>,
    auto: Option<Auto>,
) -> Result<(usize, FxHashSet<ValueId>), CompileError> {
    if !has_unrollable_loop(&kernel.body, auto.is_some()) {
        return Ok((0, FxHashSet::default()));
    }
    let mut consts = FxHashMap::default();
    collect_consts(&kernel.body, &mut consts);
    let mut body = std::mem::take(&mut kernel.body);
    let mut cx = Unroller {
        kernel,
        lds,
        consts,
        repl: FxHashMap::default(),
        added: 0,
        auto,
        auto_added: 0,
        auto_loops: 0,
        hosts: FxHashSet::default(),
    };
    // Until a loop unrolls the walk changes nothing, so a kernel none of
    // whose loops unrolls as it stands comes out of it as it went in.
    if !cx.any_unrolls(&body) {
        cx.kernel.body = body;
        return Ok((0, FxHashSet::default()));
    }
    let done = cx.region(&mut body);
    let repl = std::mem::take(&mut cx.repl);
    let (auto_loops, hosts) = (cx.auto_loops, std::mem::take(&mut cx.hosts));
    kernel.body = body;
    done?;
    if !repl.is_empty() {
        substitute(&mut kernel.body, &repl);
    }
    Ok((auto_loops, hosts))
}

/// Unrolls the loops `auto` admits in a kernel `opt::optimize` (with
/// `keep_lds_loads` as it had it) has finished, then redoes the passes that
/// end it. Code motion out of loops has run by then, so what each copy
/// computes stays in the loop around it; unrolled before it, every copy
/// would leave one value hoisted out of that loop and live across all of it.
/// The scalars code motion did take out of the loops that receive copies go
/// back in, and the copies' constant offsets go into their buffer accesses,
/// so that the unrolled body needs about the registers the loops did.
/// Returns the facts for the kernel and the loops unrolled.
pub fn unroll_optimized(
    kernel: &mut Kernel,
    lds: &mut FxHashMap<ValueId, u32>,
    facts: Facts,
    auto: Auto,
    keep_lds_loads: bool,
) -> Result<(Facts, usize), CompileError> {
    let (unrolled, hosts) = unroll_walk(kernel, lds, Some(auto))?;
    if unrolled == 0 {
        return Ok((facts, 0));
    }
    sink_invariants(kernel, &hosts);
    let mut ws = Workspace::default();
    ws.keep_lds_loads = keep_lds_loads;
    opt::sweep(kernel, &mut ws)?;
    if fold_select_offsets(kernel, &ws.facts) {
        opt::sweep(kernel, &mut ws)?;
    }
    unshare_offsets(kernel, &mut ws.facts);
    opt::dce(kernel);
    opt::compact(kernel, &mut ws, lds);
    super::loadorder::order_global_loads(&mut kernel.body);
    Ok((ws.facts, unrolled))
}

/// Moves the cheap scalar ops (uniform i32, or i1 of any uniformity) that
/// loop-invariant code motion hoisted out of a loop in `hosts` (by
/// induction variable) back into its body, right before the first op there
/// that reads them, when nothing outside reads them. Unrolled, such a
/// loop's body holds the copies of many loops that each read their own few
/// scalars; hoisted, all of those stay live across the whole body at once,
/// past the SGPRs there are. Moved, each is computed once per iteration
/// where its copies start, as it was before code motion.
fn sink_invariants(kernel: &mut Kernel, hosts: &FxHashSet<ValueId>) {
    if hosts.is_empty() {
        return;
    }
    let uniform = super::analysis::classify(kernel);
    let mut body = std::mem::take(&mut kernel.body);
    let mut total: FxHashMap<ValueId, u32> = FxHashMap::default();
    let mut walk = vec![&body];
    while let Some(region) = walk.pop() {
        for op in &region.ops {
            for &v in op.operands.iter() {
                *total.entry(v).or_default() += 1;
            }
            walk.extend(op.regions.iter());
        }
    }
    let kernel_ref: &Kernel = kernel;
    let cheap = |op: &Op| {
        matches!(
            op.kind,
            OpKind::Add
                | OpKind::Sub
                | OpKind::Mul
                | OpKind::And
                | OpKind::Or
                | OpKind::Xor
                | OpKind::Shl
                | OpKind::Lshr
                | OpKind::Ashr
                | OpKind::Smin
                | OpKind::Smax
                | OpKind::Cmp
                | OpKind::Select
                | OpKind::Not
        ) && op.regions.is_empty()
            && !op.results.is_empty()
            && op.results.iter().all(|r| match kernel_ref.ty(*r) {
                // Divergent too: a wave32 lane mask is one SGPR.
                Type::Scalar(Scalar::I1) => true,
                Type::Scalar(Scalar::I32) => uniform.get(r.0 as usize).copied().unwrap_or(false),
                _ => false,
            })
    };
    let mut stack = vec![&mut body];
    while let Some(region) = stack.pop() {
        let loops: Vec<usize> = (0..region.ops.len())
            .filter(|&i| region.ops[i].kind == OpKind::For && hosts.contains(&region.ops[i].block_args[0]))
            .collect();
        let mut moved = vec![None; region.ops.len()];
        for &l in loops.iter().rev() {
            // Reads inside the loop body, and the first top-level op reading each value.
            let mut inside: FxHashMap<ValueId, (u32, usize)> = FxHashMap::default();
            for (k, top) in region.ops[l].regions[0].ops.iter().enumerate() {
                let mut walk = vec![top];
                while let Some(op) = walk.pop() {
                    for &v in op.operands.iter() {
                        let e = inside.entry(v).or_insert((0, k));
                        e.0 += 1;
                    }
                    walk.extend(op.regions.iter().flat_map(|r| r.ops.iter()));
                }
            }
            for j in (0..l).rev() {
                let op = &region.ops[j];
                if moved[j].is_some() || !cheap(op) {
                    continue;
                }
                let at = op.results.iter().map(|r| match inside.get(r) {
                    Some(&(n, k)) if total.get(r) == Some(&n) => Some(k),
                    _ => None,
                });
                let Some(target) = at.collect::<Option<Vec<usize>>>().and_then(|ks| ks.into_iter().min()) else {
                    continue;
                };
                moved[j] = Some((l, target));
                for &v in op.operands.iter() {
                    let e = inside.entry(v).or_insert((0, target));
                    e.0 += 1;
                    e.1 = e.1.min(target);
                }
            }
        }
        if moved.iter().any(Option::is_some) {
            let mut sunk: FxHashMap<usize, Vec<(usize, Op)>> = FxHashMap::default();
            let mut kept = Vec::with_capacity(region.ops.len());
            for (j, op) in std::mem::take(&mut region.ops).into_iter().enumerate() {
                match moved[j] {
                    Some((l, target)) => sunk.entry(l).or_default().push((target, op)),
                    None => kept.push((j, op)),
                }
            }
            for (j, op) in kept.iter_mut() {
                let Some(mut ops) = sunk.remove(j) else {
                    continue;
                };
                // Program order among those moved before the same op.
                ops.sort_by_key(|&(target, _)| target);
                let old = std::mem::take(&mut op.regions[0].ops);
                let mut new = Vec::with_capacity(old.len() + ops.len());
                let mut ops = ops.into_iter().peekable();
                for (k, inner) in old.into_iter().enumerate() {
                    while let Some((_, sunk)) = ops.next_if(|&(target, _)| target == k) {
                        new.push(sunk);
                    }
                    new.push(inner);
                }
                new.extend(ops.map(|(_, sunk)| sunk));
                op.regions[0].ops = new;
            }
            region.ops = kept.into_iter().map(|(_, op)| op).collect();
        }
        stack.extend(region.ops.iter_mut().flat_map(|op| op.regions.iter_mut()));
    }
    kernel.body = body;
}

/// Gives each select of a buffer access's offset its own copy of an
/// `x + k` arm (`k` constant) that other ops share. Unrolled copies of
/// different loops compute equal sums that the sweep merged into one value,
/// live from the first copy that reads it to the last; as before unrolling,
/// each copy adds `k` where it uses the sum. Runs after the last sweep,
/// which would merge them again; each copy takes the facts of the sum.
fn unshare_offsets(kernel: &mut Kernel, facts: &mut Facts) {
    let mut uses: FxHashMap<ValueId, u32> = FxHashMap::default();
    let mut sums: FxHashMap<ValueId, Op> = FxHashMap::default();
    let mut offsets: FxHashMap<ValueId, ()> = FxHashMap::default();
    let mut stack = vec![&kernel.body];
    while let Some(region) = stack.pop() {
        for op in &region.ops {
            for &v in op.operands.iter() {
                *uses.entry(v).or_default() += 1;
            }
            if op.kind == OpKind::Add && op.operands.len() == 2 && facts.int(op.operands[1]).is_some() {
                sums.insert(op.results[0], op.clone());
            }
            if matches!(op.kind, OpKind::BufferLoadScalar | OpKind::BufferLoadVec | OpKind::BufferStoreScalar | OpKind::BufferStoreVecF16) {
                if let Some(&v) = op.operands.get(1) {
                    offsets.insert(v, ());
                }
            }
            stack.extend(op.regions.iter());
        }
    }
    let shared = |v: &ValueId| sums.contains_key(v) && uses.get(v).is_some_and(|&n| n > 1);
    let mut body = std::mem::take(&mut kernel.body);
    let mut stack = vec![&mut body];
    while let Some(region) = stack.pop() {
        let ops = std::mem::take(&mut region.ops);
        let mut out = Vec::with_capacity(ops.len());
        for mut op in ops {
            if op.kind == OpKind::Select && op.operands.len() == 3 && offsets.contains_key(&op.results[0]) {
                for arm in 1..3 {
                    let v = op.operands[arm];
                    if !shared(&v) {
                        continue;
                    }
                    let mut copy = sums[&v].clone();
                    let name = kernel.name_of(v).to_string();
                    let ty = kernel.ty(v).clone();
                    let new = kernel.new_value(&name, ty);
                    copy.results = [new].into();
                    copy_facts(facts, v, new);
                    out.push(copy);
                    op.operands[arm] = new;
                }
            }
            out.push(op);
        }
        region.ops = out;
        stack.extend(region.ops.iter_mut().flat_map(|op| op.regions.iter_mut()));
    }
    kernel.body = body;
}

/// Immediate offsets of buffer accesses take 12 bits.
const BUFFER_OFFSET_BITS: u32 = 12;

/// Folds a constant `k` into a buffer access's immediate offset through a
/// select whose other arm is a constant `b`, as `opt::legalize_addresses`
/// folds it from a plain no-wrap `x + k` with `x` known non-negative:
/// `select(c, x + k, b)` is `select(c, x, b - k) + k` exactly, so with
/// `b - k` non-negative the select of `x` is a non-negative base. A `k` too
/// large for the field leaves its multiple of 4096 in the base, as LLVM
/// splits buffer offsets, so copies whose `k` share it share that sum.
/// Unrolled copies address `x + k` this way for each `k`; folded, no copy
/// keeps its own sum of `x` live across the copies of other loops.
fn fold_select_offsets(kernel: &mut Kernel, facts: &Facts) -> bool {
    // Adds and selects by result: kind, operands, and whether no-wrap.
    let mut defs: FxHashMap<ValueId, (OpKind, [ValueId; 3], bool)> = FxHashMap::default();
    let mut stack = vec![&kernel.body];
    while let Some(region) = stack.pop() {
        for op in &region.ops {
            let arity = match op.kind {
                OpKind::Add => 2,
                OpKind::Select => 3,
                _ => 0,
            };
            if arity > 0 && op.operands.len() == arity && op.results.len() == 1 {
                let at = |i: usize| op.operands[i.min(arity - 1)];
                defs.insert(op.results[0], (op.kind, [at(0), at(1), at(2)], op.attr("wrap").is_none()));
            }
            stack.extend(op.regions.iter());
        }
    }
    let split = |v: &ValueId| match defs.get(v) {
        Some(&(OpKind::Add, [x, y, _], true)) => facts.int(y).map(|k| (x, k)),
        _ => None,
    };
    // The select's condition, `x`, `k`, `b`, and whether `x` is the true arm.
    let plan = |op: &Op| -> Option<(ValueId, ValueId, i64, i64, bool)> {
        if !matches!(op.kind, OpKind::BufferLoadScalar | OpKind::BufferLoadVec | OpKind::BufferStoreScalar | OpKind::BufferStoreVecF16) {
            return None;
        }
        let &(kind, [c, t, f], _) = defs.get(op.operands.get(1)?)?;
        if kind != OpKind::Select {
            return None;
        }
        let offset = op.int_attr("offset").unwrap_or(0);
        let fits = |x: ValueId, k: i64, b: i64| {
            let low = k & ((1 << BUFFER_OFFSET_BITS) - 1);
            k > 0 && low > 0 && offset + low < 1 << BUFFER_OFFSET_BITS && b >= low && facts.nonneg(x)
        };
        if let (Some((x, k)), Some(b)) = (split(&t), facts.int(f)) {
            if fits(x, k, b) {
                return Some((c, x, k, b, true));
            }
        }
        if let (Some(b), Some((x, k))) = (facts.int(t), split(&f)) {
            if fits(x, k, b) {
                return Some((c, x, k, b, false));
            }
        }
        None
    };
    let mut changed = false;
    let mut body = std::mem::take(&mut kernel.body);
    let mut stack = vec![&mut body];
    while let Some(region) = stack.pop() {
        let ops = std::mem::take(&mut region.ops);
        let mut out = Vec::with_capacity(ops.len());
        for mut op in ops {
            if let Some((c, x, k, b, x_true)) = plan(&op) {
                let low = k & ((1 << BUFFER_OFFSET_BITS) - 1);
                let mut x = x;
                if k > low {
                    // `x + (k - low)` stays below `x + k`: no wrap either.
                    let high = constant(kernel, k - low, &mut out);
                    let sum = kernel.new_value("%uadd", Type::Scalar(Scalar::I32));
                    out.push(Op {
                        kind: OpKind::Add,
                        results: [sum].into(),
                        operands: [x, high].into_iter().collect(),
                        attrs: Vec::new(),
                        regions: Vec::new(),
                        block_args: Vec::new(),
                    });
                    x = sum;
                }
                let rest = constant(kernel, b - low, &mut out);
                let base = kernel.new_value("%usel", Type::Scalar(Scalar::I32));
                let arms = if x_true { [c, x, rest] } else { [c, rest, x] };
                out.push(Op {
                    kind: OpKind::Select,
                    results: [base].into(),
                    operands: arms.into_iter().collect(),
                    attrs: Vec::new(),
                    regions: Vec::new(),
                    block_args: Vec::new(),
                });
                op.operands[1] = base;
                let offset = op.int_attr("offset").unwrap_or(0);
                op.set_attr("offset", Attr::Int(offset + low));
                changed = true;
            }
            out.push(op);
        }
        region.ops = out;
        stack.extend(region.ops.iter_mut().flat_map(|op| op.regions.iter_mut()));
    }
    kernel.body = body;
    changed
}

/// Gives `to` the facts of `from`.
fn copy_facts(facts: &mut Facts, from: ValueId, to: ValueId) {
    let len = facts.konst.len().max(facts.range.len()).max(to.0 as usize + 1);
    facts.konst.resize(len, None);
    facts.range.resize(len, None);
    facts.konst[to.0 as usize] = facts.konst[from.0 as usize];
    facts.range[to.0 as usize] = facts.range[from.0 as usize];
}

/// Appends an i32 `arith.constant` of `value` to `out`; returns its value.
fn constant(kernel: &mut Kernel, value: i64, out: &mut Vec<Op>) -> ValueId {
    let v = kernel.new_value("%uk", Type::Scalar(Scalar::I32));
    out.push(Op {
        kind: OpKind::Constant,
        results: [v].into(),
        operands: Default::default(),
        attrs: vec![("ity".into(), Attr::Str("i32".into())), ("value".into(), Attr::Int(value))],
        regions: Vec::new(),
        block_args: Vec::new(),
    });
    v
}

/// A loop marked `unroll = true`, or with `any`, any loop.
fn has_unrollable_loop(region: &Region, any: bool) -> bool {
    let mut stack = vec![region];
    while let Some(region) = stack.pop() {
        for op in &region.ops {
            if op.kind == OpKind::For && (any || op.bool_attr("unroll") == Some(true)) {
                return true;
            }
            stack.extend(op.regions.iter());
        }
    }
    false
}

/// Whether a load or a cross-lane op of the loop `op` reads a value that
/// depends on a carried value, so that each iteration's waits on the one
/// before. A carried value the body steps by a loop-invariant amount, as
/// `p + stride`, is an induction variable: its copies fold apart.
fn loads_wait_on_carried(op: &Op) -> bool {
    let body = &op.regions[0].ops;
    let mut inside: FxHashSet<ValueId> = op.block_args.iter().copied().collect();
    let mut all: Vec<&Op> = Vec::new();
    let mut stack = vec![&body[..]];
    while let Some(ops) = stack.pop() {
        for inner in ops {
            all.push(inner);
            inside.extend(inner.results.iter().copied());
            stack.extend(inner.regions.iter().map(|r| &r.ops[..]));
        }
    }
    let defs: FxHashMap<ValueId, &Op> = all.iter().flat_map(|o| o.results.iter().map(move |&r| (r, *o))).collect();
    let yielded = body.last().filter(|y| y.kind == OpKind::Yield).map_or(&[][..], |y| &y.operands[..]);
    let stepped = |arg: ValueId, next: ValueId| {
        defs.get(&next).is_some_and(|d| {
            matches!(d.kind, OpKind::Add | OpKind::Sub)
                && d.operands.len() == 2
                && d.operands[0] == arg
                && !inside.contains(&d.operands[1])
        })
    };
    let mut tainted: FxHashSet<ValueId> = op.block_args[1..]
        .iter()
        .zip(yielded)
        .filter(|&(&arg, &next)| !stepped(arg, next))
        .map(|(&arg, _)| arg)
        .collect();
    if tainted.is_empty() {
        return false;
    }
    // Values flow forward in program order; results of an op with regions
    // also take the taint of what its regions yield.
    loop {
        let mut changed = false;
        for inner in &all {
            let hit = inner.operands.iter().any(|v| tainted.contains(v))
                || inner.regions.iter().any(|r| {
                    r.ops.last().is_some_and(|y| y.kind == OpKind::Yield && y.operands.iter().any(|v| tainted.contains(v)))
                });
            if hit {
                for &r in inner.results.iter() {
                    changed |= tainted.insert(r);
                }
            }
        }
        if !changed {
            break;
        }
    }
    all.iter()
        .any(|o| o.kind.is_removable() && !o.kind.is_pure() && o.operands.iter().any(|v| tainted.contains(v)))
}

/// A loop in `ops` or in a region nested in them.
fn has_loop(ops: &[Op]) -> bool {
    let mut stack = vec![ops];
    while let Some(ops) = stack.pop() {
        for op in ops {
            if op.kind == OpKind::For {
                return true;
            }
            stack.extend(op.regions.iter().map(|r| &r.ops[..]));
        }
    }
    false
}

fn collect_consts(region: &Region, out: &mut FxHashMap<ValueId, i64>) {
    let mut stack = vec![region];
    while let Some(region) = stack.pop() {
        for op in &region.ops {
            if op.kind == OpKind::Constant {
                let value = match op.attr("value") {
                    Some(Attr::Int(v)) => Some(*v),
                    Some(Attr::Float(v)) if v.is_finite() => Some(v.trunc() as i64),
                    _ => None,
                };
                if let (Some(v), Some(&r)) = (value, op.results.first()) {
                    out.insert(r, v);
                }
            }
            stack.extend(op.regions.iter());
        }
    }
}

/// Ops in `ops` and in every region nested in them.
fn op_count(ops: &[Op]) -> usize {
    let mut n = 0;
    let mut stack = vec![ops];
    while let Some(ops) = stack.pop() {
        n += ops.len();
        stack.extend(ops.iter().flat_map(|op| op.regions.iter().map(|r| &r.ops[..])));
    }
    n
}

fn resolve(repl: &FxHashMap<ValueId, ValueId>, mut v: ValueId) -> ValueId {
    while let Some(&next) = repl.get(&v) {
        v = next;
    }
    v
}

fn substitute(region: &mut Region, repl: &FxHashMap<ValueId, ValueId>) {
    let mut stack = vec![region];
    while let Some(region) = stack.pop() {
        for op in &mut region.ops {
            for v in op.operands.iter_mut() {
                *v = resolve(repl, *v);
            }
            stack.extend(op.regions.iter_mut());
        }
    }
}

struct Unroller<'k> {
    kernel: &'k mut Kernel,
    lds: &'k mut FxHashMap<ValueId, u32>,
    consts: FxHashMap<ValueId, i64>,
    /// Results of unrolled loops -> the values they became.
    repl: FxHashMap<ValueId, ValueId>,
    /// Ops added so far, against `MAX_UNROLLED_OPS`.
    added: usize,
    auto: Option<Auto>,
    /// Ops the loops `auto` admitted added so far, against its total.
    auto_added: usize,
    auto_loops: usize,
    /// Induction variables of the loops the copies of those went into.
    hosts: FxHashSet<ValueId>,
}

/// A region being rebuilt: the ops still to visit, the ops kept, and the op
/// whose region `.1` is open below it.
struct Rebuild {
    input: std::vec::IntoIter<Op>,
    out: Vec<Op>,
    parked: Option<(Op, usize)>,
}

impl Rebuild {
    fn new(mut region: Region) -> Self {
        Rebuild { input: std::mem::take(&mut region.ops).into_iter(), out: Vec::new(), parked: None }
    }
}

/// An op being cloned: its source, the region being copied, what is left of
/// that region, the copies made so far, and the clone without its results.
struct Copy<'s> {
    src: &'s Op,
    region: usize,
    ops: std::slice::Iter<'s, Op>,
    built: Vec<Op>,
    head: Op,
}

impl Unroller<'_> {
    /// Each op after its regions, so a loop's body is unrolled before the
    /// loop around it copies it.
    fn region(&mut self, body: &mut Region) -> Result<(), CompileError> {
        let mut stack = vec![Rebuild::new(std::mem::take(body))];
        loop {
            let top = stack.last_mut().expect("the body stays open until the end");
            let mut nested = None;
            for op in top.input.by_ref() {
                if op.regions.is_empty() {
                    top.out.push(op);
                } else {
                    nested = Some(op);
                    break;
                }
            }
            let Some(mut op) = nested else {
                let done = stack.pop().expect("an open region");
                let region = Region { ops: done.out };
                // The innermost loop around the region `op` is in.
                let host = stack.len().checked_sub(1).and_then(|top| {
                    stack[..top].iter().rev().find_map(|r| r.parked.as_ref().filter(|(o, _)| o.kind == OpKind::For))
                });
                let host = host.map(|(o, _)| o.block_args[0]);
                let Some(parent) = stack.last_mut() else {
                    *body = region;
                    return Ok(());
                };
                let (mut op, index) = parent.parked.take().expect("a walked region has its op parked");
                op.regions[index] = region;
                if index + 1 < op.regions.len() {
                    let next = std::mem::take(&mut op.regions[index + 1]);
                    parent.parked = Some((op, index + 1));
                    stack.push(Rebuild::new(next));
                    continue;
                }
                if let Some((lb, step, trips)) = self.trip_count(&op) {
                    self.unroll(op, lb, step, trips, true, &mut parent.out)?;
                } else if let Some((lb, step, trips)) = self.auto_trip_count(&op) {
                    self.unroll(op, lb, step, trips, false, &mut parent.out)?;
                    self.hosts.extend(host);
                } else {
                    parent.out.push(op);
                }
                continue;
            };
            let first = std::mem::take(&mut op.regions[0]);
            top.parked = Some((op, 0));
            stack.push(Rebuild::new(first));
        }
    }

    /// Whether `region` holds a loop `region` would unroll first: one it
    /// takes as it stands.
    fn any_unrolls(&self, region: &Region) -> bool {
        let mut stack = vec![region];
        while let Some(region) = stack.pop() {
            for op in &region.ops {
                if self.trip_count(op).is_some() || self.auto_trip_count(op).is_some() {
                    return true;
                }
                stack.extend(op.regions.iter());
            }
        }
        false
    }

    /// `(lb, step, iterations)` of a loop to unroll. Bounds are i32 values.
    fn trip_count(&self, op: &Op) -> Option<(i64, i64, i64)> {
        if op.kind != OpKind::For || op.bool_attr("unroll") != Some(true) {
            return None;
        }
        self.constant_trips(op)
    }

    /// `(lb, step, iterations)` of a loop with constant bounds.
    fn constant_trips(&self, op: &Op) -> Option<(i64, i64, i64)> {
        let bound = |i: usize| op.operands.get(i).and_then(|v| self.consts.get(v)).map(|&c| c as i32 as i64);
        let (lb, ub, step) = (bound(0)?, bound(1)?, bound(2)?);
        if step == 0 {
            return None;
        }
        Some((lb, step, (ub - lb).div_euclid(step) - i64::from((ub - lb).rem_euclid(step) != 0 && step < 0)))
    }

    /// `trip_count` of a loop without the hint that `self.auto` admits.
    fn auto_trip_count(&self, op: &Op) -> Option<(i64, i64, i64)> {
        let auto = self.auto?;
        if op.kind != OpKind::For || op.bool_attr("unroll") == Some(true) {
            return None;
        }
        let (lb, step, trips) = self.constant_trips(op)?;
        let body = &op.regions[0].ops;
        if !(1..=auto.max_trips).contains(&trips) || has_loop(body) {
            return None;
        }
        let ops = trips as usize * (op_count(body) + 1);
        if ops > auto.max_ops || self.auto_added + ops > auto.max_total_ops || loads_wait_on_carried(op) {
            return None;
        }
        Some((lb, step, trips))
    }

    /// Unrolls `op`; only a `hinted` loop drops its trailing barrier.
    fn unroll(&mut self, op: Op, lb: i64, step: i64, trips: i64, hinted: bool, out: &mut Vec<Op>) -> Result<(), CompileError> {
        let body = &op.regions[0].ops;
        let trips = trips.max(0);
        // Each iteration adds its induction variable and a copy of the body.
        let per_trip = op_count(body) + 1;
        let budget = MAX_UNROLLED_OPS - self.added;
        if trips as u64 > (budget / per_trip) as u64 {
            return Err(CompileError::new(
                "ROCKE_DIRECT_UNSUPPORTED",
                format!("unrolling {trips} iterations of a {per_trip}-op loop body exceeds {MAX_UNROLLED_OPS} ops"),
            ));
        }
        self.added += trips as usize * per_trip;
        if !hinted {
            self.auto_added += trips as usize * per_trip;
            self.auto_loops += 1;
        }
        let elide = hinted && !matches!(op.attr("elide_trailing_barrier"), Some(Attr::Bool(false)));
        let trailing_sync = (elide && body.len() >= 2 && body[body.len() - 2].kind == OpKind::Sync).then(|| body.len() - 2);
        let iv = op.block_args[0];
        let iv_name = self.kernel.name_of(iv).to_string();
        let mut carried: Vec<ValueId> = op.operands[3..].iter().map(|&v| resolve(&self.repl, v)).collect();
        for i in 0..trips {
            let value = self.kernel.new_value(&iv_name, Type::Scalar(Scalar::I32));
            out.push(Op {
                kind: OpKind::Constant,
                results: [value].into(),
                operands: Default::default(),
                attrs: vec![
                    ("ity".into(), Attr::Str("i32".into())),
                    ("value".into(), Attr::Int((lb + i * step) as i32 as i64)),
                ],
                regions: Vec::new(),
                block_args: Vec::new(),
            });
            let mut map = FxHashMap::default();
            map.insert(iv, value);
            for (&arg, &v) in op.block_args[1..].iter().zip(&carried) {
                map.insert(arg, v);
            }
            for (k, inner) in body.iter().enumerate() {
                if inner.kind == OpKind::Yield {
                    carried = inner.operands.iter().map(|&v| self.mapped(&map, v)).collect();
                    continue;
                }
                if trailing_sync == Some(k) && i + 1 < trips {
                    continue;
                }
                let cloned = self.clone_op(inner, &mut map);
                out.push(cloned);
            }
        }
        for (&result, &v) in op.results.iter().zip(&carried) {
            self.repl.insert(result, v);
        }
        Ok(())
    }

    fn mapped(&self, map: &FxHashMap<ValueId, ValueId>, v: ValueId) -> ValueId {
        let v = resolve(&self.repl, v);
        map.get(&v).copied().unwrap_or(v)
    }

    fn fresh(&mut self, old: ValueId, map: &mut FxHashMap<ValueId, ValueId>) -> ValueId {
        let name = self.kernel.name_of(old).to_string();
        let ty = self.kernel.ty(old).clone();
        let new = self.kernel.new_value(&name, ty);
        map.insert(old, new);
        new
    }

    /// `op` with fresh results and block arguments, its operands mapped.
    /// Values are created in the order a recursive copy would: operands,
    /// block arguments, the regions' contents, then results.
    fn clone_op(&mut self, op: &Op, map: &mut FxHashMap<ValueId, ValueId>) -> Op {
        let head = self.clone_head(op, map);
        if op.regions.is_empty() {
            return self.with_results(head, op, map);
        }
        let mut stack = vec![Copy { src: op, region: 0, ops: op.regions[0].ops.iter(), built: Vec::new(), head }];
        loop {
            let top = stack.last_mut().expect("a copy in progress");
            if let Some(inner) = top.ops.next() {
                let head = self.clone_head(inner, map);
                if inner.regions.is_empty() {
                    let copy = self.with_results(head, inner, map);
                    stack.last_mut().expect("a copy in progress").built.push(copy);
                } else {
                    stack.push(Copy { src: inner, region: 0, ops: inner.regions[0].ops.iter(), built: Vec::new(), head });
                }
                continue;
            }
            let built = std::mem::take(&mut top.built);
            top.head.regions.push(Region { ops: built });
            top.region += 1;
            if let Some(next) = top.src.regions.get(top.region) {
                top.ops = next.ops.iter();
                continue;
            }
            let done = stack.pop().expect("a copy in progress");
            let copy = self.with_results(done.head, done.src, map);
            match stack.last_mut() {
                Some(parent) => parent.built.push(copy),
                None => return copy,
            }
        }
    }

    fn clone_head(&mut self, op: &Op, map: &mut FxHashMap<ValueId, ValueId>) -> Op {
        let operands = op.operands.iter().map(|&v| self.mapped(map, v)).collect();
        let block_args = op.block_args.iter().map(|&v| self.fresh(v, map)).collect();
        Op { kind: op.kind, results: Default::default(), operands, attrs: op.attrs.clone(), regions: Vec::new(), block_args }
    }

    fn with_results(&mut self, mut copy: Op, op: &Op, map: &mut FxHashMap<ValueId, ValueId>) -> Op {
        copy.results = op
            .results
            .iter()
            .map(|&v| {
                let new = self.fresh(v, map);
                if let Some(&offset) = self.lds.get(&v) {
                    self.lds.insert(new, offset);
                }
                new
            })
            .collect();
        copy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kernel(lb: i64, ub: i64, step: i64, elide: &str, sync: bool) -> Kernel {
        let sync = if sync { "        tile.sync (  ) { fence = s:\"compiler\" }\n" } else { "" };
        let text = format!(
            "ckdsl.ir v1\nkernel @k {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n    %Y : ptr<i32,global>\n  }}\n  \
             region @entry {{\n    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n    \
             %lb : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{lb} }}\n    \
             %ub : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{ub} }}\n    \
             %st : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{step} }}\n    \
             %r : i32 = scf.for ( %lb, %ub, %st, %tid ) {{ iter_args = l:[ {{ name = s:\"%acc\", type = s:\"i32\" }} ], \
             iv = s:\"%i\", iv_type = s:\"i32\", num_iter_args = i:1, unroll = b:true{elide} }}\n      region @body {{\n        \
             %n : i32 = arith.add ( %acc, %i )\n{sync}        scf.yield ( %n ) {{ num = i:1 }}\n      }}\n    \
             memref.global_store_typed ( %Y, %tid, %r ) {{ align = i:4, elem_type = s:\"i32\" }}\n    cf.return (  )\n  }}\n}}\n"
        );
        super::super::parse::parse(&text).expect("parses")
    }

    fn unrolled(lb: i64, ub: i64, step: i64, elide: &str, sync: bool) -> (Kernel, Vec<i64>, usize) {
        let mut k = kernel(lb, ub, step, elide, sync);
        let mut lds = FxHashMap::default();
        unroll_loops(&mut k, &mut lds).expect("unrolls");
        assert!(k.body.ops.iter().all(|op| op.kind != OpKind::For));
        let adds: Vec<&Op> = k.body.ops.iter().filter(|op| op.kind == OpKind::Add).collect();
        let mut consts = FxHashMap::default();
        collect_consts(&k.body, &mut consts);
        let ivs = adds.iter().map(|op| consts[&op.operands[1]]).collect();
        let syncs = k.body.ops.iter().filter(|op| op.kind == OpKind::Sync).count();
        (k, ivs, syncs)
    }

    #[test]
    fn trip_count_is_floor_division_like_the_reference() {
        assert_eq!(unrolled(0, 4, 1, "", false).1, [0, 1, 2, 3]);
        // (5 - 0) // 2 == 2: the reference's unrolled loop skips iv 4.
        assert_eq!(unrolled(0, 5, 2, "", false).1, [0, 2]);
        assert_eq!(unrolled(3, 3, 1, "", false).1, Vec::<i64>::new());
        // (0 - 10) // -2 == 5 and (1 - 10) // -2 == 4 under floor division.
        assert_eq!(unrolled(10, 0, -2, "", false).1, [10, 8, 6, 4, 2]);
        assert_eq!(unrolled(10, 1, -2, "", false).1, [10, 8, 6, 4]);
        assert_eq!(unrolled(10, 2, -3, "", false).1, [10, 7]);
    }

    #[test]
    fn results_chain_through_iterations_and_reach_later_uses() {
        let (k, _, _) = unrolled(0, 3, 1, "", false);
        let adds: Vec<&Op> = k.body.ops.iter().filter(|op| op.kind == OpKind::Add).collect();
        assert_eq!(adds[1].operands[0], adds[0].results[0]);
        assert_eq!(adds[2].operands[0], adds[1].results[0]);
        let store = k.body.ops.iter().find(|op| op.kind == OpKind::GlobalStore).unwrap();
        assert_eq!(store.operands[2], adds[2].results[0]);
        // Zero trips: the result is the initial value.
        let (k, _, _) = unrolled(0, 0, 1, "", false);
        let store = k.body.ops.iter().find(|op| op.kind == OpKind::GlobalStore).unwrap();
        assert_eq!(store.operands[2], store.operands[1]);
    }

    #[test]
    fn trailing_sync_survives_only_in_the_last_iteration() {
        assert_eq!(unrolled(0, 4, 1, "", true).2, 1);
        assert_eq!(unrolled(0, 4, 1, ", elide_trailing_barrier = b:true", true).2, 1);
        assert_eq!(unrolled(0, 4, 1, ", elide_trailing_barrier = b:false", true).2, 4);
    }

    #[test]
    fn unrolled_kernels_compile() {
        let mut text = String::from("ckdsl.ir v1\nkernel @k {\n  attrs { max_workgroup_size = i:64 }\n  params {\n    %Y : ptr<i32,global>\n    %n : i32\n  }\n  region @entry {\n    %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n    %c0 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }\n    %c1 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:1 }\n    %c3 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:3 }\n");
        // A runtime bound keeps its loop; the constant one inside unrolls per iteration.
        text += "    %r : i32 = scf.for ( %c0, %n, %c1, %tid ) { iter_args = l:[ { name = s:\"%a\", type = s:\"i32\" } ], iv = s:\"%i\", iv_type = s:\"i32\", num_iter_args = i:1, unroll = b:true }\n      region @body {\n";
        text += "        %s : i32 = scf.for ( %c0, %c3, %c1, %a ) { iter_args = l:[ { name = s:\"%b\", type = s:\"i32\" } ], iv = s:\"%j\", iv_type = s:\"i32\", num_iter_args = i:1, unroll = b:true }\n          region @body {\n            %m : i32 = arith.mul ( %b, %j )\n            %q : i32 = arith.add ( %m, %i )\n            scf.yield ( %q ) { num = i:1 }\n          }\n        scf.yield ( %s ) { num = i:1 }\n      }\n";
        text += "    memref.global_store_typed ( %Y, %tid, %r ) { align = i:4, elem_type = s:\"i32\" }\n    cf.return (  )\n  }\n}\n";
        let mut k = super::super::parse::parse(&text).expect("parses");
        let mut lds = FxHashMap::default();
        unroll_loops(&mut k, &mut lds).expect("unrolls");
        let outer = k.body.ops.iter().find(|op| op.kind == OpKind::For).expect("runtime-bound loop stays");
        assert!(outer.regions[0].ops.iter().all(|op| op.kind != OpKind::For));
        assert_eq!(outer.regions[0].ops.iter().filter(|op| op.kind == OpKind::Mul).count(), 3);
        super::super::compile(&text).unwrap_or_else(|e| panic!("{e}"));
    }

    #[test]
    fn huge_unrolls_are_refused() {
        let mut k = kernel(i32::MIN as i64, i32::MAX as i64, 1, "", false);
        let mut lds = FxHashMap::default();
        let error = unroll_loops(&mut k, &mut lds).expect_err("refused");
        assert_eq!(error.kind, "ROCKE_DIRECT_UNSUPPORTED");
        // Out-of-range constants are read as i32, as the bounds' type says.
        let mut k = kernel(0, (1i64 << 32) + 2, 1, "", false);
        unroll_loops(&mut k, &mut lds).expect("unrolls");
        assert_eq!(k.body.ops.iter().filter(|op| op.kind == OpKind::Add).count(), 2);
    }

    /// A kernel around one loop the IR leaves rolled: `trips` iterations
    /// of `body` (which reads `%acc` and the induction variable `%i`, and
    /// yields `%acc` as `%s`), with `extra` before the loop.
    fn rolled(trips: i64, extra: &str, body: &str) -> String {
        format!(
            "ckdsl.ir v1\nkernel @k {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n    %X : ptr<f32,global>\n    \
             %Y : ptr<f32,global>\n    %n : i32\n    %m : i32\n  }}\n  region @entry {{\n    \
             %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n    \
             %c0 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:0 }}\n    \
             %c1 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:1 }}\n    \
             %c4 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:4 }}\n    \
             %ub : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{trips} }}\n    \
             %big : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:2147483647 }}\n    \
             %z : f32 = arith.constant (  ) {{ ity = s:\"f32\", value = f:0.0 }}\n    \
             %rs : vec<i32x4> = tile.buffer_rsrc ( %X, %n )\n    \
             %ok : i1 = arith.cmp ( %tid, %m ) {{ pred = s:\"lt\" }}\n    \
             %base : i32 = arith.mul ( %tid, %c4 )\n{extra}    \
             %r : f32 = scf.for ( %c0, %ub, %c1, %z ) {{ iter_args = l:[ {{ name = s:\"%acc\", type = s:\"f32\" }} ], iv = s:\"%i\", \
             iv_type = s:\"i32\", num_iter_args = i:1, unroll = b:false }}\n      region @body {{\n{body}        \
             scf.yield ( %s ) {{ num = i:1 }}\n      }}\n    \
             memref.global_store_typed ( %Y, %tid, %r ) {{ align = i:4, elem_type = s:\"f32\" }}\n    cf.return (  )\n  }}\n}}\n"
        )
    }

    /// A masked buffer load at `base + i * stride`, as conv kernels address them.
    fn masked_load(base: &str, stride: i64) -> String {
        format!(
            "        %st : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{stride} }}\n        \
             %step : i32 = arith.mul ( %i, %st )\n        %a : i32 = arith.add ( {base}, %step )\n        \
             %o : i32 = arith.select ( %ok, %a, %big )\n        \
             %v : f32 = tile.buffer_load ( %rs, %o, %c0 ) {{ elem_type = s:\"f32\" }}\n        \
             %s : f32 = arith.fma ( %v, %v, %acc )\n"
        )
    }

    fn auto_unrolled(text: &str, auto: Auto) -> (Kernel, usize) {
        let mut k = super::super::parse::parse(text).unwrap_or_else(|e| panic!("{e}"));
        let mut lds = FxHashMap::default();
        let n = unroll_loops_with(&mut k, &mut lds, Some(auto)).expect("unrolls");
        (k, n)
    }

    fn count_kind(region: &Region, kind: OpKind) -> usize {
        let mut n = 0;
        let mut stack = vec![region];
        while let Some(region) = stack.pop() {
            n += region.ops.iter().filter(|op| op.kind == kind).count();
            stack.extend(region.ops.iter().flat_map(|op| op.regions.iter()));
        }
        n
    }

    #[test]
    fn small_constant_loops_unroll_without_a_hint() {
        let text = rolled(4, "", &masked_load("%base", 16));
        let (k, n) = auto_unrolled(&text, Auto::DEFAULT);
        assert_eq!((n, count_kind(&k.body, OpKind::For), count_kind(&k.body, OpKind::BufferLoadScalar)), (1, 0, 4));
        // Hints alone leave it rolled, as the reference lowering does.
        let mut k = super::super::parse::parse(&text).expect("parses");
        unroll_loops(&mut k, &mut FxHashMap::default()).expect("unrolls");
        assert_eq!(count_kind(&k.body, OpKind::For), 1);
    }

    #[test]
    fn loops_the_policy_does_not_admit_stay_rolled() {
        let load = masked_load("%base", 16);
        // More iterations than the policy takes.
        assert_eq!(auto_unrolled(&rolled(33, "", &load), Auto::DEFAULT).1, 0);
        // Copies past the per-loop size.
        let small = Auto { max_ops: 40, ..Auto::DEFAULT };
        assert_eq!(auto_unrolled(&rolled(8, "", &load), small).1, 0);
        // A runtime bound.
        let runtime = rolled(4, "", &load).replace("scf.for ( %c0, %ub,", "scf.for ( %c0, %m,");
        assert_eq!(auto_unrolled(&runtime, Auto::DEFAULT).1, 0);
        // A loop inside: neither it (runtime bound) nor the one around unrolls.
        let nested = "        %w : f32 = scf.for ( %c0, %m, %c1, %acc ) { iter_args = l:[ { name = s:\"%b\", type = s:\"f32\" } ], \
                      iv = s:\"%j\", iv_type = s:\"i32\", num_iter_args = i:1, unroll = b:false }\n          region @body {\n            \
                      %t : f32 = arith.fadd ( %b, %b )\n            scf.yield ( %t ) { num = i:1 }\n          }\n        \
                      %s : f32 = arith.fadd ( %w, %w )\n";
        let (k, n) = auto_unrolled(&rolled(4, "", nested), Auto::DEFAULT);
        assert_eq!((n, count_kind(&k.body, OpKind::For)), (0, 2));
    }

    #[test]
    fn loads_that_wait_on_the_iteration_before_stay_rolled() {
        // A search: each load's index is what the one before returned.
        let extra = "    %y0 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }\n";
        let chase = rolled(4, extra, "").replace(
            "iter_args = l:[ { name = s:\"%acc\", type = s:\"f32\" } ]",
            "iter_args = l:[ { name = s:\"%acc\", type = s:\"f32\" }, { name = s:\"%p\", type = s:\"i32\" } ]",
        );
        let chase = chase
            .replace("%r : f32 = scf.for ( %c0, %ub, %c1, %z )", "%r : f32, %q : i32 = scf.for ( %c0, %ub, %c1, %z, %y0 )")
            .replace("num_iter_args = i:1", "num_iter_args = i:2")
            .replace(
                "        scf.yield ( %s ) { num = i:1 }",
                "        %v : f32 = tile.buffer_load ( %rs, %p, %c0 ) { elem_type = s:\"f32\" }\n        \
                 %s : f32 = arith.fma ( %v, %v, %acc )\n        \
                 %np : i32 = memref.global_load_typed ( %X, %p ) { align = i:4, elem_type = s:\"i32\" }\n        \
                 scf.yield ( %s, %np ) { num = i:2 }",
            );
        assert_eq!(auto_unrolled(&chase, Auto::DEFAULT).1, 0);
        // A pointer stepped by a loop-invariant amount is an induction variable.
        let stepped = chase.replace(
            "%np : i32 = memref.global_load_typed ( %X, %p ) { align = i:4, elem_type = s:\"i32\" }",
            "%np : i32 = arith.add ( %p, %c4 )",
        );
        assert_eq!(auto_unrolled(&stepped, Auto::DEFAULT).1, 1);
    }

    #[test]
    fn unrolled_without_a_hint_every_barrier_stays() {
        let body = "        %s : f32 = arith.fadd ( %acc, %z )\n        tile.sync (  ) { fence = s:\"compiler\" }\n";
        let (k, n) = auto_unrolled(&rolled(4, "", body), Auto::DEFAULT);
        assert_eq!((n, count_kind(&k.body, OpKind::Sync)), (1, 4));
    }

    #[test]
    fn the_total_budget_leaves_later_loops_rolled() {
        let first = masked_load("%base", 16);
        let second = rolled(4, "", &first).replace(
            "    memref.global_store_typed ( %Y, %tid, %r )",
            "    %r2 : f32 = scf.for ( %c0, %ub, %c1, %r ) { iter_args = l:[ { name = s:\"%acc2\", type = s:\"f32\" } ], iv = s:\"%i2\", \
             iv_type = s:\"i32\", num_iter_args = i:1, unroll = b:false }\n      region @body {\n        \
             %s2 : f32 = arith.fadd ( %acc2, %acc2 )\n        scf.yield ( %s2 ) { num = i:1 }\n      }\n    \
             memref.global_store_typed ( %Y, %tid, %r2 )",
        );
        // The first loop's 4 copies of 7 ops plus induction variables fit; the second's do not.
        let (k, n) = auto_unrolled(&second, Auto { max_total_ops: 40, ..Auto::DEFAULT });
        assert_eq!((n, count_kind(&k.body, OpKind::For)), (1, 1));
        assert_eq!(auto_unrolled(&second, Auto::DEFAULT).1, 2);
    }

    /// `unroll_optimized` on `optimize`'s result, as `compile` runs it.
    fn late(text: &str) -> (Kernel, usize) {
        let mut k = super::super::parse::parse(text).unwrap_or_else(|e| panic!("{e}"));
        let (mut lds, _) = super::super::analysis::lds_layout(&k);
        let facts = opt::optimize(&mut k, &mut lds).unwrap_or_else(|e| panic!("{e}"));
        let (_, n) = unroll_optimized(&mut k, &mut lds, facts, Auto::DEFAULT, false).expect("unrolls");
        (k, n)
    }

    fn load_offsets(region: &Region) -> Vec<i64> {
        let mut out = Vec::new();
        let mut stack = vec![region];
        while let Some(region) = stack.pop() {
            out.extend(region.ops.iter().filter(|op| op.kind == OpKind::BufferLoadScalar).map(|op| op.int_attr("offset").unwrap_or(0)));
            stack.extend(region.ops.iter().flat_map(|op| op.regions.iter()));
        }
        out.sort_unstable();
        out
    }

    #[test]
    fn copies_fold_their_offsets_into_the_loads_through_the_select() {
        let (k, n) = late(&rolled(4, "", &masked_load("%base", 16)));
        assert_eq!((n, load_offsets(&k.body)), (1, vec![0, 16, 32, 48]));
        // The select of each copy takes the base itself, its other arm moved down by the offset.
        let mut arms = Vec::new();
        for op in k.body.ops.iter().filter(|op| op.kind == OpKind::Select) {
            if let Some(c) = k.body.ops.iter().find(|c| c.results.first() == Some(&op.operands[2])) {
                arms.push(c.int_attr("value").unwrap_or(-1));
            }
        }
        arms.sort_unstable();
        assert_eq!(arms, [2147483647 - 48, 2147483647 - 32, 2147483647 - 16, 2147483647]);
        // Past 12 bits, the multiple of 4096 stays in the base, as LLVM splits buffer offsets.
        let (k, _) = late(&rolled(4, "", &masked_load("%base", 2000)));
        assert_eq!(load_offsets(&k.body), [0, 6000 - 4096, 2000, 4000]);
    }

    #[test]
    fn a_base_not_known_non_negative_keeps_its_sums() {
        let extra = "    %neg : i32 = arith.sub ( %tid, %m )\n";
        let (k, n) = late(&rolled(4, extra, &masked_load("%neg", 16)));
        assert_eq!((n, load_offsets(&k.body)), (1, vec![0, 0, 0, 0]));
    }

    #[test]
    fn hoisted_scalars_go_back_into_the_loop_the_copies_went_into() {
        // `%u` reads only a parameter: code motion takes it out of the runtime loop.
        let inner = masked_load("%base", 16)
            .replace("        %st :", "        %u : i32 = arith.mul ( %m, %c3 )\n        %st :")
            .replace("%a : i32 = arith.add ( %base, %step )", "%su : i32 = arith.add ( %step, %u )\n        %a : i32 = arith.add ( %base, %su )");
        let extra = "    %c3 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:3 }\n";
        let text = rolled(4, extra, &inner).replace(
            "    %r : f32 = scf.for ( %c0, %ub, %c1, %z )",
            "    %outer : f32 = scf.for ( %c0, %n, %c1, %z ) { iter_args = l:[ { name = s:\"%oacc\", type = s:\"f32\" } ], iv = s:\"%j\", \
             iv_type = s:\"i32\", num_iter_args = i:1, unroll = b:false }\n      region @body {\n    \
             %r : f32 = scf.for ( %c0, %ub, %c1, %oacc )",
        );
        let text = text.replace(
            "    memref.global_store_typed ( %Y, %tid, %r )",
            "        scf.yield ( %r ) { num = i:1 }\n      }\n    memref.global_store_typed ( %Y, %tid, %outer )",
        );
        let (k, n) = late(&text);
        assert_eq!(n, 1);
        let outer = k.body.ops.iter().find(|op| op.kind == OpKind::For).expect("the runtime loop stays");
        let is_scale = |op: &Op| op.kind == OpKind::Mul && k.ty(op.results[0]) == &Type::Scalar(Scalar::I32);
        assert!(outer.regions[0].ops.iter().any(is_scale), "the uniform product is computed in the loop");
        assert!(!k.body.ops.iter().any(|op| is_scale(op) && op.operands.iter().any(|&v| k.name_of(v).contains("arg.m"))));
    }

    #[test]
    fn kernels_without_eligible_loops_come_out_unchanged() {
        let text = rolled(4, "", &masked_load("%base", 16)).replace("scf.for ( %c0, %ub,", "scf.for ( %c0, %m,");
        let mut k = super::super::parse::parse(&text).expect("parses");
        let (mut lds, _) = super::super::analysis::lds_layout(&k);
        let facts = opt::optimize(&mut k, &mut lds).expect("optimizes");
        let before = super::super::print::dump(&k);
        let (_, n) = unroll_optimized(&mut k, &mut lds, facts, Auto::DEFAULT, false).expect("unrolls");
        assert_eq!((n, super::super::print::dump(&k)), (0, before));
    }

    #[test]
    fn auto_unrolled_kernels_compile() {
        for stride in [16, 2000] {
            let text = rolled(8, "", &masked_load("%base", stride));
            let compiled = super::super::compile(&text).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(compiled.stats.unrolled, 1);
        }
    }

    /// An unrolled loop deep inside other regions, around deep regions of
    /// its own, on a 1 MiB thread stack.
    #[test]
    fn deep_nesting_on_a_small_stack() {
        let run = || {
            const DEPTH: usize = 50_000;
            let mut text = String::from(
                "ckdsl.ir v1\nkernel @deep {\n  attrs { max_workgroup_size = i:64 }\n  params {\n    %Y : ptr<i32,global>\n    %n : i32\n  }\n  \
                 region @entry {\n    %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n    %c : i1 = arith.cmp ( %tid, %n ) { pred = s:\"lt\" }\n    \
                 %c0 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }\n    %c2 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:2 }\n    \
                 %c1 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:1 }\n",
            );
            text += &"scf.if ( %c )\nregion @then {\n".repeat(DEPTH);
            text += "scf.for ( %c0, %c2, %c1 ) { iv = s:\"%i\", iv_type = s:\"i32\", unroll = b:true }\nregion @body {\n";
            text += &"scf.if ( %c )\nregion @then {\n".repeat(DEPTH);
            text += "memref.global_store_typed ( %Y, %tid, %i ) { align = i:4, elem_type = s:\"i32\" }\n";
            text += &"}\n".repeat(DEPTH);
            text += "scf.yield (  )\n}\n";
            text += &"}\n".repeat(DEPTH);
            text += "cf.return (  )\n}\n}\n";
            let mut k = super::super::parse::parse_with_threads(&text, 1).expect("parses");
            let mut lds = FxHashMap::default();
            unroll_loops(&mut k, &mut lds).expect("unrolls");
            // Six top-level ops, the outer ifs, and per trip the induction
            // variable, the inner ifs and the store.
            assert_eq!(op_count(&k.body.ops), 6 + DEPTH + 2 * (1 + DEPTH + 1));
        };
        std::thread::Builder::new().stack_size(1 << 20).spawn(run).expect("spawns").join().expect("no overflow");
    }
}
