// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Issue order of global loads within a block.

use super::ast::{Op, OpKind, Region, ValueId};
use super::fx::{FxHashMap, FxHashSet};

/// Global loads complete in issue order, so a load whose result is needed
/// late (a prefetch) delays the first use of every load issued after it.
/// Within each block, such a load moves below the later loads whose results
/// are needed before its own and below the ops that first read them (which
/// wait for them), so those waits do not include it. No load moves past a
/// matrix op (the work that hides its latency), a global write, a barrier or
/// fence, an explicit wait or scheduling hint, inline assembly, or an op with
/// regions.
pub fn order_global_loads(body: &mut Region) {
    // Each block's order depends only on its own ops, so any visiting order
    // gives the same result; an explicit stack bounds the native stack.
    let mut stack = vec![body];
    while let Some(region) = stack.pop() {
        order_block(region);
        for op in &mut region.ops {
            stack.extend(op.regions.iter_mut());
        }
    }
}

fn order_block(region: &mut Region) {
    let is_load = |op: &Op| {
        matches!(op.kind, OpKind::GlobalLoad | OpKind::GlobalLoadVec | OpKind::BufferLoadVec | OpKind::GLoad)
            && op.attr("ordering").is_none()
    };
    let ops = &region.ops;
    let n = ops.len();
    if ops.iter().filter(|op| is_load(op)).count() < 2 {
        return;
    }
    fn uses(op: &Op, f: &mut impl FnMut(ValueId)) {
        op.operands.iter().for_each(|&v| f(v));
        let mut stack: Vec<&Region> = op.regions.iter().collect();
        while let Some(region) = stack.pop() {
            for o in &region.ops {
                o.operands.iter().for_each(|&v| f(v));
                stack.extend(o.regions.iter());
            }
        }
    }
    // Looked up only, so room for an operand per op changes nothing else.
    let mut first_use: FxHashMap<ValueId, usize> = FxHashMap::with_capacity_and_hasher(n, Default::default());
    for (i, op) in ops.iter().enumerate() {
        uses(op, &mut |v| {
            first_use.entry(v).or_insert(i);
        });
    }
    let needed_at = |op: &Op| op.results.iter().map(|r| first_use.get(r).copied().unwrap_or(n)).min().unwrap_or(n);
    // Register renames: reading their result is not yet a wait.
    let free = |op: &Op| matches!(op.kind, OpKind::Extract | OpKind::Bitcast | OpKind::VecBitcast);
    // Sort keys: an op stays at (its index, 0); a moved load goes right after
    // the op it follows, in program order among loads placed there.
    let mut keys: Vec<(usize, usize)> = (0..n).map(|i| (i, 0)).collect();
    let mut passed = FxHashSet::default();
    for i in 0..n {
        if !is_load(&ops[i]) {
            continue;
        }
        let need = needed_at(&ops[i]);
        let mut after = None;
        passed.clear();
        for (j, o) in ops.iter().enumerate().take(need).skip(i + 1) {
            if is_load(o) {
                if needed_at(o) < need {
                    passed.extend(o.results.iter().copied());
                    after = Some(j);
                }
                continue;
            }
            if !o.regions.is_empty()
                || o.kind.writes_global()
                || matches!(o.kind, OpKind::Mma | OpKind::Sync | OpKind::SyncLdsOnly | OpKind::InlineAsmIdentity)
                || matches!(
                    o.kind,
                    OpKind::BarrierBare | OpKind::WaitCnt | OpKind::SchedBarrier | OpKind::SchedGroupBarrier | OpKind::IglpOpt | OpKind::SetPrio
                )
                || matches!(o.kind, OpKind::GlobalLoad | OpKind::GlobalLoadVec | OpKind::GLoad)
            {
                break;
            }
            if o.operands.iter().any(|v| passed.contains(v)) {
                if free(o) {
                    passed.extend(o.results.iter().copied());
                } else {
                    after = Some(j);
                }
            }
        }
        if let Some(j) = after {
            keys[i] = (j, i + 1);
        }
    }
    if keys.iter().enumerate().all(|(i, &k)| k == (i, 0)) {
        return;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| keys[i]);
    let mut taken: Vec<Option<Op>> = std::mem::take(&mut region.ops).into_iter().map(Some).collect();
    region.ops = order.into_iter().map(|i| taken[i].take().expect("each op placed once")).collect();
}
