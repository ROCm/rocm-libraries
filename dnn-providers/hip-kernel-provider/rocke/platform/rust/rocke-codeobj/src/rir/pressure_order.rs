// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Pressure-capped list scheduling of a block (lever `pressorder`, gfx950).
//!
//! `post::BlockDag::order` ranks the ready instructions of a window by when they can start; a window wide enough to
//! hide the latencies of an attention loop (LDS reads ahead of their MFMAs, the next tile's loads) holds more values
//! than the registers do, and the narrower ones give up the prefetch. This is the same walk with one more rule: it
//! tracks the VGPR dwords live as it emits, and an instruction that would take the count past the cap goes only after
//! those that would not; when every ready instruction would, the one that frees the most goes. The order is a
//! topological order of the block's dependence graph, as every order of `BlockDag` is: only register pressure and
//! issue order change, never a value.

use super::fx::FxHashMap;
use super::mir::{MInst, F_LDS, F_VMEM_LOAD};
use super::post::BlockDag;
use super::regalloc::UnitLists;

/// The versions (one per definition, plus one per unit live into the block) of the VGPR units a block touches, and
/// which instruction reads and writes which.
struct Versions {
    /// Per instruction, the versions it reads (once each) and writes.
    reads: Vec<Vec<u32>>,
    writes: Vec<Vec<u32>>,
    /// Per version, its readers in the block, and whether it is live out of it.
    readers: Vec<u32>,
    live_out: Vec<bool>,
    /// Versions live into the block.
    live_in: Vec<u32>,
}

impl Versions {
    /// `first` is the linear index of the block's first instruction in `lists`.
    fn new(lists: &UnitLists, first: usize, n: usize, live_in: &[u32], live_out: &[u32]) -> Versions {
        let mut cur: FxHashMap<u32, u32> = FxHashMap::default();
        let mut v = Versions { reads: vec![Vec::new(); n], writes: vec![Vec::new(); n], readers: Vec::new(), live_out: Vec::new(), live_in: Vec::new() };
        let fresh = |v: &mut Versions| {
            v.readers.push(0);
            v.live_out.push(false);
            v.readers.len() as u32 - 1
        };
        for &x in live_in.iter().filter(|&&x| x & 1 == 0) {
            let id = fresh(&mut v);
            v.live_in.push(id);
            cur.insert(x >> 1, id);
        }
        for k in 0..n {
            let (defs, uses) = lists.of(first + k);
            for &x in uses.iter().filter(|&&x| x & 1 == 0) {
                let id = match cur.get(&(x >> 1)) {
                    Some(&id) => id,
                    None => {
                        let id = fresh(&mut v);
                        v.live_in.push(id);
                        cur.insert(x >> 1, id);
                        id
                    }
                };
                if !v.reads[k].contains(&id) {
                    v.reads[k].push(id);
                    v.readers[id as usize] += 1;
                }
            }
            for &x in defs.iter().filter(|&&x| x & 1 == 0) {
                let id = fresh(&mut v);
                v.writes[k].push(id);
                cur.insert(x >> 1, id);
            }
        }
        for &x in live_out.iter().filter(|&&x| x & 1 == 0) {
            if let Some(&id) = cur.get(&(x >> 1)) {
                v.live_out[id as usize] = true;
            }
        }
        v
    }
}

/// The order of the block `dag` is of, with at most `cap` VGPR dwords live where the choice allows, among the ready
/// instructions at most `window` places after the first unscheduled. `lists`, `first`, `live_in` and `live_out` are
/// the block's units (`regalloc::Liveness`). A load that adds to the count must leave `reserve` dwords of room under the
/// cap, for the accumulators and results the instructions after it still have to make.
pub(super) fn order(dag: &BlockDag, insts: &[MInst], lists: &UnitLists, first: usize, live_in: &[u32], live_out: &[u32], cap: u32, reserve: u32, window: usize) -> Vec<u32> {
    let n = dag.preds.len();
    let ver = Versions::new(lists, first, n, live_in, live_out);
    let mut remaining = ver.readers.clone();
    // Dwords live now: the units live into the block that something still needs.
    let mut live = ver.live_in.iter().filter(|&&id| ver.readers[id as usize] > 0 || ver.live_out[id as usize]).count() as i64;
    let cap = i64::from(cap);
    // What issuing `i` does to the count: the values it makes that something needs, less those it reads last.
    let delta = |i: usize, remaining: &[u32]| -> i64 {
        let made = ver.writes[i].iter().filter(|&&id| ver.readers[id as usize] > 0 || ver.live_out[id as usize]).count() as i64;
        let freed = ver.reads[i].iter().filter(|&&id| remaining[id as usize] == 1 && !ver.live_out[id as usize]).count() as i64;
        made - freed
    };
    let prefetch: Vec<bool> = insts.iter().map(|x| x.op.flags() & (F_VMEM_LOAD | F_LDS) != 0 && !x.defs.is_empty()).collect();
    let mut preds = dag.preds.clone();
    let mut key: Vec<u64> = dag.height.iter().map(|&h| u64::from(!h)).collect();
    let mut done = vec![false; n];
    let mut ready = vec![0u64; n.div_ceil(64)];
    for (i, _) in preds.iter().enumerate().filter(|(_, &p)| p == 0) {
        ready[i / 64] |= 1 << (i % 64);
    }
    let pipe = !dag.pipe.is_empty();
    let mut order = Vec::with_capacity(n);
    let (mut peak, start_live) = (live, live);
    let (mut cycle, mut firstu, mut free) = (0u32, 0usize, 0u32);
    while order.len() < n {
        while done[firstu] {
            firstu += 1;
        }
        let now = u64::from(cycle) << 32;
        let at = |k: u64| k.max(now | (k & 0xffff_ffff));
        let rank = |i: usize| {
            if pipe { dag.piped(i, at(key[i]), free, cycle) } else { at(key[i]) }
        };
        // The best ready instruction under the cap, and failing that the one that frees the most.
        let mut under: Option<(u64, usize)> = None;
        let mut over: Option<(i64, u64, usize)> = None;
        let mut consider = |i: usize, remaining: &[u32]| {
            let (d, k) = (delta(i, remaining), rank(i));
            let room = if d > 0 && prefetch[i] { cap - i64::from(reserve) } else { cap };
            if live + d <= room {
                if under.is_none_or(|(b, _)| k < b) {
                    under = Some((k, i));
                }
            } else if over.is_none_or(|(bd, bk, _)| (d, k) < (bd, bk)) {
                over = Some((d, k, i));
            }
        };
        consider(firstu, &remaining);
        let (begin, end) = (firstu + 1, n.min(firstu + window));
        for w in begin / 64..end.div_ceil(64) {
            let mut bits = ready[w];
            if w == begin / 64 {
                bits &= !0u64 << (begin % 64);
            }
            if end < (w + 1) * 64 {
                bits &= (1u64 << (end % 64)) - 1;
            }
            while bits != 0 {
                let i = w * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                consider(i, &remaining);
            }
        }
        let (pick, best) = match (under, over) {
            (Some((k, i)), _) => (i, k),
            (None, Some((_, k, i))) => (i, k),
            (None, None) => unreachable!("the first unscheduled instruction is a candidate"),
        };
        let start = (best >> 32) as u32;
        let i = pick;
        done[i] = true;
        ready[i / 64] &= !(1 << (i % 64));
        order.push(i as u32);
        cycle = start + u32::from(dag.issues[i]);
        if pipe && dag.pipe[i] != 0 {
            free = start + u32::from(dag.pipe[i]);
        }
        live += ver.writes[i].iter().filter(|&&id| ver.readers[id as usize] > 0 || ver.live_out[id as usize]).count() as i64;
        peak = peak.max(live);
        for &id in &ver.reads[i] {
            remaining[id as usize] -= 1;
            if remaining[id as usize] == 0 && !ver.live_out[id as usize] {
                live -= 1;
            }
        }
        for &(j, latency) in &dag.succs[dag.succ_start[i] as usize..dag.succ_start[i + 1] as usize] {
            let j = j as usize;
            preds[j] -= 1;
            key[j] = key[j].max(u64::from(start + latency) << 32 | (key[j] & 0xffff_ffff));
            if preds[j] == 0 && !done[j] {
                ready[j / 64] |= 1 << (j % 64);
            }
        }
    }
    if super::debug_flag("trace") {
        eprintln!("trace: guard block of {n}: live in {start_live}, simulated peak {peak}, cap {cap}");
    }
    order
}

#[cfg(test)]
mod tests {
    use super::super::mir::*;
    use super::super::post::Dags;
    use super::super::regalloc::Liveness;
    use super::order;

    /// A block's order must be a permutation that keeps every dependence, whatever the cap.
    #[test]
    fn the_order_is_a_dependence_order_for_any_cap() {
        for seed in 0..30u64 {
            let f: MFunc = super::super::spill::tests::random_kernel(seed, 20 + seed as usize, 5 + seed as usize % 20);
            let live = Liveness::new(&f).expect("analyzes");
            let lists = live.unit_lists(&f);
            let dags = Dags::new(&f);
            for cap in [1, 20, 60, 1000] {
                for b in 0..f.blocks.len() {
                    let (live_in, live_out) = live.block_boundary(&lists, b);
                    let got = order(dags.dag(b), &f.blocks[b].insts, &lists, live.first_inst(b), &live_in, &live_out, cap, 8, 64);
                    let n = f.blocks[b].insts.len();
                    let mut at = vec![usize::MAX; n];
                    for (pos, &i) in got.iter().enumerate() {
                        assert_eq!(at[i as usize], usize::MAX, "seed {seed} block {b}: {i} twice");
                        at[i as usize] = pos;
                    }
                    assert_eq!(got.len(), n, "seed {seed} block {b}");
                    for i in 0..n {
                        for j in dags.successors(b, i) {
                            assert!(at[i] < at[j], "seed {seed} cap {cap} block {b}: {j} before its predecessor {i}");
                        }
                    }
                }
            }
        }
    }

    /// With room to spare the walk takes the same order as the plain one in a window as wide.
    #[test]
    fn a_cap_that_never_binds_takes_the_plain_order() {
        for seed in 0..30u64 {
            let f: MFunc = super::super::spill::tests::random_kernel(seed, 20 + seed as usize, 5 + seed as usize % 20);
            let live = Liveness::new(&f).expect("analyzes");
            let lists = live.unit_lists(&f);
            let dags = Dags::new(&f);
            for b in 0..f.blocks.len() {
                let (live_in, live_out) = live.block_boundary(&lists, b);
                let got = order(dags.dag(b), &f.blocks[b].insts, &lists, live.first_inst(b), &live_in, &live_out, 100_000, 0, 64);
                assert_eq!(got, dags.block_order(b, 64), "seed {seed} block {b}");
            }
        }
    }
}
