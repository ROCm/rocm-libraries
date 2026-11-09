// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Repeated loads, done once on gfx950.
//!
//! A second global or buffer load of the same kind, operands, attributes and
//! type in the same straight-line run as the first (no barrier, fence, wait,
//! global write, flag op, scheduling directive or region between them) reads
//! the same bytes: nothing in the run writes them, and a write by another
//! lane, wave or agent with no synchronization between would race with both
//! loads. It goes, and its readers take the first's value, as LLVM's CSE
//! does. Plain loads carry no coherence bits (`nt` is part of the key). This
//! runs after the automatic unrolling, whose copies repeat such loads, as do
//! the paged-attention kernels' page-table reads, one per DMA row.
//!
//! A load of memory the kernel never writes is also reused past those, from
//! earlier in the same region or in a region around it (whose lanes include
//! the second load's), as LLVM's GVN does in COMGR's code: a pointer or
//! resource based on a `noalias readonly` kernel argument
//! (`alias::value_tags`, which drops every such fact when a write contradicts
//! the attributes). Nothing writes that memory while the kernel runs, so no
//! barrier, fence, flag op or other wave's write between the loads changes
//! what they read. `ROCKE_DIRECT_DEBUG=norocse` keeps such loads to the
//! in-run rule.

use crate::rir::alias::{value_tags, ANY};
use crate::rir::ast::*;
use std::collections::HashMap;

/// A load's kind, operands, attributes and result type.
type Key = (OpKind, Vec<ValueId>, u64, Type);

fn attr_hash(attrs: &[(Text, Attr)]) -> Option<u64> {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for (name, attr) in attrs {
        name.hash(&mut h);
        match attr {
            Attr::Int(v) => v.hash(&mut h),
            Attr::Bool(v) => v.hash(&mut h),
            Attr::Str(v) => v.hash(&mut h),
            _ => return None,
        }
    }
    Some(h.finish())
}

/// Whether `op` ends a run: it may write global memory, order other waves' or
/// the memory system's accesses, or constrain the schedule.
fn breaks_run(op: &Op) -> bool {
    op.kind.writes_global()
        || matches!(
            op.kind,
            OpKind::Sync
                | OpKind::SyncLdsOnly
                | OpKind::BarrierBare
                | OpKind::Fence
                | OpKind::WaitCnt
                | OpKind::WaitAsyncCnt
                | OpKind::SchedBarrier
                | OpKind::SchedGroupBarrier
                | OpKind::IglpOpt
                | OpKind::InlineAsmIdentity
                | OpKind::FlagWaitEq
                | OpKind::FlagStore
        )
}

struct Reuse<'a> {
    kernel: &'a Kernel,
    tags: Vec<u16>,
    across: bool,
    /// The loads of the current run by key.
    run: HashMap<Key, ValueId>,
    /// With `across`, the loads of memory the kernel never writes by key,
    /// one table per open region, innermost last.
    scopes: Vec<HashMap<Key, ValueId>>,
    renames: HashMap<ValueId, ValueId>,
}

impl Reuse<'_> {
    /// The key of a plain load, and whether the memory it reads is never written.
    fn key(&self, op: &Op) -> Option<(Key, bool)> {
        if !matches!(op.kind, OpKind::GLoad | OpKind::BufferLoadVec | OpKind::BufferLoadScalar) || op.results.len() != 1 {
            return None;
        }
        let tag = self.tags.get(op.operands.first()?.0 as usize).copied().unwrap_or(ANY);
        let invariant = tag != ANY && self.kernel.params.get(usize::from(tag) - 1).is_some_and(|p| p.readonly);
        let ty = self.kernel.ty(op.results[0]).clone();
        Some(((op.kind, op.operands.iter().copied().collect(), attr_hash(&op.attrs)?, ty), invariant))
    }

    /// Walks a region in order: renames the operands of removed loads' readers and removes the loads.
    fn region(&mut self, region: &mut Region) -> usize {
        let mut removed = 0;
        let old = std::mem::take(&mut region.ops);
        let mut out = Vec::with_capacity(old.len());
        for mut op in old {
            for v in op.operands.iter_mut() {
                if let Some(&r) = self.renames.get(v) {
                    *v = r;
                }
            }
            if !op.regions.is_empty() {
                // The lanes change in there and back: each side starts a run.
                self.run.clear();
                for inner in op.regions.iter_mut() {
                    self.scopes.push(HashMap::new());
                    removed += self.region(inner);
                    self.scopes.pop();
                    self.run.clear();
                }
                out.push(op);
                continue;
            }
            if breaks_run(&op) {
                self.run.clear();
            }
            if let Some((key, invariant)) = self.key(&op) {
                let earlier = self.run.get(&key).copied().or_else(|| {
                    (self.across && invariant).then(|| self.scopes.iter().rev().find_map(|s| s.get(&key).copied())).flatten()
                });
                if let Some(first) = earlier {
                    self.renames.insert(op.results[0], first);
                    removed += 1;
                    continue;
                }
                if self.across && invariant {
                    self.scopes.last_mut().expect("an open region").insert(key.clone(), op.results[0]);
                }
                self.run.insert(key, op.results[0]);
            }
            out.push(op);
        }
        region.ops = out;
        removed
    }
}

/// Removes `kernel`'s repeated loads within runs; with `across` (gfx950's
/// default), also those of read-only memory past barriers, waits, global
/// writes and into regions. Returns how many. `ROCKE_DIRECT_DEBUG=noroload`
/// skips it.
pub fn reuse_loads(kernel: &mut Kernel, across: bool) -> usize {
    if crate::rir::debug_flag("noroload") {
        return 0;
    }
    let tags = value_tags(kernel);
    let mut body = std::mem::take(&mut kernel.body);
    let removed = {
        let mut reuse = Reuse { kernel, tags, across, run: HashMap::new(), scopes: vec![HashMap::new()], renames: HashMap::new() };
        reuse.region(&mut body)
    };
    kernel.body = body;
    removed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rir;
    use crate::target::Target;

    /// Two loads of B[8 tid] and of A[8 tid], with `between` between the pairs and `wrap` around the second pair
    /// and its stores; every load is stored to its own place in D after the loads.
    fn kernel(b_attrs: &str, between: &str, wrap: (&str, &str)) -> String {
        let consts: String = [0, 8, 16, 32, 48]
            .iter()
            .map(|v| format!("    %c{v} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{v} }}\n"))
            .collect();
        format!(
            "ckdsl.ir v1\nkernel @k {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n    \
             %A : ptr<bf16,global> {{ noalias = b:true }}\n    %B : ptr<bf16,global> {b_attrs}\n    \
             %D : ptr<bf16,global> {{ noalias = b:true }}\n    %A_bytes : i32\n    %B_bytes : i32\n    %D_bytes : i32\n    %n : i32\n  }}\n  \
             region @entry {{\n    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n{consts}    %o : i32 = arith.mul ( %tid, %c8 )\n    \
             %c : i1 = arith.cmp ( %tid, %n ) {{ pred = s:\"lt\" }}\n    \
             %o2 : i32 = arith.add ( %o, %c16 )\n    %o3 : i32 = arith.add ( %o, %c32 )\n    %o4 : i32 = arith.add ( %o, %c48 )\n    \
             %ra : vec<i32x4> = tile.buffer_rsrc ( %A, %A_bytes )\n    %rb : vec<i32x4> = tile.buffer_rsrc ( %B, %B_bytes )\n    \
             %rd : vec<i32x4> = tile.buffer_rsrc ( %D, %D_bytes )\n    \
             %b1 : vec<bf16x4> = tile.buffer_load_vN_bf16 ( %rb, %o, %c0 ) {{ dwords = i:2 }}\n    \
             %a1 : vec<bf16x4> = tile.buffer_load_vN_bf16 ( %ra, %o, %c0 ) {{ dwords = i:2 }}\n{between}{}    \
             %b2 : vec<bf16x4> = tile.buffer_load_vN_bf16 ( %rb, %o, %c0 ) {{ dwords = i:2 }}\n    \
             %a2 : vec<bf16x4> = tile.buffer_load_vN_bf16 ( %ra, %o, %c0 ) {{ dwords = i:2 }}\n    \
             tile.buffer_store_vN_bf16 ( %rd, %o2, %c0, %b2 ) {{ dwords = i:2 }}\n    \
             tile.buffer_store_vN_bf16 ( %rd, %o3, %c0, %a2 ) {{ dwords = i:2 }}\n{}    \
             tile.buffer_store_vN_bf16 ( %rd, %o, %c0, %b1 ) {{ dwords = i:2 }}\n    \
             tile.buffer_store_vN_bf16 ( %rd, %o4, %c0, %a1 ) {{ dwords = i:2 }}\n    cf.return (  )\n  }}\n}}\n",
            wrap.0, wrap.1
        )
    }

    /// A load of B[8 tid] in `scf.if (tid < n)`, stored there, then a second one after that region (in
    /// `scf.if (tid < 48)` when `sibling`), stored to another place.
    fn first_in_a_region(sibling: bool) -> String {
        let (open, close) = if sibling { ("    scf.if ( %c2 )\n      region @then {\n", "      }\n") } else { ("", "") };
        format!(
            "ckdsl.ir v1\nkernel @k {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n    \
             %B : ptr<bf16,global> {RO}\n    %D : ptr<bf16,global> {{ noalias = b:true }}\n    \
             %B_bytes : i32\n    %D_bytes : i32\n    %n : i32\n  }}\n  \
             region @entry {{\n    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n    \
             %c0 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:0 }}\n    \
             %c8 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:8 }}\n    \
             %c48 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:48 }}\n    \
             %o : i32 = arith.mul ( %tid, %c8 )\n    %o2 : i32 = arith.add ( %o, %c48 )\n    \
             %c : i1 = arith.cmp ( %tid, %n ) {{ pred = s:\"lt\" }}\n    \
             %c2 : i1 = arith.cmp ( %tid, %c48 ) {{ pred = s:\"lt\" }}\n    \
             %rb : vec<i32x4> = tile.buffer_rsrc ( %B, %B_bytes )\n    %rd : vec<i32x4> = tile.buffer_rsrc ( %D, %D_bytes )\n    \
             scf.if ( %c )\n      region @then {{\n    \
             %b1 : vec<bf16x4> = tile.buffer_load_vN_bf16 ( %rb, %o, %c0 ) {{ dwords = i:2 }}\n    \
             tile.buffer_store_vN_bf16 ( %rd, %o, %c0, %b1 ) {{ dwords = i:2 }}\n      }}\n{open}    \
             %b2 : vec<bf16x4> = tile.buffer_load_vN_bf16 ( %rb, %o, %c0 ) {{ dwords = i:2 }}\n    \
             tile.buffer_store_vN_bf16 ( %rd, %o2, %c0, %b2 ) {{ dwords = i:2 }}\n{close}    cf.return (  )\n  }}\n}}\n"
        )
    }

    const RO: &str = "{ noalias = b:true, readonly = b:true }";
    const SYNC: &str = "    tile.sync (  )\n";
    const STORE: &str = "    tile.buffer_store_vN_bf16 ( %rd, %o4, %c0, %a1 ) { dwords = i:2 }\n";
    const IF: (&str, &str) = ("    scf.if ( %c )\n      region @then {\n", "      }\n");

    /// The buffer loads `optimize` leaves, then those `reuse_loads` leaves.
    fn loads(text: &str, across: bool) -> (usize, usize) {
        let mut k = rir::parse::parse(text).unwrap_or_else(|e| panic!("{e}"));
        k.wave = 64;
        let (mut lds, _) = rir::analysis::lds_layout(&k);
        rir::opt::optimize(&mut k, &mut lds).unwrap_or_else(|e| panic!("{e}"));
        let count = |k: &Kernel| {
            let mut n = 0;
            rir::analysis::preorder(&k.body, |op| n += usize::from(op.kind == OpKind::BufferLoadVec));
            n
        };
        let before = count(&k);
        reuse_loads(&mut k, across);
        (before, count(&k))
    }

    #[test]
    fn a_repeated_load_in_one_run_goes() {
        for b in [RO, "{ noalias = b:true }", "{ readonly = b:true }", ""] {
            assert_eq!(loads(&kernel(b, "", ("", "")), false), (4, 2), "{b}");
        }
    }

    #[test]
    fn a_global_write_between_keeps_both() {
        assert_eq!(loads(&kernel(RO, STORE, ("", "")), false), (4, 4));
    }

    #[test]
    fn a_barrier_keeps_both_unless_asked_to_reuse_read_only_memory_across_it() {
        assert_eq!(loads(&kernel(RO, SYNC, ("", "")), false), (4, 4));
        // B's second load goes; A, not readonly, keeps both.
        assert_eq!(loads(&kernel(RO, SYNC, ("", "")), true), (4, 3));
        assert_eq!(loads(&kernel("{ noalias = b:true }", SYNC, ("", "")), true), (4, 4));
        assert_eq!(loads(&kernel("{ readonly = b:true }", SYNC, ("", "")), true), (4, 4));
    }

    #[test]
    fn a_region_takes_the_value_from_around_it_only_across() {
        assert_eq!(loads(&kernel(RO, "", IF), false), (4, 4));
        assert_eq!(loads(&kernel(RO, "", IF), true), (4, 3));
    }

    #[test]
    fn a_write_through_the_argument_keeps_both_past_a_barrier() {
        let to_d = ("", "    tile.buffer_store_vN_bf16 ( %rd, %o4, %c0, %a1 ) { dwords = i:2 }\n");
        let to_b = ("", "    tile.buffer_store_vN_bf16 ( %rb, %o4, %c0, %a1 ) { dwords = i:2 }\n");
        assert_eq!(loads(&kernel(RO, SYNC, to_d), true), (4, 3));
        assert_eq!(loads(&kernel(RO, SYNC, to_b), true), (4, 4));
    }

    #[test]
    fn a_region_gives_its_value_neither_to_the_ops_after_it_nor_to_another_region() {
        for sibling in [false, true] {
            assert_eq!(loads(&first_in_a_region(sibling), true), (2, 2), "sibling {sibling}");
        }
    }

    #[test]
    fn the_compile_keeps_one_load_of_each_address() {
        let listing = rir::compile_any_target(&kernel(RO, "", ("", "")), Target::Gfx950).unwrap_or_else(|e| panic!("{e}")).listing();
        let n = listing.lines().filter(|l| l.split_whitespace().next() == Some("buffer_load_dwordx2")).count();
        assert_eq!(n, 2, "{listing}");
    }

    #[test]
    fn the_compile_reuses_read_only_loads_past_a_barrier() {
        let count = |text: &str| {
            let listing = rir::compile_any_target(text, Target::Gfx950).unwrap_or_else(|e| panic!("{e}")).listing();
            listing.lines().filter(|l| l.split_whitespace().next() == Some("buffer_load_dwordx2")).count()
        };
        // B's second load goes; A, only noalias, keeps both.
        assert_eq!(count(&kernel(RO, SYNC, ("", ""))), 3);
        assert_eq!(count(&kernel("{ noalias = b:true }", SYNC, ("", ""))), 4);
    }
}
