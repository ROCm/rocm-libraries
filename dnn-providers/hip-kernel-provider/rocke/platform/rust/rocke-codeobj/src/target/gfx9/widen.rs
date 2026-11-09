// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Fewer, wider global stores on gfx950, as LLVM gives them in COMGR's code.
//! The attention epilogues store each 16-bit element of O in an `scf.if` of
//! its own on one bounds condition: SimplifyCFG makes one branch of them, and
//! the load-store vectorizer stores four elements at a time at immediate
//! offsets from one address.
//!
//! `fuse_ifs`: two `scf.if` of one condition without results, with only pure
//! ops without regions between them, become one. The ops between that no
//! later op reads go into the fused body ahead of the second body; the others
//! go before the first `scf.if`. Both bodies ran under the lanes of that
//! condition, and the fused body runs both under the same lanes.
//!
//! `combine_global_stores`: in a run of a region's ops with no global load,
//! no other global write, barrier, fence, wait, scheduling directive or nested
//! region, the 16- and 32-bit scalar stores through one pointer whose byte
//! addresses are one sum of values plus a constant form a class: the pointer
//! adds a non-wrapping i32 sum, sign-extended and scaled by a constant, whose
//! terms are the same values. Contiguous 4, 8 or 16 bytes of a class become
//! one store of a vector where the last of them was, and every store of the
//! class addresses through the pointer of its first at an immediate offset
//! when the field holds it. A class keeps one `nt` bit; stores of two classes
//! in one run stay apart, as do stores of one class to overlapping bytes.

use crate::rir::ast::*;
use crate::rir::fx::{FxHashMap, FxHashSet};
use std::collections::HashSet;

/// Byte offsets the immediate field of a global access takes.
const OFFSETS: std::ops::Range<i64> = -4096..4096;

/// Bytes a merged store may have: four dwords, two or one.
const WIDTHS: [i64; 3] = [16, 8, 4];

/// The condition of an `scf.if` without results or else.
fn plain_if(op: &Op) -> Option<ValueId> {
    (op.kind == OpKind::If && op.results.is_empty() && op.regions.len() == 1 && op.operands.len() == 1)
        .then(|| op.operands[0])
}

/// Every value the ops (and the regions in them) read.
fn reads(ops: &[Op], out: &mut HashSet<ValueId>) {
    for op in ops {
        out.extend(op.operands.iter().copied());
        for inner in &op.regions {
            reads(&inner.ops, out);
        }
    }
}

/// The wave64 passes before `vectorize`'s: `fuse_ifs`, then the repeated
/// cross-lane ops of the fused runs merged with their pure ops (`lane_cse`).
pub fn prepare(kernel: &mut Kernel) {
    fuse_ifs(kernel);
    super::lane_cse::merge_exchanges(kernel, true);
}

/// The gfx950 passes after the automatic unrolling, whose copies repeat
/// loads and exchanges: `ro_loads` (read-only memory past barriers unless
/// `ROCKE_DIRECT_DEBUG=norocse`) and `lane_cse` (exchanges only).
pub fn after_unroll(kernel: &mut Kernel) {
    super::ro_loads::reuse_loads(kernel, !crate::rir::debug_flag("norocse"));
    super::lane_cse::merge_exchanges(kernel, false);
}

/// Fuses the `scf.if` pairs of `kernel` that can be; returns whether any were.
/// `ROCKE_DIRECT_DEBUG=nostorewiden` skips it and `combine_global_stores`.
pub fn fuse_ifs(kernel: &mut Kernel) -> bool {
    if crate::rir::debug_flag("nostorewiden") {
        return false;
    }
    fuse_region(&mut kernel.body)
}

fn fuse_region(region: &mut Region) -> bool {
    let mut changed = false;
    for op in region.ops.iter_mut() {
        for inner in op.regions.iter_mut() {
            changed |= fuse_region(inner);
        }
    }
    let mut i = 0;
    while i < region.ops.len() {
        let Some(cond) = plain_if(&region.ops[i]) else {
            i += 1;
            continue;
        };
        let j = (i + 1..region.ops.len())
            .find(|&k| !(region.ops[k].regions.is_empty() && region.ops[k].kind.is_pure()))
            .unwrap_or(region.ops.len());
        if j == region.ops.len() || plain_if(&region.ops[j]) != Some(cond) {
            i += 1;
            continue;
        }
        // An op between stays outside when a later op, or one between that stays outside, reads it.
        let mut needed = HashSet::new();
        reads(&region.ops[j + 1..], &mut needed);
        let mut outside = vec![false; j - i - 1];
        for k in (i + 1..j).rev() {
            let op = &region.ops[k];
            if op.results.iter().any(|r| needed.contains(r)) {
                outside[k - i - 1] = true;
                needed.extend(op.operands.iter().copied());
            }
        }
        let mut tail = region.ops.split_off(i);
        let rest = tail.split_off(j - i + 1);
        let mut second = tail.pop().expect("the second scf.if");
        let mut first = tail.remove(0);
        let mut before = Vec::new();
        for (k, op) in tail.into_iter().enumerate() {
            if outside[k] {
                before.push(op);
            } else {
                first.regions[0].ops.push(op);
            }
        }
        first.regions[0].ops.append(&mut second.regions[0].ops);
        region.ops.extend(before);
        i = region.ops.len();
        region.ops.push(first);
        region.ops.extend(rest);
        changed = true;
    }
    changed
}

/// What the address analysis reads of a value's definition: for a pure op,
/// a hash of its attributes when they are integers, booleans and strings.
#[derive(Clone)]
struct Def {
    kind: OpKind,
    operands: Operands,
    value: Option<i64>,
    wraps: bool,
    attrs: Option<u64>,
}

/// The `Def` of each value an op with one result defines, by value: a slot
/// per value of the kernel into `defs`.
struct Defs {
    slot: Vec<u32>,
    defs: Vec<Def>,
}

impl Defs {
    fn new(values: usize) -> Defs {
        Defs { slot: vec![u32::MAX; values], defs: Vec::new() }
    }

    fn get(&self, v: &ValueId) -> Option<&Def> {
        self.defs.get(*self.slot.get(v.0 as usize)? as usize)
    }

    fn insert(&mut self, v: ValueId, def: Def) {
        let at = &mut self.slot[v.0 as usize];
        if *at == u32::MAX {
            *at = self.defs.len() as u32;
            self.defs.push(def);
        } else {
            self.defs[*at as usize] = def;
        }
    }
}

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

fn definitions(region: &Region, out: &mut Defs) {
    for op in &region.ops {
        if let [result] = op.results[..] {
            let value = if op.kind == OpKind::Constant { op.int_attr("value") } else { None };
            let attrs = if op.kind.is_pure() && op.regions.is_empty() { attr_hash(&op.attrs) } else { None };
            out.insert(result, Def { kind: op.kind, operands: op.operands.clone(), value, wraps: op.attr("wrap").is_some(), attrs });
        }
        for inner in &op.regions {
            definitions(inner, out);
        }
    }
}

/// A store's byte address: `base + scale * (sum of terms) + bytes`.
#[derive(Clone, PartialEq, Eq, Hash)]
struct Class {
    base: ValueId,
    scale: i64,
    terms: Vec<ValueId>,
    elem: Scalar,
    nt: bool,
}

struct Address<'a> {
    kernel: &'a Kernel,
    defs: &'a Defs,
    /// Pure definitions by kind, attributes, operands' numbers and type: the first value of each.
    numbers: FxHashMap<(OpKind, u64, Vec<ValueId>, Type), ValueId>,
    /// `number`'s answers by value and depth. It gives the same answer each
    /// time: the entries of `numbers` it reads, once made, never change.
    numbered: FxHashMap<(ValueId, u32), ValueId>,
}

impl Address<'_> {
    fn constant(&self, v: ValueId) -> Option<i64> {
        self.defs.get(&v).and_then(|d| d.value)
    }

    /// One value for every pure op of the same kind, attributes, type and operands (by their numbers), so that
    /// the copies of an address each `scf.if` computed for itself compare equal once the bodies are fused.
    fn number(&mut self, v: ValueId, depth: u32) -> ValueId {
        let defs = self.defs;
        let Some(d) = defs.get(&v) else { return v };
        let Some(attrs) = d.attrs.filter(|_| depth < 8) else { return v };
        if let Some(&n) = self.numbered.get(&(v, depth)) {
            return n;
        }
        let operands: Vec<ValueId> = d.operands.iter().map(|&o| self.number(o, depth + 1)).collect();
        let ty = self.kernel.ty(v).clone();
        let n = *self.numbers.entry((d.kind, attrs, operands, ty)).or_insert(v);
        self.numbered.insert((v, depth), n);
        n
    }

    /// Adds the terms and constant of the i32 value `v` (non-wrapping adds taken apart) to `terms` and `sum`.
    fn flatten(&self, v: ValueId, depth: u32, terms: &mut Vec<ValueId>, sum: &mut i64) {
        match self.defs.get(&v) {
            Some(d) if d.kind == OpKind::Constant && d.value.is_some() => *sum += d.value.unwrap_or(0),
            Some(d) if d.kind == OpKind::Add && !d.wraps && d.operands.len() == 2 && depth < 16
                && *self.kernel.ty(v) == Type::Scalar(Scalar::I32) =>
            {
                self.flatten(d.operands[0], depth + 1, terms, sum);
                self.flatten(d.operands[1], depth + 1, terms, sum);
            }
            _ => terms.push(v),
        }
    }

    /// The class key (without element and `nt`) and byte constant of a pointer.
    fn of(&mut self, ptr: ValueId) -> Option<(ValueId, i64, Vec<ValueId>, i64)> {
        let d = self.defs.get(&ptr).filter(|d| d.kind == OpKind::GlobalPtrAdd && d.operands.len() == 2)?;
        let (base, x) = (d.operands[0], d.operands[1]);
        let (scale, y) = match self.defs.get(&x) {
            Some(s) if s.kind == OpKind::Shl && s.operands.len() == 2 => {
                let k = self.constant(s.operands[1]).filter(|k| (0..=4).contains(k))?;
                (1i64 << k, s.operands[0])
            }
            Some(m) if m.kind == OpKind::Mul && m.operands.len() == 2 && !m.wraps => {
                let k = self.constant(m.operands[1]).filter(|k| (1..=16).contains(k))?;
                (k, m.operands[0])
            }
            _ => (1, x),
        };
        let e = self.defs.get(&y).filter(|e| e.kind == OpKind::Sext && e.operands.len() == 1)?;
        if *self.kernel.ty(e.operands[0]) != Type::Scalar(Scalar::I32) {
            return None;
        }
        let (mut terms, mut sum) = (Vec::new(), 0i64);
        self.flatten(e.operands[0], 0, &mut terms, &mut sum);
        let mut terms: Vec<ValueId> = terms.into_iter().map(|t| self.number(t, 0)).collect();
        terms.sort_unstable();
        Some((self.number(base, 0), scale, terms, scale.checked_mul(sum)?))
    }
}

/// A store of a class in a run: its position, its address operands and their
/// byte constant, its byte constant with its offset, its size and its value.
struct Member {
    at: usize,
    address: Vec<ValueId>,
    address_bytes: i64,
    bytes: i64,
    size: i64,
    value: ValueId,
}

/// Whether `op` ends a run of stores: it may read global memory, write it
/// other than as a store of a class, order other waves' accesses, or holds a
/// region.
fn breaks_run(op: &Op) -> bool {
    !op.regions.is_empty()
        || (op.kind.writes_global() && op.kind != OpKind::GStore)
        || matches!(
            op.kind,
            OpKind::GLoad
                | OpKind::GlobalLoad
                | OpKind::GlobalLoadVec
                | OpKind::BufferLoadVec
                | OpKind::BufferLoadScalar
                | OpKind::BufferLoadLds
                | OpKind::GlobalLoadLds
                | OpKind::Sync
                | OpKind::SyncLdsOnly
                | OpKind::BarrierBare
                | OpKind::WaitCnt
                | OpKind::WaitAsyncCnt
                | OpKind::SchedBarrier
                | OpKind::SchedGroupBarrier
                | OpKind::IglpOpt
                | OpKind::InlineAsmIdentity
        )
}

/// Merges the stores of `kernel` that can be; returns whether any were.
pub fn combine_global_stores(kernel: &mut Kernel) -> bool {
    if crate::rir::debug_flag("nostorewiden") {
        return false;
    }
    if !has_store_run(&kernel.body) {
        return false;
    }
    let mut defs = Defs::new(kernel.values.len());
    definitions(&kernel.body, &mut defs);
    let mut body = std::mem::take(&mut kernel.body);
    let changed = store_region(kernel, &defs, &mut body);
    kernel.body = body;
    changed
}

/// Whether `region`, or one in it, has two stores with no op between them
/// that ends a run (`breaks_run`). `store_region` merges or moves stores only
/// in a run of two or more, so without one it changes nothing.
fn has_store_run(region: &Region) -> bool {
    let mut open = false;
    for op in &region.ops {
        if op.regions.iter().any(has_store_run) {
            return true;
        }
        if breaks_run(op) {
            open = false;
        } else if op.kind == OpKind::GStore {
            if open {
                return true;
            }
            open = true;
        }
    }
    false
}

fn store_region(kernel: &mut Kernel, defs: &Defs, region: &mut Region) -> bool {
    let mut changed = false;
    for op in region.ops.iter_mut() {
        for inner in op.regions.iter_mut() {
            changed |= store_region(kernel, defs, inner);
        }
    }
    // Runs of one class's stores, in program order.
    let mut runs: Vec<Vec<Member>> = Vec::new();
    let mut current: Option<(Class, Vec<Member>)> = None;
    {
        let mut address = Address { kernel, defs, numbers: FxHashMap::default(), numbered: FxHashMap::default() };
        for (i, op) in region.ops.iter().enumerate() {
            if breaks_run(op) {
                runs.extend(current.take().map(|(_, m)| m));
                continue;
            }
            if op.kind != OpKind::GStore {
                continue;
            }
            let (&value, operands) = op.operands.split_last().expect("a store has a value");
            let elem = match kernel.ty(value) {
                Type::Scalar(s) if matches!(s.bytes(), 2 | 4) => Some(*s),
                _ => None,
            };
            // A pointer: one sum of values; a scalar base and a lane offset: those two values.
            let key = match (elem, operands) {
                (Some(elem), [ptr]) => address.of(*ptr).map(|(base, scale, terms, bytes)| (Class { base, scale, terms, elem, nt: false }, bytes)),
                (Some(elem), [sbase, voffset]) => {
                    let terms = vec![address.number(*voffset, 0)];
                    Some((Class { base: address.number(*sbase, 0), scale: 0, terms, elem, nt: false }, 0))
                }
                _ => None,
            };
            let Some((mut class, address_bytes)) = key else {
                runs.extend(current.take().map(|(_, m)| m));
                continue;
            };
            class.nt = op.bool_attr("nt") == Some(true);
            if current.as_ref().is_some_and(|(c, _)| *c != class) {
                runs.extend(current.take().map(|(_, m)| m));
            }
            let size = i64::from(class.elem.bytes());
            let (_, members) = current.get_or_insert_with(|| (class, Vec::new()));
            members.push(Member {
                at: i,
                address: operands.to_vec(),
                address_bytes,
                bytes: address_bytes + op.int_attr("offset").unwrap_or(0),
                size,
                value,
            });
        }
        runs.extend(current.take().map(|(_, m)| m));
    }
    // Per position: the merged store that goes there (its bytes' first store, values, address operands and
    // offset), or a new address for the store there; the other stores of a merge are dropped.
    let mut merged: FxHashMap<usize, (usize, Vec<ValueId>, Vec<ValueId>, i64)> = FxHashMap::default();
    let mut rebased: FxHashMap<usize, (Vec<ValueId>, i64)> = FxHashMap::default();
    let mut dropped: FxHashSet<usize> = FxHashSet::default();
    for mut members in runs {
        if members.len() < 2 {
            continue;
        }
        // The class's first store's address operands address them all where the offset field holds it.
        let (head, head_bytes) = (members[0].address.clone(), members[0].address_bytes);
        members.sort_by_key(|m| m.bytes);
        if members.windows(2).any(|w| w[1].bytes < w[0].bytes + w[0].size) {
            continue;
        }
        let address = |m: &Member| {
            if OFFSETS.contains(&(m.bytes - head_bytes)) {
                (head.clone(), m.bytes - head_bytes)
            } else {
                (m.address.clone(), m.bytes - m.address_bytes)
            }
        };
        let mut k = 0;
        while k < members.len() {
            let size = members[k].size;
            let run = (k..members.len()).take_while(|&j| members[j].bytes == members[k].bytes + (j - k) as i64 * size).count();
            let width = WIDTHS.into_iter().find(|&w| w <= run as i64 * size && w > size);
            let n = width.map_or(1, |w| (w / size) as usize);
            let group = &members[k..k + n];
            let (operands, offset) = address(&group[0]);
            if n > 1 {
                let last = group.iter().map(|m| m.at).max().expect("a group has members");
                dropped.extend(group.iter().map(|m| m.at).filter(|&at| at != last));
                merged.insert(last, (group[0].at, group.iter().map(|m| m.value).collect(), operands, offset));
            } else if operands != group[0].address {
                rebased.insert(group[0].at, (operands, offset));
            }
            k += n;
        }
    }
    if merged.is_empty() && rebased.is_empty() {
        return changed;
    }
    // A merged store takes the attributes of its bytes' first store.
    let firsts: FxHashMap<usize, Op> = merged.values().map(|(first, ..)| (*first, region.ops[*first].clone())).collect();
    let old = std::mem::take(&mut region.ops);
    let mut out = Vec::with_capacity(old.len());
    for (i, mut op) in old.into_iter().enumerate() {
        if let Some((first, values, address, offset)) = merged.remove(&i) {
            let template = &firsts[&first];
            let elem = kernel.ty(values[0]).elem();
            let packed = kernel.new_value("wide", Type::Vec(elem, values.len() as u32));
            out.push(Op {
                kind: OpKind::Pack,
                results: Results::from([packed]),
                operands: values.iter().copied().collect(),
                attrs: Vec::new(),
                regions: Vec::new(),
                block_args: Vec::new(),
            });
            let mut store = template.clone();
            store.operands = address.iter().copied().chain([packed]).collect();
            store.set_attr("offset", Attr::Int(offset));
            out.push(store);
        } else if dropped.contains(&i) {
            continue;
        } else {
            if let Some((address, offset)) = rebased.remove(&i) {
                let value = *op.operands.last().expect("a store has a value");
                op.operands = address.iter().copied().chain([value]).collect();
                op.set_attr("offset", Attr::Int(offset));
            }
            out.push(op);
        }
    }
    region.ops = out;
    true
}

#[cfg(test)]
mod tests {
    use crate::rir;
    use crate::target::Target;

    /// Y[16 tid + e] = f16(X[16 tid + e]) for each element `e`, each store in an
    /// `scf.if` of its own, on `cond` for the odd stores and `%c` for the even
    /// ones, with `between` after the fourth.
    fn kernel(elements: &[i64], cond: &str, between: &str) -> String {
        let mut s = String::from(
            "ckdsl.ir v1\nkernel @k {\n  attrs { max_workgroup_size = i:64 }\n  params {\n    %X : ptr<f32,global>\n    \
             %Y : ptr<f16,global>\n    %W : ptr<f32,global>\n    %n : i32\n  }\n  region @entry {\n    \
             %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n    %c16 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:16 }\n    \
             %b : i32 = arith.mul ( %tid, %c16 )\n    %c : i1 = arith.cmp ( %tid, %n ) { pred = s:\"lt\" }\n    \
             %d : i1 = arith.cmp ( %tid, %c16 ) { pred = s:\"lt\" }\n",
        );
        for (k, e) in elements.iter().enumerate() {
            s += &format!(
                "    %c{k} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{e} }}\n    %i{k} : i32 = arith.add ( %b, %c{k} )\n    \
                 %x{k} : f32 = memref.global_load ( %X, %i{k} ) {{ align = i:4 }}\n"
            );
        }
        for k in 0..elements.len() {
            let c = if k % 2 == 1 { cond } else { "%c" };
            s += &format!(
                "    %h{k} : f16 = arith.cast_f32_to ( %x{k} ) {{ target = s:\"f16\" }}\n    scf.if ( {c} )\n      region @then {{\n        \
                 memref.global_store_typed ( %Y, %i{k}, %h{k} ) {{ align = i:2, elem_type = s:\"f16\" }}\n      }}\n"
            );
            if k == 3 {
                s += between;
            }
        }
        s + "    cf.return (  )\n  }\n}\n"
    }

    const EIGHT: [i64; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

    fn count(listing: &str, mnemonic: &str) -> usize {
        listing.lines().filter(|l| l.split_whitespace().next() == Some(mnemonic)).count()
    }

    fn compiled(text: &str) -> String {
        rir::compile_any_target(text, Target::Gfx950).unwrap_or_else(|e| panic!("{e}")).listing()
    }

    fn narrow(listing: &str) -> usize {
        count(listing, "global_store_short") + count(listing, "global_store_short_d16_hi")
    }

    #[test]
    fn stores_in_ifs_of_one_condition_become_one_wide_store() {
        let listing = compiled(&kernel(&EIGHT, "%c", ""));
        assert_eq!((count(&listing, "global_store_dwordx4"), narrow(&listing)), (1, 0), "{listing}");
        // One branch around them.
        assert_eq!(count(&listing, "s_and_saveexec_b64"), 1, "{listing}");
    }

    #[test]
    fn groups_of_one_class_share_one_address() {
        let listing = compiled(&kernel(&[0, 1, 2, 3, 8, 9, 10, 11], "%c", ""));
        assert_eq!((count(&listing, "global_store_dwordx2"), narrow(&listing)), (2, 0), "{listing}");
        // The second group stores at an immediate offset from the first's address.
        assert!(listing.lines().any(|l| l.contains("global_store_dwordx2") && l.contains("imm:16")), "{listing}");
    }

    #[test]
    fn ifs_of_two_conditions_stay_apart() {
        let listing = compiled(&kernel(&EIGHT, "%d", ""));
        assert_eq!(count(&listing, "global_store_dwordx4") + count(&listing, "global_store_dwordx2"), 0, "{listing}");
        assert_eq!(narrow(&listing), 8, "{listing}");
    }

    #[test]
    fn a_global_store_between_keeps_the_halves_apart() {
        let store = "    %o : f32 = arith.constant (  ) { ity = s:\"f32\", value = f:1.0 }\n    \
                     memref.global_store_typed ( %W, %tid, %o ) { align = i:4, elem_type = s:\"f32\" }\n";
        let listing = compiled(&kernel(&EIGHT, "%c", store));
        // Each half of the stores merges, but not across the other store.
        assert_eq!((count(&listing, "global_store_dwordx2"), count(&listing, "global_store_dwordx4")), (2, 0), "{listing}");
        assert_eq!(narrow(&listing), 0, "{listing}");
    }

    #[test]
    fn a_global_load_between_keeps_the_ifs_apart() {
        let load = "    %l : f32 = memref.global_load ( %W, %tid ) { align = i:4 }\n    \
                    memref.global_store_typed ( %X, %tid, %l ) { align = i:4, elem_type = s:\"f32\" }\n";
        let listing = compiled(&kernel(&EIGHT, "%c", load));
        assert_eq!(count(&listing, "s_and_saveexec_b64"), 2, "{listing}");
    }

    #[test]
    fn wave32_kernels_keep_their_stores() {
        let listing = rir::compile_any_target(&kernel(&EIGHT, "%c", ""), Target::Gfx1151).unwrap_or_else(|e| panic!("{e}")).listing();
        assert_eq!(count(&listing, "global_store_b128") + count(&listing, "global_store_b64"), 0, "{listing}");
    }

    /// `combine_global_stores` as it was before its pre-check and memoized
    /// numbering, kept verbatim to compare against.
    mod reference {
        use super::super::{attr_hash, breaks_run, Class, Def, Member, OFFSETS, WIDTHS};
        use crate::rir::ast::*;
        use std::collections::{HashMap, HashSet};

        fn definitions(region: &Region, out: &mut HashMap<ValueId, Def>) {
            for op in &region.ops {
                if let [result] = op.results[..] {
                    let value = if op.kind == OpKind::Constant { op.int_attr("value") } else { None };
                    let attrs = if op.kind.is_pure() && op.regions.is_empty() { attr_hash(&op.attrs) } else { None };
                    out.insert(
                        result,
                        Def { kind: op.kind, operands: op.operands.iter().copied().collect(), value, wraps: op.attr("wrap").is_some(), attrs },
                    );
                }
                for inner in &op.regions {
                    definitions(inner, out);
                }
            }
        }

        struct Address<'a> {
            kernel: &'a Kernel,
            defs: &'a HashMap<ValueId, Def>,
            numbers: HashMap<(OpKind, u64, Vec<ValueId>, Type), ValueId>,
        }

        impl Address<'_> {
            fn constant(&self, v: ValueId) -> Option<i64> {
                self.defs.get(&v).and_then(|d| d.value)
            }

            fn number(&mut self, v: ValueId, depth: u32) -> ValueId {
                let Some(d) = self.defs.get(&v) else { return v };
                let Some(attrs) = d.attrs.filter(|_| depth < 8) else { return v };
                let (kind, operands) = (d.kind, d.operands.clone());
                let operands: Vec<ValueId> = operands.iter().map(|&o| self.number(o, depth + 1)).collect();
                let ty = self.kernel.ty(v).clone();
                *self.numbers.entry((kind, attrs, operands, ty)).or_insert(v)
            }

            fn flatten(&self, v: ValueId, depth: u32, terms: &mut Vec<ValueId>, sum: &mut i64) {
                match self.defs.get(&v) {
                    Some(d) if d.kind == OpKind::Constant && d.value.is_some() => *sum += d.value.unwrap_or(0),
                    Some(d) if d.kind == OpKind::Add && !d.wraps && d.operands.len() == 2 && depth < 16
                        && *self.kernel.ty(v) == Type::Scalar(Scalar::I32) =>
                    {
                        self.flatten(d.operands[0], depth + 1, terms, sum);
                        self.flatten(d.operands[1], depth + 1, terms, sum);
                    }
                    _ => terms.push(v),
                }
            }

            fn of(&mut self, ptr: ValueId) -> Option<(ValueId, i64, Vec<ValueId>, i64)> {
                let d = self.defs.get(&ptr).filter(|d| d.kind == OpKind::GlobalPtrAdd && d.operands.len() == 2)?;
                let (base, x) = (d.operands[0], d.operands[1]);
                let (scale, y) = match self.defs.get(&x) {
                    Some(s) if s.kind == OpKind::Shl && s.operands.len() == 2 => {
                        let k = self.constant(s.operands[1]).filter(|k| (0..=4).contains(k))?;
                        (1i64 << k, s.operands[0])
                    }
                    Some(m) if m.kind == OpKind::Mul && m.operands.len() == 2 && !m.wraps => {
                        let k = self.constant(m.operands[1]).filter(|k| (1..=16).contains(k))?;
                        (k, m.operands[0])
                    }
                    _ => (1, x),
                };
                let e = self.defs.get(&y).filter(|e| e.kind == OpKind::Sext && e.operands.len() == 1)?;
                if *self.kernel.ty(e.operands[0]) != Type::Scalar(Scalar::I32) {
                    return None;
                }
                let (mut terms, mut sum) = (Vec::new(), 0i64);
                self.flatten(e.operands[0], 0, &mut terms, &mut sum);
                let mut terms: Vec<ValueId> = terms.into_iter().map(|t| self.number(t, 0)).collect();
                terms.sort_unstable();
                Some((self.number(base, 0), scale, terms, scale.checked_mul(sum)?))
            }
        }

        pub fn combine_global_stores(kernel: &mut Kernel) -> bool {
            if crate::rir::debug_flag("nostorewiden") {
                return false;
            }
            let mut defs = HashMap::new();
            definitions(&kernel.body, &mut defs);
            let mut body = std::mem::take(&mut kernel.body);
            let changed = store_region(kernel, &defs, &mut body);
            kernel.body = body;
            changed
        }

        fn store_region(kernel: &mut Kernel, defs: &HashMap<ValueId, Def>, region: &mut Region) -> bool {
            let mut changed = false;
            for op in region.ops.iter_mut() {
                for inner in op.regions.iter_mut() {
                    changed |= store_region(kernel, defs, inner);
                }
            }
            let mut runs: Vec<Vec<Member>> = Vec::new();
            let mut current: Option<(Class, Vec<Member>)> = None;
            {
                let mut address = Address { kernel, defs, numbers: HashMap::new() };
                for (i, op) in region.ops.iter().enumerate() {
                    if breaks_run(op) {
                        runs.extend(current.take().map(|(_, m)| m));
                        continue;
                    }
                    if op.kind != OpKind::GStore {
                        continue;
                    }
                    let (&value, operands) = op.operands.split_last().expect("a store has a value");
                    let elem = match kernel.ty(value) {
                        Type::Scalar(s) if matches!(s.bytes(), 2 | 4) => Some(*s),
                        _ => None,
                    };
                    let key = match (elem, operands) {
                        (Some(elem), [ptr]) => {
                            address.of(*ptr).map(|(base, scale, terms, bytes)| (Class { base, scale, terms, elem, nt: false }, bytes))
                        }
                        (Some(elem), [sbase, voffset]) => {
                            let terms = vec![address.number(*voffset, 0)];
                            Some((Class { base: address.number(*sbase, 0), scale: 0, terms, elem, nt: false }, 0))
                        }
                        _ => None,
                    };
                    let Some((mut class, address_bytes)) = key else {
                        runs.extend(current.take().map(|(_, m)| m));
                        continue;
                    };
                    class.nt = op.bool_attr("nt") == Some(true);
                    if current.as_ref().is_some_and(|(c, _)| *c != class) {
                        runs.extend(current.take().map(|(_, m)| m));
                    }
                    let size = i64::from(class.elem.bytes());
                    let (_, members) = current.get_or_insert_with(|| (class, Vec::new()));
                    members.push(Member {
                        at: i,
                        address: operands.to_vec(),
                        address_bytes,
                        bytes: address_bytes + op.int_attr("offset").unwrap_or(0),
                        size,
                        value,
                    });
                }
                runs.extend(current.take().map(|(_, m)| m));
            }
            let mut merged: HashMap<usize, (usize, Vec<ValueId>, Vec<ValueId>, i64)> = HashMap::new();
            let mut rebased: HashMap<usize, (Vec<ValueId>, i64)> = HashMap::new();
            let mut dropped: HashSet<usize> = HashSet::new();
            for mut members in runs {
                if members.len() < 2 {
                    continue;
                }
                let (head, head_bytes) = (members[0].address.clone(), members[0].address_bytes);
                members.sort_by_key(|m| m.bytes);
                if members.windows(2).any(|w| w[1].bytes < w[0].bytes + w[0].size) {
                    continue;
                }
                let address = |m: &Member| {
                    if OFFSETS.contains(&(m.bytes - head_bytes)) {
                        (head.clone(), m.bytes - head_bytes)
                    } else {
                        (m.address.clone(), m.bytes - m.address_bytes)
                    }
                };
                let mut k = 0;
                while k < members.len() {
                    let size = members[k].size;
                    let run = (k..members.len()).take_while(|&j| members[j].bytes == members[k].bytes + (j - k) as i64 * size).count();
                    let width = WIDTHS.into_iter().find(|&w| w <= run as i64 * size && w > size);
                    let n = width.map_or(1, |w| (w / size) as usize);
                    let group = &members[k..k + n];
                    let (operands, offset) = address(&group[0]);
                    if n > 1 {
                        let last = group.iter().map(|m| m.at).max().expect("a group has members");
                        dropped.extend(group.iter().map(|m| m.at).filter(|&at| at != last));
                        merged.insert(last, (group[0].at, group.iter().map(|m| m.value).collect(), operands, offset));
                    } else if operands != group[0].address {
                        rebased.insert(group[0].at, (operands, offset));
                    }
                    k += n;
                }
            }
            if merged.is_empty() && rebased.is_empty() {
                return changed;
            }
            let firsts: HashMap<usize, Op> = merged.values().map(|(first, ..)| (*first, region.ops[*first].clone())).collect();
            let old = std::mem::take(&mut region.ops);
            let mut out = Vec::with_capacity(old.len());
            for (i, mut op) in old.into_iter().enumerate() {
                if let Some((first, values, address, offset)) = merged.remove(&i) {
                    let template = &firsts[&first];
                    let elem = kernel.ty(values[0]).elem();
                    let packed = kernel.new_value("wide", Type::Vec(elem, values.len() as u32));
                    out.push(Op {
                        kind: OpKind::Pack,
                        results: Results::from([packed]),
                        operands: values.iter().copied().collect(),
                        attrs: Vec::new(),
                        regions: Vec::new(),
                        block_args: Vec::new(),
                    });
                    let mut store = template.clone();
                    store.operands = address.iter().copied().chain([packed]).collect();
                    store.set_attr("offset", Attr::Int(offset));
                    out.push(store);
                } else if dropped.contains(&i) {
                    continue;
                } else {
                    if let Some((address, offset)) = rebased.remove(&i) {
                        let value = *op.operands.last().expect("a store has a value");
                        op.operands = address.iter().copied().chain([value]).collect();
                        op.set_attr("offset", Attr::Int(offset));
                    }
                    out.push(op);
                }
            }
            region.ops = out;
            true
        }
    }

    /// `stores` 16- or 32-bit stores into two arrays at element offsets of one
    /// row (some nested sums), each alone or in an `scf.if` of one of two
    /// conditions, with a global load and store, a barrier or nothing between
    /// some of them; a row index scaled by 16 or shifted by 4.
    fn random_stores(seed: u64) -> String {
        let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let mut next = |n: u64| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s % n
        };
        let stores = 2 + next(12) as usize;
        let row = if next(2) == 0 { "%b : i32 = arith.mul ( %tid, %c16 )" } else { "%b : i32 = arith.shl ( %tid, %c4 )" };
        let mut t = format!(
            "ckdsl.ir v1\nkernel @k {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n    %X : ptr<f32,global>\n    \
             %Y : ptr<f16,global>\n    %Z : ptr<f32,global>\n    %W : ptr<f32,global>\n    %n : i32\n  }}\n  region @entry {{\n    \
             %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n    %c16 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:16 }}\n    \
             %c4 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:4 }}\n    %c2 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:2 }}\n    \
             {row}\n    %c : i1 = arith.cmp ( %tid, %n ) {{ pred = s:\"lt\" }}\n    %d : i1 = arith.cmp ( %tid, %c16 ) {{ pred = s:\"lt\" }}\n"
        );
        for k in 0..stores {
            t += &format!("    %e{k} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{} }}\n", next(14));
            t += &match next(3) {
                0 => format!("    %j{k} : i32 = arith.add ( %b, %c2 )\n    %i{k} : i32 = arith.add ( %j{k}, %e{k} )\n"),
                _ => format!("    %i{k} : i32 = arith.add ( %b, %e{k} )\n"),
            };
            t += &format!("    %x{k} : f32 = memref.global_load ( %X, %i{k} ) {{ align = i:4 }}\n");
        }
        for k in 0..stores {
            let store = match next(3) {
                0 => format!("memref.global_store_typed ( %Z, %i{k}, %x{k} ) {{ align = i:4, elem_type = s:\"f32\" }}"),
                _ => format!(
                    "%h{k} : f16 = arith.cast_f32_to ( %x{k} ) {{ target = s:\"f16\" }}\n    \
                     memref.global_store_typed ( %Y, %i{k}, %h{k} ) {{ align = i:2, elem_type = s:\"f16\" }}"
                ),
            };
            t += &match next(4) {
                0 => format!("    {store}\n"),
                1 => format!("    scf.if ( %d )\n      region @then {{\n    {store}\n      }}\n"),
                _ => format!("    scf.if ( %c )\n      region @then {{\n    {store}\n      }}\n"),
            };
            t += &match next(9) {
                0 => "    tile.sync (  )\n".to_string(),
                1 => format!(
                    "    %l{k} : f32 = memref.global_load ( %W, %tid ) {{ align = i:4 }}\n    \
                     memref.global_store_typed ( %W, %c16, %l{k} ) {{ align = i:4, elem_type = s:\"f32\" }}\n"
                ),
                _ => String::new(),
            };
        }
        t + "    cf.return (  )\n  }\n}\n"
    }

    /// Marks every store of `region` (and the regions in it) whose number `pick` takes as nontemporal.
    fn mark_nt(region: &mut crate::rir::ast::Region, pick: &mut dyn FnMut() -> bool) {
        for op in region.ops.iter_mut() {
            for inner in op.regions.iter_mut() {
                mark_nt(inner, pick);
            }
            if op.kind == crate::rir::ast::OpKind::GStore && pick() {
                op.set_attr("nt", crate::rir::ast::Attr::Bool(true));
            }
        }
    }

    #[test]
    fn store_merging_changes_what_it_changed() {
        let (mut merged, mut kept) = (0, 0);
        for seed in 0..400u64 {
            let text = random_stores(seed);
            let mut k = rir::parse::parse(&text).unwrap_or_else(|e| panic!("{e}"));
            let (mut lds, _) = rir::analysis::lds_layout(&k);
            rir::opt::optimize(&mut k, &mut lds).unwrap_or_else(|e| panic!("{e}"));
            k.wave = 64;
            super::fuse_ifs(&mut k);
            let mut n = seed;
            mark_nt(&mut k.body, &mut || {
                n = n.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                seed % 3 == 0 && n >> 62 == 0
            });
            let mut old = k.clone();
            let changed = reference::combine_global_stores(&mut old);
            assert_eq!(super::combine_global_stores(&mut k), changed, "seed {seed}");
            assert_eq!(rir::print::dump(&k), rir::print::dump(&old), "seed {seed}");
            assert_eq!(format!("{:?}", k.values), format!("{:?}", old.values), "seed {seed}");
            if changed {
                merged += 1;
            } else {
                kept += 1;
            }
        }
        assert!(merged > 100 && kept > 20, "{merged} kernels merged, {kept} kept");
    }
}
