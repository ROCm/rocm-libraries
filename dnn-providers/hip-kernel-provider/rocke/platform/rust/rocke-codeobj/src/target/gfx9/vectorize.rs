// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Vector forms of scalar code on gfx950, as LLVM's vectorizers give them in
//! COMGR's code (the multi-D GEMM epilogues read their D tensors and combine
//! them element by element).
//!
//! `combine_global_loads`: after address legalization, in a run of a region's
//! ops with no global write, barrier, fence or nested region, the 16- and
//! 32-bit scalar loads from one address (the same address operands) whose
//! offsets cover 4, 8 or 16 contiguous bytes become one load of a vector where
//! the first of them was, and an element extract for each. The wider load
//! reads the same bytes; gfx950 takes it at any alignment.
//!
//! `pair_f16_ops`: two scalar f16 adds, multiplies or fmas of a region become
//! one packed op where the first was, when each operand pair is the two halves
//! of one dword of a vector, the two results of a pair already made, or one
//! scalar both read; each result is then that op's half.
//!
//! `pack_mfma_operands`: a 16-bit vector an MFMA reads as A or B, built by
//! inserting a global load into every element in turn, becomes one pack of
//! those values, which selection makes with a `v_perm_b32` per dword where
//! each insert takes one (the B fragments of the direct GEMMs). Halves loaded
//! from LDS stay inserted: selection writes those in place.

use crate::rir::ast::*;
use std::collections::HashMap;

/// Bytes a merged load may have: a dword, two or four.
const WIDTHS: [i64; 3] = [16, 8, 4];

/// Whether `op` ends a run: it may write global memory, order other waves'
/// accesses, or hold a region.
fn breaks_run(op: &Op) -> bool {
    !op.regions.is_empty()
        || op.kind.writes_global()
        || matches!(
            op.kind,
            OpKind::Sync | OpKind::SyncLdsOnly | OpKind::BarrierBare | OpKind::Fence | OpKind::InlineAsmIdentity | OpKind::GlobalAtomicAdd | OpKind::GAtomicAdd
        )
}

/// A load's key: its address operands and element type, and its offset.
fn candidate(kernel: &Kernel, op: &Op) -> Option<((Vec<ValueId>, Scalar), i64)> {
    if op.kind != OpKind::GLoad || op.results.len() != 1 {
        return None;
    }
    let s = match kernel.ty(op.results[0]) {
        Type::Scalar(s) => *s,
        _ => return None,
    };
    if !matches!(s.bytes(), 2 | 4) {
        return None;
    }
    Some(((op.operands.iter().copied().collect(), s), op.int_attr("offset")?))
}

/// Merges the loads of `kernel` that can be; returns whether any were.
pub fn combine_global_loads(kernel: &mut Kernel) -> bool {
    let mut body = std::mem::take(&mut kernel.body);
    let changed = region(kernel, &mut body);
    kernel.body = body;
    changed
}

fn region(kernel: &mut Kernel, region: &mut Region) -> bool {
    let mut changed = false;
    for op in region.ops.iter_mut() {
        for inner in op.regions.iter_mut() {
            changed |= self::region(kernel, inner);
        }
    }
    changed | region_ops(kernel, region)
}

/// One merge: the loads (positions in the region) by ascending offset, and
/// their element type.
struct Merge {
    loads: Vec<usize>,
    elem: Scalar,
}

fn region_ops(kernel: &mut Kernel, region: &mut Region) -> bool {
    let ops = &region.ops;
    let mut merges: Vec<Merge> = Vec::new();
    let mut start = 0;
    while start < ops.len() {
        let end = (start..ops.len()).find(|&i| breaks_run(&ops[i])).unwrap_or(ops.len());
        let mut groups: Vec<((Vec<ValueId>, Scalar), Vec<(i64, usize)>)> = Vec::new();
        for (i, op) in ops.iter().enumerate().take(end).skip(start) {
            if let Some((key, offset)) = candidate(kernel, op) {
                match groups.iter_mut().find(|(k, _)| *k == key) {
                    Some((_, list)) => list.push((offset, i)),
                    None => groups.push((key, vec![(offset, i)])),
                }
            }
        }
        for ((_, elem), mut list) in groups {
            list.sort();
            list.dedup_by_key(|(offset, _)| *offset);
            let bytes = i64::from(elem.bytes());
            let mut k = 0;
            while k < list.len() {
                let run = (k..list.len()).take_while(|&j| list[j].0 == list[k].0 + (j - k) as i64 * bytes).count();
                let width = WIDTHS.into_iter().find(|&w| w <= run as i64 * bytes && w > bytes);
                match width {
                    Some(w) => {
                        let n = (w / bytes) as usize;
                        merges.push(Merge { loads: list[k..k + n].iter().map(|&(_, i)| i).collect(), elem });
                        k += n;
                    }
                    None => k += 1,
                }
            }
        }
        start = end + 1;
    }
    if merges.is_empty() {
        return false;
    }
    // Per position: the merge it starts (at its earliest load) or a load it drops.
    let mut starts: Vec<Option<usize>> = vec![None; ops.len()];
    let mut dropped = vec![false; ops.len()];
    for (m, merge) in merges.iter().enumerate() {
        let first = *merge.loads.iter().min().expect("a merge has loads");
        starts[first] = Some(m);
        for &i in &merge.loads {
            dropped[i] = true;
        }
    }
    let old = std::mem::take(&mut region.ops);
    let mut taken: Vec<Option<Op>> = old.into_iter().map(Some).collect();
    let mut out = Vec::with_capacity(taken.len());
    for i in 0..taken.len() {
        if let Some(m) = starts[i] {
            let merge = &merges[m];
            let head = taken[merge.loads[0]].as_ref().expect("a load not yet taken");
            let (operands, align) = (head.operands.clone(), head.int_attr("align").unwrap_or(1));
            let offset = head.int_attr("offset").unwrap_or(0);
            let vector = kernel.new_value("vload", Type::Vec(merge.elem, merge.loads.len() as u32));
            let attrs = vec![("offset".into(), Attr::Int(offset)), ("align".into(), Attr::Int(align))];
            out.push(Op { kind: OpKind::GLoad, results: Results::from([vector]), operands, attrs, regions: Vec::new(), block_args: Vec::new() });
            for (index, &j) in merge.loads.iter().enumerate() {
                let load = taken[j].take().expect("each load is in one merge");
                let attrs = vec![("index".into(), Attr::Int(index as i64))];
                out.push(Op { kind: OpKind::Extract, results: load.results, operands: Operands::from([vector]), attrs, regions: Vec::new(), block_args: Vec::new() });
            }
            continue;
        }
        if dropped[i] {
            continue;
        }
        out.push(taken[i].take().expect("an op not merged"));
    }
    region.ops = out;
    true
}

/// An operand pair of two ops that may pair: lanes 2j and 2j+1 of a vector,
/// or of the pair `j` already made, or one value.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Lanes {
    Of(ValueId, u32),
    Pair(usize),
    Same(ValueId),
}

/// Pairs the f16 ops of `kernel` that can be; returns whether any were.
pub fn pair_f16_ops(kernel: &mut Kernel) -> bool {
    let mut body = std::mem::take(&mut kernel.body);
    let changed = pair_region(kernel, &mut body);
    kernel.body = body;
    changed
}

fn pair_region(kernel: &mut Kernel, region: &mut Region) -> bool {
    let mut changed = false;
    for op in region.ops.iter_mut() {
        for inner in op.regions.iter_mut() {
            changed |= pair_region(kernel, inner);
        }
    }
    let ops = &region.ops;
    let f16 = |kernel: &Kernel, v: ValueId| *kernel.ty(v) == Type::Scalar(Scalar::F16);
    // Each value an extract of an f16 vector or a paired op gives: its source and lane.
    let mut lane: HashMap<ValueId, (Lanes, u32)> = HashMap::new();
    for op in ops {
        if op.kind == OpKind::Extract && op.results.len() == 1 && f16(kernel, op.results[0]) {
            if let (Type::Vec(Scalar::F16, _), Some(i)) = (kernel.ty(op.operands[0]), op.int_attr("index")) {
                lane.insert(op.results[0], (Lanes::Of(op.operands[0], i as u32 / 2), i as u32 % 2));
            }
        }
    }
    let mut pairs: Vec<(usize, usize, Vec<Lanes>)> = Vec::new();
    let mut paired = vec![false; ops.len()];
    loop {
        let before = pairs.len();
        let mut first: HashMap<(OpKind, String, Vec<Lanes>), usize> = HashMap::new();
        for (i, op) in ops.iter().enumerate() {
            if paired[i] || !matches!(op.kind, OpKind::Fadd | OpKind::Fmul | OpKind::Fma) || op.results.len() != 1 {
                continue;
            }
            if !f16(kernel, op.results[0]) || !op.operands.iter().all(|&o| f16(kernel, o)) {
                continue;
            }
            let (mut parity, mut lanes) = (None, Vec::with_capacity(op.operands.len()));
            for &o in op.operands.iter() {
                match lane.get(&o) {
                    Some(&(l, p)) if parity.is_none_or(|q| q == p) => {
                        parity = Some(p);
                        lanes.push(l);
                    }
                    Some(_) => parity = Some(2),
                    None => lanes.push(Lanes::Same(o)),
                }
            }
            let key = (op.kind, format!("{:?}", op.attrs), lanes);
            match parity {
                Some(0) => {
                    first.entry(key).or_insert(i);
                }
                Some(1) => {
                    if let Some(a) = first.remove(&key) {
                        let j = pairs.len();
                        lane.insert(ops[a].results[0], (Lanes::Pair(j), 0));
                        lane.insert(op.results[0], (Lanes::Pair(j), 1));
                        (paired[a], paired[i]) = (true, true);
                        pairs.push((a, i, key.2));
                    }
                }
                _ => {}
            }
        }
        if pairs.len() == before {
            break;
        }
    }
    if pairs.is_empty() {
        return changed;
    }
    let mut at: Vec<Option<usize>> = vec![None; ops.len()];
    for (j, &(a, b, _)) in pairs.iter().enumerate() {
        at[a] = Some(j);
        at[b] = Some(j);
    }
    let vec2 = Type::Vec(Scalar::F16, 2);
    let mut result: Vec<Option<ValueId>> = vec![None; pairs.len()];
    let old = std::mem::take(&mut region.ops);
    let mut out = Vec::with_capacity(old.len() + 4 * pairs.len());
    let op = |kind: OpKind, results: &[ValueId], operands: &[ValueId], attrs: Vec<(Text, Attr)>| Op {
        kind,
        results: results.iter().copied().collect(),
        operands: operands.iter().copied().collect(),
        attrs,
        regions: Vec::new(),
        block_args: Vec::new(),
    };
    let index = |i: u32| vec![("index".into(), Attr::Int(i64::from(i)))];
    let mut taken: Vec<Option<Op>> = old.into_iter().map(Some).collect();
    for i in 0..taken.len() {
        let Some(j) = at[i] else {
            out.push(taken[i].take().expect("an op not paired"));
            continue;
        };
        let (a, b, lanes) = &pairs[j];
        if i != *a {
            continue;
        }
        let mut operands = Vec::with_capacity(lanes.len());
        for &l in lanes {
            operands.push(match l {
                Lanes::Pair(p) => result[p].expect("a pair is made before its readers"),
                Lanes::Of(v, d) => {
                    let (lo, hi) = (kernel.new_value("lo", Type::Scalar(Scalar::F16)), kernel.new_value("hi", Type::Scalar(Scalar::F16)));
                    out.push(op(OpKind::Extract, &[lo], &[v], index(2 * d)));
                    out.push(op(OpKind::Extract, &[hi], &[v], index(2 * d + 1)));
                    let packed = kernel.new_value("pair", vec2.clone());
                    out.push(op(OpKind::Pack, &[packed], &[lo, hi], Vec::new()));
                    packed
                }
                Lanes::Same(x) => {
                    let packed = kernel.new_value("pair", vec2.clone());
                    out.push(op(OpKind::Pack, &[packed], &[x, x], Vec::new()));
                    packed
                }
            });
        }
        let (first, second) = (taken[*a].take().expect("the first of a pair"), taken[*b].take().expect("the second of a pair"));
        let kind = match first.kind {
            OpKind::Fadd => OpKind::VecAdd,
            OpKind::Fmul => OpKind::VecMul,
            _ => OpKind::VecFma,
        };
        let r = kernel.new_value("packed", vec2.clone());
        out.push(op(kind, &[r], &operands, first.attrs.clone()));
        out.push(op(OpKind::Extract, &[first.results[0]], &[r], index(0)));
        out.push(op(OpKind::Extract, &[second.results[0]], &[r], index(1)));
        result[j] = Some(r);
    }
    region.ops = out;
    true
}

/// Uses of each value of `region` and the regions in it.
fn count_uses(region: &Region, uses: &mut [u32]) {
    for op in &region.ops {
        for v in op.operands.iter() {
            uses[v.0 as usize] += 1;
        }
        for inner in &op.regions {
            count_uses(inner, uses);
        }
    }
}

/// Rewrites the 16-bit MFMA operands of `kernel` that a chain of inserts
/// builds whole as packs; returns whether any were.
pub fn pack_mfma_operands(kernel: &mut Kernel) -> bool {
    let mut uses = vec![0u32; kernel.values.len()];
    count_uses(&kernel.body, &mut uses);
    let mut body = std::mem::take(&mut kernel.body);
    let changed = pack_region(kernel, &mut body, &uses);
    kernel.body = body;
    changed
}

fn pack_region(kernel: &Kernel, region: &mut Region, uses: &[u32]) -> bool {
    let mut changed = false;
    for op in region.ops.iter_mut() {
        for inner in op.regions.iter_mut() {
            changed |= pack_region(kernel, inner, uses);
        }
    }
    let ops = &region.ops;
    let inserts: HashMap<ValueId, usize> =
        ops.iter().enumerate().filter(|(_, op)| op.kind == OpKind::Insert && op.results.len() == 1).map(|(i, op)| (op.results[0], i)).collect();
    let global_loads: std::collections::HashSet<ValueId> = ops.iter().filter(|op| op.kind == OpKind::GLoad).flat_map(|op| op.results.iter().copied()).collect();
    let mut heads: Vec<(usize, Vec<ValueId>)> = Vec::new();
    for op in ops.iter().filter(|op| op.kind == OpKind::Mma) {
        for &v in op.operands.iter().take(2) {
            let Type::Vec(elem, n) = *kernel.ty(v) else { continue };
            let Some(&head) = inserts.get(&v) else { continue };
            if elem.bytes() != 2 || heads.iter().any(|(h, _)| *h == head) {
                continue;
            }
            // From the last insert back: the first value seen at an index is the one the vector holds.
            let mut elems: Vec<Option<ValueId>> = vec![None; n as usize];
            let mut cur = v;
            while let Some(&i) = inserts.get(&cur) {
                if cur != v && uses[cur.0 as usize] != 1 {
                    break;
                }
                let Some(index) = ops[i].int_attr("index").filter(|&k| (0..i64::from(n)).contains(&k)) else { break };
                elems[index as usize].get_or_insert(ops[i].operands[1]);
                cur = ops[i].operands[0];
            }
            // Halves an LDS load writes in place cost no merge; those of global loads do on gfx950.
            if let Some(elems) = elems.into_iter().collect::<Option<Vec<_>>>().filter(|e| e.iter().all(|x| global_loads.contains(x))) {
                heads.push((head, elems));
            }
        }
    }
    for (head, elems) in &heads {
        let op = &mut region.ops[*head];
        op.kind = OpKind::Pack;
        op.operands = elems.iter().copied().collect();
        op.attrs.clear();
    }
    changed | !heads.is_empty()
}
#[cfg(test)]
mod tests {
    use crate::rir;
    use crate::target::Target;

    /// Y[8 tid + k] = X[8 tid + k] + Z[8 tid + k] for k < 8, element by element, with `between` after the
    /// fourth pair of loads.
    fn kernel(between: &str) -> String {
        let mut s = String::from(
            "ckdsl.ir v1\nkernel @k {\n  attrs { max_workgroup_size = i:64 }\n  params {\n    %X : ptr<f16,global>\n    \
             %Z : ptr<f16,global>\n    %Y : ptr<f16,global>\n    %W : ptr<f32,global>\n  }\n  region @entry {\n    \
             %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n    %c8 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:8 }\n    \
             %b : i32 = arith.mul ( %tid, %c8 )\n",
        );
        let mut sums = Vec::new();
        for k in 0..8 {
            s += &format!(
                "    %c{k} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{k} }}\n    %i{k} : i32 = arith.add ( %b, %c{k} )\n    \
                 %x{k} : f16 = memref.global_load ( %X, %i{k} ) {{ align = i:2 }}\n    %z{k} : f16 = memref.global_load ( %Z, %i{k} ) {{ align = i:2 }}\n    \
                 %s{k} : f16 = arith.fadd ( %x{k}, %z{k} )\n"
            );
            if k == 3 {
                s += between;
            }
            sums.push(format!("%s{k}"));
        }
        s + &format!(
            "    %v : vec<f16x8> = vector.pack ( {} )\n    memref.global_store_vN ( %Y, %b, %v ) {{ align = i:16, elem_type = s:\"f16\", vec = i:8 }}\n    \
             cf.return (  )\n  }}\n}}\n",
            sums.join(", ")
        )
    }

    fn count(listing: &str, mnemonic: &str) -> usize {
        listing.lines().filter(|l| l.split_whitespace().next() == Some(mnemonic)).count()
    }

    #[test]
    fn adjacent_f16_loads_become_one_load_and_their_adds_pair() {
        let listing = rir::compile_any_target(&kernel(""), Target::Gfx950).unwrap_or_else(|e| panic!("{e}")).listing();
        assert_eq!((count(&listing, "global_load_dwordx4"), count(&listing, "global_load_ushort")), (2, 0), "{listing}");
        // Four packed adds, one per pair of elements, and no shifts to reach a high half.
        assert_eq!((count(&listing, "v_pk_add_f16"), count(&listing, "v_lshrrev_b32")), (4, 0), "{listing}");
    }

    #[test]
    fn an_mfma_operand_of_inserted_loads_packs_each_dword_once() {
        let mut s = String::from(
            "ckdsl.ir v1\nkernel @k {\n  attrs { max_workgroup_size = i:64 }\n  params {\n    %B : ptr<f16,global>\n    %Y : ptr<f32,global>\n  }\n  \
             region @entry {\n    %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n    \
             %z : vec<f16x8> = arith.constant_vec (  ) { elem = s:\"f16\", fill = f:0.0, vec = i:8 }\n    \
             %c : vec<f32x16> = arith.constant_vec (  ) { elem = s:\"f32\", fill = f:0.0, vec = i:16 }\n",
        );
        let mut vector = "%z".to_string();
        for k in 0..8 {
            // A column: the loads are 128 bytes apart, so they stay apart.
            s += &format!(
                "    %o{k} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{} }}\n    %i{k} : i32 = arith.add ( %tid, %o{k} )\n    \
                 %b{k} : f16 = memref.global_load ( %B, %i{k} ) {{ align = i:2 }}\n    %v{k} : vec<f16x8> = vector.insert ( {vector}, %b{k} ) {{ index = i:{k} }}\n",
                64 * k
            );
            vector = format!("%v{k}");
        }
        s += &format!(
            "    %d : vec<f32x16> = tile.mma ( {vector}, {vector}, %c ) {{ op_id = s:\"mfma_f32_32x32x16_f16\" }}\n    \
             %e : f32 = vector.extract ( %d ) {{ index = i:0 }}\n    \
             memref.global_store_typed ( %Y, %tid, %e ) {{ align = i:4, elem_type = s:\"f32\" }}\n    cf.return (  )\n  }}\n}}\n"
        );
        let listing = rir::compile_any_target(&s, Target::Gfx950).unwrap_or_else(|e| panic!("{e}")).listing();
        assert_eq!((count(&listing, "global_load_ushort"), count(&listing, "v_perm_b32")), (8, 4), "{listing}");
    }

    #[test]
    fn only_wave64_kernels_merge_loads() {
        let loads = |wave: u32| {
            let mut k = rir::parse::parse(&kernel("")).unwrap_or_else(|e| panic!("{e}"));
            k.wave = wave;
            let (mut lds, _) = rir::analysis::lds_layout(&k);
            rir::opt::optimize(&mut k, &mut lds).unwrap_or_else(|e| panic!("{e}"));
            k.body.ops.iter().filter(|op| op.kind == rir::ast::OpKind::GLoad).count()
        };
        assert_eq!((loads(32), loads(64)), (16, 2));
    }

    #[test]
    fn a_global_store_between_loads_keeps_them_apart() {
        let store = "    %o : f32 = arith.constant (  ) { ity = s:\"f32\", value = f:1.0 }\n    \
                     memref.global_store_typed ( %W, %tid, %o ) { align = i:4, elem_type = s:\"f32\" }\n";
        let listing = rir::compile_any_target(&kernel(store), Target::Gfx950).unwrap_or_else(|e| panic!("{e}")).listing();
        // Each half of the loads still merges, but not across the store: two dwordx2 loads per tensor.
        assert_eq!((count(&listing, "global_load_dwordx2"), count(&listing, "global_load_dwordx4")), (4, 0), "{listing}");
    }
}
