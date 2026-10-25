// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! LDS layout (a port of the Python lowering's packer), uniformity, and
//! f32 canonicality.

use super::ast::{Kernel, Op, OpKind, Region, Scalar, Type, ValueId};
use super::fx::{FxHashMap, FxHashSet};
use super::opt::{Facts, Konst};

fn subtree_size(op: &Op) -> usize {
    1 + op
        .regions
        .iter()
        .flat_map(|r| r.ops.iter())
        .map(subtree_size)
        .sum::<usize>()
}

fn smem_liveness(
    ops: &[Op],
    loop_end: Option<usize>,
    counter: &mut usize,
    allocs: &FxHashSet<ValueId>,
    intervals: &mut FxHashMap<ValueId, (usize, usize)>,
    used: &mut FxHashSet<ValueId>,
) {
    for op in ops {
        let idx = *counter;
        *counter += 1;
        if op.kind == OpKind::SmemAlloc {
            intervals.entry(op.results[0]).or_insert((idx, idx));
        }
        for operand in &op.operands {
            if allocs.contains(operand) {
                used.insert(*operand);
                let (first, last) = intervals.get(operand).copied().unwrap_or((idx, idx));
                let new_last = loop_end.unwrap_or(idx);
                intervals.insert(*operand, (first.min(idx), last.max(new_last)));
            }
        }
        for region in &op.regions {
            if op.kind == OpKind::For {
                let own_last = idx + subtree_size(op) - 1;
                let loop_last = loop_end.map_or(own_last, |end| own_last.max(end));
                smem_liveness(&region.ops, Some(loop_last), counter, allocs, intervals, used);
            } else {
                smem_liveness(&region.ops, loop_end, counter, allocs, intervals, used);
            }
        }
    }
}

fn collect_allocs(region: &Region, out: &mut Vec<(ValueId, bool)>) {
    for op in &region.ops {
        if op.kind == OpKind::SmemAlloc {
            out.push((op.results[0], op.bool_attr("exclusive") == Some(true)));
        }
        for inner in &op.regions {
            collect_allocs(inner, out);
        }
    }
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
    smem_liveness(&kernel.body.ops, None, &mut 0, &alloc_set, &mut intervals, &mut used);
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
pub fn classify(kernel: &Kernel) -> Vec<bool> {
    let mut uniform = vec![false; kernel.values.len()];
    for param in &kernel.params {
        uniform[param.value.0 as usize] = true;
    }
    let mut written = false;
    classify_region(kernel, &kernel.body, &mut uniform, &mut written);
    uniform
}

fn writes_global(region: &Region) -> bool {
    region
        .ops
        .iter()
        .any(|op| op.kind.writes_global() || op.regions.iter().any(writes_global))
}

/// Whether a load may use the scalar cache: uniform address, dword-sized
/// elements with dword alignment, and no global write that could precede it.
pub fn smem_eligible(kernel: &Kernel, op: &Op, uniform: &[bool], written: bool) -> bool {
    if op.kind != OpKind::GLoad || written {
        return false;
    }
    let ty = kernel.ty(op.results[0]);
    let elem = ty.elem();
    op.operands.iter().all(|v| uniform[v.0 as usize])
        && matches!(elem, Scalar::I32 | Scalar::F32 | Scalar::I64)
        && op.int_attr("align").unwrap_or(1) >= 4
        && op.int_attr("offset").unwrap_or(0) >= 0
}

fn classify_region(kernel: &Kernel, region: &Region, uniform: &mut Vec<bool>, written: &mut bool) {
    for op in &region.ops {
        let all = |u: &Vec<bool>| op.operands.iter().all(|v| u[v.0 as usize]);
        let result_uniform = match op.kind {
            OpKind::Constant | OpKind::KernArg | OpKind::BlockId | OpKind::ReadFirstLane | OpKind::SmemAlloc => true,
            OpKind::ThreadId => false,
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
            | OpKind::BufferRsrc => all(uniform),
            OpKind::Extract => uniform[op.operands[0].0 as usize],
            OpKind::GLoad => smem_eligible(kernel, op, uniform, *written),
            _ => false,
        };
        match op.kind {
            OpKind::For => {
                let body = &op.regions[0];
                let body_writes = writes_global(body);
                for (i, &arg) in op.block_args.iter().enumerate() {
                    uniform[arg.0 as usize] = if i == 0 { true } else { uniform[op.operands[2 + i].0 as usize] };
                }
                for _ in 0..8 {
                    let mut inner_written = *written || body_writes;
                    classify_region(kernel, body, uniform, &mut inner_written);
                    let yields = &body.ops.last().expect("loop body ends with yield").operands;
                    let mut changed = false;
                    for (i, &arg) in op.block_args.iter().enumerate().skip(1) {
                        if uniform[arg.0 as usize] && !uniform[yields[i - 1].0 as usize] {
                            uniform[arg.0 as usize] = false;
                            changed = true;
                        }
                    }
                    if !changed {
                        break;
                    }
                }
                for (i, &result) in op.results.iter().enumerate() {
                    uniform[result.0 as usize] = uniform[op.block_args[i + 1].0 as usize];
                }
                *written |= body_writes;
            }
            OpKind::If | OpKind::IfElse => {
                for inner in &op.regions {
                    let mut inner_written = *written;
                    classify_region(kernel, inner, uniform, &mut inner_written);
                    *written |= inner_written;
                }
            }
            _ => {
                for &result in &op.results {
                    uniform[result.0 as usize] = result_uniform;
                }
                if op.kind.writes_global() {
                    *written = true;
                }
            }
        }
    }
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
pub fn canonical_f32(kernel: &Kernel, facts: &Facts) -> Vec<bool> {
    let mut canonical = vec![false; kernel.values.len()];
    canonical_region(kernel, facts, &kernel.body, &mut canonical);
    canonical
}

fn canonical_region(kernel: &Kernel, facts: &Facts, region: &Region, canonical: &mut [bool]) {
    for op in &region.ops {
        let operand = |c: &[bool], i: usize| c[op.operands[i].0 as usize];
        let value = match op.kind {
            OpKind::For => {
                canonical_loop(kernel, facts, op, canonical);
                continue;
            }
            OpKind::If | OpKind::IfElse => {
                for inner in &op.regions {
                    canonical_region(kernel, facts, inner, canonical);
                }
                continue;
            }
            OpKind::Constant => match facts.konst.get(op.results[0].0 as usize).copied().flatten() {
                Some(Konst::F32(bits)) => !is_snan_f32(bits),
                Some(Konst::Int(v)) => !is_snan_f32(v as u32),
                None => false,
            },
            OpKind::ConstantVec => {
                let fill = op.float_attr("fill").unwrap_or(0.0);
                match kernel.ty(op.results[0]).elem() {
                    Scalar::F32 => !is_snan_f32((fill as f32).to_bits()),
                    Scalar::I32 => !is_snan_f32(fill as i64 as u32),
                    _ => false,
                }
            }
            OpKind::Fadd
            | OpKind::Fsub
            | OpKind::Fmul
            | OpKind::Fma
            | OpKind::Fmax
            | OpKind::Exp2
            | OpKind::Log2
            | OpKind::Rcp
            | OpKind::CastToF32
            | OpKind::Mma => true,
            OpKind::Bitcast
            | OpKind::Extract
            | OpKind::DsSwizzleXor
            | OpKind::ReadFirstLane
            | OpKind::InlineAsmIdentity => operand(canonical, 0),
            OpKind::Select => operand(canonical, 1) && operand(canonical, 2),
            OpKind::Insert | OpKind::Pack | OpKind::Concat => {
                op.operands.iter().all(|v| canonical[v.0 as usize])
            }
            _ => false,
        };
        for &result in &op.results {
            canonical[result.0 as usize] = value && word_elements(kernel.ty(result));
        }
    }
}

/// Iteration arguments start from their initial values and stay canonical
/// while every yield is, assuming the arguments are (induction on the trip).
fn canonical_loop(kernel: &Kernel, facts: &Facts, op: &Op, canonical: &mut [bool]) {
    let body = &op.regions[0];
    let args = &op.block_args[1..];
    for (i, &arg) in args.iter().enumerate() {
        canonical[arg.0 as usize] = canonical[op.operands[3 + i].0 as usize] && word_elements(kernel.ty(arg));
    }
    for round in 0.. {
        canonical_region(kernel, facts, body, canonical);
        let yields = &body.ops.last().expect("loop body ends with yield").operands;
        let mut changed = false;
        for (i, &arg) in args.iter().enumerate() {
            // Giving up after a few rounds bounds the cost of nested loops.
            if canonical[arg.0 as usize] && (round == 8 || !canonical[yields[i].0 as usize]) {
                canonical[arg.0 as usize] = false;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    for (i, &result) in op.results.iter().enumerate() {
        canonical[result.0 as usize] = canonical[args[i].0 as usize];
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
}
