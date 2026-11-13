// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Which f16 conversions of f32 products and FMAs round once, on gfx950.
//!
//! LLVM selects `fptrunc (fma a, b, c)` and `fptrunc (fmul a, b)` to f16 as
//! `v_fma_mixlo_f16 d, a, b, c` (`c` is -0.0 for the product): the hardware
//! rounds the exact result to f16, where `v_fma_f32` and `v_cvt_f16_f32` round
//! it to f32 first and then to f16, which differs by one f16 ulp when the f32
//! result lands on an f16 tie (about 1 element in 10^5 of a depthwise
//! convolution's output). `rir::isel` selects the conversions this marks as one
//! `v_fma_mixlo_f16` (`scalar_trunc_mix`); a product or FMA that other
//! instructions read stays for them and is computed again inside it, as LLVM
//! does.
//!
//! Whether LLVM does so depends on what it sees of the kernel, which the
//! direct compiler's own passes change (a loop it unrolls leaves the last FMA
//! of its carried value beside the conversion, a phi in LLVM's IR), so the
//! conversions are marked on the kernel as parsed, before optimization. A
//! scalar `arith.trunc_f32_to_f16` / `arith.cast_f32_to` (f16) of an f32
//! `arith.fma` or `arith.fmul` of the same region is marked when
//! - every reader of the f16 is a `tile.buffer_store_*` (an intrinsic call,
//!   which LLVM's SLP vectorizer takes no seed from): the 12 to 176 conversions
//!   of a depthwise convolution tile, all of them one `v_fma_mixlo_f16` in
//!   COMGR's code, or
//! - every reader is a plain store (`memref.global_store_typed`,
//!   `tile.smem_store`), and it is the only such conversion of its region:
//!   LLVM vectorizes the stores of two or more, with their products, as a
//!   `<2 x float>` multiply and `v_cvt_pk_f16_f32`, and leaves one alone (the
//!   output of an unified attention's reduction: one element of a head of 64
//!   is one `v_fma_mixlo_f16`, two or four are `v_pk_mul_f32` and
//!   `v_cvt_pk_f16_f32`).
//!
//! The conversions that `rir::isel::find_f16_packs` makes pairs of are not
//! these. `ROCKE_DIRECT_DEBUG=nomixtrunc` marks none.

use crate::rir::ast::{Attr, Kernel, Op, OpKind, Region, Scalar, Type, ValueId};
use crate::rir::fx::{FxHashMap, FxHashSet};

/// The op attribute marking a conversion `rir::isel` selects as `v_fma_mixlo_f16`.
pub const MARK: &str = "mixtrunc";

/// Marks the conversions of `kernel` that LLVM selects as one `v_fma_mixlo_f16` (see the module).
pub fn mark(kernel: &mut Kernel) {
    if crate::rir::debug_flag("nomixtrunc") {
        return;
    }
    let f32_scalar = Type::Scalar(Scalar::F32);
    let mut products: FxHashSet<ValueId> = FxHashSet::default();
    let mut readers: FxHashMap<ValueId, Vec<OpKind>> = FxHashMap::default();
    crate::rir::analysis::preorder(&kernel.body, |op| {
        if matches!(op.kind, OpKind::Fmul | OpKind::Fma)
            && op.results.len() == 1
            && *kernel.ty(op.results[0]) == f32_scalar
            && op.operands.iter().all(|&o| *kernel.ty(o) == f32_scalar)
        {
            products.insert(op.results[0]);
        }
        for &v in &op.operands {
            readers.entry(v).or_default().push(op.kind);
        }
    });
    let mut marked = FxHashSet::default();
    collect(&kernel.body, &products, &readers, &mut marked);
    if !marked.is_empty() {
        set(&mut kernel.body, &marked);
    }
}

/// Whether `op` converts an f32 to f16 (scalar).
fn is_f16_conversion(op: &Op) -> bool {
    match op.kind {
        OpKind::TruncF32ToF16 => true,
        OpKind::CastF32To => op.str_attr("target") == Some("f16"),
        _ => false,
    }
}

fn collect(region: &Region, products: &FxHashSet<ValueId>, readers: &FxHashMap<ValueId, Vec<OpKind>>, marked: &mut FxHashSet<ValueId>) {
    // The conversions whose readers are plain stores, which LLVM may vectorize with each other.
    let mut plain = Vec::new();
    for op in &region.ops {
        for inner in &op.regions {
            collect(inner, products, readers, marked);
        }
        if !is_f16_conversion(op) || op.results.len() != 1 || !products.contains(&op.operands[0]) {
            continue;
        }
        let result = op.results[0];
        let Some(kinds) = readers.get(&result).filter(|k| !k.is_empty()) else { continue };
        if kinds.iter().all(|&k| k == OpKind::BufferStoreScalar) {
            marked.insert(result);
        } else if kinds.iter().all(|&k| matches!(k, OpKind::GlobalStore | OpKind::SmemStore)) {
            plain.push(result);
        }
    }
    if let [only] = plain[..] {
        marked.insert(only);
    }
}

fn set(region: &mut Region, marked: &FxHashSet<ValueId>) {
    for op in region.ops.iter_mut() {
        for inner in op.regions.iter_mut() {
            set(inner, marked);
        }
        if op.results.first().is_some_and(|r| marked.contains(r)) {
            op.attrs.push((MARK.into(), Attr::Bool(true)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A kernel with `body` (after three f32 loads `%a`, `%b`, `%c`).
    fn marked(body: &str) -> Vec<(OpKind, bool)> {
        let text = format!(
            "ckdsl.ir v1\nkernel @k {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n    %X : ptr<f32,global>\n    \
             %Y : ptr<f16,global>\n  }}\n  region @entry {{\n    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n    \
             %a : f32 = memref.global_load_typed ( %X, %tid ) {{ align = i:4, elem_type = s:\"f32\" }}\n    \
             %b : f32 = memref.global_load_typed ( %X, %tid ) {{ align = i:4, elem_type = s:\"f32\" }}\n    \
             %c : f32 = memref.global_load_typed ( %X, %tid ) {{ align = i:4, elem_type = s:\"f32\" }}\n{body}    \
             cf.return (  )\n  }}\n}}\n"
        );
        let mut kernel = crate::rir::parse::parse(&text).unwrap_or_else(|e| panic!("{e}"));
        mark(&mut kernel);
        let mut found = Vec::new();
        crate::rir::analysis::preorder(&kernel.body, |op| {
            if is_f16_conversion(op) {
                found.push((op.kind, op.attr(MARK).is_some()));
            }
        });
        found
    }

    const FMA: &str = "    %m : f32 = arith.fma ( %a, %b, %c )\n";
    const MUL: &str = "    %m : f32 = arith.fmul ( %a, %b )\n";
    const TRUNC: &str = "    %h : f16 = arith.trunc_f32_to_f16 ( %m )\n";
    const STORE: &str = "    memref.global_store_typed ( %Y, %tid, %h ) { align = i:2, elem_type = s:\"f16\" }\n";

    #[test]
    fn a_conversion_read_by_a_buffer_store_is_marked() {
        let body = format!(
            "{FMA}{TRUNC}    %rs : vec<i32x4> = tile.buffer_rsrc ( %Y, %tid )\n    \
             %c0 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:0 }}\n    \
             tile.buffer_store_f16 ( %rs, %tid, %c0, %h )\n"
        );
        assert_eq!(marked(&body), [(OpKind::TruncF32ToF16, true)]);
    }

    #[test]
    fn a_lone_conversion_read_by_a_store_is_marked_and_two_are_not() {
        let one = format!("{MUL}{TRUNC}{STORE}");
        assert_eq!(marked(&one), [(OpKind::TruncF32ToF16, true)]);
        let two = format!(
            "{MUL}{TRUNC}{STORE}    %m2 : f32 = arith.fmul ( %b, %c )\n    %h2 : f16 = arith.trunc_f32_to_f16 ( %m2 )\n    \
             memref.global_store_typed ( %Y, %tid, %h2 ) {{ align = i:2, elem_type = s:\"f16\" }}\n"
        );
        assert_eq!(marked(&two), [(OpKind::TruncF32ToF16, false), (OpKind::TruncF32ToF16, false)]);
    }

    #[test]
    fn other_values_and_readers_are_not_marked() {
        let sum = "    %m : f32 = arith.fadd ( %a, %b )\n";
        assert_eq!(marked(&format!("{sum}{TRUNC}{STORE}")), [(OpKind::TruncF32ToF16, false)]);
        // read by something that is no store
        let other = format!("{MUL}{TRUNC}    %x : f32 = arith.cast_to_f32 ( %h )\n");
        assert_eq!(marked(&other), [(OpKind::TruncF32ToF16, false)]);
    }
}
