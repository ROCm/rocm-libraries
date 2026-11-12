// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! 64-bit integer arithmetic that only its low dword reaches (gfx950).
//!
//! rocKE's LDS-DMA addresses are `smem_ptr_add(base, ring * 17408 + slab)`
//! in i64, where `ring` is the zero-extended ring-buffer slot. LDS pointers
//! are 32 bits, so [`lds_dma::Facts::low`] already computes only the low
//! dword of the `smem_ptr_add`; its offset operand was still built whole: an
//! `s_mul_i32` + `s_mul_hi_u32` for the product and a 64-bit add (a VALU
//! `v_lshl_add_u64` with two `v_mov` for the halves) for every one of the
//! sixteen loads of a K/V tile, each loop iteration.
//!
//! The i64 `add`, `sub`, `mul`, `and`, `or` and `xor` are exact on the low
//! dword (no carry or borrow moves up), so a value of them whose every use
//! reads at most its low dword is computed in 32 bits, as LLVM's demanded-bits
//! narrowing does. Uses that read only the low dword: the offset of a
//! low-dword `smem_ptr_add`, an operand of another such op, a truncation to
//! i32. Any other use (a compare, a store, a loop-carried value, a shift)
//! keeps the 64-bit computation.

use crate::rir::ast::{Kernel, OpKind, Scalar, Type, ValueId};
use crate::rir::fx::{FxHashMap, FxHashSet};

fn narrowable(kind: OpKind) -> bool {
    matches!(kind, OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::And | OpKind::Or | OpKind::Xor)
}

/// The i64 results of narrowable ops that only their low dword reaches, given
/// `low`, the LDS addresses [`lds_dma::Facts`] computes in one dword.
pub fn of(kernel: &Kernel, low: &FxHashSet<ValueId>) -> FxHashSet<ValueId> {
    let is_i64 = |v: ValueId| *kernel.ty(v) == Type::Scalar(Scalar::I64);
    let mut candidates: FxHashSet<ValueId> = FxHashSet::default();
    crate::rir::analysis::preorder(&kernel.body, |op| {
        if narrowable(op.kind) && op.results.len() == 1 && is_i64(op.results[0]) {
            candidates.insert(op.results[0]);
        }
    });
    if candidates.is_empty() || low.is_empty() {
        return FxHashSet::default();
    }
    // Each candidate's uses: (using op's kind, the using op's result).
    let mut uses: FxHashMap<ValueId, Vec<(OpKind, Option<ValueId>, bool)>> = FxHashMap::default();
    crate::rir::analysis::preorder(&kernel.body, |op| {
        for v in op.operands.iter().filter(|v| candidates.contains(v)) {
            let to_i32 = op.kind == OpKind::Trunc && op.results.first().is_some_and(|&r| *kernel.ty(r) == Type::Scalar(Scalar::I32));
            uses.entry(*v).or_default().push((op.kind, op.results.first().copied(), to_i32));
        }
    });
    let mut narrow = candidates;
    loop {
        let dropped: Vec<ValueId> = narrow
            .iter()
            .copied()
            .filter(|v| match uses.get(v) {
                None => true,
                Some(list) => list.iter().any(|&(kind, result, to_i32)| match kind {
                    OpKind::SmemPtrAdd => !result.is_some_and(|r| low.contains(&r)),
                    OpKind::Trunc => !to_i32,
                    k if narrowable(k) => !result.is_some_and(|r| narrow.contains(&r)),
                    _ => true,
                }),
            })
            .collect();
        if dropped.is_empty() {
            return narrow;
        }
        for v in dropped {
            narrow.remove(&v);
        }
    }
}
