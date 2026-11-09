// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Vector selects by a splat mask, as scalar selects (gfx950): the census's
//! only vectors of i1 are `vector.splat` of an i1 whose every use is the mask
//! of a `vector.select`. LLVM's select of vectors by a splat mask is the select
//! by its condition, so `vector.select (vector.splat (c), a, b)` becomes
//! `arith.select (c, a, b)`, which selection already gives vector operands,
//! and the splats go. A splat with any other use stays (and is refused).

use crate::rir::analysis::preorder;
use crate::rir::ast::{Kernel, OpKind, Region, Scalar, Type, ValueId};
use crate::rir::fx::{FxHashMap, FxHashSet};

/// Rewrites the selects of `kernel` whose mask is a splat of i1 used only as
/// a select mask.
pub fn fold(kernel: &mut Kernel) {
    let mut splats: FxHashMap<ValueId, ValueId> = FxHashMap::default();
    preorder(&kernel.body, |op| {
        if op.kind == OpKind::VecSplat
            && op.results.len() == 1
            && op.operands.len() == 1
            && matches!(kernel.ty(op.results[0]), Type::Vec(Scalar::I1, _))
        {
            splats.insert(op.results[0], op.operands[0]);
        }
    });
    if splats.is_empty() {
        return;
    }
    let mut other: FxHashSet<ValueId> = FxHashSet::default();
    preorder(&kernel.body, |op| {
        for (i, v) in op.operands.iter().enumerate() {
            if splats.contains_key(v) && !(op.kind == OpKind::VecSelect && i == 0) {
                other.insert(*v);
            }
        }
    });
    splats.retain(|v, _| !other.contains(v));
    if !splats.is_empty() {
        rewrite(&mut kernel.body, &splats);
    }
}

fn rewrite(region: &mut Region, splats: &FxHashMap<ValueId, ValueId>) {
    region.ops.retain(|op| !(op.kind == OpKind::VecSplat && op.results.first().is_some_and(|r| splats.contains_key(r))));
    for op in region.ops.iter_mut() {
        if op.kind == OpKind::VecSelect {
            if let Some(&c) = op.operands.first().and_then(|m| splats.get(m)) {
                op.kind = OpKind::Select;
                op.operands[0] = c;
            }
        }
        for r in op.regions.iter_mut() {
            rewrite(r, splats);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::rir;
    use crate::target::Target;

    /// A guarded choice between two loaded `vec<f16x4>`, written as `select`
    /// lines that define `%r`.
    fn kernel(select: &str) -> String {
        format!(
            "ckdsl.ir v1\nkernel @msel {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n    %X : ptr<f16,global>\n    \
             %Y : ptr<f16,global>\n    %n : i32\n  }}\n  region @entry {{\n    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n    \
             %c4 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:4 }}\n    \
             %c256 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:256 }}\n    \
             %o : i32 = arith.mul ( %tid, %c4 )\n    %o2 : i32 = arith.add ( %o, %c256 )\n    \
             %a : vec<f16x4> = memref.global_load_vN ( %X, %o ) {{ align = i:8, elem_type = s:\"f16\", vec = i:4 }}\n    \
             %b : vec<f16x4> = memref.global_load_vN ( %X, %o2 ) {{ align = i:8, elem_type = s:\"f16\", vec = i:4 }}\n    \
             %lt : i1 = arith.cmp ( %tid, %n ) {{ pred = s:\"lt\" }}\n    {select}\n    \
             memref.global_store_vN ( %Y, %o, %r ) {{ align = i:8, elem_type = s:\"f16\", vec = i:4 }}\n    \
             cf.return (  )\n  }}\n}}\n"
        )
    }

    const SPLAT: &str = "%m : vec<i1x4> = vector.splat ( %lt ) { vec = i:4 }\n    %r : vec<f16x4> = vector.select ( %m, %a, %b )";

    #[test]
    fn a_select_by_a_splat_mask_is_the_select_by_its_condition() {
        let scalar = rir::compile_any_target(&kernel("%r : vec<f16x4> = arith.select ( %lt, %a, %b )"), Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        let splat = rir::compile_any_target(&kernel(SPLAT), Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        assert!(splat.hsaco == scalar.hsaco, "{}\n{}", splat.listing(), scalar.listing());
    }

    #[test]
    fn splats_with_other_uses_and_gfx1151_keep_their_masks() {
        // A splat also passed through a second select's value operand stays.
        let other = "%m : vec<i1x4> = vector.splat ( %lt ) { vec = i:4 }\n    \
                     %m2 : vec<i1x4> = vector.select ( %m, %m, %m )\n    %r : vec<f16x4> = vector.select ( %m2, %a, %b )";
        for (text, target) in [(kernel(other), Target::Gfx950), (kernel(SPLAT), Target::Gfx1151)] {
            let e = rir::compile_any_target(&text, target).err().map(|e| e.to_string()).unwrap_or_default();
            assert!(e.contains("vectors of i1"), "{target:?}: {e}");
        }
    }
}
