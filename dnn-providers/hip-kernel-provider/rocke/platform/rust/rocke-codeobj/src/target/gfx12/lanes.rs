// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Kernels that state a wave wider than gfx12's.
//!
//! A kernel written for a 64-lane wave can say so in its IR: it permutes
//! across the wave with `tile.ds_bpermute` from the lane
//! `tile.lane_id() ^ 32` (or `+ 32`, `| 32`), the other half of a wave64.
//! A wave32 has no such lane, so the code would compute something else than
//! the kernel means. `check` refuses exactly that: a `ds_bpermute` whose
//! source lane is the reader's own lane id combined with a constant that
//! puts it at or past the wave size. Any other `ds_bpermute` address is the
//! kernel's business.

use crate::error::CompileError;
use crate::rir::analysis::preorder;
use crate::rir::ast::{Kernel, Op, OpKind, ValueId};

/// `Err(ROCKE_DIRECT_UNSUPPORTED)` when `kernel` reads, with
/// `tile.ds_bpermute`, a lane that no wave of `kernel.wave` lanes has.
pub fn check(kernel: &Kernel) -> Result<(), CompileError> {
    let mut defs: Vec<Option<&Op>> = vec![None; kernel.values.len()];
    let mut permutes = Vec::new();
    preorder(&kernel.body, |op| {
        if let [result] = op.results[..] {
            defs[result.0 as usize] = Some(op);
        }
        if op.kind == OpKind::DsBpermute {
            permutes.push(op);
        }
    });
    let def = |v: ValueId| defs.get(v.0 as usize).copied().flatten();
    let constant = |v: ValueId| def(v).filter(|op| op.kind == OpKind::Constant).and_then(|op| op.int_attr("value"));
    let wave = i64::from(kernel.wave);
    for permute in permutes {
        // The address is the source lane times four.
        let Some(lane) = permute.operands.first().and_then(|&address| def(address)).and_then(|scaled| {
            let [x, k] = scaled.operands[..] else { return None };
            match (scaled.kind, constant(k)) {
                (OpKind::Shl, Some(2)) | (OpKind::Mul, Some(4)) => def(x),
                _ => None,
            }
        }) else {
            continue;
        };
        let [a, b] = lane.operands[..] else { continue };
        for (id, other) in [(a, b), (b, a)] {
            let own_lane = def(id).is_some_and(|op| op.kind == OpKind::LaneId);
            let past = match (lane.kind, constant(other)) {
                // Lane ids are below the wave size, a power of two.
                (OpKind::Xor | OpKind::Or, Some(k)) => k > 0 && k & !(wave - 1) != 0,
                (OpKind::Add, Some(k)) => k >= wave,
                _ => false,
            };
            if own_lane && past {
                return Err(CompileError::new(
                    "ROCKE_DIRECT_UNSUPPORTED",
                    format!("tile.ds_bpermute reads a lane past the {wave} lanes of a wave: the kernel is written for a wider wave"),
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::target::Target;

    fn kernel(lane: &str) -> String {
        format!(
            "ckdsl.ir v1\nkernel @probe {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n    %Y : ptr<f32,global>\n  }}\n  \
             region @entry {{\n    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n    %l : i32 = tile.lane_id (  )\n    \
             %c32 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:32 }}\n    \
             %c1 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:1 }}\n    \
             %c2 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:2 }}\n    {lane}\n    \
             %a : i32 = arith.shl ( %s, %c2 )\n    %p : i32 = tile.ds_bpermute ( %a, %tid )\n    \
             %e : f32 = arith.bitcast ( %p ) {{ target = s:\"f32\" }}\n    \
             memref.global_store_typed ( %Y, %tid, %e ) {{ align = i:4, elem_type = s:\"f32\" }}\n    cf.return (  )\n  }}\n}}\n"
        )
    }

    #[test]
    fn a_permute_from_the_other_half_of_a_wave64_is_refused_on_gfx12() {
        let compile = |lane: &str, target| crate::rir::compile_any_target(&kernel(lane), target);
        for other_half in ["%s : i32 = arith.xor ( %l, %c32 )", "%s : i32 = arith.add ( %c32, %l )", "%s : i32 = arith.or ( %l, %c32 )"] {
            for target in [Target::Gfx1201, Target::Gfx12Generic] {
                let error = compile(other_half, target).err().unwrap_or_else(|| panic!("{target:?} accepted {other_half}"));
                assert_eq!(error.kind, "ROCKE_DIRECT_UNSUPPORTED", "{error}");
                assert!(error.message.contains("wider wave"), "{error}");
            }
            // gfx950's wave has that half.
            compile(other_half, Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        }
        // Permutes within the wave, and by anything that is not the lane id
        // plus a constant, compile.
        for within in ["%s : i32 = arith.xor ( %l, %c1 )", "%s : i32 = arith.add ( %l, %c1 )", "%s : i32 = arith.xor ( %tid, %c32 )"] {
            for target in [Target::Gfx1201, Target::Gfx12Generic] {
                compile(within, target).unwrap_or_else(|e| panic!("{target:?}: {within}: {e}"));
            }
        }
    }
}
