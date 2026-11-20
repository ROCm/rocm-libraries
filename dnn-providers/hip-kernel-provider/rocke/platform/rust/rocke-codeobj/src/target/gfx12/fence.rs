// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! GFX12.0 HSA fences, in WGP mode (the mode in our descriptor). See
//! LLVM AMDGPUUsage, “Memory Model GFX12”, and SIGfx12CacheControl in
//! SIMemoryLegalizer. rocKE's scopes are cross-address-space, not `-one-as`.

use super::{inst, Gfx12Op};
use crate::error::CompileError;
use crate::rir::analysis::preorder;
use crate::rir::ast::{Kernel, Op, OpKind, Scalar, Type};
use crate::rir::mir::{MInst, Opc};

const SE: i64 = 1;
const DEV: i64 = 2;
const SYS: i64 = 3;

pub fn workgroup_barrier(op: &Op) -> bool {
    matches!(op.kind, OpKind::Sync | OpKind::SyncLdsOnly | OpKind::BarrierBare) && op.str_attr("fence") == Some("workgroup")
}

/// Mutable data in a communicating kernel must not use the incoherent
/// scalar cache. Only kernargs and loads from constant memory may use it.
/// A fence also cannot fix a global RMW issued with the wrong SCOPE bits.
/// Check before optimization so neither fact depends on unrolling or DCE.
pub fn check_kernel(k: &Kernel) -> Result<bool, CompileError> {
    let (mut fence, mut unqualified_atomic) = (false, false);
    preorder(&k.body, |op| {
        fence |= op.kind == OpKind::Fence || workgroup_barrier(op);
        match op.kind {
            // The i32 form lowers to a system-scope atomicrmw. Unlike
            // floating-point adds it needs no allocation or FP-mode promises.
            OpKind::GlobalAtomicAdd | OpKind::GAtomicAdd => {
                unqualified_atomic |= op.attr("scope").is_some()
                    || !op.results.first().is_some_and(|&v| *k.ty(v) == Type::Scalar(Scalar::I32));
            }
            OpKind::GlobalAtomicAddF32 | OpKind::GAtomicAddF32 => unqualified_atomic = true,
            _ => {}
        }
    });
    if fence && unqualified_atomic {
        return Err(CompileError::new("ROCKE_DIRECT_UNSUPPORTED", "gfx12 global atomic scope in a fenced kernel"));
    }
    Ok(fence)
}

/// Insert the optional workgroup rendezvous *between* a release and an
/// acquire. The release already drained this wave; the rendezvous is not
/// itself a memory fence and does not replace the invalidate.
pub fn emit(scope: &str, order: &str, barrier: Option<Opc>, mut push: impl FnMut(MInst)) -> Result<(), CompileError> {
    let scope = match scope {
        "workgroup" => SE, // A workgroup in WGP mode can span two CUs.
        "agent" => DEV,
        "system" => SYS,
        _ => return Err(CompileError::new("ROCKE_DIRECT_UNSUPPORTED", format!("gfx12 fence scope {scope}"))),
    };
    let (release, acquire) = match order {
        "acquire" => (false, true),
        "release" => (true, false),
        // LLVM's SC *fence* sequence, not a rule for SC atomic accesses.
        "acq_rel" | "seq_cst" => (true, true),
        _ => return Err(CompileError::new("ROCKE_DIRECT_UNSUPPORTED", format!("gfx12 fence ordering {order}"))),
    };
    if release && scope == SYS {
        push(inst(Gfx12Op::GlobalWb, SYS)); // STOREcnt includes the write-back.
    }
    // These are all the mutable memory pipelines this target can issue:
    // VMEM load/store and LDS. No image/sample/BVH instructions or opaque
    // asm. Unlike LLVM we do not elide waits for provably idle counters.
    push(inst(Gfx12Op::SWaitStorecnt, 0));
    push(inst(Gfx12Op::SWaitLoadcntDscnt, 0));
    if let Some(op) = barrier {
        push(MInst::new(op, [], []));
    }
    if acquire {
        push(inst(Gfx12Op::GlobalInv, scope));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::rir::compile_for;
    use crate::target::Target;

    // Both uniform-address loads are mutable and must go through VMEM.
    // The store before the fence and the independent load after it can
    // otherwise change places (the pointers explicitly do not alias).
    fn kernel(site: &str, group: u32) -> String {
        format!(r#"ckdsl.ir v1
kernel @exchange {{
  attrs {{ max_workgroup_size = i:{group} }}
  params {{
    %X : ptr<i32,global> {{ noalias = b:true }}
    %Y : ptr<i32,global> {{ noalias = b:true }}
  }}
  region @entry {{
    %tid : i32 = gpu.thread_id (  ) {{ axis = s:"x" }}
    %z : i32 = arith.constant (  ) {{ ity = s:"i32", value = i:0 }}
    %a : i32 = memref.global_load_typed ( %X, %z ) {{ align = i:4, elem_type = s:"i32" }}
    memref.global_store_typed ( %Y, %tid, %a ) {{ align = i:4, elem_type = s:"i32" }}
    {site}
    %b : i32 = memref.global_load_typed ( %X, %z ) {{ align = i:4, elem_type = s:"i32" }}
    memref.global_store_typed ( %Y, %tid, %b ) {{ align = i:4, elem_type = s:"i32" }}
    cf.return (  )
  }}
}}
"#)
    }

    #[test]
    fn standalone_fences_match_the_gfx12_wgp_sequences() {
        for (scope, bits) in [("workgroup", "00040000"), ("agent", "00080000"), ("system", "000C0000")] {
            for order in ["acquire", "release", "acq_rel", "seq_cst"] {
                let site = format!(r#"memref.fence (  ) {{ ordering = s:"{order}", scope = s:"{scope}" }}"#);
                for target in [Target::Gfx1201, Target::Gfx12Generic] {
                    let listing = compile_for(&kernel(&site, 128), target).unwrap_or_else(|e| panic!("{e}")).listing();
                    // The final scheduled stream, not isel output. COMGR reference
                    // words with global+LDS traffic (no images or BVH queries).
                    let inv = format!("EE0AC07C {bits} 00000000");
                    let mut expected = vec!["global_load_b32", "global_store_b32"];
                    if scope == "system" && order != "acquire" { expected.push("EE0B007C 000C0000 00000000"); }
                    expected.extend(["BFC10000", "BFC80000"]);
                    if order != "release" { expected.push(&inv); }
                    expected.extend(["global_load_b32", "global_store_b32"]);
                    let mut rest = listing.as_str();
                    for part in expected {
                        rest = rest.split_once(part).unwrap_or_else(|| panic!("missing {part} at {site}:\n{listing}")).1;
                    }
                    assert_eq!(listing.matches("global_inv").count(), usize::from(order != "release"));
                    assert_eq!(listing.matches("global_wb").count(), usize::from(scope == "system" && order != "acquire"));
                    assert!(!listing.contains("barrier"));
                }
            }
        }
    }

    #[test]
    fn uniform_i32_metadata_uses_vmem_without_broadcasting_divergent_values() {
        let site = r#"memref.fence (  ) { scope = s:"agent", ordering = s:"acquire" }"#;
        let uniform = kernel(site, 128);
        for (ir, scalarize) in [(uniform.clone(), true), (uniform.replace("%X, %z", "%X, %tid"), false)] {
            let listing = compile_for(&ir, Target::Gfx1201).unwrap_or_else(|e| panic!("{e}")).listing();
            assert_eq!(listing.matches("global_load_b32").count(), 2, "{listing}");
            assert_eq!(listing.contains("v_readfirstlane_b32"), scalarize, "{listing}");
        }
    }

    #[test]
    fn fenced_barrier_keeps_both_memory_edges_even_for_one_wave() {
        for group in [32, 128] {
            for op in ["tile.sync", "tile.sync_lds_only", "tile.s_barrier_bare"] {
                let site = format!(r#"{op} (  ) {{ fence = s:"workgroup" }}"#);
                let listing = compile_for(&kernel(&site, group), Target::Gfx1201).unwrap_or_else(|e| panic!("{e}")).listing();
                let mut rest = listing.as_str();
                for word in ["global_store_b32", "BFC10000", "BFC80000", "BE804EC1", "BF94FFFF", "EE0AC07C 00040000 00000000", "global_load_b32"] {
                    rest = rest.split_once(word).unwrap_or_else(|| panic!("missing {word}:\n{listing}")).1;
                }
            }
        }
    }

    #[test]
    fn malformed_fences_and_unqualified_global_atomics_are_not_approximated() {
        for site in [r#"memref.fence (  ) { scope = s:"agent" }"#,
                     r#"memref.fence (  ) { scope = s:"wavefront", ordering = s:"acquire" }"#,
                     r#"memref.fence (  ) { scope = s:"agent", ordering = s:"monotonic" }"#] {
            assert_eq!(compile_for(&kernel(site, 64), Target::Gfx1201).err().unwrap().kind, "ROCKE_DIRECT_INVALID");
        }
        let rmw = "%r : i32 = memref.global_atomic_add ( %Y, %tid, %a )";
        for site in [format!("{rmw} {{ ordering = s:\"release\" }}"),
                     format!("{rmw} {{ scope = s:\"workgroup\" }}\n    memref.fence (  ) {{ scope = s:\"agent\", ordering = s:\"acquire\" }}")] {
            let e = compile_for(&kernel(&site, 64), Target::Gfx12Generic).err().unwrap();
            assert_eq!(e.kind, "ROCKE_DIRECT_UNSUPPORTED", "{e}");
            assert!(e.message.contains("atomic"), "{e}");
        }
    }
}
