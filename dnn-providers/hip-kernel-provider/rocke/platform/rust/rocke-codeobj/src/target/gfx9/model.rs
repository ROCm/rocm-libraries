// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! gfx950 machine model for the scheduler (design section 8.5): the result
//! latency of an instruction, in the unit the scheduler counts, one issued
//! instruction of one wave. A wave64 VALU instruction occupies its SIMD for
//! several shader cycles, so a dependent VALU instruction can issue right
//! after its producer; memory results take many such slots.
//!
//! The classes were fitted on gfx950 with the `mb950` microbenchmarks (one
//! wave timing chains of dependent instructions against independent ones):
//! a dependent VALU chain issues about as fast as independent instructions,
//! including 32-bit integer multiplies and f64 adds (full rate there);
//! transcendentals take about two slots; an LDS load returns after about
//! twenty, `ds_swizzle`/`ds_bpermute` after about ten, and a vector memory
//! load after about 45 from L2 and 180 from beyond the caches, which the
//! streaming kernels see.
//!
//! The hazard recognizer's wait states (`hazards.rs`) are part of a result's
//! latency where the producer alone determines them, so the scheduler can
//! fill them with independent instructions instead of `s_nop`s: a VALU write
//! of an SGPR read by a VALU (H1, 2 wait states), a transcendental result
//! (H8, 1) and a forwarded destination (H9, 1).

use super::gen_isa as t;
use crate::rir::mir::*;

/// A VALU instruction's result.
const VALU: u32 = 1;
/// A transcendental's result, and one forwarded through dst-sel logic.
const TRANS: u32 = 2;
/// An SGPR (or VCC) written by a VALU instruction.
const VALU_SGPR: u32 = 3;
const SALU: u32 = 1;
const SMEM: u32 = 10;
const LDS: u32 = 20;
const LDS_CROSS_LANE: u32 = 11;
const VMEM_LOAD: u32 = 180;

/// Issue slots from `inst`'s issue until a reader of its result may issue
/// without stalling (the gfx950 counterpart of `post::result_latency`).
pub fn result_latency(inst: &MInst) -> u32 {
    let op = inst.op;
    let flags = op.flags();
    let Some(info) = t::info(op) else {
        // Pseudos: copies and the like, expanded into moves (or nothing).
        return if flags & F_USE_EXEC != 0 { VALU } else { 0 };
    };
    if flags & F_VMEM_LOAD != 0 {
        return VMEM_LOAD;
    }
    if flags & F_LDS != 0 {
        return if inst.defs.is_empty() {
            0
        } else if matches!(op, Opc::DsSwizzleB32 | Opc::DsBpermuteB32) {
            LDS_CROSS_LANE
        } else {
            LDS
        };
    }
    if flags & F_SMEM != 0 {
        return SMEM;
    }
    if let Some(waits) = super::hazards::mfma_result_latency(op) {
        return waits;
    }
    match info.fmt {
        Fmt::Sop1 | Fmt::Sop2 | Fmt::Sopc | Fmt::Sopk => SALU,
        Fmt::Vopc | Fmt::Vop3b => VALU_SGPR,
        Fmt::Vop1 | Fmt::Vop2 | Fmt::Vop3 | Fmt::Vop3p => {
            if matches!(op, Opc::VReadfirstlaneB32 | Opc::VReadlaneB32) {
                VALU_SGPR
            } else if flags & F_TRANS != 0
                || info.fmt == Fmt::Vop3p
                || matches!(op, Opc::VFmaMixloF16 | Opc::VFmaMixhiF16)
                || inst.mods & M_OPSEL_DST != 0
            {
                TRANS
            } else {
                VALU
            }
        }
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inst(op: Opc) -> MInst {
        MInst::new(op, [], [])
    }

    fn with_def(op: Opc) -> MInst {
        MInst::new(op, [Opnd::P(Phys::V(0), 1)], [])
    }

    #[test]
    fn latencies_follow_the_classes_measured_on_gfx950() {
        let l = |i: &MInst| result_latency(i);
        // Dependent VALU instructions issue back to back, integer multiplies
        // and f64 adds included.
        assert_eq!(l(&inst(Opc::VAddF32)), 1);
        assert_eq!(l(&inst(Opc::VMulLoU32)), 1);
        assert_eq!(l(&inst(Opc::VAddF64)), 1);
        // Results the hazard recognizer would wait for.
        assert_eq!(l(&inst(Opc::VExpF32)), 2);
        assert_eq!(l(&inst(Opc::VPkMulF16)), 2);
        assert_eq!(l(&inst(Opc::VCmp(0x44))), 3);
        assert_eq!(l(&inst(Opc::VReadfirstlaneB32)), 3);
        // Memory, by level.
        assert_eq!(l(&with_def(Opc::DsLoadB32)), 20);
        assert_eq!(l(&with_def(Opc::DsBpermuteB32)), 11);
        assert_eq!(l(&inst(Opc::DsStoreB32)), 0);
        assert!(l(&with_def(Opc::GLoadB32)) > 8 * l(&with_def(Opc::DsLoadB32)));
        assert_eq!(l(&inst(Opc::SAddU32)), 1);
        assert_eq!(l(&inst(Opc::Copy)), 0);
    }
}
