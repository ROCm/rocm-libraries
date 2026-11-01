// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! AGPRs as spill space on gfx950 (`rir::spill`): spilled VGPR webs take
//! AGPR slots before LDS and scratch, as far as the unified register file
//! (architected VGPRs from v0, AGPRs from `ACCUM_OFFSET`) keeps the waves the
//! architected VGPRs leave. A store is `v_accvgpr_write_b32`, a reload
//! `v_accvgpr_read_b32`: VALU instructions, under the EXEC of the store
//! like the LDS and scratch accesses they replace.

use super::asm::AGPR0;
use super::gen_isa::Gfx9Op;
use crate::rir::mir::*;
use crate::target::TargetInfo;

/// The AGPRs that cost no waves next to `vgprs` architected VGPRs, with
/// `lds` bytes of LDS per workgroup of `wg` lanes: the unified count
/// `align4(vgprs) + slots` stays within what the waves of `vgprs` allow.
pub fn agpr_slots(t: &TargetInfo, vgprs: u32, lds: u32, wg: u32) -> u32 {
    let waves = t.occupancy(vgprs.max(1), lds, wg).max(1);
    let unified = t.vgprs_per_simd / waves / t.vgpr_granule * t.vgpr_granule;
    unified.saturating_sub(vgprs.next_multiple_of(4)).min(256)
}

/// The move of one dword between VGPR `vgpr` and AGPR slot `slot`: a reload
/// into the VGPR when `load`, else a store from it.
pub fn agpr_move(load: bool, vgpr: u16, slot: u32) -> MInst {
    let (v, a) = (Opnd::P(Phys::V(vgpr), 1), Opnd::P(Phys::V(AGPR0 + slot as u16), 1));
    if load {
        MInst::new(Opc::Gfx9(Gfx9Op::VAccvgprReadB32), [v], [a])
    } else {
        MInst::new(Opc::Gfx9(Gfx9Op::VAccvgprWriteB32), [a], [v])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::GFX950;

    #[test]
    fn slots_keep_the_waves_of_the_architected_vgprs() {
        // 16 or 12 VGPRs: eight waves, 64 unified registers.
        assert_eq!((agpr_slots(&GFX950, 16, 0, 256), agpr_slots(&GFX950, 12, 0, 256)), (48, 52));
        // 256 VGPRs give two waves, which 256 registers fill.
        assert_eq!(agpr_slots(&GFX950, 256, 0, 256), 0);
        // LDS holding one 256-lane workgroup per CU: one wave, all 256 AGPRs.
        assert_eq!(agpr_slots(&GFX950, 256, 96 << 10, 256), 256);
        // 100 VGPRs: four waves, which 128 registers keep: 28 slots.
        assert_eq!(agpr_slots(&GFX950, 100, 0, 256), 28);
        // 60 VGPRs: eight waves need at most 64: four slots.
        assert_eq!(agpr_slots(&GFX950, 60, 0, 256), 4);
    }

    #[test]
    fn moves_go_between_a_vgpr_and_its_slot() {
        let store = agpr_move(false, 7, 3);
        let reload = agpr_move(true, 7, 3);
        assert_eq!((store.op, store.defs[0], store.uses[0]), (Opc::Gfx9(Gfx9Op::VAccvgprWriteB32), Opnd::P(Phys::V(AGPR0 + 3), 1), Opnd::P(Phys::V(7), 1)));
        assert_eq!((reload.op, reload.defs[0], reload.uses[0]), (Opc::Gfx9(Gfx9Op::VAccvgprReadB32), Opnd::P(Phys::V(7), 1), Opnd::P(Phys::V(AGPR0 + 3), 1)));
    }
}
