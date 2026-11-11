// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! The GPU a compile is for, and the facts about it the shared passes read.
//!
//! A compile request names its target (`rir::compile_for`, the FFI v2 entry
//! points, a session's options); nothing global selects one, and
//! `rir::compile` means gfx1151. The machine function carries the target
//! from instruction selection on (`mir::MFunc::target`), so the passes after
//! it find the facts there.

pub mod gfx9;

use crate::error::CompileError;
use crate::hsaco_writer::{ObjectTarget, GFX1151_OBJECT, GFX950_OBJECT};

/// A GPU the direct compiler knows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Target {
    #[default]
    Gfx1151,
    Gfx950,
}

impl Target {
    /// Every target the compiler knows, supported or not.
    pub const ALL: [Target; 2] = [Target::Gfx1151, Target::Gfx950];

    #[inline]
    pub fn info(self) -> &'static TargetInfo {
        match self {
            Target::Gfx1151 => &GFX1151,
            Target::Gfx950 => &GFX950,
        }
    }

    pub fn name(self) -> &'static str {
        self.info().name
    }

    /// The target named `name` (`"gfx1151"`, `"gfx950"`).
    pub fn from_name(name: &str) -> Option<Target> {
        Target::ALL.into_iter().find(|t| t.name() == name)
    }

    /// Whether the direct compiler compiles kernels for this target. On
    /// gfx950 it compiles the families of phase G1; a kernel using an
    /// operation it does not lower there fails with `ROCKE_DIRECT_UNSUPPORTED`.
    pub fn supported(self) -> bool {
        matches!(self, Target::Gfx1151 | Target::Gfx950)
    }

    /// `self`, or the `request` error a compile request for an unsupported
    /// target gets.
    pub fn require_supported(self) -> Result<Target, CompileError> {
        if self.supported() {
            return Ok(self);
        }
        Err(CompileError::new("request", format!("the direct compiler does not compile for {} yet", self.name())))
    }

    /// Whether this target's encoder encodes `op`. gfx950's opcode facts come
    /// from its description (`gfx9::gen_isa`), its own opcodes (`Opc::Gfx9`)
    /// included; gfx1151 encodes every shared opcode that is no pseudo, and
    /// no `Opc::Gfx9` (a pseudo in the gfx11 tables).
    pub fn encodes(self, op: crate::rir::mir::Opc) -> bool {
        match self {
            Target::Gfx1151 => op.fmt() != crate::rir::mir::Fmt::Pseudo,
            Target::Gfx950 => gfx9::gen_isa::info(op).is_some_and(|i| i.table),
        }
    }

    /// The supported target named `name`, or the `request` error for an
    /// unknown or unsupported one.
    pub fn requested(name: &str) -> Result<Target, CompileError> {
        match Target::from_name(name) {
            Some(target) => target.require_supported(),
            None => Err(CompileError::new(
                "request",
                format!("unknown target {name:?}; the direct compiler knows {}", supported_names()),
            )),
        }
    }
}

/// The supported targets' names, comma-separated (`"gfx1151,gfx950"`).
pub fn supported_names() -> String {
    Target::ALL.into_iter().filter(|t| t.supported()).map(Target::name).collect::<Vec<_>>().join(",")
}

/// `supported_names()` as the static C string the FFI hands out.
pub const SUPPORTED_NAMES: &std::ffi::CStr = c"gfx1151,gfx950";

/// What the shared passes read about a target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetInfo {
    pub target: Target,
    /// The processor name of the code object (`amdgcn-amd-amdhsa--<name>`).
    pub name: &'static str,
    /// Lanes per wave.
    pub wave: u32,
    /// SGPR dwords of a lane mask (EXEC, VCC, a compare result).
    pub lane_mask_dwords: u8,
    /// Architected VGPRs an allocation may hand out.
    pub max_vgprs: u32,
    /// SGPRs an allocation may hand out; VCC follows them.
    pub max_sgprs: u32,
    /// LDS bytes one workgroup may allocate.
    pub lds_bytes_per_group: u32,
    /// Waves one SIMD holds at most.
    pub max_waves: u32,
    /// VGPRs per lane of one SIMD's register file, and the granule waves
    /// take them in.
    pub vgprs_per_simd: u32,
    pub vgpr_granule: u32,
    /// LDS bytes the SIMDs of one LDS share (a gfx11 WGP, a gfx9 CU), and
    /// how many SIMDs share it.
    pub lds_bytes_shared: u32,
    pub simds_per_lds: u32,
    /// The alignment, in registers, of a VGPR tuple's first register
    /// (gfx90a and later keep tuples at even registers).
    pub vgpr_tuple_align: u8,
    /// What the code object states about the processor.
    pub(crate) object: ObjectTarget,
}

/// gfx1151 in wave32 WGP mode: VGPRs come in blocks of 24 out of 1536 per
/// SIMD, and a WGP's four SIMDs share 128 KiB of LDS.
pub const GFX1151: TargetInfo = TargetInfo {
    target: Target::Gfx1151,
    name: "gfx1151",
    wave: 32,
    lane_mask_dwords: 1,
    max_vgprs: 256,
    max_sgprs: 106,
    lds_bytes_per_group: 64 << 10,
    max_waves: 16,
    vgprs_per_simd: 1536,
    vgpr_granule: 24,
    lds_bytes_shared: 128 << 10,
    simds_per_lds: 4,
    vgpr_tuple_align: 1,
    object: GFX1151_OBJECT,
};

/// gfx950 in wave64: 512 VGPRs per lane (architected and accumulation
/// registers together) in granules of 8, eight waves per SIMD, s0..s101
/// (s102..s105 are FLAT_SCRATCH and XNACK_MASK), and a CU's four SIMDs
/// share 160 KiB of LDS.
pub const GFX950: TargetInfo = TargetInfo {
    target: Target::Gfx950,
    name: "gfx950",
    wave: 64,
    lane_mask_dwords: 2,
    max_vgprs: 256,
    max_sgprs: 102,
    lds_bytes_per_group: 160 << 10,
    max_waves: 8,
    vgprs_per_simd: 512,
    vgpr_granule: 8,
    lds_bytes_shared: 160 << 10,
    simds_per_lds: 4,
    vgpr_tuple_align: 2,
    object: GFX950_OBJECT,
};

impl TargetInfo {
    /// Waves per SIMD that can be resident with `vgprs` VGPRs per lane and
    /// `lds_bytes` of LDS per workgroup of `workgroup_size` lanes.
    pub fn occupancy(&self, vgprs: u32, lds_bytes: u32, workgroup_size: u32) -> u32 {
        let by_vgprs = self.vgprs_per_simd / vgprs.max(1).next_multiple_of(self.vgpr_granule);
        let by_vgprs = match self.target {
            Target::Gfx950 => gfx9::rules::whole_workgroups(by_vgprs, workgroup_size.div_ceil(self.wave), self.simds_per_lds),
            Target::Gfx1151 => by_vgprs,
        };
        let by_lds = match lds_bytes {
            0 => self.max_waves,
            lds => ((self.lds_bytes_shared / lds) * workgroup_size.div_ceil(self.wave) / self.simds_per_lds).max(1),
        };
        by_vgprs.min(by_lds).min(self.max_waves)
    }

    /// The most VGPRs that still give `waves` waves per SIMD.
    pub fn vgpr_budget(&self, waves: u32, lds_bytes: u32, workgroup_size: u32) -> u32 {
        (1..=self.max_vgprs).rev().find(|&v| self.occupancy(v, lds_bytes, workgroup_size) >= waves).unwrap_or(1)
    }

    /// LDS dwords per work-item that spill slots may add without lowering
    /// the workgroups an LDS holds, as limited by `vgprs`, the LDS already
    /// used, and the waves per SIMD; a workgroup's LDS stays within
    /// `lds_bytes_per_group`.
    pub fn lds_spill_slots(&self, lds_bytes: u32, workgroup_size: u32, vgprs: u32) -> u32 {
        let wg = workgroup_size.max(1);
        let waves_per_group = wg.div_ceil(self.wave);
        let waves_per_simd = (self.vgprs_per_simd / vgprs.max(1).next_multiple_of(self.vgpr_granule)).min(self.max_waves);
        let by_vgprs = self.simds_per_lds * waves_per_simd / waves_per_group;
        let by_lds = if lds_bytes == 0 { u32::MAX } else { self.lds_bytes_shared / lds_bytes };
        let groups = by_vgprs.min(by_lds).max(1);
        // A slot's address is a DS instruction's 16-bit offset.
        let most = (self.lds_bytes_shared / groups).min(self.lds_bytes_per_group).min(1 << 16);
        most.saturating_sub(lds_bytes) / (4 * wg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budgets_keep_the_occupancy_of_the_pressure() {
        let t = &GFX1151;
        // 143 live: 10 waves, which 144 VGPRs keep and 145 do not.
        assert_eq!(t.occupancy(143, 512, 32), 10);
        assert_eq!(t.vgpr_budget(10, 512, 32), 144);
        // LDS caps occupancy at one wave: every VGPR count keeps it.
        assert_eq!(t.occupancy(181, 33792, 32), 1);
        assert_eq!(t.vgpr_budget(1, 33792, 32), 256);
        assert_eq!(t.vgpr_budget(16, 0, 64), 96);
    }

    #[test]
    fn lds_spill_slots_keep_the_workgroups_per_wgp() {
        let t = &GFX1151;
        // One-wave workgroups at 256 VGPRs: 5 waves per SIMD, 20 per WGP, so
        // up to 6553 bytes of LDS each; 512 are used, the rest is 47 slots.
        assert_eq!(t.lds_spill_slots(512, 32, 256), 47);
        // 41472 bytes leave three workgroups per WGP: up to 43690 bytes.
        assert_eq!(t.lds_spill_slots(41472, 32, 256), (43690 - 41472) / 128);
        // A workgroup never gets past 64 KiB: two 1024-lane workgroups fill
        // a WGP, each with 64 KiB, 16 slots of 4 KiB.
        assert_eq!(t.lds_spill_slots(0, 1024, 16), 16);
        assert_eq!(t.lds_spill_slots(60000, 32, 256), (64 * 1024 - 60000) / 128);
    }

    #[test]
    fn gfx950_occupancy_follows_its_register_file_and_lds() {
        let t = &GFX950;
        // 512 VGPRs in granules of 8, at most eight waves.
        assert_eq!(t.occupancy(64, 0, 256), 8);
        assert_eq!(t.occupancy(65, 0, 256), 7);
        assert_eq!(t.occupancy(128, 0, 256), 4);
        assert_eq!(t.occupancy(256, 0, 256), 2);
        assert_eq!(t.vgpr_budget(4, 0, 256), 128);
        // 160 KiB per CU: 64 KiB workgroups of four waves fit twice, two
        // waves per SIMD.
        assert_eq!(t.occupancy(32, 64 << 10, 256), 2);
        // A workgroup may hold all 160 KiB: one per CU.
        assert_eq!(t.occupancy(32, 160 << 10, 1024), 4);
        assert_eq!(t.lds_spill_slots(160 << 10, 256, 32), 0);
        // One 1024-lane workgroup per CU could hold 160 KiB of slots, but a
        // slot's address is a 16-bit DS offset: 64 KiB, 16 slots.
        assert_eq!(t.lds_spill_slots(0, 1024, 256), 16);
        assert_eq!((GFX1151.vgpr_tuple_align, t.vgpr_tuple_align), (1, 2));
    }

    #[test]
    fn opcodes_are_encoded_per_target() {
        use crate::rir::mir::Opc;
        use gfx9::gen_isa::Gfx9Op;
        let both = |op: Opc| (Target::Gfx1151.encodes(op), Target::Gfx950.encodes(op));
        assert_eq!(both(Opc::VAddF32), (true, true));
        assert_eq!(both(Opc::Gfx9(Gfx9Op::VPkMulF32)), (false, true));
        assert_eq!(both(Opc::Gfx9(Gfx9Op::BufferInv)), (false, true));
        assert_eq!(both(Opc::VMaxminI32), (true, false));
        assert_eq!(both(Opc::VDual), (true, false));
        assert_eq!(both(Opc::Copy), (false, false));
        for op in Gfx9Op::ALL {
            // `KEEP_ALIVE` and `WAIT_RELEASE` are the gfx9 back end's own pseudos (dropped before encoding).
            let gfx950 = op != Gfx9Op::KeepAlive && op != Gfx9Op::WaitRelease;
            assert_eq!(both(Opc::Gfx9(op)), (false, gfx950), "{op:?}");
        }
    }

    #[test]
    fn requests_name_their_target() {
        assert_eq!(Target::default(), Target::Gfx1151);
        assert_eq!(Target::from_name("gfx1151"), Some(Target::Gfx1151));
        assert_eq!(Target::from_name("gfx950"), Some(Target::Gfx950));
        assert_eq!(Target::from_name("gfx942"), None);
        assert_eq!(Target::requested("gfx1151").ok(), Some(Target::Gfx1151));
        assert_eq!(Target::requested("gfx950").ok(), Some(Target::Gfx950));
        for name in ["gfx942", "", "GFX1151", "gfx950:xnack-"] {
            assert_eq!(Target::requested(name).err().map(|e| e.kind), Some("request"), "{name}");
        }
        assert_eq!(supported_names(), "gfx1151,gfx950");
        assert_eq!(SUPPORTED_NAMES.to_str().ok(), Some(supported_names().as_str()));
        for t in Target::ALL {
            assert_eq!(t.info().target, t);
        }
    }
}
