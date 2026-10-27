// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! The shared rewrites gfx950 takes: the proven peephole rules
//! (`rir::peephole`) of the families confirmed on gfx950, and VALU fusion
//! (`rir::post::fuse_valu`) into the three-source forms gfx950 has, under
//! gfx9's operand rule. Also how many waves gfx950 really keeps resident
//! (`TargetInfo::occupancy`).

use crate::rir::mir::*;
use crate::rir::peephole::{self, FAMILIES};
use crate::target::Target;

/// The peephole families gfx950 runs: those a GPU differential confirmed,
/// by changing G1 kernels whose every launch stayed bitwise equal to
/// COMGR's. Every family's rules hold on gfx9 (integer, bit and select
/// operations with gfx11's semantics, f32 max/min chains into `v_max3`/
/// `v_min3` in IEEE mode on both, f16 conversion pairs into `PACK_F16`,
/// which `select` lowers; the rules on 32-bit lane masks never match
/// wave64's 64-bit ones), but `bounds`, `cselect`, `perm`, `pack`, `sshl`
/// and `cndmask` change no G1 kernel, so they wait for a differential of a
/// kernel they change.
pub const CONFIRMED: [&str; 5] = ["addr", "minmax", "bfe", "and", "shl"];

pub fn peephole(f: &mut MFunc) {
    peephole::optimize_with(f, std::array::from_fn(|k| CONFIRMED.contains(&FAMILIES[k])));
}

/// Whether gfx950 has the fused instruction `op` and reads `srcs` with its
/// one constant-bus read: one scalar register or literal at most (VOP3 and
/// VOP3P take no literal on gfx9; `legalize` moves one to an SGPR).
pub fn fuses(op: Opc, srcs: &[Opnd], vregs: &[VRegInfo]) -> bool {
    if !Target::Gfx950.encodes(op) {
        return false;
    }
    let mut scalars: Vec<Opnd> = Vec::new();
    for &o in srcs {
        let scalar = match o {
            Opnd::Imm(v) => !(crate::rir::isel::is_inline_int(v) || crate::rir::isel::is_inline_f32(v)),
            Opnd::R(v) => vregs[v.reg as usize].class == Class::S,
            Opnd::P(Phys::V(_), _) => false,
            Opnd::P(..) => true,
        };
        if scalar && !scalars.contains(&o) {
            scalars.push(o);
        }
    }
    scalars.len() <= 1
}

/// The waves per SIMD that `by_vgprs` (what the register file allows)
/// leaves once only whole workgroups count: a workgroup runs on one CU with
/// its `workgroup_waves` spread over the CU's `simds`, so each SIMD holds
/// them in shares of `ceil(workgroup_waves / simds)`. Workgroups of up to
/// one wave per SIMD lose nothing.
pub fn whole_workgroups(by_vgprs: u32, workgroup_waves: u32, simds: u32) -> u32 {
    let share = workgroup_waves.div_ceil(simds).max(1);
    if by_vgprs < share {
        return by_vgprs;
    }
    by_vgprs - by_vgprs % share
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(reg: u32) -> Opnd {
        Opnd::R(VRef { reg, off: 0, n: 1 })
    }

    #[test]
    fn confirmed_families_are_peephole_families() {
        assert!(CONFIRMED.iter().all(|c| FAMILIES.contains(c)));
    }

    #[test]
    fn fusion_keeps_to_one_constant_bus_read() {
        let vregs = [VRegInfo { class: Class::V, size: 1, align: 1, fixed: None }, VRegInfo { class: Class::S, size: 1, align: 1, fixed: None }];
        let (v, s, s2) = (r(0), r(1), Opnd::R(VRef { reg: 1, off: 0, n: 1 }));
        assert!(fuses(Opc::VAdd3U32, &[v, v, v], &vregs));
        assert!(fuses(Opc::VAdd3U32, &[s, v, Opnd::Imm(64)], &vregs));
        assert!(fuses(Opc::VAdd3U32, &[s, s2, v], &vregs), "the same SGPR twice is one read");
        assert!(fuses(Opc::VLshlAddU32, &[v, Opnd::Imm(2), Opnd::Imm(0x1234)], &vregs));
        assert!(!fuses(Opc::VAdd3U32, &[s, v, Opnd::Imm(0x1234)], &vregs), "an SGPR and a literal");
        // gfx950 has no v_minmax_i32 / v_maxmin_i32.
        assert!(!fuses(Opc::VMinmaxI32, &[v, v, v], &vregs));
        assert!(!fuses(Opc::VMaxminI32, &[v, v, v], &vregs));
    }

    #[test]
    fn occupancy_counts_whole_workgroups() {
        let t = &crate::target::GFX950;
        // 1024 lanes are 16 waves, four per SIMD: 80 VGPRs (six waves) and
        // 92 (five) both hold one workgroup per CU.
        assert_eq!((t.occupancy(80, 0, 1024), t.occupancy(92, 0, 1024)), (4, 4));
        assert_eq!(t.occupancy(64, 0, 1024), 8);
        assert_eq!(t.vgpr_budget(4, 0, 1024), 128);
        // 512 lanes, two waves per SIMD.
        assert_eq!((t.occupancy(80, 0, 512), t.occupancy(92, 0, 512)), (6, 4));
        // Up to 256 lanes, one wave per SIMD: every count stands.
        assert_eq!((t.occupancy(92, 0, 256), t.occupancy(92, 0, 64)), (5, 5));
        // LDS already counted workgroups: 96 KiB fits once in 160.
        assert_eq!(t.occupancy(32, 96 << 10, 1024), 4);
        // gfx1151 keeps its own count.
        assert_eq!(crate::target::GFX1151.occupancy(143, 0, 1024), 10);
    }
}
