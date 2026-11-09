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

/// The most unified registers (VGPRs and AGPRs) per wave with which one
/// workgroup of `workgroup_size` lanes still fits a CU: each SIMD's file
/// holds its share of the workgroup's waves. Past it the dispatch is
/// refused (a 512-lane workgroup: 256, 1024 lanes: 128).
pub fn workgroup_registers(t: &crate::target::TargetInfo, workgroup_size: u32) -> u32 {
    let share = workgroup_size.div_ceil(t.wave).div_ceil(t.simds_per_lds).max(1);
    t.vgprs_per_simd / share / t.vgpr_granule * t.vgpr_granule
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

    #[test]
    fn a_workgroup_fits_a_cu_only_within_its_share_of_the_registers() {
        let t = &crate::target::GFX950;
        let room = |lanes: u32| workgroup_registers(t, lanes);
        assert_eq!((room(64), room(256), room(512), room(768), room(1024)), (512, 512, 256, 168, 128));
        // A 1024-lane kernel with more values live than 128 VGPRs hold spills
        // within them: four of its waves share each SIMD's 512 registers.
        let mut text = String::from(
            "ckdsl.ir v1\nkernel @wide {\n  attrs { max_workgroup_size = i:1024 }\n  params {\n    %X : ptr<f32,global>\n    \
             %Y : ptr<f32,global>\n  }\n  region @entry {\n    %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n",
        );
        for i in 0..160 {
            text += &format!("    %c{i} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{} }}\n", 1024 * i);
            text += &format!("    %p{i} : i32 = arith.add ( %tid, %c{i} )\n");
            text += &format!("    %v{i} : f32 = memref.global_load_typed ( %X, %p{i} ) {{ align = i:4, elem_type = s:\"f32\" }}\n");
        }
        let (mut fwd, mut rev) = ("%v0".to_string(), "%v159".to_string());
        for i in 1..160 {
            text += &format!("    %f{i} : f32 = arith.fadd ( {fwd}, %v{i} )\n    %r{i} : f32 = arith.fadd ( {rev}, %v{} )\n", 159 - i);
            (fwd, rev) = (format!("%f{i}"), format!("%r{i}"));
        }
        text += &format!(
            "    %y : f32 = arith.fmul ( {fwd}, {rev} )\n    memref.global_store_typed ( %Y, %tid, %y ) {{ align = i:4, elem_type = s:\"f32\" }}\n    \
             cf.return (  )\n  }}\n}}\n"
        );
        let compiled = crate::rir::compile_any_target(&text, crate::target::Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        let s = &compiled.stats;
        assert!(s.vgprs + s.spill.agpr_dwords <= 128 && s.spill.lds_dwords + s.spill.scratch_dwords > 0, "{s:?}");
    }
}
