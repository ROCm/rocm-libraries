// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! The gfx9 back end (gfx950, wave64): the encoder over the gfx950
//! description (`asm`, `gen_isa`), the wave64 choices of instruction
//! selection (`isel_hooks`), gfx9's operand rules before allocation
//! (`legalize`), the passes after it (`finish`: `waitcnt`, `hazards`), and
//! here the kernel descriptor and metadata resources. The resource rules are LLVM's for a
//! gfx90a-class target with architected flat scratch (AMDGPUUsage, and
//! `IsaInfo::getNumExtraSGPRs` / `getNumSGPRBlocks` / `getNumVGPRBlocks`),
//! and agree with every G1-family kernel of the COMGR gfx950 corpus.

use crate::error::CompileError;
use crate::rir::mir::MFunc;

pub mod asm;
#[rustfmt::skip]
pub mod gen_isa;
pub mod hazards;
pub mod isel_hooks;
pub mod legalize;
pub mod model;
pub mod pack;
pub mod rules;
pub mod select;
pub mod spill;
pub mod waitcnt;

/// What gfx950 does between the shared instruction selection and `legalize`:
/// the peephole rules confirmed on gfx950 (`rules`), its own instructions
/// (`select`, whose conversions also take the peephole's `PACK_F16`), VALU
/// fusion under gfx9's operand rule, then packed f32 (`pack`) over what the
/// fusion leaves. `ROCKE_DIRECT_DEBUG=nofuse` and `nopack` skip those two.
pub fn prepare(f: &mut MFunc) {
    rules::peephole(f);
    select::run(f);
    if !crate::rir::debug_flag("nofuse") {
        crate::rir::post::fuse_valu(f);
    }
    if !crate::rir::debug_flag("nopack") {
        pack::run(f);
    }
}

/// The passes after register allocation on gfx950: pseudos become moves
/// and conversions (the shared expansion: a gfx9 copy is the same moves),
/// the 64-bit shift fix, `s_waitcnt`, then the hazard `s_nop`s. gfx950 has
/// no `s_delay_alu`, clauses, VOPD or VGPR release message, so none of the
/// gfx11 passes that place them run.
pub fn finish(func: &mut MFunc, num_vgprs: u32) -> Result<(), CompileError> {
    crate::rir::post::expand_pseudos(func)?;
    hazards::fix_shift64(func, num_vgprs)?;
    waitcnt::insert(func);
    hazards::insert_nops(func);
    Ok(())
}

/// SGPRs a gfx950 code object counts besides those the kernel uses: VCC,
/// FLAT_SCRATCH and XNACK_MASK, whether or not the code touches them.
pub const EXTRA_SGPRS: u32 = 6;

/// The descriptor's COMPUTE_PGM_RSRC1/2/3 of a kernel, and the metadata's
/// `.sgpr_count` and `.vgpr_count`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProgramResources {
    pub rsrc1: u32,
    pub rsrc2: u32,
    pub rsrc3: u32,
    pub sgpr_count: u32,
    pub vgpr_count: u32,
}

/// The resources of a kernel with `vgprs` architected VGPRs (no AGPRs),
/// SGPRs s0..s(`sgprs`-1) in use, the kernarg pointer in s[0:1] and the
/// workgroup ids `wg_ids` after it, and private segment (scratch) when
/// `scratch`. The float mode is the default (denormals kept, IEEE and DX10
/// clamp on), as COMGR sets it.
pub fn program_resources(vgprs: u32, sgprs: u32, wg_ids: [bool; 3], scratch: bool) -> ProgramResources {
    // The hardware writes the kernarg pointer and the workgroup ids into
    // the first SGPRs, used or not.
    let inputs = 2 + wg_ids.iter().filter(|&&on| on).count() as u32;
    let vgpr_count = vgprs.max(1);
    let sgpr_count = sgprs.max(inputs) + EXTRA_SGPRS;
    let vgpr_blocks = vgpr_count.div_ceil(8) - 1;
    let sgpr_blocks = sgpr_count.div_ceil(8) - 1;
    let rsrc1 = 0x00af_0000 | sgpr_blocks << 6 | vgpr_blocks;
    let rsrc2 = (2 << 1)
        | (wg_ids[0] as u32) << 7
        | (wg_ids[1] as u32) << 8
        | (wg_ids[2] as u32) << 9
        | u32::from(scratch);
    // ACCUM_OFFSET: the first AGPR's position in the unified register file,
    // in 4-register units minus one; TG_SPLIT stays off.
    let rsrc3 = (vgpr_count.div_ceil(4) - 1) & 0x3f;
    ProgramResources { rsrc1, rsrc2, rsrc3, sgpr_count, vgpr_count }
}

/// `program_resources` with `agprs` AGPRs as well: they start at
/// ACCUM_OFFSET (the VGPRs rounded up to four), and the unified count
/// covers both, in the descriptor's VGPR granules and the metadata's
/// `.vgpr_count`.
pub fn program_resources_with_agprs(vgprs: u32, agprs: u32, sgprs: u32, wg_ids: [bool; 3], scratch: bool) -> ProgramResources {
    let mut r = program_resources(vgprs, sgprs, wg_ids, scratch);
    if agprs > 0 {
        r.vgpr_count = vgprs.max(1).next_multiple_of(4) + agprs;
        r.rsrc1 = r.rsrc1 & !0x3f | (r.vgpr_count.div_ceil(8) - 1);
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rows of the COMGR gfx950 corpus: (VGPRs, the `.numbered_sgpr`
    /// symbol, workgroup ids) and the descriptor and metadata it holds.
    #[test]
    fn resources_match_the_comgr_corpus() {
        let x = [true, false, false];
        let xy = [true, true, false];
        let rows: [(u32, u32, [bool; 3], u32, u32, u32, u32); 9] = [
            // permute_nd s32x32, elementwise relu f64, elementwise relu f16
            (4, 5, x, 0x00af_0040, 0x84, 0x00, 11),
            (6, 8, x, 0x00af_0040, 0x84, 0x01, 14),
            (10, 8, x, 0x00af_0041, 0x84, 0x02, 14),
            // elementwise silu bf16, gelu_tanh f16, tanh f16
            (37, 18, x, 0x00af_0084, 0x84, 0x09, 24),
            (35, 20, x, 0x00af_00c4, 0x84, 0x08, 26),
            (41, 22, x, 0x00af_00c5, 0x84, 0x0a, 28),
            // topk_softmax, the largest SGPR count
            (98, 86, x, 0x00af_02cc, 0x84, 0x18, 92),
            (195, 100, x, 0x00af_0358, 0x84, 0x30, 106),
            // transpose2d f16 16x16 (two workgroup ids)
            (6, 10, xy, 0x00af_0040, 0x184, 0x01, 16),
        ];
        for (vgprs, sgprs, ids, rsrc1, rsrc2, rsrc3, sgpr_count) in rows {
            let r = program_resources(vgprs, sgprs, ids, false);
            assert_eq!((r.rsrc1, r.rsrc2, r.rsrc3, r.sgpr_count, r.vgpr_count), (rsrc1, rsrc2, rsrc3, sgpr_count, vgprs));
        }
    }

    /// COMGR's AGPR kernels: (VGPRs, AGPRs, `.numbered_sgpr`) and the
    /// descriptor and `.vgpr_count` they hold; with fewer VGPRs, the
    /// accumulator offset and the unified count (whose granules COMGR
    /// rounds up to the waves LDS allows).
    #[test]
    fn agprs_follow_the_accumulator_offset() {
        let xy = [true, true, false];
        for (vgprs, agprs, sgprs, rsrc1, vgpr_count) in [(256, 192, 100, 0x00af_0377, 448), (256, 196, 100, 0x00af_0378, 452)] {
            let r = program_resources_with_agprs(vgprs, agprs, sgprs, xy, false);
            assert_eq!((r.rsrc1, r.rsrc2, r.rsrc3, r.sgpr_count, r.vgpr_count), (rsrc1, 0x184, 0x3f, 106, vgpr_count));
        }
        for (vgprs, agprs, rsrc3, vgpr_count) in [(54, 16, 0xd, 72), (52, 16, 0xc, 68), (70, 16, 0x11, 88), (75, 16, 0x12, 92), (82, 16, 0x14, 100)] {
            let r = program_resources_with_agprs(vgprs, agprs, 40, xy, false);
            assert_eq!((r.rsrc3, r.vgpr_count), (rsrc3, vgpr_count));
        }
        assert_eq!(program_resources_with_agprs(37, 0, 18, xy, false), program_resources(37, 18, xy, false));
    }

    #[test]
    fn inputs_and_scratch_count() {
        // An empty kernel still has its kernarg pointer and workgroup id
        // written: three SGPRs, plus the six extra.
        let r = program_resources(0, 0, [true, false, false], false);
        assert_eq!((r.sgpr_count, r.vgpr_count, r.rsrc1, r.rsrc3), (9, 1, 0x00af_0040, 0));
        assert_eq!(program_resources(8, 12, [true, false, false], true).rsrc2, 0x85);
    }

    #[test]
    fn kernels_compile_through_the_gfx9_back_end() {
        use crate::rir;
        use crate::target::Target;
        // A guarded store of a VOP3 product with a literal: a wave64 mask,
        // the literal in an SGPR, no null register.
        let text = "ckdsl.ir v1\nkernel @probe {\n  attrs { max_workgroup_size = i:256 }\n  params {\n    \
                    %X : ptr<f32,global>\n    %Y : ptr<f32,global>\n    %n : i32\n  }\n  region @entry {\n    \
                    %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n    \
                    %lt : i1 = arith.cmp ( %tid, %n ) { pred = s:\"lt\" }\n    scf.if ( %lt )\n      region @then {\n        \
                    %x : f32 = memref.global_load_typed ( %X, %tid ) { align = i:4, elem_type = s:\"f32\" }\n        \
                    %y : f32 = math.rcp ( %x )\n        \
                    memref.global_store_typed ( %Y, %tid, %y ) { align = i:4, elem_type = s:\"f32\" }\n      }\n  }\n}\n";
        let compiled = rir::compile_for(text, Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        let any = rir::compile_any_target(text, Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        assert!(compiled.hsaco == any.hsaco);
        assert_eq!(compiled.target, Target::Gfx950);
        assert_eq!(u32::from_le_bytes(compiled.hsaco[48..52].try_into().unwrap()), 0x54f);
        let listing = compiled.listing();
        for wanted in ["s_and_saveexec_b64", "v_div_scale_f32", "v_div_fmas_f32", "s_mov_b64 vcc", "s_endpgm"] {
            assert!(listing.contains(wanted), "{wanted} missing:\n{listing}");
        }
        assert!(!listing.contains("null") && !listing.contains("exec_lo"), "{listing}");
        // The same kernel for gfx1151 is the wave32 code it always was.
        let wave32 = rir::compile(text).unwrap_or_else(|e| panic!("{e}")).listing();
        assert!(wave32.contains("s_and_saveexec_b32") && !wave32.contains("s_and_saveexec_b64"));
    }

    /// `count` loaded values summed forward and in reverse, and the sums
    /// multiplied: whatever the order, every value is live at once.
    fn many_live(count: usize) -> String {
        let mut t = String::from(
            "ckdsl.ir v1\nkernel @spills {\n  attrs { max_workgroup_size = i:64 }\n  params {\n    %X : ptr<f32,global>\n    \
             %Y : ptr<f32,global>\n  }\n  region @entry {\n    %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n",
        );
        for i in 0..count {
            t += &format!("    %c{i} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{} }}\n", 64 * i);
            t += &format!("    %p{i} : i32 = arith.add ( %tid, %c{i} )\n");
            t += &format!("    %v{i} : f32 = memref.global_load_typed ( %X, %p{i} ) {{ align = i:4, elem_type = s:\"f32\" }}\n");
        }
        let (mut fwd, mut rev) = ("%v0".to_string(), format!("%v{}", count - 1));
        for i in 1..count {
            t += &format!("    %f{i} : f32 = arith.fadd ( {fwd}, %v{i} )\n");
            t += &format!("    %r{i} : f32 = arith.fadd ( {rev}, %v{} )\n", count - 1 - i);
            (fwd, rev) = (format!("%f{i}"), format!("%r{i}"));
        }
        t + &format!(
            "    %y : f32 = arith.fmul ( {fwd}, {rev} )\n    \
             memref.global_store_typed ( %Y, %tid, %y ) {{ align = i:4, elem_type = s:\"f32\" }}\n    cf.return (  )\n  }}\n}}\n"
        )
    }

    #[test]
    fn kernels_spill_within_a_vgpr_limit() {
        use crate::rir;
        use crate::target::Target;
        let text = many_live(64);
        let full = rir::compile_any_target(&text, Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(full.stats.spill.lds_dwords + full.stats.spill.scratch_dwords, 0, "fits without a limit");
        let limited = rir::compile_any_target_limited(&text, Target::Gfx950, 16).unwrap_or_else(|e| panic!("{e}"));
        let s = &limited.stats;
        assert!(s.vgprs <= 16, "{} VGPRs", s.vgprs);
        // 16 VGPRs keep eight waves per SIMD, which 64 unified registers
        // keep too: the hottest webs take the 48 AGPRs, the rest LDS.
        assert_eq!(super::spill::agpr_slots(&crate::target::GFX950, 16, 0, 64), 48);
        assert_eq!((s.spill.agpr_dwords, s.spill.scratch_dwords, s.waves), (48, 0, 8), "{:?}", s.spill);
        assert!(s.spill.lds_dwords > 0, "{:?}", s.spill);
        let listing = limited.listing();
        for wanted in ["v_accvgpr_write_b32", "v_accvgpr_read_b32", "ds_write_b32", "ds_read_b32"] {
            assert!(listing.contains(wanted), "{wanted} missing:\n{listing}");
        }
        // Past the AGPRs and the LDS: a one-wave workgroup at eight waves
        // per SIMD leaves 32 workgroups a CU's LDS, 20 slots of 256 bytes
        // each, then scratch.
        let more = rir::compile_any_target_limited(&many_live(160), Target::Gfx950, 16).unwrap_or_else(|e| panic!("{e}"));
        let s = &more.stats;
        assert_eq!(crate::target::GFX950.lds_spill_slots(0, 64, 16), 20);
        assert_eq!((s.spill.agpr_dwords, s.spill.lds_dwords), (48, 20), "{:?}", s.spill);
        assert!(s.spill.scratch_dwords > 0, "{:?}", s.spill);
        assert_eq!((s.lds_bytes, s.spill.scratch_bytes), (20 * 256, 4 * s.spill.scratch_dwords));
        let listing = more.listing();
        for wanted in ["scratch_store_dword", "scratch_load_dword"] {
            assert!(listing.contains(wanted), "{wanted} missing:\n{listing}");
        }
        // A limit at or above what the kernel needs changes nothing.
        let roomy = rir::compile_any_target_limited(&text, Target::Gfx950, 256).unwrap_or_else(|e| panic!("{e}"));
        assert!(roomy.hsaco == full.hsaco);
    }
}
