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

pub mod agpr;
pub mod agpr_spill;
pub mod asm;
pub mod atomics;
pub mod bank_wide;
pub mod clauses;
pub mod attn_ops;
pub mod conv_ops;
pub mod cpol;
#[rustfmt::skip]
pub mod gen_isa;
pub mod hazards;
pub mod iglp;
pub mod isel_hooks;
pub mod lane_cse;
pub mod early_constants;
pub mod fp8;
pub mod lanes;
pub mod latency;
pub mod lds_dma;
pub mod lds_tr;
pub mod lits;
pub mod m0_order;
pub mod mask_splat;
pub mod masks;
pub mod legalize;
pub mod m0_hoist;
pub mod mfma;
pub mod model;
pub mod pack;
pub mod relief;
pub mod ro_loads;
pub mod rules;
pub mod sched;
pub mod select;
pub mod spill;
pub mod trade;
pub mod unrolled;
pub mod vectorize;
pub mod waitcnt;
pub mod wide_add;
pub mod zero_acc;
pub mod zero_wide;
pub mod widen;

/// What gfx950 does between the shared instruction selection and `legalize`:
/// MFMAs that start from zero read C as the constant (`zero_acc`), LDS DMA
/// addresses read into SGPRs where they are defined (`m0_hoist`), the
/// peephole rules confirmed on gfx950 (`rules`), its own instructions
/// (`select`, whose conversions also take the peephole's `PACK_F16`, and
/// `wide_add`), VALU fusion under gfx9's operand rule, then packed f32
/// (`pack`) over what the fusion leaves. `ROCKE_DIRECT_DEBUG=nofuse` and
/// `nopack` skip those two.
pub fn prepare(f: &mut MFunc) {
    zero_acc::run(f);
    m0_hoist::run(f);
    rules::peephole(f);
    select::run(f);
    wide_add::run(f);
    if !crate::rir::debug_flag("nofuse") {
        crate::rir::post::fuse_valu(f);
    }
    if !crate::rir::debug_flag("nopack") {
        pack::run(f);
    }
    masks::reuse(f);
}

/// The passes after register allocation on gfx950: pseudos become moves
/// and conversions (the shared expansion: a gfx9 copy is the same moves),
/// MFMAs read the zeros moved into their C as the constant
/// (`zero_acc::after_allocation`; copied from zeros of an earlier block,
/// `zero_wide`), the 64-bit shift fix, `s_waitcnt`, then
/// the hazard `s_nop`s. gfx950 has no `s_delay_alu`, clauses, VOPD or VGPR
/// release message, so none of the gfx11 passes that place them run.
pub fn finish(func: &mut MFunc, num_vgprs: u32) -> Result<(), CompileError> {
    finish_lapped(func, num_vgprs, &mut |_| ())
}

/// `finish`, calling `lap` after each of its stages with the stage's name
/// (`expand`, `waitcnt`, `hazards`), for the compile's phase times.
pub fn finish_lapped(func: &mut MFunc, num_vgprs: u32, lap: &mut dyn FnMut(&'static str)) -> Result<(), CompileError> {
    crate::rir::post::expand_pseudos(func)?;
    zero_acc::after_allocation(func);
    zero_wide::after_allocation(func);
    hazards::fix_shift64(func, num_vgprs)?;
    lap("expand");
    waitcnt::insert(func);
    lap("waitcnt");
    hazards::insert_nops(func);
    lap("hazards");
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

    /// A K loop of `accs` 32x32x8 f16 MFMA accumulators, each stored after it.
    fn accumulators(accs: usize) -> String {
        let mut s = String::from(
            "ckdsl.ir v1\nkernel @acc {\n  attrs { max_workgroup_size = i:64 }\n  params {\n    %A : ptr<f16,global>\n    \
             %Y : ptr<f32,global>\n    %n : i32\n  }\n  region @entry {\n    %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n    \
             %c0 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }\n    %c1 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:1 }\n    \
             %c4 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:4 }\n    \
             %z : vec<f32x16> = arith.constant_vec (  ) { elem = s:\"f32\", fill = f:0.0, vec = i:16 }\n",
        );
        let results: Vec<String> = (0..accs).map(|a| format!("%r{a} : vec<f32x16>")).collect();
        let iter_args: Vec<String> = (0..accs).map(|a| format!("{{ name = s:\"%acc{a}\", type = s:\"vec<f32x16>\" }}")).collect();
        s += &format!(
            "    {} = scf.for ( %c0, %n, %c1, {} ) {{ iter_args = l:[ {} ], iv = s:\"%k\", iv_type = s:\"i32\", num_iter_args = i:{accs}, unroll = b:false }}\n      region @body {{\n",
            results.join(", "),
            vec!["%z"; accs].join(", "),
            iter_args.join(", ")
        );
        s += "        %o : i32 = arith.mul ( %k, %c4 )\n        %p : i32 = arith.add ( %o, %tid )\n        \
              %a : vec<f16x4> = memref.global_load_vN ( %A, %p ) { align = i:8, elem_type = s:\"f16\", vec = i:4 }\n";
        for a in 0..accs {
            s += &format!("        %d{a} : vec<f32x16> = tile.mma ( %a, %a, %acc{a} ) {{ op_id = s:\"mfma_f32_32x32x8_f16\" }}\n");
        }
        let yields: Vec<String> = (0..accs).map(|a| format!("%d{a}")).collect();
        s += &format!("        scf.yield ( {} ) {{ num = i:{accs} }}\n      }}\n", yields.join(", "));
        for a in 0..accs {
            s += &format!(
                "    %e{a} : f32 = vector.extract ( %r{a} ) {{ index = i:15 }}\n    \
                 %i{a} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{} }}\n    %q{a} : i32 = arith.add ( %tid, %i{a} )\n    \
                 memref.global_store_typed ( %Y, %q{a}, %e{a} ) {{ align = i:4, elem_type = s:\"f32\" }}\n",
                64 * a
            );
        }
        s + "    cf.return (  )\n  }\n}\n"
    }

    #[test]
    fn uniform_loads_below_their_index_base_are_scalar_on_gfx950() {
        use crate::rir;
        use crate::target::Target;
        // `X[for - 1]` of a loop result carried through a load: `opt` leaves
        // the -4 bytes in the offset, and the address is uniform.
        let text = "ckdsl.ir v1\nkernel @neg {\n  attrs { max_workgroup_size = i:64 }\n  params {\n    \
                    %X : ptr<i32,global> { align = i:4, readonly = b:true }\n    %Y : ptr<i32,global>\n    %n : i32\n  }\n  \
                    region @entry {\n    %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n    \
                    %c0 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }\n    \
                    %c1 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:1 }\n    \
                    %c2 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:2 }\n    \
                    %cm : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:-1 }\n    \
                    %for : i32 = scf.for ( %c0, %n, %c1, %c0 ) { iter_args = l:[ { name = s:\"%acc\", type = s:\"i32\" } ], \
                    iv = s:\"%i\", iv_type = s:\"i32\", num_iter_args = i:1, unroll = b:false }\n      region @body {\n        \
                    %g : i32 = memref.global_load_typed ( %X, %acc ) { align = i:4, elem_type = s:\"i32\" }\n        \
                    %a2 : i32 = arith.add ( %g, %c2 )\n        scf.yield ( %a2 ) { num = i:1 }\n      }\n    \
                    %m : i32 = arith.add ( %for, %cm )\n    \
                    %x : i32 = memref.global_load_typed ( %X, %m ) { align = i:4, elem_type = s:\"i32\" }\n    \
                    %y : i32 = arith.add ( %x, %tid )\n    \
                    memref.global_store_typed ( %Y, %tid, %y ) { align = i:4, elem_type = s:\"i32\" }\n    cf.return (  )\n  }\n}\n";
        let loads = |target| {
            let listing = rir::compile_any_target(text, target).unwrap_or_else(|e| panic!("{e}")).listing();
            listing.lines().filter(|l| l.contains("global_load")).count()
        };
        assert_eq!((loads(Target::Gfx950), loads(Target::Gfx1151)), (0, 1));
    }

    #[test]
    fn kernels_with_agpr_alloc_0_use_no_agprs() {
        use crate::rir;
        use crate::target::Target;
        let none = |text: String| text.replacen("attrs { ", "attrs { agpr_alloc = l:[ i:0, i:0 ], ", 1);
        // Spills short of VGPRs go to LDS instead of AGPRs.
        let s = rir::compile_any_target_limited(&none(many_live(64)), Target::Gfx950, 16).unwrap_or_else(|e| panic!("{e}")).stats;
        assert!(s.vgprs <= 16 && s.spill.agpr_dwords == 0 && s.spill.lds_dwords > 0, "{:?}", s.spill);
        // Accumulators short of VGPRs bank in AGPRs without the attribute, and spill with it.
        let banked = rir::compile_any_target_limited(&accumulators(6), Target::Gfx950, 64).unwrap_or_else(|e| panic!("{e}"));
        assert!(banked.stats.spill.agpr_dwords > 0, "{:?}", banked.stats.spill);
        let spilled = rir::compile_any_target_limited(&none(accumulators(6)), Target::Gfx950, 64).unwrap_or_else(|e| panic!("{e}"));
        let s = &spilled.stats.spill;
        assert!(s.agpr_dwords == 0 && s.lds_dwords + s.scratch_dwords > 0, "{s:?}");
        assert!(!spilled.listing().contains("accvgpr"), "{}", spilled.listing());
        // Without spills the attribute changes nothing.
        let plain = rir::compile_any_target(&accumulators(2), Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        assert!(rir::compile_any_target(&none(accumulators(2)), Target::Gfx950).is_ok_and(|c| c.hsaco == plain.hsaco));
        // Only [0, 0] is supported.
        for range in ["l:[ i:0, i:8 ]", "l:[ i:4, i:4 ]", "i:0"] {
            let text = many_live(4).replacen("attrs { ", &format!("attrs {{ agpr_alloc = {range}, "), 1);
            let e = rir::compile_any_target(&text, Target::Gfx950).err().map(|e| e.to_string()).unwrap_or_default();
            assert!(e.contains("agpr_alloc"), "{range}: {e}");
        }
    }
}
