// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Custom_Cijk_Alik_Bljk_S_B_MT512x128x8_ScalarB_gfx1150
// Hand-tuned gfx1150 f32 GEMM, TN (A and B K-contiguous); B delivered by s_load_b128 per column into SGPRs.
// Derived from a TensileLite-generated MT128x128x8 f32 TN source kernel: 512x128 macro tile,
// 16 wave32s with 16x8 outputs per lane, v_dual_fmac_f32 MACs with B as a wave-uniform SGPR operand.
// Each block also issues one global_load of this wave's B columns 2 blocks (64 B) ahead into an unused
// VGPR, warming GL1/L2 so the scalar loads (waited on one MAC step later) hit cache.
// gfx1150 has 1024 VGPRs per SIMD, so the 16-wave workgroup runs in WGP mode (4 waves per SIMD) and the
// A global reads are pipelined two blocks deep (second staging buffer) to cover the longer memory latency.
// Supported: complete 512x128 tiles, K % 16 == 0, K >= 32, GSU == 1, StaggerU == 0, batch == 1,
// no bias/activation/scaling; selected only through exact-size Equality logic.

/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx1150"
.text
.protected Custom_Cijk_Alik_Bljk_S_B_MT512x128x8_ScalarB_gfx1150
.globl Custom_Cijk_Alik_Bljk_S_B_MT512x128x8_ScalarB_gfx1150
.p2align 8
.type Custom_Cijk_Alik_Bljk_S_B_MT512x128x8_ScalarB_gfx1150,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Custom_Cijk_Alik_Bljk_S_B_MT512x128x8_ScalarB_gfx1150
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 199 // vgprs
  .amdhsa_next_free_sgpr 106 // sgprs
  .amdhsa_group_segment_fixed_size 53504 // lds bytes
  .amdhsa_wavefront_size32 1 // 32-thread wavefronts
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
  .amdhsa_workgroup_processor_mode 1
.end_amdhsa_kernel
.text
/* Num VGPR   =197 */
/* Num AccVGPR=0 */
/* Num SGPR   =74 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 16 x 8 */
/* SubGroup= 32 x 16 */
/* VectorWidthA=2 */
/* VectorWidthB=2 */
/* GlobalReadVectorWidthA=2, GlobalReadVectorWidthB=2 */
/* DirectToLdsA=False */
/* DirectToLdsB=False */
/* UseSgprForGRO=True */
.amdgpu_metadata
---
custom.config:
  InternalSupportParams:
    KernArgsVersion: 2
    SupportCustomStaggerU: false
  StaggerU: 0
amdhsa.version:
  - 1
  - 1
amdhsa.kernels:
  - .name: Custom_Cijk_Alik_Bljk_S_B_MT512x128x8_ScalarB_gfx1150
    .symbol: 'Custom_Cijk_Alik_Bljk_S_B_MT512x128x8_ScalarB_gfx1150.kd'
    .language:                   OpenCL C
    .language_version:
      - 2
      - 0
    .args:
      - .name:            Gemm info
        .size:            4
        .offset:          0
        .value_kind:      by_value
        .value_type:      u32
      - .name:            kernel info0
        .size:            4
        .offset:          4
        .value_kind:      by_value
        .value_type:      u32
      - .name:            kernel info1
        .size:            4
        .offset:          8
        .value_kind:      by_value
        .value_type:      u32
      - .name:            numWG
        .size:            4
        .offset:          12
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree0
        .size:            4
        .offset:          16
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree1
        .size:            4
        .offset:          20
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree2
        .size:            4
        .offset:          24
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesSum0
        .size:            4
        .offset:          28
        .value_kind:      by_value
        .value_type:      u32
      - .name:            D
        .size:            8
        .offset:          32
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            C
        .size:            8
        .offset:          40
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            A
        .size:            8
        .offset:          48
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            B
        .size:            8
        .offset:          56
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            strideD0
        .size:            4
        .offset:          64
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideD1
        .size:            4
        .offset:          68
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC0
        .size:            4
        .offset:          72
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC1
        .size:            4
        .offset:          76
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA0
        .size:            4
        .offset:          80
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA1
        .size:            4
        .offset:          84
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB0
        .size:            4
        .offset:          88
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB1
        .size:            4
        .offset:          92
        .value_kind:      by_value
        .value_type:      u32
      - .name:            alpha
        .size:            4
        .offset:          96
        .value_kind:      by_value
        .value_type:      f32
      - .name:            beta
        .size:            4
        .offset:          100
        .value_kind:      by_value
        .value_type:      f32
      - .name:            batchOffsetD
        .size:            8
        .offset:          104
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetC
        .size:            8
        .offset:          112
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetA
        .size:            8
        .offset:          120
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetB
        .size:            8
        .offset:          128
        .value_kind:      by_value
        .value_type:      u64
    .group_segment_fixed_size:   53504
    .kernarg_segment_align:      8
    .kernarg_segment_size:       136
    .max_flat_workgroup_size:    512
    .private_segment_fixed_size: 0
    .sgpr_count:                 106
    .sgpr_spill_count:           0
    .vgpr_count:                 199
    .vgpr_spill_count:           0
    .wavefront_size:             32
...
.end_amdgpu_metadata
Custom_Cijk_Alik_Bljk_S_B_MT512x128x8_ScalarB_gfx1150:
label_ASM_Start:  /// Main body of the asm kernel
s_mov_b64 s[104:105], s[0:1]                         // kernarg pointer for the epilogue

/******************************************/
/* VGPR Assignments for MX                */
/******************************************/
.set vgprMXSBase, 0

/******************************************/
/* VGPR Macro Assignments for MX          */
/******************************************/

/******************************************/
/* VGPR Assignments                       */
/******************************************/
/* ValuC range: [0-128),  */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprBase, 136
.set vgprLocalWriteAddrA, 130
.set vgprLocalWriteAddrB, 131
.set vgprGlobalReadOffsetA, 128
.set vgprGlobalReadOffsetB, 129
.set vgprLocalReadAddrA, 132
.set vgprLocalReadAddrB, 133
.set vgprSerial, 190

/******************************************/
/* VGPR Macro Assignments                 */
/******************************************/
.set vgprValuA_X0_I0_BASE, vgprBase+2
.set vgprValuB_X0_I0_BASE, vgprBase+36
.set vgprG2LA_BASE, vgprBase+44
.set vgprG2LB_BASE, vgprBase+52
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+16
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X1_I0, vgprValuB_X0_I0_BASE+0
.set vgprG2LA, vgprG2LA_BASE+0
.set vgprG2LA2, 191
.set vgprG2LB, vgprG2LB_BASE+0

/******************************************/
/* SGPR Assignments                       */
/******************************************/
.set sgprKernArgAddress, 0
.set sgprWorkGroup0, 2
.set sgprWorkGroup1, 3
.set sgprWorkGroup2, 4
.set sgprArgType, 5
.set sgprGSUSumIdx, 6
.set sgprGSULog2BpeC, 8
.set sgprGSULog2BpeD, 9
.set sgprStaggerU, 10
.set sgprWGM, 11
.set sgprLoopCounterL, 12
.set sgprOrigLoopCounter, 13
.set sgprSrdD, 16
.set sgprSrdC, 20
.set sgprNumWorkGroups0, 14
.set sgprNumWorkGroups1, 15
.set sgprSizesFree, 24
.set sgprSizesSum, 27
.set sgprAddressD, 28
.set sgprAddressC, 30
.set sgprAddressA, 32
.set sgprAddressB, 34
.set sgprStridesD, 36
.set sgprStridesC, 38
.set sgprStridesA, 40
.set sgprStridesB, 42
.set sgprAlpha, 44
.set sgprBeta, 45
.set sgprGSU, 46

/* Size Assignments */
.set sgprSizeI, sgprSizesFree+0
.set sgprSizeJ, sgprSizesFree+1
.set sgprSizeK, sgprSizesFree+2
.set sgprSizeL, sgprSizesSum+0

/* Stride Assignments */
.set constStrideD0I, 1
.set sgprStrideD1J, sgprStridesD+0
.set sgprStrideDK, sgprStridesD+1
.set constStrideC0I, 1
.set sgprStrideC1J, sgprStridesC+0
.set sgprStrideCK, sgprStridesC+1
.set constStrideAL, 1
.set sgprStrideA0I, sgprStridesA+0
.set sgprStrideAK, sgprStridesA+1
.set constStrideBL, 1
.set sgprStrideB1J, sgprStridesB+0
.set sgprStrideBK, sgprStridesB+1

.set MT0, 512
.set MT1, 128
.set DepthU, 8
/* Number of elements to shift-left SRD */
.set SrdShiftLeftA, 2
.set SrdShiftLeftB, 2
/* 2GB limit - set offsets to -1 to exceed this and clamp */
.set BufferLimit, 0xffffffff
.set BufferOOB, 0xfffff000

/******************************************/
/* Bits 127:96 of SRD.                    */
/* hex: 0x31004000                        */
/* dst_sel_x (3b): 0                      */
/* dst_sel_y (3b): 0                      */
/* dst_sel_z (3b): 0                      */
/* dst_sel_w (3b): 0                      */
/* format (7b): 4                         */
/* _unusedA (2b): 0                       */
/* index_stride (2b): 0                   */
/* add_tid_enable (1b): 0                 */
/* resource_level (1b): 1                 */
/* _unusedB (1b): 0                       */
/* LLC_noalloc (2b): 0                    */
/* oob_select (2b): 3                     */
/* type (2b): 0                           */
/******************************************/
.set Srd127_96, 0x31004000

/* Global Offset A */

/* Global Offset B */

/******************************************/
/* Allocate Resources                     */
/******************************************/

/* Load num of Gemms */
s_load_b32 s20, s[sgprKernArgAddress:sgprKernArgAddress+1], 0

/* Load packed kernel args (StaggerU/GSU) */
s_load_b32 s22, s[sgprKernArgAddress:sgprKernArgAddress+1], 4

/* Load WGM data */
s_load_b32 s[sgprWGM], s[sgprKernArgAddress:sgprKernArgAddress+1], 8

/* Load num of WGs */
s_load_b32 s23, s[sgprKernArgAddress:sgprKernArgAddress+1], 12
s_waitcnt lgkmcnt(0)                               // load args
s_lshr_b32 s21, s20, 0x1e                          // Get arg type
s_and_b32 s20, 0x3fffffff, s20                     // Get nums of gemm
s_cmp_eq_u32 s21, 3                                // Is kernel argType == 3
s_cbranch_scc1 label_Bypass_ArgType3_to_ArgType0_Instance1
s_cmp_eq_u32 s21, 0                                // Is kernel args
s_cbranch_scc0 label_HBMArgs
label_Bypass_ArgType3_to_ArgType0_Instance1:
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], 0x10 // Shift common args
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_b512 s[24:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 0 // 0
s_load_b128 s[40:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 64 // 64
s_load_b64 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
s_branch label_LoadArgsEnd
label_HBMArgs:

/* Load address of kernel arguments */
s_load_b64 s[sgprKernArgAddress:sgprKernArgAddress+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 16
s_waitcnt lgkmcnt(0)                               // wait for args to load
label_LoadArgsEnd:
s_and_b32 s[sgprStaggerU], s22, 0xffff0000         // Restore StaggerU related vars
s_lshr_b32 s[sgprStaggerU], s[sgprStaggerU], 0x10
s_and_b32 s[sgprGSU], s22, 0xffff                  // Restore GSUConfig and GSU
s_mov_b32 s[sgprArgType], s21
s_mov_b32 m0, 0xd100                               // LDS clamp at 53504 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id
s_mov_b32 vcc_hi, 0                                // Ensure hi bits are zero

/* remap workgroup to XCCs */
s_lshr_b32 s52, s[sgprWGM], 0x10                   // Get WGMXCC
s_ff1_i32_b32 s52, s52                             // Get log(WGMXCC)
s_lshr_b32 s53, s[sgprWGM], 0x16                   // Get CU_Count
/* remap WGs if WGMXCC > 1 ( log(WGMXCC) > 0 ) */
s_cmp_gt_i32 s52, 0
s_cbranch_scc0 label_skip_WGMXCC
/* only remap WGs in the range */
s_lshr_b32 s49, s23, s52
s_lshl_b32 s49, s49, s52
s_cmp_ge_u32 s[sgprWorkGroup0], s49
s_cbranch_scc1 label_skip_WGMXCC
s_cmp_eq_u32 s53, 0                                // CU_Count == 0 ?
s_cbranch_scc0 label_XCCG_nonzero
s_lshr_b32 s49, s[sgprWorkGroup0], s52
s_bfm_b32 s50, s52, 0
s_and_b32 s50, s[sgprWorkGroup0], s50
s_lshr_b32 s51, s23, s52
s_mul_i32 s50, s50, s51
s_add_u32 s[sgprWorkGroup0], s49, s50
s_branch label_skip_WGMXCC
label_XCCG_nonzero:
/* temp0 = (wg//CU_Count)*CU_Count */
v_cvt_f64_u32 v[6:7], s53                          // s49 = s[sgprWorkGroup0] / s53
v_rcp_f64 v[6:7], v[6:7]                           // s49 = s[sgprWorkGroup0] / s53
v_cvt_f64_u32 v[8:9], s[sgprWorkGroup0]            // s49 = s[sgprWorkGroup0] / s53
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s49 = s[sgprWorkGroup0] / s53
v_cvt_u32_f64 v6, v[6:7]                           // s49 = s[sgprWorkGroup0] / s53
v_mul_lo_u32 v7, v6, s53                           // s49 = s[sgprWorkGroup0] / s53
v_sub_nc_u32 v8, s[sgprWorkGroup0], v7             // s49 = s[sgprWorkGroup0] / s53
v_cmp_ge_u32 vcc_lo, v8, s53                       // s49 = s[sgprWorkGroup0] / s53
s_mov_b32 exec_lo, vcc_lo                          // s49 = s[sgprWorkGroup0] / s53
v_add_nc_u32 v6, v6, 1                             // s49 = s[sgprWorkGroup0] / s53
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v7, v6, s53                           // s49 = s[sgprWorkGroup0] / s53
v_sub_nc_u32 v8, s[sgprWorkGroup0], v7             // s49 = s[sgprWorkGroup0] / s53
v_readfirstlane_b32 s49, v6                        // quotient
v_readfirstlane_b32 s50, v8                        // remainder
s_mul_i32 s49, s49, s53
/* temp1 = (wg%CU_Count)//WGMXCC */
s_lshr_b32 s50, s50, s52
/* temp0 = temp0 + temp1 */
s_add_u32 s49, s49, s50
/* temp1 = (wg%WGMXCC) * ((WGs - (WGs//CU_Count) * CU_Count) if (wg > (WGs//CU_Count) * CU_Count) else CU_Count)//WGMXCC */
v_cvt_f64_u32 v[6:7], s53                          // s50 = s23 / s53
v_rcp_f64 v[6:7], v[6:7]                           // s50 = s23 / s53
v_cvt_f64_u32 v[8:9], s23                          // s50 = s23 / s53
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s50 = s23 / s53
v_cvt_u32_f64 v6, v[6:7]                           // s50 = s23 / s53
v_mul_lo_u32 v7, v6, s53                           // s50 = s23 / s53
v_sub_nc_u32 v8, s23, v7                           // s50 = s23 / s53
v_cmp_ge_u32 vcc_lo, v8, s53                       // s50 = s23 / s53
s_mov_b32 exec_lo, vcc_lo                          // s50 = s23 / s53
v_add_nc_u32 v6, v6, 1                             // s50 = s23 / s53
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s50, v6                        // quotient
s_mul_i32 s50, s50, s53
s_sub_u32 s51, s23, s50
s_cmp_gt_u32 s[sgprWorkGroup0], s50
s_cselect_b32 s50, s51, s53
s_lshr_b32 s50, s50, s52
s_bfm_b32 s51, s52, 0
s_and_b32 s51, s[sgprWorkGroup0], s51
s_mul_i32 s50, s50, s51
/* WorkGroup0 = temp0 + temp1 */
s_add_u32 s[sgprWorkGroup0], s49, s50
label_skip_WGMXCC:  /// skip WGMXCC if no enough WGs to remap
s_cmp_eq_u32 s21, 3
s_cbranch_scc1 label_ArgType3_Routed_To_ArgType0
s_cmp_eq_u32 s21, 0
s_cbranch_scc0 label_MultiGemm
label_ArgType3_Routed_To_ArgType0:
/* init: add vgpr [136...321) to pool */
/* init: add vgpr [0...128) to pool */
/* init: add agpr [0...0) to pool */

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
v_lshrrev_b32 v0, 5, v[vgprSerial]                 // 0 = Serial / 32
v_and_b32 v1, 31, v[vgprSerial]                    // 1 = Serial % 32
v_lshrrev_b32 v2, 4, v0                            // 2 = 0 / 16
v_and_b32 v3, 15, v0                               // 3 = 0 % 16

/* local read addresses: final offsets a */
v_lshrrev_b32 v0, 9, v[vgprSerial]                 // LSU offset: sgid = Serial / subGroup(512)
s_mov_b32 s16, 516                                 // LSU offset: lsuoffset = sgid*(MT0+PAD)
v_mul_lo_u32 v0, s16, v0                           // LSU offset: lsuoffset = sgid*(MT0+PAD)
v_lshlrev_b32 v1, 1, v1                            // Final Offset: lrAOffset * VW
v_add_lshl_u32 v[vgprLocalReadAddrA], v0, v1, 0x2  // Final Offset: add padding 16 per block 0

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 9, v[vgprSerial]                 // LSU offset: sgid = Serial / subGroup(512)
s_mov_b32 s16, 132                                 // LSU offset: lsuoffset = sgid*(MT1+PAD)
v_mul_lo_u32 v0, s16, v0                           // LSU offset: lsuoffset = sgid*(MT1+PAD)
v_lshlrev_b32 v3, 1, v3                            // Final Offset: lrBOffset * VW
v_add_lshl_u32 v[vgprLocalReadAddrB], v0, v3, 0x2  // Final Offset: add padding 16 per block 0

/* local read addresses: declare addresses a */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, 0x4080, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 4 */
/* v1 = A-unroll = serial%LVCA */
v_lshrrev_b32 v0, 2, v[vgprSerial]                 // 0 = Serial / 4
v_and_b32 v1, 3, v[vgprSerial]                     // 1 = Serial % 4
/* unroll *= glvw */
v_lshlrev_b32 v1, 1, v1                            // v1 = v1 * 2
v_mov_b32 v4, v1                                   // copy for GlobalSplitU
/* LVCB = 4 */
/* v3 = B-unroll = serial%LVCB */
v_lshrrev_b32 v2, 2, v[vgprSerial]                 // 2 = Serial / 4
v_and_b32 v3, 3, v[vgprSerial]                     // 3 = Serial % 4
/* unroll *= glvw */
v_lshlrev_b32 v3, 1, v3                            // v3 = v3 * 2
v_mov_b32 v5, v3                                   // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v4 */
/* lwaUnrollAssignmentB = v5 */

/* local write addresses: first offset a */
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x204, v4    // lwAL**(MTA + PAD)
v_add_nc_u32 v[vgprLocalWriteAddrA], v0, v[vgprLocalWriteAddrA] // lwFOA = (lwAA + lwAL*(MT0I+PAD))
v_lshlrev_b32 v[vgprLocalWriteAddrA], 2, v[vgprLocalWriteAddrA] //  (multiple bpe)

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x84, v5     // lwBL**(MTB + PAD)
v_add_nc_u32 v[vgprLocalWriteAddrB], v2, v[vgprLocalWriteAddrB] // lwFOB = (lwBB + lwBL*(MT1J+PAD))
v_lshlrev_b32 v[vgprLocalWriteAddrB], 2, v[vgprLocalWriteAddrB] //  (multiple bpe)
v_add_co_u32 v[vgprLocalWriteAddrB], vcc_lo, 0x4080, v[vgprLocalWriteAddrB] // lwFOB = lw1J + lwL*MT1J + LDS_OFFSET_B=16512
s_waitcnt lgkmcnt(0)                               // wait for 88/0 bytes of kern args
v_mov_b32 v8, MT0                                  // set MT0 into sgpr
v_mov_b32 v7, s[sgprSizesFree+0]                   // set Free0 size
v_cvt_f32_u32 v6, v8                               // v6 = ceil(v7 / v8)
v_rcp_iflag_f32 v6, v6                             // v6 = ceil(v7 / v8)
v_cvt_f32_u32 v9, v7                               // v6 = ceil(v7 / v8)
v_mul_f32 v6, v6, v9                               // v6 = ceil(v7 / v8)
v_cvt_u32_f32 v6, v6                               // v6 = ceil(v7 / v8)
v_mul_u32_u24 v9, v6, v8                           // v6 = ceil(v7 / v8)
v_sub_nc_u32 v9, v7, v9                            // v6 = ceil(v7 / v8)
v_cmp_ne_u32 vcc_lo, v9, 0                         // v6 = ceil(v7 / v8)
v_add_co_ci_u32 v6, vcc_lo, v6, 0, vcc_lo          // ceil
v_mov_b32 v8, MT1                                  // set MT1 into sgpr
v_mov_b32 v7, s[sgprSizesFree+1]                   // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v6      // set back to numWorkGroup0
v_cvt_f32_u32 v6, v8                               // v6 = ceil(v7 / v8)
v_rcp_iflag_f32 v6, v6                             // v6 = ceil(v7 / v8)
v_cvt_f32_u32 v9, v7                               // v6 = ceil(v7 / v8)
v_mul_f32 v6, v6, v9                               // v6 = ceil(v7 / v8)
v_cvt_u32_f32 v6, v6                               // v6 = ceil(v7 / v8)
v_mul_u32_u24 v9, v6, v8                           // v6 = ceil(v7 / v8)
v_sub_nc_u32 v9, v7, v9                            // v6 = ceil(v7 / v8)
v_cmp_ne_u32 vcc_lo, v9, 0                         // v6 = ceil(v7 / v8)
v_add_co_ci_u32 v6, vcc_lo, v6, 0, vcc_lo          // ceil
v_readfirstlane_b32 s[sgprNumWorkGroups1], v6      // set back to numWorkGroup1

/* remap wg from 1D(idxWG012) to 3D(wg2,wg1,wg0) */
/* wg2 = idxWG012 * smallMagicNumber(1/(numWG0*numWG1)) */
s_mul_i32 s16, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_and_b32 s17, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s16, s16, s17
v_cvt_f32_u32 v6, s16                              // s16 = s[sgprWorkGroup0] / s16
v_rcp_iflag_f32 v6, v6                             // s16 = s[sgprWorkGroup0] / s16
v_cvt_f32_u32 v7, s[sgprWorkGroup0]                // s16 = s[sgprWorkGroup0] / s16
v_mul_f32 v6, v6, v7                               // s16 = s[sgprWorkGroup0] / s16
v_cvt_u32_f32 v6, v6                               // s16 = s[sgprWorkGroup0] / s16
v_mul_u32_u24 v7, v6, s16                          // s16 = s[sgprWorkGroup0] / s16
v_sub_nc_u32 v7, s[sgprWorkGroup0], v7             // s16 = s[sgprWorkGroup0] / s16
v_cmp_eq_u32 vcc_lo, v7, s16                       // s16 = s[sgprWorkGroup0] / s16
s_mov_b32 exec_lo, vcc_lo                          // s16 = s[sgprWorkGroup0] / s16
v_add_nc_u32 v6, 1, v6                             // s16 = s[sgprWorkGroup0] / s16
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v7, s16                       // overflow happened in remainder
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v6, v6, 1                             // quotient - 1
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s16, v6                        // quotient
s_mov_b32 s[sgprWorkGroup2], s16
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s16, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s16, s16, s[sgprWorkGroup2]
s_mul_i32 s16, s16, s17
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s16
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v6, s[sgprNumWorkGroups0]            // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v6, v6                             // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_f32_u32 v7, s[sgprWorkGroup0]                // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_f32 v6, v6, v7                               // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_u32_f32 v6, v6                               // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_u32_u24 v7, v6, s[sgprNumWorkGroups0]        // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_sub_nc_u32 v7, s[sgprWorkGroup0], v7             // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmp_eq_u32 vcc_lo, v7, s[sgprNumWorkGroups0]     // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b32 exec_lo, vcc_lo                          // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_add_nc_u32 v6, 1, v6                             // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v7, s[sgprNumWorkGroups0]     // overflow happened in remainder
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v6, v6, 1                             // quotient - 1
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s16, v6                        // quotient
s_mov_b32 s[sgprWorkGroup1], s16
/* wg0 = idxWG01 - wg1 * numWG0 */
s_mul_i32 s16, s[sgprWorkGroup1], s[sgprNumWorkGroups0]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s16
s_branch label_MultiGemmEnd
label_MultiGemm:

/* Check if custom structure pointer is null */
s_and_b32 s16, s[sgprArgType], 0xff                // mask ArgType domain (bit 8 = TDM wave-parity)
s_cmp_eq_u32 s16, 2                                // ArgType == 2 ?
s_cbranch_scc1 label_IsExternalValid               // branch if ArgType == 2
s_mov_b32 s15, 88                                  // KernArgAddressOffset
s_mul_i32 s54, s20, 4
s_mov_b64 s[48:49], s[sgprKernArgAddress:sgprKernArgAddress+1]
s_branch label_IsExternalValidEnd
label_IsExternalValid:
s_mov_b32 s15, 228
s_mov_b32 s54, 0
s_mov_b64 s[48:49], s[sgprKernArgAddress:sgprKernArgAddress+1]
label_IsExternalValidEnd:

/* Grouped Gemm:: prefetch 1 arg load */
s_mov_b32 s14, 1
s_mov_b32 s55, 0
s_load_b128 s[24:27], s[48:49], s54
s_cmpk_eq_u32 s20, 1                               // if gemm_count is 1?
s_cbranch_scc1 label_wgTable_noLoadLoop

/* Grouped Gemm:: accumulate numTiles for each gemm */
/* Grouped Gemm:: loop start */
label_Loop_GemmCount:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s52, s24, 9                             // s52 = s24 / 512
s_and_b32 s50, 511, s24                            // s50 = s24 % 512
s_addc_u32 s52, s52, 0
s_lshr_b32 s53, s25, 7                             // s53 = s25 / 128
s_and_b32 s50, 127, s25                            // s50 = s25 % 128
s_addc_u32 s53, s53, 0
s_mul_i32 s52, s52, s53
s_mul_i32 s52, s52, s26
s_and_b32 s53, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s52, s52, s53
s_add_u32 s55, s55, s52
s_cmp_lt_u32 s[sgprWorkGroup0], s55
s_cbranch_scc1 label_FOUND
s_add_u32 s54, s54, s15
s_load_b128 s[24:27], s[48:49], s54
s_add_u32 s14, s14, 1
s_cmp_lt_u32 s14, s20
s_cbranch_scc1 label_Loop_GemmCount

/* Grouped Gemm:: noLoadLoop */
label_wgTable_noLoadLoop:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s52, s24, 9                             // s52 = s24 / 512
s_and_b32 s50, 511, s24                            // s50 = s24 % 512
s_addc_u32 s52, s52, 0
s_lshr_b32 s53, s25, 7                             // s53 = s25 / 128
s_and_b32 s50, 127, s25                            // s50 = s25 % 128
s_addc_u32 s53, s53, 0
s_mul_i32 s52, s52, s53
s_mul_i32 s52, s52, s26
s_and_b32 s48, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s52, s52, s48
s_add_u32 s55, s55, s52

/* Grouped Gemm:: gemmIndex found */
label_FOUND:
s_sub_u32 s49, s14, 1
s_sub_u32 s48, s55, s52
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s48
/* Check if custom structure pointer is null */
s_and_b32 s16, s[sgprArgType], 0xff                // mask ArgType domain (bit 8 = TDM wave-parity)
s_cmp_eq_u32 s16, 2                                // ArgType == 2 ?
s_cbranch_scc1 label_LoadExternalStruct            // branch if ArgType == 2

/* Grouped Gemm: offset argument address to gemm */
/* Grouped Gemm: offset address from wg_table_start to args_start */
s_lshl2_add_u32 s[sgprKernArgAddress], s20, s[sgprKernArgAddress]
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s49, s49, 88                             // KernArgAddressOffset
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s49
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_b512 s[28:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_b64 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
s_branch label_LoadExternalStructEnd
label_LoadExternalStruct:
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s49, s49, 228
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s49
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
s_load_b512 s[28:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_b32 s44, s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
// Read Beta
s_load_b32 s45, s[sgprKernArgAddress:sgprKernArgAddress+1], 96 // 96
label_LoadExternalStructEnd:
/* init: add vgpr [136...321) to pool */
/* init: add vgpr [0...128) to pool */
/* init: add agpr [0...0) to pool */

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
v_lshrrev_b32 v0, 5, v[vgprSerial]                 // 0 = Serial / 32
v_and_b32 v1, 31, v[vgprSerial]                    // 1 = Serial % 32
v_lshrrev_b32 v2, 4, v0                            // 2 = 0 / 16
v_and_b32 v3, 15, v0                               // 3 = 0 % 16

/* local read addresses: final offsets a */
v_lshrrev_b32 v0, 9, v[vgprSerial]                 // LSU offset: sgid = Serial / subGroup(512)
s_mov_b32 s16, 516                                 // LSU offset: lsuoffset = sgid*(MT0+PAD)
v_mul_lo_u32 v0, s16, v0                           // LSU offset: lsuoffset = sgid*(MT0+PAD)
v_lshlrev_b32 v1, 1, v1                            // Final Offset: lrAOffset * VW
v_add_lshl_u32 v[vgprLocalReadAddrA], v0, v1, 0x2  // Final Offset: add padding 16 per block 0

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 9, v[vgprSerial]                 // LSU offset: sgid = Serial / subGroup(512)
s_mov_b32 s16, 132                                 // LSU offset: lsuoffset = sgid*(MT1+PAD)
v_mul_lo_u32 v0, s16, v0                           // LSU offset: lsuoffset = sgid*(MT1+PAD)
v_lshlrev_b32 v3, 1, v3                            // Final Offset: lrBOffset * VW
v_add_lshl_u32 v[vgprLocalReadAddrB], v0, v3, 0x2  // Final Offset: add padding 16 per block 0

/* local read addresses: declare addresses a */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, 0x4080, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 4 */
/* v1 = A-unroll = serial%LVCA */
v_lshrrev_b32 v0, 2, v[vgprSerial]                 // 0 = Serial / 4
v_and_b32 v1, 3, v[vgprSerial]                     // 1 = Serial % 4
/* unroll *= glvw */
v_lshlrev_b32 v1, 1, v1                            // v1 = v1 * 2
v_mov_b32 v4, v1                                   // copy for GlobalSplitU
/* LVCB = 4 */
/* v3 = B-unroll = serial%LVCB */
v_lshrrev_b32 v2, 2, v[vgprSerial]                 // 2 = Serial / 4
v_and_b32 v3, 3, v[vgprSerial]                     // 3 = Serial % 4
/* unroll *= glvw */
v_lshlrev_b32 v3, 1, v3                            // v3 = v3 * 2
v_mov_b32 v5, v3                                   // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v4 */
/* lwaUnrollAssignmentB = v5 */

/* local write addresses: first offset a */
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x204, v4    // lwAL**(MTA + PAD)
v_add_nc_u32 v[vgprLocalWriteAddrA], v0, v[vgprLocalWriteAddrA] // lwFOA = (lwAA + lwAL*(MT0I+PAD))
v_lshlrev_b32 v[vgprLocalWriteAddrA], 2, v[vgprLocalWriteAddrA] //  (multiple bpe)

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x84, v5     // lwBL**(MTB + PAD)
v_add_nc_u32 v[vgprLocalWriteAddrB], v2, v[vgprLocalWriteAddrB] // lwFOB = (lwBB + lwBL*(MT1J+PAD))
v_lshlrev_b32 v[vgprLocalWriteAddrB], 2, v[vgprLocalWriteAddrB] //  (multiple bpe)
v_add_co_u32 v[vgprLocalWriteAddrB], vcc_lo, 0x4080, v[vgprLocalWriteAddrB] // lwFOB = lw1J + lwL*MT1J + LDS_OFFSET_B=16512
s_waitcnt lgkmcnt(0)                               // wait for 88/0 bytes of kern args
v_mov_b32 v8, MT0                                  // set MT0 into sgpr
v_mov_b32 v7, s[sgprSizesFree+0]                   // set Free0 size
v_cvt_f32_u32 v6, v8                               // v6 = ceil(v7 / v8)
v_rcp_iflag_f32 v6, v6                             // v6 = ceil(v7 / v8)
v_cvt_f32_u32 v9, v7                               // v6 = ceil(v7 / v8)
v_mul_f32 v6, v6, v9                               // v6 = ceil(v7 / v8)
v_cvt_u32_f32 v6, v6                               // v6 = ceil(v7 / v8)
v_mul_u32_u24 v9, v6, v8                           // v6 = ceil(v7 / v8)
v_sub_nc_u32 v9, v7, v9                            // v6 = ceil(v7 / v8)
v_cmp_ne_u32 vcc_lo, v9, 0                         // v6 = ceil(v7 / v8)
v_add_co_ci_u32 v6, vcc_lo, v6, 0, vcc_lo          // ceil
v_mov_b32 v8, MT1                                  // set MT1 into sgpr
v_mov_b32 v7, s[sgprSizesFree+1]                   // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v6      // set back to numWorkGroup0
v_cvt_f32_u32 v6, v8                               // v6 = ceil(v7 / v8)
v_rcp_iflag_f32 v6, v6                             // v6 = ceil(v7 / v8)
v_cvt_f32_u32 v9, v7                               // v6 = ceil(v7 / v8)
v_mul_f32 v6, v6, v9                               // v6 = ceil(v7 / v8)
v_cvt_u32_f32 v6, v6                               // v6 = ceil(v7 / v8)
v_mul_u32_u24 v9, v6, v8                           // v6 = ceil(v7 / v8)
v_sub_nc_u32 v9, v7, v9                            // v6 = ceil(v7 / v8)
v_cmp_ne_u32 vcc_lo, v9, 0                         // v6 = ceil(v7 / v8)
v_add_co_ci_u32 v6, vcc_lo, v6, 0, vcc_lo          // ceil
v_readfirstlane_b32 s[sgprNumWorkGroups1], v6      // set back to numWorkGroup1

/* Early stop if N(SizeFreeJ) == 0 */
s_cmp_eq_u32 s[sgprSizeJ], 0
s_cbranch_scc0 label_NoEarlyStop_N0
label_EarlyStop_if_N_is_0:
s_endpgm
label_NoEarlyStop_N0:

/* remap wg from 1D(idxWG012) to 3D(wg2,wg1,wg0) */
/* wg2 = idxWG012 * smallMagicNumber(1/(numWG0*numWG1)) */
s_mul_i32 s16, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_and_b32 s17, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s16, s16, s17
v_cvt_f32_u32 v6, s16                              // s16 = s[sgprWorkGroup0] / s16
v_rcp_iflag_f32 v6, v6                             // s16 = s[sgprWorkGroup0] / s16
v_cvt_f32_u32 v7, s[sgprWorkGroup0]                // s16 = s[sgprWorkGroup0] / s16
v_mul_f32 v6, v6, v7                               // s16 = s[sgprWorkGroup0] / s16
v_cvt_u32_f32 v6, v6                               // s16 = s[sgprWorkGroup0] / s16
v_mul_u32_u24 v7, v6, s16                          // s16 = s[sgprWorkGroup0] / s16
v_sub_nc_u32 v7, s[sgprWorkGroup0], v7             // s16 = s[sgprWorkGroup0] / s16
v_cmp_eq_u32 vcc_lo, v7, s16                       // s16 = s[sgprWorkGroup0] / s16
s_mov_b32 exec_lo, vcc_lo                          // s16 = s[sgprWorkGroup0] / s16
v_add_nc_u32 v6, 1, v6                             // s16 = s[sgprWorkGroup0] / s16
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v7, s16                       // overflow happened in remainder
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v6, v6, 1                             // quotient - 1
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s16, v6                        // quotient
s_mov_b32 s[sgprWorkGroup2], s16
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s16, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s16, s16, s[sgprWorkGroup2]
s_mul_i32 s16, s16, s17
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s16
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v6, s[sgprNumWorkGroups0]            // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v6, v6                             // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_f32_u32 v7, s[sgprWorkGroup0]                // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_f32 v6, v6, v7                               // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_u32_f32 v6, v6                               // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_u32_u24 v7, v6, s[sgprNumWorkGroups0]        // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_sub_nc_u32 v7, s[sgprWorkGroup0], v7             // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmp_eq_u32 vcc_lo, v7, s[sgprNumWorkGroups0]     // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b32 exec_lo, vcc_lo                          // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_add_nc_u32 v6, 1, v6                             // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v7, s[sgprNumWorkGroups0]     // overflow happened in remainder
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v6, v6, 1                             // quotient - 1
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s16, v6                        // quotient
s_mov_b32 s[sgprWorkGroup1], s16
/* wg0 = idxWG01 - wg1 * numWG0 */
s_mul_i32 s16, s[sgprWorkGroup1], s[sgprNumWorkGroups0]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s16

/* Early stop if wg exceed */
s_cmp_ge_u32 s[sgprWorkGroup2], s[sgprSizesFree+2]
s_cbranch_scc0 label_NoEarlyStop_wgExceed
label_EarlyStop_if_wg_exceed:
s_endpgm
label_NoEarlyStop_wgExceed:

label_MultiGemmEnd:
.set sgprSrdA, 48
.set sgprSrdB, 52
.set sgprShadowLimitA, 56
.set sgprShadowLimitB, 58
.set sgprStaggerUIter, 47
.set sgprWrapUA, 60
.set sgprWrapUB, 62
.set sgprGlobalReadIncsA, 64
.set sgprGlobalReadIncsB, 65
.set sgprScalarGlobalReadOffsetA, 66
s_and_b32 s16, s[sgprArgType], 0xff                // mask ArgType domain (bit 8 = TDM wave-parity)
s_cmp_eq_u32 s16, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_Skip_Address_Prepad_For_Pointer_Array
s_sub_u32 s[sgprAddressA+0], s[sgprAddressA+0], 8  // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressA+1], s[sgprAddressA+1], 0 // pre-pad to make room for possible pointer shift
s_sub_u32 s[sgprAddressB+0], s[sgprAddressB+0], 8  // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressB+1], s[sgprAddressB+1], 0 // pre-pad to make room for possible pointer shift
label_Skip_Address_Prepad_For_Pointer_Array:  /// Skip pre-padding of address for pointer array case

/* Short circuit condition if Alpha == 0, then sumDims=0 */
v_cmp_eq_f32 vcc_lo, s[sgprAlpha], 0.0             // s[Alpha] == 0.0f ?
s_cbranch_vccz label_AlphaNonZero                  // branch if s[Alpha] != 0
label_AlphaNonZero:

/******************************************/
/* Begin setupNewTile                     */
/******************************************/

/* global read addresses: work-group */
/* graWorkGroup mapping */
s_and_b32 s16, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s16, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU                           // branch if GSU == 1
// GSU-not-WGMapRR :nwg1 = (size1J + MT1J - 1) / MT1J;
s_and_b32 s16, s[sgprGSU], 0x4000                  // SCC = (GSUWGMRR == 1) ?
s_cbranch_scc1 label_GSUWGMRR                      // branch if GSUWGMRR == 1
s_and_b32 s16, s[sgprGSU], 0x3fff                  // Restore GSU
v_cvt_f32_u32 v6, s16                              // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_rcp_iflag_f32 v6, v6                             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_cvt_f32_u32 v7, s[sgprWorkGroup1]                // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_mul_f32 v6, v6, v7                               // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_cvt_u32_f32 v6, v6                               // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_mul_u32_u24 v7, v6, s16                          // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_sub_nc_u32 v7, s[sgprWorkGroup1], v7             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_cmp_eq_u32 vcc_lo, v7, s16                       // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
s_mov_b32 exec_lo, vcc_lo                          // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_add_nc_u32 v6, 1, v6                             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_mov_b32 v7, 0                                    // s[sgprGSUSumIdx] = s[sgprWorkGroup1] % s16
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v7, s16                       // overflow happened in remainder
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v6, v6, 1                             // quotient - 1
v_mul_u32_u24 v7, v6, s16                          // re-calculate remainder
v_sub_nc_u32 v7, s[sgprWorkGroup1], v7             // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprWorkGroup1], v6          // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx], v7           // remainder
s_branch label_GSUWGMRR_End
label_GSUWGMRR:
v_cvt_f32_u32 v6, s[sgprNumWorkGroups1]            // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_rcp_iflag_f32 v6, v6                             // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_cvt_f32_u32 v7, s[sgprWorkGroup1]                // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_mul_f32 v6, v6, v7                               // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_cvt_u32_f32 v6, v6                               // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_mul_u32_u24 v7, v6, s[sgprNumWorkGroups1]        // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_sub_nc_u32 v7, s[sgprWorkGroup1], v7             // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_cmp_eq_u32 vcc_lo, v7, s[sgprNumWorkGroups1]     // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
s_mov_b32 exec_lo, vcc_lo                          // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_add_nc_u32 v6, 1, v6                             // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_mov_b32 v7, 0                                    // s[sgprWorkGroup1] = s[sgprWorkGroup1] % s[sgprNumWorkGroups1]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v7, s[sgprNumWorkGroups1]     // overflow happened in remainder
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v6, v6, 1                             // quotient - 1
v_mul_u32_u24 v7, v6, s[sgprNumWorkGroups1]        // re-calculate remainder
v_sub_nc_u32 v7, s[sgprWorkGroup1], v7             // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprGSUSumIdx], v6           // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v7          // remainder
label_GSUWGMRR_End:
s_mov_b32 s[sgprGSULog2BpeC], 2
s_mov_b32 s[sgprGSULog2BpeD], 2
s_branch label_GSU_End
label_GSU:
s_mov_b64 s[sgprGSUSumIdx:sgprGSUSumIdx+1], 0      // Set GSUSumIdx to 0
s_mov_b32 s[sgprGSULog2BpeC], 2
s_mov_b32 s[sgprGSULog2BpeD], 2
label_GSU_End:
/* WGM Calculation */
s_mov_b32 s16, s[sgprWGM]                          // Restore WGM
s_sext_i32_i16 s16, s16                            // Restore WGM
s_cmp_gt_i32 s16, 1                                // WGM > 1 ?
s_cbranch_scc1 label_WGMPositive                   // branch if WGM > 1
s_cmp_ge_i32 s16, 0                                // WGM >= 0 ?
s_cbranch_scc1 label_WGM                           // branch if WGM >= 0
s_abs_i32 s16, s16                                 // abs(WGM)
v_cvt_f64_u32 v[6:7], s16                          // s17 = s[sgprWorkGroup0] / s16
v_rcp_f64 v[6:7], v[6:7]                           // s17 = s[sgprWorkGroup0] / s16
v_cvt_f64_u32 v[8:9], s[sgprWorkGroup0]            // s17 = s[sgprWorkGroup0] / s16
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s17 = s[sgprWorkGroup0] / s16
v_cvt_u32_f64 v6, v[6:7]                           // s17 = s[sgprWorkGroup0] / s16
v_mul_lo_u32 v7, v6, s16                           // s17 = s[sgprWorkGroup0] / s16
v_sub_nc_u32 v8, s[sgprWorkGroup0], v7             // s17 = s[sgprWorkGroup0] / s16
v_cmp_ge_u32 vcc_lo, v8, s16                       // s17 = s[sgprWorkGroup0] / s16
s_mov_b32 exec_lo, vcc_lo                          // s17 = s[sgprWorkGroup0] / s16
v_add_nc_u32 v6, v6, 1                             // s17 = s[sgprWorkGroup0] / s16
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s17, v6                        // quotient
s_mul_i32 s20, s17, s16                            // quotient * non-magic divisor
s_sub_u32 s20, s[sgprWorkGroup0], s20              // WorkGroup0=remainder
s_mul_i32 s20, s20, s[sgprNumWorkGroups1]          // (wg1 % WGM)*NumWorkGroups1
s_add_u32 s20, s20, s[sgprWorkGroup1]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups1
v_cvt_f64_u32 v[6:7], s16                          // s18 = s[sgprNumWorkGroups0] / s16
v_rcp_f64 v[6:7], v[6:7]                           // s18 = s[sgprNumWorkGroups0] / s16
v_cvt_f64_u32 v[8:9], s[sgprNumWorkGroups0]        // s18 = s[sgprNumWorkGroups0] / s16
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s18 = s[sgprNumWorkGroups0] / s16
v_cvt_u32_f64 v6, v[6:7]                           // s18 = s[sgprNumWorkGroups0] / s16
v_mul_lo_u32 v7, v6, s16                           // s18 = s[sgprNumWorkGroups0] / s16
v_sub_nc_u32 v8, s[sgprNumWorkGroups0], v7         // s18 = s[sgprNumWorkGroups0] / s16
v_cmp_ge_u32 vcc_lo, v8, s16                       // s18 = s[sgprNumWorkGroups0] / s16
s_mov_b32 exec_lo, vcc_lo                          // s18 = s[sgprNumWorkGroups0] / s16
v_add_nc_u32 v6, v6, 1                             // s18 = s[sgprNumWorkGroups0] / s16
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s18, v6                        // quotient
s_mul_i32 s19, s16, s18                            // quotient * non-magic divisor
s_sub_u32 s19, s[sgprNumWorkGroups0], s19          // NumWorkGroups0=remainder
s_cmp_eq_u32 s19, 0                                // remainder == 0 ?
s_cmov_b32 s19, s16                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s17, s18                              // blockId >= numFullBlocks ?
s_cselect_b32 s18, s19, s16
v_cvt_f64_u32 v[6:7], s18                          // s[sgprWorkGroup1] = s20 / s18
v_rcp_f64 v[6:7], v[6:7]                           // s[sgprWorkGroup1] = s20 / s18
v_cvt_f64_u32 v[8:9], s20                          // s[sgprWorkGroup1] = s20 / s18
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s[sgprWorkGroup1] = s20 / s18
v_cvt_u32_f64 v6, v[6:7]                           // s[sgprWorkGroup1] = s20 / s18
v_mul_lo_u32 v7, v6, s18                           // s[sgprWorkGroup1] = s20 / s18
v_sub_nc_u32 v8, s20, v7                           // s[sgprWorkGroup1] = s20 / s18
v_cmp_ge_u32 vcc_lo, v8, s18                       // s[sgprWorkGroup1] = s20 / s18
s_mov_b32 exec_lo, vcc_lo                          // s[sgprWorkGroup1] = s20 / s18
v_add_nc_u32 v6, v6, 1                             // s[sgprWorkGroup1] = s20 / s18
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v7, v6, s18                           // s[sgprWorkGroup1] = s20 / s18
v_sub_nc_u32 v8, s20, v7                           // s[sgprWorkGroup1] = s20 / s18
v_readfirstlane_b32 s[sgprWorkGroup1], v6          // quotient
v_readfirstlane_b32 s[sgprWorkGroup0], v8          // remainder
s_mul_i32 s[sgprWorkGroup0], s[sgprWorkGroup1], s18 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup0], s20, s[sgprWorkGroup0] // WorkGroup0=remainder
s_mul_i32 s17, s17, s16                            // blockId * WGM
s_add_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s17 // wg1 += blockId * WGM
s_branch label_WGM
label_WGMPositive:
s_mov_b32 s16, s16                                 // WGM
v_cvt_f64_u32 v[6:7], s16                          // s17 = s[sgprWorkGroup1] / s16
v_rcp_f64 v[6:7], v[6:7]                           // s17 = s[sgprWorkGroup1] / s16
v_cvt_f64_u32 v[8:9], s[sgprWorkGroup1]            // s17 = s[sgprWorkGroup1] / s16
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s17 = s[sgprWorkGroup1] / s16
v_cvt_u32_f64 v6, v[6:7]                           // s17 = s[sgprWorkGroup1] / s16
v_mul_lo_u32 v7, v6, s16                           // s17 = s[sgprWorkGroup1] / s16
v_sub_nc_u32 v8, s[sgprWorkGroup1], v7             // s17 = s[sgprWorkGroup1] / s16
v_cmp_ge_u32 vcc_lo, v8, s16                       // s17 = s[sgprWorkGroup1] / s16
s_mov_b32 exec_lo, vcc_lo                          // s17 = s[sgprWorkGroup1] / s16
v_add_nc_u32 v6, v6, 1                             // s17 = s[sgprWorkGroup1] / s16
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s17, v6                        // quotient
s_mul_i32 s20, s17, s16                            // quotient * non-magic divisor
s_sub_u32 s20, s[sgprWorkGroup1], s20              // WorkGroup1=remainder
s_mul_i32 s20, s20, s[sgprNumWorkGroups0]          // (wg1 % WGM)*NumWorkGroups0
s_add_u32 s20, s20, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups0
v_cvt_f64_u32 v[6:7], s16                          // s18 = s[sgprNumWorkGroups1] / s16
v_rcp_f64 v[6:7], v[6:7]                           // s18 = s[sgprNumWorkGroups1] / s16
v_cvt_f64_u32 v[8:9], s[sgprNumWorkGroups1]        // s18 = s[sgprNumWorkGroups1] / s16
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s18 = s[sgprNumWorkGroups1] / s16
v_cvt_u32_f64 v6, v[6:7]                           // s18 = s[sgprNumWorkGroups1] / s16
v_mul_lo_u32 v7, v6, s16                           // s18 = s[sgprNumWorkGroups1] / s16
v_sub_nc_u32 v8, s[sgprNumWorkGroups1], v7         // s18 = s[sgprNumWorkGroups1] / s16
v_cmp_ge_u32 vcc_lo, v8, s16                       // s18 = s[sgprNumWorkGroups1] / s16
s_mov_b32 exec_lo, vcc_lo                          // s18 = s[sgprNumWorkGroups1] / s16
v_add_nc_u32 v6, v6, 1                             // s18 = s[sgprNumWorkGroups1] / s16
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s18, v6                        // quotient
s_mul_i32 s19, s16, s18                            // quotient * non-magic divisor
s_sub_u32 s19, s[sgprNumWorkGroups1], s19          // NumWorkGroups1=remainder
s_cmp_eq_u32 s19, 0                                // remainder == 0 ?
s_cmov_b32 s19, s16                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s17, s18                              // blockId >= numFullBlocks ?
s_cselect_b32 s18, s19, s16
v_cvt_f64_u32 v[6:7], s18                          // s[sgprWorkGroup0] = s20 / s18
v_rcp_f64 v[6:7], v[6:7]                           // s[sgprWorkGroup0] = s20 / s18
v_cvt_f64_u32 v[8:9], s20                          // s[sgprWorkGroup0] = s20 / s18
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s[sgprWorkGroup0] = s20 / s18
v_cvt_u32_f64 v6, v[6:7]                           // s[sgprWorkGroup0] = s20 / s18
v_mul_lo_u32 v7, v6, s18                           // s[sgprWorkGroup0] = s20 / s18
v_sub_nc_u32 v8, s20, v7                           // s[sgprWorkGroup0] = s20 / s18
v_cmp_ge_u32 vcc_lo, v8, s18                       // s[sgprWorkGroup0] = s20 / s18
s_mov_b32 exec_lo, vcc_lo                          // s[sgprWorkGroup0] = s20 / s18
v_add_nc_u32 v6, v6, 1                             // s[sgprWorkGroup0] = s20 / s18
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v7, v6, s18                           // s[sgprWorkGroup0] = s20 / s18
v_sub_nc_u32 v8, s20, v7                           // s[sgprWorkGroup0] = s20 / s18
v_readfirstlane_b32 s[sgprWorkGroup0], v6          // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v8          // remainder
s_mul_i32 s[sgprWorkGroup1], s[sgprWorkGroup0], s18 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup1], s20, s[sgprWorkGroup1] // WorkGroup1=remainder
s_mul_i32 s17, s17, s16                            // blockId * WGM
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s17 // wg1 += blockId * WGM
label_WGM:

/* global read addresses: tile offset assignment a */
/* graTileAssignmentA = v0 */

/* global read addresses: tile offset assignment b */
/* graTileAssignmentB = v2 */

/* global read addresses: unroll assignment a */
/* v1 */

/* global read addresses: unroll assignment b */
/* v3 */

/* global read addresses: other free assignments */
/* s[sgprWorkGroup2] */

/* global read addresses: tile offsets a */

/* global read addresses: tile offsets b */

/* global read addresses: unroll offsets a */

/* global read addresses: unroll offsets b */

/* global read addresses: addresses a */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s19, s[sgprWorkGroup0], 512           // WorkGroup[01] * MT
s_mul_i32 s18, s[sgprWorkGroup0], 512              // WorkGroup[01] * MT
s_mul_hi_u32 s19, s18, s[sgprStrideA0I]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s18, s18, s[sgprStrideA0I]               // tlu=0, scaled tile-offset by stride
s_and_b32 s16, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_A                        // branch if GSUC == 1
s_mul_hi_u32 s17, 8, s[sgprGSUSumIdx]              // gsuOffset = DepthU*GSUSumIdx
s_mul_i32 s16, 8, s[sgprGSUSumIdx]                 // gsuOffset = DepthU*GSUSumIdx
s_branch label_GSUC_A_End
label_GSUC_A:
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum], 3 // s[LoopCounterL] = s[sgprSizesSum] / 8
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0x3fff   // Restore GSU
v_cvt_f32_u32 v6, s[sgprGSUSumIdx+1]               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v6, v6                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v7, s[sgprLoopCounterL]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v6, v6, v7                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v6, v6                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v7, v6, s[sgprGSUSumIdx+1]           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_nc_u32 v7, s[sgprLoopCounterL], v7           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmp_eq_u32 vcc_lo, v7, s[sgprGSUSumIdx+1]        // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, vcc_lo                          // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_nc_u32 v6, 1, v6                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v7, 0                                    // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v7, s[sgprGSUSumIdx+1]        // overflow happened in remainder
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v6, v6, 1                             // quotient - 1
v_mul_u32_u24 v7, v6, s[sgprGSUSumIdx+1]           // re-calculate remainder
v_sub_nc_u32 v7, s[sgprLoopCounterL], v7           // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v6        // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v7         // remainder
s_mul_i32 s17, s[sgprLoopCounterL], s[sgprGSUSumIdx] // quotient*GSUSumIdx
s_add_u32 s16, 1, s[sgprLoopCounterL]              // quotient+1
s_add_u32 s17, s17, s[sgprGSUSumIdx+1]             // quotient*GSUSumIdx+remainder
s_mul_i32 s16, s16, s[sgprGSUSumIdx]               // (quotient+1)*GSUSumIdx
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cselect_b32 s16, s16, s17                        // (quotient+1)*GSUSumIdx if needed
s_mul_hi_u32 s17, s16, 8                           // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
s_mul_i32 s16, s16, 8                              // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
label_GSUC_A_End:
s_add_u32 s18, s18, s16                            // accum GsuOffset term to tilestart
s_addc_u32 s19, s19, s17                           // accum GsuOffset term to tilestart
s_mov_b64 s[sgprShadowLimitA+0:sgprShadowLimitA+0+1], 1 // Init tensor size
s_sub_u32 s16, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s17, constStrideAL, s16               // stride x (size-1)
s_mul_i32 s16, constStrideAL, s16                  // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s16 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s17 // sum tensor size
s_sub_u32 s16, s[sgprSizeI], 1                     // (size-1)
s_mul_hi_u32 s17, s[sgprStrideA0I], s16            // stride x (size-1)
s_mul_i32 s16, s[sgprStrideA0I], s16               // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s16 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s17 // sum tensor size
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s18 // sub tileStart
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s19 // sub tileStart
s_lshl_b64 s[sgprShadowLimitA:sgprShadowLimitA+1], s[sgprShadowLimitA:sgprShadowLimitA+1], 2 // Set limit to use bytes (multiple bpe)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 8 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_and_b32 s20, s[sgprArgType], 0xff                // mask ArgType domain (bit 8 = TDM wave-parity)
s_cmp_eq_u32 s20, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_StridedBatchedGemmLoadA
s_mul_i32 s16, 8, s[sgprWorkGroup2]                // Compute Offset into Pointer Array
s_cmp_eq_u32 s[sgprSizesSum], 0x0                  // Don't dereference Pointer array if SizesSum == 0
s_cbranch_scc1 label_StridedBatchedGemmLoadA_End
s_add_u32 s16, s16, s[sgprAddressA+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s17, s[sgprAddressA+1], 0               // Offsetting to the location [Higher half of address]
s_load_b64 s[sgprSrdA:sgprSrdA+1], s[16:17], 0     // Load the Matrix Address in the Pointer Array
s_load_b64 s[16:17], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x68 // Load batchOffsetA from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s16        // Add batch offset to A address (low)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s17       // Add batch offset to A address (high)
s_sub_u32 s[sgprSrdA+0], s[sgprSrdA+0], 8          // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0         // pre-pad to make room for possible pointer shift
s_lshl_b64 s[18:19], s[18:19], 2                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdA+0], s18, s[sgprSrdA+0]        // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s19, s[sgprSrdA+1]       // SRD base = Address+ tileStart1
s_branch label_StridedBatchedGemmLoadA_End
label_StridedBatchedGemmLoadA:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s17, s[sgprStrideAK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s16, s[sgprStrideAK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s18, s18, s16                            // accum wg term to tilestart
s_addc_u32 s19, s19, s17                           // accum wg term to tilestart
s_lshl_b64 s[18:19], s[18:19], 2                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdA+0], s[sgprAddressA+0], s18    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprAddressA+1], s19   // SRD base = Address+ tileStart1
label_StridedBatchedGemmLoadA_End:  /// End Computing the Batch Matrix's base address for Strided Batched
s_mov_b32 s[sgprSrdA+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: addresses b */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s19, s[sgprWorkGroup1], 128           // WorkGroup[01] * MT
s_mul_i32 s18, s[sgprWorkGroup1], 128              // WorkGroup[01] * MT
s_mul_hi_u32 s19, s18, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s18, s18, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_and_b32 s16, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_B                        // branch if GSUC == 1
s_mul_hi_u32 s17, 8, s[sgprGSUSumIdx]              // gsuOffset = DepthU*GSUSumIdx
s_mul_i32 s16, 8, s[sgprGSUSumIdx]                 // gsuOffset = DepthU*GSUSumIdx
s_branch label_GSUC_B_End
label_GSUC_B:
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum], 3 // s[LoopCounterL] = s[sgprSizesSum] / 8
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0x3fff   // Restore GSU
v_cvt_f32_u32 v6, s[sgprGSUSumIdx+1]               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v6, v6                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v7, s[sgprLoopCounterL]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v6, v6, v7                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v6, v6                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v7, v6, s[sgprGSUSumIdx+1]           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_nc_u32 v7, s[sgprLoopCounterL], v7           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmp_eq_u32 vcc_lo, v7, s[sgprGSUSumIdx+1]        // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, vcc_lo                          // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_nc_u32 v6, 1, v6                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v7, 0                                    // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v7, s[sgprGSUSumIdx+1]        // overflow happened in remainder
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v6, v6, 1                             // quotient - 1
v_mul_u32_u24 v7, v6, s[sgprGSUSumIdx+1]           // re-calculate remainder
v_sub_nc_u32 v7, s[sgprLoopCounterL], v7           // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v6        // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v7         // remainder
s_mul_i32 s17, s[sgprLoopCounterL], s[sgprGSUSumIdx] // quotient*GSUSumIdx
s_add_u32 s16, 1, s[sgprLoopCounterL]              // quotient+1
s_add_u32 s17, s17, s[sgprGSUSumIdx+1]             // quotient*GSUSumIdx+remainder
s_mul_i32 s16, s16, s[sgprGSUSumIdx]               // (quotient+1)*GSUSumIdx
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cselect_b32 s16, s16, s17                        // (quotient+1)*GSUSumIdx if needed
s_mul_hi_u32 s17, s16, 8                           // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
s_mul_i32 s16, s16, 8                              // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
label_GSUC_B_End:
s_add_u32 s18, s18, s16                            // accum GsuOffset term to tilestart
s_addc_u32 s19, s19, s17                           // accum GsuOffset term to tilestart
s_mov_b64 s[sgprShadowLimitB+0:sgprShadowLimitB+0+1], 1 // Init tensor size
s_sub_u32 s16, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s17, constStrideBL, s16               // stride x (size-1)
s_mul_i32 s16, constStrideBL, s16                  // stride x (size-1)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s16 // sum tensor size
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s17 // sum tensor size
s_sub_u32 s16, s[sgprSizeJ], 1                     // (size-1)
s_mul_hi_u32 s17, s[sgprStrideB1J], s16            // stride x (size-1)
s_mul_i32 s16, s[sgprStrideB1J], s16               // stride x (size-1)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s16 // sum tensor size
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s17 // sum tensor size
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s18 // sub tileStart
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s19 // sub tileStart
s_lshl_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], 2 // Set limit to use bytes (multiple bpe)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 8 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_and_b32 s20, s[sgprArgType], 0xff                // mask ArgType domain (bit 8 = TDM wave-parity)
s_cmp_eq_u32 s20, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_StridedBatchedGemmLoadB
s_mul_i32 s16, 8, s[sgprWorkGroup2]                // Compute Offset into Pointer Array
s_cmp_eq_u32 s[sgprSizesSum], 0x0                  // Don't dereference Pointer array if SizesSum == 0
s_cbranch_scc1 label_StridedBatchedGemmLoadB_End
s_add_u32 s16, s16, s[sgprAddressB+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s17, s[sgprAddressB+1], 0               // Offsetting to the location [Higher half of address]
s_load_b64 s[sgprSrdB:sgprSrdB+1], s[16:17], 0     // Load the Matrix Address in the Pointer Array
s_load_b64 s[16:17], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x70 // Load batchOffsetB from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s16        // Add batch offset to B address (low)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s17       // Add batch offset to B address (high)
s_sub_u32 s[sgprSrdB+0], s[sgprSrdB+0], 8          // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0         // pre-pad to make room for possible pointer shift
s_lshl_b64 s[18:19], s[18:19], 2                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdB+0], s18, s[sgprSrdB+0]        // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s19, s[sgprSrdB+1]       // SRD base = Address+ tileStart1
s_branch label_StridedBatchedGemmLoadB_End
label_StridedBatchedGemmLoadB:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s17, s[sgprStrideBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s16, s[sgprStrideBK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s18, s18, s16                            // accum wg term to tilestart
s_addc_u32 s19, s19, s17                           // accum wg term to tilestart
s_lshl_b64 s[18:19], s[18:19], 2                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdB+0], s[sgprAddressB+0], s18    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprAddressB+1], s19   // SRD base = Address+ tileStart1
label_StridedBatchedGemmLoadB_End:  /// End Computing the Batch Matrix's base address for Strided Batched
s_mov_b32 s[sgprSrdB+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: final offsets a */
/* ============================================================= */
v_mul_lo_u32 v6, s[sgprStrideA0I], v[0]            // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+0+0], vcc_lo, v[1], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+0+0], 0x2, v[vgprGlobalReadOffsetA+0+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetA+0], 2, v[vgprGlobalReadOffsetA+0] //  (multiple bpe)
s_mul_i32 s[sgprScalarGlobalReadOffsetA+0], s[sgprStrideA0I], 128 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetA+0], s[sgprScalarGlobalReadOffsetA+0], 2 //  (multiple bpe)
s_mul_i32 s[sgprScalarGlobalReadOffsetA+1], s[sgprStrideA0I], 256 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetA+1], s[sgprScalarGlobalReadOffsetA+1], 2 //  (multiple bpe)
s_mul_i32 s[sgprScalarGlobalReadOffsetA+2], s[sgprStrideA0I], 384 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetA+2], s[sgprScalarGlobalReadOffsetA+2], 2 //  (multiple bpe)
/* ============================================================= */

/* global read addresses: final offsets b */
/* ============================================================= */
v_mul_lo_u32 v6, s[sgprStrideB1J], v[2]            // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+0+0], vcc_lo, v[3], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+0+0], 0x2, v[vgprGlobalReadOffsetB+0+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+0], 2, v[vgprGlobalReadOffsetB+0] //  (multiple bpe)
/* ============================================================= */

/* global read addresses: increments a */
s_and_b32 s17, s[sgprGSU], 0x3fff                  // Restore GSU
s_mov_b32 s[sgprGlobalReadIncsA+0], 32             // GSU*DepthU*Bpe*MI_dim(1)
s_mul_i32 s17, s17, s[sgprGlobalReadIncsA+0]       // GSU*DepthU*Bpe*MI_dim(1)
s_and_b32 s16, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cselect_b32 s[sgprGlobalReadIncsA+0], s[sgprGlobalReadIncsA+0], s17 // incrA (unrollIdx)

/* global read addresses: increments b */
s_and_b32 s17, s[sgprGSU], 0x3fff                  // Restore GSU
s_mov_b32 s[sgprGlobalReadIncsB+0], 32             // GSU*DepthU*Bpe*MI_dim(1)
s_mul_i32 s17, s17, s[sgprGlobalReadIncsB+0]       // GSU*DepthU*Bpe*MI_dim(1)
s_and_b32 s16, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cselect_b32 s[sgprGlobalReadIncsB+0], s[sgprGlobalReadIncsB+0], s17 // incrB (unrollIdx)
/* declare loop num iterations */
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 3 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 8
s_and_b32 s16, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s16, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU_1                         // branch if GSU == 1
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0x3fff   // Restore GSU
v_cvt_f32_u32 v0, s[sgprGSUSumIdx+1]               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v0, v0                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v1, s[sgprLoopCounterL]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v0, v0, v1                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v0, v0                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v1, v0, s[sgprGSUSumIdx+1]           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_nc_u32 v1, s[sgprLoopCounterL], v1           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmp_eq_u32 vcc_lo, v1, s[sgprGSUSumIdx+1]        // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, vcc_lo                          // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_nc_u32 v0, 1, v0                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v1, 0                                    // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v1, s[sgprGSUSumIdx+1]        // overflow happened in remainder
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v0, v0, 1                             // quotient - 1
v_mul_u32_u24 v1, v0, s[sgprGSUSumIdx+1]           // re-calculate remainder
v_sub_nc_u32 v1, s[sgprLoopCounterL], v1           // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v0        // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v1         // remainder
s_add_u32 s16, 1, s[sgprLoopCounterL]              // tmp<-numIterMyWg+1
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cmov_b32 s[sgprLoopCounterL], s16                // numIterMyWg++ if needed
label_GSU_1:
s_mov_b32 s[sgprOrigLoopCounter], s[sgprLoopCounterL] // copy loop counter
s_and_b32 s18, s[sgprStaggerU], 0x1f00
s_lshr_b32 s18, s18, 0x8
s_and_b32 s19, s[sgprStaggerU], 0xe000
s_and_b32 s[sgprStaggerU], s[sgprStaggerU], 0xff
s_mov_b32 s16, s[sgprStaggerU]                     // init staggerU
label_beginStaggerUIter:
s_lshl_b32 s17, s16, s18                           // shift by StaggerUStride
s_cmp_ge_u32 s[sgprOrigLoopCounter], s17           // loopCount >= current shift Count
s_cbranch_scc1 label_endStaggerUIter               // jump to end
s_lshr_b32 s16, s16, 1                             // step down to smaller stagger
s_branch label_beginStaggerUIter                   // jump to begin
label_endStaggerUIter:
s_sub_u32 s17, s16, 1                              // staggerU mask
s_cmp_ge_u32 s16, 1                                // if current staggerU >= 1
s_cselect_b32 s[sgprStaggerUIter], s17, 0          // set Mask
s_cmp_eq_u32 s19, 0x0
s_cbranch_scc0 label_StaggerUMapping
s_mov_b32 s16, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping:
s_cmp_eq_u32 s19, 0x2000
s_cbranch_scc0 label_StaggerUMapping_1
s_mov_b32 s16, s[sgprWorkGroup1]
s_branch label_staggerInputEnd
label_StaggerUMapping_1:
s_cmp_eq_u32 s19, 0x4000
s_cbranch_scc0 label_StaggerUMapping_2
s_mov_b32 s16, -0x1
s_branch label_staggerInputEnd
label_StaggerUMapping_2:
s_cmp_eq_u32 s19, 0x6000
s_cbranch_scc0 label_StaggerUMapping_3
s_mul_i32 s17, s[sgprNumWorkGroups0], s[sgprWorkGroup1]
s_add_u32 s16, s16, s17
s_add_u32 s16, s16, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping_3:
s_cmp_eq_u32 s19, 0x8000
s_cbranch_scc0 label_staggerInputEnd
s_mov_b32 s16, -0x1
s_branch label_staggerInputEnd
label_staggerInputEnd:
s_and_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s16 // Compute actual stagger start for this tile
s_lshl_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s18 // shift by StaggerUStride

/* addr += (StaggerUIter) * GlobalReadIncsA+0 */
s_mul_hi_i32 s17, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_i32 s16, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_hi_i32 s[sgprWrapUA+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUA+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0], s[sgprWrapUA+0] // remove one iteration
s_subb_u32 s[sgprWrapUA+1], 0, s[sgprWrapUA+1]     // remove one iteration
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s16        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s17       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s16 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s17 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* addr += (StaggerUIter) * GlobalReadIncsB+0 */
s_mul_hi_i32 s17, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_i32 s16, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_hi_i32 s[sgprWrapUB+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUB+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0], s[sgprWrapUB+0] // remove one iteration
s_subb_u32 s[sgprWrapUB+1], 0, s[sgprWrapUB+1]     // remove one iteration
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s16        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s17       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s16 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s17 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_add_u32 s[sgprStaggerUIter], s[sgprStaggerUIter], 2 // Subtract (PGR-1); StaggerUIter now contains target iteration to wrap
/* local read addresses: init pointers a */

/* localReadInitPointers */
/* local read addresses: init pointers b */

/* localReadInitPointers */

/* prefetch: global -> local */
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?
s_cbranch_scc1 label_ShadowInitStart               // skip to ShadowInitStart iter b/c numIter==0
buffer_load_b64 v[vgprG2LA+0:vgprG2LA+0+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b64 v[vgprG2LA+2:vgprG2LA+2+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b64 v[vgprG2LA+4:vgprG2LA+4+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b64 v[vgprG2LA+6:vgprG2LA+6+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:0 // G -> Reg 0_0_3_0
// SCALAR_GLOBAL_B: buffer_load_b64 v[vgprG2LB+0:vgprG2LB+0+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0

/* global read inc A loopL */
s_add_u32 s18, s[sgprLoopCounterL], 1              // remove pf(1)
s_mov_b32 s16, s[sgprGlobalReadIncsA+0]                // plain increment (StaggerU unsupported)
s_mov_b32 s17, 0
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s16        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s17       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s16 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s17 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s18, s[sgprLoopCounterL], 1              // remove pf(1)
s_mov_b32 s16, s[sgprGlobalReadIncsB+0]                // plain increment (StaggerU unsupported)
s_mov_b32 s17, 0
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s16        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s17       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s16 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s17 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32

/******************************************/
/* End setupNewTile                       */
/******************************************/
label_ShadowInitStart:
s_and_b32 s69, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s69, 1                                // GSU == 1 ?
s_cbranch_scc1 label_ArgTypeCheckD                 // Handling General Batched GEMM SRD initialization
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], s[sgprAddressD+0:sgprAddressD+0+1] // init SRD base address
s_branch label_GeneralBatchedGemmSrdInitiationD_End // End of handling General Batched GEMM SRD initialization
label_ArgTypeCheckD:  /// Check if ArgType is for General Batched GEMM for D
s_and_b32 s69, s[sgprArgType], 0xff                // mask ArgType domain (bit 8 = TDM wave-parity)
s_cmp_eq_u32 s69, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_RegularSrdInitializationD
s_branch label_GeneralBatchedGemmSrdInitiationD    // General Batched GEMM, Srd initialized to 0
label_RegularSrdInitializationD:  /// Regular SRD initialization for non-General Batched GEMM for D
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], s[sgprAddressD+0:sgprAddressD+0+1] // init SRD base address
s_branch label_GeneralBatchedGemmSrdInitiationD_End
label_GeneralBatchedGemmSrdInitiationD:  /// Handling General Batched GEMM SRD initialization
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], 0            // init SRD to 0
label_GeneralBatchedGemmSrdInitiationD_End:  /// End of handling General Batched GEMM SRD initialization
s_mov_b32 s[sgprSrdD+2], BufferOOB
s_mov_b32 s[sgprSrdD+3], Srd127_96                 // Set bits 127_96 in post-loop SRD

s_and_b32 s69, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s69, 1                                // GSU == 1 ?
s_cbranch_scc1 label_ArgTypeCheckC                 // Handling General Batched GEMM SRD initialization
s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], s[sgprAddressC+0:sgprAddressC+0+1] // init SRD base address
s_branch label_GeneralBatchedGemmSrdInitiationC_End // End of handling General Batched GEMM SRD initialization
label_ArgTypeCheckC:  /// Check if ArgType is for General Batched GEMM for C
s_and_b32 s69, s[sgprArgType], 0xff                // mask ArgType domain (bit 8 = TDM wave-parity)
s_cmp_eq_u32 s69, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_RegularSrdInitializationC
s_branch label_GeneralBatchedGemmSrdInitiationC    // General Batched GEMM, Srd initialized to 0
label_RegularSrdInitializationC:  /// Regular SRD initialization for non-General Batched GEMM for C
s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], s[sgprAddressC+0:sgprAddressC+0+1] // init SRD base address
s_branch label_GeneralBatchedGemmSrdInitiationC_End
label_GeneralBatchedGemmSrdInitiationC:  /// Handling General Batched GEMM SRD initialization
s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], 0            // init SRD to 0
label_GeneralBatchedGemmSrdInitiationC_End:  /// End of handling General Batched GEMM SRD initialization
s_mov_b32 s[sgprSrdC+2], BufferOOB
s_mov_b32 s[sgprSrdC+3], Srd127_96                 // Set bits 127_96 in post-loop SRD


s_mul_i32 s72, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_and_b32 s71, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_hi_u32 s71, s72, s[sgprStrideC1J]            // ScaleC s72 by Stride
s_mul_i32 s70, s72, s[sgprStrideC1J]               // ScaleC s72 by Stride
s_lshl_b64 s[70:71], s[70:71], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s70        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s71       // add hi to SRD
s_and_b32 s71, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_hi_u32 s71, s72, s[sgprStrideD1J]            // ScaleD s72 by Stride
s_mul_i32 s70, s72, s[sgprStrideD1J]               // ScaleD s72 by Stride
s_lshl_b64 s[70:71], s[70:71], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s70        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s71       // add hi to SRD

s_and_b32 s71, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s71, 1                                // GSU == 1 ?
s_cbranch_scc0 label_StridedBatchedGemmLoadC
s_and_b32 s69, s[sgprArgType], 0xff                // mask ArgType domain (bit 8 = TDM wave-parity)
s_cmp_eq_u32 s69, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_GeneralBatchedGemmLoadC
label_StridedBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s71, s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride
s_mul_i32 s70, s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride
s_lshl_b64 s[70:71], s[70:71], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s70        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s71       // add hi to SRD
s_branch label_GeneralBatchedGemmLoadC_End
label_GeneralBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for General Batched GEMM
s_mul_i32 s70, 8, s[sgprWorkGroup2]                // Compute stride in bytes into Pointer Array
s_add_u32 s70, s70, s[sgprAddressC+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s71, s[sgprAddressC+1], 0               // Offsetting to the location [Higher half of address]
s_load_b64 s[70:71], s[70:71], 0                   // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for the Matrix Address Load from the Pointer Array
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s70        // Offsetting within the Batch Matrix [Lower half of address]
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s71       // Offsetting within the Batch Matrix [Higher half of address]
s_load_b64 s[70:71], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x60 // Load batchOffsetC from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s70        // Add matrix address to SRD (low)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s71       // Add matrix address to SRD (high)
label_GeneralBatchedGemmLoadC_End:  /// End of label GeneralBatchedGemmLoadC
s_and_b32 s71, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s71, 1                                // GSU == 1 ?
s_cbranch_scc0 label_StridedBatchedGemmLoadD
s_and_b32 s69, s[sgprArgType], 0xff                // mask ArgType domain (bit 8 = TDM wave-parity)
s_cmp_eq_u32 s69, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_GeneralBatchedGemmLoadD
label_StridedBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s71, s[sgprWorkGroup2], s[sgprStrideDK] // ScaleD s[sgprWorkGroup2] by Stride
s_mul_i32 s70, s[sgprWorkGroup2], s[sgprStrideDK]  // ScaleD s[sgprWorkGroup2] by Stride
s_lshl_b64 s[70:71], s[70:71], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s70        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s71       // add hi to SRD
s_branch label_GeneralBatchedGemmLoadD_End
label_GeneralBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for General Batched GEMM
s_mul_i32 s70, 8, s[sgprWorkGroup2]                // Compute stride in bytes into Pointer Array
s_add_u32 s70, s70, s[sgprAddressD+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s71, s[sgprAddressD+1], 0               // Offsetting to the location [Higher half of address]
s_load_b64 s[70:71], s[70:71], 0                   // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for the Matrix Address Load from the Pointer Array
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s70        // Offsetting within the Batch Matrix [Lower half of address]
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s71       // Offsetting within the Batch Matrix [Higher half of address]
s_load_b64 s[70:71], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x58 // Load batchOffsetD from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s70        // Add matrix address to SRD (low)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s71       // Add matrix address to SRD (high)
label_GeneralBatchedGemmLoadD_End:  /// End of label GeneralBatchedGemmLoadD

s_and_b32 s69, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s69, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU_2                         // branch if GSU == 1
// GSU Output Buffer offset: Free0 + (Free1-1)*StrideC1J + (Free2-1)*StrideCK * GSUIdx * bpe%s
s_mul_hi_u32 s71, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_mul_i32 s70, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_sub_u32 s69, s[sgprSizesFree+1], 1               // Free1
s_mul_i32 s69, s69, s[sgprGSUSumIdx]               // Free1
s_mul_hi_u32 s72, s69, s[sgprStrideC1J]            // Free1
s_mul_i32 s69, s69, s[sgprStrideC1J]               // Free1
s_add_u32 s70, s70, s69                            // Free1
s_addc_u32 s71, s71, s72                           // Free1
s_sub_u32 s69, s[sgprSizesFree+2], 1               // Free2
s_mul_i32 s69, s69, s[sgprGSUSumIdx]               // Free2
s_mul_hi_u32 s72, s69, s[sgprStrideCK]             // Free2
s_mul_i32 s69, s69, s[sgprStrideCK]                // Free2
s_add_u32 s70, s70, s69                            // Free2
s_addc_u32 s71, s71, s72                           // Free2
s_lshl_b64 s[70:71], s[70:71], 2                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s70        // add lo GSU offset to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s71       // add hi GSU offset to SRD
label_GSU_2:
.set sgprGSULog2BpeC, UNDEF
.set sgprAddressC, UNDEF

/* initC: remove ValuC vgpr buffer [0...128) from pool */

/* initC: remove acc vgpr buffer [0...0) from pool */

/* initC: remove ValuA/B vgpr buffer [136...185) from pool */
v_mov_b32 v[vgprValuC+0], 0                        // initC
v_mov_b32 v[vgprValuC+1], 0                        // initC
v_mov_b32 v[vgprValuC+2], 0                        // initC
v_mov_b32 v[vgprValuC+3], 0                        // initC
v_mov_b32 v[vgprValuC+4], 0                        // initC
v_mov_b32 v[vgprValuC+5], 0                        // initC
v_mov_b32 v[vgprValuC+6], 0                        // initC
v_mov_b32 v[vgprValuC+7], 0                        // initC
v_mov_b32 v[vgprValuC+8], 0                        // initC
v_mov_b32 v[vgprValuC+9], 0                        // initC
v_mov_b32 v[vgprValuC+10], 0                       // initC
v_mov_b32 v[vgprValuC+11], 0                       // initC
v_mov_b32 v[vgprValuC+12], 0                       // initC
v_mov_b32 v[vgprValuC+13], 0                       // initC
v_mov_b32 v[vgprValuC+14], 0                       // initC
v_mov_b32 v[vgprValuC+15], 0                       // initC
v_mov_b32 v[vgprValuC+16], 0                       // initC
v_mov_b32 v[vgprValuC+17], 0                       // initC
v_mov_b32 v[vgprValuC+18], 0                       // initC
v_mov_b32 v[vgprValuC+19], 0                       // initC
v_mov_b32 v[vgprValuC+20], 0                       // initC
v_mov_b32 v[vgprValuC+21], 0                       // initC
v_mov_b32 v[vgprValuC+22], 0                       // initC
v_mov_b32 v[vgprValuC+23], 0                       // initC
v_mov_b32 v[vgprValuC+24], 0                       // initC
v_mov_b32 v[vgprValuC+25], 0                       // initC
v_mov_b32 v[vgprValuC+26], 0                       // initC
v_mov_b32 v[vgprValuC+27], 0                       // initC
v_mov_b32 v[vgprValuC+28], 0                       // initC
v_mov_b32 v[vgprValuC+29], 0                       // initC
v_mov_b32 v[vgprValuC+30], 0                       // initC
v_mov_b32 v[vgprValuC+31], 0                       // initC
v_mov_b32 v[vgprValuC+32], 0                       // initC
v_mov_b32 v[vgprValuC+33], 0                       // initC
v_mov_b32 v[vgprValuC+34], 0                       // initC
v_mov_b32 v[vgprValuC+35], 0                       // initC
v_mov_b32 v[vgprValuC+36], 0                       // initC
v_mov_b32 v[vgprValuC+37], 0                       // initC
v_mov_b32 v[vgprValuC+38], 0                       // initC
v_mov_b32 v[vgprValuC+39], 0                       // initC
v_mov_b32 v[vgprValuC+40], 0                       // initC
v_mov_b32 v[vgprValuC+41], 0                       // initC
v_mov_b32 v[vgprValuC+42], 0                       // initC
v_mov_b32 v[vgprValuC+43], 0                       // initC
v_mov_b32 v[vgprValuC+44], 0                       // initC
v_mov_b32 v[vgprValuC+45], 0                       // initC
v_mov_b32 v[vgprValuC+46], 0                       // initC
v_mov_b32 v[vgprValuC+47], 0                       // initC
v_mov_b32 v[vgprValuC+48], 0                       // initC
v_mov_b32 v[vgprValuC+49], 0                       // initC
v_mov_b32 v[vgprValuC+50], 0                       // initC
v_mov_b32 v[vgprValuC+51], 0                       // initC
v_mov_b32 v[vgprValuC+52], 0                       // initC
v_mov_b32 v[vgprValuC+53], 0                       // initC
v_mov_b32 v[vgprValuC+54], 0                       // initC
v_mov_b32 v[vgprValuC+55], 0                       // initC
v_mov_b32 v[vgprValuC+56], 0                       // initC
v_mov_b32 v[vgprValuC+57], 0                       // initC
v_mov_b32 v[vgprValuC+58], 0                       // initC
v_mov_b32 v[vgprValuC+59], 0                       // initC
v_mov_b32 v[vgprValuC+60], 0                       // initC
v_mov_b32 v[vgprValuC+61], 0                       // initC
v_mov_b32 v[vgprValuC+62], 0                       // initC
v_mov_b32 v[vgprValuC+63], 0                       // initC
v_mov_b32 v[vgprValuC+64], 0                       // initC
v_mov_b32 v[vgprValuC+65], 0                       // initC
v_mov_b32 v[vgprValuC+66], 0                       // initC
v_mov_b32 v[vgprValuC+67], 0                       // initC
v_mov_b32 v[vgprValuC+68], 0                       // initC
v_mov_b32 v[vgprValuC+69], 0                       // initC
v_mov_b32 v[vgprValuC+70], 0                       // initC
v_mov_b32 v[vgprValuC+71], 0                       // initC
v_mov_b32 v[vgprValuC+72], 0                       // initC
v_mov_b32 v[vgprValuC+73], 0                       // initC
v_mov_b32 v[vgprValuC+74], 0                       // initC
v_mov_b32 v[vgprValuC+75], 0                       // initC
v_mov_b32 v[vgprValuC+76], 0                       // initC
v_mov_b32 v[vgprValuC+77], 0                       // initC
v_mov_b32 v[vgprValuC+78], 0                       // initC
v_mov_b32 v[vgprValuC+79], 0                       // initC
v_mov_b32 v[vgprValuC+80], 0                       // initC
v_mov_b32 v[vgprValuC+81], 0                       // initC
v_mov_b32 v[vgprValuC+82], 0                       // initC
v_mov_b32 v[vgprValuC+83], 0                       // initC
v_mov_b32 v[vgprValuC+84], 0                       // initC
v_mov_b32 v[vgprValuC+85], 0                       // initC
v_mov_b32 v[vgprValuC+86], 0                       // initC
v_mov_b32 v[vgprValuC+87], 0                       // initC
v_mov_b32 v[vgprValuC+88], 0                       // initC
v_mov_b32 v[vgprValuC+89], 0                       // initC
v_mov_b32 v[vgprValuC+90], 0                       // initC
v_mov_b32 v[vgprValuC+91], 0                       // initC
v_mov_b32 v[vgprValuC+92], 0                       // initC
v_mov_b32 v[vgprValuC+93], 0                       // initC
v_mov_b32 v[vgprValuC+94], 0                       // initC
v_mov_b32 v[vgprValuC+95], 0                       // initC
v_mov_b32 v[vgprValuC+96], 0                       // initC
v_mov_b32 v[vgprValuC+97], 0                       // initC
v_mov_b32 v[vgprValuC+98], 0                       // initC
v_mov_b32 v[vgprValuC+99], 0                       // initC
v_mov_b32 v[vgprValuC+100], 0                      // initC
v_mov_b32 v[vgprValuC+101], 0                      // initC
v_mov_b32 v[vgprValuC+102], 0                      // initC
v_mov_b32 v[vgprValuC+103], 0                      // initC
v_mov_b32 v[vgprValuC+104], 0                      // initC
v_mov_b32 v[vgprValuC+105], 0                      // initC
v_mov_b32 v[vgprValuC+106], 0                      // initC
v_mov_b32 v[vgprValuC+107], 0                      // initC
v_mov_b32 v[vgprValuC+108], 0                      // initC
v_mov_b32 v[vgprValuC+109], 0                      // initC
v_mov_b32 v[vgprValuC+110], 0                      // initC
v_mov_b32 v[vgprValuC+111], 0                      // initC
v_mov_b32 v[vgprValuC+112], 0                      // initC
v_mov_b32 v[vgprValuC+113], 0                      // initC
v_mov_b32 v[vgprValuC+114], 0                      // initC
v_mov_b32 v[vgprValuC+115], 0                      // initC
v_mov_b32 v[vgprValuC+116], 0                      // initC
v_mov_b32 v[vgprValuC+117], 0                      // initC
v_mov_b32 v[vgprValuC+118], 0                      // initC
v_mov_b32 v[vgprValuC+119], 0                      // initC
v_mov_b32 v[vgprValuC+120], 0                      // initC
v_mov_b32 v[vgprValuC+121], 0                      // initC
v_mov_b32 v[vgprValuC+122], 0                      // initC
v_mov_b32 v[vgprValuC+123], 0                      // initC
v_mov_b32 v[vgprValuC+124], 0                      // initC
v_mov_b32 v[vgprValuC+125], 0                      // initC
v_mov_b32 v[vgprValuC+126], 0                      // initC
v_mov_b32 v[vgprValuC+127], 0                      // initC
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?

/* after InitC, skip to end of prefetch last iter if numIter==0 */

/* label_PrefetchGlobalLastIterEnd */
s_cbranch_scc0 label_NoBranch_0                    // Only branch on scc1
s_getpc_b64 s[70:71]                               // addr of next instr
s_add_i32 s72, label_PrefetchGlobalLastIterEnd, 4  // target branch offset
s_add_u32 s70, s70, s72                            // add target branch offset
s_addc_u32 s71, s71, 0                             // add high and carry
s_setpc_b64 s[70:71]                               // branch to label_PrefetchGlobalLastIterEnd
label_NoBranch_0:
s_waitcnt vmcnt(0)                                 // wait for global read

/* local write a */
ds_store_b32 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0] offset:0 // lwoA_0_0_0_0 = (0 + 0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
ds_store_b32 v[vgprLocalWriteAddrA+0], v[vgprG2LA+1] offset:2064 // lwoA_0_1_0_0 = (1 + 0*LSCA)*(MT0I+PAD) + (0*LSPA) = 2064 sync LDS0
ds_store_b32 v[vgprLocalWriteAddrA+0], v[vgprG2LA+2] offset:512 // lwoA_0_0_1_0 = (0 + 0*LSCA)*(MT0I+PAD) + (1*LSPA) = 512 sync LDS0
ds_store_b32 v[vgprLocalWriteAddrA+0], v[vgprG2LA+3] offset:2576 // lwoA_0_1_1_0 = (1 + 0*LSCA)*(MT0I+PAD) + (1*LSPA) = 2576 sync LDS0
ds_store_b32 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4] offset:1024 // lwoA_0_0_2_0 = (0 + 0*LSCA)*(MT0I+PAD) + (2*LSPA) = 1024 sync LDS0
ds_store_b32 v[vgprLocalWriteAddrA+0], v[vgprG2LA+5] offset:3088 // lwoA_0_1_2_0 = (1 + 0*LSCA)*(MT0I+PAD) + (2*LSPA) = 3088 sync LDS0
ds_store_b32 v[vgprLocalWriteAddrA+0], v[vgprG2LA+6] offset:1536 // lwoA_0_0_3_0 = (0 + 0*LSCA)*(MT0I+PAD) + (3*LSPA) = 1536 sync LDS0
ds_store_b32 v[vgprLocalWriteAddrA+0], v[vgprG2LA+7] offset:3600 // lwoA_0_1_3_0 = (1 + 0*LSCA)*(MT0I+PAD) + (3*LSPA) = 3600 sync LDS0

/* local write b */
// SCALAR_GLOBAL_B: ds_store_b32 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0] offset:0 // lwoB_0_0_0_0 = (0 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0
// SCALAR_GLOBAL_B: ds_store_b32 v[vgprLocalWriteAddrB+0], v[vgprG2LB+1] offset:528 // lwoB_0_1_0_0 = (1 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 528 sync LDS0

/* local write swap a */

/* (EPS=1) local write swap internal offset -> 32768 */

/* local write swap b */

/* (EPS=1) local write swap internal offset -> 32768 */
s_waitcnt lgkmcnt(0)                               // 0prefetch wait for local write
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier                                          // LW to PLR, sync

/* local read prefetch a */
ds_load_b64 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+1], v[vgprLocalReadAddrA+0] offset:0 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+2:vgprValuA_X0_I0+2+1], v[vgprLocalReadAddrA+0] offset:256 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+1], v[vgprLocalReadAddrA+0] offset:512 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+6:vgprValuA_X0_I0+6+1], v[vgprLocalReadAddrA+0] offset:768 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+1], v[vgprLocalReadAddrA+0] offset:1024 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+10:vgprValuA_X0_I0+10+1], v[vgprLocalReadAddrA+0] offset:1280 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+1], v[vgprLocalReadAddrA+0] offset:1536 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+14:vgprValuA_X0_I0+14+1], v[vgprLocalReadAddrA+0] offset:1792 // sync LDS0

/* local read prefetch b */
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB+0] offset:0 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB+0] offset:128 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrB+0] offset:256 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrB+0] offset:384 // sync LDS0

/* local read inc a */
/* N/A, lro->516 */
/* localReadDoCntA 1 localReadDoCntMXSA 0 localReadDoCntB 1 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read inc b */
/* N/A, lro->132 */
/* localReadDoCntA 1 localReadDoCntMXSA 0 localReadDoCntB 1 localReadDoCntMXSB 0 localReadDoCntM 0 */

/******************************************/
/* Unrolled Loop(s) - Begin               */
/******************************************/
v_lshrrev_b32 v134, 5, v[vgprSerial]
v_readfirstlane_b32 s72, v134
s_lshl_b32 s72, s72, 3
s_lshl_b32 s73, s[sgprWorkGroup1], 7
s_add_u32 s72, s72, s73
s_lshl_b32 s[sgprStrideB1J], s[sgprStrideB1J], 2
s_mul_hi_u32 s73, s72, s[sgprStrideB1J]
s_mul_i32 s72, s72, s[sgprStrideB1J]
s_add_u32 s72, s72, 8
s_addc_u32 s73, s73, 0
s_add_u32 s[sgprAddressB], s[sgprAddressB], s72
s_addc_u32 s[sgprAddressB+1], s[sgprAddressB+1], s73
s_mov_b32 s[sgprStrideBK], s[sgprSizeL]
s_mov_b64 s[30:31], s[2:3]
s_mov_b64 s[32:33], s[40:41]
s_mov_b32 s69, s42
s_mov_b32 s8, s47
s_mov_b64 s[10:11], s[60:61]
s_mov_b64 s[14:15], s[56:57]
.set sgprWorkGroup0, 30
.set sgprWorkGroup1, 31
.set sgprStrideA0I, 32
.set sgprStrideAK, 33
.set sgprStrideB1J, 69
.set sgprStaggerUIter, 8
.set sgprWrapUA, 10
.set sgprShadowLimitA, 14
s_mul_i32 s37, s[sgprStrideB1J], 2
s_mul_i32 s38, s[sgprStrideB1J], 3
s_mul_i32 s39, s[sgprStrideB1J], 4
s_mul_i32 s52, s[sgprStrideB1J], 5
s_mul_i32 s53, s[sgprStrideB1J], 6
s_mul_i32 s54, s[sgprStrideB1J], 7
v_and_b32 v189, 7, v[vgprSerial]
v_mul_lo_u32 v189, v189, s[sgprStrideB1J]
s_load_b128 s[0:3], s[sgprAddressB:sgprAddressB+1], 0
s_load_b128 s[4:7], s[sgprAddressB:sgprAddressB+1], s69
s_load_b128 s[16:19], s[sgprAddressB:sgprAddressB+1], s37
s_load_b128 s[20:23], s[sgprAddressB:sgprAddressB+1], s38
s_load_b128 s[40:43], s[sgprAddressB:sgprAddressB+1], s39
s_load_b128 s[44:47], s[sgprAddressB:sgprAddressB+1], s52
s_load_b128 s[56:59], s[sgprAddressB:sgprAddressB+1], s53
s_load_b128 s[60:63], s[sgprAddressB:sgprAddressB+1], s54
/* PGR2 prologue: block 1 -> G2LA2 */
buffer_load_b64 v[vgprG2LA2+0:vgprG2LA2+0+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b64 v[vgprG2LA2+2:vgprG2LA2+2+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b64 v[vgprG2LA2+4:vgprG2LA2+4+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b64 v[vgprG2LA2+6:vgprG2LA2+6+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:0 // G -> Reg 0_0_3_0
/* global read inc A loopL */
s_mov_b32 s70, s[sgprGlobalReadIncsA+0]                // plain increment (StaggerU unsupported)
s_mov_b32 s71, 0
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s70        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s71       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s70 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s71 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
label_openLoopL:
s_cmp_le_u32 s[sgprLoopCounterL], 0x1              // LoopCounterL < EndCounter
s_cbranch_scc1 label_LoopEndL                      // do not enter LoopL
.align 16
label_LoopBeginL:

/******************************************/
/* Unrolled Loop 1/2 - Begin              */
/******************************************/

/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */
buffer_load_b64 v[vgprG2LA+0:vgprG2LA+0+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b64 v[vgprG2LA+2:vgprG2LA+2+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b64 v[vgprG2LA+4:vgprG2LA+4+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b64 v[vgprG2LA+6:vgprG2LA+6+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:0 // G -> Reg 0_0_3_0
s_cmp_gt_u32 s[sgprLoopCounterL], 4 // PF B in range?
s_cbranch_scc1 label_pfb_far1
global_load_b32 v188, v189, s[sgprAddressB:sgprAddressB+1] offset:0 // PF B (tail: current block)
s_branch label_pfb_done1
label_pfb_far1:
global_load_b32 v188, v189, s[sgprAddressB:sgprAddressB+1] offset:64 // PF B
label_pfb_done1:
// SCALAR_GLOBAL_B: buffer_load_b64 v[vgprG2LB+0:vgprG2LB+0+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0

/* global read inc A loopL */
s_mov_b32 s70, s[sgprGlobalReadIncsA+0]                // plain increment (StaggerU unsupported)
s_mov_b32 s71, 0
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s70        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s71       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s70 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s71 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32


/* iter 0 */

/* local read a */
s_waitcnt lgkmcnt(0)
s_waitcnt_depctr 0
ds_load_b64 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+1], v[vgprLocalReadAddrA+0] offset:2064 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+2:vgprValuA_X1_I0+2+1], v[vgprLocalReadAddrA+0] offset:2320 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+1], v[vgprLocalReadAddrA+0] offset:2576 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+6:vgprValuA_X1_I0+6+1], v[vgprLocalReadAddrA+0] offset:2832 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+1], v[vgprLocalReadAddrA+0] offset:3088 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+10:vgprValuA_X1_I0+10+1], v[vgprLocalReadAddrA+0] offset:3344 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+1], v[vgprLocalReadAddrA+0] offset:3600 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+14:vgprValuA_X1_I0+14+1], v[vgprLocalReadAddrA+0] offset:3856 // sync LDS0

/* local read b */
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB+0] offset:528 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB+0] offset:656 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrB+0] offset:784 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrB+0] offset:912 // sync LDS0

/* local read increment a */
/* N/A, lro->1032 */
/* localReadDoCntA 2 localReadDoCntMXSA 0 localReadDoCntB 2 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->264 */
/* localReadDoCntA 2 localReadDoCntMXSA 0 localReadDoCntB 2 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=0 newLR=0 for iteration == 0
s_add_u32 s[sgprAddressB], s[sgprAddressB], 16
s_addc_u32 s[sgprAddressB+1], s[sgprAddressB+1], 0
s_load_b128 s[72:75], s[sgprAddressB:sgprAddressB+1], 0
s_load_b128 s[76:79], s[sgprAddressB:sgprAddressB+1], s69
s_load_b128 s[80:83], s[sgprAddressB:sgprAddressB+1], s37
s_load_b128 s[84:87], s[sgprAddressB:sgprAddressB+1], s38
s_load_b128 s[88:91], s[sgprAddressB:sgprAddressB+1], s39
s_load_b128 s[92:95], s[sgprAddressB:sgprAddressB+1], s52
s_load_b128 s[96:99], s[sgprAddressB:sgprAddressB+1], s53
s_load_b128 s[100:103], s[sgprAddressB:sgprAddressB+1], s54
v_dual_fmac_f32 v[vgprValuC+0], s0, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s0, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s0, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s0, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s4, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s4, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s4, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s4, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s16, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s16, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s16, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s16, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s20, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s20, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s20, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s20, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s40, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s40, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s40, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s40, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s44, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s44, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s44, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s44, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s56, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s56, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s56, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s56, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s60, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s60, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s60, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s60, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s0, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s0, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s0, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s0, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s4, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s4, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s4, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s4, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s16, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s16, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s16, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s16, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s20, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s20, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s20, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s20, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s40, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s40, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s40, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s40, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s44, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s44, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s44, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s44, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s56, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s56, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s56, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s56, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s60, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s60, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s60, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s60, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s0, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s0, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s0, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s0, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s4, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s4, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s4, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s4, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s16, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s16, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s16, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s16, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s20, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s20, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s20, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s20, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s40, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s40, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s40, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s40, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s44, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s44, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s44, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s44, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s56, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s56, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s56, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s56, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s60, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s60, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s60, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s60, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s0, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s0, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s0, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s0, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s4, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s4, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s4, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s4, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s16, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s16, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s16, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s16, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s20, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s20, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s20, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s20, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s40, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s40, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s40, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s40, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s44, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s44, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s44, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s44, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s56, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s56, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s56, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s56, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s60, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s60, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s60, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s60, v[vgprValuA_X0_I0+15]


/* iter 1 */

/* local read a */
s_waitcnt lgkmcnt(0)
s_waitcnt_depctr 0
ds_load_b64 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+1], v[vgprLocalReadAddrA+0] offset:4128 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+2:vgprValuA_X0_I0+2+1], v[vgprLocalReadAddrA+0] offset:4384 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+1], v[vgprLocalReadAddrA+0] offset:4640 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+6:vgprValuA_X0_I0+6+1], v[vgprLocalReadAddrA+0] offset:4896 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+1], v[vgprLocalReadAddrA+0] offset:5152 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+10:vgprValuA_X0_I0+10+1], v[vgprLocalReadAddrA+0] offset:5408 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+1], v[vgprLocalReadAddrA+0] offset:5664 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+14:vgprValuA_X0_I0+14+1], v[vgprLocalReadAddrA+0] offset:5920 // sync LDS0

/* local read b */
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB+0] offset:1056 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB+0] offset:1184 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrB+0] offset:1312 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrB+0] offset:1440 // sync LDS0

/* local read increment a */
/* N/A, lro->1548 */
/* localReadDoCntA 3 localReadDoCntMXSA 0 localReadDoCntB 3 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->396 */
/* localReadDoCntA 3 localReadDoCntMXSA 0 localReadDoCntB 3 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=0 newLR=0
v_dual_fmac_f32 v[vgprValuC+0], s1, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s1, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s1, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s1, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s5, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s5, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s5, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s5, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s17, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s17, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s17, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s17, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s21, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s21, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s21, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s21, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s41, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s41, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s41, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s41, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s45, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s45, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s45, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s45, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s57, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s57, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s57, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s57, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s61, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s61, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s61, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s61, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s1, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s1, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s1, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s1, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s5, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s5, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s5, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s5, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s17, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s17, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s17, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s17, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s21, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s21, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s21, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s21, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s41, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s41, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s41, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s41, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s45, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s45, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s45, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s45, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s57, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s57, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s57, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s57, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s61, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s61, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s61, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s61, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s1, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s1, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s1, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s1, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s5, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s5, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s5, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s5, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s17, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s17, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s17, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s17, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s21, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s21, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s21, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s21, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s41, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s41, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s41, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s41, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s45, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s45, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s45, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s45, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s57, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s57, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s57, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s57, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s61, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s61, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s61, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s61, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s1, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s1, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s1, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s1, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s5, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s5, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s5, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s5, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s17, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s17, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s17, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s17, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s21, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s21, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s21, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s21, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s41, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s41, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s41, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s41, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s45, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s45, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s45, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s45, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s57, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s57, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s57, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s57, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s61, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s61, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s61, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s61, v[vgprValuA_X1_I0+15]


/* iter 2 */

/* local read a */
s_waitcnt lgkmcnt(0)
s_waitcnt_depctr 0
ds_load_b64 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+1], v[vgprLocalReadAddrA+0] offset:6192 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+2:vgprValuA_X1_I0+2+1], v[vgprLocalReadAddrA+0] offset:6448 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+1], v[vgprLocalReadAddrA+0] offset:6704 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+6:vgprValuA_X1_I0+6+1], v[vgprLocalReadAddrA+0] offset:6960 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+1], v[vgprLocalReadAddrA+0] offset:7216 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+10:vgprValuA_X1_I0+10+1], v[vgprLocalReadAddrA+0] offset:7472 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+1], v[vgprLocalReadAddrA+0] offset:7728 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+14:vgprValuA_X1_I0+14+1], v[vgprLocalReadAddrA+0] offset:7984 // sync LDS0

/* local read b */
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB+0] offset:1584 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB+0] offset:1712 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrB+0] offset:1840 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrB+0] offset:1968 // sync LDS0

/* local read increment a */
/* N/A, lro->2064 */
/* localReadDoCntA 4 localReadDoCntMXSA 0 localReadDoCntB 4 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->528 */
/* localReadDoCntA 4 localReadDoCntMXSA 0 localReadDoCntB 4 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=0 newLR=0
v_dual_fmac_f32 v[vgprValuC+0], s2, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s2, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s2, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s2, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s6, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s6, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s6, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s6, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s18, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s18, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s18, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s18, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s22, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s22, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s22, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s22, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s42, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s42, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s42, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s42, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s46, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s46, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s46, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s46, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s58, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s58, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s58, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s58, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s62, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s62, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s62, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s62, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s2, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s2, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s2, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s2, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s6, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s6, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s6, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s6, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s18, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s18, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s18, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s18, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s22, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s22, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s22, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s22, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s42, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s42, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s42, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s42, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s46, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s46, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s46, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s46, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s58, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s58, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s58, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s58, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s62, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s62, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s62, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s62, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s2, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s2, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s2, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s2, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s6, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s6, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s6, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s6, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s18, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s18, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s18, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s18, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s22, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s22, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s22, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s22, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s42, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s42, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s42, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s42, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s46, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s46, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s46, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s46, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s58, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s58, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s58, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s58, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s62, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s62, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s62, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s62, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s2, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s2, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s2, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s2, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s6, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s6, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s6, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s6, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s18, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s18, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s18, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s18, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s22, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s22, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s22, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s22, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s42, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s42, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s42, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s42, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s46, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s46, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s46, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s46, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s58, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s58, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s58, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s58, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s62, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s62, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s62, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s62, v[vgprValuA_X0_I0+15]


/* iter 3 */

/* local read a */
s_waitcnt lgkmcnt(0)
s_waitcnt_depctr 0
ds_load_b64 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+1], v[vgprLocalReadAddrA+0] offset:8256 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+2:vgprValuA_X0_I0+2+1], v[vgprLocalReadAddrA+0] offset:8512 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+1], v[vgprLocalReadAddrA+0] offset:8768 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+6:vgprValuA_X0_I0+6+1], v[vgprLocalReadAddrA+0] offset:9024 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+1], v[vgprLocalReadAddrA+0] offset:9280 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+10:vgprValuA_X0_I0+10+1], v[vgprLocalReadAddrA+0] offset:9536 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+1], v[vgprLocalReadAddrA+0] offset:9792 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+14:vgprValuA_X0_I0+14+1], v[vgprLocalReadAddrA+0] offset:10048 // sync LDS0

/* local read b */
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB+0] offset:2112 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB+0] offset:2240 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrB+0] offset:2368 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrB+0] offset:2496 // sync LDS0

/* local read increment a */
/* N/A, lro->2580 */
/* localReadDoCntA 5 localReadDoCntMXSA 0 localReadDoCntB 5 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->660 */
/* localReadDoCntA 5 localReadDoCntMXSA 0 localReadDoCntB 5 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=0 newLR=0
v_dual_fmac_f32 v[vgprValuC+0], s3, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s3, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s3, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s3, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s7, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s7, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s7, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s7, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s19, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s19, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s19, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s19, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s23, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s23, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s23, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s23, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s43, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s43, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s43, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s43, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s47, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s47, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s47, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s47, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s59, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s59, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s59, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s59, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s63, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s63, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s63, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s63, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s3, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s3, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s3, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s3, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s7, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s7, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s7, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s7, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s19, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s19, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s19, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s19, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s23, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s23, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s23, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s23, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s43, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s43, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s43, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s43, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s47, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s47, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s47, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s47, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s59, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s59, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s59, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s59, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s63, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s63, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s63, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s63, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s3, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s3, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s3, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s3, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s7, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s7, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s7, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s7, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s19, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s19, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s19, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s19, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s23, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s23, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s23, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s23, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s43, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s43, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s43, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s43, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s47, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s47, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s47, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s47, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s59, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s59, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s59, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s59, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s63, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s63, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s63, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s63, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s3, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s3, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s3, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s3, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s7, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s7, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s7, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s7, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s19, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s19, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s19, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s19, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s23, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s23, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s23, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s23, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s43, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s43, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s43, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s43, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s47, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s47, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s47, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s47, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s59, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s59, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s59, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s59, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s63, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s63, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s63, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s63, v[vgprValuA_X1_I0+15]


/* iter 4 */

/* local read a */
s_waitcnt lgkmcnt(0)
s_waitcnt_depctr 0
ds_load_b64 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+1], v[vgprLocalReadAddrA+0] offset:10320 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+2:vgprValuA_X1_I0+2+1], v[vgprLocalReadAddrA+0] offset:10576 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+1], v[vgprLocalReadAddrA+0] offset:10832 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+6:vgprValuA_X1_I0+6+1], v[vgprLocalReadAddrA+0] offset:11088 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+1], v[vgprLocalReadAddrA+0] offset:11344 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+10:vgprValuA_X1_I0+10+1], v[vgprLocalReadAddrA+0] offset:11600 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+1], v[vgprLocalReadAddrA+0] offset:11856 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+14:vgprValuA_X1_I0+14+1], v[vgprLocalReadAddrA+0] offset:12112 // sync LDS0

/* local read b */
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB+0] offset:2640 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB+0] offset:2768 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrB+0] offset:2896 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrB+0] offset:3024 // sync LDS0

/* local read increment a */
/* N/A, lro->3096 */
/* localReadDoCntA 6 localReadDoCntMXSA 0 localReadDoCntB 6 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->792 */
/* localReadDoCntA 6 localReadDoCntMXSA 0 localReadDoCntB 6 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=0 newLR=0
s_add_u32 s[sgprAddressB], s[sgprAddressB], 16
s_addc_u32 s[sgprAddressB+1], s[sgprAddressB+1], 0
s_load_b128 s[0:3], s[sgprAddressB:sgprAddressB+1], 0
s_load_b128 s[4:7], s[sgprAddressB:sgprAddressB+1], s69
s_load_b128 s[16:19], s[sgprAddressB:sgprAddressB+1], s37
s_load_b128 s[20:23], s[sgprAddressB:sgprAddressB+1], s38
s_load_b128 s[40:43], s[sgprAddressB:sgprAddressB+1], s39
s_load_b128 s[44:47], s[sgprAddressB:sgprAddressB+1], s52
s_load_b128 s[56:59], s[sgprAddressB:sgprAddressB+1], s53
s_load_b128 s[60:63], s[sgprAddressB:sgprAddressB+1], s54
v_dual_fmac_f32 v[vgprValuC+0], s72, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s72, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s72, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s72, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s76, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s76, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s76, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s76, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s80, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s80, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s80, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s80, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s84, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s84, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s84, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s84, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s88, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s88, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s88, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s88, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s92, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s92, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s92, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s92, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s96, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s96, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s96, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s96, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s100, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s100, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s100, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s100, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s72, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s72, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s72, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s72, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s76, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s76, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s76, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s76, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s80, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s80, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s80, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s80, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s84, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s84, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s84, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s84, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s88, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s88, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s88, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s88, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s92, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s92, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s92, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s92, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s96, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s96, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s96, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s96, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s100, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s100, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s100, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s100, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s72, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s72, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s72, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s72, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s76, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s76, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s76, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s76, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s80, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s80, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s80, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s80, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s84, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s84, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s84, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s84, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s88, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s88, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s88, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s88, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s92, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s92, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s92, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s92, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s96, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s96, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s96, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s96, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s100, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s100, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s100, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s100, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s72, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s72, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s72, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s72, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s76, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s76, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s76, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s76, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s80, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s80, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s80, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s80, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s84, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s84, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s84, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s84, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s88, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s88, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s88, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s88, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s92, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s92, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s92, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s92, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s96, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s96, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s96, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s96, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s100, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s100, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s100, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s100, v[vgprValuA_X0_I0+15]


/* iter 5 */

/* local read a */
s_waitcnt lgkmcnt(0)
s_waitcnt_depctr 0
ds_load_b64 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+1], v[vgprLocalReadAddrA+0] offset:12384 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+2:vgprValuA_X0_I0+2+1], v[vgprLocalReadAddrA+0] offset:12640 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+1], v[vgprLocalReadAddrA+0] offset:12896 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+6:vgprValuA_X0_I0+6+1], v[vgprLocalReadAddrA+0] offset:13152 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+1], v[vgprLocalReadAddrA+0] offset:13408 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+10:vgprValuA_X0_I0+10+1], v[vgprLocalReadAddrA+0] offset:13664 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+1], v[vgprLocalReadAddrA+0] offset:13920 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+14:vgprValuA_X0_I0+14+1], v[vgprLocalReadAddrA+0] offset:14176 // sync LDS0

/* local read b */
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB+0] offset:3168 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB+0] offset:3296 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrB+0] offset:3424 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrB+0] offset:3552 // sync LDS0

/* local read increment a */
/* N/A, lro->3612 */
/* localReadDoCntA 7 localReadDoCntMXSA 0 localReadDoCntB 7 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->924 */
/* localReadDoCntA 7 localReadDoCntMXSA 0 localReadDoCntB 7 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=0 newLR=0
v_dual_fmac_f32 v[vgprValuC+0], s73, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s73, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s73, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s73, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s77, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s77, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s77, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s77, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s81, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s81, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s81, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s81, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s85, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s85, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s85, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s85, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s89, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s89, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s89, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s89, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s93, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s93, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s93, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s93, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s97, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s97, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s97, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s97, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s101, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s101, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s101, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s101, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s73, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s73, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s73, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s73, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s77, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s77, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s77, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s77, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s81, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s81, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s81, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s81, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s85, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s85, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s85, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s85, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s89, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s89, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s89, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s89, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s93, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s93, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s93, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s93, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s97, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s97, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s97, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s97, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s101, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s101, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s101, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s101, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s73, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s73, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s73, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s73, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s77, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s77, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s77, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s77, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s81, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s81, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s81, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s81, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s85, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s85, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s85, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s85, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s89, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s89, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s89, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s89, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s93, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s93, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s93, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s93, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s97, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s97, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s97, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s97, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s101, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s101, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s101, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s101, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s73, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s73, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s73, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s73, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s77, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s77, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s77, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s77, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s81, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s81, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s81, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s81, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s85, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s85, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s85, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s85, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s89, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s89, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s89, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s89, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s93, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s93, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s93, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s93, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s97, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s97, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s97, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s97, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s101, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s101, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s101, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s101, v[vgprValuA_X1_I0+15]


/* iter 6 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */

/* local read a */
s_waitcnt lgkmcnt(0)
s_waitcnt_depctr 0
ds_load_b64 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+1], v[vgprLocalReadAddrA+0] offset:14448 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+2:vgprValuA_X1_I0+2+1], v[vgprLocalReadAddrA+0] offset:14704 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+1], v[vgprLocalReadAddrA+0] offset:14960 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+6:vgprValuA_X1_I0+6+1], v[vgprLocalReadAddrA+0] offset:15216 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+1], v[vgprLocalReadAddrA+0] offset:15472 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+10:vgprValuA_X1_I0+10+1], v[vgprLocalReadAddrA+0] offset:15728 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+1], v[vgprLocalReadAddrA+0] offset:15984 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+14:vgprValuA_X1_I0+14+1], v[vgprLocalReadAddrA+0] offset:16240 // sync LDS0

/* local read b */
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB+0] offset:3696 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB+0] offset:3824 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrB+0] offset:3952 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrB+0] offset:4080 // sync LDS0
s_waitcnt vmcnt(5) // PGR2: block i+1 landed; block i+2 (+warming) in flight

/* local write A */
ds_store_b32 v[vgprLocalWriteAddrA+0], v[vgprG2LA2+0] offset:32768 // lwoA_0_0_0_0 = (0 + 0*LSCA)*(MT0I+PAD) + (0*LSPA) = 32768 sync LDS1
ds_store_b32 v[vgprLocalWriteAddrA+0], v[vgprG2LA2+1] offset:34832 // lwoA_0_1_0_0 = (1 + 0*LSCA)*(MT0I+PAD) + (0*LSPA) = 34832 sync LDS1
ds_store_b32 v[vgprLocalWriteAddrA+0], v[vgprG2LA2+2] offset:33280 // lwoA_0_0_1_0 = (0 + 0*LSCA)*(MT0I+PAD) + (1*LSPA) = 33280 sync LDS1
ds_store_b32 v[vgprLocalWriteAddrA+0], v[vgprG2LA2+3] offset:35344 // lwoA_0_1_1_0 = (1 + 0*LSCA)*(MT0I+PAD) + (1*LSPA) = 35344 sync LDS1
ds_store_b32 v[vgprLocalWriteAddrA+0], v[vgprG2LA2+4] offset:33792 // lwoA_0_0_2_0 = (0 + 0*LSCA)*(MT0I+PAD) + (2*LSPA) = 33792 sync LDS1
ds_store_b32 v[vgprLocalWriteAddrA+0], v[vgprG2LA2+5] offset:35856 // lwoA_0_1_2_0 = (1 + 0*LSCA)*(MT0I+PAD) + (2*LSPA) = 35856 sync LDS1
ds_store_b32 v[vgprLocalWriteAddrA+0], v[vgprG2LA2+6] offset:34304 // lwoA_0_0_3_0 = (0 + 0*LSCA)*(MT0I+PAD) + (3*LSPA) = 34304 sync LDS1
ds_store_b32 v[vgprLocalWriteAddrA+0], v[vgprG2LA2+7] offset:36368 // lwoA_0_1_3_0 = (1 + 0*LSCA)*(MT0I+PAD) + (3*LSPA) = 36368 sync LDS1

/* local write MXSA */

/* local write MXSB */

/* local write B */
// SCALAR_GLOBAL_B: ds_store_b32 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0] offset:32768 // lwoB_0_0_0_0 = (0 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 32768 sync LDS1
// SCALAR_GLOBAL_B: ds_store_b32 v[vgprLocalWriteAddrB+0], v[vgprG2LB+1] offset:33296 // lwoB_0_1_0_0 = (1 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 33296 sync LDS1

/* local write swap offsets a */

/* (EPS=1) local write swap internal offset -> 0 */

/* local write swap offsets b */

/* (EPS=1) local write swap internal offset -> 0 */

/* local read swap offsets a */

/* local read swap internal offset -> 32768 */

/* local read swap offsets b */

/* local read swap internal offset -> 32768 */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
s_waitcnt lgkmcnt(22)                              // wait for prior local read local write old=0, new=22 newLW=0 newLR=0
v_dual_fmac_f32 v[vgprValuC+0], s74, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s74, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s74, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s74, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s78, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s78, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s78, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s78, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s82, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s82, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s82, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s82, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s86, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s86, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s86, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s86, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s90, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s90, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s90, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s90, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s94, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s94, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s94, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s94, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s98, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s98, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s98, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s98, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s102, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s102, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s102, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s102, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s74, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s74, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s74, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s74, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s78, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s78, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s78, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s78, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s82, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s82, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s82, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s82, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s86, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s86, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s86, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s86, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s90, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s90, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s90, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s90, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s94, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s94, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s94, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s94, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s98, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s98, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s98, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s98, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s102, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s102, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s102, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s102, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s74, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s74, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s74, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s74, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s78, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s78, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s78, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s78, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s82, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s82, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s82, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s82, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s86, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s86, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s86, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s86, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s90, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s90, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s90, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s90, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s94, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s94, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s94, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s94, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s98, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s98, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s98, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s98, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s102, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s102, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s102, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s102, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s74, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s74, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s74, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s74, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s78, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s78, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s78, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s78, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s82, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s82, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s82, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s82, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s86, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s86, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s86, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s86, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s90, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s90, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s90, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s90, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s94, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s94, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s94, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s94, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s98, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s98, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s98, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s98, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s102, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s102, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s102, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s102, v[vgprValuA_X0_I0+15]


/* iter 7 */
s_waitcnt lgkmcnt(0)                               // 3wait for local write
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier                                          // PGR, and wait until LW done to sync

/* local read a */
s_waitcnt lgkmcnt(0)
s_waitcnt_depctr 0
ds_load_b64 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+1], v[vgprLocalReadAddrA+0] offset:32768 // sync LDS1
ds_load_b64 v[vgprValuA_X0_I0+2:vgprValuA_X0_I0+2+1], v[vgprLocalReadAddrA+0] offset:33024 // sync LDS1
ds_load_b64 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+1], v[vgprLocalReadAddrA+0] offset:33280 // sync LDS1
ds_load_b64 v[vgprValuA_X0_I0+6:vgprValuA_X0_I0+6+1], v[vgprLocalReadAddrA+0] offset:33536 // sync LDS1
ds_load_b64 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+1], v[vgprLocalReadAddrA+0] offset:33792 // sync LDS1
ds_load_b64 v[vgprValuA_X0_I0+10:vgprValuA_X0_I0+10+1], v[vgprLocalReadAddrA+0] offset:34048 // sync LDS1
ds_load_b64 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+1], v[vgprLocalReadAddrA+0] offset:34304 // sync LDS1
ds_load_b64 v[vgprValuA_X0_I0+14:vgprValuA_X0_I0+14+1], v[vgprLocalReadAddrA+0] offset:34560 // sync LDS1

/* local read b */
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB+0] offset:32768 // sync LDS1
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB+0] offset:32896 // sync LDS1
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrB+0] offset:33024 // sync LDS1
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrB+0] offset:33152 // sync LDS1

/* local read increment a */
/* N/A, lro->516 */
/* localReadDoCntA 9 localReadDoCntMXSA 0 localReadDoCntB 9 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->132 */
/* localReadDoCntA 9 localReadDoCntMXSA 0 localReadDoCntB 9 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=0 newLR=0
v_dual_fmac_f32 v[vgprValuC+0], s75, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s75, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s75, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s75, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s79, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s79, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s79, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s79, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s83, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s83, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s83, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s83, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s87, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s87, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s87, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s87, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s91, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s91, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s91, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s91, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s95, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s95, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s95, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s95, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s99, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s99, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s99, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s99, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s103, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s103, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s103, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s103, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s75, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s75, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s75, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s75, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s79, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s79, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s79, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s79, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s83, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s83, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s83, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s83, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s87, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s87, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s87, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s87, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s91, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s91, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s91, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s91, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s95, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s95, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s95, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s95, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s99, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s99, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s99, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s99, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s103, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s103, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s103, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s103, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s75, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s75, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s75, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s75, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s79, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s79, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s79, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s79, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s83, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s83, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s83, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s83, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s87, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s87, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s87, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s87, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s91, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s91, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s91, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s91, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s95, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s95, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s95, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s95, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s99, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s99, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s99, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s99, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s103, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s103, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s103, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s103, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s75, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s75, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s75, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s75, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s79, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s79, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s79, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s79, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s83, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s83, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s83, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s83, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s87, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s87, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s87, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s87, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s91, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s91, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s91, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s91, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s95, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s95, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s95, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s95, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s99, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s99, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s99, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s99, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s103, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s103, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s103, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s103, v[vgprValuA_X1_I0+15]


/******************************************/
/* Unrolled Loop - End 1/2                */
/******************************************/

/* closeLoop loopL finalLoop=0 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x1              // counterL==1
s_cbranch_scc1 label_LoopEndL_oddexit              // exit LoopL

/******************************************/
/* Unrolled Loop 2/2 - Begin              */
/******************************************/

/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */
buffer_load_b64 v[vgprG2LA2+0:vgprG2LA2+0+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b64 v[vgprG2LA2+2:vgprG2LA2+2+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b64 v[vgprG2LA2+4:vgprG2LA2+4+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b64 v[vgprG2LA2+6:vgprG2LA2+6+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:0 // G -> Reg 0_0_3_0
s_cmp_gt_u32 s[sgprLoopCounterL], 4 // PF B in range?
s_cbranch_scc1 label_pfb_far2
global_load_b32 v188, v189, s[sgprAddressB:sgprAddressB+1] offset:0 // PF B (tail: current block)
s_branch label_pfb_done2
label_pfb_far2:
global_load_b32 v188, v189, s[sgprAddressB:sgprAddressB+1] offset:64 // PF B
label_pfb_done2:
// SCALAR_GLOBAL_B: buffer_load_b64 v[vgprG2LB+0:vgprG2LB+0+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0

/* global read inc A loopL */
s_mov_b32 s70, s[sgprGlobalReadIncsA+0]                // plain increment (StaggerU unsupported)
s_mov_b32 s71, 0
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s70        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s71       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s70 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s71 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32


/* iter 0 */

/* local read a */
s_waitcnt lgkmcnt(0)
s_waitcnt_depctr 0
ds_load_b64 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+1], v[vgprLocalReadAddrA+0] offset:34832 // sync LDS1
ds_load_b64 v[vgprValuA_X1_I0+2:vgprValuA_X1_I0+2+1], v[vgprLocalReadAddrA+0] offset:35088 // sync LDS1
ds_load_b64 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+1], v[vgprLocalReadAddrA+0] offset:35344 // sync LDS1
ds_load_b64 v[vgprValuA_X1_I0+6:vgprValuA_X1_I0+6+1], v[vgprLocalReadAddrA+0] offset:35600 // sync LDS1
ds_load_b64 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+1], v[vgprLocalReadAddrA+0] offset:35856 // sync LDS1
ds_load_b64 v[vgprValuA_X1_I0+10:vgprValuA_X1_I0+10+1], v[vgprLocalReadAddrA+0] offset:36112 // sync LDS1
ds_load_b64 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+1], v[vgprLocalReadAddrA+0] offset:36368 // sync LDS1
ds_load_b64 v[vgprValuA_X1_I0+14:vgprValuA_X1_I0+14+1], v[vgprLocalReadAddrA+0] offset:36624 // sync LDS1

/* local read b */
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB+0] offset:33296 // sync LDS1
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB+0] offset:33424 // sync LDS1
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrB+0] offset:33552 // sync LDS1
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrB+0] offset:33680 // sync LDS1

/* local read increment a */
/* N/A, lro->1032 */
/* localReadDoCntA 10 localReadDoCntMXSA 0 localReadDoCntB 10 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->264 */
/* localReadDoCntA 10 localReadDoCntMXSA 0 localReadDoCntB 10 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=0 newLR=0 for iteration == 0
s_add_u32 s[sgprAddressB], s[sgprAddressB], 16
s_addc_u32 s[sgprAddressB+1], s[sgprAddressB+1], 0
s_load_b128 s[72:75], s[sgprAddressB:sgprAddressB+1], 0
s_load_b128 s[76:79], s[sgprAddressB:sgprAddressB+1], s69
s_load_b128 s[80:83], s[sgprAddressB:sgprAddressB+1], s37
s_load_b128 s[84:87], s[sgprAddressB:sgprAddressB+1], s38
s_load_b128 s[88:91], s[sgprAddressB:sgprAddressB+1], s39
s_load_b128 s[92:95], s[sgprAddressB:sgprAddressB+1], s52
s_load_b128 s[96:99], s[sgprAddressB:sgprAddressB+1], s53
s_load_b128 s[100:103], s[sgprAddressB:sgprAddressB+1], s54
v_dual_fmac_f32 v[vgprValuC+0], s0, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s0, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s0, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s0, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s4, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s4, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s4, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s4, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s16, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s16, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s16, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s16, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s20, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s20, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s20, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s20, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s40, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s40, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s40, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s40, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s44, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s44, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s44, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s44, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s56, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s56, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s56, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s56, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s60, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s60, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s60, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s60, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s0, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s0, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s0, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s0, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s4, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s4, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s4, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s4, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s16, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s16, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s16, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s16, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s20, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s20, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s20, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s20, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s40, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s40, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s40, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s40, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s44, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s44, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s44, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s44, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s56, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s56, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s56, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s56, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s60, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s60, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s60, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s60, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s0, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s0, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s0, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s0, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s4, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s4, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s4, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s4, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s16, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s16, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s16, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s16, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s20, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s20, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s20, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s20, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s40, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s40, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s40, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s40, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s44, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s44, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s44, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s44, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s56, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s56, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s56, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s56, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s60, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s60, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s60, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s60, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s0, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s0, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s0, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s0, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s4, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s4, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s4, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s4, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s16, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s16, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s16, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s16, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s20, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s20, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s20, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s20, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s40, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s40, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s40, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s40, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s44, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s44, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s44, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s44, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s56, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s56, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s56, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s56, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s60, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s60, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s60, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s60, v[vgprValuA_X0_I0+15]


/* iter 1 */

/* local read a */
s_waitcnt lgkmcnt(0)
s_waitcnt_depctr 0
ds_load_b64 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+1], v[vgprLocalReadAddrA+0] offset:36896 // sync LDS1
ds_load_b64 v[vgprValuA_X0_I0+2:vgprValuA_X0_I0+2+1], v[vgprLocalReadAddrA+0] offset:37152 // sync LDS1
ds_load_b64 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+1], v[vgprLocalReadAddrA+0] offset:37408 // sync LDS1
ds_load_b64 v[vgprValuA_X0_I0+6:vgprValuA_X0_I0+6+1], v[vgprLocalReadAddrA+0] offset:37664 // sync LDS1
ds_load_b64 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+1], v[vgprLocalReadAddrA+0] offset:37920 // sync LDS1
ds_load_b64 v[vgprValuA_X0_I0+10:vgprValuA_X0_I0+10+1], v[vgprLocalReadAddrA+0] offset:38176 // sync LDS1
ds_load_b64 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+1], v[vgprLocalReadAddrA+0] offset:38432 // sync LDS1
ds_load_b64 v[vgprValuA_X0_I0+14:vgprValuA_X0_I0+14+1], v[vgprLocalReadAddrA+0] offset:38688 // sync LDS1

/* local read b */
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB+0] offset:33824 // sync LDS1
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB+0] offset:33952 // sync LDS1
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrB+0] offset:34080 // sync LDS1
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrB+0] offset:34208 // sync LDS1

/* local read increment a */
/* N/A, lro->1548 */
/* localReadDoCntA 11 localReadDoCntMXSA 0 localReadDoCntB 11 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->396 */
/* localReadDoCntA 11 localReadDoCntMXSA 0 localReadDoCntB 11 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=0 newLR=0
v_dual_fmac_f32 v[vgprValuC+0], s1, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s1, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s1, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s1, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s5, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s5, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s5, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s5, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s17, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s17, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s17, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s17, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s21, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s21, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s21, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s21, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s41, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s41, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s41, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s41, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s45, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s45, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s45, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s45, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s57, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s57, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s57, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s57, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s61, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s61, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s61, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s61, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s1, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s1, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s1, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s1, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s5, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s5, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s5, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s5, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s17, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s17, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s17, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s17, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s21, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s21, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s21, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s21, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s41, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s41, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s41, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s41, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s45, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s45, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s45, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s45, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s57, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s57, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s57, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s57, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s61, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s61, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s61, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s61, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s1, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s1, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s1, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s1, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s5, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s5, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s5, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s5, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s17, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s17, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s17, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s17, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s21, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s21, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s21, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s21, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s41, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s41, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s41, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s41, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s45, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s45, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s45, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s45, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s57, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s57, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s57, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s57, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s61, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s61, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s61, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s61, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s1, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s1, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s1, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s1, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s5, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s5, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s5, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s5, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s17, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s17, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s17, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s17, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s21, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s21, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s21, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s21, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s41, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s41, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s41, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s41, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s45, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s45, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s45, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s45, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s57, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s57, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s57, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s57, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s61, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s61, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s61, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s61, v[vgprValuA_X1_I0+15]


/* iter 2 */

/* local read a */
s_waitcnt lgkmcnt(0)
s_waitcnt_depctr 0
ds_load_b64 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+1], v[vgprLocalReadAddrA+0] offset:38960 // sync LDS1
ds_load_b64 v[vgprValuA_X1_I0+2:vgprValuA_X1_I0+2+1], v[vgprLocalReadAddrA+0] offset:39216 // sync LDS1
ds_load_b64 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+1], v[vgprLocalReadAddrA+0] offset:39472 // sync LDS1
ds_load_b64 v[vgprValuA_X1_I0+6:vgprValuA_X1_I0+6+1], v[vgprLocalReadAddrA+0] offset:39728 // sync LDS1
ds_load_b64 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+1], v[vgprLocalReadAddrA+0] offset:39984 // sync LDS1
ds_load_b64 v[vgprValuA_X1_I0+10:vgprValuA_X1_I0+10+1], v[vgprLocalReadAddrA+0] offset:40240 // sync LDS1
ds_load_b64 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+1], v[vgprLocalReadAddrA+0] offset:40496 // sync LDS1
ds_load_b64 v[vgprValuA_X1_I0+14:vgprValuA_X1_I0+14+1], v[vgprLocalReadAddrA+0] offset:40752 // sync LDS1

/* local read b */
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB+0] offset:34352 // sync LDS1
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB+0] offset:34480 // sync LDS1
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrB+0] offset:34608 // sync LDS1
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrB+0] offset:34736 // sync LDS1

/* local read increment a */
/* N/A, lro->2064 */
/* localReadDoCntA 12 localReadDoCntMXSA 0 localReadDoCntB 12 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->528 */
/* localReadDoCntA 12 localReadDoCntMXSA 0 localReadDoCntB 12 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=0 newLR=0
v_dual_fmac_f32 v[vgprValuC+0], s2, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s2, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s2, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s2, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s6, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s6, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s6, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s6, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s18, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s18, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s18, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s18, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s22, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s22, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s22, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s22, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s42, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s42, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s42, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s42, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s46, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s46, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s46, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s46, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s58, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s58, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s58, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s58, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s62, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s62, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s62, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s62, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s2, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s2, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s2, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s2, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s6, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s6, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s6, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s6, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s18, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s18, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s18, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s18, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s22, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s22, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s22, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s22, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s42, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s42, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s42, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s42, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s46, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s46, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s46, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s46, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s58, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s58, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s58, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s58, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s62, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s62, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s62, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s62, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s2, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s2, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s2, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s2, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s6, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s6, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s6, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s6, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s18, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s18, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s18, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s18, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s22, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s22, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s22, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s22, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s42, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s42, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s42, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s42, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s46, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s46, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s46, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s46, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s58, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s58, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s58, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s58, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s62, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s62, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s62, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s62, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s2, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s2, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s2, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s2, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s6, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s6, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s6, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s6, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s18, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s18, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s18, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s18, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s22, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s22, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s22, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s22, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s42, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s42, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s42, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s42, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s46, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s46, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s46, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s46, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s58, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s58, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s58, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s58, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s62, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s62, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s62, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s62, v[vgprValuA_X0_I0+15]


/* iter 3 */

/* local read a */
s_waitcnt lgkmcnt(0)
s_waitcnt_depctr 0
ds_load_b64 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+1], v[vgprLocalReadAddrA+0] offset:41024 // sync LDS1
ds_load_b64 v[vgprValuA_X0_I0+2:vgprValuA_X0_I0+2+1], v[vgprLocalReadAddrA+0] offset:41280 // sync LDS1
ds_load_b64 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+1], v[vgprLocalReadAddrA+0] offset:41536 // sync LDS1
ds_load_b64 v[vgprValuA_X0_I0+6:vgprValuA_X0_I0+6+1], v[vgprLocalReadAddrA+0] offset:41792 // sync LDS1
ds_load_b64 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+1], v[vgprLocalReadAddrA+0] offset:42048 // sync LDS1
ds_load_b64 v[vgprValuA_X0_I0+10:vgprValuA_X0_I0+10+1], v[vgprLocalReadAddrA+0] offset:42304 // sync LDS1
ds_load_b64 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+1], v[vgprLocalReadAddrA+0] offset:42560 // sync LDS1
ds_load_b64 v[vgprValuA_X0_I0+14:vgprValuA_X0_I0+14+1], v[vgprLocalReadAddrA+0] offset:42816 // sync LDS1

/* local read b */
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB+0] offset:34880 // sync LDS1
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB+0] offset:35008 // sync LDS1
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrB+0] offset:35136 // sync LDS1
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrB+0] offset:35264 // sync LDS1

/* local read increment a */
/* N/A, lro->2580 */
/* localReadDoCntA 13 localReadDoCntMXSA 0 localReadDoCntB 13 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->660 */
/* localReadDoCntA 13 localReadDoCntMXSA 0 localReadDoCntB 13 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=0 newLR=0
v_dual_fmac_f32 v[vgprValuC+0], s3, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s3, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s3, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s3, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s7, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s7, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s7, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s7, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s19, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s19, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s19, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s19, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s23, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s23, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s23, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s23, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s43, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s43, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s43, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s43, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s47, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s47, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s47, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s47, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s59, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s59, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s59, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s59, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s63, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s63, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s63, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s63, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s3, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s3, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s3, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s3, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s7, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s7, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s7, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s7, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s19, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s19, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s19, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s19, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s23, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s23, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s23, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s23, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s43, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s43, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s43, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s43, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s47, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s47, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s47, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s47, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s59, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s59, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s59, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s59, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s63, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s63, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s63, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s63, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s3, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s3, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s3, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s3, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s7, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s7, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s7, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s7, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s19, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s19, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s19, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s19, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s23, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s23, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s23, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s23, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s43, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s43, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s43, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s43, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s47, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s47, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s47, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s47, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s59, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s59, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s59, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s59, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s63, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s63, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s63, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s63, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s3, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s3, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s3, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s3, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s7, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s7, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s7, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s7, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s19, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s19, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s19, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s19, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s23, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s23, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s23, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s23, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s43, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s43, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s43, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s43, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s47, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s47, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s47, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s47, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s59, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s59, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s59, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s59, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s63, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s63, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s63, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s63, v[vgprValuA_X1_I0+15]


/* iter 4 */

/* local read a */
s_waitcnt lgkmcnt(0)
s_waitcnt_depctr 0
ds_load_b64 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+1], v[vgprLocalReadAddrA+0] offset:43088 // sync LDS1
ds_load_b64 v[vgprValuA_X1_I0+2:vgprValuA_X1_I0+2+1], v[vgprLocalReadAddrA+0] offset:43344 // sync LDS1
ds_load_b64 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+1], v[vgprLocalReadAddrA+0] offset:43600 // sync LDS1
ds_load_b64 v[vgprValuA_X1_I0+6:vgprValuA_X1_I0+6+1], v[vgprLocalReadAddrA+0] offset:43856 // sync LDS1
ds_load_b64 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+1], v[vgprLocalReadAddrA+0] offset:44112 // sync LDS1
ds_load_b64 v[vgprValuA_X1_I0+10:vgprValuA_X1_I0+10+1], v[vgprLocalReadAddrA+0] offset:44368 // sync LDS1
ds_load_b64 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+1], v[vgprLocalReadAddrA+0] offset:44624 // sync LDS1
ds_load_b64 v[vgprValuA_X1_I0+14:vgprValuA_X1_I0+14+1], v[vgprLocalReadAddrA+0] offset:44880 // sync LDS1

/* local read b */
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB+0] offset:35408 // sync LDS1
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB+0] offset:35536 // sync LDS1
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrB+0] offset:35664 // sync LDS1
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrB+0] offset:35792 // sync LDS1

/* local read increment a */
/* N/A, lro->3096 */
/* localReadDoCntA 14 localReadDoCntMXSA 0 localReadDoCntB 14 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->792 */
/* localReadDoCntA 14 localReadDoCntMXSA 0 localReadDoCntB 14 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=0 newLR=0
s_add_u32 s[sgprAddressB], s[sgprAddressB], 16
s_addc_u32 s[sgprAddressB+1], s[sgprAddressB+1], 0
s_load_b128 s[0:3], s[sgprAddressB:sgprAddressB+1], 0
s_load_b128 s[4:7], s[sgprAddressB:sgprAddressB+1], s69
s_load_b128 s[16:19], s[sgprAddressB:sgprAddressB+1], s37
s_load_b128 s[20:23], s[sgprAddressB:sgprAddressB+1], s38
s_load_b128 s[40:43], s[sgprAddressB:sgprAddressB+1], s39
s_load_b128 s[44:47], s[sgprAddressB:sgprAddressB+1], s52
s_load_b128 s[56:59], s[sgprAddressB:sgprAddressB+1], s53
s_load_b128 s[60:63], s[sgprAddressB:sgprAddressB+1], s54
v_dual_fmac_f32 v[vgprValuC+0], s72, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s72, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s72, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s72, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s76, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s76, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s76, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s76, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s80, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s80, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s80, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s80, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s84, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s84, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s84, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s84, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s88, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s88, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s88, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s88, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s92, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s92, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s92, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s92, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s96, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s96, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s96, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s96, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s100, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s100, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s100, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s100, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s72, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s72, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s72, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s72, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s76, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s76, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s76, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s76, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s80, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s80, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s80, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s80, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s84, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s84, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s84, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s84, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s88, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s88, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s88, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s88, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s92, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s92, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s92, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s92, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s96, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s96, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s96, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s96, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s100, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s100, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s100, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s100, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s72, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s72, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s72, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s72, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s76, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s76, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s76, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s76, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s80, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s80, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s80, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s80, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s84, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s84, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s84, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s84, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s88, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s88, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s88, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s88, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s92, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s92, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s92, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s92, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s96, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s96, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s96, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s96, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s100, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s100, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s100, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s100, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s72, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s72, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s72, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s72, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s76, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s76, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s76, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s76, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s80, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s80, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s80, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s80, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s84, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s84, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s84, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s84, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s88, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s88, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s88, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s88, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s92, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s92, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s92, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s92, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s96, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s96, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s96, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s96, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s100, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s100, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s100, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s100, v[vgprValuA_X0_I0+15]


/* iter 5 */

/* local read a */
s_waitcnt lgkmcnt(0)
s_waitcnt_depctr 0
ds_load_b64 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+1], v[vgprLocalReadAddrA+0] offset:45152 // sync LDS1
ds_load_b64 v[vgprValuA_X0_I0+2:vgprValuA_X0_I0+2+1], v[vgprLocalReadAddrA+0] offset:45408 // sync LDS1
ds_load_b64 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+1], v[vgprLocalReadAddrA+0] offset:45664 // sync LDS1
ds_load_b64 v[vgprValuA_X0_I0+6:vgprValuA_X0_I0+6+1], v[vgprLocalReadAddrA+0] offset:45920 // sync LDS1
ds_load_b64 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+1], v[vgprLocalReadAddrA+0] offset:46176 // sync LDS1
ds_load_b64 v[vgprValuA_X0_I0+10:vgprValuA_X0_I0+10+1], v[vgprLocalReadAddrA+0] offset:46432 // sync LDS1
ds_load_b64 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+1], v[vgprLocalReadAddrA+0] offset:46688 // sync LDS1
ds_load_b64 v[vgprValuA_X0_I0+14:vgprValuA_X0_I0+14+1], v[vgprLocalReadAddrA+0] offset:46944 // sync LDS1

/* local read b */
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB+0] offset:35936 // sync LDS1
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB+0] offset:36064 // sync LDS1
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrB+0] offset:36192 // sync LDS1
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrB+0] offset:36320 // sync LDS1

/* local read increment a */
/* N/A, lro->3612 */
/* localReadDoCntA 15 localReadDoCntMXSA 0 localReadDoCntB 15 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->924 */
/* localReadDoCntA 15 localReadDoCntMXSA 0 localReadDoCntB 15 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=0 newLR=0
v_dual_fmac_f32 v[vgprValuC+0], s73, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s73, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s73, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s73, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s77, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s77, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s77, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s77, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s81, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s81, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s81, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s81, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s85, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s85, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s85, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s85, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s89, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s89, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s89, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s89, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s93, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s93, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s93, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s93, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s97, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s97, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s97, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s97, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s101, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s101, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s101, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s101, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s73, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s73, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s73, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s73, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s77, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s77, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s77, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s77, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s81, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s81, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s81, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s81, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s85, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s85, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s85, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s85, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s89, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s89, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s89, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s89, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s93, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s93, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s93, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s93, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s97, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s97, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s97, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s97, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s101, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s101, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s101, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s101, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s73, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s73, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s73, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s73, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s77, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s77, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s77, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s77, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s81, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s81, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s81, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s81, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s85, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s85, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s85, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s85, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s89, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s89, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s89, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s89, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s93, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s93, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s93, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s93, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s97, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s97, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s97, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s97, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s101, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s101, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s101, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s101, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s73, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s73, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s73, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s73, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s77, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s77, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s77, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s77, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s81, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s81, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s81, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s81, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s85, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s85, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s85, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s85, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s89, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s89, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s89, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s89, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s93, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s93, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s93, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s93, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s97, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s97, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s97, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s97, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s101, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s101, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s101, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s101, v[vgprValuA_X1_I0+15]


/* iter 6 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */

/* local read a */
s_waitcnt lgkmcnt(0)
s_waitcnt_depctr 0
ds_load_b64 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+1], v[vgprLocalReadAddrA+0] offset:47216 // sync LDS1
ds_load_b64 v[vgprValuA_X1_I0+2:vgprValuA_X1_I0+2+1], v[vgprLocalReadAddrA+0] offset:47472 // sync LDS1
ds_load_b64 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+1], v[vgprLocalReadAddrA+0] offset:47728 // sync LDS1
ds_load_b64 v[vgprValuA_X1_I0+6:vgprValuA_X1_I0+6+1], v[vgprLocalReadAddrA+0] offset:47984 // sync LDS1
ds_load_b64 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+1], v[vgprLocalReadAddrA+0] offset:48240 // sync LDS1
ds_load_b64 v[vgprValuA_X1_I0+10:vgprValuA_X1_I0+10+1], v[vgprLocalReadAddrA+0] offset:48496 // sync LDS1
ds_load_b64 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+1], v[vgprLocalReadAddrA+0] offset:48752 // sync LDS1
ds_load_b64 v[vgprValuA_X1_I0+14:vgprValuA_X1_I0+14+1], v[vgprLocalReadAddrA+0] offset:49008 // sync LDS1

/* local read b */
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB+0] offset:36464 // sync LDS1
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB+0] offset:36592 // sync LDS1
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrB+0] offset:36720 // sync LDS1
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrB+0] offset:36848 // sync LDS1
s_waitcnt vmcnt(5) // PGR2: block i+1 landed; block i+2 (+warming) in flight

/* local write A */
ds_store_b32 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0] offset:0 // lwoA_0_0_0_0 = (0 + 0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
ds_store_b32 v[vgprLocalWriteAddrA+0], v[vgprG2LA+1] offset:2064 // lwoA_0_1_0_0 = (1 + 0*LSCA)*(MT0I+PAD) + (0*LSPA) = 2064 sync LDS0
ds_store_b32 v[vgprLocalWriteAddrA+0], v[vgprG2LA+2] offset:512 // lwoA_0_0_1_0 = (0 + 0*LSCA)*(MT0I+PAD) + (1*LSPA) = 512 sync LDS0
ds_store_b32 v[vgprLocalWriteAddrA+0], v[vgprG2LA+3] offset:2576 // lwoA_0_1_1_0 = (1 + 0*LSCA)*(MT0I+PAD) + (1*LSPA) = 2576 sync LDS0
ds_store_b32 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4] offset:1024 // lwoA_0_0_2_0 = (0 + 0*LSCA)*(MT0I+PAD) + (2*LSPA) = 1024 sync LDS0
ds_store_b32 v[vgprLocalWriteAddrA+0], v[vgprG2LA+5] offset:3088 // lwoA_0_1_2_0 = (1 + 0*LSCA)*(MT0I+PAD) + (2*LSPA) = 3088 sync LDS0
ds_store_b32 v[vgprLocalWriteAddrA+0], v[vgprG2LA+6] offset:1536 // lwoA_0_0_3_0 = (0 + 0*LSCA)*(MT0I+PAD) + (3*LSPA) = 1536 sync LDS0
ds_store_b32 v[vgprLocalWriteAddrA+0], v[vgprG2LA+7] offset:3600 // lwoA_0_1_3_0 = (1 + 0*LSCA)*(MT0I+PAD) + (3*LSPA) = 3600 sync LDS0

/* local write MXSA */

/* local write MXSB */

/* local write B */
// SCALAR_GLOBAL_B: ds_store_b32 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0] offset:0 // lwoB_0_0_0_0 = (0 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0
// SCALAR_GLOBAL_B: ds_store_b32 v[vgprLocalWriteAddrB+0], v[vgprG2LB+1] offset:528 // lwoB_0_1_0_0 = (1 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 528 sync LDS0

/* local write swap offsets a */

/* (EPS=1) local write swap internal offset -> 32768 */

/* local write swap offsets b */

/* (EPS=1) local write swap internal offset -> 32768 */

/* local read swap offsets a */

/* local read swap internal offset -> 0 */

/* local read swap offsets b */

/* local read swap internal offset -> 0 */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
s_waitcnt lgkmcnt(22)                              // wait for prior local read local write old=0, new=22 newLW=0 newLR=0
v_dual_fmac_f32 v[vgprValuC+0], s74, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s74, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s74, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s74, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s78, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s78, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s78, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s78, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s82, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s82, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s82, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s82, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s86, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s86, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s86, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s86, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s90, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s90, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s90, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s90, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s94, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s94, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s94, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s94, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s98, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s98, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s98, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s98, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s102, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s102, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s102, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s102, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s74, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s74, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s74, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s74, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s78, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s78, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s78, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s78, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s82, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s82, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s82, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s82, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s86, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s86, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s86, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s86, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s90, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s90, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s90, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s90, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s94, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s94, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s94, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s94, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s98, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s98, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s98, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s98, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s102, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s102, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s102, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s102, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s74, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s74, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s74, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s74, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s78, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s78, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s78, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s78, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s82, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s82, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s82, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s82, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s86, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s86, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s86, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s86, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s90, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s90, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s90, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s90, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s94, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s94, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s94, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s94, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s98, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s98, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s98, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s98, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s102, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s102, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s102, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s102, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s74, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s74, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s74, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s74, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s78, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s78, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s78, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s78, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s82, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s82, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s82, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s82, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s86, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s86, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s86, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s86, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s90, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s90, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s90, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s90, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s94, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s94, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s94, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s94, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s98, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s98, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s98, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s98, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s102, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s102, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s102, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s102, v[vgprValuA_X0_I0+15]


/* iter 7 */
s_waitcnt lgkmcnt(0)                               // 3wait for local write
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier                                          // PGR, and wait until LW done to sync

/* local read a */
s_waitcnt lgkmcnt(0)
s_waitcnt_depctr 0
ds_load_b64 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+1], v[vgprLocalReadAddrA+0] offset:0 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+2:vgprValuA_X0_I0+2+1], v[vgprLocalReadAddrA+0] offset:256 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+1], v[vgprLocalReadAddrA+0] offset:512 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+6:vgprValuA_X0_I0+6+1], v[vgprLocalReadAddrA+0] offset:768 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+1], v[vgprLocalReadAddrA+0] offset:1024 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+10:vgprValuA_X0_I0+10+1], v[vgprLocalReadAddrA+0] offset:1280 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+1], v[vgprLocalReadAddrA+0] offset:1536 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+14:vgprValuA_X0_I0+14+1], v[vgprLocalReadAddrA+0] offset:1792 // sync LDS0

/* local read b */
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB+0] offset:0 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB+0] offset:128 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrB+0] offset:256 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrB+0] offset:384 // sync LDS0

/* local read increment a */
/* N/A, lro->516 */
/* localReadDoCntA 17 localReadDoCntMXSA 0 localReadDoCntB 17 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->132 */
/* localReadDoCntA 17 localReadDoCntMXSA 0 localReadDoCntB 17 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=0 newLR=0
v_dual_fmac_f32 v[vgprValuC+0], s75, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s75, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s75, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s75, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s79, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s79, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s79, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s79, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s83, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s83, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s83, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s83, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s87, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s87, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s87, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s87, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s91, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s91, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s91, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s91, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s95, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s95, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s95, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s95, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s99, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s99, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s99, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s99, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s103, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s103, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s103, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s103, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s75, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s75, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s75, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s75, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s79, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s79, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s79, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s79, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s83, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s83, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s83, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s83, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s87, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s87, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s87, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s87, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s91, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s91, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s91, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s91, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s95, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s95, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s95, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s95, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s99, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s99, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s99, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s99, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s103, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s103, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s103, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s103, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s75, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s75, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s75, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s75, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s79, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s79, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s79, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s79, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s83, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s83, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s83, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s83, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s87, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s87, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s87, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s87, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s91, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s91, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s91, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s91, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s95, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s95, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s95, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s95, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s99, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s99, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s99, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s99, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s103, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s103, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s103, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s103, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s75, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s75, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s75, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s75, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s79, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s79, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s79, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s79, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s83, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s83, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s83, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s83, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s87, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s87, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s87, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s87, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s91, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s91, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s91, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s91, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s95, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s95, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s95, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s95, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s99, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s99, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s99, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s99, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s103, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s103, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s103, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s103, v[vgprValuA_X1_I0+15]


/******************************************/
/* Unrolled Loop - End 2/2 (final)        */
/******************************************/

/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x1              // counterL==1
s_cbranch_scc0 label_LoopBeginL                    // restart LoopL
label_LoopEndL_evenexit:  /// unroll loop eveniter exit
s_branch label_LoopEndL                            // exit unroll loopL (and skip second exit code)
label_LoopEndL_oddexit:  /// unroll loop odditer exit

/* Select high bank of LDS */
v_xor_b32 v[vgprLocalReadAddrA], 0x8000, v[vgprLocalReadAddrA] // swap Red Blk
v_xor_b32 v[vgprLocalReadAddrB], 0x8000, v[vgprLocalReadAddrB] // swap Red Blk
label_LoopEndL:

// Full-tile alpha1/beta0/GSU1 path.
/* iter 0 (last unrolled loop) */

/* local read a */
s_waitcnt lgkmcnt(0)
s_waitcnt_depctr 0
ds_load_b64 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+1], v[vgprLocalReadAddrA+0] offset:2064 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+2:vgprValuA_X1_I0+2+1], v[vgprLocalReadAddrA+0] offset:2320 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+1], v[vgprLocalReadAddrA+0] offset:2576 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+6:vgprValuA_X1_I0+6+1], v[vgprLocalReadAddrA+0] offset:2832 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+1], v[vgprLocalReadAddrA+0] offset:3088 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+10:vgprValuA_X1_I0+10+1], v[vgprLocalReadAddrA+0] offset:3344 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+1], v[vgprLocalReadAddrA+0] offset:3600 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+14:vgprValuA_X1_I0+14+1], v[vgprLocalReadAddrA+0] offset:3856 // sync LDS0

/* local read b */
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB+0] offset:528 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB+0] offset:656 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrB+0] offset:784 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrB+0] offset:912 // sync LDS0

/* local read increment a */
/* N/A, lro->1032 */
/* localReadDoCntA 18 localReadDoCntMXSA 0 localReadDoCntB 18 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->264 */
/* localReadDoCntA 18 localReadDoCntMXSA 0 localReadDoCntB 18 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=0 newLR=0 for iteration == 0
s_add_u32 s[sgprAddressB], s[sgprAddressB], 16
s_addc_u32 s[sgprAddressB+1], s[sgprAddressB+1], 0
s_load_b128 s[72:75], s[sgprAddressB:sgprAddressB+1], 0
s_load_b128 s[76:79], s[sgprAddressB:sgprAddressB+1], s69
s_load_b128 s[80:83], s[sgprAddressB:sgprAddressB+1], s37
s_load_b128 s[84:87], s[sgprAddressB:sgprAddressB+1], s38
s_load_b128 s[88:91], s[sgprAddressB:sgprAddressB+1], s39
s_load_b128 s[92:95], s[sgprAddressB:sgprAddressB+1], s52
s_load_b128 s[96:99], s[sgprAddressB:sgprAddressB+1], s53
s_load_b128 s[100:103], s[sgprAddressB:sgprAddressB+1], s54
v_dual_fmac_f32 v[vgprValuC+0], s0, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s0, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s0, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s0, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s4, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s4, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s4, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s4, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s16, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s16, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s16, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s16, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s20, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s20, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s20, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s20, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s40, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s40, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s40, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s40, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s44, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s44, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s44, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s44, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s56, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s56, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s56, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s56, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s60, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s60, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s60, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s60, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s0, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s0, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s0, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s0, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s4, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s4, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s4, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s4, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s16, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s16, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s16, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s16, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s20, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s20, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s20, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s20, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s40, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s40, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s40, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s40, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s44, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s44, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s44, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s44, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s56, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s56, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s56, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s56, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s60, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s60, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s60, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s60, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s0, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s0, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s0, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s0, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s4, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s4, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s4, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s4, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s16, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s16, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s16, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s16, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s20, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s20, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s20, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s20, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s40, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s40, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s40, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s40, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s44, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s44, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s44, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s44, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s56, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s56, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s56, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s56, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s60, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s60, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s60, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s60, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s0, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s0, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s0, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s0, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s4, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s4, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s4, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s4, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s16, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s16, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s16, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s16, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s20, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s20, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s20, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s20, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s40, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s40, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s40, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s40, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s44, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s44, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s44, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s44, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s56, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s56, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s56, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s56, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s60, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s60, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s60, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s60, v[vgprValuA_X0_I0+15]


/* iter 1 (last unrolled loop) */

/* local read a */
s_waitcnt lgkmcnt(0)
s_waitcnt_depctr 0
ds_load_b64 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+1], v[vgprLocalReadAddrA+0] offset:4128 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+2:vgprValuA_X0_I0+2+1], v[vgprLocalReadAddrA+0] offset:4384 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+1], v[vgprLocalReadAddrA+0] offset:4640 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+6:vgprValuA_X0_I0+6+1], v[vgprLocalReadAddrA+0] offset:4896 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+1], v[vgprLocalReadAddrA+0] offset:5152 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+10:vgprValuA_X0_I0+10+1], v[vgprLocalReadAddrA+0] offset:5408 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+1], v[vgprLocalReadAddrA+0] offset:5664 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+14:vgprValuA_X0_I0+14+1], v[vgprLocalReadAddrA+0] offset:5920 // sync LDS0

/* local read b */
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB+0] offset:1056 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB+0] offset:1184 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrB+0] offset:1312 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrB+0] offset:1440 // sync LDS0

/* local read increment a */
/* N/A, lro->1548 */
/* localReadDoCntA 19 localReadDoCntMXSA 0 localReadDoCntB 19 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->396 */
/* localReadDoCntA 19 localReadDoCntMXSA 0 localReadDoCntB 19 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=0 newLR=0
v_dual_fmac_f32 v[vgprValuC+0], s1, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s1, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s1, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s1, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s5, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s5, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s5, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s5, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s17, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s17, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s17, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s17, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s21, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s21, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s21, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s21, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s41, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s41, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s41, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s41, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s45, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s45, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s45, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s45, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s57, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s57, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s57, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s57, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s61, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s61, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s61, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s61, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s1, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s1, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s1, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s1, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s5, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s5, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s5, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s5, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s17, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s17, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s17, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s17, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s21, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s21, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s21, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s21, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s41, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s41, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s41, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s41, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s45, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s45, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s45, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s45, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s57, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s57, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s57, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s57, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s61, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s61, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s61, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s61, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s1, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s1, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s1, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s1, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s5, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s5, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s5, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s5, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s17, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s17, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s17, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s17, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s21, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s21, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s21, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s21, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s41, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s41, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s41, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s41, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s45, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s45, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s45, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s45, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s57, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s57, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s57, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s57, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s61, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s61, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s61, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s61, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s1, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s1, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s1, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s1, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s5, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s5, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s5, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s5, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s17, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s17, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s17, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s17, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s21, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s21, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s21, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s21, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s41, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s41, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s41, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s41, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s45, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s45, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s45, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s45, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s57, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s57, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s57, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s57, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s61, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s61, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s61, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s61, v[vgprValuA_X1_I0+15]


/* iter 2 (last unrolled loop) */

/* local read a */
s_waitcnt lgkmcnt(0)
s_waitcnt_depctr 0
ds_load_b64 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+1], v[vgprLocalReadAddrA+0] offset:6192 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+2:vgprValuA_X1_I0+2+1], v[vgprLocalReadAddrA+0] offset:6448 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+1], v[vgprLocalReadAddrA+0] offset:6704 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+6:vgprValuA_X1_I0+6+1], v[vgprLocalReadAddrA+0] offset:6960 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+1], v[vgprLocalReadAddrA+0] offset:7216 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+10:vgprValuA_X1_I0+10+1], v[vgprLocalReadAddrA+0] offset:7472 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+1], v[vgprLocalReadAddrA+0] offset:7728 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+14:vgprValuA_X1_I0+14+1], v[vgprLocalReadAddrA+0] offset:7984 // sync LDS0

/* local read b */
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB+0] offset:1584 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB+0] offset:1712 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrB+0] offset:1840 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrB+0] offset:1968 // sync LDS0

/* local read increment a */
/* N/A, lro->2064 */
/* localReadDoCntA 20 localReadDoCntMXSA 0 localReadDoCntB 20 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->528 */
/* localReadDoCntA 20 localReadDoCntMXSA 0 localReadDoCntB 20 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=0 newLR=0
v_dual_fmac_f32 v[vgprValuC+0], s2, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s2, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s2, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s2, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s6, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s6, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s6, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s6, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s18, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s18, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s18, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s18, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s22, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s22, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s22, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s22, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s42, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s42, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s42, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s42, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s46, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s46, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s46, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s46, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s58, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s58, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s58, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s58, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s62, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s62, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s62, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s62, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s2, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s2, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s2, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s2, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s6, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s6, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s6, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s6, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s18, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s18, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s18, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s18, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s22, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s22, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s22, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s22, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s42, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s42, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s42, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s42, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s46, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s46, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s46, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s46, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s58, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s58, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s58, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s58, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s62, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s62, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s62, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s62, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s2, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s2, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s2, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s2, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s6, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s6, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s6, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s6, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s18, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s18, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s18, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s18, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s22, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s22, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s22, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s22, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s42, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s42, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s42, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s42, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s46, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s46, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s46, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s46, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s58, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s58, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s58, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s58, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s62, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s62, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s62, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s62, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s2, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s2, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s2, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s2, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s6, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s6, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s6, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s6, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s18, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s18, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s18, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s18, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s22, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s22, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s22, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s22, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s42, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s42, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s42, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s42, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s46, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s46, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s46, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s46, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s58, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s58, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s58, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s58, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s62, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s62, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s62, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s62, v[vgprValuA_X0_I0+15]


/* iter 3 (last unrolled loop) */

/* local read a */
s_waitcnt lgkmcnt(0)
s_waitcnt_depctr 0
ds_load_b64 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+1], v[vgprLocalReadAddrA+0] offset:8256 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+2:vgprValuA_X0_I0+2+1], v[vgprLocalReadAddrA+0] offset:8512 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+1], v[vgprLocalReadAddrA+0] offset:8768 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+6:vgprValuA_X0_I0+6+1], v[vgprLocalReadAddrA+0] offset:9024 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+1], v[vgprLocalReadAddrA+0] offset:9280 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+10:vgprValuA_X0_I0+10+1], v[vgprLocalReadAddrA+0] offset:9536 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+1], v[vgprLocalReadAddrA+0] offset:9792 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+14:vgprValuA_X0_I0+14+1], v[vgprLocalReadAddrA+0] offset:10048 // sync LDS0

/* local read b */
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB+0] offset:2112 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB+0] offset:2240 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrB+0] offset:2368 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrB+0] offset:2496 // sync LDS0

/* local read increment a */
/* N/A, lro->2580 */
/* localReadDoCntA 21 localReadDoCntMXSA 0 localReadDoCntB 21 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->660 */
/* localReadDoCntA 21 localReadDoCntMXSA 0 localReadDoCntB 21 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=0 newLR=0
v_dual_fmac_f32 v[vgprValuC+0], s3, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s3, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s3, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s3, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s7, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s7, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s7, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s7, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s19, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s19, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s19, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s19, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s23, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s23, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s23, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s23, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s43, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s43, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s43, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s43, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s47, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s47, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s47, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s47, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s59, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s59, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s59, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s59, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s63, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s63, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s63, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s63, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s3, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s3, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s3, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s3, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s7, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s7, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s7, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s7, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s19, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s19, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s19, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s19, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s23, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s23, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s23, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s23, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s43, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s43, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s43, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s43, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s47, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s47, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s47, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s47, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s59, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s59, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s59, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s59, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s63, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s63, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s63, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s63, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s3, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s3, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s3, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s3, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s7, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s7, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s7, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s7, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s19, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s19, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s19, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s19, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s23, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s23, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s23, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s23, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s43, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s43, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s43, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s43, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s47, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s47, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s47, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s47, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s59, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s59, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s59, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s59, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s63, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s63, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s63, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s63, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s3, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s3, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s3, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s3, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s7, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s7, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s7, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s7, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s19, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s19, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s19, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s19, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s23, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s23, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s23, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s23, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s43, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s43, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s43, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s43, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s47, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s47, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s47, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s47, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s59, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s59, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s59, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s59, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s63, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s63, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s63, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s63, v[vgprValuA_X1_I0+15]


/* iter 4 (last unrolled loop) */

/* local read a */
s_waitcnt lgkmcnt(0)
s_waitcnt_depctr 0
ds_load_b64 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+1], v[vgprLocalReadAddrA+0] offset:10320 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+2:vgprValuA_X1_I0+2+1], v[vgprLocalReadAddrA+0] offset:10576 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+1], v[vgprLocalReadAddrA+0] offset:10832 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+6:vgprValuA_X1_I0+6+1], v[vgprLocalReadAddrA+0] offset:11088 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+1], v[vgprLocalReadAddrA+0] offset:11344 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+10:vgprValuA_X1_I0+10+1], v[vgprLocalReadAddrA+0] offset:11600 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+1], v[vgprLocalReadAddrA+0] offset:11856 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+14:vgprValuA_X1_I0+14+1], v[vgprLocalReadAddrA+0] offset:12112 // sync LDS0

/* local read b */
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB+0] offset:2640 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB+0] offset:2768 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrB+0] offset:2896 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrB+0] offset:3024 // sync LDS0

/* local read increment a */
/* N/A, lro->3096 */
/* localReadDoCntA 22 localReadDoCntMXSA 0 localReadDoCntB 22 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->792 */
/* localReadDoCntA 22 localReadDoCntMXSA 0 localReadDoCntB 22 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=0 newLR=0
v_dual_fmac_f32 v[vgprValuC+0], s72, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s72, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s72, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s72, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s76, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s76, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s76, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s76, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s80, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s80, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s80, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s80, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s84, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s84, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s84, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s84, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s88, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s88, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s88, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s88, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s92, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s92, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s92, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s92, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s96, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s96, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s96, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s96, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s100, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s100, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s100, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s100, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s72, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s72, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s72, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s72, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s76, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s76, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s76, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s76, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s80, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s80, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s80, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s80, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s84, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s84, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s84, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s84, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s88, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s88, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s88, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s88, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s92, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s92, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s92, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s92, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s96, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s96, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s96, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s96, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s100, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s100, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s100, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s100, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s72, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s72, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s72, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s72, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s76, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s76, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s76, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s76, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s80, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s80, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s80, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s80, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s84, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s84, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s84, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s84, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s88, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s88, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s88, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s88, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s92, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s92, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s92, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s92, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s96, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s96, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s96, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s96, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s100, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s100, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s100, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s100, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s72, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s72, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s72, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s72, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s76, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s76, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s76, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s76, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s80, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s80, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s80, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s80, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s84, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s84, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s84, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s84, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s88, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s88, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s88, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s88, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s92, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s92, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s92, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s92, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s96, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s96, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s96, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s96, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s100, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s100, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s100, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s100, v[vgprValuA_X0_I0+15]


/* iter 5 (last unrolled loop) */

/* local read a */
s_waitcnt lgkmcnt(0)
s_waitcnt_depctr 0
ds_load_b64 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+1], v[vgprLocalReadAddrA+0] offset:12384 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+2:vgprValuA_X0_I0+2+1], v[vgprLocalReadAddrA+0] offset:12640 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+1], v[vgprLocalReadAddrA+0] offset:12896 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+6:vgprValuA_X0_I0+6+1], v[vgprLocalReadAddrA+0] offset:13152 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+1], v[vgprLocalReadAddrA+0] offset:13408 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+10:vgprValuA_X0_I0+10+1], v[vgprLocalReadAddrA+0] offset:13664 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+1], v[vgprLocalReadAddrA+0] offset:13920 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+14:vgprValuA_X0_I0+14+1], v[vgprLocalReadAddrA+0] offset:14176 // sync LDS0

/* local read b */
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB+0] offset:3168 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB+0] offset:3296 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrB+0] offset:3424 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrB+0] offset:3552 // sync LDS0

/* local read increment a */
/* N/A, lro->3612 */
/* localReadDoCntA 23 localReadDoCntMXSA 0 localReadDoCntB 23 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->924 */
/* localReadDoCntA 23 localReadDoCntMXSA 0 localReadDoCntB 23 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=0 newLR=0
v_dual_fmac_f32 v[vgprValuC+0], s73, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s73, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s73, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s73, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s77, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s77, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s77, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s77, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s81, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s81, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s81, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s81, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s85, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s85, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s85, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s85, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s89, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s89, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s89, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s89, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s93, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s93, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s93, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s93, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s97, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s97, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s97, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s97, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s101, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s101, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s101, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s101, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s73, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s73, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s73, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s73, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s77, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s77, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s77, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s77, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s81, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s81, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s81, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s81, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s85, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s85, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s85, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s85, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s89, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s89, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s89, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s89, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s93, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s93, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s93, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s93, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s97, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s97, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s97, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s97, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s101, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s101, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s101, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s101, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s73, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s73, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s73, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s73, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s77, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s77, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s77, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s77, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s81, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s81, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s81, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s81, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s85, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s85, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s85, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s85, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s89, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s89, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s89, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s89, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s93, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s93, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s93, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s93, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s97, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s97, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s97, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s97, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s101, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s101, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s101, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s101, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s73, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s73, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s73, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s73, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s77, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s77, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s77, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s77, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s81, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s81, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s81, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s81, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s85, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s85, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s85, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s85, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s89, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s89, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s89, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s89, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s93, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s93, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s93, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s93, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s97, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s97, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s97, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s97, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s101, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s101, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s101, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s101, v[vgprValuA_X1_I0+15]


/* iter 6 (last unrolled loop) */

/* local read a */
s_waitcnt lgkmcnt(0)
s_waitcnt_depctr 0
ds_load_b64 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+1], v[vgprLocalReadAddrA+0] offset:14448 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+2:vgprValuA_X1_I0+2+1], v[vgprLocalReadAddrA+0] offset:14704 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+1], v[vgprLocalReadAddrA+0] offset:14960 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+6:vgprValuA_X1_I0+6+1], v[vgprLocalReadAddrA+0] offset:15216 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+1], v[vgprLocalReadAddrA+0] offset:15472 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+10:vgprValuA_X1_I0+10+1], v[vgprLocalReadAddrA+0] offset:15728 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+1], v[vgprLocalReadAddrA+0] offset:15984 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+14:vgprValuA_X1_I0+14+1], v[vgprLocalReadAddrA+0] offset:16240 // sync LDS0

/* local read b */
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB+0] offset:3696 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB+0] offset:3824 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrB+0] offset:3952 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrB+0] offset:4080 // sync LDS0
s_waitcnt vmcnt(0)                                 // 1wait for global read

/* local write A */

/* local write MXSA */

/* local write MXSB */

/* local write B */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=0 newLR=0
v_dual_fmac_f32 v[vgprValuC+0], s74, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s74, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s74, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s74, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s78, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s78, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s78, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s78, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s82, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s82, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s82, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s82, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s86, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s86, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s86, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s86, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s90, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s90, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s90, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s90, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s94, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s94, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s94, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s94, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s98, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s98, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s98, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s98, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s102, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s102, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s102, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s102, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s74, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s74, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s74, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s74, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s78, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s78, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s78, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s78, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s82, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s82, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s82, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s82, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s86, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s86, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s86, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s86, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s90, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s90, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s90, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s90, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s94, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s94, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s94, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s94, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s98, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s98, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s98, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s98, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s102, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s102, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s102, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s102, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s74, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s74, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s74, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s74, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s78, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s78, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s78, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s78, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s82, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s82, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s82, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s82, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s86, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s86, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s86, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s86, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s90, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s90, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s90, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s90, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s94, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s94, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s94, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s94, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s98, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s98, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s98, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s98, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s102, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s102, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s102, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s102, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s74, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s74, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s74, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s74, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s78, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s78, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s78, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s78, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s82, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s82, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s82, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s82, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s86, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s86, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s86, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s86, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s90, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s90, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s90, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s90, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s94, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s94, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s94, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s94, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s98, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s98, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s98, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s98, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s102, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s102, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s102, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s102, v[vgprValuA_X0_I0+15]


/* iter 7 (last unrolled loop) */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->4128 */
/* localReadDoCntA 25 localReadDoCntMXSA 0 localReadDoCntB 25 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->1056 */
/* localReadDoCntA 25 localReadDoCntMXSA 0 localReadDoCntB 25 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
s_waitcnt lgkmcnt(0)
s_waitcnt_depctr 0
v_dual_fmac_f32 v[vgprValuC+0], s75, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s75, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s75, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s75, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s79, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s79, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s79, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s79, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s83, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s83, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s83, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s83, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s87, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s87, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s87, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s87, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s91, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s91, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s91, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s91, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s95, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s95, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s95, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s95, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s99, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s99, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s99, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s99, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s103, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s103, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s103, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s103, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s75, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s75, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s75, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s75, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s79, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s79, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s79, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s79, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s83, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s83, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s83, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s83, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s87, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s87, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s87, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s87, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s91, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s91, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s91, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s91, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s95, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s95, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s95, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s95, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s99, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s99, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s99, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s99, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s103, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s103, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s103, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s103, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s75, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s75, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s75, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s75, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s79, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s79, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s79, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s79, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s83, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s83, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s83, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s83, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s87, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s87, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s87, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s87, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s91, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s91, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s91, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s91, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s95, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s95, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s95, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s95, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s99, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s99, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s99, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s99, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s103, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s103, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s103, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s103, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s75, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s75, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s75, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s75, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s79, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s79, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s79, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s79, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s83, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s83, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s83, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s83, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s87, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s87, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s87, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s87, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s91, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s91, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s91, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s91, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s95, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s95, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s95, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s95, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s99, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s99, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s99, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s99, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s103, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s103, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s103, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s103, v[vgprValuA_X1_I0+15]

label_toPGR1end_OptNLL:
/* Stores for OptNLL */
label_Summation_End_OptNLL:
s_branch label_Summation_End_6
/* endSummation: add vgpr [128...196) to pool */
/* load store sgprs */
/* computeStoreVgprs */
v_lshrrev_b32 v129, 5, v[vgprSerial]               // 129 = Serial / 32
v_and_b32 v128, 31, v[vgprSerial]                  // 128 = Serial % 32
v_lshlrev_b32 v128, 1, v128                        // v128 = v128 * 2
v_lshlrev_b32 v129, 1, v129                        // v129 = v129 * 2
v_mul_lo_u32 v130, v129, s[sgprStrideC1J]          // rowStart vgpr
v_mul_lo_u32 v131, v129, s[sgprStrideD1J]          // rowStart vgpr

s_mul_i32 s30, 0x200, s[sgprWorkGroup0]            // s30 = wg0*MT0
v_add_nc_u32 v128, s30, v128                       // coord0 = tid0*VW + wg0*MT0
s_mul_i32 s32, 0x80, s[sgprWorkGroup1]             // <- wg1*MT1
v_add_nc_u32 v129, s32, v129                       // coord1 = tid1*VW + wg1*MT1

/******************************************/
/* Global Write Elements                  */
/******************************************/
label_GW_B0_OptNLL_MB:
label_GW_B0_FD0_OptNLL_MB:
label_GW_B0_FD0_VW2_OptNLL_MB_Then:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=4 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw2); (0,1,0,0:vw2); (0,2,0,0:vw2); (0,3,0,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
v_add_lshl_u32 v135, v131, v128, 2                 // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=128, coord0Vgpr=128 (multiple bpe)

/* apply mask, calc new C and issue writes */
buffer_store_b64 v[0:1], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b64 v[2:3], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b64 v[4:5], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
buffer_store_b64 v[6:7], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:768 // store D
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #1 (d1,d0,vc1,vc0) = */
/*    (0,4,0,0:vw2); (0,5,0,0:vw2); (0,6,0,0:vw2); (0,7,0,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
/* (d1,vc1,d0,vc0)=(0,0,7,0) */

/* apply mask, calc new C and issue writes */
buffer_store_b64 v[8:9], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:1024 // store D
buffer_store_b64 v[10:11], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:1280 // store D
buffer_store_b64 v[12:13], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:1536 // store D
buffer_store_b64 v[14:15], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:1792 // store D
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #2 (d1,d0,vc1,vc0) = */
/*    (0,0,1,0:vw2); (0,1,1,0:vw2); (0,2,1,0:vw2); (0,3,1,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
/* (d1,vc1,d0,vc0)=(0,1,2,0) */
/* (d1,vc1,d0,vc0)=(0,1,3,0) */

/* apply mask, calc new C and issue writes */
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_b64 v[16:17], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b64 v[18:19], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b64 v[20:21], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
buffer_store_b64 v[22:23], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:768 // store D
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #3 (d1,d0,vc1,vc0) = */
/*    (0,4,1,0:vw2); (0,5,1,0:vw2); (0,6,1,0:vw2); (0,7,1,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,1,4,0) */
/* (d1,vc1,d0,vc0)=(0,1,5,0) */
/* (d1,vc1,d0,vc0)=(0,1,6,0) */
/* (d1,vc1,d0,vc0)=(0,1,7,0) */

/* apply mask, calc new C and issue writes */
buffer_store_b64 v[24:25], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:1024 // store D
buffer_store_b64 v[26:27], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:1280 // store D
buffer_store_b64 v[28:29], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:1536 // store D
buffer_store_b64 v[30:31], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:1792 // store D
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #4 (d1,d0,vc1,vc0) = */
/*    (1,0,0,0:vw2); (1,1,0,0:vw2); (1,2,0,0:vw2); (1,3,0,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
/* (d1,vc1,d0,vc0)=(1,0,3,0) */

/* apply mask, calc new C and issue writes */
s_mul_i32 s8, s[sgprStrideD1J], 124                // scale StrideD *= numRows(31) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(31): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(31): gra SRD += inc(upper)
buffer_store_b64 v[32:33], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b64 v[34:35], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b64 v[36:37], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
buffer_store_b64 v[38:39], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:768 // store D
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #5 (d1,d0,vc1,vc0) = */
/*    (1,4,0,0:vw2); (1,5,0,0:vw2); (1,6,0,0:vw2); (1,7,0,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(1,0,4,0) */
/* (d1,vc1,d0,vc0)=(1,0,5,0) */
/* (d1,vc1,d0,vc0)=(1,0,6,0) */
/* (d1,vc1,d0,vc0)=(1,0,7,0) */

/* apply mask, calc new C and issue writes */
buffer_store_b64 v[40:41], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:1024 // store D
buffer_store_b64 v[42:43], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:1280 // store D
buffer_store_b64 v[44:45], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:1536 // store D
buffer_store_b64 v[46:47], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:1792 // store D
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #6 (d1,d0,vc1,vc0) = */
/*    (1,0,1,0:vw2); (1,1,1,0:vw2); (1,2,1,0:vw2); (1,3,1,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(1,1,0,0) */
/* (d1,vc1,d0,vc0)=(1,1,1,0) */
/* (d1,vc1,d0,vc0)=(1,1,2,0) */
/* (d1,vc1,d0,vc0)=(1,1,3,0) */

/* apply mask, calc new C and issue writes */
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_b64 v[48:49], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b64 v[50:51], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b64 v[52:53], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
buffer_store_b64 v[54:55], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:768 // store D
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #7 (d1,d0,vc1,vc0) = */
/*    (1,4,1,0:vw2); (1,5,1,0:vw2); (1,6,1,0:vw2); (1,7,1,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(1,1,4,0) */
/* (d1,vc1,d0,vc0)=(1,1,5,0) */
/* (d1,vc1,d0,vc0)=(1,1,6,0) */
/* (d1,vc1,d0,vc0)=(1,1,7,0) */

/* apply mask, calc new C and issue writes */
buffer_store_b64 v[56:57], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:1024 // store D
buffer_store_b64 v[58:59], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:1280 // store D
buffer_store_b64 v[60:61], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:1536 // store D
buffer_store_b64 v[62:63], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:1792 // store D
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #8 (d1,d0,vc1,vc0) = */
/*    (2,0,0,0:vw2); (2,1,0,0:vw2); (2,2,0,0:vw2); (2,3,0,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
/* (d1,vc1,d0,vc0)=(2,0,3,0) */

/* apply mask, calc new C and issue writes */
s_mul_i32 s8, s[sgprStrideD1J], 124                // scale StrideD *= numRows(31) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(31): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(31): gra SRD += inc(upper)
buffer_store_b64 v[64:65], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b64 v[66:67], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b64 v[68:69], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
buffer_store_b64 v[70:71], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:768 // store D
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #9 (d1,d0,vc1,vc0) = */
/*    (2,4,0,0:vw2); (2,5,0,0:vw2); (2,6,0,0:vw2); (2,7,0,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(2,0,4,0) */
/* (d1,vc1,d0,vc0)=(2,0,5,0) */
/* (d1,vc1,d0,vc0)=(2,0,6,0) */
/* (d1,vc1,d0,vc0)=(2,0,7,0) */

/* apply mask, calc new C and issue writes */
buffer_store_b64 v[72:73], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:1024 // store D
buffer_store_b64 v[74:75], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:1280 // store D
buffer_store_b64 v[76:77], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:1536 // store D
buffer_store_b64 v[78:79], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:1792 // store D
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #10 (d1,d0,vc1,vc0) = */
/*    (2,0,1,0:vw2); (2,1,1,0:vw2); (2,2,1,0:vw2); (2,3,1,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(2,1,0,0) */
/* (d1,vc1,d0,vc0)=(2,1,1,0) */
/* (d1,vc1,d0,vc0)=(2,1,2,0) */
/* (d1,vc1,d0,vc0)=(2,1,3,0) */

/* apply mask, calc new C and issue writes */
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_b64 v[80:81], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b64 v[82:83], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b64 v[84:85], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
buffer_store_b64 v[86:87], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:768 // store D
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #11 (d1,d0,vc1,vc0) = */
/*    (2,4,1,0:vw2); (2,5,1,0:vw2); (2,6,1,0:vw2); (2,7,1,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(2,1,4,0) */
/* (d1,vc1,d0,vc0)=(2,1,5,0) */
/* (d1,vc1,d0,vc0)=(2,1,6,0) */
/* (d1,vc1,d0,vc0)=(2,1,7,0) */

/* apply mask, calc new C and issue writes */
buffer_store_b64 v[88:89], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:1024 // store D
buffer_store_b64 v[90:91], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:1280 // store D
buffer_store_b64 v[92:93], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:1536 // store D
buffer_store_b64 v[94:95], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:1792 // store D
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #12 (d1,d0,vc1,vc0) = */
/*    (3,0,0,0:vw2); (3,1,0,0:vw2); (3,2,0,0:vw2); (3,3,0,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
/* (d1,vc1,d0,vc0)=(3,0,3,0) */

/* apply mask, calc new C and issue writes */
s_mul_i32 s8, s[sgprStrideD1J], 124                // scale StrideD *= numRows(31) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(31): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(31): gra SRD += inc(upper)
buffer_store_b64 v[96:97], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b64 v[98:99], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b64 v[100:101], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
buffer_store_b64 v[102:103], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:768 // store D
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #13 (d1,d0,vc1,vc0) = */
/*    (3,4,0,0:vw2); (3,5,0,0:vw2); (3,6,0,0:vw2); (3,7,0,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(3,0,4,0) */
/* (d1,vc1,d0,vc0)=(3,0,5,0) */
/* (d1,vc1,d0,vc0)=(3,0,6,0) */
/* (d1,vc1,d0,vc0)=(3,0,7,0) */

/* apply mask, calc new C and issue writes */
buffer_store_b64 v[104:105], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:1024 // store D
buffer_store_b64 v[106:107], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:1280 // store D
buffer_store_b64 v[108:109], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:1536 // store D
buffer_store_b64 v[110:111], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:1792 // store D
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #14 (d1,d0,vc1,vc0) = */
/*    (3,0,1,0:vw2); (3,1,1,0:vw2); (3,2,1,0:vw2); (3,3,1,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(3,1,0,0) */
/* (d1,vc1,d0,vc0)=(3,1,1,0) */
/* (d1,vc1,d0,vc0)=(3,1,2,0) */
/* (d1,vc1,d0,vc0)=(3,1,3,0) */

/* apply mask, calc new C and issue writes */
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_b64 v[112:113], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b64 v[114:115], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b64 v[116:117], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
buffer_store_b64 v[118:119], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:768 // store D
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #15 (d1,d0,vc1,vc0) = */
/*    (3,4,1,0:vw2); (3,5,1,0:vw2); (3,6,1,0:vw2); (3,7,1,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(3,1,4,0) */
/* (d1,vc1,d0,vc0)=(3,1,5,0) */
/* (d1,vc1,d0,vc0)=(3,1,6,0) */
/* (d1,vc1,d0,vc0)=(3,1,7,0) */

/* apply mask, calc new C and issue writes */
buffer_store_b64 v[120:121], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:1024 // store D
buffer_store_b64 v[122:123], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:1280 // store D
buffer_store_b64 v[124:125], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:1536 // store D
buffer_store_b64 v[126:127], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:1792 // store D
s_branch label_GW_End                              // jump to end
label_GW_End:

s_endpgm                                           // Kernel End
label_OptNLL_End:
label_GSU_3:

/******************************************/
/* Ord. NoLoadLoop - Begin                */
/******************************************/

/* iter 0 (last unrolled loop) */

/* local read a */
s_waitcnt lgkmcnt(0)
s_waitcnt_depctr 0
ds_load_b64 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+1], v[vgprLocalReadAddrA+0] offset:2064 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+2:vgprValuA_X1_I0+2+1], v[vgprLocalReadAddrA+0] offset:2320 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+1], v[vgprLocalReadAddrA+0] offset:2576 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+6:vgprValuA_X1_I0+6+1], v[vgprLocalReadAddrA+0] offset:2832 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+1], v[vgprLocalReadAddrA+0] offset:3088 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+10:vgprValuA_X1_I0+10+1], v[vgprLocalReadAddrA+0] offset:3344 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+1], v[vgprLocalReadAddrA+0] offset:3600 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+14:vgprValuA_X1_I0+14+1], v[vgprLocalReadAddrA+0] offset:3856 // sync LDS0

/* local read b */
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB+0] offset:528 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB+0] offset:656 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrB+0] offset:784 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrB+0] offset:912 // sync LDS0

/* local read increment a */
/* N/A, lro->1032 */
/* localReadDoCntA 18 localReadDoCntMXSA 0 localReadDoCntB 18 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->264 */
/* localReadDoCntA 18 localReadDoCntMXSA 0 localReadDoCntB 18 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=0 newLR=0 for iteration == 0
s_add_u32 s[sgprAddressB], s[sgprAddressB], 16
s_addc_u32 s[sgprAddressB+1], s[sgprAddressB+1], 0
s_load_b128 s[72:75], s[sgprAddressB:sgprAddressB+1], 0
s_load_b128 s[76:79], s[sgprAddressB:sgprAddressB+1], s69
s_load_b128 s[80:83], s[sgprAddressB:sgprAddressB+1], s37
s_load_b128 s[84:87], s[sgprAddressB:sgprAddressB+1], s38
s_load_b128 s[88:91], s[sgprAddressB:sgprAddressB+1], s39
s_load_b128 s[92:95], s[sgprAddressB:sgprAddressB+1], s52
s_load_b128 s[96:99], s[sgprAddressB:sgprAddressB+1], s53
s_load_b128 s[100:103], s[sgprAddressB:sgprAddressB+1], s54
v_dual_fmac_f32 v[vgprValuC+0], s0, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s0, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s0, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s0, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s4, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s4, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s4, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s4, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s16, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s16, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s16, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s16, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s20, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s20, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s20, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s20, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s40, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s40, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s40, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s40, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s44, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s44, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s44, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s44, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s56, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s56, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s56, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s56, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s60, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s60, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s60, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s60, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s0, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s0, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s0, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s0, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s4, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s4, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s4, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s4, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s16, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s16, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s16, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s16, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s20, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s20, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s20, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s20, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s40, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s40, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s40, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s40, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s44, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s44, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s44, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s44, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s56, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s56, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s56, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s56, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s60, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s60, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s60, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s60, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s0, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s0, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s0, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s0, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s4, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s4, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s4, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s4, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s16, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s16, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s16, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s16, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s20, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s20, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s20, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s20, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s40, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s40, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s40, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s40, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s44, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s44, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s44, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s44, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s56, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s56, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s56, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s56, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s60, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s60, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s60, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s60, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s0, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s0, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s0, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s0, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s4, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s4, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s4, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s4, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s16, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s16, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s16, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s16, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s20, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s20, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s20, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s20, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s40, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s40, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s40, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s40, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s44, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s44, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s44, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s44, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s56, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s56, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s56, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s56, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s60, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s60, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s60, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s60, v[vgprValuA_X0_I0+15]


/* iter 1 (last unrolled loop) */

/* local read a */
s_waitcnt lgkmcnt(0)
s_waitcnt_depctr 0
ds_load_b64 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+1], v[vgprLocalReadAddrA+0] offset:4128 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+2:vgprValuA_X0_I0+2+1], v[vgprLocalReadAddrA+0] offset:4384 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+1], v[vgprLocalReadAddrA+0] offset:4640 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+6:vgprValuA_X0_I0+6+1], v[vgprLocalReadAddrA+0] offset:4896 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+1], v[vgprLocalReadAddrA+0] offset:5152 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+10:vgprValuA_X0_I0+10+1], v[vgprLocalReadAddrA+0] offset:5408 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+1], v[vgprLocalReadAddrA+0] offset:5664 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+14:vgprValuA_X0_I0+14+1], v[vgprLocalReadAddrA+0] offset:5920 // sync LDS0

/* local read b */
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB+0] offset:1056 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB+0] offset:1184 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrB+0] offset:1312 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrB+0] offset:1440 // sync LDS0

/* local read increment a */
/* N/A, lro->1548 */
/* localReadDoCntA 19 localReadDoCntMXSA 0 localReadDoCntB 19 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->396 */
/* localReadDoCntA 19 localReadDoCntMXSA 0 localReadDoCntB 19 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=0 newLR=0
v_dual_fmac_f32 v[vgprValuC+0], s1, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s1, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s1, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s1, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s5, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s5, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s5, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s5, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s17, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s17, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s17, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s17, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s21, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s21, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s21, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s21, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s41, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s41, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s41, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s41, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s45, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s45, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s45, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s45, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s57, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s57, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s57, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s57, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s61, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s61, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s61, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s61, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s1, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s1, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s1, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s1, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s5, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s5, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s5, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s5, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s17, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s17, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s17, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s17, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s21, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s21, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s21, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s21, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s41, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s41, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s41, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s41, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s45, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s45, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s45, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s45, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s57, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s57, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s57, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s57, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s61, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s61, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s61, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s61, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s1, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s1, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s1, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s1, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s5, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s5, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s5, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s5, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s17, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s17, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s17, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s17, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s21, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s21, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s21, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s21, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s41, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s41, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s41, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s41, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s45, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s45, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s45, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s45, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s57, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s57, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s57, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s57, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s61, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s61, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s61, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s61, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s1, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s1, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s1, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s1, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s5, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s5, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s5, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s5, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s17, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s17, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s17, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s17, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s21, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s21, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s21, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s21, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s41, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s41, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s41, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s41, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s45, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s45, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s45, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s45, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s57, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s57, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s57, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s57, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s61, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s61, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s61, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s61, v[vgprValuA_X1_I0+15]


/* iter 2 (last unrolled loop) */

/* local read a */
s_waitcnt lgkmcnt(0)
s_waitcnt_depctr 0
ds_load_b64 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+1], v[vgprLocalReadAddrA+0] offset:6192 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+2:vgprValuA_X1_I0+2+1], v[vgprLocalReadAddrA+0] offset:6448 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+1], v[vgprLocalReadAddrA+0] offset:6704 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+6:vgprValuA_X1_I0+6+1], v[vgprLocalReadAddrA+0] offset:6960 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+1], v[vgprLocalReadAddrA+0] offset:7216 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+10:vgprValuA_X1_I0+10+1], v[vgprLocalReadAddrA+0] offset:7472 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+1], v[vgprLocalReadAddrA+0] offset:7728 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+14:vgprValuA_X1_I0+14+1], v[vgprLocalReadAddrA+0] offset:7984 // sync LDS0

/* local read b */
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB+0] offset:1584 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB+0] offset:1712 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrB+0] offset:1840 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrB+0] offset:1968 // sync LDS0

/* local read increment a */
/* N/A, lro->2064 */
/* localReadDoCntA 20 localReadDoCntMXSA 0 localReadDoCntB 20 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->528 */
/* localReadDoCntA 20 localReadDoCntMXSA 0 localReadDoCntB 20 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=0 newLR=0
v_dual_fmac_f32 v[vgprValuC+0], s2, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s2, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s2, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s2, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s6, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s6, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s6, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s6, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s18, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s18, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s18, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s18, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s22, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s22, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s22, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s22, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s42, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s42, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s42, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s42, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s46, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s46, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s46, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s46, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s58, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s58, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s58, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s58, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s62, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s62, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s62, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s62, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s2, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s2, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s2, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s2, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s6, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s6, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s6, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s6, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s18, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s18, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s18, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s18, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s22, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s22, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s22, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s22, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s42, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s42, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s42, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s42, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s46, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s46, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s46, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s46, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s58, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s58, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s58, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s58, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s62, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s62, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s62, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s62, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s2, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s2, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s2, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s2, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s6, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s6, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s6, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s6, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s18, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s18, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s18, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s18, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s22, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s22, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s22, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s22, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s42, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s42, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s42, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s42, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s46, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s46, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s46, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s46, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s58, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s58, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s58, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s58, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s62, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s62, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s62, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s62, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s2, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s2, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s2, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s2, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s6, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s6, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s6, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s6, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s18, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s18, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s18, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s18, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s22, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s22, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s22, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s22, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s42, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s42, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s42, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s42, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s46, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s46, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s46, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s46, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s58, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s58, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s58, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s58, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s62, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s62, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s62, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s62, v[vgprValuA_X0_I0+15]


/* iter 3 (last unrolled loop) */

/* local read a */
s_waitcnt lgkmcnt(0)
s_waitcnt_depctr 0
ds_load_b64 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+1], v[vgprLocalReadAddrA+0] offset:8256 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+2:vgprValuA_X0_I0+2+1], v[vgprLocalReadAddrA+0] offset:8512 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+1], v[vgprLocalReadAddrA+0] offset:8768 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+6:vgprValuA_X0_I0+6+1], v[vgprLocalReadAddrA+0] offset:9024 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+1], v[vgprLocalReadAddrA+0] offset:9280 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+10:vgprValuA_X0_I0+10+1], v[vgprLocalReadAddrA+0] offset:9536 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+1], v[vgprLocalReadAddrA+0] offset:9792 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+14:vgprValuA_X0_I0+14+1], v[vgprLocalReadAddrA+0] offset:10048 // sync LDS0

/* local read b */
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB+0] offset:2112 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB+0] offset:2240 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrB+0] offset:2368 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrB+0] offset:2496 // sync LDS0

/* local read increment a */
/* N/A, lro->2580 */
/* localReadDoCntA 21 localReadDoCntMXSA 0 localReadDoCntB 21 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->660 */
/* localReadDoCntA 21 localReadDoCntMXSA 0 localReadDoCntB 21 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=0 newLR=0
v_dual_fmac_f32 v[vgprValuC+0], s3, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s3, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s3, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s3, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s7, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s7, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s7, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s7, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s19, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s19, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s19, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s19, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s23, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s23, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s23, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s23, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s43, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s43, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s43, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s43, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s47, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s47, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s47, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s47, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s59, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s59, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s59, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s59, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s63, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s63, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s63, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s63, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s3, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s3, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s3, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s3, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s7, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s7, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s7, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s7, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s19, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s19, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s19, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s19, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s23, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s23, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s23, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s23, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s43, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s43, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s43, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s43, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s47, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s47, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s47, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s47, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s59, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s59, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s59, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s59, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s63, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s63, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s63, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s63, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s3, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s3, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s3, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s3, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s7, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s7, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s7, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s7, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s19, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s19, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s19, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s19, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s23, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s23, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s23, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s23, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s43, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s43, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s43, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s43, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s47, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s47, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s47, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s47, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s59, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s59, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s59, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s59, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s63, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s63, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s63, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s63, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s3, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s3, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s3, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s3, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s7, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s7, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s7, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s7, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s19, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s19, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s19, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s19, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s23, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s23, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s23, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s23, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s43, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s43, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s43, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s43, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s47, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s47, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s47, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s47, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s59, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s59, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s59, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s59, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s63, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s63, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s63, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s63, v[vgprValuA_X1_I0+15]


/* iter 4 (last unrolled loop) */

/* local read a */
s_waitcnt lgkmcnt(0)
s_waitcnt_depctr 0
ds_load_b64 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+1], v[vgprLocalReadAddrA+0] offset:10320 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+2:vgprValuA_X1_I0+2+1], v[vgprLocalReadAddrA+0] offset:10576 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+1], v[vgprLocalReadAddrA+0] offset:10832 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+6:vgprValuA_X1_I0+6+1], v[vgprLocalReadAddrA+0] offset:11088 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+1], v[vgprLocalReadAddrA+0] offset:11344 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+10:vgprValuA_X1_I0+10+1], v[vgprLocalReadAddrA+0] offset:11600 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+1], v[vgprLocalReadAddrA+0] offset:11856 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+14:vgprValuA_X1_I0+14+1], v[vgprLocalReadAddrA+0] offset:12112 // sync LDS0

/* local read b */
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB+0] offset:2640 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB+0] offset:2768 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrB+0] offset:2896 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrB+0] offset:3024 // sync LDS0

/* local read increment a */
/* N/A, lro->3096 */
/* localReadDoCntA 22 localReadDoCntMXSA 0 localReadDoCntB 22 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->792 */
/* localReadDoCntA 22 localReadDoCntMXSA 0 localReadDoCntB 22 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=0 newLR=0
v_dual_fmac_f32 v[vgprValuC+0], s72, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s72, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s72, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s72, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s76, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s76, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s76, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s76, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s80, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s80, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s80, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s80, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s84, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s84, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s84, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s84, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s88, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s88, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s88, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s88, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s92, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s92, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s92, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s92, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s96, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s96, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s96, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s96, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s100, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s100, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s100, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s100, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s72, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s72, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s72, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s72, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s76, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s76, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s76, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s76, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s80, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s80, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s80, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s80, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s84, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s84, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s84, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s84, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s88, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s88, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s88, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s88, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s92, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s92, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s92, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s92, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s96, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s96, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s96, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s96, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s100, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s100, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s100, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s100, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s72, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s72, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s72, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s72, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s76, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s76, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s76, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s76, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s80, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s80, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s80, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s80, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s84, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s84, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s84, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s84, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s88, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s88, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s88, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s88, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s92, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s92, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s92, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s92, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s96, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s96, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s96, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s96, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s100, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s100, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s100, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s100, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s72, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s72, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s72, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s72, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s76, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s76, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s76, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s76, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s80, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s80, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s80, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s80, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s84, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s84, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s84, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s84, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s88, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s88, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s88, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s88, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s92, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s92, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s92, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s92, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s96, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s96, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s96, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s96, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s100, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s100, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s100, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s100, v[vgprValuA_X0_I0+15]


/* iter 5 (last unrolled loop) */

/* local read a */
s_waitcnt lgkmcnt(0)
s_waitcnt_depctr 0
ds_load_b64 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+1], v[vgprLocalReadAddrA+0] offset:12384 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+2:vgprValuA_X0_I0+2+1], v[vgprLocalReadAddrA+0] offset:12640 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+1], v[vgprLocalReadAddrA+0] offset:12896 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+6:vgprValuA_X0_I0+6+1], v[vgprLocalReadAddrA+0] offset:13152 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+1], v[vgprLocalReadAddrA+0] offset:13408 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+10:vgprValuA_X0_I0+10+1], v[vgprLocalReadAddrA+0] offset:13664 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+1], v[vgprLocalReadAddrA+0] offset:13920 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+14:vgprValuA_X0_I0+14+1], v[vgprLocalReadAddrA+0] offset:14176 // sync LDS0

/* local read b */
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB+0] offset:3168 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB+0] offset:3296 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrB+0] offset:3424 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrB+0] offset:3552 // sync LDS0

/* local read increment a */
/* N/A, lro->3612 */
/* localReadDoCntA 23 localReadDoCntMXSA 0 localReadDoCntB 23 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->924 */
/* localReadDoCntA 23 localReadDoCntMXSA 0 localReadDoCntB 23 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=0 newLR=0
v_dual_fmac_f32 v[vgprValuC+0], s73, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s73, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s73, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s73, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s77, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s77, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s77, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s77, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s81, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s81, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s81, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s81, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s85, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s85, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s85, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s85, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s89, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s89, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s89, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s89, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s93, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s93, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s93, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s93, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s97, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s97, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s97, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s97, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s101, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s101, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s101, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s101, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s73, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s73, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s73, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s73, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s77, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s77, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s77, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s77, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s81, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s81, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s81, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s81, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s85, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s85, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s85, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s85, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s89, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s89, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s89, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s89, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s93, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s93, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s93, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s93, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s97, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s97, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s97, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s97, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s101, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s101, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s101, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s101, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s73, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s73, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s73, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s73, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s77, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s77, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s77, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s77, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s81, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s81, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s81, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s81, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s85, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s85, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s85, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s85, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s89, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s89, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s89, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s89, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s93, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s93, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s93, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s93, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s97, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s97, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s97, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s97, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s101, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s101, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s101, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s101, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s73, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s73, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s73, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s73, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s77, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s77, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s77, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s77, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s81, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s81, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s81, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s81, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s85, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s85, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s85, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s85, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s89, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s89, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s89, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s89, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s93, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s93, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s93, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s93, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s97, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s97, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s97, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s97, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s101, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s101, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s101, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s101, v[vgprValuA_X1_I0+15]


/* iter 6 (last unrolled loop) */

/* local read a */
s_waitcnt lgkmcnt(0)
s_waitcnt_depctr 0
ds_load_b64 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+1], v[vgprLocalReadAddrA+0] offset:14448 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+2:vgprValuA_X1_I0+2+1], v[vgprLocalReadAddrA+0] offset:14704 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+1], v[vgprLocalReadAddrA+0] offset:14960 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+6:vgprValuA_X1_I0+6+1], v[vgprLocalReadAddrA+0] offset:15216 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+1], v[vgprLocalReadAddrA+0] offset:15472 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+10:vgprValuA_X1_I0+10+1], v[vgprLocalReadAddrA+0] offset:15728 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+1], v[vgprLocalReadAddrA+0] offset:15984 // sync LDS0
ds_load_b64 v[vgprValuA_X1_I0+14:vgprValuA_X1_I0+14+1], v[vgprLocalReadAddrA+0] offset:16240 // sync LDS0

/* local read b */
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB+0] offset:3696 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB+0] offset:3824 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrB+0] offset:3952 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrB+0] offset:4080 // sync LDS0
s_waitcnt vmcnt(0)                                 // 1wait for global read

/* local write A */

/* local write MXSA */

/* local write MXSB */

/* local write B */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=0 newLR=0
v_dual_fmac_f32 v[vgprValuC+0], s74, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s74, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s74, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s74, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s78, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s78, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s78, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s78, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s82, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s82, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s82, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s82, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s86, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s86, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s86, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s86, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s90, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s90, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s90, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s90, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s94, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s94, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s94, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s94, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s98, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s98, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s98, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s98, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s102, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s102, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s102, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s102, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s74, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s74, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s74, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s74, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s78, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s78, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s78, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s78, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s82, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s82, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s82, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s82, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s86, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s86, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s86, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s86, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s90, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s90, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s90, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s90, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s94, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s94, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s94, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s94, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s98, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s98, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s98, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s98, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s102, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s102, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s102, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s102, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s74, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s74, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s74, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s74, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s78, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s78, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s78, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s78, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s82, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s82, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s82, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s82, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s86, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s86, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s86, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s86, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s90, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s90, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s90, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s90, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s94, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s94, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s94, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s94, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s98, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s98, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s98, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s98, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s102, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s102, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s102, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s102, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s74, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s74, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s74, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s74, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s78, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s78, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s78, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s78, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s82, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s82, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s82, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s82, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s86, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s86, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s86, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s86, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s90, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s90, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s90, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s90, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s94, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s94, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s94, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s94, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s98, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s98, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s98, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s98, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s102, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s102, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s102, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s102, v[vgprValuA_X0_I0+15]


/* iter 7 (last unrolled loop) */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->4128 */
/* localReadDoCntA 25 localReadDoCntMXSA 0 localReadDoCntB 25 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->1056 */
/* localReadDoCntA 25 localReadDoCntMXSA 0 localReadDoCntB 25 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
s_waitcnt lgkmcnt(0)
s_waitcnt_depctr 0
v_dual_fmac_f32 v[vgprValuC+0], s75, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s75, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s75, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s75, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s79, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s79, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s79, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s79, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s83, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s83, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s83, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s83, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s87, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s87, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s87, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s87, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s91, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s91, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s91, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s91, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s95, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s95, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s95, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s95, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s99, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s99, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s99, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s99, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s103, v[vgprValuA_X1_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s103, v[vgprValuA_X1_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s103, v[vgprValuA_X1_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s103, v[vgprValuA_X1_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s75, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s75, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s75, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s75, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s79, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s79, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s79, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s79, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s83, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s83, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s83, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s83, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s87, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s87, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s87, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s87, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s91, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s91, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s91, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s91, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s95, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s95, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s95, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s95, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s99, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s99, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s99, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s99, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s103, v[vgprValuA_X1_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s103, v[vgprValuA_X1_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s103, v[vgprValuA_X1_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s103, v[vgprValuA_X1_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s75, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s75, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s75, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s75, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s79, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s79, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s79, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s79, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s83, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s83, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s83, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s83, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s87, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s87, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s87, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s87, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s91, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s91, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s91, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s91, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s95, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s95, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s95, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s95, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s99, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s99, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s99, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s99, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s103, v[vgprValuA_X1_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s103, v[vgprValuA_X1_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s103, v[vgprValuA_X1_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s103, v[vgprValuA_X1_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s75, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s75, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s75, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s75, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s79, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s79, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s79, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s79, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s83, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s83, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s83, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s83, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s87, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s87, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s87, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s87, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s91, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s91, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s91, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s91, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s95, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s95, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s95, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s95, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s99, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s99, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s99, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s99, v[vgprValuA_X1_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s103, v[vgprValuA_X1_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s103, v[vgprValuA_X1_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s103, v[vgprValuA_X1_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s103, v[vgprValuA_X1_I0+15]

label_toPGR1end_OrdNLL:
label_PrefetchGlobalLastIterEnd:

/* Tail: add ValuA/B vgpr buffer [136...185) to pool */

/* Tail: add address/G2L vgpr [185...196) to pool */

/******************************************/
/* Tail Loop                              */
/******************************************/

/* local write reset offsets a */
v_and_b32 v[vgprLocalWriteAddrA], 0xf07fff, v[vgprLocalWriteAddrA] // reset to Red

/* local write reset offsets b */
v_and_b32 v[vgprLocalWriteAddrB], 0xf07fff, v[vgprLocalWriteAddrB] // reset to Red
/* Check out VGPR (numG2LA,numG2LB,numG2LMXSA,numG2LMXSB,numG2LMetadata) = (8,2,0,0,0) */
.set vgprG2LA_BASE, 136
.set vgprG2LA, vgprG2LA_BASE+0
.set vgprG2LB_BASE, 144
.set vgprG2LB, vgprG2LB_BASE+0

// numIterL = LOCAL_SPLITU * min(sizeL % LOCAL_DEPTHU, DEPTHU / LOCAL_SPLITU)
s_and_b32 s[sgprLoopCounterL], 7, s[sgprSizesSum+0] // s[sgprLoopCounterL] = s[sgprSizesSum+0] % 8
s_and_b32 s70, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_TL                       // branch if GSUC == 1
s_cmp_lg_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx == numIterPerWgRemainder
s_cmov_b32 s[sgprLoopCounterL], 0                  // numIter=0 if gsuSimIdx != numIterPerWgRemainder
s_branch label_GSUC_TL_End
label_GSUC_TL:
s_lshr_b32 s71, s[sgprSizesSum], 3                 // s71 = s[sgprSizesSum] / 8
s_and_b32 s72, s[sgprGSU], 0x3fff                  // Restore GSU
v_cvt_f32_u32 v146, s72                            // s70 = s71 / s72
v_rcp_iflag_f32 v146, v146                         // s70 = s71 / s72
v_cvt_f32_u32 v147, s71                            // s70 = s71 / s72
v_mul_f32 v146, v146, v147                         // s70 = s71 / s72
v_cvt_u32_f32 v146, v146                           // s70 = s71 / s72
v_mul_u32_u24 v147, v146, s72                      // s70 = s71 / s72
v_sub_nc_u32 v147, s71, v147                       // s70 = s71 / s72
v_cmp_eq_u32 vcc_lo, v147, s72                     // s70 = s71 / s72
s_mov_b32 exec_lo, vcc_lo                          // s70 = s71 / s72
v_add_nc_u32 v146, 1, v146                         // s70 = s71 / s72
v_mov_b32 v147, 0                                  // s[sgprGSUSumIdx+1] = s71 % s72
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v147, s72                     // overflow happened in remainder
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v146, v146, 1                         // quotient - 1
v_mul_u32_u24 v147, v146, s72                      // re-calculate remainder
v_sub_nc_u32 v147, s71, v147                       // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s70, v146                      // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v147       // remainder
s_sub_u32 s71, s72, 1                              // GSU-1
s_cmp_eq_u32 s70, 0                                // quotient == 0
s_cselect_b32 s70, s[sgprGSUSumIdx+1], s71         // lastWg = (quotient==0) ? numIterPerWgRemainder : GSU-1
s_cmp_lg_u32 s[sgprGSUSumIdx], s70                 // gsuSumIdx == lastWg
s_cmov_b32 s[sgprLoopCounterL], 0                  // numIter=0 if gsuSumIdx != lastWg
label_GSUC_TL_End:
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // numIterL == 0
s_mov_b32 s[sgprOrigLoopCounter], 0                // repurpose to count each localRead increment
s_cbranch_scc1 label_SkipTailLoopL                 // skip to end of tail loop b/c numIter==0

/* remove stagger offsets for tail loop */
//  removeStagger A
s_sub_i32 s70, 3, s[sgprStaggerUIter]
s_cmp_ge_i32 s70, 0
s_cbranch_scc0 label_Negative_2
s_mul_hi_u32 s71, s70, s[sgprGlobalReadIncsA+0]    // start offset S in bytes
s_mul_i32 s70, s70, s[sgprGlobalReadIncsA+0]       // start offset S in bytes
s_branch label_MultiplyDone_3
label_Negative_2:
s_abs_i32 s70, s70
s_mul_hi_u32 s71, s70, s[sgprGlobalReadIncsA+0]    // start offset S in bytes
s_mul_i32 s70, s70, s[sgprGlobalReadIncsA+0]       // start offset S in bytes
s_xor_b32 s70, s70, 0xffffffff
s_xor_b32 s71, s71, 0xffffffff
s_add_u32 s70, s70, 0x1
s_addc_u32 s71, s71, 0
label_MultiplyDone_3:
s_sub_u32 s70, s70, s[sgprWrapUA]                  // S - WrapU
s_subb_u32 s71, s71, s[sgprWrapUA+1]               // S - WrapU
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s70        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s71       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s70 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s71 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
//  removeStagger B
s_sub_i32 s70, 3, s[sgprStaggerUIter]
s_cmp_ge_i32 s70, 0
s_cbranch_scc0 label_Negative_4
s_mul_hi_u32 s71, s70, s[sgprGlobalReadIncsB+0]    // start offset S in bytes
s_mul_i32 s70, s70, s[sgprGlobalReadIncsB+0]       // start offset S in bytes
s_branch label_MultiplyDone_5
label_Negative_4:
s_abs_i32 s70, s70
s_mul_hi_u32 s71, s70, s[sgprGlobalReadIncsB+0]    // start offset S in bytes
s_mul_i32 s70, s70, s[sgprGlobalReadIncsB+0]       // start offset S in bytes
s_xor_b32 s70, s70, 0xffffffff
s_xor_b32 s71, s71, 0xffffffff
s_add_u32 s70, s70, 0x1
s_addc_u32 s71, s71, 0
label_MultiplyDone_5:
s_sub_u32 s70, s70, s[sgprWrapUB]                  // S - WrapU
s_subb_u32 s71, s71, s[sgprWrapUB+1]               // S - WrapU
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s70        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s71       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s70 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s71 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32

/* Update M0 for DTLDS */

/* Tail global read A */
/* g2l=0, load component 0 */
buffer_load_b64 v[vgprG2LA+0+0:vgprG2LA+0+0+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // load one buffer value
/* g2l=2, load component 0 */
buffer_load_b64 v[vgprG2LA+2+0:vgprG2LA+2+0+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:0 // load one buffer value
/* g2l=4, load component 0 */
buffer_load_b64 v[vgprG2LA+4+0:vgprG2LA+4+0+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:0 // load one buffer value
/* g2l=6, load component 0 */
buffer_load_b64 v[vgprG2LA+6+0:vgprG2LA+6+0+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:0 // load one buffer value

/* Update M0 for DTLDS */

/* Tail global read B */
/* g2l=0, load component 0 */
// SCALAR_GLOBAL_B: buffer_load_b64 v[vgprG2LB+0+0:vgprG2LB+0+0+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)                                 // 2wait for global read
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier

/* local write a */
ds_store_b32 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0] offset:0 // lwoA_0_0_0_0 = (0 + 0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
ds_store_b32 v[vgprLocalWriteAddrA+0], v[vgprG2LA+1] offset:2064 // lwoA_0_1_0_0 = (1 + 0*LSCA)*(MT0I+PAD) + (0*LSPA) = 2064 sync LDS0
ds_store_b32 v[vgprLocalWriteAddrA+0], v[vgprG2LA+2] offset:512 // lwoA_0_0_1_0 = (0 + 0*LSCA)*(MT0I+PAD) + (1*LSPA) = 512 sync LDS0
ds_store_b32 v[vgprLocalWriteAddrA+0], v[vgprG2LA+3] offset:2576 // lwoA_0_1_1_0 = (1 + 0*LSCA)*(MT0I+PAD) + (1*LSPA) = 2576 sync LDS0
ds_store_b32 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4] offset:1024 // lwoA_0_0_2_0 = (0 + 0*LSCA)*(MT0I+PAD) + (2*LSPA) = 1024 sync LDS0
ds_store_b32 v[vgprLocalWriteAddrA+0], v[vgprG2LA+5] offset:3088 // lwoA_0_1_2_0 = (1 + 0*LSCA)*(MT0I+PAD) + (2*LSPA) = 3088 sync LDS0
ds_store_b32 v[vgprLocalWriteAddrA+0], v[vgprG2LA+6] offset:1536 // lwoA_0_0_3_0 = (0 + 0*LSCA)*(MT0I+PAD) + (3*LSPA) = 1536 sync LDS0
ds_store_b32 v[vgprLocalWriteAddrA+0], v[vgprG2LA+7] offset:3600 // lwoA_0_1_3_0 = (1 + 0*LSCA)*(MT0I+PAD) + (3*LSPA) = 3600 sync LDS0

/* local write b */
// SCALAR_GLOBAL_B: ds_store_b32 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0] offset:0 // lwoB_0_0_0_0 = (0 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0
// SCALAR_GLOBAL_B: ds_store_b32 v[vgprLocalWriteAddrB+0], v[vgprG2LB+1] offset:528 // lwoB_0_1_0_0 = (1 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 528 sync LDS0

/* Recalc local read offsets */
s_waitcnt lgkmcnt(0)                               // 5wait for local write
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier                                          // Tail loop LW->LR, sync LDS0
.set vgprG2LA_BASE, UNDEF
.set vgprG2LA, UNDEF
.set vgprG2LB_BASE, UNDEF
.set vgprG2LB, UNDEF
.set vgprValuA_X0_I0_BASE, 136
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+16
.set vgprValuB_X0_I0_BASE, 168
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X1_I0, vgprValuB_X0_I0_BASE+0

/* Tail: local read reset offsets a */

/* localReadResetOffsets */
/* handled internally */
v_and_b32 v[vgprLocalReadAddrA+0], 0x7fff, v[vgprLocalReadAddrA+0] // reset Red,Blk -> Red

/* Tail: local read reset offsets b */

/* localReadResetOffsets */
/* handled internally */
v_and_b32 v[vgprLocalReadAddrB+0], 0x7fff, v[vgprLocalReadAddrB+0] // reset Red,Blk -> Red

/* Tail: local read init pointers a */

/* localReadInitPointers */

/* Tail: local read init pointers b */

/* localReadInitPointers */

/* tail loop: macs */
.align 16
label_TailLoopBeginL:

/* local read a */
ds_load_b64 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+1], v[vgprLocalReadAddrA+0] offset:0 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+2:vgprValuA_X0_I0+2+1], v[vgprLocalReadAddrA+0] offset:256 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+1], v[vgprLocalReadAddrA+0] offset:512 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+6:vgprValuA_X0_I0+6+1], v[vgprLocalReadAddrA+0] offset:768 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+1], v[vgprLocalReadAddrA+0] offset:1024 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+10:vgprValuA_X0_I0+10+1], v[vgprLocalReadAddrA+0] offset:1280 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+1], v[vgprLocalReadAddrA+0] offset:1536 // sync LDS0
ds_load_b64 v[vgprValuA_X0_I0+14:vgprValuA_X0_I0+14+1], v[vgprLocalReadAddrA+0] offset:1792 // sync LDS0

/* local read b */
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB+0] offset:0 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB+0] offset:128 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrB+0] offset:256 // sync LDS0
// SCALAR_GLOBAL_B: ds_load_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrB+0] offset:384 // sync LDS0

/* local read inc a */
s_mov_b32 s8, 2064                                 // inc
v_add_co_u32 v[vgprLocalReadAddrA+0], vcc_lo, s8, v[vgprLocalReadAddrA+0] // lrA += 2064 ((MT+PAD)*bpeDS)

/* local read inc b */
s_mov_b32 s8, 528                                  // inc
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, s8, v[vgprLocalReadAddrB+0] // lrB += 528 ((MT+PAD)*bpeDS)
s_waitcnt lgkmcnt(0)                               // 4wait for local read
s_waitcnt lgkmcnt(0)
s_waitcnt lgkmcnt(0)
s_sub_u32 s[sgprStrideBK], s[sgprStrideBK], 1
s_cmp_eq_u32 s[sgprStrideBK], 0
s_cbranch_scc1 .Llarge_scalar_prefetch_done_32
s_add_u32 s[sgprAddressB], s[sgprAddressB], s[sgprStrideB1J]
s_addc_u32 s[sgprAddressB+1], s[sgprAddressB+1], 0
s_load_b256 s[88:95], s[sgprAddressB:sgprAddressB+1], 0
.Llarge_scalar_prefetch_done_32:
s_waitcnt_depctr 0
v_dual_fmac_f32 v[vgprValuC+0], s73, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+1], s73, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+2], s73, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+3], s73, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+16], s77, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+17], s77, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+18], s77, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+19], s77, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+32], s81, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+33], s81, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+34], s81, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+35], s81, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+48], s85, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+49], s85, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+50], s85, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+51], s85, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+64], s89, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+65], s89, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+66], s89, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+67], s89, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+80], s93, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+81], s93, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+82], s93, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+83], s93, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+96], s97, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+97], s97, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+98], s97, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+99], s97, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+112], s101, v[vgprValuA_X0_I0+0] :: v_dual_fmac_f32 v[vgprValuC+113], s101, v[vgprValuA_X0_I0+1]
v_dual_fmac_f32 v[vgprValuC+114], s101, v[vgprValuA_X0_I0+2] :: v_dual_fmac_f32 v[vgprValuC+115], s101, v[vgprValuA_X0_I0+3]
v_dual_fmac_f32 v[vgprValuC+4], s73, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+5], s73, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+6], s73, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+7], s73, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+20], s77, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+21], s77, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+22], s77, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+23], s77, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+36], s81, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+37], s81, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+38], s81, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+39], s81, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+52], s85, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+53], s85, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+54], s85, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+55], s85, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+68], s89, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+69], s89, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+70], s89, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+71], s89, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+84], s93, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+85], s93, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+86], s93, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+87], s93, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+100], s97, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+101], s97, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+102], s97, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+103], s97, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+116], s101, v[vgprValuA_X0_I0+4] :: v_dual_fmac_f32 v[vgprValuC+117], s101, v[vgprValuA_X0_I0+5]
v_dual_fmac_f32 v[vgprValuC+118], s101, v[vgprValuA_X0_I0+6] :: v_dual_fmac_f32 v[vgprValuC+119], s101, v[vgprValuA_X0_I0+7]
v_dual_fmac_f32 v[vgprValuC+8], s73, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+9], s73, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+10], s73, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+11], s73, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+24], s77, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+25], s77, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+26], s77, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+27], s77, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+40], s81, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+41], s81, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+42], s81, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+43], s81, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+56], s85, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+57], s85, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+58], s85, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+59], s85, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+72], s89, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+73], s89, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+74], s89, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+75], s89, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+88], s93, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+89], s93, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+90], s93, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+91], s93, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+104], s97, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+105], s97, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+106], s97, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+107], s97, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+120], s101, v[vgprValuA_X0_I0+8] :: v_dual_fmac_f32 v[vgprValuC+121], s101, v[vgprValuA_X0_I0+9]
v_dual_fmac_f32 v[vgprValuC+122], s101, v[vgprValuA_X0_I0+10] :: v_dual_fmac_f32 v[vgprValuC+123], s101, v[vgprValuA_X0_I0+11]
v_dual_fmac_f32 v[vgprValuC+12], s73, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+13], s73, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+14], s73, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+15], s73, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+28], s77, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+29], s77, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+30], s77, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+31], s77, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+44], s81, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+45], s81, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+46], s81, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+47], s81, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+60], s85, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+61], s85, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+62], s85, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+63], s85, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+76], s89, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+77], s89, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+78], s89, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+79], s89, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+92], s93, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+93], s93, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+94], s93, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+95], s93, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+108], s97, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+109], s97, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+110], s97, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+111], s97, v[vgprValuA_X0_I0+15]
v_dual_fmac_f32 v[vgprValuC+124], s101, v[vgprValuA_X0_I0+12] :: v_dual_fmac_f32 v[vgprValuC+125], s101, v[vgprValuA_X0_I0+13]
v_dual_fmac_f32 v[vgprValuC+126], s101, v[vgprValuA_X0_I0+14] :: v_dual_fmac_f32 v[vgprValuC+127], s101, v[vgprValuA_X0_I0+15]


/* closeLoop loopL finalLoop=1 tailLoop=1 */
s_sub_i32 s[sgprLoopCounterL], s[sgprLoopCounterL], 0x1 // dec counterL (tailLoop)
s_add_u32 s[sgprOrigLoopCounter], s[sgprOrigLoopCounter], 0x1 // inc counterL
s_cmp_le_i32 s[sgprLoopCounterL], 0x0              // counterL<=0
s_cbranch_scc0 label_TailLoopBeginL                // restart LoopL
label_TailLoopEndL:
label_SkipTailLoopL:
.set vgprValuA_X0_I0_BASE, UNDEF
.set vgprValuA_X0_I0, UNDEF
.set vgprValuA_X1_I0, UNDEF
.set vgprValuB_X0_I0_BASE, UNDEF
.set vgprValuB_X0_I0, UNDEF
.set vgprValuB_X1_I0, UNDEF

/* Tail: add MISC Vgpr [128...136) to pool */
label_Summation_End_6:
// D = alpha * acc + beta * C
s_load_b64 s[76:77], s[104:105], 96                 // alpha, beta
s_load_b64 s[78:79], s[104:105], 40                 // C
s_load_b32 s80, s[104:105], 72                    // ldc
s_waitcnt lgkmcnt(0)
v_cmp_eq_f32 vcc_lo, 0, s76
s_cbranch_vccz label_AlphaScale
v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, 0
v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0
v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v5, 0
v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v7, 0
v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v9, 0
v_dual_mov_b32 v10, 0 :: v_dual_mov_b32 v11, 0
v_dual_mov_b32 v12, 0 :: v_dual_mov_b32 v13, 0
v_dual_mov_b32 v14, 0 :: v_dual_mov_b32 v15, 0
v_dual_mov_b32 v16, 0 :: v_dual_mov_b32 v17, 0
v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v19, 0
v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v21, 0
v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v23, 0
v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v25, 0
v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v27, 0
v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v29, 0
v_dual_mov_b32 v30, 0 :: v_dual_mov_b32 v31, 0
v_dual_mov_b32 v32, 0 :: v_dual_mov_b32 v33, 0
v_dual_mov_b32 v34, 0 :: v_dual_mov_b32 v35, 0
v_dual_mov_b32 v36, 0 :: v_dual_mov_b32 v37, 0
v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v39, 0
v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v41, 0
v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v43, 0
v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v45, 0
v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v47, 0
v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v49, 0
v_dual_mov_b32 v50, 0 :: v_dual_mov_b32 v51, 0
v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v53, 0
v_dual_mov_b32 v54, 0 :: v_dual_mov_b32 v55, 0
v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v57, 0
v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v59, 0
v_dual_mov_b32 v60, 0 :: v_dual_mov_b32 v61, 0
v_dual_mov_b32 v62, 0 :: v_dual_mov_b32 v63, 0
v_dual_mov_b32 v64, 0 :: v_dual_mov_b32 v65, 0
v_dual_mov_b32 v66, 0 :: v_dual_mov_b32 v67, 0
v_dual_mov_b32 v68, 0 :: v_dual_mov_b32 v69, 0
v_dual_mov_b32 v70, 0 :: v_dual_mov_b32 v71, 0
v_dual_mov_b32 v72, 0 :: v_dual_mov_b32 v73, 0
v_dual_mov_b32 v74, 0 :: v_dual_mov_b32 v75, 0
v_dual_mov_b32 v76, 0 :: v_dual_mov_b32 v77, 0
v_dual_mov_b32 v78, 0 :: v_dual_mov_b32 v79, 0
v_dual_mov_b32 v80, 0 :: v_dual_mov_b32 v81, 0
v_dual_mov_b32 v82, 0 :: v_dual_mov_b32 v83, 0
v_dual_mov_b32 v84, 0 :: v_dual_mov_b32 v85, 0
v_dual_mov_b32 v86, 0 :: v_dual_mov_b32 v87, 0
v_dual_mov_b32 v88, 0 :: v_dual_mov_b32 v89, 0
v_dual_mov_b32 v90, 0 :: v_dual_mov_b32 v91, 0
v_dual_mov_b32 v92, 0 :: v_dual_mov_b32 v93, 0
v_dual_mov_b32 v94, 0 :: v_dual_mov_b32 v95, 0
v_dual_mov_b32 v96, 0 :: v_dual_mov_b32 v97, 0
v_dual_mov_b32 v98, 0 :: v_dual_mov_b32 v99, 0
v_dual_mov_b32 v100, 0 :: v_dual_mov_b32 v101, 0
v_dual_mov_b32 v102, 0 :: v_dual_mov_b32 v103, 0
v_dual_mov_b32 v104, 0 :: v_dual_mov_b32 v105, 0
v_dual_mov_b32 v106, 0 :: v_dual_mov_b32 v107, 0
v_dual_mov_b32 v108, 0 :: v_dual_mov_b32 v109, 0
v_dual_mov_b32 v110, 0 :: v_dual_mov_b32 v111, 0
v_dual_mov_b32 v112, 0 :: v_dual_mov_b32 v113, 0
v_dual_mov_b32 v114, 0 :: v_dual_mov_b32 v115, 0
v_dual_mov_b32 v116, 0 :: v_dual_mov_b32 v117, 0
v_dual_mov_b32 v118, 0 :: v_dual_mov_b32 v119, 0
v_dual_mov_b32 v120, 0 :: v_dual_mov_b32 v121, 0
v_dual_mov_b32 v122, 0 :: v_dual_mov_b32 v123, 0
v_dual_mov_b32 v124, 0 :: v_dual_mov_b32 v125, 0
v_dual_mov_b32 v126, 0 :: v_dual_mov_b32 v127, 0
s_branch label_AlphaDone
label_AlphaScale:
v_dual_mul_f32 v0, s76, v0 :: v_dual_mul_f32 v1, s76, v1
v_dual_mul_f32 v2, s76, v2 :: v_dual_mul_f32 v3, s76, v3
v_dual_mul_f32 v4, s76, v4 :: v_dual_mul_f32 v5, s76, v5
v_dual_mul_f32 v6, s76, v6 :: v_dual_mul_f32 v7, s76, v7
v_dual_mul_f32 v8, s76, v8 :: v_dual_mul_f32 v9, s76, v9
v_dual_mul_f32 v10, s76, v10 :: v_dual_mul_f32 v11, s76, v11
v_dual_mul_f32 v12, s76, v12 :: v_dual_mul_f32 v13, s76, v13
v_dual_mul_f32 v14, s76, v14 :: v_dual_mul_f32 v15, s76, v15
v_dual_mul_f32 v16, s76, v16 :: v_dual_mul_f32 v17, s76, v17
v_dual_mul_f32 v18, s76, v18 :: v_dual_mul_f32 v19, s76, v19
v_dual_mul_f32 v20, s76, v20 :: v_dual_mul_f32 v21, s76, v21
v_dual_mul_f32 v22, s76, v22 :: v_dual_mul_f32 v23, s76, v23
v_dual_mul_f32 v24, s76, v24 :: v_dual_mul_f32 v25, s76, v25
v_dual_mul_f32 v26, s76, v26 :: v_dual_mul_f32 v27, s76, v27
v_dual_mul_f32 v28, s76, v28 :: v_dual_mul_f32 v29, s76, v29
v_dual_mul_f32 v30, s76, v30 :: v_dual_mul_f32 v31, s76, v31
v_dual_mul_f32 v32, s76, v32 :: v_dual_mul_f32 v33, s76, v33
v_dual_mul_f32 v34, s76, v34 :: v_dual_mul_f32 v35, s76, v35
v_dual_mul_f32 v36, s76, v36 :: v_dual_mul_f32 v37, s76, v37
v_dual_mul_f32 v38, s76, v38 :: v_dual_mul_f32 v39, s76, v39
v_dual_mul_f32 v40, s76, v40 :: v_dual_mul_f32 v41, s76, v41
v_dual_mul_f32 v42, s76, v42 :: v_dual_mul_f32 v43, s76, v43
v_dual_mul_f32 v44, s76, v44 :: v_dual_mul_f32 v45, s76, v45
v_dual_mul_f32 v46, s76, v46 :: v_dual_mul_f32 v47, s76, v47
v_dual_mul_f32 v48, s76, v48 :: v_dual_mul_f32 v49, s76, v49
v_dual_mul_f32 v50, s76, v50 :: v_dual_mul_f32 v51, s76, v51
v_dual_mul_f32 v52, s76, v52 :: v_dual_mul_f32 v53, s76, v53
v_dual_mul_f32 v54, s76, v54 :: v_dual_mul_f32 v55, s76, v55
v_dual_mul_f32 v56, s76, v56 :: v_dual_mul_f32 v57, s76, v57
v_dual_mul_f32 v58, s76, v58 :: v_dual_mul_f32 v59, s76, v59
v_dual_mul_f32 v60, s76, v60 :: v_dual_mul_f32 v61, s76, v61
v_dual_mul_f32 v62, s76, v62 :: v_dual_mul_f32 v63, s76, v63
v_dual_mul_f32 v64, s76, v64 :: v_dual_mul_f32 v65, s76, v65
v_dual_mul_f32 v66, s76, v66 :: v_dual_mul_f32 v67, s76, v67
v_dual_mul_f32 v68, s76, v68 :: v_dual_mul_f32 v69, s76, v69
v_dual_mul_f32 v70, s76, v70 :: v_dual_mul_f32 v71, s76, v71
v_dual_mul_f32 v72, s76, v72 :: v_dual_mul_f32 v73, s76, v73
v_dual_mul_f32 v74, s76, v74 :: v_dual_mul_f32 v75, s76, v75
v_dual_mul_f32 v76, s76, v76 :: v_dual_mul_f32 v77, s76, v77
v_dual_mul_f32 v78, s76, v78 :: v_dual_mul_f32 v79, s76, v79
v_dual_mul_f32 v80, s76, v80 :: v_dual_mul_f32 v81, s76, v81
v_dual_mul_f32 v82, s76, v82 :: v_dual_mul_f32 v83, s76, v83
v_dual_mul_f32 v84, s76, v84 :: v_dual_mul_f32 v85, s76, v85
v_dual_mul_f32 v86, s76, v86 :: v_dual_mul_f32 v87, s76, v87
v_dual_mul_f32 v88, s76, v88 :: v_dual_mul_f32 v89, s76, v89
v_dual_mul_f32 v90, s76, v90 :: v_dual_mul_f32 v91, s76, v91
v_dual_mul_f32 v92, s76, v92 :: v_dual_mul_f32 v93, s76, v93
v_dual_mul_f32 v94, s76, v94 :: v_dual_mul_f32 v95, s76, v95
v_dual_mul_f32 v96, s76, v96 :: v_dual_mul_f32 v97, s76, v97
v_dual_mul_f32 v98, s76, v98 :: v_dual_mul_f32 v99, s76, v99
v_dual_mul_f32 v100, s76, v100 :: v_dual_mul_f32 v101, s76, v101
v_dual_mul_f32 v102, s76, v102 :: v_dual_mul_f32 v103, s76, v103
v_dual_mul_f32 v104, s76, v104 :: v_dual_mul_f32 v105, s76, v105
v_dual_mul_f32 v106, s76, v106 :: v_dual_mul_f32 v107, s76, v107
v_dual_mul_f32 v108, s76, v108 :: v_dual_mul_f32 v109, s76, v109
v_dual_mul_f32 v110, s76, v110 :: v_dual_mul_f32 v111, s76, v111
v_dual_mul_f32 v112, s76, v112 :: v_dual_mul_f32 v113, s76, v113
v_dual_mul_f32 v114, s76, v114 :: v_dual_mul_f32 v115, s76, v115
v_dual_mul_f32 v116, s76, v116 :: v_dual_mul_f32 v117, s76, v117
v_dual_mul_f32 v118, s76, v118 :: v_dual_mul_f32 v119, s76, v119
v_dual_mul_f32 v120, s76, v120 :: v_dual_mul_f32 v121, s76, v121
v_dual_mul_f32 v122, s76, v122 :: v_dual_mul_f32 v123, s76, v123
v_dual_mul_f32 v124, s76, v124 :: v_dual_mul_f32 v125, s76, v125
v_dual_mul_f32 v126, s76, v126 :: v_dual_mul_f32 v127, s76, v127
label_AlphaDone:
v_cmp_eq_f32 vcc_lo, 0, s77
s_cbranch_vccnz label_BetaDone
v_and_b32 v128, 31, v[vgprSerial]
v_lshlrev_b32 v128, 1, v128
s_lshl_b32 s72, s[sgprWorkGroup0], 9
v_add_nc_u32 v128, s72, v128                       // row = wg0*512 + lane*2
v_lshrrev_b32 v129, 5, v[vgprSerial]
v_lshlrev_b32 v129, 3, v129
s_lshl_b32 s72, s[sgprWorkGroup1], 7
v_add_nc_u32 v129, s72, v129                       // col = wg1*128 + wave*8
v_mul_lo_u32 v129, s80, v129
v_add_nc_u32 v128, v128, v129
v_lshlrev_b32 v128, 2, v128
v_add_co_u32 v132, vcc_lo, s78, v128
v_add_co_ci_u32 v133, vcc_lo, s79, 0, vcc_lo
s_lshl_b32 s81, s80, 2                             // ldc bytes
global_load_b64 v[134:135], v[132:133], off offset:0
global_load_b64 v[136:137], v[132:133], off offset:256
global_load_b64 v[138:139], v[132:133], off offset:512
global_load_b64 v[140:141], v[132:133], off offset:768
global_load_b64 v[142:143], v[132:133], off offset:1024
global_load_b64 v[144:145], v[132:133], off offset:1280
global_load_b64 v[146:147], v[132:133], off offset:1536
global_load_b64 v[148:149], v[132:133], off offset:1792
v_add_co_u32 v132, vcc_lo, s81, v132
v_add_co_ci_u32 v133, vcc_lo, 0, v133, vcc_lo
global_load_b64 v[150:151], v[132:133], off offset:0
global_load_b64 v[152:153], v[132:133], off offset:256
global_load_b64 v[154:155], v[132:133], off offset:512
global_load_b64 v[156:157], v[132:133], off offset:768
global_load_b64 v[158:159], v[132:133], off offset:1024
global_load_b64 v[160:161], v[132:133], off offset:1280
global_load_b64 v[162:163], v[132:133], off offset:1536
global_load_b64 v[164:165], v[132:133], off offset:1792
v_add_co_u32 v132, vcc_lo, s81, v132
v_add_co_ci_u32 v133, vcc_lo, 0, v133, vcc_lo
s_waitcnt vmcnt(8)
v_dual_fmac_f32 v0, s77, v134 :: v_dual_fmac_f32 v1, s77, v135
v_dual_fmac_f32 v2, s77, v136 :: v_dual_fmac_f32 v3, s77, v137
v_dual_fmac_f32 v4, s77, v138 :: v_dual_fmac_f32 v5, s77, v139
v_dual_fmac_f32 v6, s77, v140 :: v_dual_fmac_f32 v7, s77, v141
v_dual_fmac_f32 v8, s77, v142 :: v_dual_fmac_f32 v9, s77, v143
v_dual_fmac_f32 v10, s77, v144 :: v_dual_fmac_f32 v11, s77, v145
v_dual_fmac_f32 v12, s77, v146 :: v_dual_fmac_f32 v13, s77, v147
v_dual_fmac_f32 v14, s77, v148 :: v_dual_fmac_f32 v15, s77, v149
global_load_b64 v[134:135], v[132:133], off offset:0
global_load_b64 v[136:137], v[132:133], off offset:256
global_load_b64 v[138:139], v[132:133], off offset:512
global_load_b64 v[140:141], v[132:133], off offset:768
global_load_b64 v[142:143], v[132:133], off offset:1024
global_load_b64 v[144:145], v[132:133], off offset:1280
global_load_b64 v[146:147], v[132:133], off offset:1536
global_load_b64 v[148:149], v[132:133], off offset:1792
v_add_co_u32 v132, vcc_lo, s81, v132
v_add_co_ci_u32 v133, vcc_lo, 0, v133, vcc_lo
s_waitcnt vmcnt(8)
v_dual_fmac_f32 v16, s77, v150 :: v_dual_fmac_f32 v17, s77, v151
v_dual_fmac_f32 v18, s77, v152 :: v_dual_fmac_f32 v19, s77, v153
v_dual_fmac_f32 v20, s77, v154 :: v_dual_fmac_f32 v21, s77, v155
v_dual_fmac_f32 v22, s77, v156 :: v_dual_fmac_f32 v23, s77, v157
v_dual_fmac_f32 v24, s77, v158 :: v_dual_fmac_f32 v25, s77, v159
v_dual_fmac_f32 v26, s77, v160 :: v_dual_fmac_f32 v27, s77, v161
v_dual_fmac_f32 v28, s77, v162 :: v_dual_fmac_f32 v29, s77, v163
v_dual_fmac_f32 v30, s77, v164 :: v_dual_fmac_f32 v31, s77, v165
global_load_b64 v[150:151], v[132:133], off offset:0
global_load_b64 v[152:153], v[132:133], off offset:256
global_load_b64 v[154:155], v[132:133], off offset:512
global_load_b64 v[156:157], v[132:133], off offset:768
global_load_b64 v[158:159], v[132:133], off offset:1024
global_load_b64 v[160:161], v[132:133], off offset:1280
global_load_b64 v[162:163], v[132:133], off offset:1536
global_load_b64 v[164:165], v[132:133], off offset:1792
v_add_co_u32 v132, vcc_lo, s81, v132
v_add_co_ci_u32 v133, vcc_lo, 0, v133, vcc_lo
s_waitcnt vmcnt(8)
v_dual_fmac_f32 v32, s77, v134 :: v_dual_fmac_f32 v33, s77, v135
v_dual_fmac_f32 v34, s77, v136 :: v_dual_fmac_f32 v35, s77, v137
v_dual_fmac_f32 v36, s77, v138 :: v_dual_fmac_f32 v37, s77, v139
v_dual_fmac_f32 v38, s77, v140 :: v_dual_fmac_f32 v39, s77, v141
v_dual_fmac_f32 v40, s77, v142 :: v_dual_fmac_f32 v41, s77, v143
v_dual_fmac_f32 v42, s77, v144 :: v_dual_fmac_f32 v43, s77, v145
v_dual_fmac_f32 v44, s77, v146 :: v_dual_fmac_f32 v45, s77, v147
v_dual_fmac_f32 v46, s77, v148 :: v_dual_fmac_f32 v47, s77, v149
global_load_b64 v[134:135], v[132:133], off offset:0
global_load_b64 v[136:137], v[132:133], off offset:256
global_load_b64 v[138:139], v[132:133], off offset:512
global_load_b64 v[140:141], v[132:133], off offset:768
global_load_b64 v[142:143], v[132:133], off offset:1024
global_load_b64 v[144:145], v[132:133], off offset:1280
global_load_b64 v[146:147], v[132:133], off offset:1536
global_load_b64 v[148:149], v[132:133], off offset:1792
v_add_co_u32 v132, vcc_lo, s81, v132
v_add_co_ci_u32 v133, vcc_lo, 0, v133, vcc_lo
s_waitcnt vmcnt(8)
v_dual_fmac_f32 v48, s77, v150 :: v_dual_fmac_f32 v49, s77, v151
v_dual_fmac_f32 v50, s77, v152 :: v_dual_fmac_f32 v51, s77, v153
v_dual_fmac_f32 v52, s77, v154 :: v_dual_fmac_f32 v53, s77, v155
v_dual_fmac_f32 v54, s77, v156 :: v_dual_fmac_f32 v55, s77, v157
v_dual_fmac_f32 v56, s77, v158 :: v_dual_fmac_f32 v57, s77, v159
v_dual_fmac_f32 v58, s77, v160 :: v_dual_fmac_f32 v59, s77, v161
v_dual_fmac_f32 v60, s77, v162 :: v_dual_fmac_f32 v61, s77, v163
v_dual_fmac_f32 v62, s77, v164 :: v_dual_fmac_f32 v63, s77, v165
global_load_b64 v[150:151], v[132:133], off offset:0
global_load_b64 v[152:153], v[132:133], off offset:256
global_load_b64 v[154:155], v[132:133], off offset:512
global_load_b64 v[156:157], v[132:133], off offset:768
global_load_b64 v[158:159], v[132:133], off offset:1024
global_load_b64 v[160:161], v[132:133], off offset:1280
global_load_b64 v[162:163], v[132:133], off offset:1536
global_load_b64 v[164:165], v[132:133], off offset:1792
v_add_co_u32 v132, vcc_lo, s81, v132
v_add_co_ci_u32 v133, vcc_lo, 0, v133, vcc_lo
s_waitcnt vmcnt(8)
v_dual_fmac_f32 v64, s77, v134 :: v_dual_fmac_f32 v65, s77, v135
v_dual_fmac_f32 v66, s77, v136 :: v_dual_fmac_f32 v67, s77, v137
v_dual_fmac_f32 v68, s77, v138 :: v_dual_fmac_f32 v69, s77, v139
v_dual_fmac_f32 v70, s77, v140 :: v_dual_fmac_f32 v71, s77, v141
v_dual_fmac_f32 v72, s77, v142 :: v_dual_fmac_f32 v73, s77, v143
v_dual_fmac_f32 v74, s77, v144 :: v_dual_fmac_f32 v75, s77, v145
v_dual_fmac_f32 v76, s77, v146 :: v_dual_fmac_f32 v77, s77, v147
v_dual_fmac_f32 v78, s77, v148 :: v_dual_fmac_f32 v79, s77, v149
global_load_b64 v[134:135], v[132:133], off offset:0
global_load_b64 v[136:137], v[132:133], off offset:256
global_load_b64 v[138:139], v[132:133], off offset:512
global_load_b64 v[140:141], v[132:133], off offset:768
global_load_b64 v[142:143], v[132:133], off offset:1024
global_load_b64 v[144:145], v[132:133], off offset:1280
global_load_b64 v[146:147], v[132:133], off offset:1536
global_load_b64 v[148:149], v[132:133], off offset:1792
v_add_co_u32 v132, vcc_lo, s81, v132
v_add_co_ci_u32 v133, vcc_lo, 0, v133, vcc_lo
s_waitcnt vmcnt(8)
v_dual_fmac_f32 v80, s77, v150 :: v_dual_fmac_f32 v81, s77, v151
v_dual_fmac_f32 v82, s77, v152 :: v_dual_fmac_f32 v83, s77, v153
v_dual_fmac_f32 v84, s77, v154 :: v_dual_fmac_f32 v85, s77, v155
v_dual_fmac_f32 v86, s77, v156 :: v_dual_fmac_f32 v87, s77, v157
v_dual_fmac_f32 v88, s77, v158 :: v_dual_fmac_f32 v89, s77, v159
v_dual_fmac_f32 v90, s77, v160 :: v_dual_fmac_f32 v91, s77, v161
v_dual_fmac_f32 v92, s77, v162 :: v_dual_fmac_f32 v93, s77, v163
v_dual_fmac_f32 v94, s77, v164 :: v_dual_fmac_f32 v95, s77, v165
global_load_b64 v[150:151], v[132:133], off offset:0
global_load_b64 v[152:153], v[132:133], off offset:256
global_load_b64 v[154:155], v[132:133], off offset:512
global_load_b64 v[156:157], v[132:133], off offset:768
global_load_b64 v[158:159], v[132:133], off offset:1024
global_load_b64 v[160:161], v[132:133], off offset:1280
global_load_b64 v[162:163], v[132:133], off offset:1536
global_load_b64 v[164:165], v[132:133], off offset:1792
v_add_co_u32 v132, vcc_lo, s81, v132
v_add_co_ci_u32 v133, vcc_lo, 0, v133, vcc_lo
s_waitcnt vmcnt(8)
v_dual_fmac_f32 v96, s77, v134 :: v_dual_fmac_f32 v97, s77, v135
v_dual_fmac_f32 v98, s77, v136 :: v_dual_fmac_f32 v99, s77, v137
v_dual_fmac_f32 v100, s77, v138 :: v_dual_fmac_f32 v101, s77, v139
v_dual_fmac_f32 v102, s77, v140 :: v_dual_fmac_f32 v103, s77, v141
v_dual_fmac_f32 v104, s77, v142 :: v_dual_fmac_f32 v105, s77, v143
v_dual_fmac_f32 v106, s77, v144 :: v_dual_fmac_f32 v107, s77, v145
v_dual_fmac_f32 v108, s77, v146 :: v_dual_fmac_f32 v109, s77, v147
v_dual_fmac_f32 v110, s77, v148 :: v_dual_fmac_f32 v111, s77, v149
s_waitcnt vmcnt(0)
v_dual_fmac_f32 v112, s77, v150 :: v_dual_fmac_f32 v113, s77, v151
v_dual_fmac_f32 v114, s77, v152 :: v_dual_fmac_f32 v115, s77, v153
v_dual_fmac_f32 v116, s77, v154 :: v_dual_fmac_f32 v117, s77, v155
v_dual_fmac_f32 v118, s77, v156 :: v_dual_fmac_f32 v119, s77, v157
v_dual_fmac_f32 v120, s77, v158 :: v_dual_fmac_f32 v121, s77, v159
v_dual_fmac_f32 v122, s77, v160 :: v_dual_fmac_f32 v123, s77, v161
v_dual_fmac_f32 v124, s77, v162 :: v_dual_fmac_f32 v125, s77, v163
v_dual_fmac_f32 v126, s77, v164 :: v_dual_fmac_f32 v127, s77, v165
label_BetaDone:
v_and_b32 v128, 31, v[vgprSerial]
v_lshlrev_b32 v128, 1, v128
s_lshl_b32 s72, s[sgprWorkGroup0], 9
v_add_nc_u32 v128, s72, v128
v_lshrrev_b32 v129, 5, v[vgprSerial]
v_lshlrev_b32 v129, 3, v129
s_lshl_b32 s72, s[sgprWorkGroup1], 7
v_add_nc_u32 v129, s72, v129
v_mul_lo_u32 v129, s[sgprStrideD1J], v129
v_add_nc_u32 v128, v128, v129
v_lshlrev_b32 v128, 2, v128
v_add_co_u32 v130, vcc_lo, s[sgprAddressD], v128
v_add_co_ci_u32 v131, vcc_lo, s[sgprAddressD+1], 0, vcc_lo
s_lshl_b32 s72, s[sgprStrideD1J], 2
s_waitcnt_depctr 0
flat_store_b64 v[130:131], v[0:1] offset:0
flat_store_b64 v[130:131], v[2:3] offset:256
flat_store_b64 v[130:131], v[4:5] offset:512
flat_store_b64 v[130:131], v[6:7] offset:768
flat_store_b64 v[130:131], v[8:9] offset:1024
flat_store_b64 v[130:131], v[10:11] offset:1280
flat_store_b64 v[130:131], v[12:13] offset:1536
flat_store_b64 v[130:131], v[14:15] offset:1792
s_mul_i32 s73, s72, 1
v_add_co_u32 v130, vcc_lo, s73, v130
v_add_co_ci_u32 v131, vcc_lo, 0, v131, vcc_lo
flat_store_b64 v[130:131], v[16:17] offset:0
flat_store_b64 v[130:131], v[18:19] offset:256
flat_store_b64 v[130:131], v[20:21] offset:512
flat_store_b64 v[130:131], v[22:23] offset:768
flat_store_b64 v[130:131], v[24:25] offset:1024
flat_store_b64 v[130:131], v[26:27] offset:1280
flat_store_b64 v[130:131], v[28:29] offset:1536
flat_store_b64 v[130:131], v[30:31] offset:1792
s_mul_i32 s73, s72, 1
v_add_co_u32 v130, vcc_lo, s73, v130
v_add_co_ci_u32 v131, vcc_lo, 0, v131, vcc_lo
flat_store_b64 v[130:131], v[32:33] offset:0
flat_store_b64 v[130:131], v[34:35] offset:256
flat_store_b64 v[130:131], v[36:37] offset:512
flat_store_b64 v[130:131], v[38:39] offset:768
flat_store_b64 v[130:131], v[40:41] offset:1024
flat_store_b64 v[130:131], v[42:43] offset:1280
flat_store_b64 v[130:131], v[44:45] offset:1536
flat_store_b64 v[130:131], v[46:47] offset:1792
s_mul_i32 s73, s72, 1
v_add_co_u32 v130, vcc_lo, s73, v130
v_add_co_ci_u32 v131, vcc_lo, 0, v131, vcc_lo
flat_store_b64 v[130:131], v[48:49] offset:0
flat_store_b64 v[130:131], v[50:51] offset:256
flat_store_b64 v[130:131], v[52:53] offset:512
flat_store_b64 v[130:131], v[54:55] offset:768
flat_store_b64 v[130:131], v[56:57] offset:1024
flat_store_b64 v[130:131], v[58:59] offset:1280
flat_store_b64 v[130:131], v[60:61] offset:1536
flat_store_b64 v[130:131], v[62:63] offset:1792
s_mul_i32 s73, s72, 1
v_add_co_u32 v130, vcc_lo, s73, v130
v_add_co_ci_u32 v131, vcc_lo, 0, v131, vcc_lo
flat_store_b64 v[130:131], v[64:65] offset:0
flat_store_b64 v[130:131], v[66:67] offset:256
flat_store_b64 v[130:131], v[68:69] offset:512
flat_store_b64 v[130:131], v[70:71] offset:768
flat_store_b64 v[130:131], v[72:73] offset:1024
flat_store_b64 v[130:131], v[74:75] offset:1280
flat_store_b64 v[130:131], v[76:77] offset:1536
flat_store_b64 v[130:131], v[78:79] offset:1792
s_mul_i32 s73, s72, 1
v_add_co_u32 v130, vcc_lo, s73, v130
v_add_co_ci_u32 v131, vcc_lo, 0, v131, vcc_lo
flat_store_b64 v[130:131], v[80:81] offset:0
flat_store_b64 v[130:131], v[82:83] offset:256
flat_store_b64 v[130:131], v[84:85] offset:512
flat_store_b64 v[130:131], v[86:87] offset:768
flat_store_b64 v[130:131], v[88:89] offset:1024
flat_store_b64 v[130:131], v[90:91] offset:1280
flat_store_b64 v[130:131], v[92:93] offset:1536
flat_store_b64 v[130:131], v[94:95] offset:1792
s_mul_i32 s73, s72, 1
v_add_co_u32 v130, vcc_lo, s73, v130
v_add_co_ci_u32 v131, vcc_lo, 0, v131, vcc_lo
flat_store_b64 v[130:131], v[96:97] offset:0
flat_store_b64 v[130:131], v[98:99] offset:256
flat_store_b64 v[130:131], v[100:101] offset:512
flat_store_b64 v[130:131], v[102:103] offset:768
flat_store_b64 v[130:131], v[104:105] offset:1024
flat_store_b64 v[130:131], v[106:107] offset:1280
flat_store_b64 v[130:131], v[108:109] offset:1536
flat_store_b64 v[130:131], v[110:111] offset:1792
s_mul_i32 s73, s72, 1
v_add_co_u32 v130, vcc_lo, s73, v130
v_add_co_ci_u32 v131, vcc_lo, 0, v131, vcc_lo
flat_store_b64 v[130:131], v[112:113] offset:0
flat_store_b64 v[130:131], v[114:115] offset:256
flat_store_b64 v[130:131], v[116:117] offset:512
flat_store_b64 v[130:131], v[118:119] offset:768
flat_store_b64 v[130:131], v[120:121] offset:1024
flat_store_b64 v[130:131], v[122:123] offset:1280
flat_store_b64 v[130:131], v[124:125] offset:1536
flat_store_b64 v[130:131], v[126:127] offset:1792
label_KernelEnd:
s_endpgm
label_ASM_End:
