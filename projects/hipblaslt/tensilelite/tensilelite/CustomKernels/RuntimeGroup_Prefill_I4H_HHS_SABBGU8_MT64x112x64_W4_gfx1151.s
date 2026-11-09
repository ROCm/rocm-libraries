// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
.text
.protected RuntimeGroup_Prefill_I4H_HHS_SABBGU8_MT64x112x64_W4_gfx1151
.globl RuntimeGroup_Prefill_I4H_HHS_SABBGU8_MT64x112x64_W4_gfx1151
.p2align 8
.type RuntimeGroup_Prefill_I4H_HHS_SABBGU8_MT64x112x64_W4_gfx1151,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel RuntimeGroup_Prefill_I4H_HHS_SABBGU8_MT64x112x64_W4_gfx1151
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 221 // vgprs
  .amdhsa_next_free_sgpr 104 // sgprs
  .amdhsa_group_segment_fixed_size 25344 // lds bytes
  .amdhsa_wavefront_size32 1 // 32-thread wavefronts
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel
.text
/* Num VGPR   =221 */
/* Num AccVGPR=0 */
/* Num SGPR   =96 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 8 x 7 */
/* SubGroup= 8 x 16 */
/* VectorWidthA=1 */
/* VectorWidthB=1 */
/* GlobalReadVectorWidthA=8, GlobalReadVectorWidthB=4 */
/* DirectToLdsA=False */
/* DirectToLdsB=False */
/* UseSgprForGRO=False */
.amdgpu_metadata
---
custom.config:
  InternalSupportParams:
    KernArgsVersion: 3
  StaggerU: 0
amdhsa.version:
  - 1
  - 1
amdhsa.kernels:
  - .name: RuntimeGroup_Prefill_I4H_HHS_SABBGU8_MT64x112x64_W4_gfx1151
    .symbol: 'RuntimeGroup_Prefill_I4H_HHS_SABBGU8_MT64x112x64_W4_gfx1151.kd'
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
      - .name:            A
        .size:            8
        .offset:          32
        .value_kind:      global_buffer
        .value_type:      i4
        .address_space:   generic
      - .name:            B
        .size:            8
        .offset:          40
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            strideA0
        .size:            4
        .offset:          48
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA1
        .size:            4
        .offset:          52
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB0
        .size:            4
        .offset:          56
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB1
        .size:            4
        .offset:          60
        .value_kind:      by_value
        .value_type:      u32
      - .name:            alpha
        .size:            4
        .offset:          64
        .value_kind:      by_value
        .value_type:      f32
      - .name:            beta
        .size:            4
        .offset:          68
        .value_kind:      by_value
        .value_type:      f32
      - .name:            D
        .size:            8
        .offset:          72
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            C
        .size:            8
        .offset:          80
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            strideD0
        .size:            4
        .offset:          88
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideD1
        .size:            4
        .offset:          92
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC0
        .size:            4
        .offset:          96
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC1
        .size:            4
        .offset:          100
        .value_kind:      by_value
        .value_type:      u32
      - .name:            AddressScaleA
        .size:            8
        .offset:          104
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            AddressScaleB
        .size:            8
        .offset:          112
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            AddressScaleZeroA
        .size:            8
        .offset:          120
        .value_kind:      global_buffer
        .value_type:      void
        .address_space:   generic
      - .name:            batchOffsetD
        .size:            8
        .offset:          128
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetC
        .size:            8
        .offset:          136
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetA
        .size:            8
        .offset:          144
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetB
        .size:            8
        .offset:          152
        .value_kind:      by_value
        .value_type:      u64
      - .name:            groupSize
        .size:            4
        .offset:          160
        .value_kind:      by_value
        .value_type:      u32
    .group_segment_fixed_size:   25344
    .kernarg_segment_align:      8
    .kernarg_segment_size:       168
    .max_flat_workgroup_size:    128
    .private_segment_fixed_size: 0
    .sgpr_count:                 104
    .sgpr_spill_count:           0
    .vgpr_count:                 221
    .vgpr_spill_count:           0
    .wavefront_size:             32
...
.end_amdgpu_metadata
RuntimeGroup_Prefill_I4H_HHS_SABBGU8_MT64x112x64_W4_gfx1151:
s_mov_b32 s94, 0xe408e408
s_mov_b32 s95, 0xd480d480
s_mov_b32 s92, 0x000f000f
s_mov_b32 s93, 0x00f000f0
label_ASM_Start:  /// Main body of the asm kernel

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
/* ValuC range: [0-56), serializedStore enabled */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprBase, 100
.set vgprLocalWriteAddrA, 94
.set vgprLocalWriteAddrB, 95
.set vgprGlobalReadOffsetA, 56
.set vgprGlobalReadOffsetB, 60
.set vgprGlobalReadOffsetScaleA, 74
.set vgprG2LScaleA, 78
.set vgprG2LScaleZeroA, 82
.set vgprGlobalReadOffsetScaleZeroA, 86
.set vgprGlobalReadByteOffsetScaleZeroA, 90
.set vgprLocalReadAddrA, 96
.set vgprLocalReadAddrB, 97
.set vgprSerial, 210

/******************************************/
/* VGPR Macro Assignments                 */
/******************************************/
.set vgprValuA_X0_I0_BASE, vgprBase+0
.set vgprValuB_X0_I0_BASE, vgprBase+9
.set vgprG2LA_BASE, vgprBase+66
.set vgprG2LB_BASE, vgprBase+82
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprG2LA, vgprG2LA_BASE+0
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
.set sgprAddressA, 28
.set sgprAddressB, 30
.set sgprStridesA, 32
.set sgprStridesB, 34
.set sgprAlpha, 36
.set sgprBeta, 37
.set sgprAddressD, 38
.set sgprAddressC, 40
.set sgprStridesD, 42
.set sgprStridesC, 44
.set sgprGSU, 46
.set sgprAddressScaleA, 48
.set sgprAddressScaleB, 50
.set sgprStrideScaleA, 47
.set sgprSrdScaleA, 52
.set sgprScaleAPkMagic, 56
.set sgprScaleAPkPermute, 60
.set sgprAddressScaleZeroA, 62
.set sgprSrdScaleZeroA, 64

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

.set MT0, 64
.set MT1, 112
.set DepthU, 64
/* Number of elements to shift-left SRD */
.set SrdShiftLeftA, 8
.set SrdShiftLeftB, 4
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

// Uniform group-size parameters for G=32,64,128.
s_load_b32 s96, s[sgprKernArgAddress:sgprKernArgAddress+1], 160
s_waitcnt lgkmcnt(0)
s_ff1_i32_b32 s97, s96
s_lshr_b32 s98, s96, 6
s_max_u32 s98, s98, 1
s_sub_u32 s98, s98, 1
s_mov_b32 s100, 64
s_lshr_b32 s100, s100, s97
s_max_u32 s100, s100, 1
s_lshl_b32 s99, s100, 1
s_sub_u32 s96, s96, 1
s_mov_b32 s101, 0

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
s_load_b64 s[sgprAddressScaleA:sgprAddressScaleA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x58
s_load_b64 s[sgprAddressScaleB:sgprAddressScaleB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x60
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
s_mov_b32 m0, 0x6300                               // LDS clamp at 25344 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id
s_mov_b32 vcc_hi, 0                                // Ensure hi bits are zero

/* remap workgroup to XCCs */
s_lshr_b32 s72, s[sgprWGM], 0x10                   // Get WGMXCC
s_ff1_i32_b32 s72, s72                             // Get log(WGMXCC)
s_lshr_b32 s73, s[sgprWGM], 0x16                   // Get CU_Count
/* remap WGs if WGMXCC > 1 ( log(WGMXCC) > 0 ) */
s_cmp_gt_i32 s72, 0
s_cbranch_scc0 label_skip_WGMXCC
/* only remap WGs in the range */
s_lshr_b32 s69, s23, s72
s_lshl_b32 s69, s69, s72
s_cmp_ge_u32 s[sgprWorkGroup0], s69
s_cbranch_scc1 label_skip_WGMXCC
s_cmp_eq_u32 s73, 0                                // CU_Count == 0 ?
s_cbranch_scc0 label_XCCG_nonzero
s_lshr_b32 s69, s[sgprWorkGroup0], s72
s_bfm_b32 s70, s72, 0
s_and_b32 s70, s[sgprWorkGroup0], s70
s_lshr_b32 s71, s23, s72
s_mul_i32 s70, s70, s71
s_add_u32 s[sgprWorkGroup0], s69, s70
s_branch label_skip_WGMXCC
label_XCCG_nonzero:
/* temp0 = (wg//CU_Count)*CU_Count */
v_cvt_f64_u32 v[6:7], s73                          // s69 = s[sgprWorkGroup0] / s73
v_rcp_f64 v[6:7], v[6:7]                           // s69 = s[sgprWorkGroup0] / s73
v_cvt_f64_u32 v[8:9], s[sgprWorkGroup0]            // s69 = s[sgprWorkGroup0] / s73
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s69 = s[sgprWorkGroup0] / s73
v_cvt_u32_f64 v6, v[6:7]                           // s69 = s[sgprWorkGroup0] / s73
v_mul_lo_u32 v7, v6, s73                           // s69 = s[sgprWorkGroup0] / s73
v_sub_nc_u32 v8, s[sgprWorkGroup0], v7             // s69 = s[sgprWorkGroup0] / s73
v_cmp_ge_u32 vcc_lo, v8, s73                       // s69 = s[sgprWorkGroup0] / s73
s_mov_b32 exec_lo, vcc_lo                          // s69 = s[sgprWorkGroup0] / s73
v_add_nc_u32 v6, v6, 1                             // s69 = s[sgprWorkGroup0] / s73
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v7, v6, s73                           // s69 = s[sgprWorkGroup0] / s73
v_sub_nc_u32 v8, s[sgprWorkGroup0], v7             // s69 = s[sgprWorkGroup0] / s73
v_readfirstlane_b32 s69, v6                        // quotient
v_readfirstlane_b32 s70, v8                        // remainder
s_mul_i32 s69, s69, s73
/* temp1 = (wg%CU_Count)//WGMXCC */
s_lshr_b32 s70, s70, s72
/* temp0 = temp0 + temp1 */
s_add_u32 s69, s69, s70
/* temp1 = (wg%WGMXCC) * ((WGs - (WGs//CU_Count) * CU_Count) if (wg > (WGs//CU_Count) * CU_Count) else CU_Count)//WGMXCC */
v_cvt_f64_u32 v[6:7], s73                          // s70 = s23 / s73
v_rcp_f64 v[6:7], v[6:7]                           // s70 = s23 / s73
v_cvt_f64_u32 v[8:9], s23                          // s70 = s23 / s73
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s70 = s23 / s73
v_cvt_u32_f64 v6, v[6:7]                           // s70 = s23 / s73
v_mul_lo_u32 v7, v6, s73                           // s70 = s23 / s73
v_sub_nc_u32 v8, s23, v7                           // s70 = s23 / s73
v_cmp_ge_u32 vcc_lo, v8, s73                       // s70 = s23 / s73
s_mov_b32 exec_lo, vcc_lo                          // s70 = s23 / s73
v_add_nc_u32 v6, v6, 1                             // s70 = s23 / s73
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s70, v6                        // quotient
s_mul_i32 s70, s70, s73
s_sub_u32 s71, s23, s70
s_cmp_gt_u32 s[sgprWorkGroup0], s70
s_cselect_b32 s70, s71, s73
s_lshr_b32 s70, s70, s72
s_bfm_b32 s71, s72, 0
s_and_b32 s71, s[sgprWorkGroup0], s71
s_mul_i32 s70, s70, s71
/* WorkGroup0 = temp0 + temp1 */
s_add_u32 s[sgprWorkGroup0], s69, s70
label_skip_WGMXCC:  /// skip WGMXCC if no enough WGs to remap
s_cmp_eq_u32 s21, 3
s_cbranch_scc1 label_ArgType3_Routed_To_ArgType0
s_cmp_eq_u32 s21, 0
s_cbranch_scc0 label_MultiGemm
label_ArgType3_Routed_To_ArgType0:
/* init: add vgpr [100...265) to pool */
/* init: add vgpr [0...56) to pool */
/* init: add agpr [0...0) to pool */

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
/* lr0I */
v_and_b32 v1, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
v_and_b32 v0, 15, v1                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v0, 6, v0                            // 1. N offset: nOffset = nIdx * nStride(64)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v4, 5, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(32)
v_and_b32 v4, 3, v4                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(4)
v_lshl_add_u32 v0, v4, 10, v0                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(1024); 7. final local read offset: flrOffset = lrOffset + WOffset
/* lr1J */
v_and_b32 v2, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v1, 6, v1                            // 1. N offset: nOffset = nIdx * nStride(64)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)

/* local read addresses: final offsets a */
v_lshrrev_b32 v2, 5, v[vgprSerial]                 // 2 = Serial / 32
v_lshrrev_b32 v2, 2, v2                            // LSU offset: Get LSU wave_id
s_mov_b32 s16, 64                                  // LSU offset: stride = lsuStride(64) when umlds==True
v_mul_lo_u32 v2, s16, v2                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_nc_u32 v[vgprLocalReadAddrA], v2, v0         // Final Offset: offset = (lro0+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrA], 1, v[vgprLocalReadAddrA] //  (multiple bpe)
v_lshrrev_b32 v3, 7, v[vgprLocalReadAddrA]         // Final Offset: padding 16 per block 128
v_lshl_add_u32 v[vgprLocalReadAddrA], v3, 4, v[vgprLocalReadAddrA] // Final Offset: padding 16 per block 128

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 5, v[vgprSerial]                 // 0 = Serial / 32
v_lshrrev_b32 v0, 2, v0                            // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(64) when umlds==True (dup assign opt.)
v_mul_lo_u32 v0, s16, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_nc_u32 v[vgprLocalReadAddrB], v0, v1         // Final Offset: offset = (lro1+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrB], 1, v[vgprLocalReadAddrB] //  (multiple bpe)
v_lshrrev_b32 v2, 7, v[vgprLocalReadAddrB]         // Final Offset: padding 16 per block 128
v_lshl_add_u32 v[vgprLocalReadAddrB], v2, 4, v[vgprLocalReadAddrB] // Final Offset: padding 16 per block 128

/* local read addresses: declare addresses a */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, 0x2400, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 8 */
/* v1 = A-unroll = serial%LVCA */
v_lshrrev_b32 v0, 3, v[vgprSerial]                 // 0 = Serial / 8
v_and_b32 v1, 7, v[vgprSerial]                     // 1 = Serial % 8
/* unroll *= glvw */
v_lshlrev_b32 v1, 3, v1                            // v1 = v1 * 8
v_mov_b32 v4, v1                                   // copy for GlobalSplitU
/* LVCB = 16 */
/* v3 = B-unroll = serial%LVCB */
v_lshrrev_b32 v2, 4, v[vgprSerial]                 // 2 = Serial / 16
v_and_b32 v3, 15, v[vgprSerial]                    // 3 = Serial % 16
/* unroll *= glvw */
v_lshlrev_b32 v3, 2, v3                            // v3 = v3 * 4
v_mov_b32 v5, v3                                   // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v4 */
/* lwaUnrollAssignmentB = v5 */

/* local write addresses: first offset a */
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x40, v0     // lwAL**(DepthU_Compute + PAD)
v_add_nc_u32 v[vgprLocalWriteAddrA], v4, v[vgprLocalWriteAddrA] // lwFOA = (lwAA + lwAL*(DepthU+PAD))
v_lshlrev_b32 v[vgprLocalWriteAddrA], 1, v[vgprLocalWriteAddrA] //  (multiple bpe)
v_lshrrev_b32 v6, 7, v[vgprLocalWriteAddrA]        // padding 16 per block 128
v_lshl_add_u32 v[vgprLocalWriteAddrA], v6, 4, v[vgprLocalWriteAddrA] // padding 16 per block 128

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x40, v2     // lwBL**(DepthU_Compute + PAD)
v_add_nc_u32 v[vgprLocalWriteAddrB], v5, v[vgprLocalWriteAddrB] // lwFOB = (lwBB + lwBL*(DepthU+PAD))
v_lshlrev_b32 v[vgprLocalWriteAddrB], 1, v[vgprLocalWriteAddrB] //  (multiple bpe)
v_lshrrev_b32 v6, 7, v[vgprLocalWriteAddrB]        // padding 16 per block 128
v_lshl_add_u32 v[vgprLocalWriteAddrB], v6, 4, v[vgprLocalWriteAddrB] // padding 16 per block 128
v_add_co_u32 v[vgprLocalWriteAddrB], vcc_lo, 0x2400, v[vgprLocalWriteAddrB] // lwFOB = lw1J + lwL*MT1J + LDS_OFFSET_B=9216
s_waitcnt lgkmcnt(0)                               // wait for 88/0 bytes of kern args over preload
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
s_and_b32 s17, s[sgprGSU], 0xfff                   // Restore GSU
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
s_and_b32 s16, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s16, 2                                // ArgType == 2 ?
s_cbranch_scc1 label_IsExternalValid               // branch if ArgType == 2
s_mov_b32 s15, 112                                 // KernArgAddressOffset
s_mul_i32 s74, s20, 4
s_mov_b64 s[68:69], s[sgprKernArgAddress:sgprKernArgAddress+1]
s_branch label_IsExternalValidEnd
label_IsExternalValid:
s_mov_b32 s15, 228
s_mov_b32 s74, 0
s_mov_b64 s[68:69], s[sgprKernArgAddress:sgprKernArgAddress+1]
label_IsExternalValidEnd:

/* Grouped Gemm:: prefetch 1 arg load */
s_mov_b32 s14, 1
s_mov_b32 s75, 0
s_load_b128 s[24:27], s[68:69], s74
s_cmpk_eq_u32 s20, 1                               // if gemm_count is 1?
s_cbranch_scc1 label_wgTable_noLoadLoop

/* Grouped Gemm:: accumulate numTiles for each gemm */
/* Grouped Gemm:: loop start */
label_Loop_GemmCount:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s72, s24, 6                             // s72 = s24 / 64
s_and_b32 s70, 63, s24                             // s70 = s24 % 64
s_addc_u32 s72, s72, 0
s_mul_hi_u32 s71, s25, 76695845                    // STATIC_DIV: divisor=112
s_mul_i32 s70, s25, 76695845                       // tmp = dividend * magic
s_lshr_b64 s[70:71], s[70:71], 33                  // tmp0 = quotient
s_mul_i32 s71, s70, 112                            // tmp1 = quotient * divisor
s_cmp_lg_u32 s71, s25                              // if (quotient * divisor != dividend), result+=1
s_addc_u32 s73, s70, 0                             // if (quotient * divisor != dividend), result+=1
s_mul_i32 s72, s72, s73
s_mul_i32 s72, s72, s26
s_and_b32 s73, s[sgprGSU], 0xfff                   // Restore GSU
s_mul_i32 s72, s72, s73
s_add_u32 s75, s75, s72
s_cmp_lt_u32 s[sgprWorkGroup0], s75
s_cbranch_scc1 label_FOUND
s_add_u32 s74, s74, s15
s_load_b128 s[24:27], s[68:69], s74
s_add_u32 s14, s14, 1
s_cmp_lt_u32 s14, s20
s_cbranch_scc1 label_Loop_GemmCount

/* Grouped Gemm:: noLoadLoop */
label_wgTable_noLoadLoop:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s72, s24, 6                             // s72 = s24 / 64
s_and_b32 s70, 63, s24                             // s70 = s24 % 64
s_addc_u32 s72, s72, 0
s_mul_hi_u32 s71, s25, 76695845                    // STATIC_DIV: divisor=112
s_mul_i32 s70, s25, 76695845                       // tmp = dividend * magic
s_lshr_b64 s[70:71], s[70:71], 33                  // tmp0 = quotient
s_mul_i32 s71, s70, 112                            // tmp1 = quotient * divisor
s_cmp_lg_u32 s71, s25                              // if (quotient * divisor != dividend), result+=1
s_addc_u32 s73, s70, 0                             // if (quotient * divisor != dividend), result+=1
s_mul_i32 s72, s72, s73
s_mul_i32 s72, s72, s26
s_and_b32 s68, s[sgprGSU], 0xfff                   // Restore GSU
s_mul_i32 s72, s72, s68
s_add_u32 s75, s75, s72

/* Grouped Gemm:: gemmIndex found */
label_FOUND:
s_sub_u32 s69, s14, 1
s_sub_u32 s68, s75, s72
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s68
/* Check if custom structure pointer is null */
s_and_b32 s16, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s16, 2                                // ArgType == 2 ?
s_cbranch_scc1 label_LoadExternalStruct            // branch if ArgType == 2

/* Grouped Gemm: offset argument address to gemm */
/* Grouped Gemm: offset address from wg_table_start to args_start */
s_lshl2_add_u32 s[sgprKernArgAddress], s20, s[sgprKernArgAddress]
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s69, s69, 112                            // KernArgAddressOffset
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s69
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_b512 s[28:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_b64 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
s_load_b64 s[sgprAddressScaleA:sgprAddressScaleA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x58
s_load_b64 s[sgprAddressScaleB:sgprAddressScaleB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x60
s_branch label_LoadExternalStructEnd
label_LoadExternalStruct:
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s69, s69, 228
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s69
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
s_load_b64 s[sgprAddressD:sgprAddressD+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x10
s_load_b64 s[sgprAddressC:sgprAddressC+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x18
s_load_b64 s[sgprAddressA:sgprAddressA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x20
s_load_b64 s[sgprAddressB:sgprAddressB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x28
s_load_b64 s[sgprStridesD:sgprStridesD+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x30
s_load_b64 s[sgprStridesC:sgprStridesC+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x38
s_load_b64 s[sgprStridesA:sgprStridesA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x40
s_load_b64 s[sgprStridesB:sgprStridesB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x48
s_load_b32 s[sgprAlpha], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x50
// Read Beta
s_load_b32 s37, s[sgprKernArgAddress:sgprKernArgAddress+1], 96 // 96
s_load_b64 s[sgprAddressScaleA:sgprAddressScaleA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x64
s_load_b64 s[sgprAddressScaleB:sgprAddressScaleB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x6c
label_LoadExternalStructEnd:
/* init: add vgpr [100...265) to pool */
/* init: add vgpr [0...56) to pool */
/* init: add agpr [0...0) to pool */

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
/* lr0I */
v_and_b32 v1, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
v_and_b32 v0, 15, v1                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v0, 6, v0                            // 1. N offset: nOffset = nIdx * nStride(64)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v4, 5, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(32)
v_and_b32 v4, 3, v4                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(4)
v_lshl_add_u32 v0, v4, 10, v0                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(1024); 7. final local read offset: flrOffset = lrOffset + WOffset
/* lr1J */
v_and_b32 v2, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v1, 6, v1                            // 1. N offset: nOffset = nIdx * nStride(64)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)

/* local read addresses: final offsets a */
v_lshrrev_b32 v2, 5, v[vgprSerial]                 // 2 = Serial / 32
v_lshrrev_b32 v2, 2, v2                            // LSU offset: Get LSU wave_id
s_mov_b32 s16, 64                                  // LSU offset: stride = lsuStride(64) when umlds==True
v_mul_lo_u32 v2, s16, v2                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_nc_u32 v[vgprLocalReadAddrA], v2, v0         // Final Offset: offset = (lro0+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrA], 1, v[vgprLocalReadAddrA] //  (multiple bpe)
v_lshrrev_b32 v3, 7, v[vgprLocalReadAddrA]         // Final Offset: padding 16 per block 128
v_lshl_add_u32 v[vgprLocalReadAddrA], v3, 4, v[vgprLocalReadAddrA] // Final Offset: padding 16 per block 128

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 5, v[vgprSerial]                 // 0 = Serial / 32
v_lshrrev_b32 v0, 2, v0                            // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(64) when umlds==True (dup assign opt.)
v_mul_lo_u32 v0, s16, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_nc_u32 v[vgprLocalReadAddrB], v0, v1         // Final Offset: offset = (lro1+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrB], 1, v[vgprLocalReadAddrB] //  (multiple bpe)
v_lshrrev_b32 v2, 7, v[vgprLocalReadAddrB]         // Final Offset: padding 16 per block 128
v_lshl_add_u32 v[vgprLocalReadAddrB], v2, 4, v[vgprLocalReadAddrB] // Final Offset: padding 16 per block 128

/* local read addresses: declare addresses a */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, 0x2400, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 8 */
/* v1 = A-unroll = serial%LVCA */
v_lshrrev_b32 v0, 3, v[vgprSerial]                 // 0 = Serial / 8
v_and_b32 v1, 7, v[vgprSerial]                     // 1 = Serial % 8
/* unroll *= glvw */
v_lshlrev_b32 v1, 3, v1                            // v1 = v1 * 8
v_mov_b32 v4, v1                                   // copy for GlobalSplitU
/* LVCB = 16 */
/* v3 = B-unroll = serial%LVCB */
v_lshrrev_b32 v2, 4, v[vgprSerial]                 // 2 = Serial / 16
v_and_b32 v3, 15, v[vgprSerial]                    // 3 = Serial % 16
/* unroll *= glvw */
v_lshlrev_b32 v3, 2, v3                            // v3 = v3 * 4
v_mov_b32 v5, v3                                   // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v4 */
/* lwaUnrollAssignmentB = v5 */

/* local write addresses: first offset a */
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x40, v0     // lwAL**(DepthU_Compute + PAD)
v_add_nc_u32 v[vgprLocalWriteAddrA], v4, v[vgprLocalWriteAddrA] // lwFOA = (lwAA + lwAL*(DepthU+PAD))
v_lshlrev_b32 v[vgprLocalWriteAddrA], 1, v[vgprLocalWriteAddrA] //  (multiple bpe)
v_lshrrev_b32 v6, 7, v[vgprLocalWriteAddrA]        // padding 16 per block 128
v_lshl_add_u32 v[vgprLocalWriteAddrA], v6, 4, v[vgprLocalWriteAddrA] // padding 16 per block 128

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x40, v2     // lwBL**(DepthU_Compute + PAD)
v_add_nc_u32 v[vgprLocalWriteAddrB], v5, v[vgprLocalWriteAddrB] // lwFOB = (lwBB + lwBL*(DepthU+PAD))
v_lshlrev_b32 v[vgprLocalWriteAddrB], 1, v[vgprLocalWriteAddrB] //  (multiple bpe)
v_lshrrev_b32 v6, 7, v[vgprLocalWriteAddrB]        // padding 16 per block 128
v_lshl_add_u32 v[vgprLocalWriteAddrB], v6, 4, v[vgprLocalWriteAddrB] // padding 16 per block 128
v_add_co_u32 v[vgprLocalWriteAddrB], vcc_lo, 0x2400, v[vgprLocalWriteAddrB] // lwFOB = lw1J + lwL*MT1J + LDS_OFFSET_B=9216
s_waitcnt lgkmcnt(0)                               // wait for 88/0 bytes of kern args over preload
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
s_and_b32 s17, s[sgprGSU], 0xfff                   // Restore GSU
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
.set sgprSrdA, 68
.set sgprSrdB, 72
.set sgprShadowLimitA, 76
.set sgprShadowLimitB, 78
.set sgprStaggerUIter, 80
.set sgprWrapUA, 81
.set sgprWrapUB, 83
.set sgprGlobalReadIncsA, 85
.set sgprGlobalReadIncsB, 86
s_and_b32 s16, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s16, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_Skip_Address_Prepad_For_Pointer_Array
s_sub_u32 s[sgprAddressA+0], s[sgprAddressA+0], 4  // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressA+1], s[sgprAddressA+1], 0 // pre-pad to make room for possible pointer shift
s_sub_u32 s[sgprAddressB+0], s[sgprAddressB+0], 8  // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressB+1], s[sgprAddressB+1], 0 // pre-pad to make room for possible pointer shift
label_Skip_Address_Prepad_For_Pointer_Array:  /// Skip pre-padding of address for pointer array case

/* Short circuit condition if Alpha == 0, then sumDims=0 */
v_cmp_eq_f32 vcc_lo, s[sgprAlpha], 0.0             // s[Alpha] == 0.0f ?
s_cbranch_vccz label_AlphaNonZero                  // branch if s[Alpha] != 0
s_mov_b32 s[sgprSizesSum+0], 0                     // Set summation dim=0 if Alpha == 0
label_AlphaNonZero:

/******************************************/
/* Begin setupNewTile                     */
/******************************************/

/* global read addresses: work-group */
/* graWorkGroup mapping */
s_and_b32 s16, s[sgprGSU], 0xfff                   // Restore GSU
s_cmp_eq_u32 s16, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU                           // branch if GSU == 1
// GSU-not-WGMapRR :nwg1 = (size1J + MT1J - 1) / MT1J;
s_and_b32 s16, s[sgprGSU], 0x4000                  // SCC = (GSUWGMRR == 1) ?
s_cbranch_scc1 label_GSUWGMRR                      // branch if GSUWGMRR == 1
s_and_b32 s16, s[sgprGSU], 0xfff                   // Restore GSU
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
s_mov_b32 s[sgprGSULog2BpeC], 1
s_mov_b32 s[sgprGSULog2BpeD], 2
s_branch label_GSU_End
label_GSU:
s_mov_b64 s[sgprGSUSumIdx:sgprGSUSumIdx+1], 0      // Set GSUSumIdx to 0
s_mov_b32 s[sgprGSULog2BpeC], 1
s_mov_b32 s[sgprGSULog2BpeD], 1
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
v_mov_b32 v6, v0                                   // groA0I_0
v_add_co_u32 v7, vcc_lo, 16, v6                    // groA0I_1 += LSPA
v_add_co_u32 v8, vcc_lo, 16, v7                    // groA0I_2 += LSPA
v_add_co_u32 v9, vcc_lo, 16, v8                    // groA0I_3 += LSPA

/* global read addresses: tile offsets b */
v_mov_b32 v10, v2                                  // groB1J_0
v_add_co_u32 v11, vcc_lo, 8, v10                   // groB1J_1 += LSPB
v_add_co_u32 v12, vcc_lo, 8, v11                   // groB1J_2 += LSPB
v_add_co_u32 v13, vcc_lo, 8, v12                   // groB1J_3 += LSPB
v_add_co_u32 v14, vcc_lo, 8, v13                   // groB1J_4 += LSPB
v_add_co_u32 v15, vcc_lo, 8, v14                   // groB1J_5 += LSPB
v_add_co_u32 v16, vcc_lo, 8, v15                   // groB1J_6 += LSPB
v_add_co_u32 v17, vcc_lo, 8, v16                   // groB1J_7 += LSPB
v_add_co_u32 v18, vcc_lo, 8, v17                   // groB1J_8 += LSPB
v_add_co_u32 v19, vcc_lo, 8, v18                   // groB1J_9 += LSPB
v_add_co_u32 v20, vcc_lo, 8, v19                   // groB1J_10 += LSPB
v_add_co_u32 v21, vcc_lo, 8, v20                   // groB1J_11 += LSPB
v_add_co_u32 v22, vcc_lo, 8, v21                   // groB1J_12 += LSPB
v_add_co_u32 v23, vcc_lo, 8, v22                   // groB1J_13 += LSPB

/* global read addresses: unroll offsets a */
v_mov_b32 v24, v1                                  // groAL_0

/* global read addresses: unroll offsets b */
v_mov_b32 v25, v3                                  // groBL_0

/* global read addresses: addresses a */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s19, s[sgprWorkGroup0], 64            // WorkGroup[01] * MT
s_mul_i32 s18, s[sgprWorkGroup0], 64               // WorkGroup[01] * MT
s_mul_hi_u32 s19, s18, s[sgprStrideA0I]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s18, s18, s[sgprStrideA0I]               // tlu=0, scaled tile-offset by stride
s_and_b32 s16, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_A                        // branch if GSUC == 1
s_mul_hi_u32 s17, 64, s[sgprGSUSumIdx]             // gsuOffset = DepthU*GSUSumIdx
s_mul_i32 s16, 64, s[sgprGSUSumIdx]                // gsuOffset = DepthU*GSUSumIdx
s_branch label_GSUC_A_End
label_GSUC_A:
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum], 6 // s[LoopCounterL] = s[sgprSizesSum] / 64
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0xfff    // Restore GSU
v_cvt_f32_u32 v26, s[sgprGSUSumIdx+1]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v26, v26                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v27, s[sgprLoopCounterL]             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v26, v26, v27                            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v26, v26                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v27, v26, s[sgprGSUSumIdx+1]         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_nc_u32 v27, s[sgprLoopCounterL], v27         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmp_eq_u32 vcc_lo, v27, s[sgprGSUSumIdx+1]       // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, vcc_lo                          // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_nc_u32 v26, 1, v26                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v27, 0                                   // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v27, s[sgprGSUSumIdx+1]       // overflow happened in remainder
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v26, v26, 1                           // quotient - 1
v_mul_u32_u24 v27, v26, s[sgprGSUSumIdx+1]         // re-calculate remainder
v_sub_nc_u32 v27, s[sgprLoopCounterL], v27         // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v26       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v27        // remainder
s_mul_i32 s17, s[sgprLoopCounterL], s[sgprGSUSumIdx] // quotient*GSUSumIdx
s_add_u32 s16, 1, s[sgprLoopCounterL]              // quotient+1
s_add_u32 s17, s17, s[sgprGSUSumIdx+1]             // quotient*GSUSumIdx+remainder
s_mul_i32 s16, s16, s[sgprGSUSumIdx]               // (quotient+1)*GSUSumIdx
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cselect_b32 s16, s16, s17                        // (quotient+1)*GSUSumIdx if needed
s_mul_hi_u32 s17, s16, 64                          // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
s_mul_i32 s16, s16, 64                             // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
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
s_lshr_b64 s[sgprShadowLimitA:sgprShadowLimitA+1], s[sgprShadowLimitA:sgprShadowLimitA+1], 1 // Set limit to use bytes (multiple bpe)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 4 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_and_b32 s20, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s20, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_StridedBatchedGemmLoadA
s_mul_i32 s16, 8, s[sgprWorkGroup2]                // Compute Offset into Pointer Array
s_cmp_eq_u32 s[sgprSizesSum], 0x0                  // Don't dereference Pointer array if SizesSum == 0
s_cbranch_scc1 label_StridedBatchedGemmLoadA_End
s_add_u32 s16, s16, s[sgprAddressA+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s17, s[sgprAddressA+1], 0               // Offsetting to the location [Higher half of address]
s_load_b64 s[sgprSrdA:sgprSrdA+1], s[16:17], 0     // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for pointer-array SRD load before reusing base SGPR
s_load_b64 s[16:17], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x80 // Load batchOffsetA from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s16        // Add batch offset to A address (low)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s17       // Add batch offset to A address (high)
s_sub_u32 s[sgprSrdA+0], s[sgprSrdA+0], 4          // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0         // pre-pad to make room for possible pointer shift
s_lshr_b64 s[18:19], s[18:19], 1                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdA+0], s18, s[sgprSrdA+0]        // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s19, s[sgprSrdA+1]       // SRD base = Address+ tileStart1
s_branch label_StridedBatchedGemmLoadA_End
label_StridedBatchedGemmLoadA:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s17, s[sgprStrideAK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s16, s[sgprStrideAK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s18, s18, s16                            // accum wg term to tilestart
s_addc_u32 s19, s19, s17                           // accum wg term to tilestart
s_lshr_b64 s[18:19], s[18:19], 1                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdA+0], s[sgprAddressA+0], s18    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprAddressA+1], s19   // SRD base = Address+ tileStart1
label_StridedBatchedGemmLoadA_End:  /// End Computing the Batch Matrix's base address for Strided Batched
s_mov_b32 s[sgprSrdA+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: block-scale A srd */
s_add_u32 s[sgprStrideScaleA], s[sgprSizeL], s96
s_lshr_b32 s[sgprStrideScaleA], s[sgprStrideScaleA], s97
s_mul_i32 s16, s[sgprWorkGroup0], 64               // scaleA: workgroup row origin
s_mul_i32 s16, s16, s[sgprStrideScaleA]            // scaleA: * row stride
s_mul_i32 s16, s16, 2                              // scaleA: elements -> bytes
s_mul_i32 s17, s[sgprSizeI], s[sgprStrideScaleA]   // scaleA: SizeI * row stride
s_mul_i32 s17, s17, 2                              // scaleA: tensor bytes
s_sub_u32 s[sgprSrdScaleA+2], s17, s16             // scaleA: buffer limit from the workgroup origin
s_add_u32 s[sgprSrdScaleA+0], s[sgprAddressScaleA+0], s16 // scaleA: SRD base lo
s_addc_u32 s[sgprSrdScaleA+1], s[sgprAddressScaleA+1], 0 // scaleA: SRD base hi
s_mov_b32 s[sgprSrdScaleA+3], Srd127_96            // scaleA: set bits 127_96 in SRD
s_mov_b32 s[sgprScaleAPkMagic+0], 0x64006400       // w4a16: two fp16 holding 1024, for the fused mask+OR
s_mov_b32 s[sgprScaleAPkMagic+2], 0xe400e400       // w4a16: two fp16 holding -1024, for the fused bias
s_mov_b32 s[sgprScaleAPkMagic+1], 0x54005400       // w4a16: two fp16 holding 64, for the fused mask+OR
s_mov_b32 s[sgprScaleAPkMagic+3], 0xd400d400       // w4a16: two fp16 holding -64, for the fused bias
s_mov_b32 s[sgprScaleAPkPermute+0], 0x5040100      // w4a16: sequential FP16 pair selector
s_mov_b32 s[sgprScaleAPkPermute+1], 0x7060302      // w4a16: sequential FP16 pair selector

/* global read addresses: block-scale A zero-point srd */
s_mul_i32 s16, s[sgprWorkGroup0], 64               // scaleZeroA: workgroup row origin
s_lshr_b32 s16, s16, 1                             // scaleZeroA: aligned row origin to bytes
s_mul_i32 s16, s16, s[sgprStrideScaleA]            // scaleZeroA: * kGroups
s_add_u32 s17, s[sgprSizeI], 7                     // scaleZeroA: SizeI + 7
s_lshr_b32 s17, s17, 3                             // scaleZeroA: ceil(SizeI/8) words
s_lshl_b32 s17, s17, 2                             // scaleZeroA: words to bytes
s_mul_i32 s17, s17, s[sgprStrideScaleA]            // scaleZeroA: total bytes

/* global read addresses: addresses b */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s19, s[sgprWorkGroup1], 112           // WorkGroup[01] * MT
s_mul_i32 s18, s[sgprWorkGroup1], 112              // WorkGroup[01] * MT
s_mul_hi_u32 s19, s18, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s18, s18, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_and_b32 s16, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_B                        // branch if GSUC == 1
s_mul_hi_u32 s17, 64, s[sgprGSUSumIdx]             // gsuOffset = DepthU*GSUSumIdx
s_mul_i32 s16, 64, s[sgprGSUSumIdx]                // gsuOffset = DepthU*GSUSumIdx
s_branch label_GSUC_B_End
label_GSUC_B:
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum], 6 // s[LoopCounterL] = s[sgprSizesSum] / 64
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0xfff    // Restore GSU
v_cvt_f32_u32 v26, s[sgprGSUSumIdx+1]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v26, v26                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v27, s[sgprLoopCounterL]             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v26, v26, v27                            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v26, v26                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v27, v26, s[sgprGSUSumIdx+1]         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_nc_u32 v27, s[sgprLoopCounterL], v27         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmp_eq_u32 vcc_lo, v27, s[sgprGSUSumIdx+1]       // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, vcc_lo                          // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_nc_u32 v26, 1, v26                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v27, 0                                   // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v27, s[sgprGSUSumIdx+1]       // overflow happened in remainder
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v26, v26, 1                           // quotient - 1
v_mul_u32_u24 v27, v26, s[sgprGSUSumIdx+1]         // re-calculate remainder
v_sub_nc_u32 v27, s[sgprLoopCounterL], v27         // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v26       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v27        // remainder
s_mul_i32 s17, s[sgprLoopCounterL], s[sgprGSUSumIdx] // quotient*GSUSumIdx
s_add_u32 s16, 1, s[sgprLoopCounterL]              // quotient+1
s_add_u32 s17, s17, s[sgprGSUSumIdx+1]             // quotient*GSUSumIdx+remainder
s_mul_i32 s16, s16, s[sgprGSUSumIdx]               // (quotient+1)*GSUSumIdx
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cselect_b32 s16, s16, s17                        // (quotient+1)*GSUSumIdx if needed
s_mul_hi_u32 s17, s16, 64                          // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
s_mul_i32 s16, s16, 64                             // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
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
s_lshl_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], 1 // Set limit to use bytes (multiple bpe)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 8 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_and_b32 s20, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s20, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_StridedBatchedGemmLoadB
s_mul_i32 s16, 8, s[sgprWorkGroup2]                // Compute Offset into Pointer Array
s_cmp_eq_u32 s[sgprSizesSum], 0x0                  // Don't dereference Pointer array if SizesSum == 0
s_cbranch_scc1 label_StridedBatchedGemmLoadB_End
s_add_u32 s16, s16, s[sgprAddressB+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s17, s[sgprAddressB+1], 0               // Offsetting to the location [Higher half of address]
s_load_b64 s[sgprSrdB:sgprSrdB+1], s[16:17], 0     // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for pointer-array SRD load before reusing base SGPR
s_load_b64 s[16:17], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x88 // Load batchOffsetB from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s16        // Add batch offset to B address (low)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s17       // Add batch offset to B address (high)
s_sub_u32 s[sgprSrdB+0], s[sgprSrdB+0], 8          // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0         // pre-pad to make room for possible pointer shift
s_lshl_b64 s[18:19], s[18:19], 1                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdB+0], s18, s[sgprSrdB+0]        // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s19, s[sgprSrdB+1]       // SRD base = Address+ tileStart1
s_branch label_StridedBatchedGemmLoadB_End
label_StridedBatchedGemmLoadB:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s17, s[sgprStrideBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s16, s[sgprStrideBK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s18, s18, s16                            // accum wg term to tilestart
s_addc_u32 s19, s19, s17                           // accum wg term to tilestart
s_lshl_b64 s[18:19], s[18:19], 1                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdB+0], s[sgprAddressB+0], s18    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprAddressB+1], s19   // SRD base = Address+ tileStart1
label_StridedBatchedGemmLoadB_End:  /// End Computing the Batch Matrix's base address for Strided Batched
s_mov_b32 s[sgprSrdB+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: final offsets a */
/* ============================================================= */
v_mul_lo_u32 v26, s[sgprStrideA0I], v[6]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+0+0], vcc_lo, v[24], v[26+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+0+0], 0x8, v[vgprGlobalReadOffsetA+0+0] // add prepad for pointer shift
v_lshrrev_b32 v26, s97, v24                          // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+0], s[sgprStrideScaleA], v6 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+0], v26, v[vgprGlobalReadOffsetScaleA+0] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+0], 1, v[vgprGlobalReadOffsetScaleA+0] // scaleA: elements -> bytes
v_and_b32 v26, 7, v6                               // scaleZeroA: nibble = row & 7
v_lshrrev_b32 v[vgprGlobalReadOffsetA+0], 1, v[vgprGlobalReadOffsetA+0] //  (multiple bpe)
v_mul_lo_u32 v26, s[sgprStrideA0I], v[7]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+1+0], vcc_lo, v[24], v[26+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+1+0], 0x8, v[vgprGlobalReadOffsetA+1+0] // add prepad for pointer shift
v_lshrrev_b32 v26, s97, v24                          // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+1], s[sgprStrideScaleA], v7 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+1], v26, v[vgprGlobalReadOffsetScaleA+1] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+1], 1, v[vgprGlobalReadOffsetScaleA+1] // scaleA: elements -> bytes
v_and_b32 v26, 7, v7                               // scaleZeroA: nibble = row & 7
v_lshrrev_b32 v[vgprGlobalReadOffsetA+1], 1, v[vgprGlobalReadOffsetA+1] //  (multiple bpe)
v_mul_lo_u32 v26, s[sgprStrideA0I], v[8]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+2+0], vcc_lo, v[24], v[26+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+2+0], 0x8, v[vgprGlobalReadOffsetA+2+0] // add prepad for pointer shift
v_lshrrev_b32 v26, s97, v24                          // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+2], s[sgprStrideScaleA], v8 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+2], v26, v[vgprGlobalReadOffsetScaleA+2] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+2], 1, v[vgprGlobalReadOffsetScaleA+2] // scaleA: elements -> bytes
v_and_b32 v26, 7, v8                               // scaleZeroA: nibble = row & 7
v_lshrrev_b32 v[vgprGlobalReadOffsetA+2], 1, v[vgprGlobalReadOffsetA+2] //  (multiple bpe)
v_mul_lo_u32 v26, s[sgprStrideA0I], v[9]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+3+0], vcc_lo, v[24], v[26+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+3+0], 0x8, v[vgprGlobalReadOffsetA+3+0] // add prepad for pointer shift
v_lshrrev_b32 v26, s97, v24                          // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+3], s[sgprStrideScaleA], v9 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+3], v26, v[vgprGlobalReadOffsetScaleA+3] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+3], 1, v[vgprGlobalReadOffsetScaleA+3] // scaleA: elements -> bytes
v_and_b32 v26, 7, v9                               // scaleZeroA: nibble = row & 7
v_lshrrev_b32 v[vgprGlobalReadOffsetA+3], 1, v[vgprGlobalReadOffsetA+3] //  (multiple bpe)
/* ============================================================= */

/* global read addresses: final offsets b */
/* ============================================================= */
v_mul_lo_u32 v6, s[sgprStrideB1J], v[10]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+0+0], vcc_lo, v[25], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+0+0], 0x4, v[vgprGlobalReadOffsetB+0+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+0], 1, v[vgprGlobalReadOffsetB+0] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[11]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+1+0], vcc_lo, v[25], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+1+0], 0x4, v[vgprGlobalReadOffsetB+1+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+1], 1, v[vgprGlobalReadOffsetB+1] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[12]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+2+0], vcc_lo, v[25], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+2+0], 0x4, v[vgprGlobalReadOffsetB+2+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+2], 1, v[vgprGlobalReadOffsetB+2] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[13]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+3+0], vcc_lo, v[25], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+3+0], 0x4, v[vgprGlobalReadOffsetB+3+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+3], 1, v[vgprGlobalReadOffsetB+3] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[14]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+4+0], vcc_lo, v[25], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+4+0], 0x4, v[vgprGlobalReadOffsetB+4+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+4], 1, v[vgprGlobalReadOffsetB+4] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[15]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+5+0], vcc_lo, v[25], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+5+0], 0x4, v[vgprGlobalReadOffsetB+5+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+5], 1, v[vgprGlobalReadOffsetB+5] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[16]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+6+0], vcc_lo, v[25], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+6+0], 0x4, v[vgprGlobalReadOffsetB+6+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+6], 1, v[vgprGlobalReadOffsetB+6] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[17]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+7+0], vcc_lo, v[25], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+7+0], 0x4, v[vgprGlobalReadOffsetB+7+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+7], 1, v[vgprGlobalReadOffsetB+7] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[18]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+8+0], vcc_lo, v[25], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+8+0], 0x4, v[vgprGlobalReadOffsetB+8+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+8], 1, v[vgprGlobalReadOffsetB+8] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[19]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+9+0], vcc_lo, v[25], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+9+0], 0x4, v[vgprGlobalReadOffsetB+9+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+9], 1, v[vgprGlobalReadOffsetB+9] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[20]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+10+0], vcc_lo, v[25], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+10+0], 0x4, v[vgprGlobalReadOffsetB+10+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+10], 1, v[vgprGlobalReadOffsetB+10] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[21]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+11+0], vcc_lo, v[25], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+11+0], 0x4, v[vgprGlobalReadOffsetB+11+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+11], 1, v[vgprGlobalReadOffsetB+11] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[22]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+12+0], vcc_lo, v[25], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+12+0], 0x4, v[vgprGlobalReadOffsetB+12+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+12], 1, v[vgprGlobalReadOffsetB+12] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[23]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+13+0], vcc_lo, v[25], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+13+0], 0x4, v[vgprGlobalReadOffsetB+13+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+13], 1, v[vgprGlobalReadOffsetB+13] //  (multiple bpe)
/* ============================================================= */

/* global read addresses: increments a */
s_and_b32 s17, s[sgprGSU], 0xfff                   // Restore GSU
s_mov_b32 s[sgprGlobalReadIncsA+0], 32             // GSU*DepthU*Bpe*MI_dim(1)
s_mul_i32 s17, s17, s[sgprGlobalReadIncsA+0]       // GSU*DepthU*Bpe*MI_dim(1)
s_and_b32 s16, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cselect_b32 s[sgprGlobalReadIncsA+0], s[sgprGlobalReadIncsA+0], s17 // incrA (unrollIdx)

/* global read addresses: increments b */
s_and_b32 s17, s[sgprGSU], 0xfff                   // Restore GSU
s_mov_b32 s[sgprGlobalReadIncsB+0], 128            // GSU*DepthU*Bpe*MI_dim(1)
s_mul_i32 s17, s17, s[sgprGlobalReadIncsB+0]       // GSU*DepthU*Bpe*MI_dim(1)
s_and_b32 s16, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cselect_b32 s[sgprGlobalReadIncsB+0], s[sgprGlobalReadIncsB+0], s17 // incrB (unrollIdx)
/* declare loop num iterations */
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 6 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 64
s_and_b32 s16, s[sgprGSU], 0xfff                   // Restore GSU
s_cmp_eq_u32 s16, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU_1                         // branch if GSU == 1
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0xfff    // Restore GSU
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
s_mul_hi_u32 s17, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_i32 s16, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_hi_u32 s[sgprWrapUA+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
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
s_mul_hi_u32 s17, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_i32 s16, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_hi_u32 s[sgprWrapUB+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
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
buffer_load_b32 v[vgprG2LA+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b32 v[vgprG2LA+6], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b32 v[vgprG2LA+10], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b32 v[vgprG2LA+14], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_3_0

/* global read block-scale A */
buffer_load_d16_b16 v[vgprG2LScaleA+0], v[vgprGlobalReadOffsetScaleA+0], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 0
buffer_load_d16_b16 v[vgprG2LScaleA+1], v[vgprGlobalReadOffsetScaleA+1], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 1
buffer_load_d16_b16 v[vgprG2LScaleA+2], v[vgprGlobalReadOffsetScaleA+2], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 2
buffer_load_d16_b16 v[vgprG2LScaleA+3], v[vgprGlobalReadOffsetScaleA+3], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 3
buffer_load_b64 v[vgprG2LB+0:vgprG2LB+0+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b64 v[vgprG2LB+2:vgprG2LB+2+1], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b64 v[vgprG2LB+4:vgprG2LB+4+1], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b64 v[vgprG2LB+6:vgprG2LB+6+1], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0
buffer_load_b64 v[vgprG2LB+8:vgprG2LB+8+1], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_4_0
buffer_load_b64 v[vgprG2LB+10:vgprG2LB+10+1], v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_5_0
buffer_load_b64 v[vgprG2LB+12:vgprG2LB+12+1], v[vgprGlobalReadOffsetB+6], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_6_0
buffer_load_b64 v[vgprG2LB+14:vgprG2LB+14+1], v[vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_7_0
buffer_load_b64 v[vgprG2LB+16:vgprG2LB+16+1], v[vgprGlobalReadOffsetB+8], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_8_0
buffer_load_b64 v[vgprG2LB+18:vgprG2LB+18+1], v[vgprGlobalReadOffsetB+9], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_9_0
buffer_load_b64 v[vgprG2LB+20:vgprG2LB+20+1], v[vgprGlobalReadOffsetB+10], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_10_0
buffer_load_b64 v[vgprG2LB+22:vgprG2LB+22+1], v[vgprGlobalReadOffsetB+11], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_11_0
buffer_load_b64 v[vgprG2LB+24:vgprG2LB+24+1], v[vgprGlobalReadOffsetB+12], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_12_0
buffer_load_b64 v[vgprG2LB+26:vgprG2LB+26+1], v[vgprGlobalReadOffsetB+13], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_13_0

/* global read inc A loopL */
s_add_u32 s18, s[sgprLoopCounterL], 1              // remove pf(1)
s_cmp_eq_u32 s[sgprStaggerUIter], s18              // Is this wrapIter? (pf)
s_cselect_b32 s16, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s17, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s16        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s17       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s16 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s17 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* global read inc block-scale A (4 bytes) */
s_add_u32 s101, s101, 1
s_and_b32 s102, s101, s98
s_cselect_b32 s102, 0, s99
s_cselect_b32 s103, 0, s100
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], s102
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], s102

/* global read inc B loopL */
s_add_u32 s18, s[sgprLoopCounterL], 1              // remove pf(1)
s_cmp_eq_u32 s[sgprStaggerUIter], s18              // Is this wrapIter? (pf)
s_cselect_b32 s16, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s17, s[sgprWrapUB+1], 0              // incUpper <- ?
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
s_and_b32 s87, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s87, 1                                // GSU == 1 ?
s_cbranch_scc1 label_ArgTypeCheckD                 // Handling General Batched GEMM SRD initialization
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], s[sgprAddressD+0:sgprAddressD+0+1] // init SRD base address
s_branch label_GeneralBatchedGemmSrdInitiationD_End // End of handling General Batched GEMM SRD initialization
label_ArgTypeCheckD:  /// Check if ArgType is for General Batched GEMM for D
s_and_b32 s87, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s87, 3                                // ArgType == 3 for General Batched GEMM
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

s_and_b32 s87, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s87, 1                                // GSU == 1 ?
s_cbranch_scc1 label_ArgTypeCheckC                 // Handling General Batched GEMM SRD initialization
s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], s[sgprAddressC+0:sgprAddressC+0+1] // init SRD base address
s_branch label_GeneralBatchedGemmSrdInitiationC_End // End of handling General Batched GEMM SRD initialization
label_ArgTypeCheckC:  /// Check if ArgType is for General Batched GEMM for C
s_and_b32 s87, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s87, 3                                // ArgType == 3 for General Batched GEMM
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


s_mul_i32 s90, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_and_b32 s89, s[sgprGSU], 0xfff                   // Restore GSU
s_mul_hi_u32 s89, s90, s[sgprStrideC1J]            // ScaleC s90 by Stride
s_mul_i32 s88, s90, s[sgprStrideC1J]               // ScaleC s90 by Stride
s_lshl_b64 s[88:89], s[88:89], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s88        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s89       // add hi to SRD
s_and_b32 s89, s[sgprGSU], 0xfff                   // Restore GSU
s_mul_hi_u32 s89, s90, s[sgprStrideD1J]            // ScaleD s90 by Stride
s_mul_i32 s88, s90, s[sgprStrideD1J]               // ScaleD s90 by Stride
s_lshl_b64 s[88:89], s[88:89], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s88        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s89       // add hi to SRD

s_and_b32 s89, s[sgprGSU], 0xfff                   // Restore GSU
s_cmp_eq_u32 s89, 1                                // GSU == 1 ?
s_cbranch_scc0 label_StridedBatchedGemmLoadC
s_and_b32 s87, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s87, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_GeneralBatchedGemmLoadC
label_StridedBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s89, s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride
s_mul_i32 s88, s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride
s_lshl_b64 s[88:89], s[88:89], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s88        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s89       // add hi to SRD
s_branch label_GeneralBatchedGemmLoadC_End
label_GeneralBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for General Batched GEMM
s_mul_i32 s88, 8, s[sgprWorkGroup2]                // Compute stride in bytes into Pointer Array
s_add_u32 s88, s88, s[sgprAddressC+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s89, s[sgprAddressC+1], 0               // Offsetting to the location [Higher half of address]
s_load_b64 s[88:89], s[88:89], 0                   // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for the Matrix Address Load from the Pointer Array
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s88        // Offsetting within the Batch Matrix [Lower half of address]
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s89       // Offsetting within the Batch Matrix [Higher half of address]
s_load_b64 s[88:89], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x78 // Load batchOffsetC from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s88        // Add matrix address to SRD (low)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s89       // Add matrix address to SRD (high)
label_GeneralBatchedGemmLoadC_End:  /// End of label GeneralBatchedGemmLoadC
s_and_b32 s89, s[sgprGSU], 0xfff                   // Restore GSU
s_cmp_eq_u32 s89, 1                                // GSU == 1 ?
s_cbranch_scc0 label_StridedBatchedGemmLoadD
s_and_b32 s87, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s87, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_GeneralBatchedGemmLoadD
label_StridedBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s89, s[sgprWorkGroup2], s[sgprStrideDK] // ScaleD s[sgprWorkGroup2] by Stride
s_mul_i32 s88, s[sgprWorkGroup2], s[sgprStrideDK]  // ScaleD s[sgprWorkGroup2] by Stride
s_lshl_b64 s[88:89], s[88:89], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s88        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s89       // add hi to SRD
s_branch label_GeneralBatchedGemmLoadD_End
label_GeneralBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for General Batched GEMM
s_mul_i32 s88, 8, s[sgprWorkGroup2]                // Compute stride in bytes into Pointer Array
s_add_u32 s88, s88, s[sgprAddressD+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s89, s[sgprAddressD+1], 0               // Offsetting to the location [Higher half of address]
s_load_b64 s[88:89], s[88:89], 0                   // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for the Matrix Address Load from the Pointer Array
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s88        // Offsetting within the Batch Matrix [Lower half of address]
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s89       // Offsetting within the Batch Matrix [Higher half of address]
s_load_b64 s[88:89], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x70 // Load batchOffsetD from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s88        // Add matrix address to SRD (low)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s89       // Add matrix address to SRD (high)
label_GeneralBatchedGemmLoadD_End:  /// End of label GeneralBatchedGemmLoadD

s_and_b32 s87, s[sgprGSU], 0xfff                   // Restore GSU
s_cmp_eq_u32 s87, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU_2                         // branch if GSU == 1
// GSU Output Buffer offset: Free0 + (Free1-1)*StrideC1J + (Free2-1)*StrideCK * GSUIdx * bpe%s
s_mul_hi_u32 s89, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_mul_i32 s88, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_sub_u32 s87, s[sgprSizesFree+1], 1               // Free1
s_mul_i32 s87, s87, s[sgprGSUSumIdx]               // Free1
s_mul_hi_u32 s90, s87, s[sgprStrideC1J]            // Free1
s_mul_i32 s87, s87, s[sgprStrideC1J]               // Free1
s_add_u32 s88, s88, s87                            // Free1
s_addc_u32 s89, s89, s90                           // Free1
s_sub_u32 s87, s[sgprSizesFree+2], 1               // Free2
s_mul_i32 s87, s87, s[sgprGSUSumIdx]               // Free2
s_mul_hi_u32 s90, s87, s[sgprStrideCK]             // Free2
s_mul_i32 s87, s87, s[sgprStrideCK]                // Free2
s_add_u32 s88, s88, s87                            // Free2
s_addc_u32 s89, s89, s90                           // Free2
s_lshl_b64 s[88:89], s[88:89], 2                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s88        // add lo GSU offset to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s89       // add hi GSU offset to SRD
label_GSU_2:
.set sgprGSULog2BpeC, UNDEF
.set sgprAddressC, UNDEF

/* initC: remove ValuC vgpr buffer [0...56) from pool */

/* initC: remove acc vgpr buffer [0...0) from pool */

/* initC: remove ValuA/B vgpr buffer [100...165) from pool */
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
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?

/* after InitC, skip to end of prefetch last iter if numIter==0 */

/* label_PrefetchGlobalLastIterEnd */
s_cbranch_scc0 label_NoBranch_0                    // Only branch on scc1
s_getpc_b64 s[88:89]                               // addr of next instr
s_add_i32 s90, label_PrefetchGlobalLastIterEnd, 4  // target branch offset
s_add_u32 s88, s88, s90                            // add target branch offset
s_addc_u32 s89, s89, 0                             // add high and carry
s_setpc_b64 s[88:89]                               // branch to label_PrefetchGlobalLastIterEnd
label_NoBranch_0:
s_waitcnt vmcnt(0)                                 // wait for global read

/* local write a */
v_pack_b32_f16 v214, v[vgprG2LScaleA+0], v[vgprG2LScaleA+0]
v_and_or_b32 v216, v[vgprG2LA+2+0], s92, 0x64006400
v_and_or_b32 v217, v[vgprG2LA+2+0], s93, 0x54005400
v_lshrrev_b32 v213, 8, v[vgprG2LA+2+0]
v_and_or_b32 v218, v213, s92, 0x64006400
v_and_or_b32 v219, v213, s93, 0x54005400
v_pk_add_f16 v216, v216, s94
v_pk_add_f16 v217, v217, s95
v_pk_add_f16 v218, v218, s94
v_pk_add_f16 v219, v219, s95
v_pk_mul_f16 v216, v216, v214
v_pk_mul_f16 v217, v217, v214
v_pk_mul_f16 v218, v218, v214
v_pk_mul_f16 v219, v219, v214
v_perm_b32 v[vgprG2LA+0+0], v217, v216, 0x05040100
v_perm_b32 v[vgprG2LA+0+1], v219, v218, 0x05040100
v_perm_b32 v[vgprG2LA+0+2], v217, v216, 0x07060302
v_perm_b32 v[vgprG2LA+0+3], v219, v218, 0x07060302
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
v_pack_b32_f16 v214, v[vgprG2LScaleA+1], v[vgprG2LScaleA+1]
v_and_or_b32 v216, v[vgprG2LA+6+0], s92, 0x64006400
v_and_or_b32 v217, v[vgprG2LA+6+0], s93, 0x54005400
v_lshrrev_b32 v213, 8, v[vgprG2LA+6+0]
v_and_or_b32 v218, v213, s92, 0x64006400
v_and_or_b32 v219, v213, s93, 0x54005400
v_pk_add_f16 v216, v216, s94
v_pk_add_f16 v217, v217, s95
v_pk_add_f16 v218, v218, s94
v_pk_add_f16 v219, v219, s95
v_pk_mul_f16 v216, v216, v214
v_pk_mul_f16 v217, v217, v214
v_pk_mul_f16 v218, v218, v214
v_pk_mul_f16 v219, v219, v214
v_perm_b32 v[vgprG2LA+4+0], v217, v216, 0x05040100
v_perm_b32 v[vgprG2LA+4+1], v219, v218, 0x05040100
v_perm_b32 v[vgprG2LA+4+2], v217, v216, 0x07060302
v_perm_b32 v[vgprG2LA+4+3], v219, v218, 0x07060302
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:2304 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 2304 sync LDS0
v_pack_b32_f16 v214, v[vgprG2LScaleA+2], v[vgprG2LScaleA+2]
v_and_or_b32 v216, v[vgprG2LA+10+0], s92, 0x64006400
v_and_or_b32 v217, v[vgprG2LA+10+0], s93, 0x54005400
v_lshrrev_b32 v213, 8, v[vgprG2LA+10+0]
v_and_or_b32 v218, v213, s92, 0x64006400
v_and_or_b32 v219, v213, s93, 0x54005400
v_pk_add_f16 v216, v216, s94
v_pk_add_f16 v217, v217, s95
v_pk_add_f16 v218, v218, s94
v_pk_add_f16 v219, v219, s95
v_pk_mul_f16 v216, v216, v214
v_pk_mul_f16 v217, v217, v214
v_pk_mul_f16 v218, v218, v214
v_pk_mul_f16 v219, v219, v214
v_perm_b32 v[vgprG2LA+8+0], v217, v216, 0x05040100
v_perm_b32 v[vgprG2LA+8+1], v219, v218, 0x05040100
v_perm_b32 v[vgprG2LA+8+2], v217, v216, 0x07060302
v_perm_b32 v[vgprG2LA+8+3], v219, v218, 0x07060302
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:4608 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 4608 sync LDS0
v_pack_b32_f16 v214, v[vgprG2LScaleA+3], v[vgprG2LScaleA+3]
v_and_or_b32 v216, v[vgprG2LA+14+0], s92, 0x64006400
v_and_or_b32 v217, v[vgprG2LA+14+0], s93, 0x54005400
v_lshrrev_b32 v213, 8, v[vgprG2LA+14+0]
v_and_or_b32 v218, v213, s92, 0x64006400
v_and_or_b32 v219, v213, s93, 0x54005400
v_pk_add_f16 v216, v216, s94
v_pk_add_f16 v217, v217, s95
v_pk_add_f16 v218, v218, s94
v_pk_add_f16 v219, v219, s95
v_pk_mul_f16 v216, v216, v214
v_pk_mul_f16 v217, v217, v214
v_pk_mul_f16 v218, v218, v214
v_pk_mul_f16 v219, v219, v214
v_perm_b32 v[vgprG2LA+12+0], v217, v216, 0x05040100
v_perm_b32 v[vgprG2LA+12+1], v219, v218, 0x05040100
v_perm_b32 v[vgprG2LA+12+2], v217, v216, 0x07060302
v_perm_b32 v[vgprG2LA+12+3], v219, v218, 0x07060302
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+12:vgprG2LA+12+3] offset:6912 // lwoA_0_0_3_0 = (0*LSCA)*(MT0I+PAD) + (3*LSPA) = 6912 sync LDS0

/* local write b */
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+1] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+2:vgprG2LB+2+1] offset:1152 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 1152 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4:vgprG2LB+4+1] offset:2304 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 2304 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+6:vgprG2LB+6+1] offset:3456 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 3456 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8:vgprG2LB+8+1] offset:4608 // lwoB_0_0_4_0 = (0*LSCB)*(MT1J+PAD) + (4*LSPB) = 4608 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+10:vgprG2LB+10+1] offset:5760 // lwoB_0_0_5_0 = (0*LSCB)*(MT1J+PAD) + (5*LSPB) = 5760 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12:vgprG2LB+12+1] offset:6912 // lwoB_0_0_6_0 = (0*LSCB)*(MT1J+PAD) + (6*LSPB) = 6912 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+14:vgprG2LB+14+1] offset:8064 // lwoB_0_0_7_0 = (0*LSCB)*(MT1J+PAD) + (7*LSPB) = 8064 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+16:vgprG2LB+16+1] offset:9216 // lwoB_0_0_8_0 = (0*LSCB)*(MT1J+PAD) + (8*LSPB) = 9216 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+18:vgprG2LB+18+1] offset:10368 // lwoB_0_0_9_0 = (0*LSCB)*(MT1J+PAD) + (9*LSPB) = 10368 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+20:vgprG2LB+20+1] offset:11520 // lwoB_0_0_10_0 = (0*LSCB)*(MT1J+PAD) + (10*LSPB) = 11520 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+22:vgprG2LB+22+1] offset:12672 // lwoB_0_0_11_0 = (0*LSCB)*(MT1J+PAD) + (11*LSPB) = 12672 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+24:vgprG2LB+24+1] offset:13824 // lwoB_0_0_12_0 = (0*LSCB)*(MT1J+PAD) + (12*LSPB) = 13824 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+26:vgprG2LB+26+1] offset:14976 // lwoB_0_0_13_0 = (0*LSCB)*(MT1J+PAD) + (13*LSPB) = 14976 sync LDS0

/* local write swap a */

/* local write swap b */

/******************************************/
/* Unrolled Loop(s) - Begin               */
/******************************************/
label_openLoopL:
s_cmp_le_u32 s[sgprLoopCounterL], 0x1              // LoopCounterL < EndCounter
s_cbranch_scc1 label_LoopEndL                      // do not enter LoopL
.align 16
label_LoopBeginL:

/******************************************/
/* Unrolled Loop 1/1 - Begin              */
/******************************************/
s_waitcnt lgkmcnt(0)                               // 1wait for local write
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier                                          // 4sync for global read, PGR->LW needs sync

/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */

/* iter 0 */
/*  grEndMfmaIndex:18, lwStartMfmaIndex:27, lwEndMfmaIndex:27  */
/*  numMfmaForLR:6, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:26 */
/*  mfmaIndex:0  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:2304 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:2320 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b32 v[vgprG2LA+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:1  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:4608 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:4624 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b32 v[vgprG2LA+6], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_1_0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:2  */
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:6912 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:6928 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b32 v[vgprG2LA+10], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_2_0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:3  */
ds_load_b128 v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprLocalReadAddrB+0] offset:9216 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[vgprLocalReadAddrB+0] offset:9232 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b32 v[vgprG2LA+14], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_3_0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:4  */
ds_load_b128 v[vgprValuB_X0_I0+40:vgprValuB_X0_I0+40+3], v[vgprLocalReadAddrB+0] offset:11520 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=5 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+44:vgprValuB_X0_I0+44+3], v[vgprLocalReadAddrB+0] offset:11536 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=5 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* global read block-scale A */
buffer_load_d16_b16 v[vgprG2LScaleA+0], v[vgprGlobalReadOffsetScaleA+0], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 0
buffer_load_d16_b16 v[vgprG2LScaleA+1], v[vgprGlobalReadOffsetScaleA+1], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 1
buffer_load_d16_b16 v[vgprG2LScaleA+2], v[vgprGlobalReadOffsetScaleA+2], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 2
buffer_load_d16_b16 v[vgprG2LScaleA+3], v[vgprGlobalReadOffsetScaleA+3], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 3
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:5  */
ds_load_b128 v[vgprValuB_X0_I0+48:vgprValuB_X0_I0+48+3], v[vgprLocalReadAddrB+0] offset:13824 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=6 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+52:vgprValuB_X0_I0+52+3], v[vgprLocalReadAddrB+0] offset:13840 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=6 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b64 v[vgprG2LB+0:vgprG2LB+0+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+40+0+0:vgprValuB_X0_I0+40+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:6  */
/* localReadsVacancy: latencyLeft 5 */
buffer_load_b64 v[vgprG2LB+2:vgprG2LB+2+1], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+48+0+0:vgprValuB_X0_I0+48+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=14 */

/* iter 1 */
/*  grEndMfmaIndex:18, lwStartMfmaIndex:27, lwEndMfmaIndex:27  */
/*  numMfmaForLR:6, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:26 */
/*  mfmaIndex:7  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:2336 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:2352 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b64 v[vgprG2LB+4:vgprG2LB+4+1], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:8  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:4640 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:4656 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b64 v[vgprG2LB+6:vgprG2LB+6+1], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:9  */
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:6944 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:6960 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b64 v[vgprG2LB+8:vgprG2LB+8+1], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_4_0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:10  */
ds_load_b128 v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprLocalReadAddrB+0] offset:9248 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[vgprLocalReadAddrB+0] offset:9264 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b64 v[vgprG2LB+10:vgprG2LB+10+1], v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_5_0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:11  */
ds_load_b128 v[vgprValuB_X0_I0+40:vgprValuB_X0_I0+40+3], v[vgprLocalReadAddrB+0] offset:11552 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=5 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+44:vgprValuB_X0_I0+44+3], v[vgprLocalReadAddrB+0] offset:11568 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=5 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b64 v[vgprG2LB+12:vgprG2LB+12+1], v[vgprGlobalReadOffsetB+6], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_6_0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:12  */
ds_load_b128 v[vgprValuB_X0_I0+48:vgprValuB_X0_I0+48+3], v[vgprLocalReadAddrB+0] offset:13856 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=6 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+52:vgprValuB_X0_I0+52+3], v[vgprLocalReadAddrB+0] offset:13872 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=6 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b64 v[vgprG2LB+14:vgprG2LB+14+1], v[vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_7_0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+40+0+0:vgprValuB_X0_I0+40+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:13  */
/* localReadsVacancy: latencyLeft 5 */
buffer_load_b64 v[vgprG2LB+16:vgprG2LB+16+1], v[vgprGlobalReadOffsetB+8], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_8_0
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+48+0+0:vgprValuB_X0_I0+48+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=0 readsPerIterB=14 */

/* iter 2 */
/*  grEndMfmaIndex:18, lwStartMfmaIndex:27, lwEndMfmaIndex:27  */
/*  numMfmaForLR:6, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:26 */
/*  mfmaIndex:14  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:2368 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:2384 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b64 v[vgprG2LB+18:vgprG2LB+18+1], v[vgprGlobalReadOffsetB+9], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_9_0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:15  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:4672 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:4688 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b64 v[vgprG2LB+20:vgprG2LB+20+1], v[vgprGlobalReadOffsetB+10], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_10_0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:16  */
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:6976 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:6992 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b64 v[vgprG2LB+22:vgprG2LB+22+1], v[vgprGlobalReadOffsetB+11], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_11_0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:17  */
ds_load_b128 v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprLocalReadAddrB+0] offset:9280 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[vgprLocalReadAddrB+0] offset:9296 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b64 v[vgprG2LB+24:vgprG2LB+24+1], v[vgprGlobalReadOffsetB+12], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_12_0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:18  */
ds_load_b128 v[vgprValuB_X0_I0+40:vgprValuB_X0_I0+40+3], v[vgprLocalReadAddrB+0] offset:11584 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=5 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+44:vgprValuB_X0_I0+44+3], v[vgprLocalReadAddrB+0] offset:11600 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=5 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b64 v[vgprG2LB+26:vgprG2LB+26+1], v[vgprGlobalReadOffsetB+13], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_13_0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:19  */
ds_load_b128 v[vgprValuB_X0_I0+48:vgprValuB_X0_I0+48+3], v[vgprLocalReadAddrB+0] offset:13888 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=6 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+52:vgprValuB_X0_I0+52+3], v[vgprLocalReadAddrB+0] offset:13904 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=6 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s88, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s89, s[sgprWrapUA+1], 0              // incUpper <- ?
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+40+0+0:vgprValuB_X0_I0+40+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:20  */
/* localReadsVacancy: latencyLeft 5 */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s88        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s89       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s88 // limit -= inc)
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+48+0+0:vgprValuB_X0_I0+48+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=0 readsPerIterB=14 */

/* iter 3 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:18, lwStartMfmaIndex:27, lwEndMfmaIndex:27  */
/*  numMfmaForLR:6, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:26 */
/*  mfmaIndex:21  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:2400 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:2416 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s89 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:22  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:4704 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:4720 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* global read inc block-scale A (4 bytes) */
s_add_u32 s101, s101, 1
s_and_b32 s102, s101, s98
s_cselect_b32 s102, 0, s99
s_cselect_b32 s103, 0, s100
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], s102
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], s102
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:23  */
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:7008 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:7024 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:24  */
ds_load_b128 v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprLocalReadAddrB+0] offset:9312 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[vgprLocalReadAddrB+0] offset:9328 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s88, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s89, s[sgprWrapUB+1], 0              // incUpper <- ?
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:25  */
ds_load_b128 v[vgprValuB_X0_I0+40:vgprValuB_X0_I0+40+3], v[vgprLocalReadAddrB+0] offset:11616 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=5 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+44:vgprValuB_X0_I0+44+3], v[vgprLocalReadAddrB+0] offset:11632 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=5 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s88        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s89       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s88 // limit -= inc)
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:26  */
/* schedule remaining localreads for one buffer scheduling */
ds_load_b128 v[vgprValuB_X0_I0+48:vgprValuB_X0_I0+48+3], v[vgprLocalReadAddrB+0] offset:13920 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=6 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+52:vgprValuB_X0_I0+52+3], v[vgprLocalReadAddrB+0] offset:13936 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=6 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s89 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+40+0+0:vgprValuB_X0_I0+40+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:27  */
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(17)                                // wait for global read before writing to local
v_pack_b32_f16 v214, v[vgprG2LScaleA+0], v[vgprG2LScaleA+0]
v_and_or_b32 v216, v[vgprG2LA+2+0], s92, 0x64006400
v_and_or_b32 v217, v[vgprG2LA+2+0], s93, 0x54005400
v_lshrrev_b32 v213, 8, v[vgprG2LA+2+0]
v_and_or_b32 v218, v213, s92, 0x64006400
v_and_or_b32 v219, v213, s93, 0x54005400
v_pk_add_f16 v216, v216, s94
v_pk_add_f16 v217, v217, s95
v_pk_add_f16 v218, v218, s94
v_pk_add_f16 v219, v219, s95
v_pk_mul_f16 v216, v216, v214
v_pk_mul_f16 v217, v217, v214
v_pk_mul_f16 v218, v218, v214
v_pk_mul_f16 v219, v219, v214
v_perm_b32 v[vgprG2LA+0+0], v217, v216, 0x05040100
v_perm_b32 v[vgprG2LA+0+1], v219, v218, 0x05040100
v_perm_b32 v[vgprG2LA+0+2], v217, v216, 0x07060302
v_perm_b32 v[vgprG2LA+0+3], v219, v218, 0x07060302
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(16)                                // wait for global read before writing to local
v_pack_b32_f16 v214, v[vgprG2LScaleA+1], v[vgprG2LScaleA+1]
v_and_or_b32 v216, v[vgprG2LA+6+0], s92, 0x64006400
v_and_or_b32 v217, v[vgprG2LA+6+0], s93, 0x54005400
v_lshrrev_b32 v213, 8, v[vgprG2LA+6+0]
v_and_or_b32 v218, v213, s92, 0x64006400
v_and_or_b32 v219, v213, s93, 0x54005400
v_pk_add_f16 v216, v216, s94
v_pk_add_f16 v217, v217, s95
v_pk_add_f16 v218, v218, s94
v_pk_add_f16 v219, v219, s95
v_pk_mul_f16 v216, v216, v214
v_pk_mul_f16 v217, v217, v214
v_pk_mul_f16 v218, v218, v214
v_pk_mul_f16 v219, v219, v214
v_perm_b32 v[vgprG2LA+4+0], v217, v216, 0x05040100
v_perm_b32 v[vgprG2LA+4+1], v219, v218, 0x05040100
v_perm_b32 v[vgprG2LA+4+2], v217, v216, 0x07060302
v_perm_b32 v[vgprG2LA+4+3], v219, v218, 0x07060302
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:2304 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 2304 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(15)                                // wait for global read before writing to local
v_pack_b32_f16 v214, v[vgprG2LScaleA+2], v[vgprG2LScaleA+2]
v_and_or_b32 v216, v[vgprG2LA+10+0], s92, 0x64006400
v_and_or_b32 v217, v[vgprG2LA+10+0], s93, 0x54005400
v_lshrrev_b32 v213, 8, v[vgprG2LA+10+0]
v_and_or_b32 v218, v213, s92, 0x64006400
v_and_or_b32 v219, v213, s93, 0x54005400
v_pk_add_f16 v216, v216, s94
v_pk_add_f16 v217, v217, s95
v_pk_add_f16 v218, v218, s94
v_pk_add_f16 v219, v219, s95
v_pk_mul_f16 v216, v216, v214
v_pk_mul_f16 v217, v217, v214
v_pk_mul_f16 v218, v218, v214
v_pk_mul_f16 v219, v219, v214
v_perm_b32 v[vgprG2LA+8+0], v217, v216, 0x05040100
v_perm_b32 v[vgprG2LA+8+1], v219, v218, 0x05040100
v_perm_b32 v[vgprG2LA+8+2], v217, v216, 0x07060302
v_perm_b32 v[vgprG2LA+8+3], v219, v218, 0x07060302
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:4608 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 4608 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(14)                                // wait for global read before writing to local
v_pack_b32_f16 v214, v[vgprG2LScaleA+3], v[vgprG2LScaleA+3]
v_and_or_b32 v216, v[vgprG2LA+14+0], s92, 0x64006400
v_and_or_b32 v217, v[vgprG2LA+14+0], s93, 0x54005400
v_lshrrev_b32 v213, 8, v[vgprG2LA+14+0]
v_and_or_b32 v218, v213, s92, 0x64006400
v_and_or_b32 v219, v213, s93, 0x54005400
v_pk_add_f16 v216, v216, s94
v_pk_add_f16 v217, v217, s95
v_pk_add_f16 v218, v218, s94
v_pk_add_f16 v219, v219, s95
v_pk_mul_f16 v216, v216, v214
v_pk_mul_f16 v217, v217, v214
v_pk_mul_f16 v218, v218, v214
v_pk_mul_f16 v219, v219, v214
v_perm_b32 v[vgprG2LA+12+0], v217, v216, 0x05040100
v_perm_b32 v[vgprG2LA+12+1], v219, v218, 0x05040100
v_perm_b32 v[vgprG2LA+12+2], v217, v216, 0x07060302
v_perm_b32 v[vgprG2LA+12+3], v219, v218, 0x07060302
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+12:vgprG2LA+12+3] offset:6912 // lwoA_0_0_3_0 = (0*LSCA)*(MT0I+PAD) + (3*LSPA) = 6912 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(13)                                // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+1] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(12)                                // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+2:vgprG2LB+2+1] offset:1152 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 1152 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(11)                                // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4:vgprG2LB+4+1] offset:2304 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 2304 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(10)                                // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+6:vgprG2LB+6+1] offset:3456 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 3456 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(9)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8:vgprG2LB+8+1] offset:4608 // lwoB_0_0_4_0 = (0*LSCB)*(MT1J+PAD) + (4*LSPB) = 4608 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(8)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+10:vgprG2LB+10+1] offset:5760 // lwoB_0_0_5_0 = (0*LSCB)*(MT1J+PAD) + (5*LSPB) = 5760 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12:vgprG2LB+12+1] offset:6912 // lwoB_0_0_6_0 = (0*LSCB)*(MT1J+PAD) + (6*LSPB) = 6912 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(6)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+14:vgprG2LB+14+1] offset:8064 // lwoB_0_0_7_0 = (0*LSCB)*(MT1J+PAD) + (7*LSPB) = 8064 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(5)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+16:vgprG2LB+16+1] offset:9216 // lwoB_0_0_8_0 = (0*LSCB)*(MT1J+PAD) + (8*LSPB) = 9216 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(4)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+18:vgprG2LB+18+1] offset:10368 // lwoB_0_0_9_0 = (0*LSCB)*(MT1J+PAD) + (9*LSPB) = 10368 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(3)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+20:vgprG2LB+20+1] offset:11520 // lwoB_0_0_10_0 = (0*LSCB)*(MT1J+PAD) + (10*LSPB) = 11520 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(2)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+22:vgprG2LB+22+1] offset:12672 // lwoB_0_0_11_0 = (0*LSCB)*(MT1J+PAD) + (11*LSPB) = 12672 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(1)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+24:vgprG2LB+24+1] offset:13824 // lwoB_0_0_12_0 = (0*LSCB)*(MT1J+PAD) + (12*LSPB) = 13824 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(0)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+26:vgprG2LB+26+1] offset:14976 // lwoB_0_0_13_0 = (0*LSCB)*(MT1J+PAD) + (13*LSPB) = 14976 sync LDS0

/* local write swap offsets a */

/* local write swap offsets b */

/* local read swap offsets a */

/* local read swap offsets b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+48+0+0:vgprValuB_X0_I0+48+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/* numPrefetchIter=0 */
/* dataAtIterA=3 numReadsIterA=4 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=3 numReadsIterB=4 skipReadsIterB=0 readsPerIterB=14 */

/******************************************/
/* Unrolled Loop - End                    */
/******************************************/

/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x1              // counterL==1
s_cbranch_scc0 label_LoopBeginL                    // restart LoopL
label_LoopEndL:

/* Before NLL: Check VGPR.checkin for INT8 LW */
s_and_b32 s8, s[sgprGSU], 0xfff                    // Restore GSU
s_cmp_eq_u32 s8, 1                                 // GSU == 1 ?
s_cbranch_scc0 label_GSU_3                         // branch if GSU != 1
label_GSU_3:

/******************************************/
/* Ord. NoLoadLoop - Begin                */
/******************************************/
s_waitcnt lgkmcnt(0)                               // 4wait for local write
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier                                          // wait for local write done, sync

/* iter 0 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:27, lwEndMfmaIndex:27  */
/*  numMfmaForLR:6, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:26 */
/*  mfmaIndex:0  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:2304 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:2320 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:1  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:4608 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:4624 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:2  */
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:6912 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:6928 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:3  */
ds_load_b128 v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprLocalReadAddrB+0] offset:9216 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[vgprLocalReadAddrB+0] offset:9232 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:4  */
ds_load_b128 v[vgprValuB_X0_I0+40:vgprValuB_X0_I0+40+3], v[vgprLocalReadAddrB+0] offset:11520 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=5 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+44:vgprValuB_X0_I0+44+3], v[vgprLocalReadAddrB+0] offset:11536 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=5 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:5  */
ds_load_b128 v[vgprValuB_X0_I0+48:vgprValuB_X0_I0+48+3], v[vgprLocalReadAddrB+0] offset:13824 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=6 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+52:vgprValuB_X0_I0+52+3], v[vgprLocalReadAddrB+0] offset:13840 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=6 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+40+0+0:vgprValuB_X0_I0+40+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:6  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+48+0+0:vgprValuB_X0_I0+48+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=14 */

/* iter 1 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:27, lwEndMfmaIndex:27  */
/*  numMfmaForLR:6, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:26 */
/*  mfmaIndex:7  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:2336 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:2352 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:8  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:4640 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:4656 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:9  */
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:6944 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:6960 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:10  */
ds_load_b128 v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprLocalReadAddrB+0] offset:9248 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[vgprLocalReadAddrB+0] offset:9264 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:11  */
ds_load_b128 v[vgprValuB_X0_I0+40:vgprValuB_X0_I0+40+3], v[vgprLocalReadAddrB+0] offset:11552 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=5 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+44:vgprValuB_X0_I0+44+3], v[vgprLocalReadAddrB+0] offset:11568 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=5 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:12  */
ds_load_b128 v[vgprValuB_X0_I0+48:vgprValuB_X0_I0+48+3], v[vgprLocalReadAddrB+0] offset:13856 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=6 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+52:vgprValuB_X0_I0+52+3], v[vgprLocalReadAddrB+0] offset:13872 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=6 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+40+0+0:vgprValuB_X0_I0+40+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:13  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+48+0+0:vgprValuB_X0_I0+48+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=0 readsPerIterB=14 */

/* iter 2 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:27, lwEndMfmaIndex:27  */
/*  numMfmaForLR:6, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:26 */
/*  mfmaIndex:14  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:2368 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:2384 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:15  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:4672 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:4688 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:16  */
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:6976 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:6992 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:17  */
ds_load_b128 v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprLocalReadAddrB+0] offset:9280 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[vgprLocalReadAddrB+0] offset:9296 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:18  */
ds_load_b128 v[vgprValuB_X0_I0+40:vgprValuB_X0_I0+40+3], v[vgprLocalReadAddrB+0] offset:11584 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=5 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+44:vgprValuB_X0_I0+44+3], v[vgprLocalReadAddrB+0] offset:11600 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=5 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:19  */
ds_load_b128 v[vgprValuB_X0_I0+48:vgprValuB_X0_I0+48+3], v[vgprLocalReadAddrB+0] offset:13888 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=6 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+52:vgprValuB_X0_I0+52+3], v[vgprLocalReadAddrB+0] offset:13904 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=6 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+40+0+0:vgprValuB_X0_I0+40+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:20  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+48+0+0:vgprValuB_X0_I0+48+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=0 readsPerIterB=14 */

/* iter 3 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:27, lwEndMfmaIndex:27  */
/*  numMfmaForLR:6, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:26 */
/*  mfmaIndex:21  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:2400 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:2416 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:22  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:4704 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:4720 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:23  */
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:7008 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:7024 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:24  */
ds_load_b128 v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprLocalReadAddrB+0] offset:9312 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[vgprLocalReadAddrB+0] offset:9328 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:25  */
ds_load_b128 v[vgprValuB_X0_I0+40:vgprValuB_X0_I0+40+3], v[vgprLocalReadAddrB+0] offset:11616 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=5 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+44:vgprValuB_X0_I0+44+3], v[vgprLocalReadAddrB+0] offset:11632 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=5 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:26  */
/* schedule remaining localreads for one buffer scheduling */
ds_load_b128 v[vgprValuB_X0_I0+48:vgprValuB_X0_I0+48+3], v[vgprLocalReadAddrB+0] offset:13920 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=6 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+52:vgprValuB_X0_I0+52+3], v[vgprLocalReadAddrB+0] offset:13936 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=6 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+40+0+0:vgprValuB_X0_I0+40+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:27  */
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+48+0+0:vgprValuB_X0_I0+48+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/* numPrefetchIter=0 */
/* dataAtIterA=3 numReadsIterA=4 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=3 numReadsIterB=4 skipReadsIterB=0 readsPerIterB=14 */
label_toPGR1end_OrdNLL:
label_PrefetchGlobalLastIterEnd:

/* Tail: add ValuA/B vgpr buffer [100...165) to pool */

/* Tail: add address/G2L vgpr [165...210) to pool */
label_Summation_End_2:
.set sgprWGM, UNDEF
.set sgprLoopCounterL, UNDEF
.set sgprOrigLoopCounter, UNDEF
.set sgprAddressA, UNDEF
.set sgprAddressB, UNDEF
.set sgprStridesA, UNDEF
.set sgprStridesB, UNDEF
.set sgprStrideScaleA, UNDEF
.set sgprAddressScaleA, UNDEF
.set sgprAddressScaleB, UNDEF
.set sgprSrdScaleA, UNDEF
.set sgprScaleAPkMagic, UNDEF
.set sgprScaleAPkPermute, UNDEF
.set sgprAddressScaleZeroA, UNDEF
.set sgprSrdScaleZeroA, UNDEF
.set sgprSrdA, UNDEF
.set sgprSrdB, UNDEF
.set sgprShadowLimitA, UNDEF
.set sgprShadowLimitB, UNDEF
.set sgprStaggerUIter, UNDEF
.set sgprWrapUA, UNDEF
.set sgprWrapUB, UNDEF
.set sgprGlobalReadIncsA, UNDEF
.set sgprGlobalReadIncsB, UNDEF
/* load store sgprs */

/* Mapping of Acc register -> C Vgpr register */

/* Multiply MI out register with Alpha -> C Vgpr register */

/* not-LocalSplitU: global write indices */
/* computeStoreVgprs */
v_lshrrev_b32 v104, 5, v[vgprSerial]               // 104 = Serial / 32
v_lshrrev_b32 v105, 2, v104                        // 105 = 104 / 4
v_mul_lo_u32 v105, 0x10, v105                      // wave coordination offset 1
v_and_b32 v101, 31, v[vgprSerial]                  // v101 = v[vgprSerial] % 32
v_lshrrev_b32 v101, 4, v101                        // 101 = 101 / 16
                                                   // thread0 * continuous_output (multiplier is 1, do nothing)
v_add_lshl_u32 v101, v105, v101, 0                 // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v102, v101, s[sgprStrideC1J]          //  offset 1
v_mul_lo_u32 v103, v101, s[sgprStrideD1J]          //  offset 1
v_and_b32 v100, 3, v104                            // v100 = v104 % 4
v_mul_lo_u32 v100, 0x10, v100                      // wave coordination offset 0
v_and_b32 v105, 15, v[vgprSerial]                  // v105 = v[vgprSerial] % 16
v_add_lshl_u32 v100, v105, v100, 0                 // coordination 0 = vwA * (wave_id0 + tid0)
s_mul_i32 s8, 64, s[sgprWorkGroup0]                // wgp0 * MT0
v_add_nc_u32 v100, s8, v100                        // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s8, 112, s[sgprWorkGroup1]               // wgp1 * MT1
v_add_nc_u32 v101, s8, v101                        // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1

/* not-LocalSplitU: global write */

/******************************************/
/* Global Write Elements                  */
/******************************************/
s_and_b32 s8, s[sgprGSU], 0xfff                    // Restore GSU
s_cmp_eq_u32 s8, 1                                 // GSU == 1 ?
s_cbranch_scc1 label_GSU_4                         // branch if GSU == 1
label_GW_B0_MB:
label_GW_B0_FD0_MB:

/* Edge/NonEdge store path check (M): Size % 64 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s28, 63, s[sgprSizeI]                    // s28 = s[sgprSizeI] % 64
s_add_u32 s29, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s29                // wg0 >= nwg0-1 ?
s_cselect_b32 s28, s28, 0                          // set rem
s_cmpk_gt_u32 s28, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_MB_Else         // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 112 > 0 -> Edge store; else -> NonEdge store */
s_mul_hi_u32 s31, s[sgprSizeJ], 76695845           // STATIC_DIV: divisor=112
s_mul_i32 s30, s[sgprSizeJ], 76695845              // tmp = dividend * magic
s_lshr_b64 s[30:31], s[30:31], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s29, s30                                 // quotient
s_mul_i32 s30, s29, 112                            // quotient*divisor
s_sub_u32 s28, s[sgprSizeJ], s30                   // rReg = dividend - quotient*divisor
s_add_u32 s29, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s29                // wg1 >= nwg1-1
s_cselect_b32 s28, s28, 0                          // set rem
s_cmpk_gt_u32 s28, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_MB_Then         // jump if edges required
label_GW_B0_FD0_VW1_MB_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=16 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (1,0,0,0:vw1); (2,0,0,0:vw1); (3,0,0,0:vw1); (4,0,0,0:vw1); (5,0,0,0:vw1); (6,0,0,0:vw1); (7,0,0,0:vw1); (8,0,0,0:vw1); (9,0,0,0:vw1); (10,0,0,0:vw1); (11,0,0,0:vw1); (12,0,0,0:vw1); (13,0,0,0:vw1); (14,0,0,0:vw1); (15,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
v_add_lshl_u32 v111, v103, v100, 2                 // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=100, coord0Vgpr=100 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0), (3, 0, 0, 0), (4, 0, 0, 0), (5, 0, 0, 0), (6, 0, 0, 0), (7, 0, 0, 0), (8, 0, 0, 0), (9, 0, 0, 0), (10, 0, 0, 0), (11, 0, 0, 0), (12, 0, 0, 0), (13, 0, 0, 0), (14, 0, 0, 0), (15, 0, 0, 0)] */
v_mov_b32 v[vgprValuC+113], v[vgprValuC+0]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+114], v[vgprValuC+1]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+115], v[vgprValuC+2]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+116], v[vgprValuC+3]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+117], v[vgprValuC+4]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+118], v[vgprValuC+5]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+119], v[vgprValuC+6]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+120], v[vgprValuC+7]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+121], v[vgprValuC+8]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+122], v[vgprValuC+9]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+123], v[vgprValuC+10]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+124], v[vgprValuC+11]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+125], v[vgprValuC+12]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+126], v[vgprValuC+13]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+127], v[vgprValuC+14]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+128], v[vgprValuC+15]        // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b32 v113, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v114, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v115, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v116, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v117, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v118, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v119, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v120, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v121, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v122, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v123, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v124, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v125, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v126, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v127, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v128, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #1 (d1,d0,vc1,vc0) = */
/*    (16,0,0,0:vw1); (17,0,0,0:vw1); (18,0,0,0:vw1); (19,0,0,0:vw1); (20,0,0,0:vw1); (21,0,0,0:vw1); (22,0,0,0:vw1); (23,0,0,0:vw1); (24,0,0,0:vw1); (25,0,0,0:vw1); (26,0,0,0:vw1); (27,0,0,0:vw1); (28,0,0,0:vw1); (29,0,0,0:vw1); (30,0,0,0:vw1); (31,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
/* (d1,vc1,d0,vc0)=(31,0,0,0) */

/* rC *= alpha batchElements=[(16, 0, 0, 0), (17, 0, 0, 0), (18, 0, 0, 0), (19, 0, 0, 0), (20, 0, 0, 0), (21, 0, 0, 0), (22, 0, 0, 0), (23, 0, 0, 0), (24, 0, 0, 0), (25, 0, 0, 0), (26, 0, 0, 0), (27, 0, 0, 0), (28, 0, 0, 0), (29, 0, 0, 0), (30, 0, 0, 0), (31, 0, 0, 0)] */
v_mov_b32 v[vgprValuC+113], v[vgprValuC+16]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+114], v[vgprValuC+17]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+115], v[vgprValuC+18]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+116], v[vgprValuC+19]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+117], v[vgprValuC+20]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+118], v[vgprValuC+21]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+119], v[vgprValuC+22]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+120], v[vgprValuC+23]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+121], v[vgprValuC+24]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+122], v[vgprValuC+25]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+123], v[vgprValuC+26]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+124], v[vgprValuC+27]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+125], v[vgprValuC+28]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+126], v[vgprValuC+29]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+127], v[vgprValuC+30]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+128], v[vgprValuC+31]        // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v113, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v114, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v115, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v116, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v117, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v118, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v119, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v120, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v121, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v122, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v123, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v124, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v125, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v126, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v127, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v128, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #2 (d1,d0,vc1,vc0) = */
/*    (32,0,0,0:vw1); (33,0,0,0:vw1); (34,0,0,0:vw1); (35,0,0,0:vw1); (36,0,0,0:vw1); (37,0,0,0:vw1); (38,0,0,0:vw1); (39,0,0,0:vw1); (40,0,0,0:vw1); (41,0,0,0:vw1); (42,0,0,0:vw1); (43,0,0,0:vw1); (44,0,0,0:vw1); (45,0,0,0:vw1); (46,0,0,0:vw1); (47,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(32,0,0,0) */
/* (d1,vc1,d0,vc0)=(33,0,0,0) */
/* (d1,vc1,d0,vc0)=(34,0,0,0) */
/* (d1,vc1,d0,vc0)=(35,0,0,0) */
/* (d1,vc1,d0,vc0)=(36,0,0,0) */
/* (d1,vc1,d0,vc0)=(37,0,0,0) */
/* (d1,vc1,d0,vc0)=(38,0,0,0) */
/* (d1,vc1,d0,vc0)=(39,0,0,0) */
/* (d1,vc1,d0,vc0)=(40,0,0,0) */
/* (d1,vc1,d0,vc0)=(41,0,0,0) */
/* (d1,vc1,d0,vc0)=(42,0,0,0) */
/* (d1,vc1,d0,vc0)=(43,0,0,0) */
/* (d1,vc1,d0,vc0)=(44,0,0,0) */
/* (d1,vc1,d0,vc0)=(45,0,0,0) */
/* (d1,vc1,d0,vc0)=(46,0,0,0) */
/* (d1,vc1,d0,vc0)=(47,0,0,0) */

/* rC *= alpha batchElements=[(32, 0, 0, 0), (33, 0, 0, 0), (34, 0, 0, 0), (35, 0, 0, 0), (36, 0, 0, 0), (37, 0, 0, 0), (38, 0, 0, 0), (39, 0, 0, 0), (40, 0, 0, 0), (41, 0, 0, 0), (42, 0, 0, 0), (43, 0, 0, 0), (44, 0, 0, 0), (45, 0, 0, 0), (46, 0, 0, 0), (47, 0, 0, 0)] */
v_mov_b32 v[vgprValuC+113], v[vgprValuC+32]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+114], v[vgprValuC+33]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+115], v[vgprValuC+34]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+116], v[vgprValuC+35]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+117], v[vgprValuC+36]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+118], v[vgprValuC+37]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+119], v[vgprValuC+38]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+120], v[vgprValuC+39]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+121], v[vgprValuC+40]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+122], v[vgprValuC+41]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+123], v[vgprValuC+42]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+124], v[vgprValuC+43]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+125], v[vgprValuC+44]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+126], v[vgprValuC+45]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+127], v[vgprValuC+46]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+128], v[vgprValuC+47]        // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v113, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v114, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v115, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v116, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v117, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v118, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v119, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v120, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v121, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v122, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v123, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v124, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v125, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v126, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v127, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v128, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #3 (d1,d0,vc1,vc0) = */
/*    (48,0,0,0:vw1); (49,0,0,0:vw1); (50,0,0,0:vw1); (51,0,0,0:vw1); (52,0,0,0:vw1); (53,0,0,0:vw1); (54,0,0,0:vw1); (55,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(48,0,0,0) */
/* (d1,vc1,d0,vc0)=(49,0,0,0) */
/* (d1,vc1,d0,vc0)=(50,0,0,0) */
/* (d1,vc1,d0,vc0)=(51,0,0,0) */
/* (d1,vc1,d0,vc0)=(52,0,0,0) */
/* (d1,vc1,d0,vc0)=(53,0,0,0) */
/* (d1,vc1,d0,vc0)=(54,0,0,0) */
/* (d1,vc1,d0,vc0)=(55,0,0,0) */

/* rC *= alpha batchElements=[(48, 0, 0, 0), (49, 0, 0, 0), (50, 0, 0, 0), (51, 0, 0, 0), (52, 0, 0, 0), (53, 0, 0, 0), (54, 0, 0, 0), (55, 0, 0, 0)] */
v_mov_b32 v[vgprValuC+113], v[vgprValuC+48]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+114], v[vgprValuC+49]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+115], v[vgprValuC+50]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+116], v[vgprValuC+51]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+117], v[vgprValuC+52]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+118], v[vgprValuC+53]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+119], v[vgprValuC+54]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+120], v[vgprValuC+55]        // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v113, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v114, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v115, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v116, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v117, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v118, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v119, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v120, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_B0_FD0_VW1_MB_NonEdgeEnd:
label_GW_B0_FD0_VW1_MB_Else:
label_GW_B0_FD0_VW1_MB_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=16 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (1,0,0,0:vw1); (2,0,0,0:vw1); (3,0,0,0:vw1); (4,0,0,0:vw1); (5,0,0,0:vw1); (6,0,0,0:vw1); (7,0,0,0:vw1); (8,0,0,0:vw1); (9,0,0,0:vw1); (10,0,0,0:vw1); (11,0,0,0:vw1); (12,0,0,0:vw1); (13,0,0,0:vw1); (14,0,0,0:vw1); (15,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v106, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v127, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v127, v106, v127, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v128, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v128, v106, v128, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v129, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v129, v106, v129, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v130, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v106, v130, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v131, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v131, v106, v131, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v132, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v106, v132, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v133, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v133, v106, v133, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v134, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v106, v134, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v135, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v135, v106, v135, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v136, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v106, v136, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v137, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v137, v106, v137, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v138, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v106, v138, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v139, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v139, v106, v139, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v140, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v106, v140, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v141, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v141, v106, v141, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v142, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v106, v142, s30                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0), (3, 0, 0, 0), (4, 0, 0, 0), (5, 0, 0, 0), (6, 0, 0, 0), (7, 0, 0, 0), (8, 0, 0, 0), (9, 0, 0, 0), (10, 0, 0, 0), (11, 0, 0, 0), (12, 0, 0, 0), (13, 0, 0, 0), (14, 0, 0, 0), (15, 0, 0, 0)] */
v_mov_b32 v[vgprValuC+111], v[vgprValuC+0]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+112], v[vgprValuC+1]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+113], v[vgprValuC+2]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+114], v[vgprValuC+3]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+115], v[vgprValuC+4]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+116], v[vgprValuC+5]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+117], v[vgprValuC+6]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+118], v[vgprValuC+7]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+119], v[vgprValuC+8]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+120], v[vgprValuC+9]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+121], v[vgprValuC+10]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+122], v[vgprValuC+11]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+123], v[vgprValuC+12]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+124], v[vgprValuC+13]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+125], v[vgprValuC+14]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+126], v[vgprValuC+15]        // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b32 v111, v127, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v112, v128, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v113, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v114, v130, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v115, v131, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v116, v132, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v117, v133, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v118, v134, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v119, v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v120, v136, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v121, v137, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v122, v138, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v123, v139, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v124, v140, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v125, v141, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v126, v142, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (16,0,0,0:vw1); (17,0,0,0:vw1); (18,0,0,0:vw1); (19,0,0,0:vw1); (20,0,0,0:vw1); (21,0,0,0:vw1); (22,0,0,0:vw1); (23,0,0,0:vw1); (24,0,0,0:vw1); (25,0,0,0:vw1); (26,0,0,0:vw1); (27,0,0,0:vw1); (28,0,0,0:vw1); (29,0,0,0:vw1); (30,0,0,0:vw1); (31,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v106, BufferOOB
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v127, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v127, v106, v127, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v128, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v128, v106, v128, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v129, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v129, v106, v129, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v130, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v106, v130, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v131, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v131, v106, v131, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v132, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v106, v132, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v133, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v133, v106, v133, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v134, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v106, v134, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v135, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v135, v106, v135, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v136, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v106, v136, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v137, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v137, v106, v137, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v138, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v106, v138, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v139, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v139, v106, v139, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v140, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v106, v140, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v141, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v141, v106, v141, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v142, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v106, v142, s30                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(16, 0, 0, 0), (17, 0, 0, 0), (18, 0, 0, 0), (19, 0, 0, 0), (20, 0, 0, 0), (21, 0, 0, 0), (22, 0, 0, 0), (23, 0, 0, 0), (24, 0, 0, 0), (25, 0, 0, 0), (26, 0, 0, 0), (27, 0, 0, 0), (28, 0, 0, 0), (29, 0, 0, 0), (30, 0, 0, 0), (31, 0, 0, 0)] */
v_mov_b32 v[vgprValuC+111], v[vgprValuC+16]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+112], v[vgprValuC+17]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+113], v[vgprValuC+18]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+114], v[vgprValuC+19]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+115], v[vgprValuC+20]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+116], v[vgprValuC+21]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+117], v[vgprValuC+22]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+118], v[vgprValuC+23]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+119], v[vgprValuC+24]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+120], v[vgprValuC+25]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+121], v[vgprValuC+26]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+122], v[vgprValuC+27]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+123], v[vgprValuC+28]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+124], v[vgprValuC+29]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+125], v[vgprValuC+30]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+126], v[vgprValuC+31]        // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b32 v111, v127, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v112, v128, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v113, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v114, v130, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v115, v131, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v116, v132, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v117, v133, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v118, v134, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v119, v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v120, v136, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v121, v137, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v122, v138, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v123, v139, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v124, v140, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v125, v141, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v126, v142, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #2 (d1,d0,vc1,vc0) = */
/*    (32,0,0,0:vw1); (33,0,0,0:vw1); (34,0,0,0:vw1); (35,0,0,0:vw1); (36,0,0,0:vw1); (37,0,0,0:vw1); (38,0,0,0:vw1); (39,0,0,0:vw1); (40,0,0,0:vw1); (41,0,0,0:vw1); (42,0,0,0:vw1); (43,0,0,0:vw1); (44,0,0,0:vw1); (45,0,0,0:vw1); (46,0,0,0:vw1); (47,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v106, BufferOOB
/* (d1,vc1,d0,vc0)=(32,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v127, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v127, v106, v127, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(33,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v128, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v128, v106, v128, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(34,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v129, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v129, v106, v129, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(35,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v130, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v106, v130, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(36,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v131, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v131, v106, v131, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(37,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v132, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v106, v132, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(38,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v133, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v133, v106, v133, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(39,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v134, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v106, v134, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(40,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v135, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v135, v106, v135, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(41,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v136, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v106, v136, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(42,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v137, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v137, v106, v137, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(43,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v138, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v106, v138, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(44,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v139, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v139, v106, v139, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(45,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v140, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v106, v140, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(46,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v141, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v141, v106, v141, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(47,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v142, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v106, v142, s30                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(32, 0, 0, 0), (33, 0, 0, 0), (34, 0, 0, 0), (35, 0, 0, 0), (36, 0, 0, 0), (37, 0, 0, 0), (38, 0, 0, 0), (39, 0, 0, 0), (40, 0, 0, 0), (41, 0, 0, 0), (42, 0, 0, 0), (43, 0, 0, 0), (44, 0, 0, 0), (45, 0, 0, 0), (46, 0, 0, 0), (47, 0, 0, 0)] */
v_mov_b32 v[vgprValuC+111], v[vgprValuC+32]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+112], v[vgprValuC+33]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+113], v[vgprValuC+34]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+114], v[vgprValuC+35]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+115], v[vgprValuC+36]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+116], v[vgprValuC+37]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+117], v[vgprValuC+38]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+118], v[vgprValuC+39]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+119], v[vgprValuC+40]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+120], v[vgprValuC+41]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+121], v[vgprValuC+42]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+122], v[vgprValuC+43]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+123], v[vgprValuC+44]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+124], v[vgprValuC+45]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+125], v[vgprValuC+46]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+126], v[vgprValuC+47]        // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b32 v111, v127, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v112, v128, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v113, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v114, v130, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v115, v131, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v116, v132, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v117, v133, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v118, v134, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v119, v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v120, v136, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v121, v137, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v122, v138, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v123, v139, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v124, v140, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v125, v141, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v126, v142, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #3 (d1,d0,vc1,vc0) = */
/*    (48,0,0,0:vw1); (49,0,0,0:vw1); (50,0,0,0:vw1); (51,0,0,0:vw1); (52,0,0,0:vw1); (53,0,0,0:vw1); (54,0,0,0:vw1); (55,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v106, BufferOOB
/* (d1,vc1,d0,vc0)=(48,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v119, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v119, v106, v119, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(49,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v120, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v120, v106, v120, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(50,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v121, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v121, v106, v121, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(51,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v122, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v122, v106, v122, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(52,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v123, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v123, v106, v123, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(53,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v124, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v124, v106, v124, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(54,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v125, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v125, v106, v125, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(55,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v126, v103, v100, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v126, v106, v126, s30                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(48, 0, 0, 0), (49, 0, 0, 0), (50, 0, 0, 0), (51, 0, 0, 0), (52, 0, 0, 0), (53, 0, 0, 0), (54, 0, 0, 0), (55, 0, 0, 0)] */
v_mov_b32 v[vgprValuC+111], v[vgprValuC+48]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+112], v[vgprValuC+49]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+113], v[vgprValuC+50]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+114], v[vgprValuC+51]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+115], v[vgprValuC+52]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+116], v[vgprValuC+53]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+117], v[vgprValuC+54]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+118], v[vgprValuC+55]        // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b32 v111, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v112, v120, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v113, v121, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v114, v122, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v115, v123, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v116, v124, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v117, v125, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v118, v126, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_End:
s_getpc_b64 s[28:29]                               // addr of next instr
s_add_i32 s30, label_KernelEnd, 4                  // target branch offset
s_add_u32 s28, s28, s30                            // add target branch offset
s_addc_u32 s29, s29, 0                             // add high and carry
s_setpc_b64 s[28:29]                               // branch to label_KernelEnd
label_GSU_4:
s_cmpk_eq_u32 s[sgprBeta], 0                       // Beta == 0
s_cbranch_scc0 label_GW_B1_GSU1                    // Branch if Beta is not zero

label_GW_B0_GSU1:
label_GW_B0_FD0_GSU1:

/* Edge/NonEdge store path check (M): Size % 64 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s28, 63, s[sgprSizeI]                    // s28 = s[sgprSizeI] % 64
s_add_u32 s29, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s29                // wg0 >= nwg0-1 ?
s_cselect_b32 s28, s28, 0                          // set rem
s_cmpk_gt_u32 s28, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_GSU1_Else       // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 112 > 0 -> Edge store; else -> NonEdge store */
s_mul_hi_u32 s31, s[sgprSizeJ], 76695845           // STATIC_DIV: divisor=112
s_mul_i32 s30, s[sgprSizeJ], 76695845              // tmp = dividend * magic
s_lshr_b64 s[30:31], s[30:31], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s29, s30                                 // quotient
s_mul_i32 s30, s29, 112                            // quotient*divisor
s_sub_u32 s28, s[sgprSizeJ], s30                   // rReg = dividend - quotient*divisor
s_add_u32 s29, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s29                // wg1 >= nwg1-1
s_cselect_b32 s28, s28, 0                          // set rem
s_cmpk_gt_u32 s28, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_GSU1_Then       // jump if edges required
label_GW_B0_FD0_VW1_GSU1_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=16 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (1,0,0,0:vw1); (2,0,0,0:vw1); (3,0,0,0:vw1); (4,0,0,0:vw1); (5,0,0,0:vw1); (6,0,0,0:vw1); (7,0,0,0:vw1); (8,0,0,0:vw1); (9,0,0,0:vw1); (10,0,0,0:vw1); (11,0,0,0:vw1); (12,0,0,0:vw1); (13,0,0,0:vw1); (14,0,0,0:vw1); (15,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
v_add_lshl_u32 v111, v103, v100, 1                 // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=100, coord0Vgpr=100 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0), (3, 0, 0, 0), (4, 0, 0, 0), (5, 0, 0, 0), (6, 0, 0, 0), (7, 0, 0, 0), (8, 0, 0, 0), (9, 0, 0, 0), (10, 0, 0, 0), (11, 0, 0, 0), (12, 0, 0, 0), (13, 0, 0, 0), (14, 0, 0, 0), (15, 0, 0, 0)] */
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+128], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v113.l, v[vgprValuC+113]             // convert C to fp16
buffer_store_b16 v113, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v114.l, v[vgprValuC+114]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v114, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v115.l, v[vgprValuC+115]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v115, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v116.l, v[vgprValuC+116]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v116, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v117.l, v[vgprValuC+117]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v117, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v118.l, v[vgprValuC+118]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v118, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v119.l, v[vgprValuC+119]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v119, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v120.l, v[vgprValuC+120]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v120, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v121.l, v[vgprValuC+121]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v121, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v122.l, v[vgprValuC+122]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v122, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v123.l, v[vgprValuC+123]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v123, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v124.l, v[vgprValuC+124]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v124, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v125.l, v[vgprValuC+125]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v125, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v126.l, v[vgprValuC+126]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v126, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v127.l, v[vgprValuC+127]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v127, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v128.l, v[vgprValuC+128]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v128, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #1 (d1,d0,vc1,vc0) = */
/*    (16,0,0,0:vw1); (17,0,0,0:vw1); (18,0,0,0:vw1); (19,0,0,0:vw1); (20,0,0,0:vw1); (21,0,0,0:vw1); (22,0,0,0:vw1); (23,0,0,0:vw1); (24,0,0,0:vw1); (25,0,0,0:vw1); (26,0,0,0:vw1); (27,0,0,0:vw1); (28,0,0,0:vw1); (29,0,0,0:vw1); (30,0,0,0:vw1); (31,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
/* (d1,vc1,d0,vc0)=(31,0,0,0) */

/* rC *= alpha batchElements=[(16, 0, 0, 0), (17, 0, 0, 0), (18, 0, 0, 0), (19, 0, 0, 0), (20, 0, 0, 0), (21, 0, 0, 0), (22, 0, 0, 0), (23, 0, 0, 0), (24, 0, 0, 0), (25, 0, 0, 0), (26, 0, 0, 0), (27, 0, 0, 0), (28, 0, 0, 0), (29, 0, 0, 0), (30, 0, 0, 0), (31, 0, 0, 0)] */
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+128], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v113.l, v[vgprValuC+113]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v113, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v114.l, v[vgprValuC+114]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v114, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v115.l, v[vgprValuC+115]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v115, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v116.l, v[vgprValuC+116]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v116, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v117.l, v[vgprValuC+117]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v117, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v118.l, v[vgprValuC+118]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v118, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v119.l, v[vgprValuC+119]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v119, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v120.l, v[vgprValuC+120]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v120, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v121.l, v[vgprValuC+121]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v121, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v122.l, v[vgprValuC+122]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v122, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v123.l, v[vgprValuC+123]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v123, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v124.l, v[vgprValuC+124]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v124, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v125.l, v[vgprValuC+125]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v125, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v126.l, v[vgprValuC+126]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v126, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v127.l, v[vgprValuC+127]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v127, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v128.l, v[vgprValuC+128]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v128, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #2 (d1,d0,vc1,vc0) = */
/*    (32,0,0,0:vw1); (33,0,0,0:vw1); (34,0,0,0:vw1); (35,0,0,0:vw1); (36,0,0,0:vw1); (37,0,0,0:vw1); (38,0,0,0:vw1); (39,0,0,0:vw1); (40,0,0,0:vw1); (41,0,0,0:vw1); (42,0,0,0:vw1); (43,0,0,0:vw1); (44,0,0,0:vw1); (45,0,0,0:vw1); (46,0,0,0:vw1); (47,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(32,0,0,0) */
/* (d1,vc1,d0,vc0)=(33,0,0,0) */
/* (d1,vc1,d0,vc0)=(34,0,0,0) */
/* (d1,vc1,d0,vc0)=(35,0,0,0) */
/* (d1,vc1,d0,vc0)=(36,0,0,0) */
/* (d1,vc1,d0,vc0)=(37,0,0,0) */
/* (d1,vc1,d0,vc0)=(38,0,0,0) */
/* (d1,vc1,d0,vc0)=(39,0,0,0) */
/* (d1,vc1,d0,vc0)=(40,0,0,0) */
/* (d1,vc1,d0,vc0)=(41,0,0,0) */
/* (d1,vc1,d0,vc0)=(42,0,0,0) */
/* (d1,vc1,d0,vc0)=(43,0,0,0) */
/* (d1,vc1,d0,vc0)=(44,0,0,0) */
/* (d1,vc1,d0,vc0)=(45,0,0,0) */
/* (d1,vc1,d0,vc0)=(46,0,0,0) */
/* (d1,vc1,d0,vc0)=(47,0,0,0) */

/* rC *= alpha batchElements=[(32, 0, 0, 0), (33, 0, 0, 0), (34, 0, 0, 0), (35, 0, 0, 0), (36, 0, 0, 0), (37, 0, 0, 0), (38, 0, 0, 0), (39, 0, 0, 0), (40, 0, 0, 0), (41, 0, 0, 0), (42, 0, 0, 0), (43, 0, 0, 0), (44, 0, 0, 0), (45, 0, 0, 0), (46, 0, 0, 0), (47, 0, 0, 0)] */
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+128], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v113.l, v[vgprValuC+113]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v113, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v114.l, v[vgprValuC+114]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v114, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v115.l, v[vgprValuC+115]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v115, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v116.l, v[vgprValuC+116]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v116, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v117.l, v[vgprValuC+117]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v117, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v118.l, v[vgprValuC+118]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v118, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v119.l, v[vgprValuC+119]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v119, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v120.l, v[vgprValuC+120]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v120, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v121.l, v[vgprValuC+121]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v121, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v122.l, v[vgprValuC+122]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v122, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v123.l, v[vgprValuC+123]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v123, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v124.l, v[vgprValuC+124]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v124, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v125.l, v[vgprValuC+125]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v125, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v126.l, v[vgprValuC+126]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v126, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v127.l, v[vgprValuC+127]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v127, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v128.l, v[vgprValuC+128]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v128, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #3 (d1,d0,vc1,vc0) = */
/*    (48,0,0,0:vw1); (49,0,0,0:vw1); (50,0,0,0:vw1); (51,0,0,0:vw1); (52,0,0,0:vw1); (53,0,0,0:vw1); (54,0,0,0:vw1); (55,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(48,0,0,0) */
/* (d1,vc1,d0,vc0)=(49,0,0,0) */
/* (d1,vc1,d0,vc0)=(50,0,0,0) */
/* (d1,vc1,d0,vc0)=(51,0,0,0) */
/* (d1,vc1,d0,vc0)=(52,0,0,0) */
/* (d1,vc1,d0,vc0)=(53,0,0,0) */
/* (d1,vc1,d0,vc0)=(54,0,0,0) */
/* (d1,vc1,d0,vc0)=(55,0,0,0) */

/* rC *= alpha batchElements=[(48, 0, 0, 0), (49, 0, 0, 0), (50, 0, 0, 0), (51, 0, 0, 0), (52, 0, 0, 0), (53, 0, 0, 0), (54, 0, 0, 0), (55, 0, 0, 0)] */
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+48] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+49] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+50] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+51] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+52] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+53] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+54] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+55] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v113.l, v[vgprValuC+113]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v113, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v114.l, v[vgprValuC+114]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v114, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v115.l, v[vgprValuC+115]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v115, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v116.l, v[vgprValuC+116]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v116, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v117.l, v[vgprValuC+117]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v117, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v118.l, v[vgprValuC+118]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v118, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v119.l, v[vgprValuC+119]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v119, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v120.l, v[vgprValuC+120]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v120, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW1_GSU1_NonEdgeEnd:
label_GW_B0_FD0_VW1_GSU1_Else:
label_GW_B0_FD0_VW1_GSU1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=16 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (1,0,0,0:vw1); (2,0,0,0:vw1); (3,0,0,0:vw1); (4,0,0,0:vw1); (5,0,0,0:vw1); (6,0,0,0:vw1); (7,0,0,0:vw1); (8,0,0,0:vw1); (9,0,0,0:vw1); (10,0,0,0:vw1); (11,0,0,0:vw1); (12,0,0,0:vw1); (13,0,0,0:vw1); (14,0,0,0:vw1); (15,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v106, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v127, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v127, v106, v127, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v128, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v128, v106, v128, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v129, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v129, v106, v129, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v130, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v106, v130, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v131, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v131, v106, v131, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v132, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v106, v132, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v133, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v133, v106, v133, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v134, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v106, v134, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v135, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v135, v106, v135, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v136, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v106, v136, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v137, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v137, v106, v137, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v138, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v106, v138, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v139, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v139, v106, v139, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v140, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v106, v140, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v141, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v141, v106, v141, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v142, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v106, v142, s30                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0), (3, 0, 0, 0), (4, 0, 0, 0), (5, 0, 0, 0), (6, 0, 0, 0), (7, 0, 0, 0), (8, 0, 0, 0), (9, 0, 0, 0), (10, 0, 0, 0), (11, 0, 0, 0), (12, 0, 0, 0), (13, 0, 0, 0), (14, 0, 0, 0), (15, 0, 0, 0)] */
v_mul_f32 v[vgprValuC+111], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+112], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v111.l, v[vgprValuC+111]             // convert C to fp16
buffer_store_b16 v111, v127, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v112.l, v[vgprValuC+112]             // convert C to fp16
buffer_store_b16 v112, v128, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v113.l, v[vgprValuC+113]             // convert C to fp16
buffer_store_b16 v113, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v114.l, v[vgprValuC+114]             // convert C to fp16
buffer_store_b16 v114, v130, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v115.l, v[vgprValuC+115]             // convert C to fp16
buffer_store_b16 v115, v131, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v116.l, v[vgprValuC+116]             // convert C to fp16
buffer_store_b16 v116, v132, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v117.l, v[vgprValuC+117]             // convert C to fp16
buffer_store_b16 v117, v133, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v118.l, v[vgprValuC+118]             // convert C to fp16
buffer_store_b16 v118, v134, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v119.l, v[vgprValuC+119]             // convert C to fp16
buffer_store_b16 v119, v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v120.l, v[vgprValuC+120]             // convert C to fp16
buffer_store_b16 v120, v136, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v121.l, v[vgprValuC+121]             // convert C to fp16
buffer_store_b16 v121, v137, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v122.l, v[vgprValuC+122]             // convert C to fp16
buffer_store_b16 v122, v138, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v123.l, v[vgprValuC+123]             // convert C to fp16
buffer_store_b16 v123, v139, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v124.l, v[vgprValuC+124]             // convert C to fp16
buffer_store_b16 v124, v140, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v125.l, v[vgprValuC+125]             // convert C to fp16
buffer_store_b16 v125, v141, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v126.l, v[vgprValuC+126]             // convert C to fp16
buffer_store_b16 v126, v142, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (16,0,0,0:vw1); (17,0,0,0:vw1); (18,0,0,0:vw1); (19,0,0,0:vw1); (20,0,0,0:vw1); (21,0,0,0:vw1); (22,0,0,0:vw1); (23,0,0,0:vw1); (24,0,0,0:vw1); (25,0,0,0:vw1); (26,0,0,0:vw1); (27,0,0,0:vw1); (28,0,0,0:vw1); (29,0,0,0:vw1); (30,0,0,0:vw1); (31,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v106, BufferOOB
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v127, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v127, v106, v127, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v128, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v128, v106, v128, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v129, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v129, v106, v129, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v130, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v106, v130, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v131, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v131, v106, v131, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v132, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v106, v132, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v133, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v133, v106, v133, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v134, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v106, v134, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v135, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v135, v106, v135, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v136, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v106, v136, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v137, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v137, v106, v137, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v138, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v106, v138, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v139, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v139, v106, v139, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v140, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v106, v140, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v141, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v141, v106, v141, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v142, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v106, v142, s30                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(16, 0, 0, 0), (17, 0, 0, 0), (18, 0, 0, 0), (19, 0, 0, 0), (20, 0, 0, 0), (21, 0, 0, 0), (22, 0, 0, 0), (23, 0, 0, 0), (24, 0, 0, 0), (25, 0, 0, 0), (26, 0, 0, 0), (27, 0, 0, 0), (28, 0, 0, 0), (29, 0, 0, 0), (30, 0, 0, 0), (31, 0, 0, 0)] */
v_mul_f32 v[vgprValuC+111], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+112], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v111.l, v[vgprValuC+111]             // convert C to fp16
buffer_store_b16 v111, v127, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v112.l, v[vgprValuC+112]             // convert C to fp16
buffer_store_b16 v112, v128, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v113.l, v[vgprValuC+113]             // convert C to fp16
buffer_store_b16 v113, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v114.l, v[vgprValuC+114]             // convert C to fp16
buffer_store_b16 v114, v130, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v115.l, v[vgprValuC+115]             // convert C to fp16
buffer_store_b16 v115, v131, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v116.l, v[vgprValuC+116]             // convert C to fp16
buffer_store_b16 v116, v132, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v117.l, v[vgprValuC+117]             // convert C to fp16
buffer_store_b16 v117, v133, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v118.l, v[vgprValuC+118]             // convert C to fp16
buffer_store_b16 v118, v134, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v119.l, v[vgprValuC+119]             // convert C to fp16
buffer_store_b16 v119, v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v120.l, v[vgprValuC+120]             // convert C to fp16
buffer_store_b16 v120, v136, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v121.l, v[vgprValuC+121]             // convert C to fp16
buffer_store_b16 v121, v137, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v122.l, v[vgprValuC+122]             // convert C to fp16
buffer_store_b16 v122, v138, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v123.l, v[vgprValuC+123]             // convert C to fp16
buffer_store_b16 v123, v139, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v124.l, v[vgprValuC+124]             // convert C to fp16
buffer_store_b16 v124, v140, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v125.l, v[vgprValuC+125]             // convert C to fp16
buffer_store_b16 v125, v141, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v126.l, v[vgprValuC+126]             // convert C to fp16
buffer_store_b16 v126, v142, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #2 (d1,d0,vc1,vc0) = */
/*    (32,0,0,0:vw1); (33,0,0,0:vw1); (34,0,0,0:vw1); (35,0,0,0:vw1); (36,0,0,0:vw1); (37,0,0,0:vw1); (38,0,0,0:vw1); (39,0,0,0:vw1); (40,0,0,0:vw1); (41,0,0,0:vw1); (42,0,0,0:vw1); (43,0,0,0:vw1); (44,0,0,0:vw1); (45,0,0,0:vw1); (46,0,0,0:vw1); (47,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v106, BufferOOB
/* (d1,vc1,d0,vc0)=(32,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v127, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v127, v106, v127, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(33,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v128, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v128, v106, v128, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(34,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v129, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v129, v106, v129, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(35,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v130, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v106, v130, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(36,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v131, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v131, v106, v131, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(37,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v132, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v106, v132, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(38,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v133, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v133, v106, v133, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(39,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v134, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v106, v134, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(40,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v135, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v135, v106, v135, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(41,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v136, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v106, v136, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(42,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v137, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v137, v106, v137, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(43,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v138, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v106, v138, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(44,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v139, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v139, v106, v139, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(45,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v140, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v106, v140, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(46,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v141, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v141, v106, v141, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(47,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v142, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v106, v142, s30                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(32, 0, 0, 0), (33, 0, 0, 0), (34, 0, 0, 0), (35, 0, 0, 0), (36, 0, 0, 0), (37, 0, 0, 0), (38, 0, 0, 0), (39, 0, 0, 0), (40, 0, 0, 0), (41, 0, 0, 0), (42, 0, 0, 0), (43, 0, 0, 0), (44, 0, 0, 0), (45, 0, 0, 0), (46, 0, 0, 0), (47, 0, 0, 0)] */
v_mul_f32 v[vgprValuC+111], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+112], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v111.l, v[vgprValuC+111]             // convert C to fp16
buffer_store_b16 v111, v127, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v112.l, v[vgprValuC+112]             // convert C to fp16
buffer_store_b16 v112, v128, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v113.l, v[vgprValuC+113]             // convert C to fp16
buffer_store_b16 v113, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v114.l, v[vgprValuC+114]             // convert C to fp16
buffer_store_b16 v114, v130, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v115.l, v[vgprValuC+115]             // convert C to fp16
buffer_store_b16 v115, v131, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v116.l, v[vgprValuC+116]             // convert C to fp16
buffer_store_b16 v116, v132, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v117.l, v[vgprValuC+117]             // convert C to fp16
buffer_store_b16 v117, v133, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v118.l, v[vgprValuC+118]             // convert C to fp16
buffer_store_b16 v118, v134, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v119.l, v[vgprValuC+119]             // convert C to fp16
buffer_store_b16 v119, v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v120.l, v[vgprValuC+120]             // convert C to fp16
buffer_store_b16 v120, v136, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v121.l, v[vgprValuC+121]             // convert C to fp16
buffer_store_b16 v121, v137, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v122.l, v[vgprValuC+122]             // convert C to fp16
buffer_store_b16 v122, v138, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v123.l, v[vgprValuC+123]             // convert C to fp16
buffer_store_b16 v123, v139, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v124.l, v[vgprValuC+124]             // convert C to fp16
buffer_store_b16 v124, v140, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v125.l, v[vgprValuC+125]             // convert C to fp16
buffer_store_b16 v125, v141, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v126.l, v[vgprValuC+126]             // convert C to fp16
buffer_store_b16 v126, v142, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #3 (d1,d0,vc1,vc0) = */
/*    (48,0,0,0:vw1); (49,0,0,0:vw1); (50,0,0,0:vw1); (51,0,0,0:vw1); (52,0,0,0:vw1); (53,0,0,0:vw1); (54,0,0,0:vw1); (55,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v106, BufferOOB
/* (d1,vc1,d0,vc0)=(48,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v119, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v119, v106, v119, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(49,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v120, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v120, v106, v120, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(50,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v121, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v121, v106, v121, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(51,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v122, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v122, v106, v122, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(52,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v123, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v123, v106, v123, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(53,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v124, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v124, v106, v124, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(54,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v125, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v125, v106, v125, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(55,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v126, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v126, v106, v126, s30                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(48, 0, 0, 0), (49, 0, 0, 0), (50, 0, 0, 0), (51, 0, 0, 0), (52, 0, 0, 0), (53, 0, 0, 0), (54, 0, 0, 0), (55, 0, 0, 0)] */
v_mul_f32 v[vgprValuC+111], s[sgprAlpha], v[vgprValuC+48] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+112], s[sgprAlpha], v[vgprValuC+49] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+50] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+51] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+52] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+53] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+54] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+55] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v111.l, v[vgprValuC+111]             // convert C to fp16
buffer_store_b16 v111, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v112.l, v[vgprValuC+112]             // convert C to fp16
buffer_store_b16 v112, v120, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v113.l, v[vgprValuC+113]             // convert C to fp16
buffer_store_b16 v113, v121, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v114.l, v[vgprValuC+114]             // convert C to fp16
buffer_store_b16 v114, v122, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v115.l, v[vgprValuC+115]             // convert C to fp16
buffer_store_b16 v115, v123, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v116.l, v[vgprValuC+116]             // convert C to fp16
buffer_store_b16 v116, v124, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v117.l, v[vgprValuC+117]             // convert C to fp16
buffer_store_b16 v117, v125, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v118.l, v[vgprValuC+118]             // convert C to fp16
buffer_store_b16 v118, v126, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_GSU1:
label_GW_B1_FD0_GSU1:

/* Edge/NonEdge store path check (M): Size % 64 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s28, 63, s[sgprSizeI]                    // s28 = s[sgprSizeI] % 64
s_add_u32 s29, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s29                // wg0 >= nwg0-1 ?
s_cselect_b32 s28, s28, 0                          // set rem
s_cmpk_gt_u32 s28, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW1_GSU1_Else       // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 112 > 0 -> Edge store; else -> NonEdge store */
s_mul_hi_u32 s31, s[sgprSizeJ], 76695845           // STATIC_DIV: divisor=112
s_mul_i32 s30, s[sgprSizeJ], 76695845              // tmp = dividend * magic
s_lshr_b64 s[30:31], s[30:31], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s29, s30                                 // quotient
s_mul_i32 s30, s29, 112                            // quotient*divisor
s_sub_u32 s28, s[sgprSizeJ], s30                   // rReg = dividend - quotient*divisor
s_add_u32 s29, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s29                // wg1 >= nwg1-1
s_cselect_b32 s28, s28, 0                          // set rem
s_cmpk_gt_u32 s28, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW1_GSU1_Then       // jump if edges required
label_GW_B1_FD0_VW1_GSU1_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=16 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (1,0,0,0:vw1); (2,0,0,0:vw1); (3,0,0,0:vw1); (4,0,0,0:vw1); (5,0,0,0:vw1); (6,0,0,0:vw1); (7,0,0,0:vw1); (8,0,0,0:vw1); (9,0,0,0:vw1); (10,0,0,0:vw1); (11,0,0,0:vw1); (12,0,0,0:vw1); (13,0,0,0:vw1); (14,0,0,0:vw1); (15,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v112, v102, v100, 1                 // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=100, coord0Vgpr=100 (multiple bpe)
buffer_load_d16_b16 v129, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v130, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v131, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v132, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v133, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v134, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v135, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v136, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v137, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v138, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v139, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v140, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v141, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v142, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v143, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v144, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v111, v103, v100, 1                 // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=100, coord0Vgpr=100 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0), (3, 0, 0, 0), (4, 0, 0, 0), (5, 0, 0, 0), (6, 0, 0, 0), (7, 0, 0, 0), (8, 0, 0, 0), (9, 0, 0, 0), (10, 0, 0, 0), (11, 0, 0, 0), (12, 0, 0, 0), (13, 0, 0, 0), (14, 0, 0, 0), (15, 0, 0, 0)] */
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+128], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(15)                                // vlcnt(15) = 16 - 1 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+113], s[sgprBeta], v129, v[vgprValuC+113] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v113.l, v[vgprValuC+113]             // convert C to fp16
buffer_store_b16 v113, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(14)                                // vlcnt(14) = 16 - 2 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+114], s[sgprBeta], v130, v[vgprValuC+114] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v114.l, v[vgprValuC+114]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v114, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(13)                                // vlcnt(13) = 16 - 3 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+115], s[sgprBeta], v131, v[vgprValuC+115] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v115.l, v[vgprValuC+115]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v115, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(12)                                // vlcnt(12) = 16 - 4 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+116], s[sgprBeta], v132, v[vgprValuC+116] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v116.l, v[vgprValuC+116]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v116, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(11)                                // vlcnt(11) = 16 - 5 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+117], s[sgprBeta], v133, v[vgprValuC+117] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v117.l, v[vgprValuC+117]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v117, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(10)                                // vlcnt(10) = 16 - 6 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+118], s[sgprBeta], v134, v[vgprValuC+118] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v118.l, v[vgprValuC+118]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v118, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(9)                                 // vlcnt(9) = 16 - 7 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+119], s[sgprBeta], v135, v[vgprValuC+119] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v119.l, v[vgprValuC+119]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v119, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(8)                                 // vlcnt(8) = 16 - 8 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+120], s[sgprBeta], v136, v[vgprValuC+120] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v120.l, v[vgprValuC+120]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v120, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(7)                                 // vlcnt(7) = 16 - 9 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+121], s[sgprBeta], v137, v[vgprValuC+121] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v121.l, v[vgprValuC+121]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v121, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(6)                                 // vlcnt(6) = 16 - 10 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+122], s[sgprBeta], v138, v[vgprValuC+122] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v122.l, v[vgprValuC+122]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v122, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(5)                                 // vlcnt(5) = 16 - 11 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+123], s[sgprBeta], v139, v[vgprValuC+123] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v123.l, v[vgprValuC+123]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v123, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(4)                                 // vlcnt(4) = 16 - 12 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+124], s[sgprBeta], v140, v[vgprValuC+124] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v124.l, v[vgprValuC+124]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v124, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(3)                                 // vlcnt(3) = 16 - 13 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+125], s[sgprBeta], v141, v[vgprValuC+125] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v125.l, v[vgprValuC+125]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v125, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(2)                                 // vlcnt(2) = 16 - 14 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+126], s[sgprBeta], v142, v[vgprValuC+126] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v126.l, v[vgprValuC+126]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v126, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(1)                                 // vlcnt(1) = 16 - 15 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+127], s[sgprBeta], v143, v[vgprValuC+127] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v127.l, v[vgprValuC+127]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v127, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(0)                                 // vlcnt(0) = 16 - 16 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+128], s[sgprBeta], v144, v[vgprValuC+128] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v128.l, v[vgprValuC+128]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v128, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #1 (d1,d0,vc1,vc0) = */
/*    (16,0,0,0:vw1); (17,0,0,0:vw1); (18,0,0,0:vw1); (19,0,0,0:vw1); (20,0,0,0:vw1); (21,0,0,0:vw1); (22,0,0,0:vw1); (23,0,0,0:vw1); (24,0,0,0:vw1); (25,0,0,0:vw1); (26,0,0,0:vw1); (27,0,0,0:vw1); (28,0,0,0:vw1); (29,0,0,0:vw1); (30,0,0,0:vw1); (31,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v129, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v130, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v131, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v132, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v133, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v134, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v135, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v136, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v137, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v138, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v139, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v140, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v141, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v142, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v143, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v144, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

/* rC *= alpha batchElements=[(16, 0, 0, 0), (17, 0, 0, 0), (18, 0, 0, 0), (19, 0, 0, 0), (20, 0, 0, 0), (21, 0, 0, 0), (22, 0, 0, 0), (23, 0, 0, 0), (24, 0, 0, 0), (25, 0, 0, 0), (26, 0, 0, 0), (27, 0, 0, 0), (28, 0, 0, 0), (29, 0, 0, 0), (30, 0, 0, 0), (31, 0, 0, 0)] */
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+128], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(15)                                // vlcnt(15) = 16 - 1 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+113], s[sgprBeta], v129, v[vgprValuC+113] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v113.l, v[vgprValuC+113]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v113, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(14)                                // vlcnt(14) = 16 - 2 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+114], s[sgprBeta], v130, v[vgprValuC+114] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v114.l, v[vgprValuC+114]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v114, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(13)                                // vlcnt(13) = 16 - 3 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+115], s[sgprBeta], v131, v[vgprValuC+115] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v115.l, v[vgprValuC+115]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v115, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(12)                                // vlcnt(12) = 16 - 4 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+116], s[sgprBeta], v132, v[vgprValuC+116] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v116.l, v[vgprValuC+116]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v116, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(11)                                // vlcnt(11) = 16 - 5 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+117], s[sgprBeta], v133, v[vgprValuC+117] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v117.l, v[vgprValuC+117]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v117, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(10)                                // vlcnt(10) = 16 - 6 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+118], s[sgprBeta], v134, v[vgprValuC+118] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v118.l, v[vgprValuC+118]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v118, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(9)                                 // vlcnt(9) = 16 - 7 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+119], s[sgprBeta], v135, v[vgprValuC+119] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v119.l, v[vgprValuC+119]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v119, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(8)                                 // vlcnt(8) = 16 - 8 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+120], s[sgprBeta], v136, v[vgprValuC+120] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v120.l, v[vgprValuC+120]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v120, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(7)                                 // vlcnt(7) = 16 - 9 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+121], s[sgprBeta], v137, v[vgprValuC+121] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v121.l, v[vgprValuC+121]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v121, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(6)                                 // vlcnt(6) = 16 - 10 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+122], s[sgprBeta], v138, v[vgprValuC+122] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v122.l, v[vgprValuC+122]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v122, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(5)                                 // vlcnt(5) = 16 - 11 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+123], s[sgprBeta], v139, v[vgprValuC+123] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v123.l, v[vgprValuC+123]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v123, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(4)                                 // vlcnt(4) = 16 - 12 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+124], s[sgprBeta], v140, v[vgprValuC+124] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v124.l, v[vgprValuC+124]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v124, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(3)                                 // vlcnt(3) = 16 - 13 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+125], s[sgprBeta], v141, v[vgprValuC+125] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v125.l, v[vgprValuC+125]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v125, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(2)                                 // vlcnt(2) = 16 - 14 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+126], s[sgprBeta], v142, v[vgprValuC+126] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v126.l, v[vgprValuC+126]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v126, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(1)                                 // vlcnt(1) = 16 - 15 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+127], s[sgprBeta], v143, v[vgprValuC+127] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v127.l, v[vgprValuC+127]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v127, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(0)                                 // vlcnt(0) = 16 - 16 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+128], s[sgprBeta], v144, v[vgprValuC+128] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v128.l, v[vgprValuC+128]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v128, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #2 (d1,d0,vc1,vc0) = */
/*    (32,0,0,0:vw1); (33,0,0,0:vw1); (34,0,0,0:vw1); (35,0,0,0:vw1); (36,0,0,0:vw1); (37,0,0,0:vw1); (38,0,0,0:vw1); (39,0,0,0:vw1); (40,0,0,0:vw1); (41,0,0,0:vw1); (42,0,0,0:vw1); (43,0,0,0:vw1); (44,0,0,0:vw1); (45,0,0,0:vw1); (46,0,0,0:vw1); (47,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(32,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v129, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(33,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v130, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(34,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v131, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(35,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v132, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(36,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v133, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(37,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v134, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(38,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v135, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(39,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v136, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(40,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v137, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(41,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v138, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(42,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v139, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(43,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v140, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(44,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v141, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(45,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v142, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(46,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v143, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(47,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v144, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

/* rC *= alpha batchElements=[(32, 0, 0, 0), (33, 0, 0, 0), (34, 0, 0, 0), (35, 0, 0, 0), (36, 0, 0, 0), (37, 0, 0, 0), (38, 0, 0, 0), (39, 0, 0, 0), (40, 0, 0, 0), (41, 0, 0, 0), (42, 0, 0, 0), (43, 0, 0, 0), (44, 0, 0, 0), (45, 0, 0, 0), (46, 0, 0, 0), (47, 0, 0, 0)] */
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+128], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(15)                                // vlcnt(15) = 16 - 1 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+113], s[sgprBeta], v129, v[vgprValuC+113] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v113.l, v[vgprValuC+113]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v113, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(14)                                // vlcnt(14) = 16 - 2 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+114], s[sgprBeta], v130, v[vgprValuC+114] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v114.l, v[vgprValuC+114]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v114, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(13)                                // vlcnt(13) = 16 - 3 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+115], s[sgprBeta], v131, v[vgprValuC+115] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v115.l, v[vgprValuC+115]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v115, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(12)                                // vlcnt(12) = 16 - 4 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+116], s[sgprBeta], v132, v[vgprValuC+116] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v116.l, v[vgprValuC+116]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v116, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(11)                                // vlcnt(11) = 16 - 5 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+117], s[sgprBeta], v133, v[vgprValuC+117] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v117.l, v[vgprValuC+117]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v117, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(10)                                // vlcnt(10) = 16 - 6 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+118], s[sgprBeta], v134, v[vgprValuC+118] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v118.l, v[vgprValuC+118]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v118, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(9)                                 // vlcnt(9) = 16 - 7 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+119], s[sgprBeta], v135, v[vgprValuC+119] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v119.l, v[vgprValuC+119]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v119, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(8)                                 // vlcnt(8) = 16 - 8 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+120], s[sgprBeta], v136, v[vgprValuC+120] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v120.l, v[vgprValuC+120]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v120, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(7)                                 // vlcnt(7) = 16 - 9 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+121], s[sgprBeta], v137, v[vgprValuC+121] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v121.l, v[vgprValuC+121]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v121, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(6)                                 // vlcnt(6) = 16 - 10 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+122], s[sgprBeta], v138, v[vgprValuC+122] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v122.l, v[vgprValuC+122]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v122, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(5)                                 // vlcnt(5) = 16 - 11 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+123], s[sgprBeta], v139, v[vgprValuC+123] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v123.l, v[vgprValuC+123]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v123, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(4)                                 // vlcnt(4) = 16 - 12 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+124], s[sgprBeta], v140, v[vgprValuC+124] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v124.l, v[vgprValuC+124]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v124, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(3)                                 // vlcnt(3) = 16 - 13 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+125], s[sgprBeta], v141, v[vgprValuC+125] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v125.l, v[vgprValuC+125]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v125, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(2)                                 // vlcnt(2) = 16 - 14 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+126], s[sgprBeta], v142, v[vgprValuC+126] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v126.l, v[vgprValuC+126]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v126, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(1)                                 // vlcnt(1) = 16 - 15 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+127], s[sgprBeta], v143, v[vgprValuC+127] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v127.l, v[vgprValuC+127]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v127, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(0)                                 // vlcnt(0) = 16 - 16 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+128], s[sgprBeta], v144, v[vgprValuC+128] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v128.l, v[vgprValuC+128]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v128, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #3 (d1,d0,vc1,vc0) = */
/*    (48,0,0,0:vw1); (49,0,0,0:vw1); (50,0,0,0:vw1); (51,0,0,0:vw1); (52,0,0,0:vw1); (53,0,0,0:vw1); (54,0,0,0:vw1); (55,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(48,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v121, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(49,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v122, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(50,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v123, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(51,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v124, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(52,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v125, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(53,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v126, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(54,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v127, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(55,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v128, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

/* rC *= alpha batchElements=[(48, 0, 0, 0), (49, 0, 0, 0), (50, 0, 0, 0), (51, 0, 0, 0), (52, 0, 0, 0), (53, 0, 0, 0), (54, 0, 0, 0), (55, 0, 0, 0)] */
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+48] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+49] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+50] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+51] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+52] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+53] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+54] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+55] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(7)                                 // vlcnt(7) = 8 - 1 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+113], s[sgprBeta], v121, v[vgprValuC+113] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v113.l, v[vgprValuC+113]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v113, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(6)                                 // vlcnt(6) = 8 - 2 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+114], s[sgprBeta], v122, v[vgprValuC+114] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v114.l, v[vgprValuC+114]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v114, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(5)                                 // vlcnt(5) = 8 - 3 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+115], s[sgprBeta], v123, v[vgprValuC+115] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v115.l, v[vgprValuC+115]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v115, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(4)                                 // vlcnt(4) = 8 - 4 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+116], s[sgprBeta], v124, v[vgprValuC+116] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v116.l, v[vgprValuC+116]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v116, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(3)                                 // vlcnt(3) = 8 - 5 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+117], s[sgprBeta], v125, v[vgprValuC+117] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v117.l, v[vgprValuC+117]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v117, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(2)                                 // vlcnt(2) = 8 - 6 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+118], s[sgprBeta], v126, v[vgprValuC+118] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v118.l, v[vgprValuC+118]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v118, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(1)                                 // vlcnt(1) = 8 - 7 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+119], s[sgprBeta], v127, v[vgprValuC+119] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v119.l, v[vgprValuC+119]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v119, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(0)                                 // vlcnt(0) = 8 - 8 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+120], s[sgprBeta], v128, v[vgprValuC+120] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v120.l, v[vgprValuC+120]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v120, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_FD0_VW1_GSU1_NonEdgeEnd:
label_GW_B1_FD0_VW1_GSU1_Else:
label_GW_B1_FD0_VW1_GSU1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=16 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (1,0,0,0:vw1); (2,0,0,0:vw1); (3,0,0,0:vw1); (4,0,0,0:vw1); (5,0,0,0:vw1); (6,0,0,0:vw1); (7,0,0,0:vw1); (8,0,0,0:vw1); (9,0,0,0:vw1); (10,0,0,0:vw1); (11,0,0,0:vw1); (12,0,0,0:vw1); (13,0,0,0:vw1); (14,0,0,0:vw1); (15,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v106, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v128, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v128, v106, v128, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v127, v128, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v128, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v128, v106, v128, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v130, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v106, v130, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v129, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v130, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v106, v130, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v132, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v106, v132, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v131, v132, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v132, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v106, v132, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v134, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v106, v134, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v133, v134, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v134, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v106, v134, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v136, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v106, v136, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v135, v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v136, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v106, v136, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v138, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v106, v138, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v137, v138, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v138, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v106, v138, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v140, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v106, v140, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v139, v140, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v140, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v106, v140, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v142, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v106, v142, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v141, v142, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v142, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v106, v142, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v144, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v144, v106, v144, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v143, v144, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v144, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v144, v106, v144, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v146, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v146, v106, v146, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v145, v146, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v146, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v146, v106, v146, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v148, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v148, v106, v148, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v147, v148, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v148, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v148, v106, v148, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v150, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v150, v106, v150, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v149, v150, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v150, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v150, v106, v150, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v152, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v152, v106, v152, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v151, v152, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v152, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v152, v106, v152, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v154, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v106, v154, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v153, v154, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v154, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v106, v154, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v156, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v106, v156, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v155, v156, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v156, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v106, v156, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v158, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v106, v158, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v157, v158, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v158, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v106, v158, s30                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0), (3, 0, 0, 0), (4, 0, 0, 0), (5, 0, 0, 0), (6, 0, 0, 0), (7, 0, 0, 0), (8, 0, 0, 0), (9, 0, 0, 0), (10, 0, 0, 0), (11, 0, 0, 0), (12, 0, 0, 0), (13, 0, 0, 0), (14, 0, 0, 0), (15, 0, 0, 0)] */
v_mul_f32 v[vgprValuC+111], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+112], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fma_mix_f32 v[vgprValuC+111], s[sgprBeta], v127, v[vgprValuC+111] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v111.l, v[vgprValuC+111]             // convert C to fp16
buffer_store_b16 v111, v128, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+112], s[sgprBeta], v129, v[vgprValuC+112] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v112.l, v[vgprValuC+112]             // convert C to fp16
buffer_store_b16 v112, v130, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+113], s[sgprBeta], v131, v[vgprValuC+113] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v113.l, v[vgprValuC+113]             // convert C to fp16
buffer_store_b16 v113, v132, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+114], s[sgprBeta], v133, v[vgprValuC+114] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v114.l, v[vgprValuC+114]             // convert C to fp16
buffer_store_b16 v114, v134, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+115], s[sgprBeta], v135, v[vgprValuC+115] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v115.l, v[vgprValuC+115]             // convert C to fp16
buffer_store_b16 v115, v136, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+116], s[sgprBeta], v137, v[vgprValuC+116] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v116.l, v[vgprValuC+116]             // convert C to fp16
buffer_store_b16 v116, v138, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+117], s[sgprBeta], v139, v[vgprValuC+117] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v117.l, v[vgprValuC+117]             // convert C to fp16
buffer_store_b16 v117, v140, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+118], s[sgprBeta], v141, v[vgprValuC+118] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v118.l, v[vgprValuC+118]             // convert C to fp16
buffer_store_b16 v118, v142, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+119], s[sgprBeta], v143, v[vgprValuC+119] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v119.l, v[vgprValuC+119]             // convert C to fp16
buffer_store_b16 v119, v144, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+120], s[sgprBeta], v145, v[vgprValuC+120] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v120.l, v[vgprValuC+120]             // convert C to fp16
buffer_store_b16 v120, v146, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+121], s[sgprBeta], v147, v[vgprValuC+121] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v121.l, v[vgprValuC+121]             // convert C to fp16
buffer_store_b16 v121, v148, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+122], s[sgprBeta], v149, v[vgprValuC+122] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v122.l, v[vgprValuC+122]             // convert C to fp16
buffer_store_b16 v122, v150, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+123], s[sgprBeta], v151, v[vgprValuC+123] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v123.l, v[vgprValuC+123]             // convert C to fp16
buffer_store_b16 v123, v152, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+124], s[sgprBeta], v153, v[vgprValuC+124] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v124.l, v[vgprValuC+124]             // convert C to fp16
buffer_store_b16 v124, v154, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+125], s[sgprBeta], v155, v[vgprValuC+125] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v125.l, v[vgprValuC+125]             // convert C to fp16
buffer_store_b16 v125, v156, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+126], s[sgprBeta], v157, v[vgprValuC+126] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v126.l, v[vgprValuC+126]             // convert C to fp16
buffer_store_b16 v126, v158, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (16,0,0,0:vw1); (17,0,0,0:vw1); (18,0,0,0:vw1); (19,0,0,0:vw1); (20,0,0,0:vw1); (21,0,0,0:vw1); (22,0,0,0:vw1); (23,0,0,0:vw1); (24,0,0,0:vw1); (25,0,0,0:vw1); (26,0,0,0:vw1); (27,0,0,0:vw1); (28,0,0,0:vw1); (29,0,0,0:vw1); (30,0,0,0:vw1); (31,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v106, BufferOOB
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v128, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v128, v106, v128, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v127, v128, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v128, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v128, v106, v128, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v130, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v106, v130, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v129, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v130, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v106, v130, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v132, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v106, v132, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v131, v132, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v132, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v106, v132, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v134, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v106, v134, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v133, v134, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v134, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v106, v134, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v136, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v106, v136, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v135, v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v136, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v106, v136, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v138, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v106, v138, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v137, v138, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v138, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v106, v138, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v140, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v106, v140, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v139, v140, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v140, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v106, v140, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v142, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v106, v142, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v141, v142, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v142, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v106, v142, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v144, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v144, v106, v144, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v143, v144, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v144, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v144, v106, v144, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v146, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v146, v106, v146, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v145, v146, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v146, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v146, v106, v146, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v148, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v148, v106, v148, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v147, v148, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v148, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v148, v106, v148, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v150, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v150, v106, v150, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v149, v150, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v150, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v150, v106, v150, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v152, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v152, v106, v152, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v151, v152, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v152, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v152, v106, v152, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v154, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v106, v154, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v153, v154, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v154, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v106, v154, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v156, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v106, v156, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v155, v156, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v156, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v106, v156, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v158, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v106, v158, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v157, v158, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v158, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v106, v158, s30                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(16, 0, 0, 0), (17, 0, 0, 0), (18, 0, 0, 0), (19, 0, 0, 0), (20, 0, 0, 0), (21, 0, 0, 0), (22, 0, 0, 0), (23, 0, 0, 0), (24, 0, 0, 0), (25, 0, 0, 0), (26, 0, 0, 0), (27, 0, 0, 0), (28, 0, 0, 0), (29, 0, 0, 0), (30, 0, 0, 0), (31, 0, 0, 0)] */
v_mul_f32 v[vgprValuC+111], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+112], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fma_mix_f32 v[vgprValuC+111], s[sgprBeta], v127, v[vgprValuC+111] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v111.l, v[vgprValuC+111]             // convert C to fp16
buffer_store_b16 v111, v128, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+112], s[sgprBeta], v129, v[vgprValuC+112] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v112.l, v[vgprValuC+112]             // convert C to fp16
buffer_store_b16 v112, v130, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+113], s[sgprBeta], v131, v[vgprValuC+113] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v113.l, v[vgprValuC+113]             // convert C to fp16
buffer_store_b16 v113, v132, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+114], s[sgprBeta], v133, v[vgprValuC+114] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v114.l, v[vgprValuC+114]             // convert C to fp16
buffer_store_b16 v114, v134, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+115], s[sgprBeta], v135, v[vgprValuC+115] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v115.l, v[vgprValuC+115]             // convert C to fp16
buffer_store_b16 v115, v136, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+116], s[sgprBeta], v137, v[vgprValuC+116] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v116.l, v[vgprValuC+116]             // convert C to fp16
buffer_store_b16 v116, v138, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+117], s[sgprBeta], v139, v[vgprValuC+117] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v117.l, v[vgprValuC+117]             // convert C to fp16
buffer_store_b16 v117, v140, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+118], s[sgprBeta], v141, v[vgprValuC+118] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v118.l, v[vgprValuC+118]             // convert C to fp16
buffer_store_b16 v118, v142, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+119], s[sgprBeta], v143, v[vgprValuC+119] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v119.l, v[vgprValuC+119]             // convert C to fp16
buffer_store_b16 v119, v144, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+120], s[sgprBeta], v145, v[vgprValuC+120] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v120.l, v[vgprValuC+120]             // convert C to fp16
buffer_store_b16 v120, v146, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+121], s[sgprBeta], v147, v[vgprValuC+121] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v121.l, v[vgprValuC+121]             // convert C to fp16
buffer_store_b16 v121, v148, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+122], s[sgprBeta], v149, v[vgprValuC+122] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v122.l, v[vgprValuC+122]             // convert C to fp16
buffer_store_b16 v122, v150, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+123], s[sgprBeta], v151, v[vgprValuC+123] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v123.l, v[vgprValuC+123]             // convert C to fp16
buffer_store_b16 v123, v152, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+124], s[sgprBeta], v153, v[vgprValuC+124] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v124.l, v[vgprValuC+124]             // convert C to fp16
buffer_store_b16 v124, v154, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+125], s[sgprBeta], v155, v[vgprValuC+125] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v125.l, v[vgprValuC+125]             // convert C to fp16
buffer_store_b16 v125, v156, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+126], s[sgprBeta], v157, v[vgprValuC+126] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v126.l, v[vgprValuC+126]             // convert C to fp16
buffer_store_b16 v126, v158, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #2 (d1,d0,vc1,vc0) = */
/*    (32,0,0,0:vw1); (33,0,0,0:vw1); (34,0,0,0:vw1); (35,0,0,0:vw1); (36,0,0,0:vw1); (37,0,0,0:vw1); (38,0,0,0:vw1); (39,0,0,0:vw1); (40,0,0,0:vw1); (41,0,0,0:vw1); (42,0,0,0:vw1); (43,0,0,0:vw1); (44,0,0,0:vw1); (45,0,0,0:vw1); (46,0,0,0:vw1); (47,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v106, BufferOOB
/* (d1,vc1,d0,vc0)=(32,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v128, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v128, v106, v128, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v127, v128, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v128, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v128, v106, v128, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(33,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v130, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v106, v130, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v129, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v130, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v106, v130, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(34,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v132, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v106, v132, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v131, v132, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v132, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v106, v132, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(35,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v134, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v106, v134, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v133, v134, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v134, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v106, v134, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(36,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v136, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v106, v136, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v135, v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v136, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v106, v136, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(37,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v138, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v106, v138, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v137, v138, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v138, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v106, v138, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(38,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v140, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v106, v140, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v139, v140, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v140, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v106, v140, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(39,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v142, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v106, v142, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v141, v142, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v142, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v106, v142, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(40,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v144, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v144, v106, v144, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v143, v144, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v144, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v144, v106, v144, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(41,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v146, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v146, v106, v146, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v145, v146, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v146, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v146, v106, v146, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(42,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v148, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v148, v106, v148, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v147, v148, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v148, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v148, v106, v148, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(43,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v150, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v150, v106, v150, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v149, v150, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v150, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v150, v106, v150, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(44,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v152, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v152, v106, v152, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v151, v152, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v152, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v152, v106, v152, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(45,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v154, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v106, v154, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v153, v154, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v154, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v106, v154, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(46,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v156, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v106, v156, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v155, v156, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v156, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v106, v156, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(47,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v158, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v106, v158, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v157, v158, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v158, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v106, v158, s30                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(32, 0, 0, 0), (33, 0, 0, 0), (34, 0, 0, 0), (35, 0, 0, 0), (36, 0, 0, 0), (37, 0, 0, 0), (38, 0, 0, 0), (39, 0, 0, 0), (40, 0, 0, 0), (41, 0, 0, 0), (42, 0, 0, 0), (43, 0, 0, 0), (44, 0, 0, 0), (45, 0, 0, 0), (46, 0, 0, 0), (47, 0, 0, 0)] */
v_mul_f32 v[vgprValuC+111], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+112], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fma_mix_f32 v[vgprValuC+111], s[sgprBeta], v127, v[vgprValuC+111] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v111.l, v[vgprValuC+111]             // convert C to fp16
buffer_store_b16 v111, v128, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+112], s[sgprBeta], v129, v[vgprValuC+112] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v112.l, v[vgprValuC+112]             // convert C to fp16
buffer_store_b16 v112, v130, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+113], s[sgprBeta], v131, v[vgprValuC+113] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v113.l, v[vgprValuC+113]             // convert C to fp16
buffer_store_b16 v113, v132, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+114], s[sgprBeta], v133, v[vgprValuC+114] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v114.l, v[vgprValuC+114]             // convert C to fp16
buffer_store_b16 v114, v134, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+115], s[sgprBeta], v135, v[vgprValuC+115] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v115.l, v[vgprValuC+115]             // convert C to fp16
buffer_store_b16 v115, v136, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+116], s[sgprBeta], v137, v[vgprValuC+116] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v116.l, v[vgprValuC+116]             // convert C to fp16
buffer_store_b16 v116, v138, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+117], s[sgprBeta], v139, v[vgprValuC+117] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v117.l, v[vgprValuC+117]             // convert C to fp16
buffer_store_b16 v117, v140, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+118], s[sgprBeta], v141, v[vgprValuC+118] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v118.l, v[vgprValuC+118]             // convert C to fp16
buffer_store_b16 v118, v142, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+119], s[sgprBeta], v143, v[vgprValuC+119] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v119.l, v[vgprValuC+119]             // convert C to fp16
buffer_store_b16 v119, v144, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+120], s[sgprBeta], v145, v[vgprValuC+120] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v120.l, v[vgprValuC+120]             // convert C to fp16
buffer_store_b16 v120, v146, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+121], s[sgprBeta], v147, v[vgprValuC+121] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v121.l, v[vgprValuC+121]             // convert C to fp16
buffer_store_b16 v121, v148, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+122], s[sgprBeta], v149, v[vgprValuC+122] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v122.l, v[vgprValuC+122]             // convert C to fp16
buffer_store_b16 v122, v150, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+123], s[sgprBeta], v151, v[vgprValuC+123] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v123.l, v[vgprValuC+123]             // convert C to fp16
buffer_store_b16 v123, v152, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+124], s[sgprBeta], v153, v[vgprValuC+124] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v124.l, v[vgprValuC+124]             // convert C to fp16
buffer_store_b16 v124, v154, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+125], s[sgprBeta], v155, v[vgprValuC+125] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v125.l, v[vgprValuC+125]             // convert C to fp16
buffer_store_b16 v125, v156, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+126], s[sgprBeta], v157, v[vgprValuC+126] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v126.l, v[vgprValuC+126]             // convert C to fp16
buffer_store_b16 v126, v158, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #3 (d1,d0,vc1,vc0) = */
/*    (48,0,0,0:vw1); (49,0,0,0:vw1); (50,0,0,0:vw1); (51,0,0,0:vw1); (52,0,0,0:vw1); (53,0,0,0:vw1); (54,0,0,0:vw1); (55,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v106, BufferOOB
/* (d1,vc1,d0,vc0)=(48,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v120, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v120, v106, v120, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v119, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v120, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v120, v106, v120, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(49,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v122, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v122, v106, v122, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v121, v122, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v122, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v122, v106, v122, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(50,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v124, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v124, v106, v124, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v123, v124, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v124, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v124, v106, v124, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(51,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v126, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v126, v106, v126, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v125, v126, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v126, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v126, v106, v126, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(52,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v128, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v128, v106, v128, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v127, v128, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v128, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v128, v106, v128, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(53,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v130, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v106, v130, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v129, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v130, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v106, v130, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(54,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v132, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v106, v132, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v131, v132, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v132, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v106, v132, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(55,0,0,0) */
v_add_co_u32 v101, vcc_lo, v101, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v102, v102, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v103, v103, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v100, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v101, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v134, v102, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v106, v134, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v133, v134, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v134, v103, v100, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v106, v134, s30                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(48, 0, 0, 0), (49, 0, 0, 0), (50, 0, 0, 0), (51, 0, 0, 0), (52, 0, 0, 0), (53, 0, 0, 0), (54, 0, 0, 0), (55, 0, 0, 0)] */
v_mul_f32 v[vgprValuC+111], s[sgprAlpha], v[vgprValuC+48] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+112], s[sgprAlpha], v[vgprValuC+49] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+50] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+51] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+52] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+53] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+54] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+55] // Multiply MI out reg with alpha
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fma_mix_f32 v[vgprValuC+111], s[sgprBeta], v119, v[vgprValuC+111] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v111.l, v[vgprValuC+111]             // convert C to fp16
buffer_store_b16 v111, v120, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+112], s[sgprBeta], v121, v[vgprValuC+112] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v112.l, v[vgprValuC+112]             // convert C to fp16
buffer_store_b16 v112, v122, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+113], s[sgprBeta], v123, v[vgprValuC+113] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v113.l, v[vgprValuC+113]             // convert C to fp16
buffer_store_b16 v113, v124, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+114], s[sgprBeta], v125, v[vgprValuC+114] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v114.l, v[vgprValuC+114]             // convert C to fp16
buffer_store_b16 v114, v126, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+115], s[sgprBeta], v127, v[vgprValuC+115] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v115.l, v[vgprValuC+115]             // convert C to fp16
buffer_store_b16 v115, v128, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+116], s[sgprBeta], v129, v[vgprValuC+116] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v116.l, v[vgprValuC+116]             // convert C to fp16
buffer_store_b16 v116, v130, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+117], s[sgprBeta], v131, v[vgprValuC+117] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v117.l, v[vgprValuC+117]             // convert C to fp16
buffer_store_b16 v117, v132, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+118], s[sgprBeta], v133, v[vgprValuC+118] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v118.l, v[vgprValuC+118]             // convert C to fp16
buffer_store_b16 v118, v134, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_End_1:
label_KernelEnd:
s_endpgm                                           // Kernel End
label_ASM_End:  /// The end of the kernel
