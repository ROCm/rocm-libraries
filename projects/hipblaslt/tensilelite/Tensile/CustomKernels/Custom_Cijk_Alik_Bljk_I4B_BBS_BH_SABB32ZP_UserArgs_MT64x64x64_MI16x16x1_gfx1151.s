
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
.text
.protected Custom_Cijk_Alik_Bljk_I4B_BBS_BH_SABB32ZP_UserArgs_MT64x64x64_MI16x16x1_gfx1151
.globl Custom_Cijk_Alik_Bljk_I4B_BBS_BH_SABB32ZP_UserArgs_MT64x64x64_MI16x16x1_gfx1151
.p2align 8
.type Custom_Cijk_Alik_Bljk_I4B_BBS_BH_SABB32ZP_UserArgs_MT64x64x64_MI16x16x1_gfx1151,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Custom_Cijk_Alik_Bljk_I4B_BBS_BH_SABB32ZP_UserArgs_MT64x64x64_MI16x16x1_gfx1151
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 254 // vgprs
  .amdhsa_next_free_sgpr 86 // sgprs
  .amdhsa_group_segment_fixed_size 51200 // lds bytes
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
/* Num VGPR   =254 */
/* Num AccVGPR=0 */
/* Num SGPR   =86 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 16 x 2 */
/* SubGroup= 4 x 32 */
/* VectorWidthA=2 */
/* VectorWidthB=2 */
/* GlobalReadVectorWidthA=8, GlobalReadVectorWidthB=8 */
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
  - .name: Custom_Cijk_Alik_Bljk_I4B_BBS_BH_SABB32ZP_UserArgs_MT64x64x64_MI16x16x1_gfx1151
    .symbol: 'Custom_Cijk_Alik_Bljk_I4B_BBS_BH_SABB32ZP_UserArgs_MT64x64x64_MI16x16x1_gfx1151.kd'
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
        .value_type:      bf16
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
        .value_type:      bf16
        .address_space:   generic
      - .name:            C
        .size:            8
        .offset:          80
        .value_kind:      global_buffer
        .value_type:      bf16
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
    .group_segment_fixed_size:   51200
    .kernarg_segment_align:      8
    .kernarg_segment_size:       160
    .max_flat_workgroup_size:    128
    .private_segment_fixed_size: 0
    .sgpr_count:                 86
    .sgpr_spill_count:           0
    .vgpr_count:                 254
    .vgpr_spill_count:           0
    .wavefront_size:             32
...
.end_amdgpu_metadata
Custom_Cijk_Alik_Bljk_I4B_BBS_BH_SABB32ZP_UserArgs_MT64x64x64_MI16x16x1_gfx1151:
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
/* ValuC range: [0-32), serializedStore enabled */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprBase, 60
.set vgprLocalWriteAddrA, 56
.set vgprLocalWriteAddrB, 57
.set vgprGlobalReadOffsetA, 32
.set vgprGlobalReadOffsetB, 36
.set vgprGlobalReadOffsetScaleA, 40
.set vgprG2LScaleA, 44
.set vgprG2LScaleZeroA, 48
.set vgprGlobalReadOffsetScaleZeroA, 52
.set vgprLocalReadAddrA, 58
.set vgprLocalReadAddrB, 59
.set vgprSerial, 222

/******************************************/
/* VGPR Macro Assignments                 */
/******************************************/
.set vgprValuA_X0_I0_BASE, vgprBase+0
.set vgprValuB_X0_I0_BASE, vgprBase+65
.set vgprG2LA_BASE, vgprBase+130
.set vgprG2LB_BASE, vgprBase+146
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+16
.set vgprValuA_X2_I0, vgprValuA_X0_I0_BASE+32
.set vgprValuA_X3_I0, vgprValuA_X0_I0_BASE+48
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X1_I0, vgprValuB_X0_I0_BASE+16
.set vgprValuB_X2_I0, vgprValuB_X0_I0_BASE+32
.set vgprValuB_X3_I0, vgprValuB_X0_I0_BASE+48
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
.set sgprAddressScaleZeroA, 56
.set sgprSrdScaleZeroA, 60

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
.set MT1, 64
.set DepthU, 64
/* Number of elements to shift-left SRD */
.set SrdShiftLeftA, 8
.set SrdShiftLeftB, 8
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
s_load_b64 s[sgprAddressScaleA:sgprAddressScaleA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x58
s_load_b64 s[sgprAddressScaleB:sgprAddressScaleB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x60
s_load_b64 s[sgprAddressScaleZeroA:sgprAddressScaleZeroA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x68
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
s_mov_b32 m0, 0xc800                               // LDS clamp at 51200 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id
s_mov_b32 vcc_hi, 0                                // Ensure hi bits are zero

/* remap workgroup to XCCs */
s_lshr_b32 s68, s[sgprWGM], 0x10                   // Get WGMXCC
s_ff1_i32_b32 s68, s68                             // Get log(WGMXCC)
s_lshr_b32 s69, s[sgprWGM], 0x16                   // Get CU_Count
/* remap WGs if WGMXCC > 1 ( log(WGMXCC) > 0 ) */
s_cmp_gt_i32 s68, 0
s_cbranch_scc0 label_skip_WGMXCC
/* only remap WGs in the range */
s_lshr_b32 s65, s23, s68
s_lshl_b32 s65, s65, s68
s_cmp_ge_u32 s[sgprWorkGroup0], s65
s_cbranch_scc1 label_skip_WGMXCC
s_cmp_eq_u32 s69, 0                                // CU_Count == 0 ?
s_cbranch_scc0 label_XCCG_nonzero
s_lshr_b32 s65, s[sgprWorkGroup0], s68
s_bfm_b32 s66, s68, 0
s_and_b32 s66, s[sgprWorkGroup0], s66
s_lshr_b32 s67, s23, s68
s_mul_i32 s66, s66, s67
s_add_u32 s[sgprWorkGroup0], s65, s66
s_branch label_skip_WGMXCC
label_XCCG_nonzero:
/* temp0 = (wg//CU_Count)*CU_Count */
v_cvt_f64_u32 v[6:7], s69                          // s65 = s[sgprWorkGroup0] / s69
v_rcp_f64 v[6:7], v[6:7]                           // s65 = s[sgprWorkGroup0] / s69
v_cvt_f64_u32 v[8:9], s[sgprWorkGroup0]            // s65 = s[sgprWorkGroup0] / s69
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s65 = s[sgprWorkGroup0] / s69
v_cvt_u32_f64 v6, v[6:7]                           // s65 = s[sgprWorkGroup0] / s69
v_mul_lo_u32 v7, v6, s69                           // s65 = s[sgprWorkGroup0] / s69
v_sub_nc_u32 v8, s[sgprWorkGroup0], v7             // s65 = s[sgprWorkGroup0] / s69
v_cmp_ge_u32 vcc_lo, v8, s69                       // s65 = s[sgprWorkGroup0] / s69
s_mov_b32 exec_lo, vcc_lo                          // s65 = s[sgprWorkGroup0] / s69
v_add_nc_u32 v6, v6, 1                             // s65 = s[sgprWorkGroup0] / s69
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v7, v6, s69                           // s65 = s[sgprWorkGroup0] / s69
v_sub_nc_u32 v8, s[sgprWorkGroup0], v7             // s65 = s[sgprWorkGroup0] / s69
v_readfirstlane_b32 s65, v6                        // quotient
v_readfirstlane_b32 s66, v8                        // remainder
s_mul_i32 s65, s65, s69
/* temp1 = (wg%CU_Count)//WGMXCC */
s_lshr_b32 s66, s66, s68
/* temp0 = temp0 + temp1 */
s_add_u32 s65, s65, s66
/* temp1 = (wg%WGMXCC) * ((WGs - (WGs//CU_Count) * CU_Count) if (wg > (WGs//CU_Count) * CU_Count) else CU_Count)//WGMXCC */
v_cvt_f64_u32 v[6:7], s69                          // s66 = s23 / s69
v_rcp_f64 v[6:7], v[6:7]                           // s66 = s23 / s69
v_cvt_f64_u32 v[8:9], s23                          // s66 = s23 / s69
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s66 = s23 / s69
v_cvt_u32_f64 v6, v[6:7]                           // s66 = s23 / s69
v_mul_lo_u32 v7, v6, s69                           // s66 = s23 / s69
v_sub_nc_u32 v8, s23, v7                           // s66 = s23 / s69
v_cmp_ge_u32 vcc_lo, v8, s69                       // s66 = s23 / s69
s_mov_b32 exec_lo, vcc_lo                          // s66 = s23 / s69
v_add_nc_u32 v6, v6, 1                             // s66 = s23 / s69
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s66, v6                        // quotient
s_mul_i32 s66, s66, s69
s_sub_u32 s67, s23, s66
s_cmp_gt_u32 s[sgprWorkGroup0], s66
s_cselect_b32 s66, s67, s69
s_lshr_b32 s66, s66, s68
s_bfm_b32 s67, s68, 0
s_and_b32 s67, s[sgprWorkGroup0], s67
s_mul_i32 s66, s66, s67
/* WorkGroup0 = temp0 + temp1 */
s_add_u32 s[sgprWorkGroup0], s65, s66
label_skip_WGMXCC:  /// skip WGMXCC if no enough WGs to remap
s_cmp_eq_u32 s21, 3
s_cbranch_scc1 label_ArgType3_Routed_To_ArgType0
s_cmp_eq_u32 s21, 0
s_cbranch_scc0 label_MultiGemm
label_ArgType3_Routed_To_ArgType0:
/* init: add vgpr [60...249) to pool */
/* init: add vgpr [0...32) to pool */
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
v_lshlrev_b32 v0, 1, v0                            // 4. apply VectorWidth: bnOffset = bnOffset * vw(2)
v_lshrrev_b32 v4, 5, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(32)
v_and_b32 v4, 1, v4                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v0, v4, 11, v0                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(2048); 7. final local read offset: flrOffset = lrOffset + WOffset
/* lr1J */
v_and_b32 v2, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v1, 6, v1                            // 1. N offset: nOffset = nIdx * nStride(64)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
v_lshlrev_b32 v1, 1, v1                            // 4. apply VectorWidth: bnOffset = bnOffset * vw(2)
v_lshrrev_b32 v3, 6, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
v_and_b32 v3, 1, v3                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v1, v3, 11, v1                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(2048); 7. final local read offset: flrOffset = lrOffset + WOffset

/* local read addresses: final offsets a */
v_lshrrev_b32 v2, 5, v[vgprSerial]                 // 2 = Serial / 32
v_lshrrev_b32 v2, 2, v2                            // LSU offset: Get LSU wave_id
s_mov_b32 s16, 64                                  // LSU offset: stride = lsuStride(64) when umlds==True
v_mul_lo_u32 v2, s16, v2                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_nc_u32 v[vgprLocalReadAddrA], v2, v0         // Final Offset: offset = (lro0+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrA], 1, v[vgprLocalReadAddrA] //  (multiple bpe)
v_lshrrev_b32 v3, 8, v[vgprLocalReadAddrA]         // Final Offset: padding 32 per block 256
v_lshl_add_u32 v[vgprLocalReadAddrA], v3, 5, v[vgprLocalReadAddrA] // Final Offset: padding 32 per block 256

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 5, v[vgprSerial]                 // 0 = Serial / 32
v_lshrrev_b32 v0, 2, v0                            // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(64) when umlds==True (dup assign opt.)
v_mul_lo_u32 v0, s16, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_nc_u32 v[vgprLocalReadAddrB], v0, v1         // Final Offset: offset = (lro1+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrB], 1, v[vgprLocalReadAddrB] //  (multiple bpe)
v_lshrrev_b32 v2, 8, v[vgprLocalReadAddrB]         // Final Offset: padding 32 per block 256
v_lshl_add_u32 v[vgprLocalReadAddrB], v2, 5, v[vgprLocalReadAddrB] // Final Offset: padding 32 per block 256

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
/* LVCB = 8 */
/* v3 = B-unroll = serial%LVCB */
v_lshrrev_b32 v2, 3, v[vgprSerial]                 // 2 = Serial / 8
v_and_b32 v3, 7, v[vgprSerial]                     // 3 = Serial % 8
/* unroll *= glvw */
v_lshlrev_b32 v3, 3, v3                            // v3 = v3 * 8
v_mov_b32 v5, v3                                   // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v4 */
/* lwaUnrollAssignmentB = v5 */

/* local write addresses: first offset a */
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x40, v0     // lwAL**(DepthU_Compute + PAD)
v_add_nc_u32 v[vgprLocalWriteAddrA], v4, v[vgprLocalWriteAddrA] // lwFOA = (lwAA + lwAL*(DepthU+PAD))
v_lshlrev_b32 v[vgprLocalWriteAddrA], 1, v[vgprLocalWriteAddrA] //  (multiple bpe)
v_lshrrev_b32 v6, 8, v[vgprLocalWriteAddrA]        // padding 32 per block 256
v_lshl_add_u32 v[vgprLocalWriteAddrA], v6, 5, v[vgprLocalWriteAddrA] // padding 32 per block 256

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x40, v2     // lwBL**(DepthU_Compute + PAD)
v_add_nc_u32 v[vgprLocalWriteAddrB], v5, v[vgprLocalWriteAddrB] // lwFOB = (lwBB + lwBL*(DepthU+PAD))
v_lshlrev_b32 v[vgprLocalWriteAddrB], 1, v[vgprLocalWriteAddrB] //  (multiple bpe)
v_lshrrev_b32 v6, 8, v[vgprLocalWriteAddrB]        // padding 32 per block 256
v_lshl_add_u32 v[vgprLocalWriteAddrB], v6, 5, v[vgprLocalWriteAddrB] // padding 32 per block 256
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
s_and_b32 s16, s[sgprArgType], 0xff                // mask ArgType domain (bit 8 = TDM wave-parity)
s_cmp_eq_u32 s16, 2                                // ArgType == 2 ?
s_cbranch_scc1 label_IsExternalValid               // branch if ArgType == 2
s_mov_b32 s15, 112                                 // KernArgAddressOffset
s_mul_i32 s70, s20, 4
s_mov_b64 s[64:65], s[sgprKernArgAddress:sgprKernArgAddress+1]
s_branch label_IsExternalValidEnd
label_IsExternalValid:
s_mov_b32 s15, 228
s_mov_b32 s70, 0
s_mov_b64 s[64:65], s[sgprKernArgAddress:sgprKernArgAddress+1]
label_IsExternalValidEnd:

/* Grouped Gemm:: prefetch 1 arg load */
s_mov_b32 s14, 1
s_mov_b32 s71, 0
s_load_b128 s[24:27], s[64:65], s70
s_cmpk_eq_u32 s20, 1                               // if gemm_count is 1?
s_cbranch_scc1 label_wgTable_noLoadLoop

/* Grouped Gemm:: accumulate numTiles for each gemm */
/* Grouped Gemm:: loop start */
label_Loop_GemmCount:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s68, s24, 6                             // s68 = s24 / 64
s_and_b32 s66, 63, s24                             // s66 = s24 % 64
s_addc_u32 s68, s68, 0
s_lshr_b32 s69, s25, 6                             // s69 = s25 / 64
s_and_b32 s66, 63, s25                             // s66 = s25 % 64
s_addc_u32 s69, s69, 0
s_mul_i32 s68, s68, s69
s_mul_i32 s68, s68, s26
s_and_b32 s69, s[sgprGSU], 0xfff                   // Restore GSU
s_mul_i32 s68, s68, s69
s_add_u32 s71, s71, s68
s_cmp_lt_u32 s[sgprWorkGroup0], s71
s_cbranch_scc1 label_FOUND
s_add_u32 s70, s70, s15
s_load_b128 s[24:27], s[64:65], s70
s_add_u32 s14, s14, 1
s_cmp_lt_u32 s14, s20
s_cbranch_scc1 label_Loop_GemmCount

/* Grouped Gemm:: noLoadLoop */
label_wgTable_noLoadLoop:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s68, s24, 6                             // s68 = s24 / 64
s_and_b32 s66, 63, s24                             // s66 = s24 % 64
s_addc_u32 s68, s68, 0
s_lshr_b32 s69, s25, 6                             // s69 = s25 / 64
s_and_b32 s66, 63, s25                             // s66 = s25 % 64
s_addc_u32 s69, s69, 0
s_mul_i32 s68, s68, s69
s_mul_i32 s68, s68, s26
s_and_b32 s64, s[sgprGSU], 0xfff                   // Restore GSU
s_mul_i32 s68, s68, s64
s_add_u32 s71, s71, s68

/* Grouped Gemm:: gemmIndex found */
label_FOUND:
s_sub_u32 s65, s14, 1
s_sub_u32 s64, s71, s68
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s64
/* Check if custom structure pointer is null */
s_and_b32 s16, s[sgprArgType], 0xff                // mask ArgType domain (bit 8 = TDM wave-parity)
s_cmp_eq_u32 s16, 2                                // ArgType == 2 ?
s_cbranch_scc1 label_LoadExternalStruct            // branch if ArgType == 2

/* Grouped Gemm: offset argument address to gemm */
/* Grouped Gemm: offset address from wg_table_start to args_start */
s_lshl2_add_u32 s[sgprKernArgAddress], s20, s[sgprKernArgAddress]
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s65, s65, 112                            // KernArgAddressOffset
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s65
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_b512 s[28:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_b64 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
s_load_b64 s[sgprAddressScaleA:sgprAddressScaleA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x58
s_load_b64 s[sgprAddressScaleB:sgprAddressScaleB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x60
s_load_b64 s[sgprAddressScaleZeroA:sgprAddressScaleZeroA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x68
s_branch label_LoadExternalStructEnd
label_LoadExternalStruct:
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s65, s65, 228
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s65
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
s_load_b64 s[sgprAddressScaleZeroA:sgprAddressScaleZeroA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x74
label_LoadExternalStructEnd:
/* init: add vgpr [60...249) to pool */
/* init: add vgpr [0...32) to pool */
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
v_lshlrev_b32 v0, 1, v0                            // 4. apply VectorWidth: bnOffset = bnOffset * vw(2)
v_lshrrev_b32 v4, 5, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(32)
v_and_b32 v4, 1, v4                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v0, v4, 11, v0                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(2048); 7. final local read offset: flrOffset = lrOffset + WOffset
/* lr1J */
v_and_b32 v2, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v1, 6, v1                            // 1. N offset: nOffset = nIdx * nStride(64)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
v_lshlrev_b32 v1, 1, v1                            // 4. apply VectorWidth: bnOffset = bnOffset * vw(2)
v_lshrrev_b32 v3, 6, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
v_and_b32 v3, 1, v3                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v1, v3, 11, v1                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(2048); 7. final local read offset: flrOffset = lrOffset + WOffset

/* local read addresses: final offsets a */
v_lshrrev_b32 v2, 5, v[vgprSerial]                 // 2 = Serial / 32
v_lshrrev_b32 v2, 2, v2                            // LSU offset: Get LSU wave_id
s_mov_b32 s16, 64                                  // LSU offset: stride = lsuStride(64) when umlds==True
v_mul_lo_u32 v2, s16, v2                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_nc_u32 v[vgprLocalReadAddrA], v2, v0         // Final Offset: offset = (lro0+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrA], 1, v[vgprLocalReadAddrA] //  (multiple bpe)
v_lshrrev_b32 v3, 8, v[vgprLocalReadAddrA]         // Final Offset: padding 32 per block 256
v_lshl_add_u32 v[vgprLocalReadAddrA], v3, 5, v[vgprLocalReadAddrA] // Final Offset: padding 32 per block 256

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 5, v[vgprSerial]                 // 0 = Serial / 32
v_lshrrev_b32 v0, 2, v0                            // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(64) when umlds==True (dup assign opt.)
v_mul_lo_u32 v0, s16, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_nc_u32 v[vgprLocalReadAddrB], v0, v1         // Final Offset: offset = (lro1+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrB], 1, v[vgprLocalReadAddrB] //  (multiple bpe)
v_lshrrev_b32 v2, 8, v[vgprLocalReadAddrB]         // Final Offset: padding 32 per block 256
v_lshl_add_u32 v[vgprLocalReadAddrB], v2, 5, v[vgprLocalReadAddrB] // Final Offset: padding 32 per block 256

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
/* LVCB = 8 */
/* v3 = B-unroll = serial%LVCB */
v_lshrrev_b32 v2, 3, v[vgprSerial]                 // 2 = Serial / 8
v_and_b32 v3, 7, v[vgprSerial]                     // 3 = Serial % 8
/* unroll *= glvw */
v_lshlrev_b32 v3, 3, v3                            // v3 = v3 * 8
v_mov_b32 v5, v3                                   // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v4 */
/* lwaUnrollAssignmentB = v5 */

/* local write addresses: first offset a */
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x40, v0     // lwAL**(DepthU_Compute + PAD)
v_add_nc_u32 v[vgprLocalWriteAddrA], v4, v[vgprLocalWriteAddrA] // lwFOA = (lwAA + lwAL*(DepthU+PAD))
v_lshlrev_b32 v[vgprLocalWriteAddrA], 1, v[vgprLocalWriteAddrA] //  (multiple bpe)
v_lshrrev_b32 v6, 8, v[vgprLocalWriteAddrA]        // padding 32 per block 256
v_lshl_add_u32 v[vgprLocalWriteAddrA], v6, 5, v[vgprLocalWriteAddrA] // padding 32 per block 256

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x40, v2     // lwBL**(DepthU_Compute + PAD)
v_add_nc_u32 v[vgprLocalWriteAddrB], v5, v[vgprLocalWriteAddrB] // lwFOB = (lwBB + lwBL*(DepthU+PAD))
v_lshlrev_b32 v[vgprLocalWriteAddrB], 1, v[vgprLocalWriteAddrB] //  (multiple bpe)
v_lshrrev_b32 v6, 8, v[vgprLocalWriteAddrB]        // padding 32 per block 256
v_lshl_add_u32 v[vgprLocalWriteAddrB], v6, 5, v[vgprLocalWriteAddrB] // padding 32 per block 256
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
.set sgprSrdA, 64
.set sgprSrdB, 68
.set sgprShadowLimitA, 58
.set sgprShadowLimitB, 72
.set sgprStaggerUIter, 74
.set sgprWrapUA, 75
.set sgprWrapUB, 77
.set sgprGlobalReadIncsA, 79
.set sgprGlobalReadIncsB, 80
s_and_b32 s16, s[sgprArgType], 0xff                // mask ArgType domain (bit 8 = TDM wave-parity)
s_cmp_eq_u32 s16, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_Skip_Address_Prepad_For_Pointer_Array
s_sub_u32 s[sgprAddressA+0], s[sgprAddressA+0], 4  // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressA+1], s[sgprAddressA+1], 0 // pre-pad to make room for possible pointer shift
s_sub_u32 s[sgprAddressB+0], s[sgprAddressB+0], 16 // pre-pad to make room for possible pointer shift
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
v_add_co_u32 v11, vcc_lo, 16, v10                  // groB1J_1 += LSPB
v_add_co_u32 v12, vcc_lo, 16, v11                  // groB1J_2 += LSPB
v_add_co_u32 v13, vcc_lo, 16, v12                  // groB1J_3 += LSPB

/* global read addresses: unroll offsets a */
v_mov_b32 v14, v1                                  // groAL_0

/* global read addresses: unroll offsets b */
v_mov_b32 v15, v3                                  // groBL_0

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
v_cvt_f32_u32 v16, s[sgprGSUSumIdx+1]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v16, v16                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v17, s[sgprLoopCounterL]             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v16, v16, v17                            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v16, v16                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v17, v16, s[sgprGSUSumIdx+1]         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_nc_u32 v17, s[sgprLoopCounterL], v17         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmp_eq_u32 vcc_lo, v17, s[sgprGSUSumIdx+1]       // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, vcc_lo                          // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_nc_u32 v16, 1, v16                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v17, 0                                   // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v17, s[sgprGSUSumIdx+1]       // overflow happened in remainder
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v16, v16, 1                           // quotient - 1
v_mul_u32_u24 v17, v16, s[sgprGSUSumIdx+1]         // re-calculate remainder
v_sub_nc_u32 v17, s[sgprLoopCounterL], v17         // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v16       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v17        // remainder
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
s_and_b32 s20, s[sgprArgType], 0xff                // mask ArgType domain (bit 8 = TDM wave-parity)
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
s_add_u32 s[sgprStrideScaleA], s[sgprSizeL], 0x1f  // SizeL + G-1
s_lshr_b32 s[sgprStrideScaleA], s[sgprStrideScaleA], 5 // StrideScaleA = ceil(SizeL/32) scale elements
s_mul_i32 s16, s[sgprWorkGroup0], 64               // scaleA: workgroup row origin
s_mul_i32 s16, s16, s[sgprStrideScaleA]            // scaleA: * row stride
s_mul_i32 s16, s16, 2                              // scaleA: elements -> bytes
s_mul_i32 s17, s[sgprSizeI], s[sgprStrideScaleA]   // scaleA: SizeI * row stride
s_mul_i32 s17, s17, 2                              // scaleA: tensor bytes
s_sub_u32 s[sgprSrdScaleA+2], s17, s16             // scaleA: buffer limit from the workgroup origin
s_add_u32 s[sgprSrdScaleA+0], s[sgprAddressScaleA+0], s16 // scaleA: SRD base lo
s_addc_u32 s[sgprSrdScaleA+1], s[sgprAddressScaleA+1], 0 // scaleA: SRD base hi
s_mov_b32 s[sgprSrdScaleA+3], Srd127_96            // scaleA: set bits 127_96 in SRD

/* global read addresses: block-scale A zero-point srd */
s_mul_i32 s16, s[sgprWorkGroup0], 64               // scaleZeroA: workgroup row origin
s_lshr_b32 s16, s16, 1                             // scaleZeroA: 2 rows per byte
s_mul_i32 s16, s16, s[sgprStrideScaleA]            // scaleZeroA: * kGroups
s_add_u32 s17, s[sgprSizeI], 1                     // scaleZeroA: SizeI + 1
s_lshr_b32 s17, s17, 1                             // scaleZeroA: ceil(SizeI/2) row pairs
s_mul_i32 s17, s17, s[sgprStrideScaleA]            // scaleZeroA: total bytes
s_sub_u32 s[sgprSrdScaleZeroA+2], s17, s16         // scaleZeroA: buffer limit from the workgroup origin
s_add_u32 s[sgprSrdScaleZeroA+0], s[sgprAddressScaleZeroA+0], s16 // scaleZeroA: SRD base lo
s_addc_u32 s[sgprSrdScaleZeroA+1], s[sgprAddressScaleZeroA+1], 0 // scaleZeroA: SRD base hi
s_mov_b32 s[sgprSrdScaleZeroA+3], Srd127_96        // scaleZeroA: set bits 127_96 in SRD

/* global read addresses: addresses b */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s19, s[sgprWorkGroup1], 64            // WorkGroup[01] * MT
s_mul_i32 s18, s[sgprWorkGroup1], 64               // WorkGroup[01] * MT
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
v_cvt_f32_u32 v16, s[sgprGSUSumIdx+1]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v16, v16                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v17, s[sgprLoopCounterL]             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v16, v16, v17                            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v16, v16                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v17, v16, s[sgprGSUSumIdx+1]         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_nc_u32 v17, s[sgprLoopCounterL], v17         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmp_eq_u32 vcc_lo, v17, s[sgprGSUSumIdx+1]       // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, vcc_lo                          // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_nc_u32 v16, 1, v16                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v17, 0                                   // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v17, s[sgprGSUSumIdx+1]       // overflow happened in remainder
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v16, v16, 1                           // quotient - 1
v_mul_u32_u24 v17, v16, s[sgprGSUSumIdx+1]         // re-calculate remainder
v_sub_nc_u32 v17, s[sgprLoopCounterL], v17         // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v16       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v17        // remainder
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
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 16 // extend limit for pre-pad
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
s_waitcnt lgkmcnt(0)                               // Wait for pointer-array SRD load before reusing base SGPR
s_load_b64 s[16:17], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x88 // Load batchOffsetB from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s16        // Add batch offset to B address (low)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s17       // Add batch offset to B address (high)
s_sub_u32 s[sgprSrdB+0], s[sgprSrdB+0], 16         // pre-pad to make room for possible pointer shift
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
v_mul_lo_u32 v16, s[sgprStrideA0I], v[6]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+0+0], vcc_lo, v[14], v[16+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+0+0], 0x8, v[vgprGlobalReadOffsetA+0+0] // add prepad for pointer shift
v_lshrrev_b32 v16, 5, v14                          // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+0], s[sgprStrideScaleA], v6 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+0], v16, v[vgprGlobalReadOffsetScaleA+0] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+0], 1, v[vgprGlobalReadOffsetScaleA+0] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetScaleZeroA+0], 1, v6 // scaleZeroA: row / 2
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleZeroA+0], s[sgprStrideScaleA], v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: (row/2) * kGroups
v_add_nc_u32 v[vgprGlobalReadOffsetScaleZeroA+0], v16, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: + kGroup -> byte offset
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleZeroA+0], 1, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: make room for the nibble bit
v_and_b32 v16, 1, v6                               // scaleZeroA: nibble = row & 1
v_or_b32 v[vgprGlobalReadOffsetScaleZeroA+0], v16, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: (byte << 1) | nibble
v_lshrrev_b32 v[vgprGlobalReadOffsetA+0], 1, v[vgprGlobalReadOffsetA+0] //  (multiple bpe)
v_mul_lo_u32 v16, s[sgprStrideA0I], v[7]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+1+0], vcc_lo, v[14], v[16+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+1+0], 0x8, v[vgprGlobalReadOffsetA+1+0] // add prepad for pointer shift
v_lshrrev_b32 v16, 5, v14                          // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+1], s[sgprStrideScaleA], v7 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+1], v16, v[vgprGlobalReadOffsetScaleA+1] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+1], 1, v[vgprGlobalReadOffsetScaleA+1] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetScaleZeroA+1], 1, v7 // scaleZeroA: row / 2
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleZeroA+1], s[sgprStrideScaleA], v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: (row/2) * kGroups
v_add_nc_u32 v[vgprGlobalReadOffsetScaleZeroA+1], v16, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: + kGroup -> byte offset
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleZeroA+1], 1, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: make room for the nibble bit
v_and_b32 v16, 1, v7                               // scaleZeroA: nibble = row & 1
v_or_b32 v[vgprGlobalReadOffsetScaleZeroA+1], v16, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: (byte << 1) | nibble
v_lshrrev_b32 v[vgprGlobalReadOffsetA+1], 1, v[vgprGlobalReadOffsetA+1] //  (multiple bpe)
v_mul_lo_u32 v16, s[sgprStrideA0I], v[8]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+2+0], vcc_lo, v[14], v[16+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+2+0], 0x8, v[vgprGlobalReadOffsetA+2+0] // add prepad for pointer shift
v_lshrrev_b32 v16, 5, v14                          // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+2], s[sgprStrideScaleA], v8 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+2], v16, v[vgprGlobalReadOffsetScaleA+2] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+2], 1, v[vgprGlobalReadOffsetScaleA+2] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetScaleZeroA+2], 1, v8 // scaleZeroA: row / 2
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleZeroA+2], s[sgprStrideScaleA], v[vgprGlobalReadOffsetScaleZeroA+2] // scaleZeroA: (row/2) * kGroups
v_add_nc_u32 v[vgprGlobalReadOffsetScaleZeroA+2], v16, v[vgprGlobalReadOffsetScaleZeroA+2] // scaleZeroA: + kGroup -> byte offset
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleZeroA+2], 1, v[vgprGlobalReadOffsetScaleZeroA+2] // scaleZeroA: make room for the nibble bit
v_and_b32 v16, 1, v8                               // scaleZeroA: nibble = row & 1
v_or_b32 v[vgprGlobalReadOffsetScaleZeroA+2], v16, v[vgprGlobalReadOffsetScaleZeroA+2] // scaleZeroA: (byte << 1) | nibble
v_lshrrev_b32 v[vgprGlobalReadOffsetA+2], 1, v[vgprGlobalReadOffsetA+2] //  (multiple bpe)
v_mul_lo_u32 v16, s[sgprStrideA0I], v[9]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+3+0], vcc_lo, v[14], v[16+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+3+0], 0x8, v[vgprGlobalReadOffsetA+3+0] // add prepad for pointer shift
v_lshrrev_b32 v16, 5, v14                          // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+3], s[sgprStrideScaleA], v9 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+3], v16, v[vgprGlobalReadOffsetScaleA+3] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+3], 1, v[vgprGlobalReadOffsetScaleA+3] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetScaleZeroA+3], 1, v9 // scaleZeroA: row / 2
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleZeroA+3], s[sgprStrideScaleA], v[vgprGlobalReadOffsetScaleZeroA+3] // scaleZeroA: (row/2) * kGroups
v_add_nc_u32 v[vgprGlobalReadOffsetScaleZeroA+3], v16, v[vgprGlobalReadOffsetScaleZeroA+3] // scaleZeroA: + kGroup -> byte offset
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleZeroA+3], 1, v[vgprGlobalReadOffsetScaleZeroA+3] // scaleZeroA: make room for the nibble bit
v_and_b32 v16, 1, v9                               // scaleZeroA: nibble = row & 1
v_or_b32 v[vgprGlobalReadOffsetScaleZeroA+3], v16, v[vgprGlobalReadOffsetScaleZeroA+3] // scaleZeroA: (byte << 1) | nibble
v_lshrrev_b32 v[vgprGlobalReadOffsetA+3], 1, v[vgprGlobalReadOffsetA+3] //  (multiple bpe)
/* ============================================================= */

/* global read addresses: final offsets b */
/* ============================================================= */
v_mul_lo_u32 v6, s[sgprStrideB1J], v[10]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+0+0], vcc_lo, v[15], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+0+0], 0x8, v[vgprGlobalReadOffsetB+0+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+0], 1, v[vgprGlobalReadOffsetB+0] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[11]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+1+0], vcc_lo, v[15], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+1+0], 0x8, v[vgprGlobalReadOffsetB+1+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+1], 1, v[vgprGlobalReadOffsetB+1] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[12]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+2+0], vcc_lo, v[15], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+2+0], 0x8, v[vgprGlobalReadOffsetB+2+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+2], 1, v[vgprGlobalReadOffsetB+2] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[13]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+3+0], vcc_lo, v[15], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+3+0], 0x8, v[vgprGlobalReadOffsetB+3+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+3], 1, v[vgprGlobalReadOffsetB+3] //  (multiple bpe)
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
buffer_load_b32 v[vgprG2LA+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b32 v[vgprG2LA+6], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b32 v[vgprG2LA+10], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b32 v[vgprG2LA+14], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_3_0

/* global read block-scale A */
buffer_load_d16_b16 v[vgprG2LScaleA+0], v[vgprGlobalReadOffsetScaleA+0], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 0
buffer_load_d16_b16 v[vgprG2LScaleA+1], v[vgprGlobalReadOffsetScaleA+1], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 1
buffer_load_d16_b16 v[vgprG2LScaleA+2], v[vgprGlobalReadOffsetScaleA+2], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 2
buffer_load_d16_b16 v[vgprG2LScaleA+3], v[vgprGlobalReadOffsetScaleA+3], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 3
v_lshrrev_b32 v0, 1, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: drop the nibble bit
buffer_load_d16_u8 v[vgprG2LScaleZeroA+0], v0, s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 0
v_lshrrev_b32 v0, 1, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: drop the nibble bit
buffer_load_d16_u8 v[vgprG2LScaleZeroA+1], v0, s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 1
v_lshrrev_b32 v0, 1, v[vgprGlobalReadOffsetScaleZeroA+2] // scaleZeroA: drop the nibble bit
buffer_load_d16_u8 v[vgprG2LScaleZeroA+2], v0, s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 2
v_lshrrev_b32 v0, 1, v[vgprGlobalReadOffsetScaleZeroA+3] // scaleZeroA: drop the nibble bit
buffer_load_d16_u8 v[vgprG2LScaleZeroA+3], v0, s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 3
buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b128 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b128 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b128 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0

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
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], 0x4 // scaleA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], 0x4 // scaleA limit -= inc
s_add_u32 s[sgprSrdScaleZeroA+0], s[sgprSrdScaleZeroA+0], 0x2 // scaleZeroA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleZeroA+1], s[sgprSrdScaleZeroA+1], 0 // scaleZeroA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleZeroA+2], s[sgprSrdScaleZeroA+2], 0x2 // scaleZeroA limit -= inc

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
s_and_b32 s81, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s81, 1                                // GSU == 1 ?
s_cbranch_scc1 label_ArgTypeCheckD                 // Handling General Batched GEMM SRD initialization
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], s[sgprAddressD+0:sgprAddressD+0+1] // init SRD base address
s_branch label_GeneralBatchedGemmSrdInitiationD_End // End of handling General Batched GEMM SRD initialization
label_ArgTypeCheckD:  /// Check if ArgType is for General Batched GEMM for D
s_and_b32 s81, s[sgprArgType], 0xff                // mask ArgType domain (bit 8 = TDM wave-parity)
s_cmp_eq_u32 s81, 3                                // ArgType == 3 for General Batched GEMM
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

s_and_b32 s81, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s81, 1                                // GSU == 1 ?
s_cbranch_scc1 label_ArgTypeCheckC                 // Handling General Batched GEMM SRD initialization
s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], s[sgprAddressC+0:sgprAddressC+0+1] // init SRD base address
s_branch label_GeneralBatchedGemmSrdInitiationC_End // End of handling General Batched GEMM SRD initialization
label_ArgTypeCheckC:  /// Check if ArgType is for General Batched GEMM for C
s_and_b32 s81, s[sgprArgType], 0xff                // mask ArgType domain (bit 8 = TDM wave-parity)
s_cmp_eq_u32 s81, 3                                // ArgType == 3 for General Batched GEMM
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


s_mul_i32 s84, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_and_b32 s83, s[sgprGSU], 0xfff                   // Restore GSU
s_mul_hi_u32 s83, s84, s[sgprStrideC1J]            // ScaleC s84 by Stride
s_mul_i32 s82, s84, s[sgprStrideC1J]               // ScaleC s84 by Stride
s_lshl_b64 s[82:83], s[82:83], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s82        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s83       // add hi to SRD
s_and_b32 s83, s[sgprGSU], 0xfff                   // Restore GSU
s_mul_hi_u32 s83, s84, s[sgprStrideD1J]            // ScaleD s84 by Stride
s_mul_i32 s82, s84, s[sgprStrideD1J]               // ScaleD s84 by Stride
s_lshl_b64 s[82:83], s[82:83], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s82        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s83       // add hi to SRD

s_and_b32 s83, s[sgprGSU], 0xfff                   // Restore GSU
s_cmp_eq_u32 s83, 1                                // GSU == 1 ?
s_cbranch_scc0 label_StridedBatchedGemmLoadC
s_and_b32 s81, s[sgprArgType], 0xff                // mask ArgType domain (bit 8 = TDM wave-parity)
s_cmp_eq_u32 s81, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_GeneralBatchedGemmLoadC
label_StridedBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s83, s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride
s_mul_i32 s82, s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride
s_lshl_b64 s[82:83], s[82:83], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s82        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s83       // add hi to SRD
s_branch label_GeneralBatchedGemmLoadC_End
label_GeneralBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for General Batched GEMM
s_mul_i32 s82, 8, s[sgprWorkGroup2]                // Compute stride in bytes into Pointer Array
s_add_u32 s82, s82, s[sgprAddressC+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s83, s[sgprAddressC+1], 0               // Offsetting to the location [Higher half of address]
s_load_b64 s[82:83], s[82:83], 0                   // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for the Matrix Address Load from the Pointer Array
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s82        // Offsetting within the Batch Matrix [Lower half of address]
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s83       // Offsetting within the Batch Matrix [Higher half of address]
s_load_b64 s[82:83], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x78 // Load batchOffsetC from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s82        // Add matrix address to SRD (low)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s83       // Add matrix address to SRD (high)
label_GeneralBatchedGemmLoadC_End:  /// End of label GeneralBatchedGemmLoadC
s_and_b32 s83, s[sgprGSU], 0xfff                   // Restore GSU
s_cmp_eq_u32 s83, 1                                // GSU == 1 ?
s_cbranch_scc0 label_StridedBatchedGemmLoadD
s_and_b32 s81, s[sgprArgType], 0xff                // mask ArgType domain (bit 8 = TDM wave-parity)
s_cmp_eq_u32 s81, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_GeneralBatchedGemmLoadD
label_StridedBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s83, s[sgprWorkGroup2], s[sgprStrideDK] // ScaleD s[sgprWorkGroup2] by Stride
s_mul_i32 s82, s[sgprWorkGroup2], s[sgprStrideDK]  // ScaleD s[sgprWorkGroup2] by Stride
s_lshl_b64 s[82:83], s[82:83], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s82        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s83       // add hi to SRD
s_branch label_GeneralBatchedGemmLoadD_End
label_GeneralBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for General Batched GEMM
s_mul_i32 s82, 8, s[sgprWorkGroup2]                // Compute stride in bytes into Pointer Array
s_add_u32 s82, s82, s[sgprAddressD+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s83, s[sgprAddressD+1], 0               // Offsetting to the location [Higher half of address]
s_load_b64 s[82:83], s[82:83], 0                   // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for the Matrix Address Load from the Pointer Array
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s82        // Offsetting within the Batch Matrix [Lower half of address]
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s83       // Offsetting within the Batch Matrix [Higher half of address]
s_load_b64 s[82:83], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x70 // Load batchOffsetD from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s82        // Add matrix address to SRD (low)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s83       // Add matrix address to SRD (high)
label_GeneralBatchedGemmLoadD_End:  /// End of label GeneralBatchedGemmLoadD

s_and_b32 s81, s[sgprGSU], 0xfff                   // Restore GSU
s_cmp_eq_u32 s81, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU_2                         // branch if GSU == 1
// GSU Output Buffer offset: Free0 + (Free1-1)*StrideC1J + (Free2-1)*StrideCK * GSUIdx * bpe%s
s_mul_hi_u32 s83, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_mul_i32 s82, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_sub_u32 s81, s[sgprSizesFree+1], 1               // Free1
s_mul_i32 s81, s81, s[sgprGSUSumIdx]               // Free1
s_mul_hi_u32 s84, s81, s[sgprStrideC1J]            // Free1
s_mul_i32 s81, s81, s[sgprStrideC1J]               // Free1
s_add_u32 s82, s82, s81                            // Free1
s_addc_u32 s83, s83, s84                           // Free1
s_sub_u32 s81, s[sgprSizesFree+2], 1               // Free2
s_mul_i32 s81, s81, s[sgprGSUSumIdx]               // Free2
s_mul_hi_u32 s84, s81, s[sgprStrideCK]             // Free2
s_mul_i32 s81, s81, s[sgprStrideCK]                // Free2
s_add_u32 s82, s82, s81                            // Free2
s_addc_u32 s83, s83, s84                           // Free2
s_lshl_b64 s[82:83], s[82:83], 2                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s82        // add lo GSU offset to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s83       // add hi GSU offset to SRD
label_GSU_2:
.set sgprGSULog2BpeC, UNDEF
.set sgprAddressC, UNDEF

/* initC: remove ValuC vgpr buffer [0...32) from pool */

/* initC: remove acc vgpr buffer [0...0) from pool */

/* initC: remove ValuA/B vgpr buffer [60...189) from pool */
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
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?

/* after InitC, skip to end of prefetch last iter if numIter==0 */

/* label_PrefetchGlobalLastIterEnd */
s_cbranch_scc0 label_NoBranch_0                    // Only branch on scc1
s_getpc_b64 s[82:83]                               // addr of next instr
s_add_i32 s84, label_PrefetchGlobalLastIterEnd, 4  // target branch offset
s_add_u32 s82, s82, s84                            // add target branch offset
s_addc_u32 s83, s83, 0                             // add high and carry
s_setpc_b64 s[82:83]                               // branch to label_PrefetchGlobalLastIterEnd
label_NoBranch_0:
s_waitcnt vmcnt(0)                                 // wait for global read

/* local write a */
v_mov_b32 v229, 0x7fff                             // w4a16: round-to-nearest-even bias for bf16
v_mov_b32 v230, 0x7fff0000                         // w4a16: bf16 Nan pattern
v_lshlrev_b32 v226, 16, v[vgprG2LScaleA+0]         // scaleA: bf16 -> f32
v_and_b32 v227, 1, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v227, 2, v227                        // scaleZeroA: -> nibble shift 0 or 4
v_bfe_i32 v223, v[vgprG2LScaleZeroA+0], v227, 0x4  // scaleZeroA: extract the selected nibble (sign-extended)
v_cvt_f32_i32 v223, v223                           // scaleZeroA: int4 -> f32
v_mul_f32 v227, v226, v223                         // scaleZeroA: z*s
v_xor_b32 v227, 0x80000000, v227                   // w4a16: negate -> -z*s
v_mov_b32 v225, v[vgprG2LA+2+0]                    // w4a16: save packed int4 dword 0
v_bfe_i32 v223, v225, 0x0, 0x4                     // w4a16: sign-extend int4 #0
v_bfe_i32 v224, v225, 0x4, 0x4                     // w4a16: sign-extend int4 #1
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+0+0], v223, v224         // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x8, 0x4                     // w4a16: sign-extend int4 #2
v_bfe_i32 v224, v225, 0xc, 0x4                     // w4a16: sign-extend int4 #3
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+0+1], v223, v224         // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x10, 0x4                    // w4a16: sign-extend int4 #4
v_bfe_i32 v224, v225, 0x14, 0x4                    // w4a16: sign-extend int4 #5
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+0+2], v223, v224         // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x18, 0x4                    // w4a16: sign-extend int4 #6
v_bfe_i32 v224, v225, 0x1c, 0x4                    // w4a16: sign-extend int4 #7
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+0+3], v223, v224         // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
v_mov_b32 v229, 0x7fff                             // w4a16: round-to-nearest-even bias for bf16
v_mov_b32 v230, 0x7fff0000                         // w4a16: bf16 Nan pattern
v_lshlrev_b32 v226, 16, v[vgprG2LScaleA+1]         // scaleA: bf16 -> f32
v_and_b32 v227, 1, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v227, 2, v227                        // scaleZeroA: -> nibble shift 0 or 4
v_bfe_i32 v223, v[vgprG2LScaleZeroA+1], v227, 0x4  // scaleZeroA: extract the selected nibble (sign-extended)
v_cvt_f32_i32 v223, v223                           // scaleZeroA: int4 -> f32
v_mul_f32 v227, v226, v223                         // scaleZeroA: z*s
v_xor_b32 v227, 0x80000000, v227                   // w4a16: negate -> -z*s
v_mov_b32 v225, v[vgprG2LA+6+0]                    // w4a16: save packed int4 dword 0
v_bfe_i32 v223, v225, 0x0, 0x4                     // w4a16: sign-extend int4 #0
v_bfe_i32 v224, v225, 0x4, 0x4                     // w4a16: sign-extend int4 #1
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+4+0], v223, v224         // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x8, 0x4                     // w4a16: sign-extend int4 #2
v_bfe_i32 v224, v225, 0xc, 0x4                     // w4a16: sign-extend int4 #3
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+4+1], v223, v224         // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x10, 0x4                    // w4a16: sign-extend int4 #4
v_bfe_i32 v224, v225, 0x14, 0x4                    // w4a16: sign-extend int4 #5
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+4+2], v223, v224         // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x18, 0x4                    // w4a16: sign-extend int4 #6
v_bfe_i32 v224, v225, 0x1c, 0x4                    // w4a16: sign-extend int4 #7
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+4+3], v223, v224         // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:2304 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 2304 sync LDS0
v_mov_b32 v229, 0x7fff                             // w4a16: round-to-nearest-even bias for bf16
v_mov_b32 v230, 0x7fff0000                         // w4a16: bf16 Nan pattern
v_lshlrev_b32 v226, 16, v[vgprG2LScaleA+2]         // scaleA: bf16 -> f32
v_and_b32 v227, 1, v[vgprGlobalReadOffsetScaleZeroA+2] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v227, 2, v227                        // scaleZeroA: -> nibble shift 0 or 4
v_bfe_i32 v223, v[vgprG2LScaleZeroA+2], v227, 0x4  // scaleZeroA: extract the selected nibble (sign-extended)
v_cvt_f32_i32 v223, v223                           // scaleZeroA: int4 -> f32
v_mul_f32 v227, v226, v223                         // scaleZeroA: z*s
v_xor_b32 v227, 0x80000000, v227                   // w4a16: negate -> -z*s
v_mov_b32 v225, v[vgprG2LA+10+0]                   // w4a16: save packed int4 dword 0
v_bfe_i32 v223, v225, 0x0, 0x4                     // w4a16: sign-extend int4 #0
v_bfe_i32 v224, v225, 0x4, 0x4                     // w4a16: sign-extend int4 #1
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+8+0], v223, v224         // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x8, 0x4                     // w4a16: sign-extend int4 #2
v_bfe_i32 v224, v225, 0xc, 0x4                     // w4a16: sign-extend int4 #3
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+8+1], v223, v224         // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x10, 0x4                    // w4a16: sign-extend int4 #4
v_bfe_i32 v224, v225, 0x14, 0x4                    // w4a16: sign-extend int4 #5
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+8+2], v223, v224         // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x18, 0x4                    // w4a16: sign-extend int4 #6
v_bfe_i32 v224, v225, 0x1c, 0x4                    // w4a16: sign-extend int4 #7
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+8+3], v223, v224         // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:4608 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 4608 sync LDS0
v_mov_b32 v229, 0x7fff                             // w4a16: round-to-nearest-even bias for bf16
v_mov_b32 v230, 0x7fff0000                         // w4a16: bf16 Nan pattern
v_lshlrev_b32 v226, 16, v[vgprG2LScaleA+3]         // scaleA: bf16 -> f32
v_and_b32 v227, 1, v[vgprGlobalReadOffsetScaleZeroA+3] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v227, 2, v227                        // scaleZeroA: -> nibble shift 0 or 4
v_bfe_i32 v223, v[vgprG2LScaleZeroA+3], v227, 0x4  // scaleZeroA: extract the selected nibble (sign-extended)
v_cvt_f32_i32 v223, v223                           // scaleZeroA: int4 -> f32
v_mul_f32 v227, v226, v223                         // scaleZeroA: z*s
v_xor_b32 v227, 0x80000000, v227                   // w4a16: negate -> -z*s
v_mov_b32 v225, v[vgprG2LA+14+0]                   // w4a16: save packed int4 dword 0
v_bfe_i32 v223, v225, 0x0, 0x4                     // w4a16: sign-extend int4 #0
v_bfe_i32 v224, v225, 0x4, 0x4                     // w4a16: sign-extend int4 #1
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+12+0], v223, v224        // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x8, 0x4                     // w4a16: sign-extend int4 #2
v_bfe_i32 v224, v225, 0xc, 0x4                     // w4a16: sign-extend int4 #3
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+12+1], v223, v224        // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x10, 0x4                    // w4a16: sign-extend int4 #4
v_bfe_i32 v224, v225, 0x14, 0x4                    // w4a16: sign-extend int4 #5
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+12+2], v223, v224        // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x18, 0x4                    // w4a16: sign-extend int4 #6
v_bfe_i32 v224, v225, 0x1c, 0x4                    // w4a16: sign-extend int4 #7
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+12+3], v223, v224        // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+12:vgprG2LA+12+3] offset:6912 // lwoA_0_0_3_0 = (0*LSCA)*(MT0I+PAD) + (3*LSPA) = 6912 sync LDS0

/* local write b */
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4:vgprG2LB+4+3] offset:2304 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 2304 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8:vgprG2LB+8+3] offset:4608 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 4608 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12:vgprG2LB+12+3] offset:6912 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 6912 sync LDS0

/* local write swap a */

/* (EPS=1) local write swap internal offset -> 32768 */

/* local write swap b */

/* (EPS=1) local write swap internal offset -> 32768 */

/******************************************/
/* Unrolled Loop(s) - Begin               */
/******************************************/
label_openLoopL:
s_cmp_le_u32 s[sgprLoopCounterL], 0x1              // LoopCounterL < EndCounter
s_cbranch_scc1 label_LoopEndL                      // do not enter LoopL
.align 16
label_LoopBeginL:

/******************************************/
/* Unrolled Loop 1/2 - Begin              */
/******************************************/
s_waitcnt lgkmcnt(0)                               // 1wait for local write
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier                                          // 4sync for global read, PGR->LW needs sync

/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */
buffer_load_b32 v[vgprG2LA+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b32 v[vgprG2LA+6], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b32 v[vgprG2LA+10], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b32 v[vgprG2LA+14], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_3_0

/* global read block-scale A */
buffer_load_d16_b16 v[vgprG2LScaleA+0], v[vgprGlobalReadOffsetScaleA+0], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 0
buffer_load_d16_b16 v[vgprG2LScaleA+1], v[vgprGlobalReadOffsetScaleA+1], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 1
buffer_load_d16_b16 v[vgprG2LScaleA+2], v[vgprGlobalReadOffsetScaleA+2], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 2
buffer_load_d16_b16 v[vgprG2LScaleA+3], v[vgprGlobalReadOffsetScaleA+3], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 3
v_lshrrev_b32 v223, 1, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: drop the nibble bit
buffer_load_d16_u8 v[vgprG2LScaleZeroA+0], v223, s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 0
v_lshrrev_b32 v223, 1, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: drop the nibble bit
buffer_load_d16_u8 v[vgprG2LScaleZeroA+1], v223, s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 1
v_lshrrev_b32 v223, 1, v[vgprGlobalReadOffsetScaleZeroA+2] // scaleZeroA: drop the nibble bit
buffer_load_d16_u8 v[vgprG2LScaleZeroA+2], v223, s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 2
v_lshrrev_b32 v223, 1, v[vgprGlobalReadOffsetScaleZeroA+3] // scaleZeroA: drop the nibble bit
buffer_load_d16_u8 v[vgprG2LScaleZeroA+3], v223, s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 3
buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b128 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b128 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b128 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s82, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s83, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s82        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s83       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s82 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s83 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* global read inc block-scale A (4 bytes) */
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], 0x4 // scaleA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], 0x4 // scaleA limit -= inc
s_add_u32 s[sgprSrdScaleZeroA+0], s[sgprSrdScaleZeroA+0], 0x2 // scaleZeroA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleZeroA+1], s[sgprSrdScaleZeroA+1], 0 // scaleZeroA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleZeroA+2], s[sgprSrdScaleZeroA+2], 0x2 // scaleZeroA limit -= inc

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s82, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s83, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s82        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s83       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s82 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s83 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32

/* iter 0 */

/* local read a */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:144 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read b */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:144 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read increment a */
/* N/A, lro->16 */
/* localReadDoCntA 1 localReadDoCntMXSA 0 localReadDoCntB 1 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->16 */
/* localReadDoCntA 1 localReadDoCntMXSA 0 localReadDoCntB 1 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0 for iteration == 0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=4 */

/* iter 1 */

/* local read a */
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA+0] offset:160 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA+0] offset:176 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0

/* local read b */
ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB+0] offset:160 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[vgprLocalReadAddrB+0] offset:176 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0

/* local read increment a */
/* N/A, lro->32 */
/* localReadDoCntA 2 localReadDoCntMXSA 0 localReadDoCntB 2 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->32 */
/* localReadDoCntA 2 localReadDoCntMXSA 0 localReadDoCntB 2 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=0 readsPerIterB=4 */

/* iter 2 */

/* local read a */
ds_load_b128 v[vgprValuA_X2_I0+0:vgprValuA_X2_I0+0+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X2_I0+4:vgprValuA_X2_I0+4+3], v[vgprLocalReadAddrA+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=2 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X2_I0+8:vgprValuA_X2_I0+8+3], v[vgprLocalReadAddrA+0] offset:192 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=2 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X2_I0+12:vgprValuA_X2_I0+12+3], v[vgprLocalReadAddrA+0] offset:208 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=2 iui=0 sync LDS0

/* local read b */
ds_load_b128 v[vgprValuB_X2_I0+0:vgprValuB_X2_I0+0+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X2_I0+4:vgprValuB_X2_I0+4+3], v[vgprLocalReadAddrB+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=2 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X2_I0+8:vgprValuB_X2_I0+8+3], v[vgprLocalReadAddrB+0] offset:192 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=2 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X2_I0+12:vgprValuB_X2_I0+12+3], v[vgprLocalReadAddrB+0] offset:208 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=2 iui=0 sync LDS0

/* local read increment a */
/* N/A, lro->48 */
/* localReadDoCntA 3 localReadDoCntMXSA 0 localReadDoCntB 3 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->48 */
/* localReadDoCntA 3 localReadDoCntMXSA 0 localReadDoCntB 3 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+7], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuA_X2_I0+8+0+0:vgprValuA_X2_I0+8+0+0+7], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+7], v[vgprValuB_X2_I0+8+0+0:vgprValuB_X2_I0+8+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuA_X2_I0+8+0+0:vgprValuA_X2_I0+8+0+0+7], v[vgprValuB_X2_I0+8+0+0:vgprValuB_X2_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=0 readsPerIterB=4 */

/* iter 3 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */

/* local read a */
ds_load_b128 v[vgprValuA_X3_I0+0:vgprValuA_X3_I0+0+3], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=3 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X3_I0+4:vgprValuA_X3_I0+4+3], v[vgprLocalReadAddrA+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=3 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X3_I0+8:vgprValuA_X3_I0+8+3], v[vgprLocalReadAddrA+0] offset:224 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=3 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X3_I0+12:vgprValuA_X3_I0+12+3], v[vgprLocalReadAddrA+0] offset:240 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=3 iui=0 sync LDS0

/* local read b */
ds_load_b128 v[vgprValuB_X3_I0+0:vgprValuB_X3_I0+0+3], v[vgprLocalReadAddrB+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=3 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X3_I0+4:vgprValuB_X3_I0+4+3], v[vgprLocalReadAddrB+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=3 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X3_I0+8:vgprValuB_X3_I0+8+3], v[vgprLocalReadAddrB+0] offset:224 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=3 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X3_I0+12:vgprValuB_X3_I0+12+3], v[vgprLocalReadAddrB+0] offset:240 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=3 iui=0 sync LDS0
s_waitcnt vmcnt(0)                                 // 1wait for global read

/* local write A */
v_mov_b32 v229, 0x7fff                             // w4a16: round-to-nearest-even bias for bf16
v_mov_b32 v230, 0x7fff0000                         // w4a16: bf16 Nan pattern
v_lshlrev_b32 v226, 16, v[vgprG2LScaleA+0]         // scaleA: bf16 -> f32
v_and_b32 v227, 1, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v227, 2, v227                        // scaleZeroA: -> nibble shift 0 or 4
v_bfe_i32 v223, v[vgprG2LScaleZeroA+0], v227, 0x4  // scaleZeroA: extract the selected nibble (sign-extended)
v_cvt_f32_i32 v223, v223                           // scaleZeroA: int4 -> f32
v_mul_f32 v227, v226, v223                         // scaleZeroA: z*s
v_xor_b32 v227, 0x80000000, v227                   // w4a16: negate -> -z*s
v_mov_b32 v225, v[vgprG2LA+2+0]                    // w4a16: save packed int4 dword 0
v_bfe_i32 v223, v225, 0x0, 0x4                     // w4a16: sign-extend int4 #0
v_bfe_i32 v224, v225, 0x4, 0x4                     // w4a16: sign-extend int4 #1
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+0+0], v223, v224         // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x8, 0x4                     // w4a16: sign-extend int4 #2
v_bfe_i32 v224, v225, 0xc, 0x4                     // w4a16: sign-extend int4 #3
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+0+1], v223, v224         // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x10, 0x4                    // w4a16: sign-extend int4 #4
v_bfe_i32 v224, v225, 0x14, 0x4                    // w4a16: sign-extend int4 #5
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+0+2], v223, v224         // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x18, 0x4                    // w4a16: sign-extend int4 #6
v_bfe_i32 v224, v225, 0x1c, 0x4                    // w4a16: sign-extend int4 #7
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+0+3], v223, v224         // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:32768 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 32768 sync LDS1
v_mov_b32 v229, 0x7fff                             // w4a16: round-to-nearest-even bias for bf16
v_mov_b32 v230, 0x7fff0000                         // w4a16: bf16 Nan pattern
v_lshlrev_b32 v226, 16, v[vgprG2LScaleA+1]         // scaleA: bf16 -> f32
v_and_b32 v227, 1, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v227, 2, v227                        // scaleZeroA: -> nibble shift 0 or 4
v_bfe_i32 v223, v[vgprG2LScaleZeroA+1], v227, 0x4  // scaleZeroA: extract the selected nibble (sign-extended)
v_cvt_f32_i32 v223, v223                           // scaleZeroA: int4 -> f32
v_mul_f32 v227, v226, v223                         // scaleZeroA: z*s
v_xor_b32 v227, 0x80000000, v227                   // w4a16: negate -> -z*s
v_mov_b32 v225, v[vgprG2LA+6+0]                    // w4a16: save packed int4 dword 0
v_bfe_i32 v223, v225, 0x0, 0x4                     // w4a16: sign-extend int4 #0
v_bfe_i32 v224, v225, 0x4, 0x4                     // w4a16: sign-extend int4 #1
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+4+0], v223, v224         // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x8, 0x4                     // w4a16: sign-extend int4 #2
v_bfe_i32 v224, v225, 0xc, 0x4                     // w4a16: sign-extend int4 #3
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+4+1], v223, v224         // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x10, 0x4                    // w4a16: sign-extend int4 #4
v_bfe_i32 v224, v225, 0x14, 0x4                    // w4a16: sign-extend int4 #5
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+4+2], v223, v224         // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x18, 0x4                    // w4a16: sign-extend int4 #6
v_bfe_i32 v224, v225, 0x1c, 0x4                    // w4a16: sign-extend int4 #7
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+4+3], v223, v224         // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:35072 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 35072 sync LDS1
v_mov_b32 v229, 0x7fff                             // w4a16: round-to-nearest-even bias for bf16
v_mov_b32 v230, 0x7fff0000                         // w4a16: bf16 Nan pattern
v_lshlrev_b32 v226, 16, v[vgprG2LScaleA+2]         // scaleA: bf16 -> f32
v_and_b32 v227, 1, v[vgprGlobalReadOffsetScaleZeroA+2] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v227, 2, v227                        // scaleZeroA: -> nibble shift 0 or 4
v_bfe_i32 v223, v[vgprG2LScaleZeroA+2], v227, 0x4  // scaleZeroA: extract the selected nibble (sign-extended)
v_cvt_f32_i32 v223, v223                           // scaleZeroA: int4 -> f32
v_mul_f32 v227, v226, v223                         // scaleZeroA: z*s
v_xor_b32 v227, 0x80000000, v227                   // w4a16: negate -> -z*s
v_mov_b32 v225, v[vgprG2LA+10+0]                   // w4a16: save packed int4 dword 0
v_bfe_i32 v223, v225, 0x0, 0x4                     // w4a16: sign-extend int4 #0
v_bfe_i32 v224, v225, 0x4, 0x4                     // w4a16: sign-extend int4 #1
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+8+0], v223, v224         // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x8, 0x4                     // w4a16: sign-extend int4 #2
v_bfe_i32 v224, v225, 0xc, 0x4                     // w4a16: sign-extend int4 #3
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+8+1], v223, v224         // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x10, 0x4                    // w4a16: sign-extend int4 #4
v_bfe_i32 v224, v225, 0x14, 0x4                    // w4a16: sign-extend int4 #5
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+8+2], v223, v224         // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x18, 0x4                    // w4a16: sign-extend int4 #6
v_bfe_i32 v224, v225, 0x1c, 0x4                    // w4a16: sign-extend int4 #7
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+8+3], v223, v224         // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:37376 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 37376 sync LDS1
v_mov_b32 v229, 0x7fff                             // w4a16: round-to-nearest-even bias for bf16
v_mov_b32 v230, 0x7fff0000                         // w4a16: bf16 Nan pattern
v_lshlrev_b32 v226, 16, v[vgprG2LScaleA+3]         // scaleA: bf16 -> f32
v_and_b32 v227, 1, v[vgprGlobalReadOffsetScaleZeroA+3] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v227, 2, v227                        // scaleZeroA: -> nibble shift 0 or 4
v_bfe_i32 v223, v[vgprG2LScaleZeroA+3], v227, 0x4  // scaleZeroA: extract the selected nibble (sign-extended)
v_cvt_f32_i32 v223, v223                           // scaleZeroA: int4 -> f32
v_mul_f32 v227, v226, v223                         // scaleZeroA: z*s
v_xor_b32 v227, 0x80000000, v227                   // w4a16: negate -> -z*s
v_mov_b32 v225, v[vgprG2LA+14+0]                   // w4a16: save packed int4 dword 0
v_bfe_i32 v223, v225, 0x0, 0x4                     // w4a16: sign-extend int4 #0
v_bfe_i32 v224, v225, 0x4, 0x4                     // w4a16: sign-extend int4 #1
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+12+0], v223, v224        // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x8, 0x4                     // w4a16: sign-extend int4 #2
v_bfe_i32 v224, v225, 0xc, 0x4                     // w4a16: sign-extend int4 #3
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+12+1], v223, v224        // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x10, 0x4                    // w4a16: sign-extend int4 #4
v_bfe_i32 v224, v225, 0x14, 0x4                    // w4a16: sign-extend int4 #5
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+12+2], v223, v224        // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x18, 0x4                    // w4a16: sign-extend int4 #6
v_bfe_i32 v224, v225, 0x1c, 0x4                    // w4a16: sign-extend int4 #7
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+12+3], v223, v224        // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+12:vgprG2LA+12+3] offset:39680 // lwoA_0_0_3_0 = (0*LSCA)*(MT0I+PAD) + (3*LSPA) = 39680 sync LDS1

/* local write MXSA */

/* local write MXSB */

/* local write B */
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+3] offset:32768 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 32768 sync LDS1
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4:vgprG2LB+4+3] offset:35072 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 35072 sync LDS1
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8:vgprG2LB+8+3] offset:37376 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 37376 sync LDS1
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12:vgprG2LB+12+3] offset:39680 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 39680 sync LDS1

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
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=8 newLR=0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+7], v[vgprValuB_X3_I0+0+0+0:vgprValuB_X3_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuA_X3_I0+8+0+0:vgprValuA_X3_I0+8+0+0+7], v[vgprValuB_X3_I0+0+0+0:vgprValuB_X3_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+7], v[vgprValuB_X3_I0+8+0+0:vgprValuB_X3_I0+8+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuA_X3_I0+8+0+0:vgprValuA_X3_I0+8+0+0+7], v[vgprValuB_X3_I0+8+0+0:vgprValuB_X3_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/* numPrefetchIter=0 */
/* dataAtIterA=3 numReadsIterA=4 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=3 numReadsIterB=4 skipReadsIterB=0 readsPerIterB=4 */

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
s_waitcnt lgkmcnt(0)                               // 1wait for local write
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier                                          // 4sync for global read, PGR->LW needs sync

/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */
buffer_load_b32 v[vgprG2LA+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b32 v[vgprG2LA+6], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b32 v[vgprG2LA+10], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b32 v[vgprG2LA+14], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_3_0

/* global read block-scale A */
buffer_load_d16_b16 v[vgprG2LScaleA+0], v[vgprGlobalReadOffsetScaleA+0], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 0
buffer_load_d16_b16 v[vgprG2LScaleA+1], v[vgprGlobalReadOffsetScaleA+1], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 1
buffer_load_d16_b16 v[vgprG2LScaleA+2], v[vgprGlobalReadOffsetScaleA+2], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 2
buffer_load_d16_b16 v[vgprG2LScaleA+3], v[vgprGlobalReadOffsetScaleA+3], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 3
v_lshrrev_b32 v223, 1, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: drop the nibble bit
buffer_load_d16_u8 v[vgprG2LScaleZeroA+0], v223, s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 0
v_lshrrev_b32 v223, 1, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: drop the nibble bit
buffer_load_d16_u8 v[vgprG2LScaleZeroA+1], v223, s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 1
v_lshrrev_b32 v223, 1, v[vgprGlobalReadOffsetScaleZeroA+2] // scaleZeroA: drop the nibble bit
buffer_load_d16_u8 v[vgprG2LScaleZeroA+2], v223, s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 2
v_lshrrev_b32 v223, 1, v[vgprGlobalReadOffsetScaleZeroA+3] // scaleZeroA: drop the nibble bit
buffer_load_d16_u8 v[vgprG2LScaleZeroA+3], v223, s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 3
buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b128 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b128 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b128 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s82, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s83, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s82        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s83       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s82 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s83 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* global read inc block-scale A (4 bytes) */
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], 0x4 // scaleA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], 0x4 // scaleA limit -= inc
s_add_u32 s[sgprSrdScaleZeroA+0], s[sgprSrdScaleZeroA+0], 0x2 // scaleZeroA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleZeroA+1], s[sgprSrdScaleZeroA+1], 0 // scaleZeroA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleZeroA+2], s[sgprSrdScaleZeroA+2], 0x2 // scaleZeroA limit -= inc

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s82, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s83, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s82        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s83       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s82 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s83 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32

/* iter 0 */

/* local read a */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:32768 // L -> Reg lro=0 swapByteOffset=32768 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:32784 // L -> Reg lro=0 swapByteOffset=32768 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:32896 // L -> Reg lro=0 swapByteOffset=32768 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:32912 // L -> Reg lro=0 swapByteOffset=32768 ti=64 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1

/* local read b */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:32768 // L -> Reg lro=0 swapByteOffset=32768 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:32784 // L -> Reg lro=0 swapByteOffset=32768 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:32896 // L -> Reg lro=0 swapByteOffset=32768 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:32912 // L -> Reg lro=0 swapByteOffset=32768 ti=64 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1

/* local read increment a */
/* N/A, lro->16 */
/* localReadDoCntA 5 localReadDoCntMXSA 0 localReadDoCntB 5 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->16 */
/* localReadDoCntA 5 localReadDoCntMXSA 0 localReadDoCntB 5 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0 for iteration == 0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=4 */

/* iter 1 */

/* local read a */
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:32800 // L -> Reg lro=16 swapByteOffset=32768 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:32816 // L -> Reg lro=16 swapByteOffset=32768 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA+0] offset:32928 // L -> Reg lro=16 swapByteOffset=32768 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA+0] offset:32944 // L -> Reg lro=16 swapByteOffset=32768 ti=64 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS1

/* local read b */
ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:32800 // L -> Reg lro=16 swapByteOffset=32768 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:32816 // L -> Reg lro=16 swapByteOffset=32768 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB+0] offset:32928 // L -> Reg lro=16 swapByteOffset=32768 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[vgprLocalReadAddrB+0] offset:32944 // L -> Reg lro=16 swapByteOffset=32768 ti=64 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS1

/* local read increment a */
/* N/A, lro->32 */
/* localReadDoCntA 6 localReadDoCntMXSA 0 localReadDoCntB 6 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->32 */
/* localReadDoCntA 6 localReadDoCntMXSA 0 localReadDoCntB 6 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=0 readsPerIterB=4 */

/* iter 2 */

/* local read a */
ds_load_b128 v[vgprValuA_X2_I0+0:vgprValuA_X2_I0+0+3], v[vgprLocalReadAddrA+0] offset:32832 // L -> Reg lro=32 swapByteOffset=32768 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X2_I0+4:vgprValuA_X2_I0+4+3], v[vgprLocalReadAddrA+0] offset:32848 // L -> Reg lro=32 swapByteOffset=32768 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=2 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X2_I0+8:vgprValuA_X2_I0+8+3], v[vgprLocalReadAddrA+0] offset:32960 // L -> Reg lro=32 swapByteOffset=32768 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=2 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X2_I0+12:vgprValuA_X2_I0+12+3], v[vgprLocalReadAddrA+0] offset:32976 // L -> Reg lro=32 swapByteOffset=32768 ti=64 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=2 iui=0 sync LDS1

/* local read b */
ds_load_b128 v[vgprValuB_X2_I0+0:vgprValuB_X2_I0+0+3], v[vgprLocalReadAddrB+0] offset:32832 // L -> Reg lro=32 swapByteOffset=32768 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X2_I0+4:vgprValuB_X2_I0+4+3], v[vgprLocalReadAddrB+0] offset:32848 // L -> Reg lro=32 swapByteOffset=32768 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=2 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X2_I0+8:vgprValuB_X2_I0+8+3], v[vgprLocalReadAddrB+0] offset:32960 // L -> Reg lro=32 swapByteOffset=32768 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=2 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X2_I0+12:vgprValuB_X2_I0+12+3], v[vgprLocalReadAddrB+0] offset:32976 // L -> Reg lro=32 swapByteOffset=32768 ti=64 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=2 iui=0 sync LDS1

/* local read increment a */
/* N/A, lro->48 */
/* localReadDoCntA 7 localReadDoCntMXSA 0 localReadDoCntB 7 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->48 */
/* localReadDoCntA 7 localReadDoCntMXSA 0 localReadDoCntB 7 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+7], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuA_X2_I0+8+0+0:vgprValuA_X2_I0+8+0+0+7], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+7], v[vgprValuB_X2_I0+8+0+0:vgprValuB_X2_I0+8+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuA_X2_I0+8+0+0:vgprValuA_X2_I0+8+0+0+7], v[vgprValuB_X2_I0+8+0+0:vgprValuB_X2_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=0 readsPerIterB=4 */

/* iter 3 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */

/* local read a */
ds_load_b128 v[vgprValuA_X3_I0+0:vgprValuA_X3_I0+0+3], v[vgprLocalReadAddrA+0] offset:32864 // L -> Reg lro=48 swapByteOffset=32768 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=3 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X3_I0+4:vgprValuA_X3_I0+4+3], v[vgprLocalReadAddrA+0] offset:32880 // L -> Reg lro=48 swapByteOffset=32768 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=3 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X3_I0+8:vgprValuA_X3_I0+8+3], v[vgprLocalReadAddrA+0] offset:32992 // L -> Reg lro=48 swapByteOffset=32768 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=3 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X3_I0+12:vgprValuA_X3_I0+12+3], v[vgprLocalReadAddrA+0] offset:33008 // L -> Reg lro=48 swapByteOffset=32768 ti=64 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=3 iui=0 sync LDS1

/* local read b */
ds_load_b128 v[vgprValuB_X3_I0+0:vgprValuB_X3_I0+0+3], v[vgprLocalReadAddrB+0] offset:32864 // L -> Reg lro=48 swapByteOffset=32768 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=3 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X3_I0+4:vgprValuB_X3_I0+4+3], v[vgprLocalReadAddrB+0] offset:32880 // L -> Reg lro=48 swapByteOffset=32768 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=3 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X3_I0+8:vgprValuB_X3_I0+8+3], v[vgprLocalReadAddrB+0] offset:32992 // L -> Reg lro=48 swapByteOffset=32768 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=3 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X3_I0+12:vgprValuB_X3_I0+12+3], v[vgprLocalReadAddrB+0] offset:33008 // L -> Reg lro=48 swapByteOffset=32768 ti=64 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=3 iui=0 sync LDS1
s_waitcnt vmcnt(0)                                 // 1wait for global read

/* local write A */
v_mov_b32 v229, 0x7fff                             // w4a16: round-to-nearest-even bias for bf16
v_mov_b32 v230, 0x7fff0000                         // w4a16: bf16 Nan pattern
v_lshlrev_b32 v226, 16, v[vgprG2LScaleA+0]         // scaleA: bf16 -> f32
v_and_b32 v227, 1, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v227, 2, v227                        // scaleZeroA: -> nibble shift 0 or 4
v_bfe_i32 v223, v[vgprG2LScaleZeroA+0], v227, 0x4  // scaleZeroA: extract the selected nibble (sign-extended)
v_cvt_f32_i32 v223, v223                           // scaleZeroA: int4 -> f32
v_mul_f32 v227, v226, v223                         // scaleZeroA: z*s
v_xor_b32 v227, 0x80000000, v227                   // w4a16: negate -> -z*s
v_mov_b32 v225, v[vgprG2LA+2+0]                    // w4a16: save packed int4 dword 0
v_bfe_i32 v223, v225, 0x0, 0x4                     // w4a16: sign-extend int4 #0
v_bfe_i32 v224, v225, 0x4, 0x4                     // w4a16: sign-extend int4 #1
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+0+0], v223, v224         // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x8, 0x4                     // w4a16: sign-extend int4 #2
v_bfe_i32 v224, v225, 0xc, 0x4                     // w4a16: sign-extend int4 #3
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+0+1], v223, v224         // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x10, 0x4                    // w4a16: sign-extend int4 #4
v_bfe_i32 v224, v225, 0x14, 0x4                    // w4a16: sign-extend int4 #5
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+0+2], v223, v224         // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x18, 0x4                    // w4a16: sign-extend int4 #6
v_bfe_i32 v224, v225, 0x1c, 0x4                    // w4a16: sign-extend int4 #7
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+0+3], v223, v224         // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
v_mov_b32 v229, 0x7fff                             // w4a16: round-to-nearest-even bias for bf16
v_mov_b32 v230, 0x7fff0000                         // w4a16: bf16 Nan pattern
v_lshlrev_b32 v226, 16, v[vgprG2LScaleA+1]         // scaleA: bf16 -> f32
v_and_b32 v227, 1, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v227, 2, v227                        // scaleZeroA: -> nibble shift 0 or 4
v_bfe_i32 v223, v[vgprG2LScaleZeroA+1], v227, 0x4  // scaleZeroA: extract the selected nibble (sign-extended)
v_cvt_f32_i32 v223, v223                           // scaleZeroA: int4 -> f32
v_mul_f32 v227, v226, v223                         // scaleZeroA: z*s
v_xor_b32 v227, 0x80000000, v227                   // w4a16: negate -> -z*s
v_mov_b32 v225, v[vgprG2LA+6+0]                    // w4a16: save packed int4 dword 0
v_bfe_i32 v223, v225, 0x0, 0x4                     // w4a16: sign-extend int4 #0
v_bfe_i32 v224, v225, 0x4, 0x4                     // w4a16: sign-extend int4 #1
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+4+0], v223, v224         // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x8, 0x4                     // w4a16: sign-extend int4 #2
v_bfe_i32 v224, v225, 0xc, 0x4                     // w4a16: sign-extend int4 #3
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+4+1], v223, v224         // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x10, 0x4                    // w4a16: sign-extend int4 #4
v_bfe_i32 v224, v225, 0x14, 0x4                    // w4a16: sign-extend int4 #5
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+4+2], v223, v224         // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x18, 0x4                    // w4a16: sign-extend int4 #6
v_bfe_i32 v224, v225, 0x1c, 0x4                    // w4a16: sign-extend int4 #7
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+4+3], v223, v224         // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:2304 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 2304 sync LDS0
v_mov_b32 v229, 0x7fff                             // w4a16: round-to-nearest-even bias for bf16
v_mov_b32 v230, 0x7fff0000                         // w4a16: bf16 Nan pattern
v_lshlrev_b32 v226, 16, v[vgprG2LScaleA+2]         // scaleA: bf16 -> f32
v_and_b32 v227, 1, v[vgprGlobalReadOffsetScaleZeroA+2] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v227, 2, v227                        // scaleZeroA: -> nibble shift 0 or 4
v_bfe_i32 v223, v[vgprG2LScaleZeroA+2], v227, 0x4  // scaleZeroA: extract the selected nibble (sign-extended)
v_cvt_f32_i32 v223, v223                           // scaleZeroA: int4 -> f32
v_mul_f32 v227, v226, v223                         // scaleZeroA: z*s
v_xor_b32 v227, 0x80000000, v227                   // w4a16: negate -> -z*s
v_mov_b32 v225, v[vgprG2LA+10+0]                   // w4a16: save packed int4 dword 0
v_bfe_i32 v223, v225, 0x0, 0x4                     // w4a16: sign-extend int4 #0
v_bfe_i32 v224, v225, 0x4, 0x4                     // w4a16: sign-extend int4 #1
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+8+0], v223, v224         // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x8, 0x4                     // w4a16: sign-extend int4 #2
v_bfe_i32 v224, v225, 0xc, 0x4                     // w4a16: sign-extend int4 #3
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+8+1], v223, v224         // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x10, 0x4                    // w4a16: sign-extend int4 #4
v_bfe_i32 v224, v225, 0x14, 0x4                    // w4a16: sign-extend int4 #5
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+8+2], v223, v224         // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x18, 0x4                    // w4a16: sign-extend int4 #6
v_bfe_i32 v224, v225, 0x1c, 0x4                    // w4a16: sign-extend int4 #7
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+8+3], v223, v224         // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:4608 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 4608 sync LDS0
v_mov_b32 v229, 0x7fff                             // w4a16: round-to-nearest-even bias for bf16
v_mov_b32 v230, 0x7fff0000                         // w4a16: bf16 Nan pattern
v_lshlrev_b32 v226, 16, v[vgprG2LScaleA+3]         // scaleA: bf16 -> f32
v_and_b32 v227, 1, v[vgprGlobalReadOffsetScaleZeroA+3] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v227, 2, v227                        // scaleZeroA: -> nibble shift 0 or 4
v_bfe_i32 v223, v[vgprG2LScaleZeroA+3], v227, 0x4  // scaleZeroA: extract the selected nibble (sign-extended)
v_cvt_f32_i32 v223, v223                           // scaleZeroA: int4 -> f32
v_mul_f32 v227, v226, v223                         // scaleZeroA: z*s
v_xor_b32 v227, 0x80000000, v227                   // w4a16: negate -> -z*s
v_mov_b32 v225, v[vgprG2LA+14+0]                   // w4a16: save packed int4 dword 0
v_bfe_i32 v223, v225, 0x0, 0x4                     // w4a16: sign-extend int4 #0
v_bfe_i32 v224, v225, 0x4, 0x4                     // w4a16: sign-extend int4 #1
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+12+0], v223, v224        // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x8, 0x4                     // w4a16: sign-extend int4 #2
v_bfe_i32 v224, v225, 0xc, 0x4                     // w4a16: sign-extend int4 #3
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+12+1], v223, v224        // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x10, 0x4                    // w4a16: sign-extend int4 #4
v_bfe_i32 v224, v225, 0x14, 0x4                    // w4a16: sign-extend int4 #5
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+12+2], v223, v224        // w4a16: pack 2 bf16
v_bfe_i32 v223, v225, 0x18, 0x4                    // w4a16: sign-extend int4 #6
v_bfe_i32 v224, v225, 0x1c, 0x4                    // w4a16: sign-extend int4 #7
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_fma_f32 v223, v223, v226, v227                   // w4a16: q*s - z*s
v_fma_f32 v224, v224, v226, v227                   // w4a16: q*s - z*s
v_cmp_u_f32 s8, v223, v223                         // w4a16: check Nan
v_bfe_u32 v228, v223, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v223, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v223, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v223, 16, v223                       // w4a16: f32 -> bf16
v_cmp_u_f32 s8, v224, v224                         // w4a16: check Nan
v_bfe_u32 v228, v224, 16, 1                        // w4a16: lsb of the bf16 mantissa
v_add3_u32 v228, v224, v228, v229                  // w4a16: add lsb + rounding bias
v_cndmask_b32 v224, v228, v230, s8                 // w4a16: keep Nan
v_lshrrev_b32 v224, 16, v224                       // w4a16: f32 -> bf16
v_pack_b32_f16 v[vgprG2LA+12+3], v223, v224        // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+12:vgprG2LA+12+3] offset:6912 // lwoA_0_0_3_0 = (0*LSCA)*(MT0I+PAD) + (3*LSPA) = 6912 sync LDS0

/* local write MXSA */

/* local write MXSB */

/* local write B */
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4:vgprG2LB+4+3] offset:2304 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 2304 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8:vgprG2LB+8+3] offset:4608 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 4608 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12:vgprG2LB+12+3] offset:6912 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 6912 sync LDS0

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
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=8 newLR=0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+7], v[vgprValuB_X3_I0+0+0+0:vgprValuB_X3_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuA_X3_I0+8+0+0:vgprValuA_X3_I0+8+0+0+7], v[vgprValuB_X3_I0+0+0+0:vgprValuB_X3_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+7], v[vgprValuB_X3_I0+8+0+0:vgprValuB_X3_I0+8+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuA_X3_I0+8+0+0:vgprValuA_X3_I0+8+0+0+7], v[vgprValuB_X3_I0+8+0+0:vgprValuB_X3_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/* numPrefetchIter=0 */
/* dataAtIterA=3 numReadsIterA=4 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=3 numReadsIterB=4 skipReadsIterB=0 readsPerIterB=4 */

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

/* local read a */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:144 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read b */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:144 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read increment a */
/* N/A, lro->16 */
/* localReadDoCntA 9 localReadDoCntMXSA 0 localReadDoCntB 9 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->16 */
/* localReadDoCntA 9 localReadDoCntMXSA 0 localReadDoCntB 9 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0 for iteration == 0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=4 */

/* iter 1 (last unrolled loop) */

/* local read a */
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA+0] offset:160 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA+0] offset:176 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0

/* local read b */
ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB+0] offset:160 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[vgprLocalReadAddrB+0] offset:176 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0

/* local read increment a */
/* N/A, lro->32 */
/* localReadDoCntA 10 localReadDoCntMXSA 0 localReadDoCntB 10 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->32 */
/* localReadDoCntA 10 localReadDoCntMXSA 0 localReadDoCntB 10 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=0 readsPerIterB=4 */

/* iter 2 (last unrolled loop) */

/* local read a */
ds_load_b128 v[vgprValuA_X2_I0+0:vgprValuA_X2_I0+0+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X2_I0+4:vgprValuA_X2_I0+4+3], v[vgprLocalReadAddrA+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=2 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X2_I0+8:vgprValuA_X2_I0+8+3], v[vgprLocalReadAddrA+0] offset:192 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=2 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X2_I0+12:vgprValuA_X2_I0+12+3], v[vgprLocalReadAddrA+0] offset:208 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=2 iui=0 sync LDS0

/* local read b */
ds_load_b128 v[vgprValuB_X2_I0+0:vgprValuB_X2_I0+0+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X2_I0+4:vgprValuB_X2_I0+4+3], v[vgprLocalReadAddrB+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=2 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X2_I0+8:vgprValuB_X2_I0+8+3], v[vgprLocalReadAddrB+0] offset:192 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=2 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X2_I0+12:vgprValuB_X2_I0+12+3], v[vgprLocalReadAddrB+0] offset:208 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=2 iui=0 sync LDS0

/* local read increment a */
/* N/A, lro->48 */
/* localReadDoCntA 11 localReadDoCntMXSA 0 localReadDoCntB 11 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->48 */
/* localReadDoCntA 11 localReadDoCntMXSA 0 localReadDoCntB 11 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+7], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuA_X2_I0+8+0+0:vgprValuA_X2_I0+8+0+0+7], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+7], v[vgprValuB_X2_I0+8+0+0:vgprValuB_X2_I0+8+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuA_X2_I0+8+0+0:vgprValuA_X2_I0+8+0+0+7], v[vgprValuB_X2_I0+8+0+0:vgprValuB_X2_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=0 readsPerIterB=4 */

/* iter 3 (last unrolled loop) */

/* local read a */
ds_load_b128 v[vgprValuA_X3_I0+0:vgprValuA_X3_I0+0+3], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=3 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X3_I0+4:vgprValuA_X3_I0+4+3], v[vgprLocalReadAddrA+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=3 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X3_I0+8:vgprValuA_X3_I0+8+3], v[vgprLocalReadAddrA+0] offset:224 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=3 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X3_I0+12:vgprValuA_X3_I0+12+3], v[vgprLocalReadAddrA+0] offset:240 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=3 iui=0 sync LDS0

/* local read b */
ds_load_b128 v[vgprValuB_X3_I0+0:vgprValuB_X3_I0+0+3], v[vgprLocalReadAddrB+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=3 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X3_I0+4:vgprValuB_X3_I0+4+3], v[vgprLocalReadAddrB+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=3 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X3_I0+8:vgprValuB_X3_I0+8+3], v[vgprLocalReadAddrB+0] offset:224 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=3 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X3_I0+12:vgprValuB_X3_I0+12+3], v[vgprLocalReadAddrB+0] offset:240 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=3 iui=0 sync LDS0
s_waitcnt vmcnt(0)                                 // 1wait for global read

/* local write A */

/* local write MXSA */

/* local write MXSB */

/* local write B */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+7], v[vgprValuB_X3_I0+0+0+0:vgprValuB_X3_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuA_X3_I0+8+0+0:vgprValuA_X3_I0+8+0+0+7], v[vgprValuB_X3_I0+0+0+0:vgprValuB_X3_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+7], v[vgprValuB_X3_I0+8+0+0:vgprValuB_X3_I0+8+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuA_X3_I0+8+0+0:vgprValuA_X3_I0+8+0+0+7], v[vgprValuB_X3_I0+8+0+0:vgprValuB_X3_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/* numPrefetchIter=0 */
/* dataAtIterA=3 numReadsIterA=4 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=3 numReadsIterB=4 skipReadsIterB=0 readsPerIterB=4 */
label_toPGR1end_OrdNLL:
label_PrefetchGlobalLastIterEnd:

/* Tail: add ValuA/B vgpr buffer [60...189) to pool */

/* Tail: add address/G2L vgpr [189...222) to pool */
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
.set sgprAddressScaleZeroA, UNDEF
.set sgprShadowLimitA, UNDEF
.set sgprSrdScaleZeroA, UNDEF
.set sgprSrdA, UNDEF
.set sgprSrdB, UNDEF
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
v_lshrrev_b32 v64, 5, v[vgprSerial]                // 64 = Serial / 32
v_lshrrev_b32 v65, 1, v64                          // 65 = 64 / 2
v_lshlrev_b32 v61, 4, v65                          // wave coordination offset 1
v_and_b32 v65, 15, v[vgprSerial]                   // v65 = v[vgprSerial] % 16
v_add_lshl_u32 v61, v65, v61, 1                    // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v62, v61, s[sgprStrideC1J]            //  offset 1
v_mul_lo_u32 v63, v61, s[sgprStrideD1J]            //  offset 1
v_and_b32 v65, 1, v64                              // v65 = v64 % 2
v_lshlrev_b32 v65, 4, v65                          // wave coordination offset 0
v_and_b32 v60, 31, v[vgprSerial]                   // v60 = v[vgprSerial] % 32
v_lshrrev_b32 v60, 4, v60                          // 60 = 60 / 16
                                                   // thread0 * continuous_output (multiplier is 1, do nothing)
v_add_lshl_u32 v60, v65, v60, 1                    // coordination 0 = vwA *(wave_id0 + tid0)
s_mul_i32 s8, 64, s[sgprWorkGroup0]                // wgp0 * MT0
v_add_nc_u32 v60, s8, v60                          // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s8, 64, s[sgprWorkGroup1]                // wgp1 * MT1
v_add_nc_u32 v61, s8, v61                          // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1

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
s_cbranch_scc1 label_GW_B0_FD0_VW2_MB_Else         // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 64 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s28, 63, s[sgprSizeJ]                    // s28 = s[sgprSizeJ] % 64
s_add_u32 s29, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s29                // wg1 >= nwg1-1
s_cselect_b32 s28, s28, 0                          // set rem
s_cmpk_gt_u32 s28, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW2_MB_Then         // jump if edges required
label_GW_B0_FD0_VW2_MB_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=90 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw2); (0,1,0,0:vw2); (0,2,0,0:vw2); (0,3,0,0:vw2); (0,4,0,0:vw2); (0,5,0,0:vw2); (0,6,0,0:vw2); (0,7,0,0:vw2); (0,0,1,0:vw2); (0,1,1,0:vw2); (0,2,1,0:vw2); (0,3,1,0:vw2); (0,4,1,0:vw2); (0,5,1,0:vw2); (0,6,1,0:vw2); (0,7,1,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
/* (d1,vc1,d0,vc0)=(0,1,2,0) */
/* (d1,vc1,d0,vc0)=(0,1,3,0) */
/* (d1,vc1,d0,vc0)=(0,1,4,0) */
/* (d1,vc1,d0,vc0)=(0,1,5,0) */
/* (d1,vc1,d0,vc0)=(0,1,6,0) */
/* (d1,vc1,d0,vc0)=(0,1,7,0) */
v_add_lshl_u32 v71, v63, v60, 2                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=60, coord0Vgpr=60 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0), (0, 0, 1, 0), (0, 1, 1, 0), (0, 2, 1, 0), (0, 3, 1, 0), (0, 4, 1, 0), (0, 5, 1, 0), (0, 6, 1, 0), (0, 7, 1, 0)] */
v_mov_b32 v[vgprValuC+74], v[vgprValuC+0]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+75], v[vgprValuC+8]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+76], v[vgprValuC+1]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+77], v[vgprValuC+9]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+78], v[vgprValuC+2]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+79], v[vgprValuC+10]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+80], v[vgprValuC+3]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+81], v[vgprValuC+11]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+82], v[vgprValuC+4]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+83], v[vgprValuC+12]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+84], v[vgprValuC+5]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+85], v[vgprValuC+13]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+86], v[vgprValuC+6]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+87], v[vgprValuC+14]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+88], v[vgprValuC+7]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+89], v[vgprValuC+15]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+90], v[vgprValuC+16]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+91], v[vgprValuC+24]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+92], v[vgprValuC+17]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+93], v[vgprValuC+25]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+94], v[vgprValuC+18]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+95], v[vgprValuC+26]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+96], v[vgprValuC+19]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+97], v[vgprValuC+27]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+98], v[vgprValuC+20]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+99], v[vgprValuC+28]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+100], v[vgprValuC+21]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+101], v[vgprValuC+29]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+102], v[vgprValuC+22]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+103], v[vgprValuC+30]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+104], v[vgprValuC+23]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+105], v[vgprValuC+31]        // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b64 v[74:75], v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b64 v[76:77], v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:16 // store D
buffer_store_b64 v[78:79], v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
buffer_store_b64 v[80:81], v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:48 // store D
buffer_store_b64 v[82:83], v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
buffer_store_b64 v[84:85], v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:80 // store D
buffer_store_b64 v[86:87], v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:96 // store D
buffer_store_b64 v[88:89], v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:112 // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_b64 v[90:91], v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b64 v[92:93], v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:16 // store D
buffer_store_b64 v[94:95], v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
buffer_store_b64 v[96:97], v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:48 // store D
buffer_store_b64 v[98:99], v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
buffer_store_b64 v[100:101], v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:80 // store D
buffer_store_b64 v[102:103], v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:96 // store D
buffer_store_b64 v[104:105], v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:112 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_B0_FD0_VW2_MB_NonEdgeEnd:
label_GW_B0_FD0_VW2_MB_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=60 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw2); (0,1,0,0:vw2); (0,2,0,0:vw2); (0,3,0,0:vw2); (0,4,0,0:vw2); (0,5,0,0:vw2); (0,6,0,0:vw2); (0,7,0,0:vw2); (0,0,1,0:vw2); (0,1,1,0:vw2); (0,2,1,0:vw2); (0,3,1,0:vw2); (0,4,1,0:vw2); (0,5,1,0:vw2); (0,6,1,0:vw2); (0,7,1,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v66, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s28, v60, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v71, v63, v60, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v71, v66, v71, s30                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v64, vcc_lo, v60, 4                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v104, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v104, v66, v104, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
v_add_co_u32 v64, vcc_lo, v60, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v105, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v105, v66, v105, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
v_add_co_u32 v64, vcc_lo, v60, 12                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v106, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v106, v66, v106, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
v_add_co_u32 v64, vcc_lo, v60, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v107, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v107, v66, v107, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
v_add_co_u32 v64, vcc_lo, v60, 20                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v108, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v108, v66, v108, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
v_add_co_u32 v64, vcc_lo, v60, 24                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v109, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v109, v66, v109, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
v_add_co_u32 v64, vcc_lo, v60, 28                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v110, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v110, v66, v110, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v61, vcc_lo, v61, 1                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v62, v62, s[sgprStrideC1J]            // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v63, v63, s[sgprStrideD1J]            // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v60, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v111, v63, v60, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v111, v66, v111, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
v_add_co_u32 v64, vcc_lo, v60, 4                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v112, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v112, v66, v112, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,2,0) */
v_add_co_u32 v64, vcc_lo, v60, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v113, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v113, v66, v113, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,3,0) */
v_add_co_u32 v64, vcc_lo, v60, 12                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v114, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v114, v66, v114, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,4,0) */
v_add_co_u32 v64, vcc_lo, v60, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v115, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v115, v66, v115, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,5,0) */
v_add_co_u32 v64, vcc_lo, v60, 20                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v116, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v116, v66, v116, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,6,0) */
v_add_co_u32 v64, vcc_lo, v60, 24                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v117, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v117, v66, v117, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,7,0) */
v_add_co_u32 v64, vcc_lo, v60, 28                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v118, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v118, v66, v118, s30                 // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0), (0, 0, 1, 0), (0, 1, 1, 0), (0, 2, 1, 0), (0, 3, 1, 0), (0, 4, 1, 0), (0, 5, 1, 0), (0, 6, 1, 0), (0, 7, 1, 0)] */
v_mov_b32 v[vgprValuC+72], v[vgprValuC+0]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+73], v[vgprValuC+8]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+74], v[vgprValuC+1]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+75], v[vgprValuC+9]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+76], v[vgprValuC+2]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+77], v[vgprValuC+10]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+78], v[vgprValuC+3]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+79], v[vgprValuC+11]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+80], v[vgprValuC+4]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+81], v[vgprValuC+12]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+82], v[vgprValuC+5]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+83], v[vgprValuC+13]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+84], v[vgprValuC+6]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+85], v[vgprValuC+14]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+86], v[vgprValuC+7]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+87], v[vgprValuC+15]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+88], v[vgprValuC+16]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+89], v[vgprValuC+24]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+90], v[vgprValuC+17]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+91], v[vgprValuC+25]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+92], v[vgprValuC+18]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+93], v[vgprValuC+26]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+94], v[vgprValuC+19]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+95], v[vgprValuC+27]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+96], v[vgprValuC+20]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+97], v[vgprValuC+28]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+98], v[vgprValuC+21]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+99], v[vgprValuC+29]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+100], v[vgprValuC+22]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+101], v[vgprValuC+30]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+102], v[vgprValuC+23]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+103], v[vgprValuC+31]        // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b64 v[72:73], v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b64 v[74:75], v104, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b64 v[76:77], v105, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b64 v[78:79], v106, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b64 v[80:81], v107, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b64 v[82:83], v108, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b64 v[84:85], v109, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b64 v[86:87], v110, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b64 v[88:89], v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b64 v[90:91], v112, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b64 v[92:93], v113, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b64 v[94:95], v114, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b64 v[96:97], v115, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b64 v[98:99], v116, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b64 v[100:101], v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b64 v[102:103], v118, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_B0_FD0_VW2_MB_Else:
label_GW_B0_FD0_VW1_MB_Else:
label_GW_B0_FD0_VW1_MB_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=90 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,1,0,0:vw1); (0,1,0,1:vw1); (0,2,0,0:vw1); (0,2,0,1:vw1); (0,3,0,0:vw1); (0,3,0,1:vw1); (0,4,0,0:vw1); (0,4,0,1:vw1); (0,5,0,0:vw1); (0,5,0,1:vw1); (0,6,0,0:vw1); (0,6,0,1:vw1); (0,7,0,0:vw1); (0,7,0,1:vw1); (0,0,1,0:vw1); (0,0,1,1:vw1); (0,1,1,0:vw1); (0,1,1,1:vw1); (0,2,1,0:vw1); (0,2,1,1:vw1); (0,3,1,0:vw1); (0,3,1,1:vw1); (0,4,1,0:vw1); (0,4,1,1:vw1); (0,5,1,0:vw1); (0,5,1,1:vw1); (0,6,1,0:vw1); (0,6,1,1:vw1); (0,7,1,0:vw1); (0,7,1,1:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v66, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s28, v60, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v103, v63, v60, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v103, v66, v103, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
v_add_co_u32 v64, vcc_lo, v60, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v104, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v104, v66, v104, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v64, vcc_lo, v60, 4                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v105, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v105, v66, v105, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,1) */
v_add_co_u32 v64, vcc_lo, v60, 5                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v106, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v106, v66, v106, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
v_add_co_u32 v64, vcc_lo, v60, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v107, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v107, v66, v107, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,1) */
v_add_co_u32 v64, vcc_lo, v60, 9                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v108, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v108, v66, v108, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
v_add_co_u32 v64, vcc_lo, v60, 12                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v109, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v109, v66, v109, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,1) */
v_add_co_u32 v64, vcc_lo, v60, 13                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v110, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v110, v66, v110, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
v_add_co_u32 v64, vcc_lo, v60, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v111, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v111, v66, v111, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,1) */
v_add_co_u32 v64, vcc_lo, v60, 17                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v112, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v112, v66, v112, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
v_add_co_u32 v64, vcc_lo, v60, 20                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v113, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v113, v66, v113, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,1) */
v_add_co_u32 v64, vcc_lo, v60, 21                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v114, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v114, v66, v114, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
v_add_co_u32 v64, vcc_lo, v60, 24                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v115, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v115, v66, v115, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,1) */
v_add_co_u32 v64, vcc_lo, v60, 25                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v116, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v116, v66, v116, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
v_add_co_u32 v64, vcc_lo, v60, 28                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v117, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v117, v66, v117, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,1) */
v_add_co_u32 v64, vcc_lo, v60, 29                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v118, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v118, v66, v118, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v61, vcc_lo, v61, 1                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v62, v62, s[sgprStrideC1J]            // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v63, v63, s[sgprStrideD1J]            // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v60, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v119, v63, v60, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v119, v66, v119, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,1) */
v_add_co_u32 v64, vcc_lo, v60, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v120, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v120, v66, v120, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
v_add_co_u32 v64, vcc_lo, v60, 4                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v121, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v121, v66, v121, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,1) */
v_add_co_u32 v64, vcc_lo, v60, 5                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v122, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v122, v66, v122, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,2,0) */
v_add_co_u32 v64, vcc_lo, v60, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v123, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v123, v66, v123, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,2,1) */
v_add_co_u32 v64, vcc_lo, v60, 9                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v124, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v124, v66, v124, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,3,0) */
v_add_co_u32 v64, vcc_lo, v60, 12                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v125, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v125, v66, v125, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,3,1) */
v_add_co_u32 v64, vcc_lo, v60, 13                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v126, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v126, v66, v126, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,4,0) */
v_add_co_u32 v64, vcc_lo, v60, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v127, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v127, v66, v127, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,4,1) */
v_add_co_u32 v64, vcc_lo, v60, 17                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v128, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v128, v66, v128, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,5,0) */
v_add_co_u32 v64, vcc_lo, v60, 20                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v129, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v129, v66, v129, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,5,1) */
v_add_co_u32 v64, vcc_lo, v60, 21                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v130, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v66, v130, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,6,0) */
v_add_co_u32 v64, vcc_lo, v60, 24                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v131, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v131, v66, v131, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,6,1) */
v_add_co_u32 v64, vcc_lo, v60, 25                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v132, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v66, v132, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,7,0) */
v_add_co_u32 v64, vcc_lo, v60, 28                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v133, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v133, v66, v133, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,7,1) */
v_add_co_u32 v64, vcc_lo, v60, 29                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v134, v63, v64, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v66, v134, s30                 // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 1, 0, 0), (0, 1, 0, 1), (0, 2, 0, 0), (0, 2, 0, 1), (0, 3, 0, 0), (0, 3, 0, 1), (0, 4, 0, 0), (0, 4, 0, 1), (0, 5, 0, 0), (0, 5, 0, 1), (0, 6, 0, 0), (0, 6, 0, 1), (0, 7, 0, 0), (0, 7, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1), (0, 1, 1, 0), (0, 1, 1, 1), (0, 2, 1, 0), (0, 2, 1, 1), (0, 3, 1, 0), (0, 3, 1, 1), (0, 4, 1, 0), (0, 4, 1, 1), (0, 5, 1, 0), (0, 5, 1, 1), (0, 6, 1, 0), (0, 6, 1, 1), (0, 7, 1, 0), (0, 7, 1, 1)] */
v_mov_b32 v[vgprValuC+71], v[vgprValuC+0]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+72], v[vgprValuC+8]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+73], v[vgprValuC+1]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+74], v[vgprValuC+9]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+75], v[vgprValuC+2]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+76], v[vgprValuC+10]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+77], v[vgprValuC+3]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+78], v[vgprValuC+11]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+79], v[vgprValuC+4]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+80], v[vgprValuC+12]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+81], v[vgprValuC+5]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+82], v[vgprValuC+13]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+83], v[vgprValuC+6]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+84], v[vgprValuC+14]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+85], v[vgprValuC+7]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+86], v[vgprValuC+15]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+87], v[vgprValuC+16]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+88], v[vgprValuC+24]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+89], v[vgprValuC+17]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+90], v[vgprValuC+25]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+91], v[vgprValuC+18]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+92], v[vgprValuC+26]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+93], v[vgprValuC+19]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+94], v[vgprValuC+27]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+95], v[vgprValuC+20]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+96], v[vgprValuC+28]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+97], v[vgprValuC+21]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+98], v[vgprValuC+29]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+99], v[vgprValuC+22]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+100], v[vgprValuC+30]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+101], v[vgprValuC+23]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+102], v[vgprValuC+31]        // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b32 v71, v103, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v72, v104, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v73, v105, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v74, v106, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v75, v107, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v76, v108, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v77, v109, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v78, v110, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v79, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v80, v112, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v81, v113, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v82, v114, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v83, v115, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v84, v116, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v85, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v86, v118, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v87, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v88, v120, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v89, v121, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v90, v122, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v91, v123, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v92, v124, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v93, v125, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v94, v126, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v95, v127, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v96, v128, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v97, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v98, v130, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v99, v131, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v100, v132, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v101, v133, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v102, v134, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
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
s_cbranch_scc1 label_GW_B0_FD0_VW2_GSU1_Else       // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 64 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s28, 63, s[sgprSizeJ]                    // s28 = s[sgprSizeJ] % 64
s_add_u32 s29, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s29                // wg1 >= nwg1-1
s_cselect_b32 s28, s28, 0                          // set rem
s_cmpk_gt_u32 s28, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW2_GSU1_Then       // jump if edges required
label_GW_B0_FD0_VW2_GSU1_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=90 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw2); (0,1,0,0:vw2); (0,2,0,0:vw2); (0,3,0,0:vw2); (0,4,0,0:vw2); (0,5,0,0:vw2); (0,6,0,0:vw2); (0,7,0,0:vw2); (0,0,1,0:vw2); (0,1,1,0:vw2); (0,2,1,0:vw2); (0,3,1,0:vw2); (0,4,1,0:vw2); (0,5,1,0:vw2); (0,6,1,0:vw2); (0,7,1,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
/* (d1,vc1,d0,vc0)=(0,1,2,0) */
/* (d1,vc1,d0,vc0)=(0,1,3,0) */
/* (d1,vc1,d0,vc0)=(0,1,4,0) */
/* (d1,vc1,d0,vc0)=(0,1,5,0) */
/* (d1,vc1,d0,vc0)=(0,1,6,0) */
/* (d1,vc1,d0,vc0)=(0,1,7,0) */
v_add_lshl_u32 v71, v63, v60, 1                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=60, coord0Vgpr=60 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0), (0, 0, 1, 0), (0, 1, 1, 0), (0, 2, 1, 0), (0, 3, 1, 0), (0, 4, 1, 0), (0, 5, 1, 0), (0, 6, 1, 0), (0, 7, 1, 0)] */
v_mul_f32 v[vgprValuC+74], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+75], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+76], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+77], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+78], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+79], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+80], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+81], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+82], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+83], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+84], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+85], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+86], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+87], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+88], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+89], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+90], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+91], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+92], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+93], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+94], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+95], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+96], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+97], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+98], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+100], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+102], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+103], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+104], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+105], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v68, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v69, 0x7fff0000                          // fp32 Nan
v_mov_b32 v70, 0x7fff                              // rounding bias for bfloat16
v_cmp_u_f32 s8, v[vgprValuC+74], v[vgprValuC+74]   // check Nan
v_bfe_u32 v67, v[vgprValuC+74], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+74], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+74], v67, v69, s8
v_lshrrev_b32 v[vgprValuC+74], 16, v[vgprValuC+74] // convert C to bf16
v_cmp_u_f32 s8, v[vgprValuC+75], v[vgprValuC+75]   // check Nan
v_bfe_u32 v67, v[vgprValuC+75], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+75], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+75], v67, v69, s8
v_and_or_b32 v74, v[vgprValuC+75], v68, v[vgprValuC+74] // pack two bf16 to dword
buffer_store_b32 v74, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+76], v[vgprValuC+76]   // check Nan
v_bfe_u32 v67, v[vgprValuC+76], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+76], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+76], v67, v69, s8
v_lshrrev_b32 v[vgprValuC+76], 16, v[vgprValuC+76] // convert C to bf16
v_cmp_u_f32 s8, v[vgprValuC+77], v[vgprValuC+77]   // check Nan
v_bfe_u32 v67, v[vgprValuC+77], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+77], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+77], v67, v69, s8
v_and_or_b32 v76, v[vgprValuC+77], v68, v[vgprValuC+76] // pack two bf16 to dword
buffer_store_b32 v76, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:8 // store D
v_cmp_u_f32 s8, v[vgprValuC+78], v[vgprValuC+78]   // check Nan
v_bfe_u32 v67, v[vgprValuC+78], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+78], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+78], v67, v69, s8
v_lshrrev_b32 v[vgprValuC+78], 16, v[vgprValuC+78] // convert C to bf16
v_cmp_u_f32 s8, v[vgprValuC+79], v[vgprValuC+79]   // check Nan
v_bfe_u32 v67, v[vgprValuC+79], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+79], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+79], v67, v69, s8
v_and_or_b32 v78, v[vgprValuC+79], v68, v[vgprValuC+78] // pack two bf16 to dword
buffer_store_b32 v78, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:16 // store D
v_cmp_u_f32 s8, v[vgprValuC+80], v[vgprValuC+80]   // check Nan
v_bfe_u32 v67, v[vgprValuC+80], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+80], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+80], v67, v69, s8
v_lshrrev_b32 v[vgprValuC+80], 16, v[vgprValuC+80] // convert C to bf16
v_cmp_u_f32 s8, v[vgprValuC+81], v[vgprValuC+81]   // check Nan
v_bfe_u32 v67, v[vgprValuC+81], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+81], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+81], v67, v69, s8
v_and_or_b32 v80, v[vgprValuC+81], v68, v[vgprValuC+80] // pack two bf16 to dword
buffer_store_b32 v80, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:24 // store D
v_cmp_u_f32 s8, v[vgprValuC+82], v[vgprValuC+82]   // check Nan
v_bfe_u32 v67, v[vgprValuC+82], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+82], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+82], v67, v69, s8
v_lshrrev_b32 v[vgprValuC+82], 16, v[vgprValuC+82] // convert C to bf16
v_cmp_u_f32 s8, v[vgprValuC+83], v[vgprValuC+83]   // check Nan
v_bfe_u32 v67, v[vgprValuC+83], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+83], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+83], v67, v69, s8
v_and_or_b32 v82, v[vgprValuC+83], v68, v[vgprValuC+82] // pack two bf16 to dword
buffer_store_b32 v82, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
v_cmp_u_f32 s8, v[vgprValuC+84], v[vgprValuC+84]   // check Nan
v_bfe_u32 v67, v[vgprValuC+84], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+84], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+84], v67, v69, s8
v_lshrrev_b32 v[vgprValuC+84], 16, v[vgprValuC+84] // convert C to bf16
v_cmp_u_f32 s8, v[vgprValuC+85], v[vgprValuC+85]   // check Nan
v_bfe_u32 v67, v[vgprValuC+85], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+85], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+85], v67, v69, s8
v_and_or_b32 v84, v[vgprValuC+85], v68, v[vgprValuC+84] // pack two bf16 to dword
buffer_store_b32 v84, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:40 // store D
v_cmp_u_f32 s8, v[vgprValuC+86], v[vgprValuC+86]   // check Nan
v_bfe_u32 v67, v[vgprValuC+86], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+86], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+86], v67, v69, s8
v_lshrrev_b32 v[vgprValuC+86], 16, v[vgprValuC+86] // convert C to bf16
v_cmp_u_f32 s8, v[vgprValuC+87], v[vgprValuC+87]   // check Nan
v_bfe_u32 v67, v[vgprValuC+87], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+87], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+87], v67, v69, s8
v_and_or_b32 v86, v[vgprValuC+87], v68, v[vgprValuC+86] // pack two bf16 to dword
buffer_store_b32 v86, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:48 // store D
v_cmp_u_f32 s8, v[vgprValuC+88], v[vgprValuC+88]   // check Nan
v_bfe_u32 v67, v[vgprValuC+88], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+88], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+88], v67, v69, s8
v_lshrrev_b32 v[vgprValuC+88], 16, v[vgprValuC+88] // convert C to bf16
v_cmp_u_f32 s8, v[vgprValuC+89], v[vgprValuC+89]   // check Nan
v_bfe_u32 v67, v[vgprValuC+89], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+89], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+89], v67, v69, s8
v_and_or_b32 v88, v[vgprValuC+89], v68, v[vgprValuC+88] // pack two bf16 to dword
buffer_store_b32 v88, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:56 // store D
v_cmp_u_f32 s8, v[vgprValuC+90], v[vgprValuC+90]   // check Nan
v_bfe_u32 v67, v[vgprValuC+90], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+90], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+90], v67, v69, s8
v_lshrrev_b32 v[vgprValuC+90], 16, v[vgprValuC+90] // convert C to bf16
v_cmp_u_f32 s8, v[vgprValuC+91], v[vgprValuC+91]   // check Nan
v_bfe_u32 v67, v[vgprValuC+91], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+91], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+91], v67, v69, s8
v_and_or_b32 v90, v[vgprValuC+91], v68, v[vgprValuC+90] // pack two bf16 to dword
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_b32 v90, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+92], v[vgprValuC+92]   // check Nan
v_bfe_u32 v67, v[vgprValuC+92], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+92], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+92], v67, v69, s8
v_lshrrev_b32 v[vgprValuC+92], 16, v[vgprValuC+92] // convert C to bf16
v_cmp_u_f32 s8, v[vgprValuC+93], v[vgprValuC+93]   // check Nan
v_bfe_u32 v67, v[vgprValuC+93], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+93], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+93], v67, v69, s8
v_and_or_b32 v92, v[vgprValuC+93], v68, v[vgprValuC+92] // pack two bf16 to dword
buffer_store_b32 v92, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:8 // store D
v_cmp_u_f32 s8, v[vgprValuC+94], v[vgprValuC+94]   // check Nan
v_bfe_u32 v67, v[vgprValuC+94], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+94], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+94], v67, v69, s8
v_lshrrev_b32 v[vgprValuC+94], 16, v[vgprValuC+94] // convert C to bf16
v_cmp_u_f32 s8, v[vgprValuC+95], v[vgprValuC+95]   // check Nan
v_bfe_u32 v67, v[vgprValuC+95], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+95], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+95], v67, v69, s8
v_and_or_b32 v94, v[vgprValuC+95], v68, v[vgprValuC+94] // pack two bf16 to dword
buffer_store_b32 v94, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:16 // store D
v_cmp_u_f32 s8, v[vgprValuC+96], v[vgprValuC+96]   // check Nan
v_bfe_u32 v67, v[vgprValuC+96], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+96], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+96], v67, v69, s8
v_lshrrev_b32 v[vgprValuC+96], 16, v[vgprValuC+96] // convert C to bf16
v_cmp_u_f32 s8, v[vgprValuC+97], v[vgprValuC+97]   // check Nan
v_bfe_u32 v67, v[vgprValuC+97], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+97], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+97], v67, v69, s8
v_and_or_b32 v96, v[vgprValuC+97], v68, v[vgprValuC+96] // pack two bf16 to dword
buffer_store_b32 v96, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:24 // store D
v_cmp_u_f32 s8, v[vgprValuC+98], v[vgprValuC+98]   // check Nan
v_bfe_u32 v67, v[vgprValuC+98], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+98], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+98], v67, v69, s8
v_lshrrev_b32 v[vgprValuC+98], 16, v[vgprValuC+98] // convert C to bf16
v_cmp_u_f32 s8, v[vgprValuC+99], v[vgprValuC+99]   // check Nan
v_bfe_u32 v67, v[vgprValuC+99], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+99], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+99], v67, v69, s8
v_and_or_b32 v98, v[vgprValuC+99], v68, v[vgprValuC+98] // pack two bf16 to dword
buffer_store_b32 v98, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
v_cmp_u_f32 s8, v[vgprValuC+100], v[vgprValuC+100] // check Nan
v_bfe_u32 v67, v[vgprValuC+100], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+100], v67, v70         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+100], v67, v69, s8
v_lshrrev_b32 v[vgprValuC+100], 16, v[vgprValuC+100] // convert C to bf16
v_cmp_u_f32 s8, v[vgprValuC+101], v[vgprValuC+101] // check Nan
v_bfe_u32 v67, v[vgprValuC+101], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+101], v67, v70         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+101], v67, v69, s8
v_and_or_b32 v100, v[vgprValuC+101], v68, v[vgprValuC+100] // pack two bf16 to dword
buffer_store_b32 v100, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:40 // store D
v_cmp_u_f32 s8, v[vgprValuC+102], v[vgprValuC+102] // check Nan
v_bfe_u32 v67, v[vgprValuC+102], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+102], v67, v70         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+102], v67, v69, s8
v_lshrrev_b32 v[vgprValuC+102], 16, v[vgprValuC+102] // convert C to bf16
v_cmp_u_f32 s8, v[vgprValuC+103], v[vgprValuC+103] // check Nan
v_bfe_u32 v67, v[vgprValuC+103], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+103], v67, v70         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+103], v67, v69, s8
v_and_or_b32 v102, v[vgprValuC+103], v68, v[vgprValuC+102] // pack two bf16 to dword
buffer_store_b32 v102, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:48 // store D
v_cmp_u_f32 s8, v[vgprValuC+104], v[vgprValuC+104] // check Nan
v_bfe_u32 v67, v[vgprValuC+104], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+104], v67, v70         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+104], v67, v69, s8
v_lshrrev_b32 v[vgprValuC+104], 16, v[vgprValuC+104] // convert C to bf16
v_cmp_u_f32 s8, v[vgprValuC+105], v[vgprValuC+105] // check Nan
v_bfe_u32 v67, v[vgprValuC+105], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+105], v67, v70         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+105], v67, v69, s8
v_and_or_b32 v104, v[vgprValuC+105], v68, v[vgprValuC+104] // pack two bf16 to dword
buffer_store_b32 v104, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:56 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW2_GSU1_NonEdgeEnd:
label_GW_B0_FD0_VW2_GSU1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=60 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw2); (0,1,0,0:vw2); (0,2,0,0:vw2); (0,3,0,0:vw2); (0,4,0,0:vw2); (0,5,0,0:vw2); (0,6,0,0:vw2); (0,7,0,0:vw2); (0,0,1,0:vw2); (0,1,1,0:vw2); (0,2,1,0:vw2); (0,3,1,0:vw2); (0,4,1,0:vw2); (0,5,1,0:vw2); (0,6,1,0:vw2); (0,7,1,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v66, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s28, v60, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v71, v63, v60, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v71, v66, v71, s30                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v64, vcc_lo, v60, 4                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v104, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v104, v66, v104, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
v_add_co_u32 v64, vcc_lo, v60, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v105, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v105, v66, v105, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
v_add_co_u32 v64, vcc_lo, v60, 12                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v106, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v106, v66, v106, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
v_add_co_u32 v64, vcc_lo, v60, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v107, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v107, v66, v107, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
v_add_co_u32 v64, vcc_lo, v60, 20                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v108, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v108, v66, v108, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
v_add_co_u32 v64, vcc_lo, v60, 24                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v109, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v109, v66, v109, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
v_add_co_u32 v64, vcc_lo, v60, 28                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v110, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v110, v66, v110, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v61, vcc_lo, v61, 1                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v62, v62, s[sgprStrideC1J]            // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v63, v63, s[sgprStrideD1J]            // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v60, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v111, v63, v60, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v111, v66, v111, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
v_add_co_u32 v64, vcc_lo, v60, 4                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v112, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v112, v66, v112, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,2,0) */
v_add_co_u32 v64, vcc_lo, v60, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v113, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v113, v66, v113, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,3,0) */
v_add_co_u32 v64, vcc_lo, v60, 12                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v114, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v114, v66, v114, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,4,0) */
v_add_co_u32 v64, vcc_lo, v60, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v115, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v115, v66, v115, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,5,0) */
v_add_co_u32 v64, vcc_lo, v60, 20                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v116, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v116, v66, v116, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,6,0) */
v_add_co_u32 v64, vcc_lo, v60, 24                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v117, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v117, v66, v117, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,7,0) */
v_add_co_u32 v64, vcc_lo, v60, 28                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v118, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v118, v66, v118, s30                 // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0), (0, 0, 1, 0), (0, 1, 1, 0), (0, 2, 1, 0), (0, 3, 1, 0), (0, 4, 1, 0), (0, 5, 1, 0), (0, 6, 1, 0), (0, 7, 1, 0)] */
v_mul_f32 v[vgprValuC+72], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+73], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+74], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+75], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+76], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+77], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+78], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+79], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+80], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+81], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+82], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+83], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+84], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+85], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+86], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+87], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+88], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+89], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+90], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+91], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+92], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+93], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+94], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+95], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+96], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+97], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+98], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+100], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+102], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+103], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v68, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v69, 0x7fff0000                          // fp32 Nan
v_mov_b32 v70, 0x7fff                              // rounding bias for bfloat16
v_cmp_u_f32 s28, v[vgprValuC+72], v[vgprValuC+72]  // check Nan
v_bfe_u32 v67, v[vgprValuC+72], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+72], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+72], v67, v69, s28
v_lshrrev_b32 v[vgprValuC+72], 16, v[vgprValuC+72] // convert C to bf16
v_cmp_u_f32 s28, v[vgprValuC+73], v[vgprValuC+73]  // check Nan
v_bfe_u32 v67, v[vgprValuC+73], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+73], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+73], v67, v69, s28
v_and_or_b32 v72, v[vgprValuC+73], v68, v[vgprValuC+72] // pack two bf16 to dword
buffer_store_b32 v72, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+74], v[vgprValuC+74]  // check Nan
v_bfe_u32 v67, v[vgprValuC+74], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+74], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+74], v67, v69, s28
v_lshrrev_b32 v[vgprValuC+74], 16, v[vgprValuC+74] // convert C to bf16
v_cmp_u_f32 s28, v[vgprValuC+75], v[vgprValuC+75]  // check Nan
v_bfe_u32 v67, v[vgprValuC+75], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+75], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+75], v67, v69, s28
v_and_or_b32 v74, v[vgprValuC+75], v68, v[vgprValuC+74] // pack two bf16 to dword
buffer_store_b32 v74, v104, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+76], v[vgprValuC+76]  // check Nan
v_bfe_u32 v67, v[vgprValuC+76], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+76], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+76], v67, v69, s28
v_lshrrev_b32 v[vgprValuC+76], 16, v[vgprValuC+76] // convert C to bf16
v_cmp_u_f32 s28, v[vgprValuC+77], v[vgprValuC+77]  // check Nan
v_bfe_u32 v67, v[vgprValuC+77], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+77], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+77], v67, v69, s28
v_and_or_b32 v76, v[vgprValuC+77], v68, v[vgprValuC+76] // pack two bf16 to dword
buffer_store_b32 v76, v105, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+78], v[vgprValuC+78]  // check Nan
v_bfe_u32 v67, v[vgprValuC+78], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+78], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+78], v67, v69, s28
v_lshrrev_b32 v[vgprValuC+78], 16, v[vgprValuC+78] // convert C to bf16
v_cmp_u_f32 s28, v[vgprValuC+79], v[vgprValuC+79]  // check Nan
v_bfe_u32 v67, v[vgprValuC+79], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+79], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+79], v67, v69, s28
v_and_or_b32 v78, v[vgprValuC+79], v68, v[vgprValuC+78] // pack two bf16 to dword
buffer_store_b32 v78, v106, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+80], v[vgprValuC+80]  // check Nan
v_bfe_u32 v67, v[vgprValuC+80], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+80], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+80], v67, v69, s28
v_lshrrev_b32 v[vgprValuC+80], 16, v[vgprValuC+80] // convert C to bf16
v_cmp_u_f32 s28, v[vgprValuC+81], v[vgprValuC+81]  // check Nan
v_bfe_u32 v67, v[vgprValuC+81], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+81], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+81], v67, v69, s28
v_and_or_b32 v80, v[vgprValuC+81], v68, v[vgprValuC+80] // pack two bf16 to dword
buffer_store_b32 v80, v107, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+82], v[vgprValuC+82]  // check Nan
v_bfe_u32 v67, v[vgprValuC+82], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+82], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+82], v67, v69, s28
v_lshrrev_b32 v[vgprValuC+82], 16, v[vgprValuC+82] // convert C to bf16
v_cmp_u_f32 s28, v[vgprValuC+83], v[vgprValuC+83]  // check Nan
v_bfe_u32 v67, v[vgprValuC+83], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+83], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+83], v67, v69, s28
v_and_or_b32 v82, v[vgprValuC+83], v68, v[vgprValuC+82] // pack two bf16 to dword
buffer_store_b32 v82, v108, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+84], v[vgprValuC+84]  // check Nan
v_bfe_u32 v67, v[vgprValuC+84], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+84], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+84], v67, v69, s28
v_lshrrev_b32 v[vgprValuC+84], 16, v[vgprValuC+84] // convert C to bf16
v_cmp_u_f32 s28, v[vgprValuC+85], v[vgprValuC+85]  // check Nan
v_bfe_u32 v67, v[vgprValuC+85], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+85], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+85], v67, v69, s28
v_and_or_b32 v84, v[vgprValuC+85], v68, v[vgprValuC+84] // pack two bf16 to dword
buffer_store_b32 v84, v109, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+86], v[vgprValuC+86]  // check Nan
v_bfe_u32 v67, v[vgprValuC+86], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+86], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+86], v67, v69, s28
v_lshrrev_b32 v[vgprValuC+86], 16, v[vgprValuC+86] // convert C to bf16
v_cmp_u_f32 s28, v[vgprValuC+87], v[vgprValuC+87]  // check Nan
v_bfe_u32 v67, v[vgprValuC+87], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+87], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+87], v67, v69, s28
v_and_or_b32 v86, v[vgprValuC+87], v68, v[vgprValuC+86] // pack two bf16 to dword
buffer_store_b32 v86, v110, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+88], v[vgprValuC+88]  // check Nan
v_bfe_u32 v67, v[vgprValuC+88], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+88], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+88], v67, v69, s28
v_lshrrev_b32 v[vgprValuC+88], 16, v[vgprValuC+88] // convert C to bf16
v_cmp_u_f32 s28, v[vgprValuC+89], v[vgprValuC+89]  // check Nan
v_bfe_u32 v67, v[vgprValuC+89], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+89], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+89], v67, v69, s28
v_and_or_b32 v88, v[vgprValuC+89], v68, v[vgprValuC+88] // pack two bf16 to dword
buffer_store_b32 v88, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+90], v[vgprValuC+90]  // check Nan
v_bfe_u32 v67, v[vgprValuC+90], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+90], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+90], v67, v69, s28
v_lshrrev_b32 v[vgprValuC+90], 16, v[vgprValuC+90] // convert C to bf16
v_cmp_u_f32 s28, v[vgprValuC+91], v[vgprValuC+91]  // check Nan
v_bfe_u32 v67, v[vgprValuC+91], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+91], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+91], v67, v69, s28
v_and_or_b32 v90, v[vgprValuC+91], v68, v[vgprValuC+90] // pack two bf16 to dword
buffer_store_b32 v90, v112, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+92], v[vgprValuC+92]  // check Nan
v_bfe_u32 v67, v[vgprValuC+92], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+92], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+92], v67, v69, s28
v_lshrrev_b32 v[vgprValuC+92], 16, v[vgprValuC+92] // convert C to bf16
v_cmp_u_f32 s28, v[vgprValuC+93], v[vgprValuC+93]  // check Nan
v_bfe_u32 v67, v[vgprValuC+93], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+93], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+93], v67, v69, s28
v_and_or_b32 v92, v[vgprValuC+93], v68, v[vgprValuC+92] // pack two bf16 to dword
buffer_store_b32 v92, v113, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+94], v[vgprValuC+94]  // check Nan
v_bfe_u32 v67, v[vgprValuC+94], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+94], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+94], v67, v69, s28
v_lshrrev_b32 v[vgprValuC+94], 16, v[vgprValuC+94] // convert C to bf16
v_cmp_u_f32 s28, v[vgprValuC+95], v[vgprValuC+95]  // check Nan
v_bfe_u32 v67, v[vgprValuC+95], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+95], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+95], v67, v69, s28
v_and_or_b32 v94, v[vgprValuC+95], v68, v[vgprValuC+94] // pack two bf16 to dword
buffer_store_b32 v94, v114, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+96], v[vgprValuC+96]  // check Nan
v_bfe_u32 v67, v[vgprValuC+96], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+96], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+96], v67, v69, s28
v_lshrrev_b32 v[vgprValuC+96], 16, v[vgprValuC+96] // convert C to bf16
v_cmp_u_f32 s28, v[vgprValuC+97], v[vgprValuC+97]  // check Nan
v_bfe_u32 v67, v[vgprValuC+97], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+97], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+97], v67, v69, s28
v_and_or_b32 v96, v[vgprValuC+97], v68, v[vgprValuC+96] // pack two bf16 to dword
buffer_store_b32 v96, v115, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+98], v[vgprValuC+98]  // check Nan
v_bfe_u32 v67, v[vgprValuC+98], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+98], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+98], v67, v69, s28
v_lshrrev_b32 v[vgprValuC+98], 16, v[vgprValuC+98] // convert C to bf16
v_cmp_u_f32 s28, v[vgprValuC+99], v[vgprValuC+99]  // check Nan
v_bfe_u32 v67, v[vgprValuC+99], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+99], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+99], v67, v69, s28
v_and_or_b32 v98, v[vgprValuC+99], v68, v[vgprValuC+98] // pack two bf16 to dword
buffer_store_b32 v98, v116, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+100], v[vgprValuC+100] // check Nan
v_bfe_u32 v67, v[vgprValuC+100], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+100], v67, v70         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+100], v67, v69, s28
v_lshrrev_b32 v[vgprValuC+100], 16, v[vgprValuC+100] // convert C to bf16
v_cmp_u_f32 s28, v[vgprValuC+101], v[vgprValuC+101] // check Nan
v_bfe_u32 v67, v[vgprValuC+101], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+101], v67, v70         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+101], v67, v69, s28
v_and_or_b32 v100, v[vgprValuC+101], v68, v[vgprValuC+100] // pack two bf16 to dword
buffer_store_b32 v100, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+102], v[vgprValuC+102] // check Nan
v_bfe_u32 v67, v[vgprValuC+102], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+102], v67, v70         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+102], v67, v69, s28
v_lshrrev_b32 v[vgprValuC+102], 16, v[vgprValuC+102] // convert C to bf16
v_cmp_u_f32 s28, v[vgprValuC+103], v[vgprValuC+103] // check Nan
v_bfe_u32 v67, v[vgprValuC+103], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+103], v67, v70         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+103], v67, v69, s28
v_and_or_b32 v102, v[vgprValuC+103], v68, v[vgprValuC+102] // pack two bf16 to dword
buffer_store_b32 v102, v118, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW2_GSU1_Else:
label_GW_B0_FD0_VW1_GSU1_Else:
label_GW_B0_FD0_VW1_GSU1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=90 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,1,0,0:vw1); (0,1,0,1:vw1); (0,2,0,0:vw1); (0,2,0,1:vw1); (0,3,0,0:vw1); (0,3,0,1:vw1); (0,4,0,0:vw1); (0,4,0,1:vw1); (0,5,0,0:vw1); (0,5,0,1:vw1); (0,6,0,0:vw1); (0,6,0,1:vw1); (0,7,0,0:vw1); (0,7,0,1:vw1); (0,0,1,0:vw1); (0,0,1,1:vw1); (0,1,1,0:vw1); (0,1,1,1:vw1); (0,2,1,0:vw1); (0,2,1,1:vw1); (0,3,1,0:vw1); (0,3,1,1:vw1); (0,4,1,0:vw1); (0,4,1,1:vw1); (0,5,1,0:vw1); (0,5,1,1:vw1); (0,6,1,0:vw1); (0,6,1,1:vw1); (0,7,1,0:vw1); (0,7,1,1:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v66, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s28, v60, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v103, v63, v60, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v103, v66, v103, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
v_add_co_u32 v64, vcc_lo, v60, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v104, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v104, v66, v104, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v64, vcc_lo, v60, 4                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v105, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v105, v66, v105, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,1) */
v_add_co_u32 v64, vcc_lo, v60, 5                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v106, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v106, v66, v106, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
v_add_co_u32 v64, vcc_lo, v60, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v107, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v107, v66, v107, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,1) */
v_add_co_u32 v64, vcc_lo, v60, 9                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v108, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v108, v66, v108, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
v_add_co_u32 v64, vcc_lo, v60, 12                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v109, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v109, v66, v109, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,1) */
v_add_co_u32 v64, vcc_lo, v60, 13                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v110, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v110, v66, v110, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
v_add_co_u32 v64, vcc_lo, v60, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v111, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v111, v66, v111, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,1) */
v_add_co_u32 v64, vcc_lo, v60, 17                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v112, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v112, v66, v112, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
v_add_co_u32 v64, vcc_lo, v60, 20                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v113, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v113, v66, v113, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,1) */
v_add_co_u32 v64, vcc_lo, v60, 21                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v114, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v114, v66, v114, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
v_add_co_u32 v64, vcc_lo, v60, 24                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v115, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v115, v66, v115, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,1) */
v_add_co_u32 v64, vcc_lo, v60, 25                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v116, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v116, v66, v116, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
v_add_co_u32 v64, vcc_lo, v60, 28                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v117, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v117, v66, v117, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,1) */
v_add_co_u32 v64, vcc_lo, v60, 29                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v118, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v118, v66, v118, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v61, vcc_lo, v61, 1                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v62, v62, s[sgprStrideC1J]            // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v63, v63, s[sgprStrideD1J]            // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v60, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v119, v63, v60, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v119, v66, v119, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,1) */
v_add_co_u32 v64, vcc_lo, v60, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v120, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v120, v66, v120, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
v_add_co_u32 v64, vcc_lo, v60, 4                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v121, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v121, v66, v121, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,1) */
v_add_co_u32 v64, vcc_lo, v60, 5                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v122, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v122, v66, v122, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,2,0) */
v_add_co_u32 v64, vcc_lo, v60, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v123, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v123, v66, v123, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,2,1) */
v_add_co_u32 v64, vcc_lo, v60, 9                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v124, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v124, v66, v124, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,3,0) */
v_add_co_u32 v64, vcc_lo, v60, 12                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v125, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v125, v66, v125, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,3,1) */
v_add_co_u32 v64, vcc_lo, v60, 13                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v126, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v126, v66, v126, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,4,0) */
v_add_co_u32 v64, vcc_lo, v60, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v127, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v127, v66, v127, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,4,1) */
v_add_co_u32 v64, vcc_lo, v60, 17                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v128, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v128, v66, v128, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,5,0) */
v_add_co_u32 v64, vcc_lo, v60, 20                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v129, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v129, v66, v129, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,5,1) */
v_add_co_u32 v64, vcc_lo, v60, 21                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v130, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v66, v130, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,6,0) */
v_add_co_u32 v64, vcc_lo, v60, 24                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v131, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v131, v66, v131, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,6,1) */
v_add_co_u32 v64, vcc_lo, v60, 25                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v132, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v66, v132, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,7,0) */
v_add_co_u32 v64, vcc_lo, v60, 28                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v133, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v133, v66, v133, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,7,1) */
v_add_co_u32 v64, vcc_lo, v60, 29                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v134, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v66, v134, s30                 // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 1, 0, 0), (0, 1, 0, 1), (0, 2, 0, 0), (0, 2, 0, 1), (0, 3, 0, 0), (0, 3, 0, 1), (0, 4, 0, 0), (0, 4, 0, 1), (0, 5, 0, 0), (0, 5, 0, 1), (0, 6, 0, 0), (0, 6, 0, 1), (0, 7, 0, 0), (0, 7, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1), (0, 1, 1, 0), (0, 1, 1, 1), (0, 2, 1, 0), (0, 2, 1, 1), (0, 3, 1, 0), (0, 3, 1, 1), (0, 4, 1, 0), (0, 4, 1, 1), (0, 5, 1, 0), (0, 5, 1, 1), (0, 6, 1, 0), (0, 6, 1, 1), (0, 7, 1, 0), (0, 7, 1, 1)] */
v_mul_f32 v[vgprValuC+71], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+72], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+73], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+74], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+75], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+76], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+77], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+78], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+79], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+80], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+81], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+82], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+83], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+84], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+85], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+86], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+87], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+88], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+89], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+90], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+91], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+92], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+93], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+94], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+95], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+96], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+97], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+98], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+100], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+102], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v68, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v69, 0x7fff0000                          // fp32 Nan
v_mov_b32 v70, 0x7fff                              // rounding bias for bfloat16
v_cmp_u_f32 s28, v[vgprValuC+71], v[vgprValuC+71]  // check Nan
v_bfe_u32 v67, v[vgprValuC+71], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+71], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+71], v67, v69, s28
v_lshrrev_b32 v71, 16, v[vgprValuC+71]             // convert C to bf16
buffer_store_b16 v71, v103, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+72], v[vgprValuC+72]  // check Nan
v_bfe_u32 v67, v[vgprValuC+72], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+72], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+72], v67, v69, s28
v_lshrrev_b32 v72, 16, v[vgprValuC+72]             // convert C to bf16
buffer_store_b16 v72, v104, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+73], v[vgprValuC+73]  // check Nan
v_bfe_u32 v67, v[vgprValuC+73], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+73], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+73], v67, v69, s28
v_lshrrev_b32 v73, 16, v[vgprValuC+73]             // convert C to bf16
buffer_store_b16 v73, v105, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+74], v[vgprValuC+74]  // check Nan
v_bfe_u32 v67, v[vgprValuC+74], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+74], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+74], v67, v69, s28
v_lshrrev_b32 v74, 16, v[vgprValuC+74]             // convert C to bf16
buffer_store_b16 v74, v106, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+75], v[vgprValuC+75]  // check Nan
v_bfe_u32 v67, v[vgprValuC+75], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+75], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+75], v67, v69, s28
v_lshrrev_b32 v75, 16, v[vgprValuC+75]             // convert C to bf16
buffer_store_b16 v75, v107, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+76], v[vgprValuC+76]  // check Nan
v_bfe_u32 v67, v[vgprValuC+76], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+76], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+76], v67, v69, s28
v_lshrrev_b32 v76, 16, v[vgprValuC+76]             // convert C to bf16
buffer_store_b16 v76, v108, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+77], v[vgprValuC+77]  // check Nan
v_bfe_u32 v67, v[vgprValuC+77], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+77], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+77], v67, v69, s28
v_lshrrev_b32 v77, 16, v[vgprValuC+77]             // convert C to bf16
buffer_store_b16 v77, v109, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+78], v[vgprValuC+78]  // check Nan
v_bfe_u32 v67, v[vgprValuC+78], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+78], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+78], v67, v69, s28
v_lshrrev_b32 v78, 16, v[vgprValuC+78]             // convert C to bf16
buffer_store_b16 v78, v110, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+79], v[vgprValuC+79]  // check Nan
v_bfe_u32 v67, v[vgprValuC+79], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+79], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+79], v67, v69, s28
v_lshrrev_b32 v79, 16, v[vgprValuC+79]             // convert C to bf16
buffer_store_b16 v79, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+80], v[vgprValuC+80]  // check Nan
v_bfe_u32 v67, v[vgprValuC+80], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+80], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+80], v67, v69, s28
v_lshrrev_b32 v80, 16, v[vgprValuC+80]             // convert C to bf16
buffer_store_b16 v80, v112, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+81], v[vgprValuC+81]  // check Nan
v_bfe_u32 v67, v[vgprValuC+81], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+81], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+81], v67, v69, s28
v_lshrrev_b32 v81, 16, v[vgprValuC+81]             // convert C to bf16
buffer_store_b16 v81, v113, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+82], v[vgprValuC+82]  // check Nan
v_bfe_u32 v67, v[vgprValuC+82], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+82], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+82], v67, v69, s28
v_lshrrev_b32 v82, 16, v[vgprValuC+82]             // convert C to bf16
buffer_store_b16 v82, v114, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+83], v[vgprValuC+83]  // check Nan
v_bfe_u32 v67, v[vgprValuC+83], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+83], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+83], v67, v69, s28
v_lshrrev_b32 v83, 16, v[vgprValuC+83]             // convert C to bf16
buffer_store_b16 v83, v115, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+84], v[vgprValuC+84]  // check Nan
v_bfe_u32 v67, v[vgprValuC+84], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+84], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+84], v67, v69, s28
v_lshrrev_b32 v84, 16, v[vgprValuC+84]             // convert C to bf16
buffer_store_b16 v84, v116, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+85], v[vgprValuC+85]  // check Nan
v_bfe_u32 v67, v[vgprValuC+85], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+85], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+85], v67, v69, s28
v_lshrrev_b32 v85, 16, v[vgprValuC+85]             // convert C to bf16
buffer_store_b16 v85, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+86], v[vgprValuC+86]  // check Nan
v_bfe_u32 v67, v[vgprValuC+86], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+86], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+86], v67, v69, s28
v_lshrrev_b32 v86, 16, v[vgprValuC+86]             // convert C to bf16
buffer_store_b16 v86, v118, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+87], v[vgprValuC+87]  // check Nan
v_bfe_u32 v67, v[vgprValuC+87], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+87], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+87], v67, v69, s28
v_lshrrev_b32 v87, 16, v[vgprValuC+87]             // convert C to bf16
buffer_store_b16 v87, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+88], v[vgprValuC+88]  // check Nan
v_bfe_u32 v67, v[vgprValuC+88], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+88], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+88], v67, v69, s28
v_lshrrev_b32 v88, 16, v[vgprValuC+88]             // convert C to bf16
buffer_store_b16 v88, v120, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+89], v[vgprValuC+89]  // check Nan
v_bfe_u32 v67, v[vgprValuC+89], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+89], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+89], v67, v69, s28
v_lshrrev_b32 v89, 16, v[vgprValuC+89]             // convert C to bf16
buffer_store_b16 v89, v121, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+90], v[vgprValuC+90]  // check Nan
v_bfe_u32 v67, v[vgprValuC+90], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+90], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+90], v67, v69, s28
v_lshrrev_b32 v90, 16, v[vgprValuC+90]             // convert C to bf16
buffer_store_b16 v90, v122, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+91], v[vgprValuC+91]  // check Nan
v_bfe_u32 v67, v[vgprValuC+91], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+91], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+91], v67, v69, s28
v_lshrrev_b32 v91, 16, v[vgprValuC+91]             // convert C to bf16
buffer_store_b16 v91, v123, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+92], v[vgprValuC+92]  // check Nan
v_bfe_u32 v67, v[vgprValuC+92], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+92], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+92], v67, v69, s28
v_lshrrev_b32 v92, 16, v[vgprValuC+92]             // convert C to bf16
buffer_store_b16 v92, v124, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+93], v[vgprValuC+93]  // check Nan
v_bfe_u32 v67, v[vgprValuC+93], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+93], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+93], v67, v69, s28
v_lshrrev_b32 v93, 16, v[vgprValuC+93]             // convert C to bf16
buffer_store_b16 v93, v125, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+94], v[vgprValuC+94]  // check Nan
v_bfe_u32 v67, v[vgprValuC+94], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+94], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+94], v67, v69, s28
v_lshrrev_b32 v94, 16, v[vgprValuC+94]             // convert C to bf16
buffer_store_b16 v94, v126, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+95], v[vgprValuC+95]  // check Nan
v_bfe_u32 v67, v[vgprValuC+95], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+95], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+95], v67, v69, s28
v_lshrrev_b32 v95, 16, v[vgprValuC+95]             // convert C to bf16
buffer_store_b16 v95, v127, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+96], v[vgprValuC+96]  // check Nan
v_bfe_u32 v67, v[vgprValuC+96], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+96], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+96], v67, v69, s28
v_lshrrev_b32 v96, 16, v[vgprValuC+96]             // convert C to bf16
buffer_store_b16 v96, v128, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+97], v[vgprValuC+97]  // check Nan
v_bfe_u32 v67, v[vgprValuC+97], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+97], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+97], v67, v69, s28
v_lshrrev_b32 v97, 16, v[vgprValuC+97]             // convert C to bf16
buffer_store_b16 v97, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+98], v[vgprValuC+98]  // check Nan
v_bfe_u32 v67, v[vgprValuC+98], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+98], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+98], v67, v69, s28
v_lshrrev_b32 v98, 16, v[vgprValuC+98]             // convert C to bf16
buffer_store_b16 v98, v130, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+99], v[vgprValuC+99]  // check Nan
v_bfe_u32 v67, v[vgprValuC+99], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+99], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+99], v67, v69, s28
v_lshrrev_b32 v99, 16, v[vgprValuC+99]             // convert C to bf16
buffer_store_b16 v99, v131, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+100], v[vgprValuC+100] // check Nan
v_bfe_u32 v67, v[vgprValuC+100], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+100], v67, v70         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+100], v67, v69, s28
v_lshrrev_b32 v100, 16, v[vgprValuC+100]           // convert C to bf16
buffer_store_b16 v100, v132, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+101], v[vgprValuC+101] // check Nan
v_bfe_u32 v67, v[vgprValuC+101], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+101], v67, v70         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+101], v67, v69, s28
v_lshrrev_b32 v101, 16, v[vgprValuC+101]           // convert C to bf16
buffer_store_b16 v101, v133, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s28, v[vgprValuC+102], v[vgprValuC+102] // check Nan
v_bfe_u32 v67, v[vgprValuC+102], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+102], v67, v70         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+102], v67, v69, s28
v_lshrrev_b32 v102, 16, v[vgprValuC+102]           // convert C to bf16
buffer_store_b16 v102, v134, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
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
s_cbranch_scc1 label_GW_B1_FD0_VW2_GSU1_Else       // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 64 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s28, 63, s[sgprSizeJ]                    // s28 = s[sgprSizeJ] % 64
s_add_u32 s29, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s29                // wg1 >= nwg1-1
s_cselect_b32 s28, s28, 0                          // set rem
s_cmpk_gt_u32 s28, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW2_GSU1_Then       // jump if edges required
label_GW_B1_FD0_VW2_GSU1_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=58 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw2); (0,1,0,0:vw2); (0,2,0,0:vw2); (0,3,0,0:vw2); (0,4,0,0:vw2); (0,5,0,0:vw2); (0,6,0,0:vw2); (0,7,0,0:vw2); (0,0,1,0:vw2); (0,1,1,0:vw2); (0,2,1,0:vw2); (0,3,1,0:vw2); (0,4,1,0:vw2); (0,5,1,0:vw2); (0,6,1,0:vw2); (0,7,1,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v72, v62, v60, 1                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=60, coord0Vgpr=60 (multiple bpe)
buffer_load_b32 v73, v72, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
buffer_load_b32 v106, v72, s[sgprSrdC:sgprSrdC+3], 0 offen offset:8 // load C
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
buffer_load_b32 v107, v72, s[sgprSrdC:sgprSrdC+3], 0 offen offset:16 // load C
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
buffer_load_b32 v108, v72, s[sgprSrdC:sgprSrdC+3], 0 offen offset:24 // load C
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
buffer_load_b32 v109, v72, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
buffer_load_b32 v110, v72, s[sgprSrdC:sgprSrdC+3], 0 offen offset:40 // load C
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
buffer_load_b32 v111, v72, s[sgprSrdC:sgprSrdC+3], 0 offen offset:48 // load C
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
buffer_load_b32 v112, v72, s[sgprSrdC:sgprSrdC+3], 0 offen offset:56 // load C
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_load_b32 v113, v72, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
buffer_load_b32 v114, v72, s[sgprSrdC:sgprSrdC+3], 0 offen offset:8 // load C
/* (d1,vc1,d0,vc0)=(0,1,2,0) */
buffer_load_b32 v115, v72, s[sgprSrdC:sgprSrdC+3], 0 offen offset:16 // load C
/* (d1,vc1,d0,vc0)=(0,1,3,0) */
buffer_load_b32 v116, v72, s[sgprSrdC:sgprSrdC+3], 0 offen offset:24 // load C
/* (d1,vc1,d0,vc0)=(0,1,4,0) */
buffer_load_b32 v117, v72, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
/* (d1,vc1,d0,vc0)=(0,1,5,0) */
buffer_load_b32 v118, v72, s[sgprSrdC:sgprSrdC+3], 0 offen offset:40 // load C
/* (d1,vc1,d0,vc0)=(0,1,6,0) */
buffer_load_b32 v119, v72, s[sgprSrdC:sgprSrdC+3], 0 offen offset:48 // load C
/* (d1,vc1,d0,vc0)=(0,1,7,0) */
buffer_load_b32 v120, v72, s[sgprSrdC:sgprSrdC+3], 0 offen offset:56 // load C
v_add_lshl_u32 v71, v63, v60, 1                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=60, coord0Vgpr=60 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0), (0, 0, 1, 0), (0, 1, 1, 0), (0, 2, 1, 0), (0, 3, 1, 0), (0, 4, 1, 0), (0, 5, 1, 0), (0, 6, 1, 0), (0, 7, 1, 0)] */
v_mul_f32 v[vgprValuC+74], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+75], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+76], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+77], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+78], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+79], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+80], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+81], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+82], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+83], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+84], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+85], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+86], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+87], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+88], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+89], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+90], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+91], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+92], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+93], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+94], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+95], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+96], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+97], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+98], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+100], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+102], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+103], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+104], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+105], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v68, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v69, 0x7fff0000                          // fp32 Nan
v_mov_b32 v70, 0x7fff                              // rounding bias for bfloat16

s_waitcnt vmcnt(15)                                // vlcnt(15) = 16 - 1 (beta) (interleaved)
v_lshlrev_b32 v64, 16, v73                         // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+74], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_and_b32 v64, v73, v68                            // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+75], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+74], v[vgprValuC+74]   // check Nan
v_bfe_u32 v67, v[vgprValuC+74], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+74], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+74], v67, v69, s8
v_lshrrev_b32 v[vgprValuC+74], 16, v[vgprValuC+74] // convert C to bf16
v_cmp_u_f32 s8, v[vgprValuC+75], v[vgprValuC+75]   // check Nan
v_bfe_u32 v67, v[vgprValuC+75], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+75], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+75], v67, v69, s8
v_and_or_b32 v74, v[vgprValuC+75], v68, v[vgprValuC+74] // pack two bf16 to dword
buffer_store_b32 v74, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(14)                                // vlcnt(14) = 16 - 2 (beta) (interleaved)
v_lshlrev_b32 v64, 16, v106                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+76], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_and_b32 v64, v106, v68                           // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+77], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+76], v[vgprValuC+76]   // check Nan
v_bfe_u32 v67, v[vgprValuC+76], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+76], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+76], v67, v69, s8
v_lshrrev_b32 v[vgprValuC+76], 16, v[vgprValuC+76] // convert C to bf16
v_cmp_u_f32 s8, v[vgprValuC+77], v[vgprValuC+77]   // check Nan
v_bfe_u32 v67, v[vgprValuC+77], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+77], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+77], v67, v69, s8
v_and_or_b32 v76, v[vgprValuC+77], v68, v[vgprValuC+76] // pack two bf16 to dword
buffer_store_b32 v76, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:8 // store D

s_waitcnt vmcnt(13)                                // vlcnt(13) = 16 - 3 (beta) (interleaved)
v_lshlrev_b32 v64, 16, v107                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+78], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_and_b32 v64, v107, v68                           // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+79], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+78], v[vgprValuC+78]   // check Nan
v_bfe_u32 v67, v[vgprValuC+78], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+78], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+78], v67, v69, s8
v_lshrrev_b32 v[vgprValuC+78], 16, v[vgprValuC+78] // convert C to bf16
v_cmp_u_f32 s8, v[vgprValuC+79], v[vgprValuC+79]   // check Nan
v_bfe_u32 v67, v[vgprValuC+79], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+79], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+79], v67, v69, s8
v_and_or_b32 v78, v[vgprValuC+79], v68, v[vgprValuC+78] // pack two bf16 to dword
buffer_store_b32 v78, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:16 // store D

s_waitcnt vmcnt(12)                                // vlcnt(12) = 16 - 4 (beta) (interleaved)
v_lshlrev_b32 v64, 16, v108                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+80], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_and_b32 v64, v108, v68                           // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+81], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+80], v[vgprValuC+80]   // check Nan
v_bfe_u32 v67, v[vgprValuC+80], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+80], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+80], v67, v69, s8
v_lshrrev_b32 v[vgprValuC+80], 16, v[vgprValuC+80] // convert C to bf16
v_cmp_u_f32 s8, v[vgprValuC+81], v[vgprValuC+81]   // check Nan
v_bfe_u32 v67, v[vgprValuC+81], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+81], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+81], v67, v69, s8
v_and_or_b32 v80, v[vgprValuC+81], v68, v[vgprValuC+80] // pack two bf16 to dword
buffer_store_b32 v80, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:24 // store D

s_waitcnt vmcnt(11)                                // vlcnt(11) = 16 - 5 (beta) (interleaved)
v_lshlrev_b32 v64, 16, v109                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+82], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_and_b32 v64, v109, v68                           // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+83], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+82], v[vgprValuC+82]   // check Nan
v_bfe_u32 v67, v[vgprValuC+82], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+82], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+82], v67, v69, s8
v_lshrrev_b32 v[vgprValuC+82], 16, v[vgprValuC+82] // convert C to bf16
v_cmp_u_f32 s8, v[vgprValuC+83], v[vgprValuC+83]   // check Nan
v_bfe_u32 v67, v[vgprValuC+83], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+83], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+83], v67, v69, s8
v_and_or_b32 v82, v[vgprValuC+83], v68, v[vgprValuC+82] // pack two bf16 to dword
buffer_store_b32 v82, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

s_waitcnt vmcnt(10)                                // vlcnt(10) = 16 - 6 (beta) (interleaved)
v_lshlrev_b32 v64, 16, v110                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+84], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_and_b32 v64, v110, v68                           // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+85], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+84], v[vgprValuC+84]   // check Nan
v_bfe_u32 v67, v[vgprValuC+84], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+84], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+84], v67, v69, s8
v_lshrrev_b32 v[vgprValuC+84], 16, v[vgprValuC+84] // convert C to bf16
v_cmp_u_f32 s8, v[vgprValuC+85], v[vgprValuC+85]   // check Nan
v_bfe_u32 v67, v[vgprValuC+85], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+85], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+85], v67, v69, s8
v_and_or_b32 v84, v[vgprValuC+85], v68, v[vgprValuC+84] // pack two bf16 to dword
buffer_store_b32 v84, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:40 // store D

s_waitcnt vmcnt(9)                                 // vlcnt(9) = 16 - 7 (beta) (interleaved)
v_lshlrev_b32 v64, 16, v111                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+86], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_and_b32 v64, v111, v68                           // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+87], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+86], v[vgprValuC+86]   // check Nan
v_bfe_u32 v67, v[vgprValuC+86], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+86], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+86], v67, v69, s8
v_lshrrev_b32 v[vgprValuC+86], 16, v[vgprValuC+86] // convert C to bf16
v_cmp_u_f32 s8, v[vgprValuC+87], v[vgprValuC+87]   // check Nan
v_bfe_u32 v67, v[vgprValuC+87], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+87], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+87], v67, v69, s8
v_and_or_b32 v86, v[vgprValuC+87], v68, v[vgprValuC+86] // pack two bf16 to dword
buffer_store_b32 v86, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:48 // store D

s_waitcnt vmcnt(8)                                 // vlcnt(8) = 16 - 8 (beta) (interleaved)
v_lshlrev_b32 v64, 16, v112                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+88], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_and_b32 v64, v112, v68                           // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+89], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+88], v[vgprValuC+88]   // check Nan
v_bfe_u32 v67, v[vgprValuC+88], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+88], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+88], v67, v69, s8
v_lshrrev_b32 v[vgprValuC+88], 16, v[vgprValuC+88] // convert C to bf16
v_cmp_u_f32 s8, v[vgprValuC+89], v[vgprValuC+89]   // check Nan
v_bfe_u32 v67, v[vgprValuC+89], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+89], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+89], v67, v69, s8
v_and_or_b32 v88, v[vgprValuC+89], v68, v[vgprValuC+88] // pack two bf16 to dword
buffer_store_b32 v88, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:56 // store D

s_waitcnt vmcnt(7)                                 // vlcnt(7) = 16 - 9 (beta) (interleaved)
v_lshlrev_b32 v64, 16, v113                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+90], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_and_b32 v64, v113, v68                           // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+91], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+90], v[vgprValuC+90]   // check Nan
v_bfe_u32 v67, v[vgprValuC+90], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+90], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+90], v67, v69, s8
v_lshrrev_b32 v[vgprValuC+90], 16, v[vgprValuC+90] // convert C to bf16
v_cmp_u_f32 s8, v[vgprValuC+91], v[vgprValuC+91]   // check Nan
v_bfe_u32 v67, v[vgprValuC+91], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+91], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+91], v67, v69, s8
v_and_or_b32 v90, v[vgprValuC+91], v68, v[vgprValuC+90] // pack two bf16 to dword
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_b32 v90, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(6)                                 // vlcnt(6) = 16 - 10 (beta) (interleaved)
v_lshlrev_b32 v64, 16, v114                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+92], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_and_b32 v64, v114, v68                           // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+93], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+92], v[vgprValuC+92]   // check Nan
v_bfe_u32 v67, v[vgprValuC+92], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+92], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+92], v67, v69, s8
v_lshrrev_b32 v[vgprValuC+92], 16, v[vgprValuC+92] // convert C to bf16
v_cmp_u_f32 s8, v[vgprValuC+93], v[vgprValuC+93]   // check Nan
v_bfe_u32 v67, v[vgprValuC+93], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+93], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+93], v67, v69, s8
v_and_or_b32 v92, v[vgprValuC+93], v68, v[vgprValuC+92] // pack two bf16 to dword
buffer_store_b32 v92, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:8 // store D

s_waitcnt vmcnt(5)                                 // vlcnt(5) = 16 - 11 (beta) (interleaved)
v_lshlrev_b32 v64, 16, v115                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+94], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_and_b32 v64, v115, v68                           // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+95], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+94], v[vgprValuC+94]   // check Nan
v_bfe_u32 v67, v[vgprValuC+94], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+94], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+94], v67, v69, s8
v_lshrrev_b32 v[vgprValuC+94], 16, v[vgprValuC+94] // convert C to bf16
v_cmp_u_f32 s8, v[vgprValuC+95], v[vgprValuC+95]   // check Nan
v_bfe_u32 v67, v[vgprValuC+95], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+95], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+95], v67, v69, s8
v_and_or_b32 v94, v[vgprValuC+95], v68, v[vgprValuC+94] // pack two bf16 to dword
buffer_store_b32 v94, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:16 // store D

s_waitcnt vmcnt(4)                                 // vlcnt(4) = 16 - 12 (beta) (interleaved)
v_lshlrev_b32 v64, 16, v116                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+96], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_and_b32 v64, v116, v68                           // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+97], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+96], v[vgprValuC+96]   // check Nan
v_bfe_u32 v67, v[vgprValuC+96], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+96], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+96], v67, v69, s8
v_lshrrev_b32 v[vgprValuC+96], 16, v[vgprValuC+96] // convert C to bf16
v_cmp_u_f32 s8, v[vgprValuC+97], v[vgprValuC+97]   // check Nan
v_bfe_u32 v67, v[vgprValuC+97], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+97], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+97], v67, v69, s8
v_and_or_b32 v96, v[vgprValuC+97], v68, v[vgprValuC+96] // pack two bf16 to dword
buffer_store_b32 v96, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:24 // store D

s_waitcnt vmcnt(3)                                 // vlcnt(3) = 16 - 13 (beta) (interleaved)
v_lshlrev_b32 v64, 16, v117                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+98], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_and_b32 v64, v117, v68                           // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+99], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+98], v[vgprValuC+98]   // check Nan
v_bfe_u32 v67, v[vgprValuC+98], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+98], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+98], v67, v69, s8
v_lshrrev_b32 v[vgprValuC+98], 16, v[vgprValuC+98] // convert C to bf16
v_cmp_u_f32 s8, v[vgprValuC+99], v[vgprValuC+99]   // check Nan
v_bfe_u32 v67, v[vgprValuC+99], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+99], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+99], v67, v69, s8
v_and_or_b32 v98, v[vgprValuC+99], v68, v[vgprValuC+98] // pack two bf16 to dword
buffer_store_b32 v98, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

s_waitcnt vmcnt(2)                                 // vlcnt(2) = 16 - 14 (beta) (interleaved)
v_lshlrev_b32 v64, 16, v118                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+100], v64, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_and_b32 v64, v118, v68                           // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+101], v64, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+100], v[vgprValuC+100] // check Nan
v_bfe_u32 v67, v[vgprValuC+100], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+100], v67, v70         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+100], v67, v69, s8
v_lshrrev_b32 v[vgprValuC+100], 16, v[vgprValuC+100] // convert C to bf16
v_cmp_u_f32 s8, v[vgprValuC+101], v[vgprValuC+101] // check Nan
v_bfe_u32 v67, v[vgprValuC+101], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+101], v67, v70         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+101], v67, v69, s8
v_and_or_b32 v100, v[vgprValuC+101], v68, v[vgprValuC+100] // pack two bf16 to dword
buffer_store_b32 v100, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:40 // store D

s_waitcnt vmcnt(1)                                 // vlcnt(1) = 16 - 15 (beta) (interleaved)
v_lshlrev_b32 v64, 16, v119                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+102], v64, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_and_b32 v64, v119, v68                           // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+103], v64, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+102], v[vgprValuC+102] // check Nan
v_bfe_u32 v67, v[vgprValuC+102], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+102], v67, v70         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+102], v67, v69, s8
v_lshrrev_b32 v[vgprValuC+102], 16, v[vgprValuC+102] // convert C to bf16
v_cmp_u_f32 s8, v[vgprValuC+103], v[vgprValuC+103] // check Nan
v_bfe_u32 v67, v[vgprValuC+103], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+103], v67, v70         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+103], v67, v69, s8
v_and_or_b32 v102, v[vgprValuC+103], v68, v[vgprValuC+102] // pack two bf16 to dword
buffer_store_b32 v102, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:48 // store D

s_waitcnt vmcnt(0)                                 // vlcnt(0) = 16 - 16 (beta) (interleaved)
v_lshlrev_b32 v64, 16, v120                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+104], v64, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_and_b32 v64, v120, v68                           // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+105], v64, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+104], v[vgprValuC+104] // check Nan
v_bfe_u32 v67, v[vgprValuC+104], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+104], v67, v70         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+104], v67, v69, s8
v_lshrrev_b32 v[vgprValuC+104], 16, v[vgprValuC+104] // convert C to bf16
v_cmp_u_f32 s8, v[vgprValuC+105], v[vgprValuC+105] // check Nan
v_bfe_u32 v67, v[vgprValuC+105], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+105], v67, v70         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+105], v67, v69, s8
v_and_or_b32 v104, v[vgprValuC+105], v68, v[vgprValuC+104] // pack two bf16 to dword
buffer_store_b32 v104, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:56 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_FD0_VW2_GSU1_NonEdgeEnd:
label_GW_B1_FD0_VW2_GSU1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=44 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw2); (0,1,0,0:vw2); (0,2,0,0:vw2); (0,3,0,0:vw2); (0,4,0,0:vw2); (0,5,0,0:vw2); (0,6,0,0:vw2); (0,7,0,0:vw2); (0,0,1,0:vw2); (0,1,1,0:vw2); (0,2,1,0:vw2); (0,3,1,0:vw2); (0,4,1,0:vw2); (0,5,1,0:vw2); (0,6,1,0:vw2); (0,7,1,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v66, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s28, v60, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v104, v62, v60, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v104, v66, v104, s30                 // LDC clip if OOB. offset
buffer_load_b32 v71, v104, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v104, v63, v60, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v104, v66, v104, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v64, vcc_lo, v60, 4                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v106, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v106, v66, v106, s30                 // LDC clip if OOB. offset
buffer_load_b32 v105, v106, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v106, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v106, v66, v106, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
v_add_co_u32 v64, vcc_lo, v60, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v108, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v108, v66, v108, s30                 // LDC clip if OOB. offset
buffer_load_b32 v107, v108, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v108, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v108, v66, v108, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
v_add_co_u32 v64, vcc_lo, v60, 12                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v110, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v110, v66, v110, s30                 // LDC clip if OOB. offset
buffer_load_b32 v109, v110, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v110, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v110, v66, v110, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
v_add_co_u32 v64, vcc_lo, v60, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v112, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v112, v66, v112, s30                 // LDC clip if OOB. offset
buffer_load_b32 v111, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v112, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v112, v66, v112, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
v_add_co_u32 v64, vcc_lo, v60, 20                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v114, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v114, v66, v114, s30                 // LDC clip if OOB. offset
buffer_load_b32 v113, v114, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v114, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v114, v66, v114, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
v_add_co_u32 v64, vcc_lo, v60, 24                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v116, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v116, v66, v116, s30                 // LDC clip if OOB. offset
buffer_load_b32 v115, v116, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v116, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v116, v66, v116, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
v_add_co_u32 v64, vcc_lo, v60, 28                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v118, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v118, v66, v118, s30                 // LDC clip if OOB. offset
buffer_load_b32 v117, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v118, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v118, v66, v118, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v61, vcc_lo, v61, 1                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v62, v62, s[sgprStrideC1J]            // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v63, v63, s[sgprStrideD1J]            // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v60, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v120, v62, v60, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v120, v66, v120, s30                 // LDC clip if OOB. offset
buffer_load_b32 v119, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v120, v63, v60, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v120, v66, v120, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
v_add_co_u32 v64, vcc_lo, v60, 4                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v122, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v122, v66, v122, s30                 // LDC clip if OOB. offset
buffer_load_b32 v121, v122, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v122, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v122, v66, v122, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,2,0) */
v_add_co_u32 v64, vcc_lo, v60, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v124, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v124, v66, v124, s30                 // LDC clip if OOB. offset
buffer_load_b32 v123, v124, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v124, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v124, v66, v124, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,3,0) */
v_add_co_u32 v64, vcc_lo, v60, 12                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v126, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v126, v66, v126, s30                 // LDC clip if OOB. offset
buffer_load_b32 v125, v126, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v126, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v126, v66, v126, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,4,0) */
v_add_co_u32 v64, vcc_lo, v60, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v128, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v128, v66, v128, s30                 // LDC clip if OOB. offset
buffer_load_b32 v127, v128, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v128, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v128, v66, v128, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,5,0) */
v_add_co_u32 v64, vcc_lo, v60, 20                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v130, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v66, v130, s30                 // LDC clip if OOB. offset
buffer_load_b32 v129, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v130, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v66, v130, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,6,0) */
v_add_co_u32 v64, vcc_lo, v60, 24                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v132, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v66, v132, s30                 // LDC clip if OOB. offset
buffer_load_b32 v131, v132, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v132, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v66, v132, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,7,0) */
v_add_co_u32 v64, vcc_lo, v60, 28                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v134, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v66, v134, s30                 // LDC clip if OOB. offset
buffer_load_b32 v133, v134, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v134, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v66, v134, s30                 // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0), (0, 0, 1, 0), (0, 1, 1, 0), (0, 2, 1, 0), (0, 3, 1, 0), (0, 4, 1, 0), (0, 5, 1, 0), (0, 6, 1, 0), (0, 7, 1, 0)] */
v_mul_f32 v[vgprValuC+72], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+73], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+74], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+75], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+76], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+77], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+78], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+79], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+80], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+81], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+82], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+83], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+84], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+85], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+86], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+87], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+88], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+89], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+90], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+91], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+92], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+93], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+94], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+95], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+96], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+97], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+98], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+100], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+102], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+103], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_mov_b32 v68, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v69, 0x7fff0000                          // fp32 Nan
v_mov_b32 v70, 0x7fff                              // rounding bias for bfloat16
v_lshlrev_b32 v64, 16, v71                         // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+72], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_and_b32 v64, v71, v68                            // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+73], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+72], v[vgprValuC+72]  // check Nan
v_bfe_u32 v67, v[vgprValuC+72], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+72], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+72], v67, v69, s28
v_lshrrev_b32 v[vgprValuC+72], 16, v[vgprValuC+72] // convert C to bf16
v_cmp_u_f32 s28, v[vgprValuC+73], v[vgprValuC+73]  // check Nan
v_bfe_u32 v67, v[vgprValuC+73], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+73], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+73], v67, v69, s28
v_and_or_b32 v72, v[vgprValuC+73], v68, v[vgprValuC+72] // pack two bf16 to dword
buffer_store_b32 v72, v104, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v105                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+74], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_and_b32 v64, v105, v68                           // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+75], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+74], v[vgprValuC+74]  // check Nan
v_bfe_u32 v67, v[vgprValuC+74], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+74], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+74], v67, v69, s28
v_lshrrev_b32 v[vgprValuC+74], 16, v[vgprValuC+74] // convert C to bf16
v_cmp_u_f32 s28, v[vgprValuC+75], v[vgprValuC+75]  // check Nan
v_bfe_u32 v67, v[vgprValuC+75], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+75], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+75], v67, v69, s28
v_and_or_b32 v74, v[vgprValuC+75], v68, v[vgprValuC+74] // pack two bf16 to dword
buffer_store_b32 v74, v106, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v107                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+76], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_and_b32 v64, v107, v68                           // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+77], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+76], v[vgprValuC+76]  // check Nan
v_bfe_u32 v67, v[vgprValuC+76], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+76], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+76], v67, v69, s28
v_lshrrev_b32 v[vgprValuC+76], 16, v[vgprValuC+76] // convert C to bf16
v_cmp_u_f32 s28, v[vgprValuC+77], v[vgprValuC+77]  // check Nan
v_bfe_u32 v67, v[vgprValuC+77], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+77], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+77], v67, v69, s28
v_and_or_b32 v76, v[vgprValuC+77], v68, v[vgprValuC+76] // pack two bf16 to dword
buffer_store_b32 v76, v108, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v109                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+78], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_and_b32 v64, v109, v68                           // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+79], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+78], v[vgprValuC+78]  // check Nan
v_bfe_u32 v67, v[vgprValuC+78], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+78], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+78], v67, v69, s28
v_lshrrev_b32 v[vgprValuC+78], 16, v[vgprValuC+78] // convert C to bf16
v_cmp_u_f32 s28, v[vgprValuC+79], v[vgprValuC+79]  // check Nan
v_bfe_u32 v67, v[vgprValuC+79], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+79], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+79], v67, v69, s28
v_and_or_b32 v78, v[vgprValuC+79], v68, v[vgprValuC+78] // pack two bf16 to dword
buffer_store_b32 v78, v110, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v111                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+80], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_and_b32 v64, v111, v68                           // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+81], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+80], v[vgprValuC+80]  // check Nan
v_bfe_u32 v67, v[vgprValuC+80], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+80], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+80], v67, v69, s28
v_lshrrev_b32 v[vgprValuC+80], 16, v[vgprValuC+80] // convert C to bf16
v_cmp_u_f32 s28, v[vgprValuC+81], v[vgprValuC+81]  // check Nan
v_bfe_u32 v67, v[vgprValuC+81], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+81], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+81], v67, v69, s28
v_and_or_b32 v80, v[vgprValuC+81], v68, v[vgprValuC+80] // pack two bf16 to dword
buffer_store_b32 v80, v112, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v113                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+82], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_and_b32 v64, v113, v68                           // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+83], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+82], v[vgprValuC+82]  // check Nan
v_bfe_u32 v67, v[vgprValuC+82], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+82], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+82], v67, v69, s28
v_lshrrev_b32 v[vgprValuC+82], 16, v[vgprValuC+82] // convert C to bf16
v_cmp_u_f32 s28, v[vgprValuC+83], v[vgprValuC+83]  // check Nan
v_bfe_u32 v67, v[vgprValuC+83], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+83], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+83], v67, v69, s28
v_and_or_b32 v82, v[vgprValuC+83], v68, v[vgprValuC+82] // pack two bf16 to dword
buffer_store_b32 v82, v114, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v115                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+84], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_and_b32 v64, v115, v68                           // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+85], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+84], v[vgprValuC+84]  // check Nan
v_bfe_u32 v67, v[vgprValuC+84], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+84], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+84], v67, v69, s28
v_lshrrev_b32 v[vgprValuC+84], 16, v[vgprValuC+84] // convert C to bf16
v_cmp_u_f32 s28, v[vgprValuC+85], v[vgprValuC+85]  // check Nan
v_bfe_u32 v67, v[vgprValuC+85], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+85], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+85], v67, v69, s28
v_and_or_b32 v84, v[vgprValuC+85], v68, v[vgprValuC+84] // pack two bf16 to dword
buffer_store_b32 v84, v116, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v117                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+86], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_and_b32 v64, v117, v68                           // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+87], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+86], v[vgprValuC+86]  // check Nan
v_bfe_u32 v67, v[vgprValuC+86], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+86], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+86], v67, v69, s28
v_lshrrev_b32 v[vgprValuC+86], 16, v[vgprValuC+86] // convert C to bf16
v_cmp_u_f32 s28, v[vgprValuC+87], v[vgprValuC+87]  // check Nan
v_bfe_u32 v67, v[vgprValuC+87], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+87], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+87], v67, v69, s28
v_and_or_b32 v86, v[vgprValuC+87], v68, v[vgprValuC+86] // pack two bf16 to dword
buffer_store_b32 v86, v118, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v119                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+88], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_and_b32 v64, v119, v68                           // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+89], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+88], v[vgprValuC+88]  // check Nan
v_bfe_u32 v67, v[vgprValuC+88], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+88], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+88], v67, v69, s28
v_lshrrev_b32 v[vgprValuC+88], 16, v[vgprValuC+88] // convert C to bf16
v_cmp_u_f32 s28, v[vgprValuC+89], v[vgprValuC+89]  // check Nan
v_bfe_u32 v67, v[vgprValuC+89], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+89], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+89], v67, v69, s28
v_and_or_b32 v88, v[vgprValuC+89], v68, v[vgprValuC+88] // pack two bf16 to dword
buffer_store_b32 v88, v120, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v121                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+90], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_and_b32 v64, v121, v68                           // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+91], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+90], v[vgprValuC+90]  // check Nan
v_bfe_u32 v67, v[vgprValuC+90], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+90], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+90], v67, v69, s28
v_lshrrev_b32 v[vgprValuC+90], 16, v[vgprValuC+90] // convert C to bf16
v_cmp_u_f32 s28, v[vgprValuC+91], v[vgprValuC+91]  // check Nan
v_bfe_u32 v67, v[vgprValuC+91], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+91], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+91], v67, v69, s28
v_and_or_b32 v90, v[vgprValuC+91], v68, v[vgprValuC+90] // pack two bf16 to dword
buffer_store_b32 v90, v122, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v123                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+92], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_and_b32 v64, v123, v68                           // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+93], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+92], v[vgprValuC+92]  // check Nan
v_bfe_u32 v67, v[vgprValuC+92], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+92], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+92], v67, v69, s28
v_lshrrev_b32 v[vgprValuC+92], 16, v[vgprValuC+92] // convert C to bf16
v_cmp_u_f32 s28, v[vgprValuC+93], v[vgprValuC+93]  // check Nan
v_bfe_u32 v67, v[vgprValuC+93], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+93], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+93], v67, v69, s28
v_and_or_b32 v92, v[vgprValuC+93], v68, v[vgprValuC+92] // pack two bf16 to dword
buffer_store_b32 v92, v124, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v125                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+94], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_and_b32 v64, v125, v68                           // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+95], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+94], v[vgprValuC+94]  // check Nan
v_bfe_u32 v67, v[vgprValuC+94], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+94], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+94], v67, v69, s28
v_lshrrev_b32 v[vgprValuC+94], 16, v[vgprValuC+94] // convert C to bf16
v_cmp_u_f32 s28, v[vgprValuC+95], v[vgprValuC+95]  // check Nan
v_bfe_u32 v67, v[vgprValuC+95], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+95], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+95], v67, v69, s28
v_and_or_b32 v94, v[vgprValuC+95], v68, v[vgprValuC+94] // pack two bf16 to dword
buffer_store_b32 v94, v126, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v127                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+96], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_and_b32 v64, v127, v68                           // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+97], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+96], v[vgprValuC+96]  // check Nan
v_bfe_u32 v67, v[vgprValuC+96], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+96], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+96], v67, v69, s28
v_lshrrev_b32 v[vgprValuC+96], 16, v[vgprValuC+96] // convert C to bf16
v_cmp_u_f32 s28, v[vgprValuC+97], v[vgprValuC+97]  // check Nan
v_bfe_u32 v67, v[vgprValuC+97], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+97], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+97], v67, v69, s28
v_and_or_b32 v96, v[vgprValuC+97], v68, v[vgprValuC+96] // pack two bf16 to dword
buffer_store_b32 v96, v128, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v129                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+98], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_and_b32 v64, v129, v68                           // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+99], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+98], v[vgprValuC+98]  // check Nan
v_bfe_u32 v67, v[vgprValuC+98], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+98], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+98], v67, v69, s28
v_lshrrev_b32 v[vgprValuC+98], 16, v[vgprValuC+98] // convert C to bf16
v_cmp_u_f32 s28, v[vgprValuC+99], v[vgprValuC+99]  // check Nan
v_bfe_u32 v67, v[vgprValuC+99], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+99], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+99], v67, v69, s28
v_and_or_b32 v98, v[vgprValuC+99], v68, v[vgprValuC+98] // pack two bf16 to dword
buffer_store_b32 v98, v130, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v131                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+100], v64, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_and_b32 v64, v131, v68                           // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+101], v64, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+100], v[vgprValuC+100] // check Nan
v_bfe_u32 v67, v[vgprValuC+100], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+100], v67, v70         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+100], v67, v69, s28
v_lshrrev_b32 v[vgprValuC+100], 16, v[vgprValuC+100] // convert C to bf16
v_cmp_u_f32 s28, v[vgprValuC+101], v[vgprValuC+101] // check Nan
v_bfe_u32 v67, v[vgprValuC+101], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+101], v67, v70         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+101], v67, v69, s28
v_and_or_b32 v100, v[vgprValuC+101], v68, v[vgprValuC+100] // pack two bf16 to dword
buffer_store_b32 v100, v132, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v133                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+102], v64, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_and_b32 v64, v133, v68                           // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+103], v64, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+102], v[vgprValuC+102] // check Nan
v_bfe_u32 v67, v[vgprValuC+102], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+102], v67, v70         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+102], v67, v69, s28
v_lshrrev_b32 v[vgprValuC+102], 16, v[vgprValuC+102] // convert C to bf16
v_cmp_u_f32 s28, v[vgprValuC+103], v[vgprValuC+103] // check Nan
v_bfe_u32 v67, v[vgprValuC+103], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+103], v67, v70         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+103], v67, v69, s28
v_and_or_b32 v102, v[vgprValuC+103], v68, v[vgprValuC+102] // pack two bf16 to dword
buffer_store_b32 v102, v134, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_FD0_VW2_GSU1_Else:
label_GW_B1_FD0_VW1_GSU1_Else:
label_GW_B1_FD0_VW1_GSU1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=60 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,1,0,0:vw1); (0,1,0,1:vw1); (0,2,0,0:vw1); (0,2,0,1:vw1); (0,3,0,0:vw1); (0,3,0,1:vw1); (0,4,0,0:vw1); (0,4,0,1:vw1); (0,5,0,0:vw1); (0,5,0,1:vw1); (0,6,0,0:vw1); (0,6,0,1:vw1); (0,7,0,0:vw1); (0,7,0,1:vw1); (0,0,1,0:vw1); (0,0,1,1:vw1); (0,1,1,0:vw1); (0,1,1,1:vw1); (0,2,1,0:vw1); (0,2,1,1:vw1); (0,3,1,0:vw1); (0,3,1,1:vw1); (0,4,1,0:vw1); (0,4,1,1:vw1); (0,5,1,0:vw1); (0,5,1,1:vw1); (0,6,1,0:vw1); (0,6,1,1:vw1); (0,7,1,0:vw1); (0,7,1,1:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v66, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s28, v60, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v104, v62, v60, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v104, v66, v104, s30                 // LDC clip if OOB. offset
buffer_load_d16_b16 v103, v104, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v104, v63, v60, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v104, v66, v104, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
v_add_co_u32 v64, vcc_lo, v60, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v106, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v106, v66, v106, s30                 // LDC clip if OOB. offset
buffer_load_d16_b16 v105, v106, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v106, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v106, v66, v106, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v64, vcc_lo, v60, 4                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v108, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v108, v66, v108, s30                 // LDC clip if OOB. offset
buffer_load_d16_b16 v107, v108, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v108, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v108, v66, v108, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,1) */
v_add_co_u32 v64, vcc_lo, v60, 5                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v110, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v110, v66, v110, s30                 // LDC clip if OOB. offset
buffer_load_d16_b16 v109, v110, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v110, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v110, v66, v110, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
v_add_co_u32 v64, vcc_lo, v60, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v112, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v112, v66, v112, s30                 // LDC clip if OOB. offset
buffer_load_d16_b16 v111, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v112, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v112, v66, v112, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,1) */
v_add_co_u32 v64, vcc_lo, v60, 9                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v114, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v114, v66, v114, s30                 // LDC clip if OOB. offset
buffer_load_d16_b16 v113, v114, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v114, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v114, v66, v114, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
v_add_co_u32 v64, vcc_lo, v60, 12                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v116, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v116, v66, v116, s30                 // LDC clip if OOB. offset
buffer_load_d16_b16 v115, v116, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v116, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v116, v66, v116, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,1) */
v_add_co_u32 v64, vcc_lo, v60, 13                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v118, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v118, v66, v118, s30                 // LDC clip if OOB. offset
buffer_load_d16_b16 v117, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v118, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v118, v66, v118, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
v_add_co_u32 v64, vcc_lo, v60, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v120, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v120, v66, v120, s30                 // LDC clip if OOB. offset
buffer_load_d16_b16 v119, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v120, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v120, v66, v120, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,1) */
v_add_co_u32 v64, vcc_lo, v60, 17                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v122, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v122, v66, v122, s30                 // LDC clip if OOB. offset
buffer_load_d16_b16 v121, v122, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v122, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v122, v66, v122, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
v_add_co_u32 v64, vcc_lo, v60, 20                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v124, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v124, v66, v124, s30                 // LDC clip if OOB. offset
buffer_load_d16_b16 v123, v124, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v124, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v124, v66, v124, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,1) */
v_add_co_u32 v64, vcc_lo, v60, 21                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v126, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v126, v66, v126, s30                 // LDC clip if OOB. offset
buffer_load_d16_b16 v125, v126, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v126, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v126, v66, v126, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
v_add_co_u32 v64, vcc_lo, v60, 24                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v128, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v128, v66, v128, s30                 // LDC clip if OOB. offset
buffer_load_d16_b16 v127, v128, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v128, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v128, v66, v128, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,1) */
v_add_co_u32 v64, vcc_lo, v60, 25                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v130, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v66, v130, s30                 // LDC clip if OOB. offset
buffer_load_d16_b16 v129, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v130, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v66, v130, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
v_add_co_u32 v64, vcc_lo, v60, 28                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v132, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v66, v132, s30                 // LDC clip if OOB. offset
buffer_load_d16_b16 v131, v132, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v132, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v66, v132, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,1) */
v_add_co_u32 v64, vcc_lo, v60, 29                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v134, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v66, v134, s30                 // LDC clip if OOB. offset
buffer_load_d16_b16 v133, v134, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v134, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v66, v134, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v61, vcc_lo, v61, 1                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v62, v62, s[sgprStrideC1J]            // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v63, v63, s[sgprStrideD1J]            // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v60, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v136, v62, v60, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v66, v136, s30                 // LDC clip if OOB. offset
buffer_load_d16_b16 v135, v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v136, v63, v60, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v66, v136, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,1) */
v_add_co_u32 v64, vcc_lo, v60, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v138, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v66, v138, s30                 // LDC clip if OOB. offset
buffer_load_d16_b16 v137, v138, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v138, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v66, v138, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
v_add_co_u32 v64, vcc_lo, v60, 4                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v140, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v66, v140, s30                 // LDC clip if OOB. offset
buffer_load_d16_b16 v139, v140, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v140, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v66, v140, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,1) */
v_add_co_u32 v64, vcc_lo, v60, 5                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v142, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v66, v142, s30                 // LDC clip if OOB. offset
buffer_load_d16_b16 v141, v142, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v142, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v66, v142, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,2,0) */
v_add_co_u32 v64, vcc_lo, v60, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v144, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v144, v66, v144, s30                 // LDC clip if OOB. offset
buffer_load_d16_b16 v143, v144, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v144, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v144, v66, v144, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,2,1) */
v_add_co_u32 v64, vcc_lo, v60, 9                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v146, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v146, v66, v146, s30                 // LDC clip if OOB. offset
buffer_load_d16_b16 v145, v146, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v146, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v146, v66, v146, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,3,0) */
v_add_co_u32 v64, vcc_lo, v60, 12                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v148, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v148, v66, v148, s30                 // LDC clip if OOB. offset
buffer_load_d16_b16 v147, v148, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v148, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v148, v66, v148, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,3,1) */
v_add_co_u32 v64, vcc_lo, v60, 13                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v150, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v150, v66, v150, s30                 // LDC clip if OOB. offset
buffer_load_d16_b16 v149, v150, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v150, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v150, v66, v150, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,4,0) */
v_add_co_u32 v64, vcc_lo, v60, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v152, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v152, v66, v152, s30                 // LDC clip if OOB. offset
buffer_load_d16_b16 v151, v152, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v152, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v152, v66, v152, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,4,1) */
v_add_co_u32 v64, vcc_lo, v60, 17                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v154, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v66, v154, s30                 // LDC clip if OOB. offset
buffer_load_d16_b16 v153, v154, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v154, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v66, v154, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,5,0) */
v_add_co_u32 v64, vcc_lo, v60, 20                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v156, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v66, v156, s30                 // LDC clip if OOB. offset
buffer_load_d16_b16 v155, v156, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v156, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v66, v156, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,5,1) */
v_add_co_u32 v64, vcc_lo, v60, 21                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v158, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v66, v158, s30                 // LDC clip if OOB. offset
buffer_load_d16_b16 v157, v158, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v158, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v66, v158, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,6,0) */
v_add_co_u32 v64, vcc_lo, v60, 24                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v160, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v160, v66, v160, s30                 // LDC clip if OOB. offset
buffer_load_d16_b16 v159, v160, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v160, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v160, v66, v160, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,6,1) */
v_add_co_u32 v64, vcc_lo, v60, 25                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v162, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v162, v66, v162, s30                 // LDC clip if OOB. offset
buffer_load_d16_b16 v161, v162, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v162, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v162, v66, v162, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,7,0) */
v_add_co_u32 v64, vcc_lo, v60, 28                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v164, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v164, v66, v164, s30                 // LDC clip if OOB. offset
buffer_load_d16_b16 v163, v164, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v164, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v164, v66, v164, s30                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,7,1) */
v_add_co_u32 v64, vcc_lo, v60, 29                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v166, v62, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v166, v66, v166, s30                 // LDC clip if OOB. offset
buffer_load_d16_b16 v165, v166, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v166, v63, v64, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v166, v66, v166, s30                 // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 1, 0, 0), (0, 1, 0, 1), (0, 2, 0, 0), (0, 2, 0, 1), (0, 3, 0, 0), (0, 3, 0, 1), (0, 4, 0, 0), (0, 4, 0, 1), (0, 5, 0, 0), (0, 5, 0, 1), (0, 6, 0, 0), (0, 6, 0, 1), (0, 7, 0, 0), (0, 7, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1), (0, 1, 1, 0), (0, 1, 1, 1), (0, 2, 1, 0), (0, 2, 1, 1), (0, 3, 1, 0), (0, 3, 1, 1), (0, 4, 1, 0), (0, 4, 1, 1), (0, 5, 1, 0), (0, 5, 1, 1), (0, 6, 1, 0), (0, 6, 1, 1), (0, 7, 1, 0), (0, 7, 1, 1)] */
v_mul_f32 v[vgprValuC+71], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+72], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+73], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+74], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+75], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+76], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+77], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+78], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+79], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+80], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+81], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+82], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+83], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+84], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+85], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+86], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+87], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+88], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+89], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+90], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+91], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+92], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+93], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+94], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+95], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+96], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+97], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+98], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+100], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+102], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_mov_b32 v68, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v69, 0x7fff0000                          // fp32 Nan
v_mov_b32 v70, 0x7fff                              // rounding bias for bfloat16
v_lshlrev_b32 v64, 16, v103                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+71], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+71], v[vgprValuC+71]  // check Nan
v_bfe_u32 v67, v[vgprValuC+71], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+71], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+71], v67, v69, s28
v_lshrrev_b32 v71, 16, v[vgprValuC+71]             // convert C to bf16
buffer_store_b16 v71, v104, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v105                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+72], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+72], v[vgprValuC+72]  // check Nan
v_bfe_u32 v67, v[vgprValuC+72], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+72], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+72], v67, v69, s28
v_lshrrev_b32 v72, 16, v[vgprValuC+72]             // convert C to bf16
buffer_store_b16 v72, v106, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v107                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+73], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+73], v[vgprValuC+73]  // check Nan
v_bfe_u32 v67, v[vgprValuC+73], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+73], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+73], v67, v69, s28
v_lshrrev_b32 v73, 16, v[vgprValuC+73]             // convert C to bf16
buffer_store_b16 v73, v108, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v109                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+74], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+74], v[vgprValuC+74]  // check Nan
v_bfe_u32 v67, v[vgprValuC+74], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+74], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+74], v67, v69, s28
v_lshrrev_b32 v74, 16, v[vgprValuC+74]             // convert C to bf16
buffer_store_b16 v74, v110, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v111                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+75], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+75], v[vgprValuC+75]  // check Nan
v_bfe_u32 v67, v[vgprValuC+75], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+75], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+75], v67, v69, s28
v_lshrrev_b32 v75, 16, v[vgprValuC+75]             // convert C to bf16
buffer_store_b16 v75, v112, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v113                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+76], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+76], v[vgprValuC+76]  // check Nan
v_bfe_u32 v67, v[vgprValuC+76], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+76], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+76], v67, v69, s28
v_lshrrev_b32 v76, 16, v[vgprValuC+76]             // convert C to bf16
buffer_store_b16 v76, v114, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v115                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+77], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+77], v[vgprValuC+77]  // check Nan
v_bfe_u32 v67, v[vgprValuC+77], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+77], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+77], v67, v69, s28
v_lshrrev_b32 v77, 16, v[vgprValuC+77]             // convert C to bf16
buffer_store_b16 v77, v116, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v117                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+78], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+78], v[vgprValuC+78]  // check Nan
v_bfe_u32 v67, v[vgprValuC+78], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+78], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+78], v67, v69, s28
v_lshrrev_b32 v78, 16, v[vgprValuC+78]             // convert C to bf16
buffer_store_b16 v78, v118, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v119                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+79], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+79], v[vgprValuC+79]  // check Nan
v_bfe_u32 v67, v[vgprValuC+79], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+79], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+79], v67, v69, s28
v_lshrrev_b32 v79, 16, v[vgprValuC+79]             // convert C to bf16
buffer_store_b16 v79, v120, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v121                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+80], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+80], v[vgprValuC+80]  // check Nan
v_bfe_u32 v67, v[vgprValuC+80], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+80], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+80], v67, v69, s28
v_lshrrev_b32 v80, 16, v[vgprValuC+80]             // convert C to bf16
buffer_store_b16 v80, v122, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v123                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+81], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+81], v[vgprValuC+81]  // check Nan
v_bfe_u32 v67, v[vgprValuC+81], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+81], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+81], v67, v69, s28
v_lshrrev_b32 v81, 16, v[vgprValuC+81]             // convert C to bf16
buffer_store_b16 v81, v124, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v125                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+82], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+82], v[vgprValuC+82]  // check Nan
v_bfe_u32 v67, v[vgprValuC+82], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+82], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+82], v67, v69, s28
v_lshrrev_b32 v82, 16, v[vgprValuC+82]             // convert C to bf16
buffer_store_b16 v82, v126, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v127                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+83], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+83], v[vgprValuC+83]  // check Nan
v_bfe_u32 v67, v[vgprValuC+83], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+83], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+83], v67, v69, s28
v_lshrrev_b32 v83, 16, v[vgprValuC+83]             // convert C to bf16
buffer_store_b16 v83, v128, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v129                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+84], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+84], v[vgprValuC+84]  // check Nan
v_bfe_u32 v67, v[vgprValuC+84], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+84], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+84], v67, v69, s28
v_lshrrev_b32 v84, 16, v[vgprValuC+84]             // convert C to bf16
buffer_store_b16 v84, v130, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v131                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+85], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+85], v[vgprValuC+85]  // check Nan
v_bfe_u32 v67, v[vgprValuC+85], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+85], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+85], v67, v69, s28
v_lshrrev_b32 v85, 16, v[vgprValuC+85]             // convert C to bf16
buffer_store_b16 v85, v132, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v133                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+86], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+86], v[vgprValuC+86]  // check Nan
v_bfe_u32 v67, v[vgprValuC+86], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+86], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+86], v67, v69, s28
v_lshrrev_b32 v86, 16, v[vgprValuC+86]             // convert C to bf16
buffer_store_b16 v86, v134, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v135                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+87], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+87], v[vgprValuC+87]  // check Nan
v_bfe_u32 v67, v[vgprValuC+87], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+87], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+87], v67, v69, s28
v_lshrrev_b32 v87, 16, v[vgprValuC+87]             // convert C to bf16
buffer_store_b16 v87, v136, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v137                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+88], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+88], v[vgprValuC+88]  // check Nan
v_bfe_u32 v67, v[vgprValuC+88], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+88], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+88], v67, v69, s28
v_lshrrev_b32 v88, 16, v[vgprValuC+88]             // convert C to bf16
buffer_store_b16 v88, v138, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v139                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+89], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+89], v[vgprValuC+89]  // check Nan
v_bfe_u32 v67, v[vgprValuC+89], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+89], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+89], v67, v69, s28
v_lshrrev_b32 v89, 16, v[vgprValuC+89]             // convert C to bf16
buffer_store_b16 v89, v140, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v141                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+90], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+90], v[vgprValuC+90]  // check Nan
v_bfe_u32 v67, v[vgprValuC+90], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+90], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+90], v67, v69, s28
v_lshrrev_b32 v90, 16, v[vgprValuC+90]             // convert C to bf16
buffer_store_b16 v90, v142, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v143                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+91], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+91], v[vgprValuC+91]  // check Nan
v_bfe_u32 v67, v[vgprValuC+91], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+91], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+91], v67, v69, s28
v_lshrrev_b32 v91, 16, v[vgprValuC+91]             // convert C to bf16
buffer_store_b16 v91, v144, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v145                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+92], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+92], v[vgprValuC+92]  // check Nan
v_bfe_u32 v67, v[vgprValuC+92], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+92], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+92], v67, v69, s28
v_lshrrev_b32 v92, 16, v[vgprValuC+92]             // convert C to bf16
buffer_store_b16 v92, v146, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v147                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+93], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+93], v[vgprValuC+93]  // check Nan
v_bfe_u32 v67, v[vgprValuC+93], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+93], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+93], v67, v69, s28
v_lshrrev_b32 v93, 16, v[vgprValuC+93]             // convert C to bf16
buffer_store_b16 v93, v148, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v149                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+94], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+94], v[vgprValuC+94]  // check Nan
v_bfe_u32 v67, v[vgprValuC+94], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+94], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+94], v67, v69, s28
v_lshrrev_b32 v94, 16, v[vgprValuC+94]             // convert C to bf16
buffer_store_b16 v94, v150, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v151                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+95], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+95], v[vgprValuC+95]  // check Nan
v_bfe_u32 v67, v[vgprValuC+95], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+95], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+95], v67, v69, s28
v_lshrrev_b32 v95, 16, v[vgprValuC+95]             // convert C to bf16
buffer_store_b16 v95, v152, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v153                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+96], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+96], v[vgprValuC+96]  // check Nan
v_bfe_u32 v67, v[vgprValuC+96], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+96], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+96], v67, v69, s28
v_lshrrev_b32 v96, 16, v[vgprValuC+96]             // convert C to bf16
buffer_store_b16 v96, v154, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v155                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+97], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+97], v[vgprValuC+97]  // check Nan
v_bfe_u32 v67, v[vgprValuC+97], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+97], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+97], v67, v69, s28
v_lshrrev_b32 v97, 16, v[vgprValuC+97]             // convert C to bf16
buffer_store_b16 v97, v156, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v157                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+98], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+98], v[vgprValuC+98]  // check Nan
v_bfe_u32 v67, v[vgprValuC+98], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+98], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+98], v67, v69, s28
v_lshrrev_b32 v98, 16, v[vgprValuC+98]             // convert C to bf16
buffer_store_b16 v98, v158, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v159                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+99], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+99], v[vgprValuC+99]  // check Nan
v_bfe_u32 v67, v[vgprValuC+99], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+99], v67, v70          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+99], v67, v69, s28
v_lshrrev_b32 v99, 16, v[vgprValuC+99]             // convert C to bf16
buffer_store_b16 v99, v160, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v161                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+100], v64, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+100], v[vgprValuC+100] // check Nan
v_bfe_u32 v67, v[vgprValuC+100], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+100], v67, v70         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+100], v67, v69, s28
v_lshrrev_b32 v100, 16, v[vgprValuC+100]           // convert C to bf16
buffer_store_b16 v100, v162, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v163                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+101], v64, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+101], v[vgprValuC+101] // check Nan
v_bfe_u32 v67, v[vgprValuC+101], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+101], v67, v70         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+101], v67, v69, s28
v_lshrrev_b32 v101, 16, v[vgprValuC+101]           // convert C to bf16
buffer_store_b16 v101, v164, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v64, 16, v165                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+102], v64, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s28, v[vgprValuC+102], v[vgprValuC+102] // check Nan
v_bfe_u32 v67, v[vgprValuC+102], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v67, v[vgprValuC+102], v67, v70         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+102], v67, v69, s28
v_lshrrev_b32 v102, 16, v[vgprValuC+102]           // convert C to bf16
buffer_store_b16 v102, v166, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_End_1:
label_KernelEnd:
s_endpgm                                           // Kernel End
label_ASM_End:  /// The end of the kernel
