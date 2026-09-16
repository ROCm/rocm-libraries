
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
.text
.protected Custom_Cijk_Alik_Bljk_I4H_HHS_BH_SABB128ZPU8X_UserArgs_MT32x32x128_MI16x16x1_gfx1151
.globl Custom_Cijk_Alik_Bljk_I4H_HHS_BH_SABB128ZPU8X_UserArgs_MT32x32x128_MI16x16x1_gfx1151
.p2align 8
.type Custom_Cijk_Alik_Bljk_I4H_HHS_BH_SABB128ZPU8X_UserArgs_MT32x32x128_MI16x16x1_gfx1151,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Custom_Cijk_Alik_Bljk_I4H_HHS_BH_SABB128ZPU8X_UserArgs_MT32x32x128_MI16x16x1_gfx1151
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 256 // vgprs
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
/* Num VGPR   =256 */
/* Num AccVGPR=0 */
/* Num SGPR   =86 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 8 x 1 */
/* SubGroup= 4 x 32 */
/* VectorWidthA=1 */
/* VectorWidthB=1 */
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
  - .name: Custom_Cijk_Alik_Bljk_I4H_HHS_BH_SABB128ZPU8X_UserArgs_MT32x32x128_MI16x16x1_gfx1151
    .symbol: 'Custom_Cijk_Alik_Bljk_I4H_HHS_BH_SABB128ZPU8X_UserArgs_MT32x32x128_MI16x16x1_gfx1151.kd'
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
    .group_segment_fixed_size:   51200
    .kernarg_segment_align:      8
    .kernarg_segment_size:       160
    .max_flat_workgroup_size:    128
    .private_segment_fixed_size: 0
    .sgpr_count:                 86
    .sgpr_spill_count:           0
    .vgpr_count:                 256
    .vgpr_spill_count:           0
    .wavefront_size:             32
...
.end_amdgpu_metadata
Custom_Cijk_Alik_Bljk_I4H_HHS_BH_SABB128ZPU8X_UserArgs_MT32x32x128_MI16x16x1_gfx1151:
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
/* ValuC range: [0-8), serializedStore enabled */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprBase, 36
.set vgprLocalWriteAddrA, 32
.set vgprLocalWriteAddrB, 33
.set vgprGlobalReadOffsetA, 8
.set vgprGlobalReadOffsetB, 12
.set vgprGlobalReadOffsetScaleA, 16
.set vgprG2LScaleA, 20
.set vgprG2LScaleZeroA, 24
.set vgprGlobalReadOffsetScaleZeroA, 28
.set vgprLocalReadAddrA, 34
.set vgprLocalReadAddrB, 35
.set vgprSerial, 198

/******************************************/
/* VGPR Macro Assignments                 */
/******************************************/
.set vgprValuA_X0_I0_BASE, vgprBase+0
.set vgprValuB_X0_I0_BASE, vgprBase+65
.set vgprG2LA_BASE, vgprBase+130
.set vgprG2LB_BASE, vgprBase+146
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+8
.set vgprValuA_X2_I0, vgprValuA_X0_I0_BASE+16
.set vgprValuA_X3_I0, vgprValuA_X0_I0_BASE+24
.set vgprValuA_X4_I0, vgprValuA_X0_I0_BASE+32
.set vgprValuA_X5_I0, vgprValuA_X0_I0_BASE+40
.set vgprValuA_X6_I0, vgprValuA_X0_I0_BASE+48
.set vgprValuA_X7_I0, vgprValuA_X0_I0_BASE+56
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X1_I0, vgprValuB_X0_I0_BASE+8
.set vgprValuB_X2_I0, vgprValuB_X0_I0_BASE+16
.set vgprValuB_X3_I0, vgprValuB_X0_I0_BASE+24
.set vgprValuB_X4_I0, vgprValuB_X0_I0_BASE+32
.set vgprValuB_X5_I0, vgprValuB_X0_I0_BASE+40
.set vgprValuB_X6_I0, vgprValuB_X0_I0_BASE+48
.set vgprValuB_X7_I0, vgprValuB_X0_I0_BASE+56
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

.set MT0, 32
.set MT1, 32
.set DepthU, 128
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
v_cvt_f64_u32 v[36:37], s69                        // s65 = s[sgprWorkGroup0] / s69
v_rcp_f64 v[36:37], v[36:37]                       // s65 = s[sgprWorkGroup0] / s69
v_cvt_f64_u32 v[38:39], s[sgprWorkGroup0]          // s65 = s[sgprWorkGroup0] / s69
v_mul_f64 v[36:37], v[36:37], v[38:39]             // s65 = s[sgprWorkGroup0] / s69
v_cvt_u32_f64 v36, v[36:37]                        // s65 = s[sgprWorkGroup0] / s69
v_mul_lo_u32 v37, v36, s69                         // s65 = s[sgprWorkGroup0] / s69
v_sub_nc_u32 v38, s[sgprWorkGroup0], v37           // s65 = s[sgprWorkGroup0] / s69
v_cmp_ge_u32 vcc_lo, v38, s69                      // s65 = s[sgprWorkGroup0] / s69
s_mov_b32 exec_lo, vcc_lo                          // s65 = s[sgprWorkGroup0] / s69
v_add_nc_u32 v36, v36, 1                           // s65 = s[sgprWorkGroup0] / s69
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v37, v36, s69                         // s65 = s[sgprWorkGroup0] / s69
v_sub_nc_u32 v38, s[sgprWorkGroup0], v37           // s65 = s[sgprWorkGroup0] / s69
v_readfirstlane_b32 s65, v36                       // quotient
v_readfirstlane_b32 s66, v38                       // remainder
s_mul_i32 s65, s65, s69
/* temp1 = (wg%CU_Count)//WGMXCC */
s_lshr_b32 s66, s66, s68
/* temp0 = temp0 + temp1 */
s_add_u32 s65, s65, s66
/* temp1 = (wg%WGMXCC) * ((WGs - (WGs//CU_Count) * CU_Count) if (wg > (WGs//CU_Count) * CU_Count) else CU_Count)//WGMXCC */
v_cvt_f64_u32 v[36:37], s69                        // s66 = s23 / s69
v_rcp_f64 v[36:37], v[36:37]                       // s66 = s23 / s69
v_cvt_f64_u32 v[38:39], s23                        // s66 = s23 / s69
v_mul_f64 v[36:37], v[36:37], v[38:39]             // s66 = s23 / s69
v_cvt_u32_f64 v36, v[36:37]                        // s66 = s23 / s69
v_mul_lo_u32 v37, v36, s69                         // s66 = s23 / s69
v_sub_nc_u32 v38, s23, v37                         // s66 = s23 / s69
v_cmp_ge_u32 vcc_lo, v38, s69                      // s66 = s23 / s69
s_mov_b32 exec_lo, vcc_lo                          // s66 = s23 / s69
v_add_nc_u32 v36, v36, 1                           // s66 = s23 / s69
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s66, v36                       // quotient
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
/* init: add vgpr [36...201) to pool */
/* init: add vgpr [0...8) to pool */
/* init: add agpr [0...0) to pool */

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
/* lr0I */
v_and_b32 v1, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
v_and_b32 v0, 15, v1                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v0, 7, v0                            // 1. N offset: nOffset = nIdx * nStride(128)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v4, 5, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(32)
v_and_b32 v4, 1, v4                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v0, v4, 11, v0                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(2048); 7. final local read offset: flrOffset = lrOffset + WOffset
/* lr1J */
v_and_b32 v2, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v1, 7, v1                            // 1. N offset: nOffset = nIdx * nStride(128)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v3, 6, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
v_and_b32 v3, 1, v3                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v1, v3, 11, v1                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(2048); 7. final local read offset: flrOffset = lrOffset + WOffset

/* local read addresses: final offsets a */
v_lshrrev_b32 v2, 5, v[vgprSerial]                 // 2 = Serial / 32
v_lshrrev_b32 v2, 2, v2                            // LSU offset: Get LSU wave_id
s_mov_b32 s16, 128                                 // LSU offset: stride = lsuStride(128) when umlds==True
v_mul_lo_u32 v2, s16, v2                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_nc_u32 v[vgprLocalReadAddrA], v2, v0         // Final Offset: offset = (lro0+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrA], 1, v[vgprLocalReadAddrA] //  (multiple bpe)
v_lshrrev_b32 v3, 8, v[vgprLocalReadAddrA]         // Final Offset: padding 32 per block 256
v_lshl_add_u32 v[vgprLocalReadAddrA], v3, 5, v[vgprLocalReadAddrA] // Final Offset: padding 32 per block 256

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 5, v[vgprSerial]                 // 0 = Serial / 32
v_lshrrev_b32 v0, 2, v0                            // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(128) when umlds==True (dup assign opt.)
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
/* LVCA = 16 */
/* v1 = A-unroll = serial%LVCA */
v_lshrrev_b32 v0, 4, v[vgprSerial]                 // 0 = Serial / 16
v_and_b32 v1, 15, v[vgprSerial]                    // 1 = Serial % 16
/* unroll *= glvw */
v_lshlrev_b32 v1, 3, v1                            // v1 = v1 * 8
v_mov_b32 v4, v1                                   // copy for GlobalSplitU
/* LVCB = 16 */
/* v3 = B-unroll = serial%LVCB */
v_lshrrev_b32 v2, 4, v[vgprSerial]                 // 2 = Serial / 16
v_and_b32 v3, 15, v[vgprSerial]                    // 3 = Serial % 16
/* unroll *= glvw */
v_lshlrev_b32 v3, 3, v3                            // v3 = v3 * 8
v_mov_b32 v5, v3                                   // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v4 */
/* lwaUnrollAssignmentB = v5 */

/* local write addresses: first offset a */
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x80, v0     // lwAL**(DepthU_Compute + PAD)
v_add_nc_u32 v[vgprLocalWriteAddrA], v4, v[vgprLocalWriteAddrA] // lwFOA = (lwAA + lwAL*(DepthU+PAD))
v_lshlrev_b32 v[vgprLocalWriteAddrA], 1, v[vgprLocalWriteAddrA] //  (multiple bpe)
v_lshrrev_b32 v6, 8, v[vgprLocalWriteAddrA]        // padding 32 per block 256
v_lshl_add_u32 v[vgprLocalWriteAddrA], v6, 5, v[vgprLocalWriteAddrA] // padding 32 per block 256

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x80, v2     // lwBL**(DepthU_Compute + PAD)
v_add_nc_u32 v[vgprLocalWriteAddrB], v5, v[vgprLocalWriteAddrB] // lwFOB = (lwBB + lwBL*(DepthU+PAD))
v_lshlrev_b32 v[vgprLocalWriteAddrB], 1, v[vgprLocalWriteAddrB] //  (multiple bpe)
v_lshrrev_b32 v6, 8, v[vgprLocalWriteAddrB]        // padding 32 per block 256
v_lshl_add_u32 v[vgprLocalWriteAddrB], v6, 5, v[vgprLocalWriteAddrB] // padding 32 per block 256
v_add_co_u32 v[vgprLocalWriteAddrB], vcc_lo, 0x2400, v[vgprLocalWriteAddrB] // lwFOB = lw1J + lwL*MT1J + LDS_OFFSET_B=9216
s_waitcnt lgkmcnt(0)                               // wait for 88/0 bytes of kern args over preload
v_mov_b32 v38, MT0                                 // set MT0 into sgpr
v_mov_b32 v37, s[sgprSizesFree+0]                  // set Free0 size
v_cvt_f32_u32 v36, v38                             // v36 = ceil(v37 / v38)
v_rcp_iflag_f32 v36, v36                           // v36 = ceil(v37 / v38)
v_cvt_f32_u32 v39, v37                             // v36 = ceil(v37 / v38)
v_mul_f32 v36, v36, v39                            // v36 = ceil(v37 / v38)
v_cvt_u32_f32 v36, v36                             // v36 = ceil(v37 / v38)
v_mul_u32_u24 v39, v36, v38                        // v36 = ceil(v37 / v38)
v_sub_nc_u32 v39, v37, v39                         // v36 = ceil(v37 / v38)
v_cmp_ne_u32 vcc_lo, v39, 0                        // v36 = ceil(v37 / v38)
v_add_co_ci_u32 v36, vcc_lo, v36, 0, vcc_lo        // ceil
v_mov_b32 v38, MT1                                 // set MT1 into sgpr
v_mov_b32 v37, s[sgprSizesFree+1]                  // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v36     // set back to numWorkGroup0
v_cvt_f32_u32 v36, v38                             // v36 = ceil(v37 / v38)
v_rcp_iflag_f32 v36, v36                           // v36 = ceil(v37 / v38)
v_cvt_f32_u32 v39, v37                             // v36 = ceil(v37 / v38)
v_mul_f32 v36, v36, v39                            // v36 = ceil(v37 / v38)
v_cvt_u32_f32 v36, v36                             // v36 = ceil(v37 / v38)
v_mul_u32_u24 v39, v36, v38                        // v36 = ceil(v37 / v38)
v_sub_nc_u32 v39, v37, v39                         // v36 = ceil(v37 / v38)
v_cmp_ne_u32 vcc_lo, v39, 0                        // v36 = ceil(v37 / v38)
v_add_co_ci_u32 v36, vcc_lo, v36, 0, vcc_lo        // ceil
v_readfirstlane_b32 s[sgprNumWorkGroups1], v36     // set back to numWorkGroup1

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
s_lshr_b32 s68, s24, 5                             // s68 = s24 / 32
s_and_b32 s66, 31, s24                             // s66 = s24 % 32
s_addc_u32 s68, s68, 0
s_lshr_b32 s69, s25, 5                             // s69 = s25 / 32
s_and_b32 s66, 31, s25                             // s66 = s25 % 32
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
s_lshr_b32 s68, s24, 5                             // s68 = s24 / 32
s_and_b32 s66, 31, s24                             // s66 = s24 % 32
s_addc_u32 s68, s68, 0
s_lshr_b32 s69, s25, 5                             // s69 = s25 / 32
s_and_b32 s66, 31, s25                             // s66 = s25 % 32
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
/* init: add vgpr [36...201) to pool */
/* init: add vgpr [0...8) to pool */
/* init: add agpr [0...0) to pool */

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
/* lr0I */
v_and_b32 v1, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
v_and_b32 v0, 15, v1                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v0, 7, v0                            // 1. N offset: nOffset = nIdx * nStride(128)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v4, 5, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(32)
v_and_b32 v4, 1, v4                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v0, v4, 11, v0                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(2048); 7. final local read offset: flrOffset = lrOffset + WOffset
/* lr1J */
v_and_b32 v2, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v1, 7, v1                            // 1. N offset: nOffset = nIdx * nStride(128)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v3, 6, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
v_and_b32 v3, 1, v3                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v1, v3, 11, v1                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(2048); 7. final local read offset: flrOffset = lrOffset + WOffset

/* local read addresses: final offsets a */
v_lshrrev_b32 v2, 5, v[vgprSerial]                 // 2 = Serial / 32
v_lshrrev_b32 v2, 2, v2                            // LSU offset: Get LSU wave_id
s_mov_b32 s16, 128                                 // LSU offset: stride = lsuStride(128) when umlds==True
v_mul_lo_u32 v2, s16, v2                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_nc_u32 v[vgprLocalReadAddrA], v2, v0         // Final Offset: offset = (lro0+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrA], 1, v[vgprLocalReadAddrA] //  (multiple bpe)
v_lshrrev_b32 v3, 8, v[vgprLocalReadAddrA]         // Final Offset: padding 32 per block 256
v_lshl_add_u32 v[vgprLocalReadAddrA], v3, 5, v[vgprLocalReadAddrA] // Final Offset: padding 32 per block 256

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 5, v[vgprSerial]                 // 0 = Serial / 32
v_lshrrev_b32 v0, 2, v0                            // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(128) when umlds==True (dup assign opt.)
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
/* LVCA = 16 */
/* v1 = A-unroll = serial%LVCA */
v_lshrrev_b32 v0, 4, v[vgprSerial]                 // 0 = Serial / 16
v_and_b32 v1, 15, v[vgprSerial]                    // 1 = Serial % 16
/* unroll *= glvw */
v_lshlrev_b32 v1, 3, v1                            // v1 = v1 * 8
v_mov_b32 v4, v1                                   // copy for GlobalSplitU
/* LVCB = 16 */
/* v3 = B-unroll = serial%LVCB */
v_lshrrev_b32 v2, 4, v[vgprSerial]                 // 2 = Serial / 16
v_and_b32 v3, 15, v[vgprSerial]                    // 3 = Serial % 16
/* unroll *= glvw */
v_lshlrev_b32 v3, 3, v3                            // v3 = v3 * 8
v_mov_b32 v5, v3                                   // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v4 */
/* lwaUnrollAssignmentB = v5 */

/* local write addresses: first offset a */
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x80, v0     // lwAL**(DepthU_Compute + PAD)
v_add_nc_u32 v[vgprLocalWriteAddrA], v4, v[vgprLocalWriteAddrA] // lwFOA = (lwAA + lwAL*(DepthU+PAD))
v_lshlrev_b32 v[vgprLocalWriteAddrA], 1, v[vgprLocalWriteAddrA] //  (multiple bpe)
v_lshrrev_b32 v6, 8, v[vgprLocalWriteAddrA]        // padding 32 per block 256
v_lshl_add_u32 v[vgprLocalWriteAddrA], v6, 5, v[vgprLocalWriteAddrA] // padding 32 per block 256

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x80, v2     // lwBL**(DepthU_Compute + PAD)
v_add_nc_u32 v[vgprLocalWriteAddrB], v5, v[vgprLocalWriteAddrB] // lwFOB = (lwBB + lwBL*(DepthU+PAD))
v_lshlrev_b32 v[vgprLocalWriteAddrB], 1, v[vgprLocalWriteAddrB] //  (multiple bpe)
v_lshrrev_b32 v6, 8, v[vgprLocalWriteAddrB]        // padding 32 per block 256
v_lshl_add_u32 v[vgprLocalWriteAddrB], v6, 5, v[vgprLocalWriteAddrB] // padding 32 per block 256
v_add_co_u32 v[vgprLocalWriteAddrB], vcc_lo, 0x2400, v[vgprLocalWriteAddrB] // lwFOB = lw1J + lwL*MT1J + LDS_OFFSET_B=9216
s_waitcnt lgkmcnt(0)                               // wait for 88/0 bytes of kern args over preload
v_mov_b32 v38, MT0                                 // set MT0 into sgpr
v_mov_b32 v37, s[sgprSizesFree+0]                  // set Free0 size
v_cvt_f32_u32 v36, v38                             // v36 = ceil(v37 / v38)
v_rcp_iflag_f32 v36, v36                           // v36 = ceil(v37 / v38)
v_cvt_f32_u32 v39, v37                             // v36 = ceil(v37 / v38)
v_mul_f32 v36, v36, v39                            // v36 = ceil(v37 / v38)
v_cvt_u32_f32 v36, v36                             // v36 = ceil(v37 / v38)
v_mul_u32_u24 v39, v36, v38                        // v36 = ceil(v37 / v38)
v_sub_nc_u32 v39, v37, v39                         // v36 = ceil(v37 / v38)
v_cmp_ne_u32 vcc_lo, v39, 0                        // v36 = ceil(v37 / v38)
v_add_co_ci_u32 v36, vcc_lo, v36, 0, vcc_lo        // ceil
v_mov_b32 v38, MT1                                 // set MT1 into sgpr
v_mov_b32 v37, s[sgprSizesFree+1]                  // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v36     // set back to numWorkGroup0
v_cvt_f32_u32 v36, v38                             // v36 = ceil(v37 / v38)
v_rcp_iflag_f32 v36, v36                           // v36 = ceil(v37 / v38)
v_cvt_f32_u32 v39, v37                             // v36 = ceil(v37 / v38)
v_mul_f32 v36, v36, v39                            // v36 = ceil(v37 / v38)
v_cvt_u32_f32 v36, v36                             // v36 = ceil(v37 / v38)
v_mul_u32_u24 v39, v36, v38                        // v36 = ceil(v37 / v38)
v_sub_nc_u32 v39, v37, v39                         // v36 = ceil(v37 / v38)
v_cmp_ne_u32 vcc_lo, v39, 0                        // v36 = ceil(v37 / v38)
v_add_co_ci_u32 v36, vcc_lo, v36, 0, vcc_lo        // ceil
v_readfirstlane_b32 s[sgprNumWorkGroups1], v36     // set back to numWorkGroup1

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
v_cvt_f64_u32 v[36:37], s16                        // s17 = s[sgprWorkGroup0] / s16
v_rcp_f64 v[36:37], v[36:37]                       // s17 = s[sgprWorkGroup0] / s16
v_cvt_f64_u32 v[38:39], s[sgprWorkGroup0]          // s17 = s[sgprWorkGroup0] / s16
v_mul_f64 v[36:37], v[36:37], v[38:39]             // s17 = s[sgprWorkGroup0] / s16
v_cvt_u32_f64 v36, v[36:37]                        // s17 = s[sgprWorkGroup0] / s16
v_mul_lo_u32 v37, v36, s16                         // s17 = s[sgprWorkGroup0] / s16
v_sub_nc_u32 v38, s[sgprWorkGroup0], v37           // s17 = s[sgprWorkGroup0] / s16
v_cmp_ge_u32 vcc_lo, v38, s16                      // s17 = s[sgprWorkGroup0] / s16
s_mov_b32 exec_lo, vcc_lo                          // s17 = s[sgprWorkGroup0] / s16
v_add_nc_u32 v36, v36, 1                           // s17 = s[sgprWorkGroup0] / s16
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s17, v36                       // quotient
s_mul_i32 s20, s17, s16                            // quotient * non-magic divisor
s_sub_u32 s20, s[sgprWorkGroup0], s20              // WorkGroup0=remainder
s_mul_i32 s20, s20, s[sgprNumWorkGroups1]          // (wg1 % WGM)*NumWorkGroups1
s_add_u32 s20, s20, s[sgprWorkGroup1]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups1
v_cvt_f64_u32 v[36:37], s16                        // s18 = s[sgprNumWorkGroups0] / s16
v_rcp_f64 v[36:37], v[36:37]                       // s18 = s[sgprNumWorkGroups0] / s16
v_cvt_f64_u32 v[38:39], s[sgprNumWorkGroups0]      // s18 = s[sgprNumWorkGroups0] / s16
v_mul_f64 v[36:37], v[36:37], v[38:39]             // s18 = s[sgprNumWorkGroups0] / s16
v_cvt_u32_f64 v36, v[36:37]                        // s18 = s[sgprNumWorkGroups0] / s16
v_mul_lo_u32 v37, v36, s16                         // s18 = s[sgprNumWorkGroups0] / s16
v_sub_nc_u32 v38, s[sgprNumWorkGroups0], v37       // s18 = s[sgprNumWorkGroups0] / s16
v_cmp_ge_u32 vcc_lo, v38, s16                      // s18 = s[sgprNumWorkGroups0] / s16
s_mov_b32 exec_lo, vcc_lo                          // s18 = s[sgprNumWorkGroups0] / s16
v_add_nc_u32 v36, v36, 1                           // s18 = s[sgprNumWorkGroups0] / s16
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s18, v36                       // quotient
s_mul_i32 s19, s16, s18                            // quotient * non-magic divisor
s_sub_u32 s19, s[sgprNumWorkGroups0], s19          // NumWorkGroups0=remainder
s_cmp_eq_u32 s19, 0                                // remainder == 0 ?
s_cmov_b32 s19, s16                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s17, s18                              // blockId >= numFullBlocks ?
s_cselect_b32 s18, s19, s16
v_cvt_f64_u32 v[36:37], s18                        // s[sgprWorkGroup1] = s20 / s18
v_rcp_f64 v[36:37], v[36:37]                       // s[sgprWorkGroup1] = s20 / s18
v_cvt_f64_u32 v[38:39], s20                        // s[sgprWorkGroup1] = s20 / s18
v_mul_f64 v[36:37], v[36:37], v[38:39]             // s[sgprWorkGroup1] = s20 / s18
v_cvt_u32_f64 v36, v[36:37]                        // s[sgprWorkGroup1] = s20 / s18
v_mul_lo_u32 v37, v36, s18                         // s[sgprWorkGroup1] = s20 / s18
v_sub_nc_u32 v38, s20, v37                         // s[sgprWorkGroup1] = s20 / s18
v_cmp_ge_u32 vcc_lo, v38, s18                      // s[sgprWorkGroup1] = s20 / s18
s_mov_b32 exec_lo, vcc_lo                          // s[sgprWorkGroup1] = s20 / s18
v_add_nc_u32 v36, v36, 1                           // s[sgprWorkGroup1] = s20 / s18
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v37, v36, s18                         // s[sgprWorkGroup1] = s20 / s18
v_sub_nc_u32 v38, s20, v37                         // s[sgprWorkGroup1] = s20 / s18
v_readfirstlane_b32 s[sgprWorkGroup1], v36         // quotient
v_readfirstlane_b32 s[sgprWorkGroup0], v38         // remainder
s_mul_i32 s[sgprWorkGroup0], s[sgprWorkGroup1], s18 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup0], s20, s[sgprWorkGroup0] // WorkGroup0=remainder
s_mul_i32 s17, s17, s16                            // blockId * WGM
s_add_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s17 // wg1 += blockId * WGM
s_branch label_WGM
label_WGMPositive:
s_mov_b32 s16, s16                                 // WGM
v_cvt_f64_u32 v[36:37], s16                        // s17 = s[sgprWorkGroup1] / s16
v_rcp_f64 v[36:37], v[36:37]                       // s17 = s[sgprWorkGroup1] / s16
v_cvt_f64_u32 v[38:39], s[sgprWorkGroup1]          // s17 = s[sgprWorkGroup1] / s16
v_mul_f64 v[36:37], v[36:37], v[38:39]             // s17 = s[sgprWorkGroup1] / s16
v_cvt_u32_f64 v36, v[36:37]                        // s17 = s[sgprWorkGroup1] / s16
v_mul_lo_u32 v37, v36, s16                         // s17 = s[sgprWorkGroup1] / s16
v_sub_nc_u32 v38, s[sgprWorkGroup1], v37           // s17 = s[sgprWorkGroup1] / s16
v_cmp_ge_u32 vcc_lo, v38, s16                      // s17 = s[sgprWorkGroup1] / s16
s_mov_b32 exec_lo, vcc_lo                          // s17 = s[sgprWorkGroup1] / s16
v_add_nc_u32 v36, v36, 1                           // s17 = s[sgprWorkGroup1] / s16
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s17, v36                       // quotient
s_mul_i32 s20, s17, s16                            // quotient * non-magic divisor
s_sub_u32 s20, s[sgprWorkGroup1], s20              // WorkGroup1=remainder
s_mul_i32 s20, s20, s[sgprNumWorkGroups0]          // (wg1 % WGM)*NumWorkGroups0
s_add_u32 s20, s20, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups0
v_cvt_f64_u32 v[36:37], s16                        // s18 = s[sgprNumWorkGroups1] / s16
v_rcp_f64 v[36:37], v[36:37]                       // s18 = s[sgprNumWorkGroups1] / s16
v_cvt_f64_u32 v[38:39], s[sgprNumWorkGroups1]      // s18 = s[sgprNumWorkGroups1] / s16
v_mul_f64 v[36:37], v[36:37], v[38:39]             // s18 = s[sgprNumWorkGroups1] / s16
v_cvt_u32_f64 v36, v[36:37]                        // s18 = s[sgprNumWorkGroups1] / s16
v_mul_lo_u32 v37, v36, s16                         // s18 = s[sgprNumWorkGroups1] / s16
v_sub_nc_u32 v38, s[sgprNumWorkGroups1], v37       // s18 = s[sgprNumWorkGroups1] / s16
v_cmp_ge_u32 vcc_lo, v38, s16                      // s18 = s[sgprNumWorkGroups1] / s16
s_mov_b32 exec_lo, vcc_lo                          // s18 = s[sgprNumWorkGroups1] / s16
v_add_nc_u32 v36, v36, 1                           // s18 = s[sgprNumWorkGroups1] / s16
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s18, v36                       // quotient
s_mul_i32 s19, s16, s18                            // quotient * non-magic divisor
s_sub_u32 s19, s[sgprNumWorkGroups1], s19          // NumWorkGroups1=remainder
s_cmp_eq_u32 s19, 0                                // remainder == 0 ?
s_cmov_b32 s19, s16                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s17, s18                              // blockId >= numFullBlocks ?
s_cselect_b32 s18, s19, s16
v_cvt_f64_u32 v[36:37], s18                        // s[sgprWorkGroup0] = s20 / s18
v_rcp_f64 v[36:37], v[36:37]                       // s[sgprWorkGroup0] = s20 / s18
v_cvt_f64_u32 v[38:39], s20                        // s[sgprWorkGroup0] = s20 / s18
v_mul_f64 v[36:37], v[36:37], v[38:39]             // s[sgprWorkGroup0] = s20 / s18
v_cvt_u32_f64 v36, v[36:37]                        // s[sgprWorkGroup0] = s20 / s18
v_mul_lo_u32 v37, v36, s18                         // s[sgprWorkGroup0] = s20 / s18
v_sub_nc_u32 v38, s20, v37                         // s[sgprWorkGroup0] = s20 / s18
v_cmp_ge_u32 vcc_lo, v38, s18                      // s[sgprWorkGroup0] = s20 / s18
s_mov_b32 exec_lo, vcc_lo                          // s[sgprWorkGroup0] = s20 / s18
v_add_nc_u32 v36, v36, 1                           // s[sgprWorkGroup0] = s20 / s18
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v37, v36, s18                         // s[sgprWorkGroup0] = s20 / s18
v_sub_nc_u32 v38, s20, v37                         // s[sgprWorkGroup0] = s20 / s18
v_readfirstlane_b32 s[sgprWorkGroup0], v36         // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v38         // remainder
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
v_mov_b32 v36, v0                                  // groA0I_0
v_add_co_u32 v37, vcc_lo, 8, v36                   // groA0I_1 += LSPA
v_add_co_u32 v38, vcc_lo, 8, v37                   // groA0I_2 += LSPA
v_add_co_u32 v39, vcc_lo, 8, v38                   // groA0I_3 += LSPA

/* global read addresses: tile offsets b */
v_mov_b32 v40, v2                                  // groB1J_0
v_add_co_u32 v41, vcc_lo, 8, v40                   // groB1J_1 += LSPB
v_add_co_u32 v42, vcc_lo, 8, v41                   // groB1J_2 += LSPB
v_add_co_u32 v43, vcc_lo, 8, v42                   // groB1J_3 += LSPB

/* global read addresses: unroll offsets a */
v_mov_b32 v6, v1                                   // groAL_0

/* global read addresses: unroll offsets b */
v_mov_b32 v7, v3                                   // groBL_0

/* global read addresses: addresses a */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s19, s[sgprWorkGroup0], 32            // WorkGroup[01] * MT
s_mul_i32 s18, s[sgprWorkGroup0], 32               // WorkGroup[01] * MT
s_mul_hi_u32 s19, s18, s[sgprStrideA0I]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s18, s18, s[sgprStrideA0I]               // tlu=0, scaled tile-offset by stride
s_and_b32 s16, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_A                        // branch if GSUC == 1
s_mul_hi_u32 s17, 128, s[sgprGSUSumIdx]            // gsuOffset = DepthU*GSUSumIdx
s_mul_i32 s16, 128, s[sgprGSUSumIdx]               // gsuOffset = DepthU*GSUSumIdx
s_branch label_GSUC_A_End
label_GSUC_A:
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum], 7 // s[LoopCounterL] = s[sgprSizesSum] / 128
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0xfff    // Restore GSU
v_cvt_f32_u32 v44, s[sgprGSUSumIdx+1]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v44, v44                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v45, s[sgprLoopCounterL]             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v44, v44, v45                            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v44, v44                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v45, v44, s[sgprGSUSumIdx+1]         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_nc_u32 v45, s[sgprLoopCounterL], v45         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmp_eq_u32 vcc_lo, v45, s[sgprGSUSumIdx+1]       // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, vcc_lo                          // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_nc_u32 v44, 1, v44                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v45, 0                                   // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v45, s[sgprGSUSumIdx+1]       // overflow happened in remainder
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v44, v44, 1                           // quotient - 1
v_mul_u32_u24 v45, v44, s[sgprGSUSumIdx+1]         // re-calculate remainder
v_sub_nc_u32 v45, s[sgprLoopCounterL], v45         // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v44       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v45        // remainder
s_mul_i32 s17, s[sgprLoopCounterL], s[sgprGSUSumIdx] // quotient*GSUSumIdx
s_add_u32 s16, 1, s[sgprLoopCounterL]              // quotient+1
s_add_u32 s17, s17, s[sgprGSUSumIdx+1]             // quotient*GSUSumIdx+remainder
s_mul_i32 s16, s16, s[sgprGSUSumIdx]               // (quotient+1)*GSUSumIdx
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cselect_b32 s16, s16, s17                        // (quotient+1)*GSUSumIdx if needed
s_mul_hi_u32 s17, s16, 128                         // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
s_mul_i32 s16, s16, 128                            // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
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
s_add_u32 s[sgprStrideScaleA], s[sgprSizeL], 0x7f  // SizeL + G-1
s_lshr_b32 s[sgprStrideScaleA], s[sgprStrideScaleA], 7 // StrideScaleA = ceil(SizeL/128) scale elements
s_mul_i32 s16, s[sgprWorkGroup0], 32               // scaleA: workgroup row origin
s_mul_i32 s16, s16, s[sgprStrideScaleA]            // scaleA: * row stride
s_mul_i32 s16, s16, 2                              // scaleA: elements -> bytes
s_mul_i32 s17, s[sgprSizeI], s[sgprStrideScaleA]   // scaleA: SizeI * row stride
s_mul_i32 s17, s17, 2                              // scaleA: tensor bytes
s_sub_u32 s[sgprSrdScaleA+2], s17, s16             // scaleA: buffer limit from the workgroup origin
s_add_u32 s[sgprSrdScaleA+0], s[sgprAddressScaleA+0], s16 // scaleA: SRD base lo
s_addc_u32 s[sgprSrdScaleA+1], s[sgprAddressScaleA+1], 0 // scaleA: SRD base hi
s_mov_b32 s[sgprSrdScaleA+3], Srd127_96            // scaleA: set bits 127_96 in SRD

/* global read addresses: block-scale A zero-point srd */
s_mul_i32 s16, s[sgprWorkGroup0], 32               // scaleZeroA: workgroup row origin
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
s_mul_hi_u32 s19, s[sgprWorkGroup1], 32            // WorkGroup[01] * MT
s_mul_i32 s18, s[sgprWorkGroup1], 32               // WorkGroup[01] * MT
s_mul_hi_u32 s19, s18, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s18, s18, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_and_b32 s16, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_B                        // branch if GSUC == 1
s_mul_hi_u32 s17, 128, s[sgprGSUSumIdx]            // gsuOffset = DepthU*GSUSumIdx
s_mul_i32 s16, 128, s[sgprGSUSumIdx]               // gsuOffset = DepthU*GSUSumIdx
s_branch label_GSUC_B_End
label_GSUC_B:
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum], 7 // s[LoopCounterL] = s[sgprSizesSum] / 128
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0xfff    // Restore GSU
v_cvt_f32_u32 v44, s[sgprGSUSumIdx+1]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v44, v44                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v45, s[sgprLoopCounterL]             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v44, v44, v45                            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v44, v44                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v45, v44, s[sgprGSUSumIdx+1]         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_nc_u32 v45, s[sgprLoopCounterL], v45         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmp_eq_u32 vcc_lo, v45, s[sgprGSUSumIdx+1]       // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, vcc_lo                          // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_nc_u32 v44, 1, v44                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v45, 0                                   // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v45, s[sgprGSUSumIdx+1]       // overflow happened in remainder
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v44, v44, 1                           // quotient - 1
v_mul_u32_u24 v45, v44, s[sgprGSUSumIdx+1]         // re-calculate remainder
v_sub_nc_u32 v45, s[sgprLoopCounterL], v45         // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v44       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v45        // remainder
s_mul_i32 s17, s[sgprLoopCounterL], s[sgprGSUSumIdx] // quotient*GSUSumIdx
s_add_u32 s16, 1, s[sgprLoopCounterL]              // quotient+1
s_add_u32 s17, s17, s[sgprGSUSumIdx+1]             // quotient*GSUSumIdx+remainder
s_mul_i32 s16, s16, s[sgprGSUSumIdx]               // (quotient+1)*GSUSumIdx
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cselect_b32 s16, s16, s17                        // (quotient+1)*GSUSumIdx if needed
s_mul_hi_u32 s17, s16, 128                         // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
s_mul_i32 s16, s16, 128                            // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
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
v_mul_lo_u32 v44, s[sgprStrideA0I], v[36]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+0+0], vcc_lo, v[6], v[44+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+0+0], 0x8, v[vgprGlobalReadOffsetA+0+0] // add prepad for pointer shift
v_lshrrev_b32 v44, 7, v6                           // scaleA: kGroup = k/128
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+0], s[sgprStrideScaleA], v36 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+0], v44, v[vgprGlobalReadOffsetScaleA+0] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+0], 1, v[vgprGlobalReadOffsetScaleA+0] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetScaleZeroA+0], 1, v36 // scaleZeroA: row / 2
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleZeroA+0], s[sgprStrideScaleA], v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: (row/2) * kGroups
v_add_nc_u32 v[vgprGlobalReadOffsetScaleZeroA+0], v44, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: + kGroup -> byte offset
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleZeroA+0], 1, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: make room for the nibble bit
v_and_b32 v44, 1, v36                              // scaleZeroA: nibble = row & 1
v_or_b32 v[vgprGlobalReadOffsetScaleZeroA+0], v44, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: (byte << 1) | nibble
v_lshrrev_b32 v[vgprGlobalReadOffsetA+0], 1, v[vgprGlobalReadOffsetA+0] //  (multiple bpe)
v_mul_lo_u32 v44, s[sgprStrideA0I], v[37]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+1+0], vcc_lo, v[6], v[44+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+1+0], 0x8, v[vgprGlobalReadOffsetA+1+0] // add prepad for pointer shift
v_lshrrev_b32 v44, 7, v6                           // scaleA: kGroup = k/128
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+1], s[sgprStrideScaleA], v37 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+1], v44, v[vgprGlobalReadOffsetScaleA+1] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+1], 1, v[vgprGlobalReadOffsetScaleA+1] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetScaleZeroA+1], 1, v37 // scaleZeroA: row / 2
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleZeroA+1], s[sgprStrideScaleA], v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: (row/2) * kGroups
v_add_nc_u32 v[vgprGlobalReadOffsetScaleZeroA+1], v44, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: + kGroup -> byte offset
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleZeroA+1], 1, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: make room for the nibble bit
v_and_b32 v44, 1, v37                              // scaleZeroA: nibble = row & 1
v_or_b32 v[vgprGlobalReadOffsetScaleZeroA+1], v44, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: (byte << 1) | nibble
v_lshrrev_b32 v[vgprGlobalReadOffsetA+1], 1, v[vgprGlobalReadOffsetA+1] //  (multiple bpe)
v_mul_lo_u32 v44, s[sgprStrideA0I], v[38]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+2+0], vcc_lo, v[6], v[44+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+2+0], 0x8, v[vgprGlobalReadOffsetA+2+0] // add prepad for pointer shift
v_lshrrev_b32 v44, 7, v6                           // scaleA: kGroup = k/128
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+2], s[sgprStrideScaleA], v38 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+2], v44, v[vgprGlobalReadOffsetScaleA+2] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+2], 1, v[vgprGlobalReadOffsetScaleA+2] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetScaleZeroA+2], 1, v38 // scaleZeroA: row / 2
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleZeroA+2], s[sgprStrideScaleA], v[vgprGlobalReadOffsetScaleZeroA+2] // scaleZeroA: (row/2) * kGroups
v_add_nc_u32 v[vgprGlobalReadOffsetScaleZeroA+2], v44, v[vgprGlobalReadOffsetScaleZeroA+2] // scaleZeroA: + kGroup -> byte offset
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleZeroA+2], 1, v[vgprGlobalReadOffsetScaleZeroA+2] // scaleZeroA: make room for the nibble bit
v_and_b32 v44, 1, v38                              // scaleZeroA: nibble = row & 1
v_or_b32 v[vgprGlobalReadOffsetScaleZeroA+2], v44, v[vgprGlobalReadOffsetScaleZeroA+2] // scaleZeroA: (byte << 1) | nibble
v_lshrrev_b32 v[vgprGlobalReadOffsetA+2], 1, v[vgprGlobalReadOffsetA+2] //  (multiple bpe)
v_mul_lo_u32 v44, s[sgprStrideA0I], v[39]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+3+0], vcc_lo, v[6], v[44+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+3+0], 0x8, v[vgprGlobalReadOffsetA+3+0] // add prepad for pointer shift
v_lshrrev_b32 v44, 7, v6                           // scaleA: kGroup = k/128
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+3], s[sgprStrideScaleA], v39 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+3], v44, v[vgprGlobalReadOffsetScaleA+3] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+3], 1, v[vgprGlobalReadOffsetScaleA+3] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetScaleZeroA+3], 1, v39 // scaleZeroA: row / 2
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleZeroA+3], s[sgprStrideScaleA], v[vgprGlobalReadOffsetScaleZeroA+3] // scaleZeroA: (row/2) * kGroups
v_add_nc_u32 v[vgprGlobalReadOffsetScaleZeroA+3], v44, v[vgprGlobalReadOffsetScaleZeroA+3] // scaleZeroA: + kGroup -> byte offset
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleZeroA+3], 1, v[vgprGlobalReadOffsetScaleZeroA+3] // scaleZeroA: make room for the nibble bit
v_and_b32 v44, 1, v39                              // scaleZeroA: nibble = row & 1
v_or_b32 v[vgprGlobalReadOffsetScaleZeroA+3], v44, v[vgprGlobalReadOffsetScaleZeroA+3] // scaleZeroA: (byte << 1) | nibble
v_lshrrev_b32 v[vgprGlobalReadOffsetA+3], 1, v[vgprGlobalReadOffsetA+3] //  (multiple bpe)
/* ============================================================= */

/* global read addresses: final offsets b */
/* ============================================================= */
v_mul_lo_u32 v36, s[sgprStrideB1J], v[40]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+0+0], vcc_lo, v[7], v[36+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+0+0], 0x8, v[vgprGlobalReadOffsetB+0+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+0], 1, v[vgprGlobalReadOffsetB+0] //  (multiple bpe)
v_mul_lo_u32 v36, s[sgprStrideB1J], v[41]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+1+0], vcc_lo, v[7], v[36+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+1+0], 0x8, v[vgprGlobalReadOffsetB+1+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+1], 1, v[vgprGlobalReadOffsetB+1] //  (multiple bpe)
v_mul_lo_u32 v36, s[sgprStrideB1J], v[42]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+2+0], vcc_lo, v[7], v[36+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+2+0], 0x8, v[vgprGlobalReadOffsetB+2+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+2], 1, v[vgprGlobalReadOffsetB+2] //  (multiple bpe)
v_mul_lo_u32 v36, s[sgprStrideB1J], v[43]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+3+0], vcc_lo, v[7], v[36+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+3+0], 0x8, v[vgprGlobalReadOffsetB+3+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+3], 1, v[vgprGlobalReadOffsetB+3] //  (multiple bpe)
/* ============================================================= */

/* global read addresses: increments a */
s_and_b32 s17, s[sgprGSU], 0xfff                   // Restore GSU
s_mov_b32 s[sgprGlobalReadIncsA+0], 64             // GSU*DepthU*Bpe*MI_dim(1)
s_mul_i32 s17, s17, s[sgprGlobalReadIncsA+0]       // GSU*DepthU*Bpe*MI_dim(1)
s_and_b32 s16, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cselect_b32 s[sgprGlobalReadIncsA+0], s[sgprGlobalReadIncsA+0], s17 // incrA (unrollIdx)

/* global read addresses: increments b */
s_and_b32 s17, s[sgprGSU], 0xfff                   // Restore GSU
s_mov_b32 s[sgprGlobalReadIncsB+0], 256            // GSU*DepthU*Bpe*MI_dim(1)
s_mul_i32 s17, s17, s[sgprGlobalReadIncsB+0]       // GSU*DepthU*Bpe*MI_dim(1)
s_and_b32 s16, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cselect_b32 s[sgprGlobalReadIncsB+0], s[sgprGlobalReadIncsB+0], s17 // incrB (unrollIdx)
/* declare loop num iterations */
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 7 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 128
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

/* global read inc block-scale A (2 bytes) */
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], 0x2 // scaleA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], 0x2 // scaleA limit -= inc
s_add_u32 s[sgprSrdScaleZeroA+0], s[sgprSrdScaleZeroA+0], 0x1 // scaleZeroA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleZeroA+1], s[sgprSrdScaleZeroA+1], 0 // scaleZeroA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleZeroA+2], s[sgprSrdScaleZeroA+2], 0x1 // scaleZeroA limit -= inc

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

/* initC: remove ValuC vgpr buffer [0...8) from pool */

/* initC: remove acc vgpr buffer [0...0) from pool */

/* initC: remove ValuA/B vgpr buffer [36...165) from pool */
v_mov_b32 v[vgprValuC+0], 0                        // initC
v_mov_b32 v[vgprValuC+1], 0                        // initC
v_mov_b32 v[vgprValuC+2], 0                        // initC
v_mov_b32 v[vgprValuC+3], 0                        // initC
v_mov_b32 v[vgprValuC+4], 0                        // initC
v_mov_b32 v[vgprValuC+5], 0                        // initC
v_mov_b32 v[vgprValuC+6], 0                        // initC
v_mov_b32 v[vgprValuC+7], 0                        // initC
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
v_cvt_f32_f16 v202, v[vgprG2LScaleA+0]             // scaleA: fp16 -> f32
v_and_b32 v203, 1, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v203, 2, v203                        // scaleZeroA: -> nibble shift 0 or 4
v_bfe_u32 v199, v[vgprG2LScaleZeroA+0], v203, 0x4  // scaleZeroA: extract the selected nibble (unsigned)
v_cvt_f32_i32 v199, v199                           // scaleZeroA: int4 -> f32
v_add_f32 v199, 0x43000000, v199                   // scaleZeroA: + 128 to cancel the magic bias
v_mul_f32 v203, v202, v199                         // scaleZeroA: z*s
v_xor_b32 v203, 0x80000000, v203                   // w4a16: negate -> -z*s
v_mov_b32 v201, v[vgprG2LA+2+0]                    // w4a16: save packed int4 dword 0
v_and_b32 v200, 0xf000f, v201                      // w4a16: isolate int4 #0 and #1
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+0+0], v199, v200         // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0x4, v201                      // w4a16: nibble pair 1
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #2 and #3
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+0+1], v199, v200         // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0x8, v201                      // w4a16: nibble pair 2
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #4 and #5
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+0+2], v199, v200         // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0xc, v201                      // w4a16: nibble pair 3
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #6 and #7
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+0+3], v199, v200         // w4a16: pack 2 fp16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
v_cvt_f32_f16 v202, v[vgprG2LScaleA+1]             // scaleA: fp16 -> f32
v_and_b32 v203, 1, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v203, 2, v203                        // scaleZeroA: -> nibble shift 0 or 4
v_bfe_u32 v199, v[vgprG2LScaleZeroA+1], v203, 0x4  // scaleZeroA: extract the selected nibble (unsigned)
v_cvt_f32_i32 v199, v199                           // scaleZeroA: int4 -> f32
v_add_f32 v199, 0x43000000, v199                   // scaleZeroA: + 128 to cancel the magic bias
v_mul_f32 v203, v202, v199                         // scaleZeroA: z*s
v_xor_b32 v203, 0x80000000, v203                   // w4a16: negate -> -z*s
v_mov_b32 v201, v[vgprG2LA+6+0]                    // w4a16: save packed int4 dword 0
v_and_b32 v200, 0xf000f, v201                      // w4a16: isolate int4 #0 and #1
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+4+0], v199, v200         // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0x4, v201                      // w4a16: nibble pair 1
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #2 and #3
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+4+1], v199, v200         // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0x8, v201                      // w4a16: nibble pair 2
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #4 and #5
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+4+2], v199, v200         // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0xc, v201                      // w4a16: nibble pair 3
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #6 and #7
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+4+3], v199, v200         // w4a16: pack 2 fp16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:2304 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 2304 sync LDS0
v_cvt_f32_f16 v202, v[vgprG2LScaleA+2]             // scaleA: fp16 -> f32
v_and_b32 v203, 1, v[vgprGlobalReadOffsetScaleZeroA+2] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v203, 2, v203                        // scaleZeroA: -> nibble shift 0 or 4
v_bfe_u32 v199, v[vgprG2LScaleZeroA+2], v203, 0x4  // scaleZeroA: extract the selected nibble (unsigned)
v_cvt_f32_i32 v199, v199                           // scaleZeroA: int4 -> f32
v_add_f32 v199, 0x43000000, v199                   // scaleZeroA: + 128 to cancel the magic bias
v_mul_f32 v203, v202, v199                         // scaleZeroA: z*s
v_xor_b32 v203, 0x80000000, v203                   // w4a16: negate -> -z*s
v_mov_b32 v201, v[vgprG2LA+10+0]                   // w4a16: save packed int4 dword 0
v_and_b32 v200, 0xf000f, v201                      // w4a16: isolate int4 #0 and #1
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+8+0], v199, v200         // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0x4, v201                      // w4a16: nibble pair 1
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #2 and #3
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+8+1], v199, v200         // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0x8, v201                      // w4a16: nibble pair 2
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #4 and #5
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+8+2], v199, v200         // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0xc, v201                      // w4a16: nibble pair 3
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #6 and #7
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+8+3], v199, v200         // w4a16: pack 2 fp16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:4608 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 4608 sync LDS0
v_cvt_f32_f16 v202, v[vgprG2LScaleA+3]             // scaleA: fp16 -> f32
v_and_b32 v203, 1, v[vgprGlobalReadOffsetScaleZeroA+3] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v203, 2, v203                        // scaleZeroA: -> nibble shift 0 or 4
v_bfe_u32 v199, v[vgprG2LScaleZeroA+3], v203, 0x4  // scaleZeroA: extract the selected nibble (unsigned)
v_cvt_f32_i32 v199, v199                           // scaleZeroA: int4 -> f32
v_add_f32 v199, 0x43000000, v199                   // scaleZeroA: + 128 to cancel the magic bias
v_mul_f32 v203, v202, v199                         // scaleZeroA: z*s
v_xor_b32 v203, 0x80000000, v203                   // w4a16: negate -> -z*s
v_mov_b32 v201, v[vgprG2LA+14+0]                   // w4a16: save packed int4 dword 0
v_and_b32 v200, 0xf000f, v201                      // w4a16: isolate int4 #0 and #1
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+12+0], v199, v200        // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0x4, v201                      // w4a16: nibble pair 1
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #2 and #3
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+12+1], v199, v200        // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0x8, v201                      // w4a16: nibble pair 2
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #4 and #5
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+12+2], v199, v200        // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0xc, v201                      // w4a16: nibble pair 3
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #6 and #7
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+12+3], v199, v200        // w4a16: pack 2 fp16
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
v_lshrrev_b32 v199, 1, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: drop the nibble bit
buffer_load_d16_u8 v[vgprG2LScaleZeroA+0], v199, s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 0
v_lshrrev_b32 v199, 1, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: drop the nibble bit
buffer_load_d16_u8 v[vgprG2LScaleZeroA+1], v199, s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 1
v_lshrrev_b32 v199, 1, v[vgprGlobalReadOffsetScaleZeroA+2] // scaleZeroA: drop the nibble bit
buffer_load_d16_u8 v[vgprG2LScaleZeroA+2], v199, s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 2
v_lshrrev_b32 v199, 1, v[vgprGlobalReadOffsetScaleZeroA+3] // scaleZeroA: drop the nibble bit
buffer_load_d16_u8 v[vgprG2LScaleZeroA+3], v199, s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 3
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

/* global read inc block-scale A (2 bytes) */
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], 0x2 // scaleA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], 0x2 // scaleA limit -= inc
s_add_u32 s[sgprSrdScaleZeroA+0], s[sgprSrdScaleZeroA+0], 0x1 // scaleZeroA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleZeroA+1], s[sgprSrdScaleZeroA+1], 0 // scaleZeroA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleZeroA+2], s[sgprSrdScaleZeroA+2], 0x1 // scaleZeroA limit -= inc

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
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read b */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read increment a */
/* N/A, lro->16 */
/* localReadDoCntA 1 localReadDoCntMXSA 0 localReadDoCntB 1 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->16 */
/* localReadDoCntA 1 localReadDoCntMXSA 0 localReadDoCntB 1 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0 for iteration == 0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=2 */

/* iter 1 */

/* local read a */
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0

/* local read b */
ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0

/* local read increment a */
/* N/A, lro->32 */
/* localReadDoCntA 2 localReadDoCntMXSA 0 localReadDoCntB 2 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->32 */
/* localReadDoCntA 2 localReadDoCntMXSA 0 localReadDoCntB 2 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=0 readsPerIterB=2 */

/* iter 2 */

/* local read a */
ds_load_b128 v[vgprValuA_X2_I0+0:vgprValuA_X2_I0+0+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X2_I0+4:vgprValuA_X2_I0+4+3], v[vgprLocalReadAddrA+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=2 iui=0 sync LDS0

/* local read b */
ds_load_b128 v[vgprValuB_X2_I0+0:vgprValuB_X2_I0+0+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X2_I0+4:vgprValuB_X2_I0+4+3], v[vgprLocalReadAddrB+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=2 iui=0 sync LDS0

/* local read increment a */
/* N/A, lro->48 */
/* localReadDoCntA 3 localReadDoCntMXSA 0 localReadDoCntB 3 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->48 */
/* localReadDoCntA 3 localReadDoCntMXSA 0 localReadDoCntB 3 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+7], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=0 readsPerIterB=2 */

/* iter 3 */

/* local read a */
ds_load_b128 v[vgprValuA_X3_I0+0:vgprValuA_X3_I0+0+3], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=3 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X3_I0+4:vgprValuA_X3_I0+4+3], v[vgprLocalReadAddrA+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=3 iui=0 sync LDS0

/* local read b */
ds_load_b128 v[vgprValuB_X3_I0+0:vgprValuB_X3_I0+0+3], v[vgprLocalReadAddrB+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=3 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X3_I0+4:vgprValuB_X3_I0+4+3], v[vgprLocalReadAddrB+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=3 iui=0 sync LDS0

/* local read increment a */
/* N/A, lro->64 */
/* localReadDoCntA 4 localReadDoCntMXSA 0 localReadDoCntB 4 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->64 */
/* localReadDoCntA 4 localReadDoCntMXSA 0 localReadDoCntB 4 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+7], v[vgprValuB_X3_I0+0+0+0:vgprValuB_X3_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=3 numReadsIterA=4 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=3 numReadsIterB=4 skipReadsIterB=0 readsPerIterB=2 */

/* iter 4 */

/* local read a */
ds_load_b128 v[vgprValuA_X4_I0+0:vgprValuA_X4_I0+0+3], v[vgprLocalReadAddrA+0] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X4_I0+4:vgprValuA_X4_I0+4+3], v[vgprLocalReadAddrA+0] offset:144 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=4 iui=0 sync LDS0

/* local read b */
ds_load_b128 v[vgprValuB_X4_I0+0:vgprValuB_X4_I0+0+3], v[vgprLocalReadAddrB+0] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X4_I0+4:vgprValuB_X4_I0+4+3], v[vgprLocalReadAddrB+0] offset:144 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=4 iui=0 sync LDS0

/* local read increment a */
/* N/A, lro->80 */
/* localReadDoCntA 5 localReadDoCntMXSA 0 localReadDoCntB 5 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->80 */
/* localReadDoCntA 5 localReadDoCntMXSA 0 localReadDoCntB 5 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X4_I0+0+0+0:vgprValuA_X4_I0+0+0+0+7], v[vgprValuB_X4_I0+0+0+0:vgprValuB_X4_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=4 numReadsIterA=5 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=4 numReadsIterB=5 skipReadsIterB=0 readsPerIterB=2 */

/* iter 5 */

/* local read a */
ds_load_b128 v[vgprValuA_X5_I0+0:vgprValuA_X5_I0+0+3], v[vgprLocalReadAddrA+0] offset:160 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=5 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X5_I0+4:vgprValuA_X5_I0+4+3], v[vgprLocalReadAddrA+0] offset:176 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=5 iui=0 sync LDS0

/* local read b */
ds_load_b128 v[vgprValuB_X5_I0+0:vgprValuB_X5_I0+0+3], v[vgprLocalReadAddrB+0] offset:160 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=5 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X5_I0+4:vgprValuB_X5_I0+4+3], v[vgprLocalReadAddrB+0] offset:176 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=5 iui=0 sync LDS0

/* local read increment a */
/* N/A, lro->96 */
/* localReadDoCntA 6 localReadDoCntMXSA 0 localReadDoCntB 6 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->96 */
/* localReadDoCntA 6 localReadDoCntMXSA 0 localReadDoCntB 6 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X5_I0+0+0+0:vgprValuA_X5_I0+0+0+0+7], v[vgprValuB_X5_I0+0+0+0:vgprValuB_X5_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=5 numReadsIterA=6 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=5 numReadsIterB=6 skipReadsIterB=0 readsPerIterB=2 */

/* iter 6 */

/* local read a */
ds_load_b128 v[vgprValuA_X6_I0+0:vgprValuA_X6_I0+0+3], v[vgprLocalReadAddrA+0] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X6_I0+4:vgprValuA_X6_I0+4+3], v[vgprLocalReadAddrA+0] offset:208 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=6 iui=0 sync LDS0

/* local read b */
ds_load_b128 v[vgprValuB_X6_I0+0:vgprValuB_X6_I0+0+3], v[vgprLocalReadAddrB+0] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X6_I0+4:vgprValuB_X6_I0+4+3], v[vgprLocalReadAddrB+0] offset:208 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=6 iui=0 sync LDS0

/* local read increment a */
/* N/A, lro->112 */
/* localReadDoCntA 7 localReadDoCntMXSA 0 localReadDoCntB 7 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->112 */
/* localReadDoCntA 7 localReadDoCntMXSA 0 localReadDoCntB 7 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X6_I0+0+0+0:vgprValuA_X6_I0+0+0+0+7], v[vgprValuB_X6_I0+0+0+0:vgprValuB_X6_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=6 numReadsIterA=7 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=6 numReadsIterB=7 skipReadsIterB=0 readsPerIterB=2 */

/* iter 7 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */

/* local read a */
ds_load_b128 v[vgprValuA_X7_I0+0:vgprValuA_X7_I0+0+3], v[vgprLocalReadAddrA+0] offset:224 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=7 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X7_I0+4:vgprValuA_X7_I0+4+3], v[vgprLocalReadAddrA+0] offset:240 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=7 iui=0 sync LDS0

/* local read b */
ds_load_b128 v[vgprValuB_X7_I0+0:vgprValuB_X7_I0+0+3], v[vgprLocalReadAddrB+0] offset:224 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=7 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X7_I0+4:vgprValuB_X7_I0+4+3], v[vgprLocalReadAddrB+0] offset:240 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=7 iui=0 sync LDS0
s_waitcnt vmcnt(0)                                 // 1wait for global read

/* local write A */
v_cvt_f32_f16 v202, v[vgprG2LScaleA+0]             // scaleA: fp16 -> f32
v_and_b32 v203, 1, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v203, 2, v203                        // scaleZeroA: -> nibble shift 0 or 4
v_bfe_u32 v199, v[vgprG2LScaleZeroA+0], v203, 0x4  // scaleZeroA: extract the selected nibble (unsigned)
v_cvt_f32_i32 v199, v199                           // scaleZeroA: int4 -> f32
v_add_f32 v199, 0x43000000, v199                   // scaleZeroA: + 128 to cancel the magic bias
v_mul_f32 v203, v202, v199                         // scaleZeroA: z*s
v_xor_b32 v203, 0x80000000, v203                   // w4a16: negate -> -z*s
v_mov_b32 v201, v[vgprG2LA+2+0]                    // w4a16: save packed int4 dword 0
v_and_b32 v200, 0xf000f, v201                      // w4a16: isolate int4 #0 and #1
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+0+0], v199, v200         // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0x4, v201                      // w4a16: nibble pair 1
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #2 and #3
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+0+1], v199, v200         // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0x8, v201                      // w4a16: nibble pair 2
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #4 and #5
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+0+2], v199, v200         // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0xc, v201                      // w4a16: nibble pair 3
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #6 and #7
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+0+3], v199, v200         // w4a16: pack 2 fp16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:32768 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 32768 sync LDS1
v_cvt_f32_f16 v202, v[vgprG2LScaleA+1]             // scaleA: fp16 -> f32
v_and_b32 v203, 1, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v203, 2, v203                        // scaleZeroA: -> nibble shift 0 or 4
v_bfe_u32 v199, v[vgprG2LScaleZeroA+1], v203, 0x4  // scaleZeroA: extract the selected nibble (unsigned)
v_cvt_f32_i32 v199, v199                           // scaleZeroA: int4 -> f32
v_add_f32 v199, 0x43000000, v199                   // scaleZeroA: + 128 to cancel the magic bias
v_mul_f32 v203, v202, v199                         // scaleZeroA: z*s
v_xor_b32 v203, 0x80000000, v203                   // w4a16: negate -> -z*s
v_mov_b32 v201, v[vgprG2LA+6+0]                    // w4a16: save packed int4 dword 0
v_and_b32 v200, 0xf000f, v201                      // w4a16: isolate int4 #0 and #1
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+4+0], v199, v200         // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0x4, v201                      // w4a16: nibble pair 1
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #2 and #3
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+4+1], v199, v200         // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0x8, v201                      // w4a16: nibble pair 2
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #4 and #5
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+4+2], v199, v200         // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0xc, v201                      // w4a16: nibble pair 3
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #6 and #7
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+4+3], v199, v200         // w4a16: pack 2 fp16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:35072 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 35072 sync LDS1
v_cvt_f32_f16 v202, v[vgprG2LScaleA+2]             // scaleA: fp16 -> f32
v_and_b32 v203, 1, v[vgprGlobalReadOffsetScaleZeroA+2] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v203, 2, v203                        // scaleZeroA: -> nibble shift 0 or 4
v_bfe_u32 v199, v[vgprG2LScaleZeroA+2], v203, 0x4  // scaleZeroA: extract the selected nibble (unsigned)
v_cvt_f32_i32 v199, v199                           // scaleZeroA: int4 -> f32
v_add_f32 v199, 0x43000000, v199                   // scaleZeroA: + 128 to cancel the magic bias
v_mul_f32 v203, v202, v199                         // scaleZeroA: z*s
v_xor_b32 v203, 0x80000000, v203                   // w4a16: negate -> -z*s
v_mov_b32 v201, v[vgprG2LA+10+0]                   // w4a16: save packed int4 dword 0
v_and_b32 v200, 0xf000f, v201                      // w4a16: isolate int4 #0 and #1
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+8+0], v199, v200         // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0x4, v201                      // w4a16: nibble pair 1
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #2 and #3
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+8+1], v199, v200         // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0x8, v201                      // w4a16: nibble pair 2
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #4 and #5
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+8+2], v199, v200         // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0xc, v201                      // w4a16: nibble pair 3
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #6 and #7
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+8+3], v199, v200         // w4a16: pack 2 fp16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:37376 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 37376 sync LDS1
v_cvt_f32_f16 v202, v[vgprG2LScaleA+3]             // scaleA: fp16 -> f32
v_and_b32 v203, 1, v[vgprGlobalReadOffsetScaleZeroA+3] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v203, 2, v203                        // scaleZeroA: -> nibble shift 0 or 4
v_bfe_u32 v199, v[vgprG2LScaleZeroA+3], v203, 0x4  // scaleZeroA: extract the selected nibble (unsigned)
v_cvt_f32_i32 v199, v199                           // scaleZeroA: int4 -> f32
v_add_f32 v199, 0x43000000, v199                   // scaleZeroA: + 128 to cancel the magic bias
v_mul_f32 v203, v202, v199                         // scaleZeroA: z*s
v_xor_b32 v203, 0x80000000, v203                   // w4a16: negate -> -z*s
v_mov_b32 v201, v[vgprG2LA+14+0]                   // w4a16: save packed int4 dword 0
v_and_b32 v200, 0xf000f, v201                      // w4a16: isolate int4 #0 and #1
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+12+0], v199, v200        // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0x4, v201                      // w4a16: nibble pair 1
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #2 and #3
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+12+1], v199, v200        // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0x8, v201                      // w4a16: nibble pair 2
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #4 and #5
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+12+2], v199, v200        // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0xc, v201                      // w4a16: nibble pair 3
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #6 and #7
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+12+3], v199, v200        // w4a16: pack 2 fp16
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
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X7_I0+0+0+0:vgprValuA_X7_I0+0+0+0+7], v[vgprValuB_X7_I0+0+0+0:vgprValuB_X7_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=7 numReadsIterA=8 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=7 numReadsIterB=8 skipReadsIterB=0 readsPerIterB=2 */

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
v_lshrrev_b32 v199, 1, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: drop the nibble bit
buffer_load_d16_u8 v[vgprG2LScaleZeroA+0], v199, s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 0
v_lshrrev_b32 v199, 1, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: drop the nibble bit
buffer_load_d16_u8 v[vgprG2LScaleZeroA+1], v199, s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 1
v_lshrrev_b32 v199, 1, v[vgprGlobalReadOffsetScaleZeroA+2] // scaleZeroA: drop the nibble bit
buffer_load_d16_u8 v[vgprG2LScaleZeroA+2], v199, s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 2
v_lshrrev_b32 v199, 1, v[vgprGlobalReadOffsetScaleZeroA+3] // scaleZeroA: drop the nibble bit
buffer_load_d16_u8 v[vgprG2LScaleZeroA+3], v199, s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 3
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

/* global read inc block-scale A (2 bytes) */
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], 0x2 // scaleA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], 0x2 // scaleA limit -= inc
s_add_u32 s[sgprSrdScaleZeroA+0], s[sgprSrdScaleZeroA+0], 0x1 // scaleZeroA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleZeroA+1], s[sgprSrdScaleZeroA+1], 0 // scaleZeroA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleZeroA+2], s[sgprSrdScaleZeroA+2], 0x1 // scaleZeroA limit -= inc

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
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:32768 // L -> Reg lro=0 swapByteOffset=32768 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:32784 // L -> Reg lro=0 swapByteOffset=32768 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1

/* local read b */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:32768 // L -> Reg lro=0 swapByteOffset=32768 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:32784 // L -> Reg lro=0 swapByteOffset=32768 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1

/* local read increment a */
/* N/A, lro->16 */
/* localReadDoCntA 9 localReadDoCntMXSA 0 localReadDoCntB 9 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->16 */
/* localReadDoCntA 9 localReadDoCntMXSA 0 localReadDoCntB 9 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0 for iteration == 0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=2 */

/* iter 1 */

/* local read a */
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:32800 // L -> Reg lro=16 swapByteOffset=32768 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:32816 // L -> Reg lro=16 swapByteOffset=32768 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS1

/* local read b */
ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:32800 // L -> Reg lro=16 swapByteOffset=32768 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:32816 // L -> Reg lro=16 swapByteOffset=32768 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS1

/* local read increment a */
/* N/A, lro->32 */
/* localReadDoCntA 10 localReadDoCntMXSA 0 localReadDoCntB 10 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->32 */
/* localReadDoCntA 10 localReadDoCntMXSA 0 localReadDoCntB 10 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=0 readsPerIterB=2 */

/* iter 2 */

/* local read a */
ds_load_b128 v[vgprValuA_X2_I0+0:vgprValuA_X2_I0+0+3], v[vgprLocalReadAddrA+0] offset:32832 // L -> Reg lro=32 swapByteOffset=32768 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X2_I0+4:vgprValuA_X2_I0+4+3], v[vgprLocalReadAddrA+0] offset:32848 // L -> Reg lro=32 swapByteOffset=32768 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=2 iui=0 sync LDS1

/* local read b */
ds_load_b128 v[vgprValuB_X2_I0+0:vgprValuB_X2_I0+0+3], v[vgprLocalReadAddrB+0] offset:32832 // L -> Reg lro=32 swapByteOffset=32768 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X2_I0+4:vgprValuB_X2_I0+4+3], v[vgprLocalReadAddrB+0] offset:32848 // L -> Reg lro=32 swapByteOffset=32768 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=2 iui=0 sync LDS1

/* local read increment a */
/* N/A, lro->48 */
/* localReadDoCntA 11 localReadDoCntMXSA 0 localReadDoCntB 11 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->48 */
/* localReadDoCntA 11 localReadDoCntMXSA 0 localReadDoCntB 11 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+7], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=0 readsPerIterB=2 */

/* iter 3 */

/* local read a */
ds_load_b128 v[vgprValuA_X3_I0+0:vgprValuA_X3_I0+0+3], v[vgprLocalReadAddrA+0] offset:32864 // L -> Reg lro=48 swapByteOffset=32768 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=3 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X3_I0+4:vgprValuA_X3_I0+4+3], v[vgprLocalReadAddrA+0] offset:32880 // L -> Reg lro=48 swapByteOffset=32768 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=3 iui=0 sync LDS1

/* local read b */
ds_load_b128 v[vgprValuB_X3_I0+0:vgprValuB_X3_I0+0+3], v[vgprLocalReadAddrB+0] offset:32864 // L -> Reg lro=48 swapByteOffset=32768 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=3 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X3_I0+4:vgprValuB_X3_I0+4+3], v[vgprLocalReadAddrB+0] offset:32880 // L -> Reg lro=48 swapByteOffset=32768 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=3 iui=0 sync LDS1

/* local read increment a */
/* N/A, lro->64 */
/* localReadDoCntA 12 localReadDoCntMXSA 0 localReadDoCntB 12 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->64 */
/* localReadDoCntA 12 localReadDoCntMXSA 0 localReadDoCntB 12 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+7], v[vgprValuB_X3_I0+0+0+0:vgprValuB_X3_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=3 numReadsIterA=4 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=3 numReadsIterB=4 skipReadsIterB=0 readsPerIterB=2 */

/* iter 4 */

/* local read a */
ds_load_b128 v[vgprValuA_X4_I0+0:vgprValuA_X4_I0+0+3], v[vgprLocalReadAddrA+0] offset:32896 // L -> Reg lro=64 swapByteOffset=32768 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X4_I0+4:vgprValuA_X4_I0+4+3], v[vgprLocalReadAddrA+0] offset:32912 // L -> Reg lro=64 swapByteOffset=32768 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=4 iui=0 sync LDS1

/* local read b */
ds_load_b128 v[vgprValuB_X4_I0+0:vgprValuB_X4_I0+0+3], v[vgprLocalReadAddrB+0] offset:32896 // L -> Reg lro=64 swapByteOffset=32768 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X4_I0+4:vgprValuB_X4_I0+4+3], v[vgprLocalReadAddrB+0] offset:32912 // L -> Reg lro=64 swapByteOffset=32768 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=4 iui=0 sync LDS1

/* local read increment a */
/* N/A, lro->80 */
/* localReadDoCntA 13 localReadDoCntMXSA 0 localReadDoCntB 13 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->80 */
/* localReadDoCntA 13 localReadDoCntMXSA 0 localReadDoCntB 13 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X4_I0+0+0+0:vgprValuA_X4_I0+0+0+0+7], v[vgprValuB_X4_I0+0+0+0:vgprValuB_X4_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=4 numReadsIterA=5 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=4 numReadsIterB=5 skipReadsIterB=0 readsPerIterB=2 */

/* iter 5 */

/* local read a */
ds_load_b128 v[vgprValuA_X5_I0+0:vgprValuA_X5_I0+0+3], v[vgprLocalReadAddrA+0] offset:32928 // L -> Reg lro=80 swapByteOffset=32768 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=5 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X5_I0+4:vgprValuA_X5_I0+4+3], v[vgprLocalReadAddrA+0] offset:32944 // L -> Reg lro=80 swapByteOffset=32768 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=5 iui=0 sync LDS1

/* local read b */
ds_load_b128 v[vgprValuB_X5_I0+0:vgprValuB_X5_I0+0+3], v[vgprLocalReadAddrB+0] offset:32928 // L -> Reg lro=80 swapByteOffset=32768 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=5 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X5_I0+4:vgprValuB_X5_I0+4+3], v[vgprLocalReadAddrB+0] offset:32944 // L -> Reg lro=80 swapByteOffset=32768 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=5 iui=0 sync LDS1

/* local read increment a */
/* N/A, lro->96 */
/* localReadDoCntA 14 localReadDoCntMXSA 0 localReadDoCntB 14 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->96 */
/* localReadDoCntA 14 localReadDoCntMXSA 0 localReadDoCntB 14 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X5_I0+0+0+0:vgprValuA_X5_I0+0+0+0+7], v[vgprValuB_X5_I0+0+0+0:vgprValuB_X5_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=5 numReadsIterA=6 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=5 numReadsIterB=6 skipReadsIterB=0 readsPerIterB=2 */

/* iter 6 */

/* local read a */
ds_load_b128 v[vgprValuA_X6_I0+0:vgprValuA_X6_I0+0+3], v[vgprLocalReadAddrA+0] offset:32960 // L -> Reg lro=96 swapByteOffset=32768 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X6_I0+4:vgprValuA_X6_I0+4+3], v[vgprLocalReadAddrA+0] offset:32976 // L -> Reg lro=96 swapByteOffset=32768 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=6 iui=0 sync LDS1

/* local read b */
ds_load_b128 v[vgprValuB_X6_I0+0:vgprValuB_X6_I0+0+3], v[vgprLocalReadAddrB+0] offset:32960 // L -> Reg lro=96 swapByteOffset=32768 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X6_I0+4:vgprValuB_X6_I0+4+3], v[vgprLocalReadAddrB+0] offset:32976 // L -> Reg lro=96 swapByteOffset=32768 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=6 iui=0 sync LDS1

/* local read increment a */
/* N/A, lro->112 */
/* localReadDoCntA 15 localReadDoCntMXSA 0 localReadDoCntB 15 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->112 */
/* localReadDoCntA 15 localReadDoCntMXSA 0 localReadDoCntB 15 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X6_I0+0+0+0:vgprValuA_X6_I0+0+0+0+7], v[vgprValuB_X6_I0+0+0+0:vgprValuB_X6_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=6 numReadsIterA=7 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=6 numReadsIterB=7 skipReadsIterB=0 readsPerIterB=2 */

/* iter 7 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */

/* local read a */
ds_load_b128 v[vgprValuA_X7_I0+0:vgprValuA_X7_I0+0+3], v[vgprLocalReadAddrA+0] offset:32992 // L -> Reg lro=112 swapByteOffset=32768 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=7 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X7_I0+4:vgprValuA_X7_I0+4+3], v[vgprLocalReadAddrA+0] offset:33008 // L -> Reg lro=112 swapByteOffset=32768 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=7 iui=0 sync LDS1

/* local read b */
ds_load_b128 v[vgprValuB_X7_I0+0:vgprValuB_X7_I0+0+3], v[vgprLocalReadAddrB+0] offset:32992 // L -> Reg lro=112 swapByteOffset=32768 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=7 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X7_I0+4:vgprValuB_X7_I0+4+3], v[vgprLocalReadAddrB+0] offset:33008 // L -> Reg lro=112 swapByteOffset=32768 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=7 iui=0 sync LDS1
s_waitcnt vmcnt(0)                                 // 1wait for global read

/* local write A */
v_cvt_f32_f16 v202, v[vgprG2LScaleA+0]             // scaleA: fp16 -> f32
v_and_b32 v203, 1, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v203, 2, v203                        // scaleZeroA: -> nibble shift 0 or 4
v_bfe_u32 v199, v[vgprG2LScaleZeroA+0], v203, 0x4  // scaleZeroA: extract the selected nibble (unsigned)
v_cvt_f32_i32 v199, v199                           // scaleZeroA: int4 -> f32
v_add_f32 v199, 0x43000000, v199                   // scaleZeroA: + 128 to cancel the magic bias
v_mul_f32 v203, v202, v199                         // scaleZeroA: z*s
v_xor_b32 v203, 0x80000000, v203                   // w4a16: negate -> -z*s
v_mov_b32 v201, v[vgprG2LA+2+0]                    // w4a16: save packed int4 dword 0
v_and_b32 v200, 0xf000f, v201                      // w4a16: isolate int4 #0 and #1
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+0+0], v199, v200         // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0x4, v201                      // w4a16: nibble pair 1
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #2 and #3
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+0+1], v199, v200         // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0x8, v201                      // w4a16: nibble pair 2
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #4 and #5
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+0+2], v199, v200         // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0xc, v201                      // w4a16: nibble pair 3
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #6 and #7
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+0+3], v199, v200         // w4a16: pack 2 fp16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
v_cvt_f32_f16 v202, v[vgprG2LScaleA+1]             // scaleA: fp16 -> f32
v_and_b32 v203, 1, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v203, 2, v203                        // scaleZeroA: -> nibble shift 0 or 4
v_bfe_u32 v199, v[vgprG2LScaleZeroA+1], v203, 0x4  // scaleZeroA: extract the selected nibble (unsigned)
v_cvt_f32_i32 v199, v199                           // scaleZeroA: int4 -> f32
v_add_f32 v199, 0x43000000, v199                   // scaleZeroA: + 128 to cancel the magic bias
v_mul_f32 v203, v202, v199                         // scaleZeroA: z*s
v_xor_b32 v203, 0x80000000, v203                   // w4a16: negate -> -z*s
v_mov_b32 v201, v[vgprG2LA+6+0]                    // w4a16: save packed int4 dword 0
v_and_b32 v200, 0xf000f, v201                      // w4a16: isolate int4 #0 and #1
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+4+0], v199, v200         // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0x4, v201                      // w4a16: nibble pair 1
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #2 and #3
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+4+1], v199, v200         // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0x8, v201                      // w4a16: nibble pair 2
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #4 and #5
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+4+2], v199, v200         // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0xc, v201                      // w4a16: nibble pair 3
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #6 and #7
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+4+3], v199, v200         // w4a16: pack 2 fp16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:2304 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 2304 sync LDS0
v_cvt_f32_f16 v202, v[vgprG2LScaleA+2]             // scaleA: fp16 -> f32
v_and_b32 v203, 1, v[vgprGlobalReadOffsetScaleZeroA+2] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v203, 2, v203                        // scaleZeroA: -> nibble shift 0 or 4
v_bfe_u32 v199, v[vgprG2LScaleZeroA+2], v203, 0x4  // scaleZeroA: extract the selected nibble (unsigned)
v_cvt_f32_i32 v199, v199                           // scaleZeroA: int4 -> f32
v_add_f32 v199, 0x43000000, v199                   // scaleZeroA: + 128 to cancel the magic bias
v_mul_f32 v203, v202, v199                         // scaleZeroA: z*s
v_xor_b32 v203, 0x80000000, v203                   // w4a16: negate -> -z*s
v_mov_b32 v201, v[vgprG2LA+10+0]                   // w4a16: save packed int4 dword 0
v_and_b32 v200, 0xf000f, v201                      // w4a16: isolate int4 #0 and #1
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+8+0], v199, v200         // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0x4, v201                      // w4a16: nibble pair 1
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #2 and #3
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+8+1], v199, v200         // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0x8, v201                      // w4a16: nibble pair 2
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #4 and #5
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+8+2], v199, v200         // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0xc, v201                      // w4a16: nibble pair 3
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #6 and #7
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+8+3], v199, v200         // w4a16: pack 2 fp16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:4608 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 4608 sync LDS0
v_cvt_f32_f16 v202, v[vgprG2LScaleA+3]             // scaleA: fp16 -> f32
v_and_b32 v203, 1, v[vgprGlobalReadOffsetScaleZeroA+3] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v203, 2, v203                        // scaleZeroA: -> nibble shift 0 or 4
v_bfe_u32 v199, v[vgprG2LScaleZeroA+3], v203, 0x4  // scaleZeroA: extract the selected nibble (unsigned)
v_cvt_f32_i32 v199, v199                           // scaleZeroA: int4 -> f32
v_add_f32 v199, 0x43000000, v199                   // scaleZeroA: + 128 to cancel the magic bias
v_mul_f32 v203, v202, v199                         // scaleZeroA: z*s
v_xor_b32 v203, 0x80000000, v203                   // w4a16: negate -> -z*s
v_mov_b32 v201, v[vgprG2LA+14+0]                   // w4a16: save packed int4 dword 0
v_and_b32 v200, 0xf000f, v201                      // w4a16: isolate int4 #0 and #1
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+12+0], v199, v200        // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0x4, v201                      // w4a16: nibble pair 1
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #2 and #3
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+12+1], v199, v200        // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0x8, v201                      // w4a16: nibble pair 2
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #4 and #5
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+12+2], v199, v200        // w4a16: pack 2 fp16
v_lshrrev_b32 v200, 0xc, v201                      // w4a16: nibble pair 3
v_and_b32 v200, 0xf000f, v200                      // w4a16: isolate int4 #6 and #7
v_or_b32 v200, 0x43004300, v200                    // w4a16: -> two bf16 holding 128+q
v_lshlrev_b32 v199, 16, v200                       // w4a16: low bf16 -> f32
v_and_b32 v200, 0xffff0000, v200                   // w4a16: high bf16 -> f32
v_fma_f32 v199, v199, v202, v203                   // w4a16: q*s - z*s
v_fma_f32 v200, v200, v202, v203                   // w4a16: q*s - z*s
v_cvt_f16_f32 v199, v199                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v200, v200                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+12+3], v199, v200        // w4a16: pack 2 fp16
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
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X7_I0+0+0+0:vgprValuA_X7_I0+0+0+0+7], v[vgprValuB_X7_I0+0+0+0:vgprValuB_X7_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=7 numReadsIterA=8 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=7 numReadsIterB=8 skipReadsIterB=0 readsPerIterB=2 */

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
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read b */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read increment a */
/* N/A, lro->16 */
/* localReadDoCntA 17 localReadDoCntMXSA 0 localReadDoCntB 17 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->16 */
/* localReadDoCntA 17 localReadDoCntMXSA 0 localReadDoCntB 17 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0 for iteration == 0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=2 */

/* iter 1 (last unrolled loop) */

/* local read a */
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0

/* local read b */
ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0

/* local read increment a */
/* N/A, lro->32 */
/* localReadDoCntA 18 localReadDoCntMXSA 0 localReadDoCntB 18 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->32 */
/* localReadDoCntA 18 localReadDoCntMXSA 0 localReadDoCntB 18 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=0 readsPerIterB=2 */

/* iter 2 (last unrolled loop) */

/* local read a */
ds_load_b128 v[vgprValuA_X2_I0+0:vgprValuA_X2_I0+0+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X2_I0+4:vgprValuA_X2_I0+4+3], v[vgprLocalReadAddrA+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=2 iui=0 sync LDS0

/* local read b */
ds_load_b128 v[vgprValuB_X2_I0+0:vgprValuB_X2_I0+0+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X2_I0+4:vgprValuB_X2_I0+4+3], v[vgprLocalReadAddrB+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=2 iui=0 sync LDS0

/* local read increment a */
/* N/A, lro->48 */
/* localReadDoCntA 19 localReadDoCntMXSA 0 localReadDoCntB 19 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->48 */
/* localReadDoCntA 19 localReadDoCntMXSA 0 localReadDoCntB 19 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+7], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=0 readsPerIterB=2 */

/* iter 3 (last unrolled loop) */

/* local read a */
ds_load_b128 v[vgprValuA_X3_I0+0:vgprValuA_X3_I0+0+3], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=3 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X3_I0+4:vgprValuA_X3_I0+4+3], v[vgprLocalReadAddrA+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=3 iui=0 sync LDS0

/* local read b */
ds_load_b128 v[vgprValuB_X3_I0+0:vgprValuB_X3_I0+0+3], v[vgprLocalReadAddrB+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=3 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X3_I0+4:vgprValuB_X3_I0+4+3], v[vgprLocalReadAddrB+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=3 iui=0 sync LDS0

/* local read increment a */
/* N/A, lro->64 */
/* localReadDoCntA 20 localReadDoCntMXSA 0 localReadDoCntB 20 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->64 */
/* localReadDoCntA 20 localReadDoCntMXSA 0 localReadDoCntB 20 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+7], v[vgprValuB_X3_I0+0+0+0:vgprValuB_X3_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=3 numReadsIterA=4 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=3 numReadsIterB=4 skipReadsIterB=0 readsPerIterB=2 */

/* iter 4 (last unrolled loop) */

/* local read a */
ds_load_b128 v[vgprValuA_X4_I0+0:vgprValuA_X4_I0+0+3], v[vgprLocalReadAddrA+0] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X4_I0+4:vgprValuA_X4_I0+4+3], v[vgprLocalReadAddrA+0] offset:144 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=4 iui=0 sync LDS0

/* local read b */
ds_load_b128 v[vgprValuB_X4_I0+0:vgprValuB_X4_I0+0+3], v[vgprLocalReadAddrB+0] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X4_I0+4:vgprValuB_X4_I0+4+3], v[vgprLocalReadAddrB+0] offset:144 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=4 iui=0 sync LDS0

/* local read increment a */
/* N/A, lro->80 */
/* localReadDoCntA 21 localReadDoCntMXSA 0 localReadDoCntB 21 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->80 */
/* localReadDoCntA 21 localReadDoCntMXSA 0 localReadDoCntB 21 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X4_I0+0+0+0:vgprValuA_X4_I0+0+0+0+7], v[vgprValuB_X4_I0+0+0+0:vgprValuB_X4_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=4 numReadsIterA=5 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=4 numReadsIterB=5 skipReadsIterB=0 readsPerIterB=2 */

/* iter 5 (last unrolled loop) */

/* local read a */
ds_load_b128 v[vgprValuA_X5_I0+0:vgprValuA_X5_I0+0+3], v[vgprLocalReadAddrA+0] offset:160 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=5 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X5_I0+4:vgprValuA_X5_I0+4+3], v[vgprLocalReadAddrA+0] offset:176 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=5 iui=0 sync LDS0

/* local read b */
ds_load_b128 v[vgprValuB_X5_I0+0:vgprValuB_X5_I0+0+3], v[vgprLocalReadAddrB+0] offset:160 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=5 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X5_I0+4:vgprValuB_X5_I0+4+3], v[vgprLocalReadAddrB+0] offset:176 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=5 iui=0 sync LDS0

/* local read increment a */
/* N/A, lro->96 */
/* localReadDoCntA 22 localReadDoCntMXSA 0 localReadDoCntB 22 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->96 */
/* localReadDoCntA 22 localReadDoCntMXSA 0 localReadDoCntB 22 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X5_I0+0+0+0:vgprValuA_X5_I0+0+0+0+7], v[vgprValuB_X5_I0+0+0+0:vgprValuB_X5_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=5 numReadsIterA=6 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=5 numReadsIterB=6 skipReadsIterB=0 readsPerIterB=2 */

/* iter 6 (last unrolled loop) */

/* local read a */
ds_load_b128 v[vgprValuA_X6_I0+0:vgprValuA_X6_I0+0+3], v[vgprLocalReadAddrA+0] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X6_I0+4:vgprValuA_X6_I0+4+3], v[vgprLocalReadAddrA+0] offset:208 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=6 iui=0 sync LDS0

/* local read b */
ds_load_b128 v[vgprValuB_X6_I0+0:vgprValuB_X6_I0+0+3], v[vgprLocalReadAddrB+0] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X6_I0+4:vgprValuB_X6_I0+4+3], v[vgprLocalReadAddrB+0] offset:208 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=6 iui=0 sync LDS0

/* local read increment a */
/* N/A, lro->112 */
/* localReadDoCntA 23 localReadDoCntMXSA 0 localReadDoCntB 23 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->112 */
/* localReadDoCntA 23 localReadDoCntMXSA 0 localReadDoCntB 23 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X6_I0+0+0+0:vgprValuA_X6_I0+0+0+0+7], v[vgprValuB_X6_I0+0+0+0:vgprValuB_X6_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=6 numReadsIterA=7 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=6 numReadsIterB=7 skipReadsIterB=0 readsPerIterB=2 */

/* iter 7 (last unrolled loop) */

/* local read a */
ds_load_b128 v[vgprValuA_X7_I0+0:vgprValuA_X7_I0+0+3], v[vgprLocalReadAddrA+0] offset:224 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=7 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X7_I0+4:vgprValuA_X7_I0+4+3], v[vgprLocalReadAddrA+0] offset:240 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=7 iui=0 sync LDS0

/* local read b */
ds_load_b128 v[vgprValuB_X7_I0+0:vgprValuB_X7_I0+0+3], v[vgprLocalReadAddrB+0] offset:224 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=7 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X7_I0+4:vgprValuB_X7_I0+4+3], v[vgprLocalReadAddrB+0] offset:240 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=7 iui=0 sync LDS0
s_waitcnt vmcnt(0)                                 // 1wait for global read

/* local write A */

/* local write MXSA */

/* local write MXSB */

/* local write B */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X7_I0+0+0+0:vgprValuA_X7_I0+0+0+0+7], v[vgprValuB_X7_I0+0+0+0:vgprValuB_X7_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=7 numReadsIterA=8 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=7 numReadsIterB=8 skipReadsIterB=0 readsPerIterB=2 */
label_toPGR1end_OrdNLL:
label_PrefetchGlobalLastIterEnd:

/* Tail: add ValuA/B vgpr buffer [36...165) to pool */

/* Tail: add address/G2L vgpr [165...198) to pool */
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
v_lshrrev_b32 v40, 5, v[vgprSerial]                // 40 = Serial / 32
v_lshrrev_b32 v41, 1, v40                          // 41 = 40 / 2
v_lshlrev_b32 v37, 4, v41                          // wave coordination offset 1
v_and_b32 v41, 15, v[vgprSerial]                   // v41 = v[vgprSerial] % 16
v_add_lshl_u32 v37, v41, v37, 0                    // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v38, v37, s[sgprStrideC1J]            //  offset 1
v_mul_lo_u32 v39, v37, s[sgprStrideD1J]            //  offset 1
v_and_b32 v41, 1, v40                              // v41 = v40 % 2
v_lshlrev_b32 v41, 4, v41                          // wave coordination offset 0
v_and_b32 v36, 31, v[vgprSerial]                   // v36 = v[vgprSerial] % 32
v_lshrrev_b32 v36, 4, v36                          // 36 = 36 / 16
                                                   // thread0 * continuous_output (multiplier is 1, do nothing)
v_add_lshl_u32 v36, v41, v36, 0                    // coordination 0 = vwA *(wave_id0 + tid0)
s_mul_i32 s8, 32, s[sgprWorkGroup0]                // wgp0 * MT0
v_add_nc_u32 v36, s8, v36                          // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s8, 32, s[sgprWorkGroup1]                // wgp1 * MT1
v_add_nc_u32 v37, s8, v37                          // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1

/* not-LocalSplitU: global write */

/******************************************/
/* Global Write Elements                  */
/******************************************/
s_and_b32 s8, s[sgprGSU], 0xfff                    // Restore GSU
s_cmp_eq_u32 s8, 1                                 // GSU == 1 ?
s_cbranch_scc1 label_GSU_4                         // branch if GSU == 1
label_GW_B0_MB:
label_GW_B0_FD0_MB:

/* Edge/NonEdge store path check (M): Size % 32 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s28, 31, s[sgprSizeI]                    // s28 = s[sgprSizeI] % 32
s_add_u32 s29, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s29                // wg0 >= nwg0-1 ?
s_cselect_b32 s28, s28, 0                          // set rem
s_cmpk_gt_u32 s28, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_MB_Else         // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 32 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s28, 31, s[sgprSizeJ]                    // s28 = s[sgprSizeJ] % 32
s_add_u32 s29, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s29                // wg1 >= nwg1-1
s_cselect_b32 s28, s28, 0                          // set rem
s_cmpk_gt_u32 s28, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_MB_Then         // jump if edges required
label_GW_B0_FD0_VW1_MB_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=206 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,4,0,0:vw1); (0,5,0,0:vw1); (0,6,0,0:vw1); (0,7,0,0:vw1) */
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
v_add_lshl_u32 v47, v39, v36, 2                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=36, coord0Vgpr=36 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0)] */
v_mov_b32 v[vgprValuC+49], v[vgprValuC+0]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+50], v[vgprValuC+1]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+51], v[vgprValuC+2]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+52], v[vgprValuC+3]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+53], v[vgprValuC+4]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+54], v[vgprValuC+5]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+55], v[vgprValuC+6]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+56], v[vgprValuC+7]          // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b32 v49, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v50, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:8 // store D
buffer_store_b32 v51, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:16 // store D
buffer_store_b32 v52, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:24 // store D
buffer_store_b32 v53, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
buffer_store_b32 v54, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:40 // store D
buffer_store_b32 v55, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:48 // store D
buffer_store_b32 v56, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:56 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_B0_FD0_VW1_MB_NonEdgeEnd:
label_GW_B0_FD0_VW1_MB_Else:
label_GW_B0_FD0_VW1_MB_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=102 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,4,0,0:vw1); (0,5,0,0:vw1); (0,6,0,0:vw1); (0,7,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v42, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s28, v36, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v37, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v55, v39, v36, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v55, v42, v55, s30                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v40, vcc_lo, v36, 2                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v40, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v37, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v56, v39, v40, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v56, v42, v56, s30                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
v_add_co_u32 v40, vcc_lo, v36, 4                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v40, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v37, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v57, v39, v40, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v57, v42, v57, s30                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
v_add_co_u32 v40, vcc_lo, v36, 6                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v40, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v37, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v58, v39, v40, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v58, v42, v58, s30                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
v_add_co_u32 v40, vcc_lo, v36, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v40, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v37, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v59, v39, v40, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v59, v42, v59, s30                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
v_add_co_u32 v40, vcc_lo, v36, 10                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v40, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v37, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v60, v39, v40, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v60, v42, v60, s30                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
v_add_co_u32 v40, vcc_lo, v36, 12                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v40, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v37, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v61, v39, v40, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v61, v42, v61, s30                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
v_add_co_u32 v40, vcc_lo, v36, 14                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v40, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v37, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v62, v39, v40, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v62, v42, v62, s30                   // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0)] */
v_mov_b32 v[vgprValuC+47], v[vgprValuC+0]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+48], v[vgprValuC+1]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+49], v[vgprValuC+2]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+50], v[vgprValuC+3]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+51], v[vgprValuC+4]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+52], v[vgprValuC+5]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+53], v[vgprValuC+6]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+54], v[vgprValuC+7]          // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b32 v47, v55, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v48, v56, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v49, v57, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v50, v58, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v51, v59, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v52, v60, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v53, v61, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v54, v62, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
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

/* Edge/NonEdge store path check (M): Size % 32 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s28, 31, s[sgprSizeI]                    // s28 = s[sgprSizeI] % 32
s_add_u32 s29, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s29                // wg0 >= nwg0-1 ?
s_cselect_b32 s28, s28, 0                          // set rem
s_cmpk_gt_u32 s28, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_GSU1_Else       // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 32 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s28, 31, s[sgprSizeJ]                    // s28 = s[sgprSizeJ] % 32
s_add_u32 s29, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s29                // wg1 >= nwg1-1
s_cselect_b32 s28, s28, 0                          // set rem
s_cmpk_gt_u32 s28, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_GSU1_Then       // jump if edges required
label_GW_B0_FD0_VW1_GSU1_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=206 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,4,0,0:vw1); (0,5,0,0:vw1); (0,6,0,0:vw1); (0,7,0,0:vw1) */
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
v_add_lshl_u32 v47, v39, v36, 1                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=36, coord0Vgpr=36 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0)] */
v_mul_f32 v[vgprValuC+49], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+50], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+51], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+52], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+53], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+54], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+55], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+56], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v49, v[vgprValuC+49]                 // convert C to fp16
buffer_store_b16 v49, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v50, v[vgprValuC+50]                 // convert C to fp16
buffer_store_b16 v50, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:4 // store D
v_cvt_f16_f32 v51, v[vgprValuC+51]                 // convert C to fp16
buffer_store_b16 v51, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:8 // store D
v_cvt_f16_f32 v52, v[vgprValuC+52]                 // convert C to fp16
buffer_store_b16 v52, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:12 // store D
v_cvt_f16_f32 v53, v[vgprValuC+53]                 // convert C to fp16
buffer_store_b16 v53, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:16 // store D
v_cvt_f16_f32 v54, v[vgprValuC+54]                 // convert C to fp16
buffer_store_b16 v54, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:20 // store D
v_cvt_f16_f32 v55, v[vgprValuC+55]                 // convert C to fp16
buffer_store_b16 v55, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:24 // store D
v_cvt_f16_f32 v56, v[vgprValuC+56]                 // convert C to fp16
buffer_store_b16 v56, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:28 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW1_GSU1_NonEdgeEnd:
label_GW_B0_FD0_VW1_GSU1_Else:
label_GW_B0_FD0_VW1_GSU1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=102 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,4,0,0:vw1); (0,5,0,0:vw1); (0,6,0,0:vw1); (0,7,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v42, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s28, v36, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v37, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v55, v39, v36, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v55, v42, v55, s30                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v40, vcc_lo, v36, 2                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v40, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v37, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v56, v39, v40, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v56, v42, v56, s30                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
v_add_co_u32 v40, vcc_lo, v36, 4                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v40, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v37, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v57, v39, v40, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v57, v42, v57, s30                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
v_add_co_u32 v40, vcc_lo, v36, 6                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v40, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v37, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v58, v39, v40, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v58, v42, v58, s30                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
v_add_co_u32 v40, vcc_lo, v36, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v40, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v37, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v59, v39, v40, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v59, v42, v59, s30                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
v_add_co_u32 v40, vcc_lo, v36, 10                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v40, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v37, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v60, v39, v40, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v60, v42, v60, s30                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
v_add_co_u32 v40, vcc_lo, v36, 12                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v40, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v37, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v61, v39, v40, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v61, v42, v61, s30                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
v_add_co_u32 v40, vcc_lo, v36, 14                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v40, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v37, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v62, v39, v40, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v62, v42, v62, s30                   // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0)] */
v_mul_f32 v[vgprValuC+47], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+48], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+49], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+50], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+51], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+52], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+53], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+54], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v47, v[vgprValuC+47]                 // convert C to fp16
buffer_store_b16 v47, v55, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v48, v[vgprValuC+48]                 // convert C to fp16
buffer_store_b16 v48, v56, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v49, v[vgprValuC+49]                 // convert C to fp16
buffer_store_b16 v49, v57, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v50, v[vgprValuC+50]                 // convert C to fp16
buffer_store_b16 v50, v58, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v51, v[vgprValuC+51]                 // convert C to fp16
buffer_store_b16 v51, v59, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v52, v[vgprValuC+52]                 // convert C to fp16
buffer_store_b16 v52, v60, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v53, v[vgprValuC+53]                 // convert C to fp16
buffer_store_b16 v53, v61, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v54, v[vgprValuC+54]                 // convert C to fp16
buffer_store_b16 v54, v62, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_GSU1:
label_GW_B1_FD0_GSU1:

/* Edge/NonEdge store path check (M): Size % 32 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s28, 31, s[sgprSizeI]                    // s28 = s[sgprSizeI] % 32
s_add_u32 s29, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s29                // wg0 >= nwg0-1 ?
s_cselect_b32 s28, s28, 0                          // set rem
s_cmpk_gt_u32 s28, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW1_GSU1_Else       // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 32 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s28, 31, s[sgprSizeJ]                    // s28 = s[sgprSizeJ] % 32
s_add_u32 s29, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s29                // wg1 >= nwg1-1
s_cselect_b32 s28, s28, 0                          // set rem
s_cmpk_gt_u32 s28, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW1_GSU1_Then       // jump if edges required
label_GW_B1_FD0_VW1_GSU1_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=102 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,4,0,0:vw1); (0,5,0,0:vw1); (0,6,0,0:vw1); (0,7,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v48, v38, v36, 1                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=36, coord0Vgpr=36 (multiple bpe)
buffer_load_d16_b16 v57, v48, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
buffer_load_d16_b16 v58, v48, s[sgprSrdC:sgprSrdC+3], 0 offen offset:4 // load C
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
buffer_load_d16_b16 v59, v48, s[sgprSrdC:sgprSrdC+3], 0 offen offset:8 // load C
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
buffer_load_d16_b16 v60, v48, s[sgprSrdC:sgprSrdC+3], 0 offen offset:12 // load C
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
buffer_load_d16_b16 v61, v48, s[sgprSrdC:sgprSrdC+3], 0 offen offset:16 // load C
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
buffer_load_d16_b16 v62, v48, s[sgprSrdC:sgprSrdC+3], 0 offen offset:20 // load C
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
buffer_load_d16_b16 v63, v48, s[sgprSrdC:sgprSrdC+3], 0 offen offset:24 // load C
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
buffer_load_d16_b16 v64, v48, s[sgprSrdC:sgprSrdC+3], 0 offen offset:28 // load C
v_add_lshl_u32 v47, v39, v36, 1                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=36, coord0Vgpr=36 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0)] */
v_mul_f32 v[vgprValuC+49], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+50], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+51], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+52], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+53], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+54], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+55], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+56], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(7)                                 // vlcnt(7) = 8 - 1 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+49], s[sgprBeta], v57, v[vgprValuC+49] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v49, v[vgprValuC+49]                 // convert C to fp16
buffer_store_b16 v49, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(6)                                 // vlcnt(6) = 8 - 2 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+50], s[sgprBeta], v58, v[vgprValuC+50] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v50, v[vgprValuC+50]                 // convert C to fp16
buffer_store_b16 v50, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:4 // store D

s_waitcnt vmcnt(5)                                 // vlcnt(5) = 8 - 3 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+51], s[sgprBeta], v59, v[vgprValuC+51] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v51, v[vgprValuC+51]                 // convert C to fp16
buffer_store_b16 v51, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:8 // store D

s_waitcnt vmcnt(4)                                 // vlcnt(4) = 8 - 4 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+52], s[sgprBeta], v60, v[vgprValuC+52] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v52, v[vgprValuC+52]                 // convert C to fp16
buffer_store_b16 v52, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:12 // store D

s_waitcnt vmcnt(3)                                 // vlcnt(3) = 8 - 5 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+53], s[sgprBeta], v61, v[vgprValuC+53] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v53, v[vgprValuC+53]                 // convert C to fp16
buffer_store_b16 v53, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:16 // store D

s_waitcnt vmcnt(2)                                 // vlcnt(2) = 8 - 6 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+54], s[sgprBeta], v62, v[vgprValuC+54] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v54, v[vgprValuC+54]                 // convert C to fp16
buffer_store_b16 v54, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:20 // store D

s_waitcnt vmcnt(1)                                 // vlcnt(1) = 8 - 7 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+55], s[sgprBeta], v63, v[vgprValuC+55] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v55, v[vgprValuC+55]                 // convert C to fp16
buffer_store_b16 v55, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:24 // store D

s_waitcnt vmcnt(0)                                 // vlcnt(0) = 8 - 8 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+56], s[sgprBeta], v64, v[vgprValuC+56] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v56, v[vgprValuC+56]                 // convert C to fp16
buffer_store_b16 v56, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:28 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_FD0_VW1_GSU1_NonEdgeEnd:
label_GW_B1_FD0_VW1_GSU1_Else:
label_GW_B1_FD0_VW1_GSU1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=68 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,4,0,0:vw1); (0,5,0,0:vw1); (0,6,0,0:vw1); (0,7,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v42, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s28, v36, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v37, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v56, v38, v36, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v56, v42, v56, s30                   // LDC clip if OOB. offset
buffer_load_d16_b16 v55, v56, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v56, v39, v36, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v56, v42, v56, s30                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v40, vcc_lo, v36, 2                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v40, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v37, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v58, v38, v40, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v58, v42, v58, s30                   // LDC clip if OOB. offset
buffer_load_d16_b16 v57, v58, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v58, v39, v40, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v58, v42, v58, s30                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
v_add_co_u32 v40, vcc_lo, v36, 4                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v40, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v37, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v60, v38, v40, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v60, v42, v60, s30                   // LDC clip if OOB. offset
buffer_load_d16_b16 v59, v60, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v60, v39, v40, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v60, v42, v60, s30                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
v_add_co_u32 v40, vcc_lo, v36, 6                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v40, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v37, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v62, v38, v40, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v62, v42, v62, s30                   // LDC clip if OOB. offset
buffer_load_d16_b16 v61, v62, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v62, v39, v40, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v62, v42, v62, s30                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
v_add_co_u32 v40, vcc_lo, v36, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v40, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v37, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v64, v38, v40, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v64, v42, v64, s30                   // LDC clip if OOB. offset
buffer_load_d16_b16 v63, v64, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v64, v39, v40, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v64, v42, v64, s30                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
v_add_co_u32 v40, vcc_lo, v36, 10                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v40, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v37, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v66, v38, v40, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v66, v42, v66, s30                   // LDC clip if OOB. offset
buffer_load_d16_b16 v65, v66, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v66, v39, v40, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v66, v42, v66, s30                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
v_add_co_u32 v40, vcc_lo, v36, 12                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v40, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v37, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v68, v38, v40, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v68, v42, v68, s30                   // LDC clip if OOB. offset
buffer_load_d16_b16 v67, v68, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v68, v39, v40, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v68, v42, v68, s30                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
v_add_co_u32 v40, vcc_lo, v36, 14                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v40, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s30, v37, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v70, v38, v40, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v70, v42, v70, s30                   // LDC clip if OOB. offset
buffer_load_d16_b16 v69, v70, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v70, v39, v40, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v70, v42, v70, s30                   // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0)] */
v_mul_f32 v[vgprValuC+47], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+48], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+49], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+50], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+51], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+52], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+53], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+54], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fma_mix_f32 v[vgprValuC+47], s[sgprBeta], v55, v[vgprValuC+47] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v47, v[vgprValuC+47]                 // convert C to fp16
buffer_store_b16 v47, v56, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+48], s[sgprBeta], v57, v[vgprValuC+48] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v48, v[vgprValuC+48]                 // convert C to fp16
buffer_store_b16 v48, v58, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+49], s[sgprBeta], v59, v[vgprValuC+49] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v49, v[vgprValuC+49]                 // convert C to fp16
buffer_store_b16 v49, v60, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+50], s[sgprBeta], v61, v[vgprValuC+50] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v50, v[vgprValuC+50]                 // convert C to fp16
buffer_store_b16 v50, v62, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+51], s[sgprBeta], v63, v[vgprValuC+51] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v51, v[vgprValuC+51]                 // convert C to fp16
buffer_store_b16 v51, v64, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+52], s[sgprBeta], v65, v[vgprValuC+52] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v52, v[vgprValuC+52]                 // convert C to fp16
buffer_store_b16 v52, v66, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+53], s[sgprBeta], v67, v[vgprValuC+53] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v53, v[vgprValuC+53]                 // convert C to fp16
buffer_store_b16 v53, v68, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+54], s[sgprBeta], v69, v[vgprValuC+54] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v54, v[vgprValuC+54]                 // convert C to fp16
buffer_store_b16 v54, v70, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_End_1:
label_KernelEnd:
s_endpgm                                           // Kernel End
label_ASM_End:  /// The end of the kernel
