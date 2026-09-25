
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
.text
.protected Custom_Cijk_Alik_Bljk_I4H_HHS_BH_SABB128_UserArgs_MT64x160x64_MI16x16x1_gfx1151
.globl Custom_Cijk_Alik_Bljk_I4H_HHS_BH_SABB128_UserArgs_MT64x160x64_MI16x16x1_gfx1151
.p2align 8
.type Custom_Cijk_Alik_Bljk_I4H_HHS_BH_SABB128_UserArgs_MT64x160x64_MI16x16x1_gfx1151,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Custom_Cijk_Alik_Bljk_I4H_HHS_BH_SABB128_UserArgs_MT64x160x64_MI16x16x1_gfx1151
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 256 // vgprs
  .amdhsa_next_free_sgpr 80 // sgprs
  .amdhsa_group_segment_fixed_size 65024 // lds bytes
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
/* Num SGPR   =80 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 16 x 5 */
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
  - .name: Custom_Cijk_Alik_Bljk_I4H_HHS_BH_SABB128_UserArgs_MT64x160x64_MI16x16x1_gfx1151
    .symbol: 'Custom_Cijk_Alik_Bljk_I4H_HHS_BH_SABB128_UserArgs_MT64x160x64_MI16x16x1_gfx1151.kd'
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
      - .name:            batchOffsetD
        .size:            8
        .offset:          120
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetC
        .size:            8
        .offset:          128
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetA
        .size:            8
        .offset:          136
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetB
        .size:            8
        .offset:          144
        .value_kind:      by_value
        .value_type:      u64
    .group_segment_fixed_size:   65024
    .kernarg_segment_align:      8
    .kernarg_segment_size:       152
    .max_flat_workgroup_size:    128
    .private_segment_fixed_size: 0
    .sgpr_count:                 80
    .sgpr_spill_count:           0
    .vgpr_count:                 256
    .vgpr_spill_count:           0
    .wavefront_size:             32
...
.end_amdgpu_metadata
Custom_Cijk_Alik_Bljk_I4H_HHS_BH_SABB128_UserArgs_MT64x160x64_MI16x16x1_gfx1151:
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
/* ValuC range: [0-80), serializedStore enabled */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprBase, 108
.set vgprLocalWriteAddrA, 102
.set vgprLocalWriteAddrB, 103
.set vgprGlobalReadOffsetA, 80
.set vgprGlobalReadOffsetB, 84
.set vgprGlobalReadOffsetScaleA, 94
.set vgprG2LScaleA, 98
.set vgprLocalReadAddrA, 104
.set vgprLocalReadAddrB, 105
.set vgprSerial, 222

/******************************************/
/* VGPR Macro Assignments                 */
/******************************************/
.set vgprValuA_X0_I0_BASE, vgprBase+0
.set vgprValuB_X0_I0_BASE, vgprBase+17
.set vgprG2LA_BASE, vgprBase+58
.set vgprG2LB_BASE, vgprBase+74
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
.set sgprScaleAKCnt, 56

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
.set MT1, 160
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
s_mov_b32 m0, 0xfe00                               // LDS clamp at 65024 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id
s_mov_b32 vcc_hi, 0                                // Ensure hi bits are zero

/* remap workgroup to XCCs */
s_lshr_b32 s62, s[sgprWGM], 0x10                   // Get WGMXCC
s_ff1_i32_b32 s62, s62                             // Get log(WGMXCC)
s_lshr_b32 s63, s[sgprWGM], 0x16                   // Get CU_Count
/* remap WGs if WGMXCC > 1 ( log(WGMXCC) > 0 ) */
s_cmp_gt_i32 s62, 0
s_cbranch_scc0 label_skip_WGMXCC
/* only remap WGs in the range */
s_lshr_b32 s59, s23, s62
s_lshl_b32 s59, s59, s62
s_cmp_ge_u32 s[sgprWorkGroup0], s59
s_cbranch_scc1 label_skip_WGMXCC
s_cmp_eq_u32 s63, 0                                // CU_Count == 0 ?
s_cbranch_scc0 label_XCCG_nonzero
s_lshr_b32 s59, s[sgprWorkGroup0], s62
s_bfm_b32 s60, s62, 0
s_and_b32 s60, s[sgprWorkGroup0], s60
s_lshr_b32 s61, s23, s62
s_mul_i32 s60, s60, s61
s_add_u32 s[sgprWorkGroup0], s59, s60
s_branch label_skip_WGMXCC
label_XCCG_nonzero:
/* temp0 = (wg//CU_Count)*CU_Count */
v_cvt_f64_u32 v[6:7], s63                          // s59 = s[sgprWorkGroup0] / s63
v_rcp_f64 v[6:7], v[6:7]                           // s59 = s[sgprWorkGroup0] / s63
v_cvt_f64_u32 v[8:9], s[sgprWorkGroup0]            // s59 = s[sgprWorkGroup0] / s63
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s59 = s[sgprWorkGroup0] / s63
v_cvt_u32_f64 v6, v[6:7]                           // s59 = s[sgprWorkGroup0] / s63
v_mul_lo_u32 v7, v6, s63                           // s59 = s[sgprWorkGroup0] / s63
v_sub_nc_u32 v8, s[sgprWorkGroup0], v7             // s59 = s[sgprWorkGroup0] / s63
v_cmp_ge_u32 vcc_lo, v8, s63                       // s59 = s[sgprWorkGroup0] / s63
s_mov_b32 exec_lo, vcc_lo                          // s59 = s[sgprWorkGroup0] / s63
v_add_nc_u32 v6, v6, 1                             // s59 = s[sgprWorkGroup0] / s63
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v7, v6, s63                           // s59 = s[sgprWorkGroup0] / s63
v_sub_nc_u32 v8, s[sgprWorkGroup0], v7             // s59 = s[sgprWorkGroup0] / s63
v_readfirstlane_b32 s59, v6                        // quotient
v_readfirstlane_b32 s60, v8                        // remainder
s_mul_i32 s59, s59, s63
/* temp1 = (wg%CU_Count)//WGMXCC */
s_lshr_b32 s60, s60, s62
/* temp0 = temp0 + temp1 */
s_add_u32 s59, s59, s60
/* temp1 = (wg%WGMXCC) * ((WGs - (WGs//CU_Count) * CU_Count) if (wg > (WGs//CU_Count) * CU_Count) else CU_Count)//WGMXCC */
v_cvt_f64_u32 v[6:7], s63                          // s60 = s23 / s63
v_rcp_f64 v[6:7], v[6:7]                           // s60 = s23 / s63
v_cvt_f64_u32 v[8:9], s23                          // s60 = s23 / s63
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s60 = s23 / s63
v_cvt_u32_f64 v6, v[6:7]                           // s60 = s23 / s63
v_mul_lo_u32 v7, v6, s63                           // s60 = s23 / s63
v_sub_nc_u32 v8, s23, v7                           // s60 = s23 / s63
v_cmp_ge_u32 vcc_lo, v8, s63                       // s60 = s23 / s63
s_mov_b32 exec_lo, vcc_lo                          // s60 = s23 / s63
v_add_nc_u32 v6, v6, 1                             // s60 = s23 / s63
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s60, v6                        // quotient
s_mul_i32 s60, s60, s63
s_sub_u32 s61, s23, s60
s_cmp_gt_u32 s[sgprWorkGroup0], s60
s_cselect_b32 s60, s61, s63
s_lshr_b32 s60, s60, s62
s_bfm_b32 s61, s62, 0
s_and_b32 s61, s[sgprWorkGroup0], s61
s_mul_i32 s60, s60, s61
/* WorkGroup0 = temp0 + temp1 */
s_add_u32 s[sgprWorkGroup0], s59, s60
label_skip_WGMXCC:  /// skip WGMXCC if no enough WGs to remap
s_cmp_eq_u32 s21, 3
s_cbranch_scc1 label_ArgType3_Routed_To_ArgType0
s_cmp_eq_u32 s21, 0
s_cbranch_scc0 label_MultiGemm
label_ArgType3_Routed_To_ArgType0:
/* init: add vgpr [108...273) to pool */
/* init: add vgpr [0...80) to pool */
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
v_and_b32 v4, 1, v4                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v0, v4, 10, v0                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(1024); 7. final local read offset: flrOffset = lrOffset + WOffset
/* lr1J */
v_and_b32 v2, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v1, 6, v1                            // 1. N offset: nOffset = nIdx * nStride(64)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v3, 6, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
v_and_b32 v3, 1, v3                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v1, v3, 10, v1                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(1024); 7. final local read offset: flrOffset = lrOffset + WOffset

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
s_and_b32 s16, s[sgprArgType], 0xff                // mask ArgType domain (bit 8 = TDM wave-parity)
s_cmp_eq_u32 s16, 2                                // ArgType == 2 ?
s_cbranch_scc1 label_IsExternalValid               // branch if ArgType == 2
s_mov_b32 s15, 104                                 // KernArgAddressOffset
s_mul_i32 s64, s20, 4
s_mov_b64 s[58:59], s[sgprKernArgAddress:sgprKernArgAddress+1]
s_branch label_IsExternalValidEnd
label_IsExternalValid:
s_mov_b32 s15, 228
s_mov_b32 s64, 0
s_mov_b64 s[58:59], s[sgprKernArgAddress:sgprKernArgAddress+1]
label_IsExternalValidEnd:

/* Grouped Gemm:: prefetch 1 arg load */
s_mov_b32 s14, 1
s_mov_b32 s65, 0
s_load_b128 s[24:27], s[58:59], s64
s_cmpk_eq_u32 s20, 1                               // if gemm_count is 1?
s_cbranch_scc1 label_wgTable_noLoadLoop

/* Grouped Gemm:: accumulate numTiles for each gemm */
/* Grouped Gemm:: loop start */
label_Loop_GemmCount:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s62, s24, 6                             // s62 = s24 / 64
s_and_b32 s60, 63, s24                             // s60 = s24 % 64
s_addc_u32 s62, s62, 0
s_mov_b32 s61, 0                                   // STATIC_DIV: divisor=160
s_mul_i32 s60, 819, s25                            // tmp1 = dividend * magic hi
s_lshl_b64 s[60:61], s[60:61], 16                  // left shift 16 bits
s_mul_i32 s63, s25, 13108                          // tmp0 = dividend * magic lo
s_add_u32 s60, s63, s60                            // add lo
s_addc_u32 s61, s61, 0                             // add hi
s_lshr_b64 s[60:61], s[60:61], 33                  // tmp0 = quotient
s_mul_i32 s61, s60, 160                            // tmp1 = quotient * divisor
s_cmp_lg_u32 s61, s25                              // if (quotient * divisor != dividend), result+=1
s_addc_u32 s63, s60, 0                             // if (quotient * divisor != dividend), result+=1
s_mul_i32 s62, s62, s63
s_mul_i32 s62, s62, s26
s_and_b32 s63, s[sgprGSU], 0xfff                   // Restore GSU
s_mul_i32 s62, s62, s63
s_add_u32 s65, s65, s62
s_cmp_lt_u32 s[sgprWorkGroup0], s65
s_cbranch_scc1 label_FOUND
s_add_u32 s64, s64, s15
s_load_b128 s[24:27], s[58:59], s64
s_add_u32 s14, s14, 1
s_cmp_lt_u32 s14, s20
s_cbranch_scc1 label_Loop_GemmCount

/* Grouped Gemm:: noLoadLoop */
label_wgTable_noLoadLoop:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s62, s24, 6                             // s62 = s24 / 64
s_and_b32 s60, 63, s24                             // s60 = s24 % 64
s_addc_u32 s62, s62, 0
s_mov_b32 s61, 0                                   // STATIC_DIV: divisor=160
s_mul_i32 s60, 819, s25                            // tmp1 = dividend * magic hi
s_lshl_b64 s[60:61], s[60:61], 16                  // left shift 16 bits
s_mul_i32 s63, s25, 13108                          // tmp0 = dividend * magic lo
s_add_u32 s60, s63, s60                            // add lo
s_addc_u32 s61, s61, 0                             // add hi
s_lshr_b64 s[60:61], s[60:61], 33                  // tmp0 = quotient
s_mul_i32 s61, s60, 160                            // tmp1 = quotient * divisor
s_cmp_lg_u32 s61, s25                              // if (quotient * divisor != dividend), result+=1
s_addc_u32 s63, s60, 0                             // if (quotient * divisor != dividend), result+=1
s_mul_i32 s62, s62, s63
s_mul_i32 s62, s62, s26
s_and_b32 s58, s[sgprGSU], 0xfff                   // Restore GSU
s_mul_i32 s62, s62, s58
s_add_u32 s65, s65, s62

/* Grouped Gemm:: gemmIndex found */
label_FOUND:
s_sub_u32 s59, s14, 1
s_sub_u32 s58, s65, s62
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s58
/* Check if custom structure pointer is null */
s_and_b32 s16, s[sgprArgType], 0xff                // mask ArgType domain (bit 8 = TDM wave-parity)
s_cmp_eq_u32 s16, 2                                // ArgType == 2 ?
s_cbranch_scc1 label_LoadExternalStruct            // branch if ArgType == 2

/* Grouped Gemm: offset argument address to gemm */
/* Grouped Gemm: offset address from wg_table_start to args_start */
s_lshl2_add_u32 s[sgprKernArgAddress], s20, s[sgprKernArgAddress]
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s59, s59, 104                            // KernArgAddressOffset
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s59
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_b512 s[28:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_b64 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
s_load_b64 s[sgprAddressScaleA:sgprAddressScaleA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x58
s_load_b64 s[sgprAddressScaleB:sgprAddressScaleB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x60
s_branch label_LoadExternalStructEnd
label_LoadExternalStruct:
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s59, s59, 228
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s59
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
/* init: add vgpr [108...273) to pool */
/* init: add vgpr [0...80) to pool */
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
v_and_b32 v4, 1, v4                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v0, v4, 10, v0                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(1024); 7. final local read offset: flrOffset = lrOffset + WOffset
/* lr1J */
v_and_b32 v2, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v1, 6, v1                            // 1. N offset: nOffset = nIdx * nStride(64)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v3, 6, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
v_and_b32 v3, 1, v3                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v1, v3, 10, v1                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(1024); 7. final local read offset: flrOffset = lrOffset + WOffset

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
.set sgprSrdA, 60
.set sgprSrdB, 64
.set sgprShadowLimitA, 58
.set sgprShadowLimitB, 68
.set sgprStaggerUIter, 57
.set sgprWrapUA, 70
.set sgprWrapUB, 72
.set sgprGlobalReadIncsA, 74
.set sgprGlobalReadIncsB, 75
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
v_add_co_u32 v14, vcc_lo, 16, v13                  // groB1J_4 += LSPB
v_add_co_u32 v15, vcc_lo, 16, v14                  // groB1J_5 += LSPB
v_add_co_u32 v16, vcc_lo, 16, v15                  // groB1J_6 += LSPB
v_add_co_u32 v17, vcc_lo, 16, v16                  // groB1J_7 += LSPB
v_add_co_u32 v18, vcc_lo, 16, v17                  // groB1J_8 += LSPB
v_add_co_u32 v19, vcc_lo, 16, v18                  // groB1J_9 += LSPB

/* global read addresses: unroll offsets a */
v_mov_b32 v20, v1                                  // groAL_0

/* global read addresses: unroll offsets b */
v_mov_b32 v21, v3                                  // groBL_0

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
v_cvt_f32_u32 v22, s[sgprGSUSumIdx+1]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v22, v22                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v23, s[sgprLoopCounterL]             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v22, v22, v23                            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v22, v22                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v23, v22, s[sgprGSUSumIdx+1]         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_nc_u32 v23, s[sgprLoopCounterL], v23         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmp_eq_u32 vcc_lo, v23, s[sgprGSUSumIdx+1]       // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, vcc_lo                          // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_nc_u32 v22, 1, v22                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v23, 0                                   // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v23, s[sgprGSUSumIdx+1]       // overflow happened in remainder
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v22, v22, 1                           // quotient - 1
v_mul_u32_u24 v23, v22, s[sgprGSUSumIdx+1]         // re-calculate remainder
v_sub_nc_u32 v23, s[sgprLoopCounterL], v23         // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v22       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v23        // remainder
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
s_load_b64 s[16:17], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x78 // Load batchOffsetA from kernel args
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
s_mul_i32 s16, s[sgprWorkGroup0], 64               // scaleA: workgroup row origin
s_mul_i32 s16, s16, s[sgprStrideScaleA]            // scaleA: * row stride
s_mul_i32 s16, s16, 2                              // scaleA: elements -> bytes
s_mul_i32 s17, s[sgprSizeI], s[sgprStrideScaleA]   // scaleA: SizeI * row stride
s_mul_i32 s17, s17, 2                              // scaleA: tensor bytes
s_sub_u32 s[sgprSrdScaleA+2], s17, s16             // scaleA: buffer limit from the workgroup origin
s_add_u32 s[sgprSrdScaleA+0], s[sgprAddressScaleA+0], s16 // scaleA: SRD base lo
s_addc_u32 s[sgprSrdScaleA+1], s[sgprAddressScaleA+1], 0 // scaleA: SRD base hi
s_mov_b32 s[sgprSrdScaleA+3], Srd127_96            // scaleA: set bits 127_96 in SRD
s_mov_b32 s[sgprScaleAKCnt], 0                     // scaleA: K-iteration counter within a group

/* global read addresses: addresses b */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s19, s[sgprWorkGroup1], 160           // WorkGroup[01] * MT
s_mul_i32 s18, s[sgprWorkGroup1], 160              // WorkGroup[01] * MT
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
v_cvt_f32_u32 v22, s[sgprGSUSumIdx+1]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v22, v22                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v23, s[sgprLoopCounterL]             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v22, v22, v23                            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v22, v22                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v23, v22, s[sgprGSUSumIdx+1]         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_nc_u32 v23, s[sgprLoopCounterL], v23         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmp_eq_u32 vcc_lo, v23, s[sgprGSUSumIdx+1]       // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, vcc_lo                          // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_nc_u32 v22, 1, v22                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v23, 0                                   // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v23, s[sgprGSUSumIdx+1]       // overflow happened in remainder
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v22, v22, 1                           // quotient - 1
v_mul_u32_u24 v23, v22, s[sgprGSUSumIdx+1]         // re-calculate remainder
v_sub_nc_u32 v23, s[sgprLoopCounterL], v23         // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v22       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v23        // remainder
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
s_load_b64 s[16:17], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x80 // Load batchOffsetB from kernel args
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
v_mul_lo_u32 v22, s[sgprStrideA0I], v[6]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+0+0], vcc_lo, v[20], v[22+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+0+0], 0x8, v[vgprGlobalReadOffsetA+0+0] // add prepad for pointer shift
v_lshrrev_b32 v22, 7, v20                          // scaleA: kGroup = k/128
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+0], s[sgprStrideScaleA], v6 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+0], v22, v[vgprGlobalReadOffsetScaleA+0] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+0], 1, v[vgprGlobalReadOffsetScaleA+0] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetA+0], 1, v[vgprGlobalReadOffsetA+0] //  (multiple bpe)
v_mul_lo_u32 v22, s[sgprStrideA0I], v[7]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+1+0], vcc_lo, v[20], v[22+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+1+0], 0x8, v[vgprGlobalReadOffsetA+1+0] // add prepad for pointer shift
v_lshrrev_b32 v22, 7, v20                          // scaleA: kGroup = k/128
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+1], s[sgprStrideScaleA], v7 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+1], v22, v[vgprGlobalReadOffsetScaleA+1] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+1], 1, v[vgprGlobalReadOffsetScaleA+1] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetA+1], 1, v[vgprGlobalReadOffsetA+1] //  (multiple bpe)
v_mul_lo_u32 v22, s[sgprStrideA0I], v[8]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+2+0], vcc_lo, v[20], v[22+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+2+0], 0x8, v[vgprGlobalReadOffsetA+2+0] // add prepad for pointer shift
v_lshrrev_b32 v22, 7, v20                          // scaleA: kGroup = k/128
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+2], s[sgprStrideScaleA], v8 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+2], v22, v[vgprGlobalReadOffsetScaleA+2] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+2], 1, v[vgprGlobalReadOffsetScaleA+2] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetA+2], 1, v[vgprGlobalReadOffsetA+2] //  (multiple bpe)
v_mul_lo_u32 v22, s[sgprStrideA0I], v[9]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+3+0], vcc_lo, v[20], v[22+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+3+0], 0x8, v[vgprGlobalReadOffsetA+3+0] // add prepad for pointer shift
v_lshrrev_b32 v22, 7, v20                          // scaleA: kGroup = k/128
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+3], s[sgprStrideScaleA], v9 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+3], v22, v[vgprGlobalReadOffsetScaleA+3] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+3], 1, v[vgprGlobalReadOffsetScaleA+3] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetA+3], 1, v[vgprGlobalReadOffsetA+3] //  (multiple bpe)
/* ============================================================= */

/* global read addresses: final offsets b */
/* ============================================================= */
v_mul_lo_u32 v6, s[sgprStrideB1J], v[10]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+0+0], vcc_lo, v[21], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+0+0], 0x8, v[vgprGlobalReadOffsetB+0+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+0], 1, v[vgprGlobalReadOffsetB+0] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[11]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+1+0], vcc_lo, v[21], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+1+0], 0x8, v[vgprGlobalReadOffsetB+1+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+1], 1, v[vgprGlobalReadOffsetB+1] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[12]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+2+0], vcc_lo, v[21], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+2+0], 0x8, v[vgprGlobalReadOffsetB+2+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+2], 1, v[vgprGlobalReadOffsetB+2] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[13]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+3+0], vcc_lo, v[21], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+3+0], 0x8, v[vgprGlobalReadOffsetB+3+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+3], 1, v[vgprGlobalReadOffsetB+3] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[14]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+4+0], vcc_lo, v[21], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+4+0], 0x8, v[vgprGlobalReadOffsetB+4+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+4], 1, v[vgprGlobalReadOffsetB+4] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[15]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+5+0], vcc_lo, v[21], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+5+0], 0x8, v[vgprGlobalReadOffsetB+5+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+5], 1, v[vgprGlobalReadOffsetB+5] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[16]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+6+0], vcc_lo, v[21], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+6+0], 0x8, v[vgprGlobalReadOffsetB+6+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+6], 1, v[vgprGlobalReadOffsetB+6] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[17]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+7+0], vcc_lo, v[21], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+7+0], 0x8, v[vgprGlobalReadOffsetB+7+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+7], 1, v[vgprGlobalReadOffsetB+7] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[18]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+8+0], vcc_lo, v[21], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+8+0], 0x8, v[vgprGlobalReadOffsetB+8+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+8], 1, v[vgprGlobalReadOffsetB+8] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[19]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+9+0], vcc_lo, v[21], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+9+0], 0x8, v[vgprGlobalReadOffsetB+9+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+9], 1, v[vgprGlobalReadOffsetB+9] //  (multiple bpe)
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
buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b128 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b128 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b128 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0
buffer_load_b128 v[vgprG2LB+16:vgprG2LB+16+3], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_4_0
buffer_load_b128 v[vgprG2LB+20:vgprG2LB+20+3], v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_5_0
buffer_load_b128 v[vgprG2LB+24:vgprG2LB+24+3], v[vgprGlobalReadOffsetB+6], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_6_0
buffer_load_b128 v[vgprG2LB+28:vgprG2LB+28+3], v[vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_7_0
buffer_load_b128 v[vgprG2LB+32:vgprG2LB+32+3], v[vgprGlobalReadOffsetB+8], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_8_0
buffer_load_b128 v[vgprG2LB+36:vgprG2LB+36+3], v[vgprGlobalReadOffsetB+9], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_9_0

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

/* global read inc block-scale A (2 bytes every 2 iters) */
s_add_u32 s[sgprScaleAKCnt], s[sgprScaleAKCnt], 1  // scaleA: one more K iteration done
s_and_b32 s16, s[sgprScaleAKCnt], 0x1              // scaleA: SCC = counter has not wrapped
s_cselect_b32 s16, 0, 0x2                          // scaleA: advance only on a wrap
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], s16 // scaleA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], s16 // scaleA limit -= inc

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
s_and_b32 s76, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s76, 1                                // GSU == 1 ?
s_cbranch_scc1 label_ArgTypeCheckD                 // Handling General Batched GEMM SRD initialization
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], s[sgprAddressD+0:sgprAddressD+0+1] // init SRD base address
s_branch label_GeneralBatchedGemmSrdInitiationD_End // End of handling General Batched GEMM SRD initialization
label_ArgTypeCheckD:  /// Check if ArgType is for General Batched GEMM for D
s_and_b32 s76, s[sgprArgType], 0xff                // mask ArgType domain (bit 8 = TDM wave-parity)
s_cmp_eq_u32 s76, 3                                // ArgType == 3 for General Batched GEMM
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

s_and_b32 s76, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s76, 1                                // GSU == 1 ?
s_cbranch_scc1 label_ArgTypeCheckC                 // Handling General Batched GEMM SRD initialization
s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], s[sgprAddressC+0:sgprAddressC+0+1] // init SRD base address
s_branch label_GeneralBatchedGemmSrdInitiationC_End // End of handling General Batched GEMM SRD initialization
label_ArgTypeCheckC:  /// Check if ArgType is for General Batched GEMM for C
s_and_b32 s76, s[sgprArgType], 0xff                // mask ArgType domain (bit 8 = TDM wave-parity)
s_cmp_eq_u32 s76, 3                                // ArgType == 3 for General Batched GEMM
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


s_mul_i32 s78, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_and_b32 s77, s[sgprGSU], 0xfff                   // Restore GSU
s_mul_hi_u32 s77, s78, s[sgprStrideC1J]            // ScaleC s78 by Stride
s_mul_i32 s76, s78, s[sgprStrideC1J]               // ScaleC s78 by Stride
s_lshl_b64 s[76:77], s[76:77], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s76        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s77       // add hi to SRD
s_and_b32 s77, s[sgprGSU], 0xfff                   // Restore GSU
s_mul_hi_u32 s77, s78, s[sgprStrideD1J]            // ScaleD s78 by Stride
s_mul_i32 s76, s78, s[sgprStrideD1J]               // ScaleD s78 by Stride
s_lshl_b64 s[76:77], s[76:77], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s76        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s77       // add hi to SRD

s_and_b32 s77, s[sgprGSU], 0xfff                   // Restore GSU
s_cmp_eq_u32 s77, 1                                // GSU == 1 ?
s_cbranch_scc0 label_StridedBatchedGemmLoadC
s_and_b32 s79, s[sgprArgType], 0xff                // mask ArgType domain (bit 8 = TDM wave-parity)
s_cmp_eq_u32 s79, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_GeneralBatchedGemmLoadC
label_StridedBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s77, s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride
s_mul_i32 s76, s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride
s_lshl_b64 s[76:77], s[76:77], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s76        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s77       // add hi to SRD
s_branch label_GeneralBatchedGemmLoadC_End
label_GeneralBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for General Batched GEMM
s_mul_i32 s76, 8, s[sgprWorkGroup2]                // Compute stride in bytes into Pointer Array
s_add_u32 s76, s76, s[sgprAddressC+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s77, s[sgprAddressC+1], 0               // Offsetting to the location [Higher half of address]
s_load_b64 s[76:77], s[76:77], 0                   // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for the Matrix Address Load from the Pointer Array
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s76        // Offsetting within the Batch Matrix [Lower half of address]
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s77       // Offsetting within the Batch Matrix [Higher half of address]
s_load_b64 s[76:77], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x70 // Load batchOffsetC from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s76        // Add matrix address to SRD (low)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s77       // Add matrix address to SRD (high)
label_GeneralBatchedGemmLoadC_End:  /// End of label GeneralBatchedGemmLoadC
s_and_b32 s77, s[sgprGSU], 0xfff                   // Restore GSU
s_cmp_eq_u32 s77, 1                                // GSU == 1 ?
s_cbranch_scc0 label_StridedBatchedGemmLoadD
s_and_b32 s79, s[sgprArgType], 0xff                // mask ArgType domain (bit 8 = TDM wave-parity)
s_cmp_eq_u32 s79, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_GeneralBatchedGemmLoadD
label_StridedBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s77, s[sgprWorkGroup2], s[sgprStrideDK] // ScaleD s[sgprWorkGroup2] by Stride
s_mul_i32 s76, s[sgprWorkGroup2], s[sgprStrideDK]  // ScaleD s[sgprWorkGroup2] by Stride
s_lshl_b64 s[76:77], s[76:77], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s76        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s77       // add hi to SRD
s_branch label_GeneralBatchedGemmLoadD_End
label_GeneralBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for General Batched GEMM
s_mul_i32 s76, 8, s[sgprWorkGroup2]                // Compute stride in bytes into Pointer Array
s_add_u32 s76, s76, s[sgprAddressD+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s77, s[sgprAddressD+1], 0               // Offsetting to the location [Higher half of address]
s_load_b64 s[76:77], s[76:77], 0                   // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for the Matrix Address Load from the Pointer Array
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s76        // Offsetting within the Batch Matrix [Lower half of address]
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s77       // Offsetting within the Batch Matrix [Higher half of address]
s_load_b64 s[76:77], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x68 // Load batchOffsetD from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s76        // Add matrix address to SRD (low)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s77       // Add matrix address to SRD (high)
label_GeneralBatchedGemmLoadD_End:  /// End of label GeneralBatchedGemmLoadD

s_and_b32 s76, s[sgprGSU], 0xfff                   // Restore GSU
s_cmp_eq_u32 s76, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU_2                         // branch if GSU == 1
// GSU Output Buffer offset: Free0 + (Free1-1)*StrideC1J + (Free2-1)*StrideCK * GSUIdx * bpe%s
s_mul_hi_u32 s77, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_mul_i32 s76, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_sub_u32 s78, s[sgprSizesFree+1], 1               // Free1
s_mul_i32 s78, s78, s[sgprGSUSumIdx]               // Free1
s_mul_hi_u32 s79, s78, s[sgprStrideC1J]            // Free1
s_mul_i32 s78, s78, s[sgprStrideC1J]               // Free1
s_add_u32 s76, s76, s78                            // Free1
s_addc_u32 s77, s77, s79                           // Free1
s_sub_u32 s78, s[sgprSizesFree+2], 1               // Free2
s_mul_i32 s78, s78, s[sgprGSUSumIdx]               // Free2
s_mul_hi_u32 s79, s78, s[sgprStrideCK]             // Free2
s_mul_i32 s78, s78, s[sgprStrideCK]                // Free2
s_add_u32 s76, s76, s78                            // Free2
s_addc_u32 s77, s77, s79                           // Free2
s_lshl_b64 s[76:77], s[76:77], 2                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s76        // add lo GSU offset to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s77       // add hi GSU offset to SRD
label_GSU_2:
.set sgprGSULog2BpeC, UNDEF
.set sgprAddressC, UNDEF

/* initC: remove ValuC vgpr buffer [0...80) from pool */

/* initC: remove acc vgpr buffer [0...0) from pool */

/* initC: remove ValuA/B vgpr buffer [108...165) from pool */
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
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?

/* after InitC, skip to end of prefetch last iter if numIter==0 */

/* label_PrefetchGlobalLastIterEnd */
s_cbranch_scc0 label_NoBranch_0                    // Only branch on scc1
s_getpc_b64 s[76:77]                               // addr of next instr
s_add_i32 s78, label_PrefetchGlobalLastIterEnd, 4  // target branch offset
s_add_u32 s76, s76, s78                            // add target branch offset
s_addc_u32 s77, s77, 0                             // add high and carry
s_setpc_b64 s[76:77]                               // branch to label_PrefetchGlobalLastIterEnd
label_NoBranch_0:
s_waitcnt vmcnt(0)                                 // wait for global read

/* local write a */
v_cvt_f32_f16 v226, v[vgprG2LScaleA+0]             // scaleA: fp16 -> f32
v_mov_b32 v225, v[vgprG2LA+2+0]                    // w4a16: save packed int4 dword 0
v_bfe_i32 v223, v225, 0x0, 0x4                     // w4a16: sign-extend int4 #0
v_bfe_i32 v224, v225, 0x4, 0x4                     // w4a16: sign-extend int4 #1
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+0+0], v223, v224         // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x8, 0x4                     // w4a16: sign-extend int4 #2
v_bfe_i32 v224, v225, 0xc, 0x4                     // w4a16: sign-extend int4 #3
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+0+1], v223, v224         // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x10, 0x4                    // w4a16: sign-extend int4 #4
v_bfe_i32 v224, v225, 0x14, 0x4                    // w4a16: sign-extend int4 #5
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+0+2], v223, v224         // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x18, 0x4                    // w4a16: sign-extend int4 #6
v_bfe_i32 v224, v225, 0x1c, 0x4                    // w4a16: sign-extend int4 #7
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+0+3], v223, v224         // w4a16: pack 2 fp16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
v_cvt_f32_f16 v226, v[vgprG2LScaleA+1]             // scaleA: fp16 -> f32
v_mov_b32 v225, v[vgprG2LA+6+0]                    // w4a16: save packed int4 dword 0
v_bfe_i32 v223, v225, 0x0, 0x4                     // w4a16: sign-extend int4 #0
v_bfe_i32 v224, v225, 0x4, 0x4                     // w4a16: sign-extend int4 #1
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+4+0], v223, v224         // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x8, 0x4                     // w4a16: sign-extend int4 #2
v_bfe_i32 v224, v225, 0xc, 0x4                     // w4a16: sign-extend int4 #3
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+4+1], v223, v224         // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x10, 0x4                    // w4a16: sign-extend int4 #4
v_bfe_i32 v224, v225, 0x14, 0x4                    // w4a16: sign-extend int4 #5
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+4+2], v223, v224         // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x18, 0x4                    // w4a16: sign-extend int4 #6
v_bfe_i32 v224, v225, 0x1c, 0x4                    // w4a16: sign-extend int4 #7
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+4+3], v223, v224         // w4a16: pack 2 fp16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:2304 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 2304 sync LDS0
v_cvt_f32_f16 v226, v[vgprG2LScaleA+2]             // scaleA: fp16 -> f32
v_mov_b32 v225, v[vgprG2LA+10+0]                   // w4a16: save packed int4 dword 0
v_bfe_i32 v223, v225, 0x0, 0x4                     // w4a16: sign-extend int4 #0
v_bfe_i32 v224, v225, 0x4, 0x4                     // w4a16: sign-extend int4 #1
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+8+0], v223, v224         // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x8, 0x4                     // w4a16: sign-extend int4 #2
v_bfe_i32 v224, v225, 0xc, 0x4                     // w4a16: sign-extend int4 #3
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+8+1], v223, v224         // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x10, 0x4                    // w4a16: sign-extend int4 #4
v_bfe_i32 v224, v225, 0x14, 0x4                    // w4a16: sign-extend int4 #5
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+8+2], v223, v224         // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x18, 0x4                    // w4a16: sign-extend int4 #6
v_bfe_i32 v224, v225, 0x1c, 0x4                    // w4a16: sign-extend int4 #7
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+8+3], v223, v224         // w4a16: pack 2 fp16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:4608 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 4608 sync LDS0
v_cvt_f32_f16 v226, v[vgprG2LScaleA+3]             // scaleA: fp16 -> f32
v_mov_b32 v225, v[vgprG2LA+14+0]                   // w4a16: save packed int4 dword 0
v_bfe_i32 v223, v225, 0x0, 0x4                     // w4a16: sign-extend int4 #0
v_bfe_i32 v224, v225, 0x4, 0x4                     // w4a16: sign-extend int4 #1
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+12+0], v223, v224        // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x8, 0x4                     // w4a16: sign-extend int4 #2
v_bfe_i32 v224, v225, 0xc, 0x4                     // w4a16: sign-extend int4 #3
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+12+1], v223, v224        // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x10, 0x4                    // w4a16: sign-extend int4 #4
v_bfe_i32 v224, v225, 0x14, 0x4                    // w4a16: sign-extend int4 #5
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+12+2], v223, v224        // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x18, 0x4                    // w4a16: sign-extend int4 #6
v_bfe_i32 v224, v225, 0x1c, 0x4                    // w4a16: sign-extend int4 #7
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+12+3], v223, v224        // w4a16: pack 2 fp16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+12:vgprG2LA+12+3] offset:6912 // lwoA_0_0_3_0 = (0*LSCA)*(MT0I+PAD) + (3*LSPA) = 6912 sync LDS0

/* local write b */
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4:vgprG2LB+4+3] offset:2304 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 2304 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8:vgprG2LB+8+3] offset:4608 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 4608 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12:vgprG2LB+12+3] offset:6912 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 6912 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+16:vgprG2LB+16+3] offset:9216 // lwoB_0_0_4_0 = (0*LSCB)*(MT1J+PAD) + (4*LSPB) = 9216 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+20:vgprG2LB+20+3] offset:11520 // lwoB_0_0_5_0 = (0*LSCB)*(MT1J+PAD) + (5*LSPB) = 11520 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+24:vgprG2LB+24+3] offset:13824 // lwoB_0_0_6_0 = (0*LSCB)*(MT1J+PAD) + (6*LSPB) = 13824 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+28:vgprG2LB+28+3] offset:16128 // lwoB_0_0_7_0 = (0*LSCB)*(MT1J+PAD) + (7*LSPB) = 16128 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+32:vgprG2LB+32+3] offset:18432 // lwoB_0_0_8_0 = (0*LSCB)*(MT1J+PAD) + (8*LSPB) = 18432 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+36:vgprG2LB+36+3] offset:20736 // lwoB_0_0_9_0 = (0*LSCB)*(MT1J+PAD) + (9*LSPB) = 20736 sync LDS0

/* local write swap a */
v_xor_b32 v[vgprLocalWriteAddrA], 0x8000, v[vgprLocalWriteAddrA] // swap Red Blk

/* local write swap b */
v_xor_b32 v[vgprLocalWriteAddrB], 0x8000, v[vgprLocalWriteAddrB] // swap Red Blk
s_cmp_eq_u32 s[sgprLoopCounterL], 0x1              // PGR=2 but only 1 loop
s_cbranch_scc1 label_skipPGR2_1                    // PGR=2 but only 1 loop
buffer_load_b32 v[vgprG2LA+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b32 v[vgprG2LA+6], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b32 v[vgprG2LA+10], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b32 v[vgprG2LA+14], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_3_0

/* global read block-scale A */
buffer_load_d16_b16 v[vgprG2LScaleA+0], v[vgprGlobalReadOffsetScaleA+0], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 0
buffer_load_d16_b16 v[vgprG2LScaleA+1], v[vgprGlobalReadOffsetScaleA+1], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 1
buffer_load_d16_b16 v[vgprG2LScaleA+2], v[vgprGlobalReadOffsetScaleA+2], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 2
buffer_load_d16_b16 v[vgprG2LScaleA+3], v[vgprGlobalReadOffsetScaleA+3], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 3
buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b128 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b128 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b128 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0
buffer_load_b128 v[vgprG2LB+16:vgprG2LB+16+3], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_4_0
buffer_load_b128 v[vgprG2LB+20:vgprG2LB+20+3], v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_5_0
buffer_load_b128 v[vgprG2LB+24:vgprG2LB+24+3], v[vgprGlobalReadOffsetB+6], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_6_0
buffer_load_b128 v[vgprG2LB+28:vgprG2LB+28+3], v[vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_7_0
buffer_load_b128 v[vgprG2LB+32:vgprG2LB+32+3], v[vgprGlobalReadOffsetB+8], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_8_0
buffer_load_b128 v[vgprG2LB+36:vgprG2LB+36+3], v[vgprGlobalReadOffsetB+9], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_9_0
s_branch label_skipPGR2_2                          // jump to PGR=2 label
label_skipPGR2_1:
label_skipPGR2_2:

/******************************************/
/* Unrolled Loop(s) - Begin               */
/******************************************/
label_openLoopL:
s_cmp_eq_u32 s[sgprLoopCounterL], 0x1              // LoopCounterL == 1 (PGR>=2, not Suppress: single-loop -> toPGR1)
s_cbranch_scc1 label_toPGR1                        // PGR=2 but only 1 loop, toPGR1
s_cmp_le_u32 s[sgprLoopCounterL], 0x2              // LoopCounterL < EndCounter
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

/* local read a */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4608 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4624 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read b */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4608 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4624 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:9216 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:9232 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:13824 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:13840 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprLocalReadAddrB+0] offset:18432 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[vgprLocalReadAddrB+0] offset:18448 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read increment a */
/* N/A, lro->16 */
/* localReadDoCntA 1 localReadDoCntMXSA 0 localReadDoCntB 1 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->16 */
/* localReadDoCntA 1 localReadDoCntMXSA 0 localReadDoCntB 1 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0 for iteration == 0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=10 */

/* iter 1 */

/* local read a */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4640 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4656 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read b */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4640 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4656 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:9248 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:9264 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:13856 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:13872 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprLocalReadAddrB+0] offset:18464 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[vgprLocalReadAddrB+0] offset:18480 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read increment a */
/* N/A, lro->32 */
/* localReadDoCntA 2 localReadDoCntMXSA 0 localReadDoCntB 2 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->32 */
/* localReadDoCntA 2 localReadDoCntMXSA 0 localReadDoCntB 2 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=0 readsPerIterB=10 */

/* iter 2 */

/* local read a */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4672 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4688 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read b */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4672 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4688 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:9280 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:9296 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:13888 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:13904 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprLocalReadAddrB+0] offset:18496 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[vgprLocalReadAddrB+0] offset:18512 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read increment a */
/* N/A, lro->48 */
/* localReadDoCntA 3 localReadDoCntMXSA 0 localReadDoCntB 3 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->48 */
/* localReadDoCntA 3 localReadDoCntMXSA 0 localReadDoCntB 3 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=0 readsPerIterB=10 */

/* iter 3 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */

/* local read a */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4704 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4720 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read b */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4704 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4720 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:9312 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:9328 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:13920 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:13936 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprLocalReadAddrB+0] offset:18528 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[vgprLocalReadAddrB+0] offset:18544 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt vmcnt(0)                                 // 1wait for global read

/* local write A */
v_cvt_f32_f16 v226, v[vgprG2LScaleA+0]             // scaleA: fp16 -> f32
v_mov_b32 v225, v[vgprG2LA+2+0]                    // w4a16: save packed int4 dword 0
v_bfe_i32 v223, v225, 0x0, 0x4                     // w4a16: sign-extend int4 #0
v_bfe_i32 v224, v225, 0x4, 0x4                     // w4a16: sign-extend int4 #1
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+0+0], v223, v224         // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x8, 0x4                     // w4a16: sign-extend int4 #2
v_bfe_i32 v224, v225, 0xc, 0x4                     // w4a16: sign-extend int4 #3
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+0+1], v223, v224         // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x10, 0x4                    // w4a16: sign-extend int4 #4
v_bfe_i32 v224, v225, 0x14, 0x4                    // w4a16: sign-extend int4 #5
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+0+2], v223, v224         // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x18, 0x4                    // w4a16: sign-extend int4 #6
v_bfe_i32 v224, v225, 0x1c, 0x4                    // w4a16: sign-extend int4 #7
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+0+3], v223, v224         // w4a16: pack 2 fp16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS1
v_cvt_f32_f16 v226, v[vgprG2LScaleA+1]             // scaleA: fp16 -> f32
v_mov_b32 v225, v[vgprG2LA+6+0]                    // w4a16: save packed int4 dword 0
v_bfe_i32 v223, v225, 0x0, 0x4                     // w4a16: sign-extend int4 #0
v_bfe_i32 v224, v225, 0x4, 0x4                     // w4a16: sign-extend int4 #1
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+4+0], v223, v224         // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x8, 0x4                     // w4a16: sign-extend int4 #2
v_bfe_i32 v224, v225, 0xc, 0x4                     // w4a16: sign-extend int4 #3
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+4+1], v223, v224         // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x10, 0x4                    // w4a16: sign-extend int4 #4
v_bfe_i32 v224, v225, 0x14, 0x4                    // w4a16: sign-extend int4 #5
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+4+2], v223, v224         // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x18, 0x4                    // w4a16: sign-extend int4 #6
v_bfe_i32 v224, v225, 0x1c, 0x4                    // w4a16: sign-extend int4 #7
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+4+3], v223, v224         // w4a16: pack 2 fp16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:2304 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 2304 sync LDS1
v_cvt_f32_f16 v226, v[vgprG2LScaleA+2]             // scaleA: fp16 -> f32
v_mov_b32 v225, v[vgprG2LA+10+0]                   // w4a16: save packed int4 dword 0
v_bfe_i32 v223, v225, 0x0, 0x4                     // w4a16: sign-extend int4 #0
v_bfe_i32 v224, v225, 0x4, 0x4                     // w4a16: sign-extend int4 #1
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+8+0], v223, v224         // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x8, 0x4                     // w4a16: sign-extend int4 #2
v_bfe_i32 v224, v225, 0xc, 0x4                     // w4a16: sign-extend int4 #3
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+8+1], v223, v224         // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x10, 0x4                    // w4a16: sign-extend int4 #4
v_bfe_i32 v224, v225, 0x14, 0x4                    // w4a16: sign-extend int4 #5
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+8+2], v223, v224         // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x18, 0x4                    // w4a16: sign-extend int4 #6
v_bfe_i32 v224, v225, 0x1c, 0x4                    // w4a16: sign-extend int4 #7
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+8+3], v223, v224         // w4a16: pack 2 fp16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:4608 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 4608 sync LDS1
v_cvt_f32_f16 v226, v[vgprG2LScaleA+3]             // scaleA: fp16 -> f32
v_mov_b32 v225, v[vgprG2LA+14+0]                   // w4a16: save packed int4 dword 0
v_bfe_i32 v223, v225, 0x0, 0x4                     // w4a16: sign-extend int4 #0
v_bfe_i32 v224, v225, 0x4, 0x4                     // w4a16: sign-extend int4 #1
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+12+0], v223, v224        // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x8, 0x4                     // w4a16: sign-extend int4 #2
v_bfe_i32 v224, v225, 0xc, 0x4                     // w4a16: sign-extend int4 #3
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+12+1], v223, v224        // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x10, 0x4                    // w4a16: sign-extend int4 #4
v_bfe_i32 v224, v225, 0x14, 0x4                    // w4a16: sign-extend int4 #5
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+12+2], v223, v224        // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x18, 0x4                    // w4a16: sign-extend int4 #6
v_bfe_i32 v224, v225, 0x1c, 0x4                    // w4a16: sign-extend int4 #7
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+12+3], v223, v224        // w4a16: pack 2 fp16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+12:vgprG2LA+12+3] offset:6912 // lwoA_0_0_3_0 = (0*LSCA)*(MT0I+PAD) + (3*LSPA) = 6912 sync LDS1

/* local write MXSA */

/* local write MXSB */

/* local write B */
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS1
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4:vgprG2LB+4+3] offset:2304 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 2304 sync LDS1
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8:vgprG2LB+8+3] offset:4608 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 4608 sync LDS1
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12:vgprG2LB+12+3] offset:6912 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 6912 sync LDS1
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+16:vgprG2LB+16+3] offset:9216 // lwoB_0_0_4_0 = (0*LSCB)*(MT1J+PAD) + (4*LSPB) = 9216 sync LDS1
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+20:vgprG2LB+20+3] offset:11520 // lwoB_0_0_5_0 = (0*LSCB)*(MT1J+PAD) + (5*LSPB) = 11520 sync LDS1
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+24:vgprG2LB+24+3] offset:13824 // lwoB_0_0_6_0 = (0*LSCB)*(MT1J+PAD) + (6*LSPB) = 13824 sync LDS1
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+28:vgprG2LB+28+3] offset:16128 // lwoB_0_0_7_0 = (0*LSCB)*(MT1J+PAD) + (7*LSPB) = 16128 sync LDS1
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+32:vgprG2LB+32+3] offset:18432 // lwoB_0_0_8_0 = (0*LSCB)*(MT1J+PAD) + (8*LSPB) = 18432 sync LDS1
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+36:vgprG2LB+36+3] offset:20736 // lwoB_0_0_9_0 = (0*LSCB)*(MT1J+PAD) + (9*LSPB) = 20736 sync LDS1

/* local write swap offsets a */
v_xor_b32 v[vgprLocalWriteAddrA], 0x8000, v[vgprLocalWriteAddrA] // swap Red Blk

/* local write swap offsets b */
v_xor_b32 v[vgprLocalWriteAddrB], 0x8000, v[vgprLocalWriteAddrB] // swap Red Blk

/* local read swap offsets a */
v_xor_b32 v[vgprLocalReadAddrA], 0x8000, v[vgprLocalReadAddrA] // swap Red Blk

/* local read swap offsets b */
v_xor_b32 v[vgprLocalReadAddrB], 0x8000, v[vgprLocalReadAddrB] // swap Red Blk

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */

/* Global Read IncA */

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s76, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s77, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s76        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s77       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s76 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s77 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* global read inc block-scale A (2 bytes every 2 iters) */
s_add_u32 s[sgprScaleAKCnt], s[sgprScaleAKCnt], 1  // scaleA: one more K iteration done
s_and_b32 s8, s[sgprScaleAKCnt], 0x1               // scaleA: SCC = counter has not wrapped
s_cselect_b32 s8, 0, 0x2                           // scaleA: advance only on a wrap
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], s8 // scaleA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], s8 // scaleA limit -= inc

/* Global Read IncB */

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s76, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s77, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s76        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s77       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s76 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s77 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32

/* Global Read A */
buffer_load_b32 v[vgprG2LA+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b32 v[vgprG2LA+6], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b32 v[vgprG2LA+10], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b32 v[vgprG2LA+14], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_3_0

/* global read block-scale A */
buffer_load_d16_b16 v[vgprG2LScaleA+0], v[vgprGlobalReadOffsetScaleA+0], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 0
buffer_load_d16_b16 v[vgprG2LScaleA+1], v[vgprGlobalReadOffsetScaleA+1], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 1
buffer_load_d16_b16 v[vgprG2LScaleA+2], v[vgprGlobalReadOffsetScaleA+2], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 2
buffer_load_d16_b16 v[vgprG2LScaleA+3], v[vgprGlobalReadOffsetScaleA+3], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 3

/* Global Read MXSA */

/* Global Read MXSB */

/* Global Read B */
buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b128 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b128 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b128 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0
buffer_load_b128 v[vgprG2LB+16:vgprG2LB+16+3], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_4_0
buffer_load_b128 v[vgprG2LB+20:vgprG2LB+20+3], v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_5_0
buffer_load_b128 v[vgprG2LB+24:vgprG2LB+24+3], v[vgprGlobalReadOffsetB+6], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_6_0
buffer_load_b128 v[vgprG2LB+28:vgprG2LB+28+3], v[vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_7_0
buffer_load_b128 v[vgprG2LB+32:vgprG2LB+32+3], v[vgprGlobalReadOffsetB+8], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_8_0
buffer_load_b128 v[vgprG2LB+36:vgprG2LB+36+3], v[vgprGlobalReadOffsetB+9], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_9_0
s_waitcnt lgkmcnt(14)                              // wait for prior local read local write old=0, new=14 newLW=14 newLR=0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
/* numPrefetchIter=0 */
/* dataAtIterA=3 numReadsIterA=4 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=3 numReadsIterB=4 skipReadsIterB=0 readsPerIterB=10 */

/******************************************/
/* Unrolled Loop - End                    */
/******************************************/

/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x2              // counterL==2
s_cbranch_scc0 label_LoopBeginL                    // restart LoopL
label_LoopEndL:

/* Before NLL: Check VGPR.checkin for INT8 LW */

/******************************************/
/* Ord. NoGlobalLoadLoop_1 - Begin        */
/******************************************/
s_waitcnt lgkmcnt(0)                               // 4wait for local write
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier                                          // wait for local write done, sync

/* iter 0 */

/* local read a */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4608 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4624 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1

/* local read b */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4608 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4624 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:9216 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:9232 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:13824 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:13840 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprLocalReadAddrB+0] offset:18432 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[vgprLocalReadAddrB+0] offset:18448 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1

/* local read increment a */
/* N/A, lro->16 */
/* localReadDoCntA 5 localReadDoCntMXSA 0 localReadDoCntB 5 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->16 */
/* localReadDoCntA 5 localReadDoCntMXSA 0 localReadDoCntB 5 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* Global Read IncA */

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s76, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s77, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s76        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s77       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s76 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s77 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* global read inc block-scale A (2 bytes every 2 iters) */
s_add_u32 s[sgprScaleAKCnt], s[sgprScaleAKCnt], 1  // scaleA: one more K iteration done
s_and_b32 s8, s[sgprScaleAKCnt], 0x1               // scaleA: SCC = counter has not wrapped
s_cselect_b32 s8, 0, 0x2                           // scaleA: advance only on a wrap
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], s8 // scaleA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], s8 // scaleA limit -= inc

/* Global Read IncB */

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s76, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s77, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s76        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s77       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s76 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s77 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0 for iteration == 0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=10 */

/* iter 1 */

/* local read a */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4640 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4656 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1

/* local read b */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4640 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4656 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:9248 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:9264 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:13856 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:13872 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprLocalReadAddrB+0] offset:18464 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[vgprLocalReadAddrB+0] offset:18480 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1

/* local read increment a */
/* N/A, lro->32 */
/* localReadDoCntA 6 localReadDoCntMXSA 0 localReadDoCntB 6 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->32 */
/* localReadDoCntA 6 localReadDoCntMXSA 0 localReadDoCntB 6 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=0 readsPerIterB=10 */

/* iter 2 */

/* local read a */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4672 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4688 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1

/* local read b */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4672 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4688 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:9280 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:9296 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:13888 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:13904 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprLocalReadAddrB+0] offset:18496 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[vgprLocalReadAddrB+0] offset:18512 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1

/* local read increment a */
/* N/A, lro->48 */
/* localReadDoCntA 7 localReadDoCntMXSA 0 localReadDoCntB 7 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->48 */
/* localReadDoCntA 7 localReadDoCntMXSA 0 localReadDoCntB 7 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=0 readsPerIterB=10 */

/* iter 3 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */

/* local read a */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4704 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4720 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1

/* local read b */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4704 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4720 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:9312 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:9328 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:13920 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:13936 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprLocalReadAddrB+0] offset:18528 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[vgprLocalReadAddrB+0] offset:18544 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
s_waitcnt vmcnt(0)                                 // 1wait for global read

/* local write A */
v_cvt_f32_f16 v226, v[vgprG2LScaleA+0]             // scaleA: fp16 -> f32
v_mov_b32 v225, v[vgprG2LA+2+0]                    // w4a16: save packed int4 dword 0
v_bfe_i32 v223, v225, 0x0, 0x4                     // w4a16: sign-extend int4 #0
v_bfe_i32 v224, v225, 0x4, 0x4                     // w4a16: sign-extend int4 #1
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+0+0], v223, v224         // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x8, 0x4                     // w4a16: sign-extend int4 #2
v_bfe_i32 v224, v225, 0xc, 0x4                     // w4a16: sign-extend int4 #3
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+0+1], v223, v224         // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x10, 0x4                    // w4a16: sign-extend int4 #4
v_bfe_i32 v224, v225, 0x14, 0x4                    // w4a16: sign-extend int4 #5
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+0+2], v223, v224         // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x18, 0x4                    // w4a16: sign-extend int4 #6
v_bfe_i32 v224, v225, 0x1c, 0x4                    // w4a16: sign-extend int4 #7
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+0+3], v223, v224         // w4a16: pack 2 fp16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
v_cvt_f32_f16 v226, v[vgprG2LScaleA+1]             // scaleA: fp16 -> f32
v_mov_b32 v225, v[vgprG2LA+6+0]                    // w4a16: save packed int4 dword 0
v_bfe_i32 v223, v225, 0x0, 0x4                     // w4a16: sign-extend int4 #0
v_bfe_i32 v224, v225, 0x4, 0x4                     // w4a16: sign-extend int4 #1
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+4+0], v223, v224         // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x8, 0x4                     // w4a16: sign-extend int4 #2
v_bfe_i32 v224, v225, 0xc, 0x4                     // w4a16: sign-extend int4 #3
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+4+1], v223, v224         // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x10, 0x4                    // w4a16: sign-extend int4 #4
v_bfe_i32 v224, v225, 0x14, 0x4                    // w4a16: sign-extend int4 #5
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+4+2], v223, v224         // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x18, 0x4                    // w4a16: sign-extend int4 #6
v_bfe_i32 v224, v225, 0x1c, 0x4                    // w4a16: sign-extend int4 #7
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+4+3], v223, v224         // w4a16: pack 2 fp16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:2304 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 2304 sync LDS0
v_cvt_f32_f16 v226, v[vgprG2LScaleA+2]             // scaleA: fp16 -> f32
v_mov_b32 v225, v[vgprG2LA+10+0]                   // w4a16: save packed int4 dword 0
v_bfe_i32 v223, v225, 0x0, 0x4                     // w4a16: sign-extend int4 #0
v_bfe_i32 v224, v225, 0x4, 0x4                     // w4a16: sign-extend int4 #1
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+8+0], v223, v224         // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x8, 0x4                     // w4a16: sign-extend int4 #2
v_bfe_i32 v224, v225, 0xc, 0x4                     // w4a16: sign-extend int4 #3
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+8+1], v223, v224         // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x10, 0x4                    // w4a16: sign-extend int4 #4
v_bfe_i32 v224, v225, 0x14, 0x4                    // w4a16: sign-extend int4 #5
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+8+2], v223, v224         // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x18, 0x4                    // w4a16: sign-extend int4 #6
v_bfe_i32 v224, v225, 0x1c, 0x4                    // w4a16: sign-extend int4 #7
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+8+3], v223, v224         // w4a16: pack 2 fp16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:4608 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 4608 sync LDS0
v_cvt_f32_f16 v226, v[vgprG2LScaleA+3]             // scaleA: fp16 -> f32
v_mov_b32 v225, v[vgprG2LA+14+0]                   // w4a16: save packed int4 dword 0
v_bfe_i32 v223, v225, 0x0, 0x4                     // w4a16: sign-extend int4 #0
v_bfe_i32 v224, v225, 0x4, 0x4                     // w4a16: sign-extend int4 #1
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+12+0], v223, v224        // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x8, 0x4                     // w4a16: sign-extend int4 #2
v_bfe_i32 v224, v225, 0xc, 0x4                     // w4a16: sign-extend int4 #3
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+12+1], v223, v224        // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x10, 0x4                    // w4a16: sign-extend int4 #4
v_bfe_i32 v224, v225, 0x14, 0x4                    // w4a16: sign-extend int4 #5
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+12+2], v223, v224        // w4a16: pack 2 fp16
v_bfe_i32 v223, v225, 0x18, 0x4                    // w4a16: sign-extend int4 #6
v_bfe_i32 v224, v225, 0x1c, 0x4                    // w4a16: sign-extend int4 #7
v_cvt_f32_i32 v223, v223                           // w4a16: int4 -> f32
v_cvt_f32_i32 v224, v224                           // w4a16: int4 -> f32
v_mul_f32 v223, v226, v223                         // w4a16: dequantize
v_mul_f32 v224, v226, v224                         // w4a16: dequantize
v_cvt_f16_f32 v223, v223                           // w4a16: f32 -> fp16
v_cvt_f16_f32 v224, v224                           // w4a16: f32 -> fp16
v_pack_b32_f16 v[vgprG2LA+12+3], v223, v224        // w4a16: pack 2 fp16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+12:vgprG2LA+12+3] offset:6912 // lwoA_0_0_3_0 = (0*LSCA)*(MT0I+PAD) + (3*LSPA) = 6912 sync LDS0

/* local write MXSA */

/* local write MXSB */

/* local write B */
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4:vgprG2LB+4+3] offset:2304 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 2304 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8:vgprG2LB+8+3] offset:4608 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 4608 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12:vgprG2LB+12+3] offset:6912 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 6912 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+16:vgprG2LB+16+3] offset:9216 // lwoB_0_0_4_0 = (0*LSCB)*(MT1J+PAD) + (4*LSPB) = 9216 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+20:vgprG2LB+20+3] offset:11520 // lwoB_0_0_5_0 = (0*LSCB)*(MT1J+PAD) + (5*LSPB) = 11520 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+24:vgprG2LB+24+3] offset:13824 // lwoB_0_0_6_0 = (0*LSCB)*(MT1J+PAD) + (6*LSPB) = 13824 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+28:vgprG2LB+28+3] offset:16128 // lwoB_0_0_7_0 = (0*LSCB)*(MT1J+PAD) + (7*LSPB) = 16128 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+32:vgprG2LB+32+3] offset:18432 // lwoB_0_0_8_0 = (0*LSCB)*(MT1J+PAD) + (8*LSPB) = 18432 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+36:vgprG2LB+36+3] offset:20736 // lwoB_0_0_9_0 = (0*LSCB)*(MT1J+PAD) + (9*LSPB) = 20736 sync LDS0

/* local write swap offsets a */
v_xor_b32 v[vgprLocalWriteAddrA], 0x8000, v[vgprLocalWriteAddrA] // swap Red Blk

/* local write swap offsets b */
v_xor_b32 v[vgprLocalWriteAddrB], 0x8000, v[vgprLocalWriteAddrB] // swap Red Blk

/* local read swap offsets a */
v_xor_b32 v[vgprLocalReadAddrA], 0x8000, v[vgprLocalReadAddrA] // swap Red Blk

/* local read swap offsets b */
v_xor_b32 v[vgprLocalReadAddrB], 0x8000, v[vgprLocalReadAddrB] // swap Red Blk

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
s_waitcnt lgkmcnt(14)                              // wait for prior local read local write old=0, new=14 newLW=14 newLR=0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
/* numPrefetchIter=0 */
/* dataAtIterA=3 numReadsIterA=4 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=3 numReadsIterB=4 skipReadsIterB=0 readsPerIterB=10 */
label_toPGR1:
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
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4608 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4624 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read b */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4608 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4624 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:9216 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:9232 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:13824 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:13840 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprLocalReadAddrB+0] offset:18432 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[vgprLocalReadAddrB+0] offset:18448 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read increment a */
/* N/A, lro->16 */
/* localReadDoCntA 9 localReadDoCntMXSA 0 localReadDoCntB 9 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->16 */
/* localReadDoCntA 9 localReadDoCntMXSA 0 localReadDoCntB 9 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0 for iteration == 0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=10 */

/* iter 1 (last unrolled loop) */

/* local read a */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4640 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4656 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read b */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4640 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4656 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:9248 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:9264 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:13856 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:13872 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprLocalReadAddrB+0] offset:18464 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[vgprLocalReadAddrB+0] offset:18480 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read increment a */
/* N/A, lro->32 */
/* localReadDoCntA 10 localReadDoCntMXSA 0 localReadDoCntB 10 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->32 */
/* localReadDoCntA 10 localReadDoCntMXSA 0 localReadDoCntB 10 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=0 readsPerIterB=10 */

/* iter 2 (last unrolled loop) */

/* local read a */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4672 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4688 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read b */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4672 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4688 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:9280 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:9296 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:13888 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:13904 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprLocalReadAddrB+0] offset:18496 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[vgprLocalReadAddrB+0] offset:18512 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read increment a */
/* N/A, lro->48 */
/* localReadDoCntA 11 localReadDoCntMXSA 0 localReadDoCntB 11 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->48 */
/* localReadDoCntA 11 localReadDoCntMXSA 0 localReadDoCntB 11 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=0 readsPerIterB=10 */

/* iter 3 (last unrolled loop) */

/* local read a */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4704 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4720 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read b */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4704 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4720 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:9312 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:9328 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:13920 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:13936 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprLocalReadAddrB+0] offset:18528 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[vgprLocalReadAddrB+0] offset:18544 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt vmcnt(0)                                 // 1wait for global read

/* local write A */

/* local write MXSA */

/* local write MXSB */

/* local write B */

/* Global Read IncA */

/* Global Read IncB */

/* Global Read A */

/* Global Read MXSA */

/* Global Read MXSB */

/* Global Read B */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
v_wmma_f32_16x16x16_f16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
/* numPrefetchIter=0 */
/* dataAtIterA=3 numReadsIterA=4 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=3 numReadsIterB=4 skipReadsIterB=0 readsPerIterB=10 */
label_toPGR1end_OrdNLL:
label_PrefetchGlobalLastIterEnd:

/* Tail: add ValuA/B vgpr buffer [108...165) to pool */

/* Tail: add address/G2L vgpr [165...222) to pool */
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
.set sgprScaleAKCnt, UNDEF
.set sgprStaggerUIter, UNDEF
.set sgprShadowLimitA, UNDEF
.set sgprSrdA, UNDEF
.set sgprSrdB, UNDEF
.set sgprShadowLimitB, UNDEF
.set sgprWrapUA, UNDEF
.set sgprWrapUB, UNDEF
.set sgprGlobalReadIncsA, UNDEF
.set sgprGlobalReadIncsB, UNDEF
/* load store sgprs */

/* Mapping of Acc register -> C Vgpr register */

/* Multiply MI out register with Alpha -> C Vgpr register */

/* not-LocalSplitU: global write indices */
/* computeStoreVgprs */
v_lshrrev_b32 v112, 5, v[vgprSerial]               // 112 = Serial / 32
v_lshrrev_b32 v113, 1, v112                        // 113 = 112 / 2
v_mul_lo_u32 v113, 0x10, v113                      // wave coordination offset 1
v_and_b32 v109, 31, v[vgprSerial]                  // v109 = v[vgprSerial] % 32
v_lshrrev_b32 v109, 4, v109                        // 109 = 109 / 16
                                                   // thread0 * continuous_output (multiplier is 1, do nothing)
v_add_lshl_u32 v109, v113, v109, 0                 // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v110, v109, s[sgprStrideC1J]          //  offset 1
v_mul_lo_u32 v111, v109, s[sgprStrideD1J]          //  offset 1
v_and_b32 v108, 1, v112                            // v108 = v112 % 2
v_mul_lo_u32 v108, 0x10, v108                      // wave coordination offset 0
v_and_b32 v113, 15, v[vgprSerial]                  // v113 = v[vgprSerial] % 16
v_add_lshl_u32 v108, v113, v108, 0                 // coordination 0 = vwA * (wave_id0 + tid0)
s_mul_i32 s8, 64, s[sgprWorkGroup0]                // wgp0 * MT0
v_add_nc_u32 v108, s8, v108                        // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s8, 160, s[sgprWorkGroup1]               // wgp1 * MT1
v_add_nc_u32 v109, s8, v109                        // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1

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

/* Edge/NonEdge store path check (N (isSize1)): Size % 160 > 0 -> Edge store; else -> NonEdge store */
s_mov_b32 s31, 0                                   // STATIC_DIV: divisor=160
s_mul_i32 s30, 819, s[sgprSizeJ]                   // tmp1 = dividend * magic hi
s_lshl_b64 s[30:31], s[30:31], 16                  // left shift 16 bits
s_mul_i32 s29, s[sgprSizeJ], 13108                 // tmp0 = dividend * magic lo
s_add_u32 s30, s29, s30                            // add lo
s_addc_u32 s31, s31, 0                             // add hi
s_lshr_b64 s[30:31], s[30:31], 33                  // tmp1 = (dividend * magic) << shift
s_mov_b32 s29, s30                                 // quotient
s_mul_i32 s30, s29, 160                            // quotient*divisor
s_sub_u32 s28, s[sgprSizeJ], s30                   // rReg = dividend - quotient*divisor
s_add_u32 s29, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s29                // wg1 >= nwg1-1
s_cselect_b32 s28, s28, 0                          // set rem
s_cmpk_gt_u32 s28, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_MB_Then         // jump if edges required
label_GW_B0_FD0_VW1_MB_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=134 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1); (24,0,0,0:vw1); (24,1,0,0:vw1); (25,0,0,0:vw1); (25,1,0,0:vw1); (26,0,0,0:vw1); (26,1,0,0:vw1); (27,0,0,0:vw1); (27,1,0,0:vw1); (28,0,0,0:vw1); (28,1,0,0:vw1); (29,0,0,0:vw1); (29,1,0,0:vw1); (30,0,0,0:vw1); (30,1,0,0:vw1); (31,0,0,0:vw1); (31,1,0,0:vw1); (32,0,0,0:vw1); (32,1,0,0:vw1); (33,0,0,0:vw1); (33,1,0,0:vw1); (34,0,0,0:vw1); (34,1,0,0:vw1); (35,0,0,0:vw1); (35,1,0,0:vw1); (36,0,0,0:vw1); (36,1,0,0:vw1); (37,0,0,0:vw1); (37,1,0,0:vw1); (38,0,0,0:vw1); (38,1,0,0:vw1); (39,0,0,0:vw1); (39,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
/* (d1,vc1,d0,vc0)=(9,0,1,0) */
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
/* (d1,vc1,d0,vc0)=(10,0,1,0) */
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
/* (d1,vc1,d0,vc0)=(11,0,1,0) */
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
/* (d1,vc1,d0,vc0)=(13,0,1,0) */
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
/* (d1,vc1,d0,vc0)=(14,0,1,0) */
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
/* (d1,vc1,d0,vc0)=(15,0,1,0) */
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
/* (d1,vc1,d0,vc0)=(17,0,1,0) */
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
/* (d1,vc1,d0,vc0)=(18,0,1,0) */
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
/* (d1,vc1,d0,vc0)=(19,0,1,0) */
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
/* (d1,vc1,d0,vc0)=(21,0,1,0) */
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
/* (d1,vc1,d0,vc0)=(22,0,1,0) */
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
/* (d1,vc1,d0,vc0)=(23,0,1,0) */
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
/* (d1,vc1,d0,vc0)=(24,0,1,0) */
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
/* (d1,vc1,d0,vc0)=(25,0,1,0) */
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
/* (d1,vc1,d0,vc0)=(26,0,1,0) */
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
/* (d1,vc1,d0,vc0)=(27,0,1,0) */
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
/* (d1,vc1,d0,vc0)=(28,0,1,0) */
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
/* (d1,vc1,d0,vc0)=(29,0,1,0) */
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
/* (d1,vc1,d0,vc0)=(30,0,1,0) */
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
/* (d1,vc1,d0,vc0)=(31,0,1,0) */
/* (d1,vc1,d0,vc0)=(32,0,0,0) */
/* (d1,vc1,d0,vc0)=(32,0,1,0) */
/* (d1,vc1,d0,vc0)=(33,0,0,0) */
/* (d1,vc1,d0,vc0)=(33,0,1,0) */
/* (d1,vc1,d0,vc0)=(34,0,0,0) */
/* (d1,vc1,d0,vc0)=(34,0,1,0) */
/* (d1,vc1,d0,vc0)=(35,0,0,0) */
/* (d1,vc1,d0,vc0)=(35,0,1,0) */
/* (d1,vc1,d0,vc0)=(36,0,0,0) */
/* (d1,vc1,d0,vc0)=(36,0,1,0) */
/* (d1,vc1,d0,vc0)=(37,0,0,0) */
/* (d1,vc1,d0,vc0)=(37,0,1,0) */
/* (d1,vc1,d0,vc0)=(38,0,0,0) */
/* (d1,vc1,d0,vc0)=(38,0,1,0) */
/* (d1,vc1,d0,vc0)=(39,0,0,0) */
/* (d1,vc1,d0,vc0)=(39,0,1,0) */
v_add_lshl_u32 v119, v111, v108, 2                 // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=108, coord0Vgpr=108 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (19, 0, 0, 0), (19, 1, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0), (22, 0, 0, 0), (22, 1, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0), (24, 0, 0, 0), (24, 1, 0, 0), (25, 0, 0, 0), (25, 1, 0, 0), (26, 0, 0, 0), (26, 1, 0, 0), (27, 0, 0, 0), (27, 1, 0, 0), (28, 0, 0, 0), (28, 1, 0, 0), (29, 0, 0, 0), (29, 1, 0, 0), (30, 0, 0, 0), (30, 1, 0, 0), (31, 0, 0, 0), (31, 1, 0, 0), (32, 0, 0, 0), (32, 1, 0, 0), (33, 0, 0, 0), (33, 1, 0, 0), (34, 0, 0, 0), (34, 1, 0, 0), (35, 0, 0, 0), (35, 1, 0, 0), (36, 0, 0, 0), (36, 1, 0, 0), (37, 0, 0, 0), (37, 1, 0, 0), (38, 0, 0, 0), (38, 1, 0, 0), (39, 0, 0, 0), (39, 1, 0, 0)] */
v_mov_b32 v[vgprValuC+121], v[vgprValuC+0]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+122], v[vgprValuC+8]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+123], v[vgprValuC+1]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+124], v[vgprValuC+9]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+125], v[vgprValuC+2]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+126], v[vgprValuC+10]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+127], v[vgprValuC+3]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+128], v[vgprValuC+11]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+129], v[vgprValuC+4]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+130], v[vgprValuC+12]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+131], v[vgprValuC+5]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+132], v[vgprValuC+13]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+133], v[vgprValuC+6]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+134], v[vgprValuC+14]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+135], v[vgprValuC+7]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+136], v[vgprValuC+15]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+137], v[vgprValuC+16]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+138], v[vgprValuC+24]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+139], v[vgprValuC+17]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+140], v[vgprValuC+25]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+141], v[vgprValuC+18]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+142], v[vgprValuC+26]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+143], v[vgprValuC+19]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+144], v[vgprValuC+27]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+145], v[vgprValuC+20]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+146], v[vgprValuC+28]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+147], v[vgprValuC+21]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+148], v[vgprValuC+29]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+149], v[vgprValuC+22]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+150], v[vgprValuC+30]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+151], v[vgprValuC+23]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+152], v[vgprValuC+31]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+153], v[vgprValuC+32]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+154], v[vgprValuC+40]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+155], v[vgprValuC+33]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+156], v[vgprValuC+41]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+157], v[vgprValuC+34]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+158], v[vgprValuC+42]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+159], v[vgprValuC+35]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+160], v[vgprValuC+43]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+161], v[vgprValuC+36]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+162], v[vgprValuC+44]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+163], v[vgprValuC+37]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+164], v[vgprValuC+45]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+165], v[vgprValuC+38]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+166], v[vgprValuC+46]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+167], v[vgprValuC+39]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+168], v[vgprValuC+47]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+169], v[vgprValuC+48]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+170], v[vgprValuC+56]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+171], v[vgprValuC+49]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+172], v[vgprValuC+57]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+173], v[vgprValuC+50]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+174], v[vgprValuC+58]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+175], v[vgprValuC+51]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+176], v[vgprValuC+59]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+177], v[vgprValuC+52]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+178], v[vgprValuC+60]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+179], v[vgprValuC+53]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+180], v[vgprValuC+61]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+181], v[vgprValuC+54]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+182], v[vgprValuC+62]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+183], v[vgprValuC+55]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+184], v[vgprValuC+63]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+185], v[vgprValuC+64]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+186], v[vgprValuC+72]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+187], v[vgprValuC+65]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+188], v[vgprValuC+73]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+189], v[vgprValuC+66]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+190], v[vgprValuC+74]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+191], v[vgprValuC+67]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+192], v[vgprValuC+75]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+193], v[vgprValuC+68]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+194], v[vgprValuC+76]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+195], v[vgprValuC+69]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+196], v[vgprValuC+77]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+197], v[vgprValuC+70]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+198], v[vgprValuC+78]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+199], v[vgprValuC+71]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+200], v[vgprValuC+79]        // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b32 v121, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v122, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v123, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v124, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v125, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v126, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v127, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v128, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v129, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v130, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v131, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v132, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v133, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v134, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v135, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v136, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 72                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b32 v137, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v138, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v139, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v140, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v141, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v142, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v143, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v144, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v145, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v146, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v147, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v148, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v149, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v150, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v151, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v152, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 72                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b32 v153, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v154, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v155, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v156, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v157, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v158, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v159, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v160, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v161, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v162, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v163, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v164, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v165, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v166, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v167, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v168, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 72                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b32 v169, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v170, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v171, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v172, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v173, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v174, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v175, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v176, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v177, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v178, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v179, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v180, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v181, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v182, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v183, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v184, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 72                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b32 v185, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v186, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v187, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v188, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v189, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v190, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v191, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v192, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v193, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v194, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v195, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v196, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v197, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v198, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v199, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v200, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_B0_FD0_VW1_MB_NonEdgeEnd:
label_GW_B0_FD0_VW1_MB_Else:
label_GW_B0_FD0_VW1_MB_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=66 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1); (24,0,0,0:vw1); (24,1,0,0:vw1); (25,0,0,0:vw1); (25,1,0,0:vw1); (26,0,0,0:vw1); (26,1,0,0:vw1); (27,0,0,0:vw1); (27,1,0,0:vw1); (28,0,0,0:vw1); (28,1,0,0:vw1); (29,0,0,0:vw1); (29,1,0,0:vw1); (30,0,0,0:vw1); (30,1,0,0:vw1); (31,0,0,0:vw1); (31,1,0,0:vw1); (32,0,0,0:vw1); (32,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v114, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v185, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v185, v114, v185, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v186, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v186, v114, v186, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v187, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v187, v114, v187, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v188, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v188, v114, v188, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v189, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v189, v114, v189, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v190, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v190, v114, v190, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v191, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v191, v114, v191, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v192, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v192, v114, v192, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v193, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v193, v114, v193, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v194, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v194, v114, v194, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v195, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v195, v114, v195, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v196, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v196, v114, v196, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v197, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v197, v114, v197, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v198, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v198, v114, v198, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v199, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v199, v114, v199, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v200, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v200, v114, v200, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 18                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v201, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v201, v114, v201, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v202, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v202, v114, v202, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v203, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v203, v114, v203, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v204, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v204, v114, v204, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v205, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v205, v114, v205, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v206, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v206, v114, v206, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v207, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v207, v114, v207, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v208, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v208, v114, v208, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v209, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v209, v114, v209, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v210, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v210, v114, v210, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v211, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v211, v114, v211, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v212, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v212, v114, v212, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v213, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v213, v114, v213, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v214, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v214, v114, v214, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v215, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v215, v114, v215, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v216, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v216, v114, v216, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 18                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v217, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v217, v114, v217, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v218, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v218, v114, v218, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v219, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v219, v114, v219, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v220, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v220, v114, v220, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v221, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v221, v114, v221, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v223, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v223, v114, v223, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v224, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v224, v114, v224, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v225, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v225, v114, v225, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v226, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v226, v114, v226, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v227, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v227, v114, v227, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v228, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v228, v114, v228, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v229, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v229, v114, v229, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v230, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v230, v114, v230, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v231, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v231, v114, v231, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v232, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v232, v114, v232, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v233, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v233, v114, v233, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 18                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v234, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v234, v114, v234, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v235, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v235, v114, v235, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v236, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v236, v114, v236, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v237, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v237, v114, v237, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v238, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v238, v114, v238, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v239, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v239, v114, v239, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v240, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v240, v114, v240, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v241, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v241, v114, v241, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v242, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v242, v114, v242, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v243, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v243, v114, v243, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v244, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v244, v114, v244, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v245, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v245, v114, v245, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v246, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v246, v114, v246, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v247, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v247, v114, v247, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v248, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v248, v114, v248, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v249, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v249, v114, v249, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(32,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 18                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v250, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v250, v114, v250, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(32,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v251, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v251, v114, v251, s30                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (19, 0, 0, 0), (19, 1, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0), (22, 0, 0, 0), (22, 1, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0), (24, 0, 0, 0), (24, 1, 0, 0), (25, 0, 0, 0), (25, 1, 0, 0), (26, 0, 0, 0), (26, 1, 0, 0), (27, 0, 0, 0), (27, 1, 0, 0), (28, 0, 0, 0), (28, 1, 0, 0), (29, 0, 0, 0), (29, 1, 0, 0), (30, 0, 0, 0), (30, 1, 0, 0), (31, 0, 0, 0), (31, 1, 0, 0), (32, 0, 0, 0), (32, 1, 0, 0)] */
v_mov_b32 v[vgprValuC+119], v[vgprValuC+0]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+120], v[vgprValuC+8]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+121], v[vgprValuC+1]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+122], v[vgprValuC+9]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+123], v[vgprValuC+2]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+124], v[vgprValuC+10]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+125], v[vgprValuC+3]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+126], v[vgprValuC+11]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+127], v[vgprValuC+4]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+128], v[vgprValuC+12]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+129], v[vgprValuC+5]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+130], v[vgprValuC+13]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+131], v[vgprValuC+6]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+132], v[vgprValuC+14]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+133], v[vgprValuC+7]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+134], v[vgprValuC+15]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+135], v[vgprValuC+16]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+136], v[vgprValuC+24]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+137], v[vgprValuC+17]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+138], v[vgprValuC+25]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+139], v[vgprValuC+18]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+140], v[vgprValuC+26]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+141], v[vgprValuC+19]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+142], v[vgprValuC+27]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+143], v[vgprValuC+20]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+144], v[vgprValuC+28]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+145], v[vgprValuC+21]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+146], v[vgprValuC+29]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+147], v[vgprValuC+22]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+148], v[vgprValuC+30]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+149], v[vgprValuC+23]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+150], v[vgprValuC+31]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+151], v[vgprValuC+32]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+152], v[vgprValuC+40]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+153], v[vgprValuC+33]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+154], v[vgprValuC+41]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+155], v[vgprValuC+34]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+156], v[vgprValuC+42]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+157], v[vgprValuC+35]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+158], v[vgprValuC+43]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+159], v[vgprValuC+36]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+160], v[vgprValuC+44]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+161], v[vgprValuC+37]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+162], v[vgprValuC+45]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+163], v[vgprValuC+38]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+164], v[vgprValuC+46]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+165], v[vgprValuC+39]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+166], v[vgprValuC+47]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+167], v[vgprValuC+48]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+168], v[vgprValuC+56]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+169], v[vgprValuC+49]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+170], v[vgprValuC+57]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+171], v[vgprValuC+50]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+172], v[vgprValuC+58]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+173], v[vgprValuC+51]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+174], v[vgprValuC+59]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+175], v[vgprValuC+52]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+176], v[vgprValuC+60]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+177], v[vgprValuC+53]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+178], v[vgprValuC+61]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+179], v[vgprValuC+54]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+180], v[vgprValuC+62]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+181], v[vgprValuC+55]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+182], v[vgprValuC+63]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+183], v[vgprValuC+64]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+184], v[vgprValuC+72]        // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b32 v119, v185, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v120, v186, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v121, v187, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v122, v188, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v123, v189, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v124, v190, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v125, v191, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v126, v192, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v127, v193, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v128, v194, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v129, v195, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v130, v196, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v131, v197, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v132, v198, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v133, v199, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v134, v200, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v135, v201, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v136, v202, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v137, v203, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v138, v204, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v139, v205, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v140, v206, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v141, v207, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v142, v208, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v143, v209, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v144, v210, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v145, v211, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v146, v212, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v147, v213, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v148, v214, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v149, v215, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v150, v216, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v151, v217, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v152, v218, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v153, v219, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v154, v220, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v155, v221, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v156, v223, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v157, v224, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v158, v225, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v159, v226, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v160, v227, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v161, v228, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v162, v229, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v163, v230, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v164, v231, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v165, v232, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v166, v233, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v167, v234, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v168, v235, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v169, v236, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v170, v237, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v171, v238, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v172, v239, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v173, v240, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v174, v241, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v175, v242, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v176, v243, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v177, v244, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v178, v245, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v179, v246, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v180, v247, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v181, v248, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v182, v249, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v183, v250, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v184, v251, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (33,0,0,0:vw1); (33,1,0,0:vw1); (34,0,0,0:vw1); (34,1,0,0:vw1); (35,0,0,0:vw1); (35,1,0,0:vw1); (36,0,0,0:vw1); (36,1,0,0:vw1); (37,0,0,0:vw1); (37,1,0,0:vw1); (38,0,0,0:vw1); (38,1,0,0:vw1); (39,0,0,0:vw1); (39,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v114, BufferOOB
/* (d1,vc1,d0,vc0)=(33,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v133, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v133, v114, v133, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(33,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v134, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v114, v134, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(34,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v135, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v135, v114, v135, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(34,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v136, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v114, v136, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(35,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v137, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v137, v114, v137, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(35,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v138, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v114, v138, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(36,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v139, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v139, v114, v139, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(36,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v140, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v114, v140, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(37,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v141, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v141, v114, v141, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(37,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v142, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v114, v142, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(38,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v143, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v143, v114, v143, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(38,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v144, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v144, v114, v144, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(39,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v145, v111, v108, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v145, v114, v145, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(39,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v146, v111, v112, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v146, v114, v146, s30                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(33, 0, 0, 0), (33, 1, 0, 0), (34, 0, 0, 0), (34, 1, 0, 0), (35, 0, 0, 0), (35, 1, 0, 0), (36, 0, 0, 0), (36, 1, 0, 0), (37, 0, 0, 0), (37, 1, 0, 0), (38, 0, 0, 0), (38, 1, 0, 0), (39, 0, 0, 0), (39, 1, 0, 0)] */
v_mov_b32 v[vgprValuC+119], v[vgprValuC+65]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+120], v[vgprValuC+73]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+121], v[vgprValuC+66]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+122], v[vgprValuC+74]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+123], v[vgprValuC+67]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+124], v[vgprValuC+75]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+125], v[vgprValuC+68]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+126], v[vgprValuC+76]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+127], v[vgprValuC+69]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+128], v[vgprValuC+77]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+129], v[vgprValuC+70]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+130], v[vgprValuC+78]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+131], v[vgprValuC+71]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+132], v[vgprValuC+79]        // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b32 v119, v133, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v120, v134, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v121, v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v122, v136, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v123, v137, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v124, v138, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v125, v139, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v126, v140, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v127, v141, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v128, v142, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v129, v143, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v130, v144, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v131, v145, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v132, v146, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
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

/* Edge/NonEdge store path check (N (isSize1)): Size % 160 > 0 -> Edge store; else -> NonEdge store */
s_mov_b32 s31, 0                                   // STATIC_DIV: divisor=160
s_mul_i32 s30, 819, s[sgprSizeJ]                   // tmp1 = dividend * magic hi
s_lshl_b64 s[30:31], s[30:31], 16                  // left shift 16 bits
s_mul_i32 s29, s[sgprSizeJ], 13108                 // tmp0 = dividend * magic lo
s_add_u32 s30, s29, s30                            // add lo
s_addc_u32 s31, s31, 0                             // add hi
s_lshr_b64 s[30:31], s[30:31], 33                  // tmp1 = (dividend * magic) << shift
s_mov_b32 s29, s30                                 // quotient
s_mul_i32 s30, s29, 160                            // quotient*divisor
s_sub_u32 s28, s[sgprSizeJ], s30                   // rReg = dividend - quotient*divisor
s_add_u32 s29, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s29                // wg1 >= nwg1-1
s_cselect_b32 s28, s28, 0                          // set rem
s_cmpk_gt_u32 s28, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_GSU1_Then       // jump if edges required
label_GW_B0_FD0_VW1_GSU1_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=134 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1); (24,0,0,0:vw1); (24,1,0,0:vw1); (25,0,0,0:vw1); (25,1,0,0:vw1); (26,0,0,0:vw1); (26,1,0,0:vw1); (27,0,0,0:vw1); (27,1,0,0:vw1); (28,0,0,0:vw1); (28,1,0,0:vw1); (29,0,0,0:vw1); (29,1,0,0:vw1); (30,0,0,0:vw1); (30,1,0,0:vw1); (31,0,0,0:vw1); (31,1,0,0:vw1); (32,0,0,0:vw1); (32,1,0,0:vw1); (33,0,0,0:vw1); (33,1,0,0:vw1); (34,0,0,0:vw1); (34,1,0,0:vw1); (35,0,0,0:vw1); (35,1,0,0:vw1); (36,0,0,0:vw1); (36,1,0,0:vw1); (37,0,0,0:vw1); (37,1,0,0:vw1); (38,0,0,0:vw1); (38,1,0,0:vw1); (39,0,0,0:vw1); (39,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
/* (d1,vc1,d0,vc0)=(9,0,1,0) */
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
/* (d1,vc1,d0,vc0)=(10,0,1,0) */
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
/* (d1,vc1,d0,vc0)=(11,0,1,0) */
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
/* (d1,vc1,d0,vc0)=(13,0,1,0) */
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
/* (d1,vc1,d0,vc0)=(14,0,1,0) */
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
/* (d1,vc1,d0,vc0)=(15,0,1,0) */
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
/* (d1,vc1,d0,vc0)=(17,0,1,0) */
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
/* (d1,vc1,d0,vc0)=(18,0,1,0) */
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
/* (d1,vc1,d0,vc0)=(19,0,1,0) */
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
/* (d1,vc1,d0,vc0)=(21,0,1,0) */
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
/* (d1,vc1,d0,vc0)=(22,0,1,0) */
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
/* (d1,vc1,d0,vc0)=(23,0,1,0) */
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
/* (d1,vc1,d0,vc0)=(24,0,1,0) */
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
/* (d1,vc1,d0,vc0)=(25,0,1,0) */
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
/* (d1,vc1,d0,vc0)=(26,0,1,0) */
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
/* (d1,vc1,d0,vc0)=(27,0,1,0) */
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
/* (d1,vc1,d0,vc0)=(28,0,1,0) */
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
/* (d1,vc1,d0,vc0)=(29,0,1,0) */
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
/* (d1,vc1,d0,vc0)=(30,0,1,0) */
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
/* (d1,vc1,d0,vc0)=(31,0,1,0) */
/* (d1,vc1,d0,vc0)=(32,0,0,0) */
/* (d1,vc1,d0,vc0)=(32,0,1,0) */
/* (d1,vc1,d0,vc0)=(33,0,0,0) */
/* (d1,vc1,d0,vc0)=(33,0,1,0) */
/* (d1,vc1,d0,vc0)=(34,0,0,0) */
/* (d1,vc1,d0,vc0)=(34,0,1,0) */
/* (d1,vc1,d0,vc0)=(35,0,0,0) */
/* (d1,vc1,d0,vc0)=(35,0,1,0) */
/* (d1,vc1,d0,vc0)=(36,0,0,0) */
/* (d1,vc1,d0,vc0)=(36,0,1,0) */
/* (d1,vc1,d0,vc0)=(37,0,0,0) */
/* (d1,vc1,d0,vc0)=(37,0,1,0) */
/* (d1,vc1,d0,vc0)=(38,0,0,0) */
/* (d1,vc1,d0,vc0)=(38,0,1,0) */
/* (d1,vc1,d0,vc0)=(39,0,0,0) */
/* (d1,vc1,d0,vc0)=(39,0,1,0) */
v_add_lshl_u32 v119, v111, v108, 1                 // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=108, coord0Vgpr=108 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (19, 0, 0, 0), (19, 1, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0), (22, 0, 0, 0), (22, 1, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0), (24, 0, 0, 0), (24, 1, 0, 0), (25, 0, 0, 0), (25, 1, 0, 0), (26, 0, 0, 0), (26, 1, 0, 0), (27, 0, 0, 0), (27, 1, 0, 0), (28, 0, 0, 0), (28, 1, 0, 0), (29, 0, 0, 0), (29, 1, 0, 0), (30, 0, 0, 0), (30, 1, 0, 0), (31, 0, 0, 0), (31, 1, 0, 0), (32, 0, 0, 0), (32, 1, 0, 0), (33, 0, 0, 0), (33, 1, 0, 0), (34, 0, 0, 0), (34, 1, 0, 0), (35, 0, 0, 0), (35, 1, 0, 0), (36, 0, 0, 0), (36, 1, 0, 0), (37, 0, 0, 0), (37, 1, 0, 0), (38, 0, 0, 0), (38, 1, 0, 0), (39, 0, 0, 0), (39, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+128], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+129], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+130], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+131], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+132], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+133], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+134], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+135], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+152], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+154], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+158], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+160], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+162], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+163], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+164], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+166], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+167], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+168], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+169], s[sgprAlpha], v[vgprValuC+48] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+170], s[sgprAlpha], v[vgprValuC+56] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+171], s[sgprAlpha], v[vgprValuC+49] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+172], s[sgprAlpha], v[vgprValuC+57] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+173], s[sgprAlpha], v[vgprValuC+50] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+174], s[sgprAlpha], v[vgprValuC+58] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+175], s[sgprAlpha], v[vgprValuC+51] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+176], s[sgprAlpha], v[vgprValuC+59] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+177], s[sgprAlpha], v[vgprValuC+52] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+178], s[sgprAlpha], v[vgprValuC+60] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+179], s[sgprAlpha], v[vgprValuC+53] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+180], s[sgprAlpha], v[vgprValuC+61] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+181], s[sgprAlpha], v[vgprValuC+54] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+182], s[sgprAlpha], v[vgprValuC+62] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+183], s[sgprAlpha], v[vgprValuC+55] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+184], s[sgprAlpha], v[vgprValuC+63] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+185], s[sgprAlpha], v[vgprValuC+64] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+186], s[sgprAlpha], v[vgprValuC+72] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+187], s[sgprAlpha], v[vgprValuC+65] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+188], s[sgprAlpha], v[vgprValuC+73] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+189], s[sgprAlpha], v[vgprValuC+66] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+190], s[sgprAlpha], v[vgprValuC+74] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+191], s[sgprAlpha], v[vgprValuC+67] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+192], s[sgprAlpha], v[vgprValuC+75] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+193], s[sgprAlpha], v[vgprValuC+68] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+194], s[sgprAlpha], v[vgprValuC+76] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+195], s[sgprAlpha], v[vgprValuC+69] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+196], s[sgprAlpha], v[vgprValuC+77] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+197], s[sgprAlpha], v[vgprValuC+70] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+198], s[sgprAlpha], v[vgprValuC+78] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+199], s[sgprAlpha], v[vgprValuC+71] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+200], s[sgprAlpha], v[vgprValuC+79] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v121, v[vgprValuC+121]               // convert C to fp16
buffer_store_b16 v121, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v122, v[vgprValuC+122]               // convert C to fp16
buffer_store_b16 v122, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v123, v[vgprValuC+123]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v123, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v124, v[vgprValuC+124]               // convert C to fp16
buffer_store_b16 v124, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v125, v[vgprValuC+125]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v125, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v126, v[vgprValuC+126]               // convert C to fp16
buffer_store_b16 v126, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v127, v[vgprValuC+127]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v127, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v128, v[vgprValuC+128]               // convert C to fp16
buffer_store_b16 v128, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v129, v[vgprValuC+129]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v129, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v130, v[vgprValuC+130]               // convert C to fp16
buffer_store_b16 v130, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v131, v[vgprValuC+131]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v131, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v132, v[vgprValuC+132]               // convert C to fp16
buffer_store_b16 v132, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v133, v[vgprValuC+133]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v133, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v134, v[vgprValuC+134]               // convert C to fp16
buffer_store_b16 v134, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v135, v[vgprValuC+135]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v135, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v136, v[vgprValuC+136]               // convert C to fp16
buffer_store_b16 v136, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v137, v[vgprValuC+137]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 36                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b16 v137, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v138, v[vgprValuC+138]               // convert C to fp16
buffer_store_b16 v138, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v139, v[vgprValuC+139]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v139, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v140, v[vgprValuC+140]               // convert C to fp16
buffer_store_b16 v140, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v141, v[vgprValuC+141]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v141, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v142, v[vgprValuC+142]               // convert C to fp16
buffer_store_b16 v142, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v143, v[vgprValuC+143]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v143, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v144, v[vgprValuC+144]               // convert C to fp16
buffer_store_b16 v144, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v145, v[vgprValuC+145]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v145, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v146, v[vgprValuC+146]               // convert C to fp16
buffer_store_b16 v146, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v147, v[vgprValuC+147]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v147, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v148, v[vgprValuC+148]               // convert C to fp16
buffer_store_b16 v148, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v149, v[vgprValuC+149]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v149, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v150, v[vgprValuC+150]               // convert C to fp16
buffer_store_b16 v150, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v151, v[vgprValuC+151]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v151, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v152, v[vgprValuC+152]               // convert C to fp16
buffer_store_b16 v152, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v153, v[vgprValuC+153]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 36                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b16 v153, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v154, v[vgprValuC+154]               // convert C to fp16
buffer_store_b16 v154, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v155, v[vgprValuC+155]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v155, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v156, v[vgprValuC+156]               // convert C to fp16
buffer_store_b16 v156, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v157, v[vgprValuC+157]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v157, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v158, v[vgprValuC+158]               // convert C to fp16
buffer_store_b16 v158, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v159, v[vgprValuC+159]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v159, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v160, v[vgprValuC+160]               // convert C to fp16
buffer_store_b16 v160, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v161, v[vgprValuC+161]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v161, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v162, v[vgprValuC+162]               // convert C to fp16
buffer_store_b16 v162, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v163, v[vgprValuC+163]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v163, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v164, v[vgprValuC+164]               // convert C to fp16
buffer_store_b16 v164, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v165, v[vgprValuC+165]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v165, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v166, v[vgprValuC+166]               // convert C to fp16
buffer_store_b16 v166, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v167, v[vgprValuC+167]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v167, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v168, v[vgprValuC+168]               // convert C to fp16
buffer_store_b16 v168, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v169, v[vgprValuC+169]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 36                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b16 v169, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v170, v[vgprValuC+170]               // convert C to fp16
buffer_store_b16 v170, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v171, v[vgprValuC+171]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v171, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v172, v[vgprValuC+172]               // convert C to fp16
buffer_store_b16 v172, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v173, v[vgprValuC+173]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v173, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v174, v[vgprValuC+174]               // convert C to fp16
buffer_store_b16 v174, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v175, v[vgprValuC+175]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v175, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v176, v[vgprValuC+176]               // convert C to fp16
buffer_store_b16 v176, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v177, v[vgprValuC+177]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v177, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v178, v[vgprValuC+178]               // convert C to fp16
buffer_store_b16 v178, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v179, v[vgprValuC+179]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v179, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v180, v[vgprValuC+180]               // convert C to fp16
buffer_store_b16 v180, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v181, v[vgprValuC+181]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v181, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v182, v[vgprValuC+182]               // convert C to fp16
buffer_store_b16 v182, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v183, v[vgprValuC+183]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v183, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v184, v[vgprValuC+184]               // convert C to fp16
buffer_store_b16 v184, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v185, v[vgprValuC+185]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 36                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b16 v185, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v186, v[vgprValuC+186]               // convert C to fp16
buffer_store_b16 v186, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v187, v[vgprValuC+187]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v187, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v188, v[vgprValuC+188]               // convert C to fp16
buffer_store_b16 v188, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v189, v[vgprValuC+189]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v189, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v190, v[vgprValuC+190]               // convert C to fp16
buffer_store_b16 v190, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v191, v[vgprValuC+191]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v191, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v192, v[vgprValuC+192]               // convert C to fp16
buffer_store_b16 v192, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v193, v[vgprValuC+193]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v193, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v194, v[vgprValuC+194]               // convert C to fp16
buffer_store_b16 v194, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v195, v[vgprValuC+195]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v195, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v196, v[vgprValuC+196]               // convert C to fp16
buffer_store_b16 v196, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v197, v[vgprValuC+197]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v197, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v198, v[vgprValuC+198]               // convert C to fp16
buffer_store_b16 v198, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v199, v[vgprValuC+199]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v199, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v200, v[vgprValuC+200]               // convert C to fp16
buffer_store_b16 v200, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW1_GSU1_NonEdgeEnd:
label_GW_B0_FD0_VW1_GSU1_Else:
label_GW_B0_FD0_VW1_GSU1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=66 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1); (24,0,0,0:vw1); (24,1,0,0:vw1); (25,0,0,0:vw1); (25,1,0,0:vw1); (26,0,0,0:vw1); (26,1,0,0:vw1); (27,0,0,0:vw1); (27,1,0,0:vw1); (28,0,0,0:vw1); (28,1,0,0:vw1); (29,0,0,0:vw1); (29,1,0,0:vw1); (30,0,0,0:vw1); (30,1,0,0:vw1); (31,0,0,0:vw1); (31,1,0,0:vw1); (32,0,0,0:vw1); (32,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v114, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v185, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v185, v114, v185, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v186, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v186, v114, v186, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v187, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v187, v114, v187, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v188, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v188, v114, v188, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v189, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v189, v114, v189, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v190, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v190, v114, v190, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v191, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v191, v114, v191, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v192, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v192, v114, v192, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v193, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v193, v114, v193, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v194, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v194, v114, v194, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v195, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v195, v114, v195, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v196, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v196, v114, v196, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v197, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v197, v114, v197, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v198, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v198, v114, v198, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v199, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v199, v114, v199, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v200, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v200, v114, v200, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 18                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v201, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v201, v114, v201, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v202, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v202, v114, v202, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v203, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v203, v114, v203, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v204, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v204, v114, v204, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v205, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v205, v114, v205, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v206, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v206, v114, v206, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v207, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v207, v114, v207, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v208, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v208, v114, v208, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v209, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v209, v114, v209, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v210, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v210, v114, v210, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v211, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v211, v114, v211, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v212, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v212, v114, v212, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v213, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v213, v114, v213, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v214, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v214, v114, v214, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v215, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v215, v114, v215, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v216, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v216, v114, v216, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 18                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v217, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v217, v114, v217, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v218, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v218, v114, v218, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v219, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v219, v114, v219, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v220, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v220, v114, v220, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v221, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v221, v114, v221, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v223, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v223, v114, v223, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v224, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v224, v114, v224, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v225, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v225, v114, v225, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v226, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v226, v114, v226, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v227, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v227, v114, v227, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v228, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v228, v114, v228, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v229, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v229, v114, v229, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v230, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v230, v114, v230, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v231, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v231, v114, v231, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v232, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v232, v114, v232, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v233, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v233, v114, v233, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 18                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v234, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v234, v114, v234, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v235, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v235, v114, v235, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v236, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v236, v114, v236, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v237, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v237, v114, v237, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v238, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v238, v114, v238, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v239, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v239, v114, v239, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v240, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v240, v114, v240, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v241, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v241, v114, v241, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v242, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v242, v114, v242, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v243, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v243, v114, v243, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v244, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v244, v114, v244, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v245, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v245, v114, v245, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v246, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v246, v114, v246, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v247, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v247, v114, v247, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v248, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v248, v114, v248, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v249, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v249, v114, v249, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(32,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 18                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v250, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v250, v114, v250, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(32,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v251, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v251, v114, v251, s30                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (19, 0, 0, 0), (19, 1, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0), (22, 0, 0, 0), (22, 1, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0), (24, 0, 0, 0), (24, 1, 0, 0), (25, 0, 0, 0), (25, 1, 0, 0), (26, 0, 0, 0), (26, 1, 0, 0), (27, 0, 0, 0), (27, 1, 0, 0), (28, 0, 0, 0), (28, 1, 0, 0), (29, 0, 0, 0), (29, 1, 0, 0), (30, 0, 0, 0), (30, 1, 0, 0), (31, 0, 0, 0), (31, 1, 0, 0), (32, 0, 0, 0), (32, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+128], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+129], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+130], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+131], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+132], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+133], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+134], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+135], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+152], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+154], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+158], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+160], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+162], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+163], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+164], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+166], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+167], s[sgprAlpha], v[vgprValuC+48] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+168], s[sgprAlpha], v[vgprValuC+56] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+169], s[sgprAlpha], v[vgprValuC+49] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+170], s[sgprAlpha], v[vgprValuC+57] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+171], s[sgprAlpha], v[vgprValuC+50] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+172], s[sgprAlpha], v[vgprValuC+58] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+173], s[sgprAlpha], v[vgprValuC+51] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+174], s[sgprAlpha], v[vgprValuC+59] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+175], s[sgprAlpha], v[vgprValuC+52] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+176], s[sgprAlpha], v[vgprValuC+60] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+177], s[sgprAlpha], v[vgprValuC+53] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+178], s[sgprAlpha], v[vgprValuC+61] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+179], s[sgprAlpha], v[vgprValuC+54] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+180], s[sgprAlpha], v[vgprValuC+62] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+181], s[sgprAlpha], v[vgprValuC+55] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+182], s[sgprAlpha], v[vgprValuC+63] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+183], s[sgprAlpha], v[vgprValuC+64] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+184], s[sgprAlpha], v[vgprValuC+72] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v119, v[vgprValuC+119]               // convert C to fp16
buffer_store_b16 v119, v185, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v120, v[vgprValuC+120]               // convert C to fp16
buffer_store_b16 v120, v186, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v121, v[vgprValuC+121]               // convert C to fp16
buffer_store_b16 v121, v187, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v122, v[vgprValuC+122]               // convert C to fp16
buffer_store_b16 v122, v188, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v123, v[vgprValuC+123]               // convert C to fp16
buffer_store_b16 v123, v189, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v124, v[vgprValuC+124]               // convert C to fp16
buffer_store_b16 v124, v190, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v125, v[vgprValuC+125]               // convert C to fp16
buffer_store_b16 v125, v191, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v126, v[vgprValuC+126]               // convert C to fp16
buffer_store_b16 v126, v192, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v127, v[vgprValuC+127]               // convert C to fp16
buffer_store_b16 v127, v193, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v128, v[vgprValuC+128]               // convert C to fp16
buffer_store_b16 v128, v194, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v129, v[vgprValuC+129]               // convert C to fp16
buffer_store_b16 v129, v195, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v130, v[vgprValuC+130]               // convert C to fp16
buffer_store_b16 v130, v196, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v131, v[vgprValuC+131]               // convert C to fp16
buffer_store_b16 v131, v197, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v132, v[vgprValuC+132]               // convert C to fp16
buffer_store_b16 v132, v198, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v133, v[vgprValuC+133]               // convert C to fp16
buffer_store_b16 v133, v199, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v134, v[vgprValuC+134]               // convert C to fp16
buffer_store_b16 v134, v200, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v135, v[vgprValuC+135]               // convert C to fp16
buffer_store_b16 v135, v201, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v136, v[vgprValuC+136]               // convert C to fp16
buffer_store_b16 v136, v202, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v137, v[vgprValuC+137]               // convert C to fp16
buffer_store_b16 v137, v203, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v138, v[vgprValuC+138]               // convert C to fp16
buffer_store_b16 v138, v204, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v139, v[vgprValuC+139]               // convert C to fp16
buffer_store_b16 v139, v205, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v140, v[vgprValuC+140]               // convert C to fp16
buffer_store_b16 v140, v206, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v141, v[vgprValuC+141]               // convert C to fp16
buffer_store_b16 v141, v207, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v142, v[vgprValuC+142]               // convert C to fp16
buffer_store_b16 v142, v208, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v143, v[vgprValuC+143]               // convert C to fp16
buffer_store_b16 v143, v209, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v144, v[vgprValuC+144]               // convert C to fp16
buffer_store_b16 v144, v210, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v145, v[vgprValuC+145]               // convert C to fp16
buffer_store_b16 v145, v211, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v146, v[vgprValuC+146]               // convert C to fp16
buffer_store_b16 v146, v212, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v147, v[vgprValuC+147]               // convert C to fp16
buffer_store_b16 v147, v213, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v148, v[vgprValuC+148]               // convert C to fp16
buffer_store_b16 v148, v214, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v149, v[vgprValuC+149]               // convert C to fp16
buffer_store_b16 v149, v215, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v150, v[vgprValuC+150]               // convert C to fp16
buffer_store_b16 v150, v216, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v151, v[vgprValuC+151]               // convert C to fp16
buffer_store_b16 v151, v217, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v152, v[vgprValuC+152]               // convert C to fp16
buffer_store_b16 v152, v218, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v153, v[vgprValuC+153]               // convert C to fp16
buffer_store_b16 v153, v219, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v154, v[vgprValuC+154]               // convert C to fp16
buffer_store_b16 v154, v220, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v155, v[vgprValuC+155]               // convert C to fp16
buffer_store_b16 v155, v221, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v156, v[vgprValuC+156]               // convert C to fp16
buffer_store_b16 v156, v223, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v157, v[vgprValuC+157]               // convert C to fp16
buffer_store_b16 v157, v224, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v158, v[vgprValuC+158]               // convert C to fp16
buffer_store_b16 v158, v225, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v159, v[vgprValuC+159]               // convert C to fp16
buffer_store_b16 v159, v226, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v160, v[vgprValuC+160]               // convert C to fp16
buffer_store_b16 v160, v227, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v161, v[vgprValuC+161]               // convert C to fp16
buffer_store_b16 v161, v228, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v162, v[vgprValuC+162]               // convert C to fp16
buffer_store_b16 v162, v229, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v163, v[vgprValuC+163]               // convert C to fp16
buffer_store_b16 v163, v230, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v164, v[vgprValuC+164]               // convert C to fp16
buffer_store_b16 v164, v231, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v165, v[vgprValuC+165]               // convert C to fp16
buffer_store_b16 v165, v232, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v166, v[vgprValuC+166]               // convert C to fp16
buffer_store_b16 v166, v233, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v167, v[vgprValuC+167]               // convert C to fp16
buffer_store_b16 v167, v234, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v168, v[vgprValuC+168]               // convert C to fp16
buffer_store_b16 v168, v235, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v169, v[vgprValuC+169]               // convert C to fp16
buffer_store_b16 v169, v236, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v170, v[vgprValuC+170]               // convert C to fp16
buffer_store_b16 v170, v237, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v171, v[vgprValuC+171]               // convert C to fp16
buffer_store_b16 v171, v238, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v172, v[vgprValuC+172]               // convert C to fp16
buffer_store_b16 v172, v239, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v173, v[vgprValuC+173]               // convert C to fp16
buffer_store_b16 v173, v240, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v174, v[vgprValuC+174]               // convert C to fp16
buffer_store_b16 v174, v241, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v175, v[vgprValuC+175]               // convert C to fp16
buffer_store_b16 v175, v242, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v176, v[vgprValuC+176]               // convert C to fp16
buffer_store_b16 v176, v243, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v177, v[vgprValuC+177]               // convert C to fp16
buffer_store_b16 v177, v244, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v178, v[vgprValuC+178]               // convert C to fp16
buffer_store_b16 v178, v245, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v179, v[vgprValuC+179]               // convert C to fp16
buffer_store_b16 v179, v246, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v180, v[vgprValuC+180]               // convert C to fp16
buffer_store_b16 v180, v247, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v181, v[vgprValuC+181]               // convert C to fp16
buffer_store_b16 v181, v248, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v182, v[vgprValuC+182]               // convert C to fp16
buffer_store_b16 v182, v249, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v183, v[vgprValuC+183]               // convert C to fp16
buffer_store_b16 v183, v250, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v184, v[vgprValuC+184]               // convert C to fp16
buffer_store_b16 v184, v251, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (33,0,0,0:vw1); (33,1,0,0:vw1); (34,0,0,0:vw1); (34,1,0,0:vw1); (35,0,0,0:vw1); (35,1,0,0:vw1); (36,0,0,0:vw1); (36,1,0,0:vw1); (37,0,0,0:vw1); (37,1,0,0:vw1); (38,0,0,0:vw1); (38,1,0,0:vw1); (39,0,0,0:vw1); (39,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v114, BufferOOB
/* (d1,vc1,d0,vc0)=(33,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v133, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v133, v114, v133, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(33,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v134, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v114, v134, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(34,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v135, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v135, v114, v135, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(34,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v136, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v114, v136, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(35,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v137, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v137, v114, v137, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(35,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v138, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v114, v138, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(36,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v139, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v139, v114, v139, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(36,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v140, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v114, v140, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(37,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v141, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v141, v114, v141, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(37,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v142, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v114, v142, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(38,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v143, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v143, v114, v143, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(38,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v144, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v144, v114, v144, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(39,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v145, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v145, v114, v145, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(39,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v146, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v146, v114, v146, s30                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(33, 0, 0, 0), (33, 1, 0, 0), (34, 0, 0, 0), (34, 1, 0, 0), (35, 0, 0, 0), (35, 1, 0, 0), (36, 0, 0, 0), (36, 1, 0, 0), (37, 0, 0, 0), (37, 1, 0, 0), (38, 0, 0, 0), (38, 1, 0, 0), (39, 0, 0, 0), (39, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+65] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+73] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+66] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+74] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+67] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+75] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+68] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+76] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+69] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+128], s[sgprAlpha], v[vgprValuC+77] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+129], s[sgprAlpha], v[vgprValuC+70] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+130], s[sgprAlpha], v[vgprValuC+78] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+131], s[sgprAlpha], v[vgprValuC+71] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+132], s[sgprAlpha], v[vgprValuC+79] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v119, v[vgprValuC+119]               // convert C to fp16
buffer_store_b16 v119, v133, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v120, v[vgprValuC+120]               // convert C to fp16
buffer_store_b16 v120, v134, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v121, v[vgprValuC+121]               // convert C to fp16
buffer_store_b16 v121, v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v122, v[vgprValuC+122]               // convert C to fp16
buffer_store_b16 v122, v136, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v123, v[vgprValuC+123]               // convert C to fp16
buffer_store_b16 v123, v137, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v124, v[vgprValuC+124]               // convert C to fp16
buffer_store_b16 v124, v138, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v125, v[vgprValuC+125]               // convert C to fp16
buffer_store_b16 v125, v139, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v126, v[vgprValuC+126]               // convert C to fp16
buffer_store_b16 v126, v140, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v127, v[vgprValuC+127]               // convert C to fp16
buffer_store_b16 v127, v141, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v128, v[vgprValuC+128]               // convert C to fp16
buffer_store_b16 v128, v142, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v129, v[vgprValuC+129]               // convert C to fp16
buffer_store_b16 v129, v143, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v130, v[vgprValuC+130]               // convert C to fp16
buffer_store_b16 v130, v144, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v131, v[vgprValuC+131]               // convert C to fp16
buffer_store_b16 v131, v145, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v132, v[vgprValuC+132]               // convert C to fp16
buffer_store_b16 v132, v146, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
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

/* Edge/NonEdge store path check (N (isSize1)): Size % 160 > 0 -> Edge store; else -> NonEdge store */
s_mov_b32 s31, 0                                   // STATIC_DIV: divisor=160
s_mul_i32 s30, 819, s[sgprSizeJ]                   // tmp1 = dividend * magic hi
s_lshl_b64 s[30:31], s[30:31], 16                  // left shift 16 bits
s_mul_i32 s29, s[sgprSizeJ], 13108                 // tmp0 = dividend * magic lo
s_add_u32 s30, s29, s30                            // add lo
s_addc_u32 s31, s31, 0                             // add hi
s_lshr_b64 s[30:31], s[30:31], 33                  // tmp1 = (dividend * magic) << shift
s_mov_b32 s29, s30                                 // quotient
s_mul_i32 s30, s29, 160                            // quotient*divisor
s_sub_u32 s28, s[sgprSizeJ], s30                   // rReg = dividend - quotient*divisor
s_add_u32 s29, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s29                // wg1 >= nwg1-1
s_cselect_b32 s28, s28, 0                          // set rem
s_cmpk_gt_u32 s28, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW1_GSU1_Then       // jump if edges required
label_GW_B1_FD0_VW1_GSU1_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=66 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1); (24,0,0,0:vw1); (24,1,0,0:vw1); (25,0,0,0:vw1); (25,1,0,0:vw1); (26,0,0,0:vw1); (26,1,0,0:vw1); (27,0,0,0:vw1); (27,1,0,0:vw1); (28,0,0,0:vw1); (28,1,0,0:vw1); (29,0,0,0:vw1); (29,1,0,0:vw1); (30,0,0,0:vw1); (30,1,0,0:vw1); (31,0,0,0:vw1); (31,1,0,0:vw1); (32,0,0,0:vw1); (32,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v120, v110, v108, 1                 // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=108, coord0Vgpr=108 (multiple bpe)
buffer_load_d16_b16 v187, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
buffer_load_d16_b16 v188, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v189, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
buffer_load_d16_b16 v190, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v191, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
buffer_load_d16_b16 v192, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v193, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
buffer_load_d16_b16 v194, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v195, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
buffer_load_d16_b16 v196, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v197, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
buffer_load_d16_b16 v198, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v199, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
buffer_load_d16_b16 v200, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v201, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
buffer_load_d16_b16 v202, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 36                 // scale StrideC *= numRows(18) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_load_d16_b16 v203, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
buffer_load_d16_b16 v204, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v205, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(9,0,1,0) */
buffer_load_d16_b16 v206, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v207, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(10,0,1,0) */
buffer_load_d16_b16 v208, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v209, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(11,0,1,0) */
buffer_load_d16_b16 v210, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v211, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
buffer_load_d16_b16 v212, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v213, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(13,0,1,0) */
buffer_load_d16_b16 v214, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v215, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(14,0,1,0) */
buffer_load_d16_b16 v216, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v217, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(15,0,1,0) */
buffer_load_d16_b16 v218, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 36                 // scale StrideC *= numRows(18) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_load_d16_b16 v219, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
buffer_load_d16_b16 v220, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v221, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(17,0,1,0) */
buffer_load_d16_b16 v223, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v224, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(18,0,1,0) */
buffer_load_d16_b16 v225, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v226, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(19,0,1,0) */
buffer_load_d16_b16 v227, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v228, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
buffer_load_d16_b16 v229, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v230, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(21,0,1,0) */
buffer_load_d16_b16 v231, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v232, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(22,0,1,0) */
buffer_load_d16_b16 v233, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v234, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(23,0,1,0) */
buffer_load_d16_b16 v235, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 36                 // scale StrideC *= numRows(18) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_load_d16_b16 v236, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(24,0,1,0) */
buffer_load_d16_b16 v237, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v238, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(25,0,1,0) */
buffer_load_d16_b16 v239, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v240, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(26,0,1,0) */
buffer_load_d16_b16 v241, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v242, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(27,0,1,0) */
buffer_load_d16_b16 v243, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v244, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(28,0,1,0) */
buffer_load_d16_b16 v245, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v246, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(29,0,1,0) */
buffer_load_d16_b16 v247, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v248, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(30,0,1,0) */
buffer_load_d16_b16 v249, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v250, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(31,0,1,0) */
buffer_load_d16_b16 v251, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(32,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 36                 // scale StrideC *= numRows(18) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_load_d16_b16 v252, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(32,0,1,0) */
buffer_load_d16_b16 v253, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
v_add_lshl_u32 v119, v111, v108, 1                 // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=108, coord0Vgpr=108 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (19, 0, 0, 0), (19, 1, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0), (22, 0, 0, 0), (22, 1, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0), (24, 0, 0, 0), (24, 1, 0, 0), (25, 0, 0, 0), (25, 1, 0, 0), (26, 0, 0, 0), (26, 1, 0, 0), (27, 0, 0, 0), (27, 1, 0, 0), (28, 0, 0, 0), (28, 1, 0, 0), (29, 0, 0, 0), (29, 1, 0, 0), (30, 0, 0, 0), (30, 1, 0, 0), (31, 0, 0, 0), (31, 1, 0, 0), (32, 0, 0, 0), (32, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+128], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+129], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+130], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+131], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+132], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+133], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+134], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+135], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+152], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+154], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+158], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+160], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+162], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+163], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+164], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+166], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+167], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+168], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+169], s[sgprAlpha], v[vgprValuC+48] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+170], s[sgprAlpha], v[vgprValuC+56] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+171], s[sgprAlpha], v[vgprValuC+49] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+172], s[sgprAlpha], v[vgprValuC+57] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+173], s[sgprAlpha], v[vgprValuC+50] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+174], s[sgprAlpha], v[vgprValuC+58] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+175], s[sgprAlpha], v[vgprValuC+51] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+176], s[sgprAlpha], v[vgprValuC+59] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+177], s[sgprAlpha], v[vgprValuC+52] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+178], s[sgprAlpha], v[vgprValuC+60] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+179], s[sgprAlpha], v[vgprValuC+53] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+180], s[sgprAlpha], v[vgprValuC+61] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+181], s[sgprAlpha], v[vgprValuC+54] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+182], s[sgprAlpha], v[vgprValuC+62] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+183], s[sgprAlpha], v[vgprValuC+55] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+184], s[sgprAlpha], v[vgprValuC+63] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+185], s[sgprAlpha], v[vgprValuC+64] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+186], s[sgprAlpha], v[vgprValuC+72] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(63)                                // vlcnt(65) = 66 - 1 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+121], s[sgprBeta], v187, v[vgprValuC+121] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v121, v[vgprValuC+121]               // convert C to fp16
buffer_store_b16 v121, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(63)                                // vlcnt(64) = 66 - 2 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+122], s[sgprBeta], v188, v[vgprValuC+122] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v122, v[vgprValuC+122]               // convert C to fp16
buffer_store_b16 v122, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(63)                                // vlcnt(63) = 66 - 3 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+123], s[sgprBeta], v189, v[vgprValuC+123] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v123, v[vgprValuC+123]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v123, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(62)                                // vlcnt(62) = 66 - 4 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+124], s[sgprBeta], v190, v[vgprValuC+124] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v124, v[vgprValuC+124]               // convert C to fp16
buffer_store_b16 v124, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(61)                                // vlcnt(61) = 66 - 5 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+125], s[sgprBeta], v191, v[vgprValuC+125] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v125, v[vgprValuC+125]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v125, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(60)                                // vlcnt(60) = 66 - 6 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+126], s[sgprBeta], v192, v[vgprValuC+126] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v126, v[vgprValuC+126]               // convert C to fp16
buffer_store_b16 v126, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(59)                                // vlcnt(59) = 66 - 7 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+127], s[sgprBeta], v193, v[vgprValuC+127] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v127, v[vgprValuC+127]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v127, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(58)                                // vlcnt(58) = 66 - 8 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+128], s[sgprBeta], v194, v[vgprValuC+128] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v128, v[vgprValuC+128]               // convert C to fp16
buffer_store_b16 v128, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(57)                                // vlcnt(57) = 66 - 9 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+129], s[sgprBeta], v195, v[vgprValuC+129] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v129, v[vgprValuC+129]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v129, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(56)                                // vlcnt(56) = 66 - 10 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+130], s[sgprBeta], v196, v[vgprValuC+130] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v130, v[vgprValuC+130]               // convert C to fp16
buffer_store_b16 v130, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(55)                                // vlcnt(55) = 66 - 11 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+131], s[sgprBeta], v197, v[vgprValuC+131] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v131, v[vgprValuC+131]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v131, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(54)                                // vlcnt(54) = 66 - 12 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+132], s[sgprBeta], v198, v[vgprValuC+132] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v132, v[vgprValuC+132]               // convert C to fp16
buffer_store_b16 v132, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(53)                                // vlcnt(53) = 66 - 13 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+133], s[sgprBeta], v199, v[vgprValuC+133] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v133, v[vgprValuC+133]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v133, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(52)                                // vlcnt(52) = 66 - 14 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+134], s[sgprBeta], v200, v[vgprValuC+134] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v134, v[vgprValuC+134]               // convert C to fp16
buffer_store_b16 v134, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(51)                                // vlcnt(51) = 66 - 15 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+135], s[sgprBeta], v201, v[vgprValuC+135] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v135, v[vgprValuC+135]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v135, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(50)                                // vlcnt(50) = 66 - 16 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+136], s[sgprBeta], v202, v[vgprValuC+136] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v136, v[vgprValuC+136]               // convert C to fp16
buffer_store_b16 v136, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(49)                                // vlcnt(49) = 66 - 17 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+137], s[sgprBeta], v203, v[vgprValuC+137] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v137, v[vgprValuC+137]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 36                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b16 v137, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(48)                                // vlcnt(48) = 66 - 18 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+138], s[sgprBeta], v204, v[vgprValuC+138] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v138, v[vgprValuC+138]               // convert C to fp16
buffer_store_b16 v138, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(47)                                // vlcnt(47) = 66 - 19 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+139], s[sgprBeta], v205, v[vgprValuC+139] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v139, v[vgprValuC+139]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v139, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(46)                                // vlcnt(46) = 66 - 20 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+140], s[sgprBeta], v206, v[vgprValuC+140] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v140, v[vgprValuC+140]               // convert C to fp16
buffer_store_b16 v140, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(45)                                // vlcnt(45) = 66 - 21 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+141], s[sgprBeta], v207, v[vgprValuC+141] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v141, v[vgprValuC+141]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v141, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(44)                                // vlcnt(44) = 66 - 22 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+142], s[sgprBeta], v208, v[vgprValuC+142] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v142, v[vgprValuC+142]               // convert C to fp16
buffer_store_b16 v142, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(43)                                // vlcnt(43) = 66 - 23 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+143], s[sgprBeta], v209, v[vgprValuC+143] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v143, v[vgprValuC+143]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v143, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(42)                                // vlcnt(42) = 66 - 24 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+144], s[sgprBeta], v210, v[vgprValuC+144] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v144, v[vgprValuC+144]               // convert C to fp16
buffer_store_b16 v144, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(41)                                // vlcnt(41) = 66 - 25 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+145], s[sgprBeta], v211, v[vgprValuC+145] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v145, v[vgprValuC+145]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v145, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(40)                                // vlcnt(40) = 66 - 26 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+146], s[sgprBeta], v212, v[vgprValuC+146] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v146, v[vgprValuC+146]               // convert C to fp16
buffer_store_b16 v146, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(39)                                // vlcnt(39) = 66 - 27 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+147], s[sgprBeta], v213, v[vgprValuC+147] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v147, v[vgprValuC+147]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v147, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(38)                                // vlcnt(38) = 66 - 28 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+148], s[sgprBeta], v214, v[vgprValuC+148] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v148, v[vgprValuC+148]               // convert C to fp16
buffer_store_b16 v148, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(37)                                // vlcnt(37) = 66 - 29 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+149], s[sgprBeta], v215, v[vgprValuC+149] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v149, v[vgprValuC+149]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v149, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(36)                                // vlcnt(36) = 66 - 30 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+150], s[sgprBeta], v216, v[vgprValuC+150] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v150, v[vgprValuC+150]               // convert C to fp16
buffer_store_b16 v150, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(35)                                // vlcnt(35) = 66 - 31 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+151], s[sgprBeta], v217, v[vgprValuC+151] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v151, v[vgprValuC+151]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v151, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(34)                                // vlcnt(34) = 66 - 32 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+152], s[sgprBeta], v218, v[vgprValuC+152] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v152, v[vgprValuC+152]               // convert C to fp16
buffer_store_b16 v152, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(33)                                // vlcnt(33) = 66 - 33 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+153], s[sgprBeta], v219, v[vgprValuC+153] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v153, v[vgprValuC+153]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 36                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b16 v153, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(32)                                // vlcnt(32) = 66 - 34 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+154], s[sgprBeta], v220, v[vgprValuC+154] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v154, v[vgprValuC+154]               // convert C to fp16
buffer_store_b16 v154, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(31)                                // vlcnt(31) = 66 - 35 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+155], s[sgprBeta], v221, v[vgprValuC+155] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v155, v[vgprValuC+155]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v155, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(30)                                // vlcnt(30) = 66 - 36 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+156], s[sgprBeta], v223, v[vgprValuC+156] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v156, v[vgprValuC+156]               // convert C to fp16
buffer_store_b16 v156, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(29)                                // vlcnt(29) = 66 - 37 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+157], s[sgprBeta], v224, v[vgprValuC+157] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v157, v[vgprValuC+157]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v157, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(28)                                // vlcnt(28) = 66 - 38 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+158], s[sgprBeta], v225, v[vgprValuC+158] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v158, v[vgprValuC+158]               // convert C to fp16
buffer_store_b16 v158, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(27)                                // vlcnt(27) = 66 - 39 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+159], s[sgprBeta], v226, v[vgprValuC+159] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v159, v[vgprValuC+159]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v159, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(26)                                // vlcnt(26) = 66 - 40 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+160], s[sgprBeta], v227, v[vgprValuC+160] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v160, v[vgprValuC+160]               // convert C to fp16
buffer_store_b16 v160, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(25)                                // vlcnt(25) = 66 - 41 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+161], s[sgprBeta], v228, v[vgprValuC+161] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v161, v[vgprValuC+161]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v161, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(24)                                // vlcnt(24) = 66 - 42 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+162], s[sgprBeta], v229, v[vgprValuC+162] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v162, v[vgprValuC+162]               // convert C to fp16
buffer_store_b16 v162, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(23)                                // vlcnt(23) = 66 - 43 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+163], s[sgprBeta], v230, v[vgprValuC+163] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v163, v[vgprValuC+163]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v163, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(22)                                // vlcnt(22) = 66 - 44 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+164], s[sgprBeta], v231, v[vgprValuC+164] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v164, v[vgprValuC+164]               // convert C to fp16
buffer_store_b16 v164, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(21)                                // vlcnt(21) = 66 - 45 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+165], s[sgprBeta], v232, v[vgprValuC+165] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v165, v[vgprValuC+165]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v165, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(20)                                // vlcnt(20) = 66 - 46 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+166], s[sgprBeta], v233, v[vgprValuC+166] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v166, v[vgprValuC+166]               // convert C to fp16
buffer_store_b16 v166, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(19)                                // vlcnt(19) = 66 - 47 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+167], s[sgprBeta], v234, v[vgprValuC+167] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v167, v[vgprValuC+167]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v167, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(18)                                // vlcnt(18) = 66 - 48 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+168], s[sgprBeta], v235, v[vgprValuC+168] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v168, v[vgprValuC+168]               // convert C to fp16
buffer_store_b16 v168, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(17)                                // vlcnt(17) = 66 - 49 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+169], s[sgprBeta], v236, v[vgprValuC+169] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v169, v[vgprValuC+169]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 36                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b16 v169, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(16)                                // vlcnt(16) = 66 - 50 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+170], s[sgprBeta], v237, v[vgprValuC+170] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v170, v[vgprValuC+170]               // convert C to fp16
buffer_store_b16 v170, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(15)                                // vlcnt(15) = 66 - 51 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+171], s[sgprBeta], v238, v[vgprValuC+171] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v171, v[vgprValuC+171]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v171, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(14)                                // vlcnt(14) = 66 - 52 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+172], s[sgprBeta], v239, v[vgprValuC+172] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v172, v[vgprValuC+172]               // convert C to fp16
buffer_store_b16 v172, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(13)                                // vlcnt(13) = 66 - 53 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+173], s[sgprBeta], v240, v[vgprValuC+173] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v173, v[vgprValuC+173]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v173, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(12)                                // vlcnt(12) = 66 - 54 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+174], s[sgprBeta], v241, v[vgprValuC+174] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v174, v[vgprValuC+174]               // convert C to fp16
buffer_store_b16 v174, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(11)                                // vlcnt(11) = 66 - 55 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+175], s[sgprBeta], v242, v[vgprValuC+175] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v175, v[vgprValuC+175]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v175, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(10)                                // vlcnt(10) = 66 - 56 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+176], s[sgprBeta], v243, v[vgprValuC+176] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v176, v[vgprValuC+176]               // convert C to fp16
buffer_store_b16 v176, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(9)                                 // vlcnt(9) = 66 - 57 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+177], s[sgprBeta], v244, v[vgprValuC+177] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v177, v[vgprValuC+177]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v177, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(8)                                 // vlcnt(8) = 66 - 58 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+178], s[sgprBeta], v245, v[vgprValuC+178] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v178, v[vgprValuC+178]               // convert C to fp16
buffer_store_b16 v178, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(7)                                 // vlcnt(7) = 66 - 59 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+179], s[sgprBeta], v246, v[vgprValuC+179] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v179, v[vgprValuC+179]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v179, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(6)                                 // vlcnt(6) = 66 - 60 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+180], s[sgprBeta], v247, v[vgprValuC+180] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v180, v[vgprValuC+180]               // convert C to fp16
buffer_store_b16 v180, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(5)                                 // vlcnt(5) = 66 - 61 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+181], s[sgprBeta], v248, v[vgprValuC+181] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v181, v[vgprValuC+181]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v181, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(4)                                 // vlcnt(4) = 66 - 62 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+182], s[sgprBeta], v249, v[vgprValuC+182] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v182, v[vgprValuC+182]               // convert C to fp16
buffer_store_b16 v182, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(3)                                 // vlcnt(3) = 66 - 63 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+183], s[sgprBeta], v250, v[vgprValuC+183] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v183, v[vgprValuC+183]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v183, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(2)                                 // vlcnt(2) = 66 - 64 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+184], s[sgprBeta], v251, v[vgprValuC+184] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v184, v[vgprValuC+184]               // convert C to fp16
buffer_store_b16 v184, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(1)                                 // vlcnt(1) = 66 - 65 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+185], s[sgprBeta], v252, v[vgprValuC+185] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v185, v[vgprValuC+185]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 36                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b16 v185, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(0)                                 // vlcnt(0) = 66 - 66 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+186], s[sgprBeta], v253, v[vgprValuC+186] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v186, v[vgprValuC+186]               // convert C to fp16
buffer_store_b16 v186, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #1 (d1,d0,vc1,vc0) = */
/*    (33,0,0,0:vw1); (33,1,0,0:vw1); (34,0,0,0:vw1); (34,1,0,0:vw1); (35,0,0,0:vw1); (35,1,0,0:vw1); (36,0,0,0:vw1); (36,1,0,0:vw1); (37,0,0,0:vw1); (37,1,0,0:vw1); (38,0,0,0:vw1); (38,1,0,0:vw1); (39,0,0,0:vw1); (39,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(33,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v135, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(33,0,1,0) */
buffer_load_d16_b16 v136, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(34,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v137, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(34,0,1,0) */
buffer_load_d16_b16 v138, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(35,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v139, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(35,0,1,0) */
buffer_load_d16_b16 v140, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(36,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v141, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(36,0,1,0) */
buffer_load_d16_b16 v142, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(37,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v143, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(37,0,1,0) */
buffer_load_d16_b16 v144, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(38,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v145, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(38,0,1,0) */
buffer_load_d16_b16 v146, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(39,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v147, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(39,0,1,0) */
buffer_load_d16_b16 v148, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C

/* rC *= alpha batchElements=[(33, 0, 0, 0), (33, 1, 0, 0), (34, 0, 0, 0), (34, 1, 0, 0), (35, 0, 0, 0), (35, 1, 0, 0), (36, 0, 0, 0), (36, 1, 0, 0), (37, 0, 0, 0), (37, 1, 0, 0), (38, 0, 0, 0), (38, 1, 0, 0), (39, 0, 0, 0), (39, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+65] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+73] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+66] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+74] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+67] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+75] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+68] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+128], s[sgprAlpha], v[vgprValuC+76] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+129], s[sgprAlpha], v[vgprValuC+69] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+130], s[sgprAlpha], v[vgprValuC+77] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+131], s[sgprAlpha], v[vgprValuC+70] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+132], s[sgprAlpha], v[vgprValuC+78] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+133], s[sgprAlpha], v[vgprValuC+71] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+134], s[sgprAlpha], v[vgprValuC+79] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(13)                                // vlcnt(13) = 14 - 1 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+121], s[sgprBeta], v135, v[vgprValuC+121] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v121, v[vgprValuC+121]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v121, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(12)                                // vlcnt(12) = 14 - 2 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+122], s[sgprBeta], v136, v[vgprValuC+122] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v122, v[vgprValuC+122]               // convert C to fp16
buffer_store_b16 v122, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(11)                                // vlcnt(11) = 14 - 3 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+123], s[sgprBeta], v137, v[vgprValuC+123] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v123, v[vgprValuC+123]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v123, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(10)                                // vlcnt(10) = 14 - 4 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+124], s[sgprBeta], v138, v[vgprValuC+124] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v124, v[vgprValuC+124]               // convert C to fp16
buffer_store_b16 v124, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(9)                                 // vlcnt(9) = 14 - 5 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+125], s[sgprBeta], v139, v[vgprValuC+125] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v125, v[vgprValuC+125]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v125, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(8)                                 // vlcnt(8) = 14 - 6 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+126], s[sgprBeta], v140, v[vgprValuC+126] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v126, v[vgprValuC+126]               // convert C to fp16
buffer_store_b16 v126, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(7)                                 // vlcnt(7) = 14 - 7 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+127], s[sgprBeta], v141, v[vgprValuC+127] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v127, v[vgprValuC+127]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v127, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(6)                                 // vlcnt(6) = 14 - 8 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+128], s[sgprBeta], v142, v[vgprValuC+128] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v128, v[vgprValuC+128]               // convert C to fp16
buffer_store_b16 v128, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(5)                                 // vlcnt(5) = 14 - 9 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+129], s[sgprBeta], v143, v[vgprValuC+129] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v129, v[vgprValuC+129]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v129, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(4)                                 // vlcnt(4) = 14 - 10 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+130], s[sgprBeta], v144, v[vgprValuC+130] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v130, v[vgprValuC+130]               // convert C to fp16
buffer_store_b16 v130, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(3)                                 // vlcnt(3) = 14 - 11 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+131], s[sgprBeta], v145, v[vgprValuC+131] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v131, v[vgprValuC+131]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v131, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(2)                                 // vlcnt(2) = 14 - 12 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+132], s[sgprBeta], v146, v[vgprValuC+132] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v132, v[vgprValuC+132]               // convert C to fp16
buffer_store_b16 v132, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(1)                                 // vlcnt(1) = 14 - 13 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+133], s[sgprBeta], v147, v[vgprValuC+133] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v133, v[vgprValuC+133]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v133, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(0)                                 // vlcnt(0) = 14 - 14 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+134], s[sgprBeta], v148, v[vgprValuC+134] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v134, v[vgprValuC+134]               // convert C to fp16
buffer_store_b16 v134, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_FD0_VW1_GSU1_NonEdgeEnd:
label_GW_B1_FD0_VW1_GSU1_Else:
label_GW_B1_FD0_VW1_GSU1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=44 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v114, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v164, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v164, v114, v164, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v163, v164, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v164, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v164, v114, v164, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v166, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v166, v114, v166, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v165, v166, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v166, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v166, v114, v166, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v168, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v168, v114, v168, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v167, v168, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v168, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v168, v114, v168, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v170, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v170, v114, v170, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v169, v170, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v170, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v170, v114, v170, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v172, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v172, v114, v172, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v171, v172, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v172, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v172, v114, v172, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v174, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v174, v114, v174, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v173, v174, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v174, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v174, v114, v174, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v176, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v176, v114, v176, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v175, v176, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v176, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v176, v114, v176, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v178, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v178, v114, v178, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v177, v178, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v178, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v178, v114, v178, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v180, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v180, v114, v180, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v179, v180, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v180, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v180, v114, v180, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v182, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v182, v114, v182, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v181, v182, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v182, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v182, v114, v182, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v184, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v184, v114, v184, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v183, v184, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v184, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v184, v114, v184, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v186, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v186, v114, v186, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v185, v186, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v186, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v186, v114, v186, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v188, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v188, v114, v188, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v187, v188, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v188, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v188, v114, v188, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v190, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v190, v114, v190, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v189, v190, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v190, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v190, v114, v190, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v192, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v192, v114, v192, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v191, v192, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v192, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v192, v114, v192, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v194, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v194, v114, v194, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v193, v194, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v194, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v194, v114, v194, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 18                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v196, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v196, v114, v196, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v195, v196, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v196, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v196, v114, v196, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v198, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v198, v114, v198, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v197, v198, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v198, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v198, v114, v198, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v200, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v200, v114, v200, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v199, v200, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v200, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v200, v114, v200, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v202, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v202, v114, v202, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v201, v202, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v202, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v202, v114, v202, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v204, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v204, v114, v204, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v203, v204, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v204, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v204, v114, v204, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v206, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v206, v114, v206, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v205, v206, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v206, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v206, v114, v206, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v208, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v208, v114, v208, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v207, v208, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v208, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v208, v114, v208, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v210, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v210, v114, v210, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v209, v210, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v210, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v210, v114, v210, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v212, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v212, v114, v212, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v211, v212, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v212, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v212, v114, v212, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v214, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v214, v114, v214, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v213, v214, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v214, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v214, v114, v214, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v216, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v216, v114, v216, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v215, v216, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v216, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v216, v114, v216, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v218, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v218, v114, v218, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v217, v218, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v218, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v218, v114, v218, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v220, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v220, v114, v220, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v219, v220, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v220, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v220, v114, v220, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v223, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v223, v114, v223, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v221, v223, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v223, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v223, v114, v223, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v225, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v225, v114, v225, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v224, v225, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v225, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v225, v114, v225, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v227, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v227, v114, v227, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v226, v227, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v227, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v227, v114, v227, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 18                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v229, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v229, v114, v229, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v228, v229, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v229, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v229, v114, v229, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v231, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v231, v114, v231, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v230, v231, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v231, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v231, v114, v231, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v233, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v233, v114, v233, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v232, v233, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v233, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v233, v114, v233, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v235, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v235, v114, v235, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v234, v235, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v235, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v235, v114, v235, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v237, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v237, v114, v237, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v236, v237, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v237, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v237, v114, v237, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v239, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v239, v114, v239, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v238, v239, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v239, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v239, v114, v239, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v241, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v241, v114, v241, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v240, v241, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v241, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v241, v114, v241, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v243, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v243, v114, v243, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v242, v243, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v243, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v243, v114, v243, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v245, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v245, v114, v245, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v244, v245, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v245, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v245, v114, v245, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v247, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v247, v114, v247, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v246, v247, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v247, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v247, v114, v247, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v249, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v249, v114, v249, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v248, v249, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v249, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v249, v114, v249, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v251, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v251, v114, v251, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v250, v251, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v251, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v251, v114, v251, s30                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (19, 0, 0, 0), (19, 1, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+128], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+129], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+130], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+131], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+132], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+133], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+134], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+135], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+152], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+154], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+158], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+160], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+162], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fma_mix_f32 v[vgprValuC+119], s[sgprBeta], v163, v[vgprValuC+119] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v119, v[vgprValuC+119]               // convert C to fp16
buffer_store_b16 v119, v164, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+120], s[sgprBeta], v165, v[vgprValuC+120] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v120, v[vgprValuC+120]               // convert C to fp16
buffer_store_b16 v120, v166, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+121], s[sgprBeta], v167, v[vgprValuC+121] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v121, v[vgprValuC+121]               // convert C to fp16
buffer_store_b16 v121, v168, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+122], s[sgprBeta], v169, v[vgprValuC+122] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v122, v[vgprValuC+122]               // convert C to fp16
buffer_store_b16 v122, v170, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+123], s[sgprBeta], v171, v[vgprValuC+123] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v123, v[vgprValuC+123]               // convert C to fp16
buffer_store_b16 v123, v172, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+124], s[sgprBeta], v173, v[vgprValuC+124] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v124, v[vgprValuC+124]               // convert C to fp16
buffer_store_b16 v124, v174, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+125], s[sgprBeta], v175, v[vgprValuC+125] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v125, v[vgprValuC+125]               // convert C to fp16
buffer_store_b16 v125, v176, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+126], s[sgprBeta], v177, v[vgprValuC+126] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v126, v[vgprValuC+126]               // convert C to fp16
buffer_store_b16 v126, v178, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+127], s[sgprBeta], v179, v[vgprValuC+127] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v127, v[vgprValuC+127]               // convert C to fp16
buffer_store_b16 v127, v180, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+128], s[sgprBeta], v181, v[vgprValuC+128] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v128, v[vgprValuC+128]               // convert C to fp16
buffer_store_b16 v128, v182, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+129], s[sgprBeta], v183, v[vgprValuC+129] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v129, v[vgprValuC+129]               // convert C to fp16
buffer_store_b16 v129, v184, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+130], s[sgprBeta], v185, v[vgprValuC+130] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v130, v[vgprValuC+130]               // convert C to fp16
buffer_store_b16 v130, v186, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+131], s[sgprBeta], v187, v[vgprValuC+131] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v131, v[vgprValuC+131]               // convert C to fp16
buffer_store_b16 v131, v188, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+132], s[sgprBeta], v189, v[vgprValuC+132] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v132, v[vgprValuC+132]               // convert C to fp16
buffer_store_b16 v132, v190, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+133], s[sgprBeta], v191, v[vgprValuC+133] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v133, v[vgprValuC+133]               // convert C to fp16
buffer_store_b16 v133, v192, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+134], s[sgprBeta], v193, v[vgprValuC+134] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v134, v[vgprValuC+134]               // convert C to fp16
buffer_store_b16 v134, v194, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+135], s[sgprBeta], v195, v[vgprValuC+135] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v135, v[vgprValuC+135]               // convert C to fp16
buffer_store_b16 v135, v196, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+136], s[sgprBeta], v197, v[vgprValuC+136] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v136, v[vgprValuC+136]               // convert C to fp16
buffer_store_b16 v136, v198, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+137], s[sgprBeta], v199, v[vgprValuC+137] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v137, v[vgprValuC+137]               // convert C to fp16
buffer_store_b16 v137, v200, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+138], s[sgprBeta], v201, v[vgprValuC+138] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v138, v[vgprValuC+138]               // convert C to fp16
buffer_store_b16 v138, v202, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+139], s[sgprBeta], v203, v[vgprValuC+139] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v139, v[vgprValuC+139]               // convert C to fp16
buffer_store_b16 v139, v204, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+140], s[sgprBeta], v205, v[vgprValuC+140] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v140, v[vgprValuC+140]               // convert C to fp16
buffer_store_b16 v140, v206, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+141], s[sgprBeta], v207, v[vgprValuC+141] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v141, v[vgprValuC+141]               // convert C to fp16
buffer_store_b16 v141, v208, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+142], s[sgprBeta], v209, v[vgprValuC+142] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v142, v[vgprValuC+142]               // convert C to fp16
buffer_store_b16 v142, v210, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+143], s[sgprBeta], v211, v[vgprValuC+143] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v143, v[vgprValuC+143]               // convert C to fp16
buffer_store_b16 v143, v212, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+144], s[sgprBeta], v213, v[vgprValuC+144] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v144, v[vgprValuC+144]               // convert C to fp16
buffer_store_b16 v144, v214, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+145], s[sgprBeta], v215, v[vgprValuC+145] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v145, v[vgprValuC+145]               // convert C to fp16
buffer_store_b16 v145, v216, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+146], s[sgprBeta], v217, v[vgprValuC+146] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v146, v[vgprValuC+146]               // convert C to fp16
buffer_store_b16 v146, v218, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+147], s[sgprBeta], v219, v[vgprValuC+147] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v147, v[vgprValuC+147]               // convert C to fp16
buffer_store_b16 v147, v220, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+148], s[sgprBeta], v221, v[vgprValuC+148] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v148, v[vgprValuC+148]               // convert C to fp16
buffer_store_b16 v148, v223, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+149], s[sgprBeta], v224, v[vgprValuC+149] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v149, v[vgprValuC+149]               // convert C to fp16
buffer_store_b16 v149, v225, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+150], s[sgprBeta], v226, v[vgprValuC+150] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v150, v[vgprValuC+150]               // convert C to fp16
buffer_store_b16 v150, v227, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+151], s[sgprBeta], v228, v[vgprValuC+151] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v151, v[vgprValuC+151]               // convert C to fp16
buffer_store_b16 v151, v229, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+152], s[sgprBeta], v230, v[vgprValuC+152] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v152, v[vgprValuC+152]               // convert C to fp16
buffer_store_b16 v152, v231, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+153], s[sgprBeta], v232, v[vgprValuC+153] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v153, v[vgprValuC+153]               // convert C to fp16
buffer_store_b16 v153, v233, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+154], s[sgprBeta], v234, v[vgprValuC+154] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v154, v[vgprValuC+154]               // convert C to fp16
buffer_store_b16 v154, v235, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+155], s[sgprBeta], v236, v[vgprValuC+155] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v155, v[vgprValuC+155]               // convert C to fp16
buffer_store_b16 v155, v237, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+156], s[sgprBeta], v238, v[vgprValuC+156] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v156, v[vgprValuC+156]               // convert C to fp16
buffer_store_b16 v156, v239, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+157], s[sgprBeta], v240, v[vgprValuC+157] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v157, v[vgprValuC+157]               // convert C to fp16
buffer_store_b16 v157, v241, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+158], s[sgprBeta], v242, v[vgprValuC+158] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v158, v[vgprValuC+158]               // convert C to fp16
buffer_store_b16 v158, v243, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+159], s[sgprBeta], v244, v[vgprValuC+159] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v159, v[vgprValuC+159]               // convert C to fp16
buffer_store_b16 v159, v245, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+160], s[sgprBeta], v246, v[vgprValuC+160] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v160, v[vgprValuC+160]               // convert C to fp16
buffer_store_b16 v160, v247, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+161], s[sgprBeta], v248, v[vgprValuC+161] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v161, v[vgprValuC+161]               // convert C to fp16
buffer_store_b16 v161, v249, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+162], s[sgprBeta], v250, v[vgprValuC+162] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v162, v[vgprValuC+162]               // convert C to fp16
buffer_store_b16 v162, v251, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (22,0,0,0:vw1); (22,1,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1); (24,0,0,0:vw1); (24,1,0,0:vw1); (25,0,0,0:vw1); (25,1,0,0:vw1); (26,0,0,0:vw1); (26,1,0,0:vw1); (27,0,0,0:vw1); (27,1,0,0:vw1); (28,0,0,0:vw1); (28,1,0,0:vw1); (29,0,0,0:vw1); (29,1,0,0:vw1); (30,0,0,0:vw1); (30,1,0,0:vw1); (31,0,0,0:vw1); (31,1,0,0:vw1); (32,0,0,0:vw1); (32,1,0,0:vw1); (33,0,0,0:vw1); (33,1,0,0:vw1); (34,0,0,0:vw1); (34,1,0,0:vw1); (35,0,0,0:vw1); (35,1,0,0:vw1); (36,0,0,0:vw1); (36,1,0,0:vw1); (37,0,0,0:vw1); (37,1,0,0:vw1); (38,0,0,0:vw1); (38,1,0,0:vw1); (39,0,0,0:vw1); (39,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v114, BufferOOB
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v156, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v114, v156, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v155, v156, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v156, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v114, v156, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v158, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v114, v158, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v157, v158, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v158, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v114, v158, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v160, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v160, v114, v160, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v159, v160, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v160, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v160, v114, v160, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v162, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v162, v114, v162, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v161, v162, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v162, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v162, v114, v162, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 18                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v164, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v164, v114, v164, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v163, v164, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v164, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v164, v114, v164, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v166, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v166, v114, v166, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v165, v166, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v166, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v166, v114, v166, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v168, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v168, v114, v168, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v167, v168, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v168, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v168, v114, v168, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v170, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v170, v114, v170, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v169, v170, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v170, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v170, v114, v170, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v172, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v172, v114, v172, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v171, v172, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v172, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v172, v114, v172, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v174, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v174, v114, v174, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v173, v174, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v174, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v174, v114, v174, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v176, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v176, v114, v176, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v175, v176, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v176, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v176, v114, v176, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v178, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v178, v114, v178, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v177, v178, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v178, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v178, v114, v178, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v180, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v180, v114, v180, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v179, v180, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v180, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v180, v114, v180, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v182, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v182, v114, v182, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v181, v182, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v182, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v182, v114, v182, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v184, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v184, v114, v184, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v183, v184, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v184, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v184, v114, v184, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v186, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v186, v114, v186, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v185, v186, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v186, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v186, v114, v186, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v188, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v188, v114, v188, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v187, v188, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v188, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v188, v114, v188, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v190, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v190, v114, v190, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v189, v190, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v190, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v190, v114, v190, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v192, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v192, v114, v192, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v191, v192, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v192, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v192, v114, v192, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v194, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v194, v114, v194, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v193, v194, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v194, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v194, v114, v194, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(32,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 18                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v196, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v196, v114, v196, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v195, v196, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v196, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v196, v114, v196, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(32,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v198, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v198, v114, v198, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v197, v198, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v198, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v198, v114, v198, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(33,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v200, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v200, v114, v200, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v199, v200, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v200, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v200, v114, v200, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(33,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v202, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v202, v114, v202, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v201, v202, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v202, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v202, v114, v202, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(34,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v204, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v204, v114, v204, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v203, v204, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v204, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v204, v114, v204, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(34,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v206, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v206, v114, v206, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v205, v206, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v206, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v206, v114, v206, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(35,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v208, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v208, v114, v208, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v207, v208, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v208, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v208, v114, v208, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(35,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v210, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v210, v114, v210, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v209, v210, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v210, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v210, v114, v210, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(36,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v212, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v212, v114, v212, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v211, v212, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v212, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v212, v114, v212, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(36,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v214, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v214, v114, v214, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v213, v214, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v214, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v214, v114, v214, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(37,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v216, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v216, v114, v216, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v215, v216, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v216, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v216, v114, v216, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(37,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v218, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v218, v114, v218, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v217, v218, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v218, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v218, v114, v218, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(38,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v220, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v220, v114, v220, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v219, v220, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v220, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v220, v114, v220, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(38,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v223, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v223, v114, v223, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v221, v223, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v223, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v223, v114, v223, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(39,0,0,0) */
v_add_co_u32 v109, vcc_lo, v109, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s28, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v110, v110, s28                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s28, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v111, v111, s28                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s28, v108, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v225, v110, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v225, v114, v225, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v224, v225, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v225, v111, v108, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v225, v114, v225, s30                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(39,0,1,0) */
v_add_co_u32 v112, vcc_lo, v108, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s28, v112, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s30, v109, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s30, s28, s30                            // in0 && in1
v_add_lshl_u32 v227, v110, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v227, v114, v227, s30                // LDC clip if OOB. offset
buffer_load_d16_b16 v226, v227, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v227, v111, v112, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v227, v114, v227, s30                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(22, 0, 0, 0), (22, 1, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0), (24, 0, 0, 0), (24, 1, 0, 0), (25, 0, 0, 0), (25, 1, 0, 0), (26, 0, 0, 0), (26, 1, 0, 0), (27, 0, 0, 0), (27, 1, 0, 0), (28, 0, 0, 0), (28, 1, 0, 0), (29, 0, 0, 0), (29, 1, 0, 0), (30, 0, 0, 0), (30, 1, 0, 0), (31, 0, 0, 0), (31, 1, 0, 0), (32, 0, 0, 0), (32, 1, 0, 0), (33, 0, 0, 0), (33, 1, 0, 0), (34, 0, 0, 0), (34, 1, 0, 0), (35, 0, 0, 0), (35, 1, 0, 0), (36, 0, 0, 0), (36, 1, 0, 0), (37, 0, 0, 0), (37, 1, 0, 0), (38, 0, 0, 0), (38, 1, 0, 0), (39, 0, 0, 0), (39, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+48] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+56] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+49] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+57] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+50] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+128], s[sgprAlpha], v[vgprValuC+58] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+129], s[sgprAlpha], v[vgprValuC+51] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+130], s[sgprAlpha], v[vgprValuC+59] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+131], s[sgprAlpha], v[vgprValuC+52] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+132], s[sgprAlpha], v[vgprValuC+60] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+133], s[sgprAlpha], v[vgprValuC+53] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+134], s[sgprAlpha], v[vgprValuC+61] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+135], s[sgprAlpha], v[vgprValuC+54] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+62] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+55] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+63] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+64] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+72] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+65] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+73] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+66] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+74] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+67] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+75] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+68] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+76] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+69] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+77] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+70] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+152], s[sgprAlpha], v[vgprValuC+78] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+71] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+154], s[sgprAlpha], v[vgprValuC+79] // Multiply MI out reg with alpha
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fma_mix_f32 v[vgprValuC+119], s[sgprBeta], v155, v[vgprValuC+119] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v119, v[vgprValuC+119]               // convert C to fp16
buffer_store_b16 v119, v156, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+120], s[sgprBeta], v157, v[vgprValuC+120] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v120, v[vgprValuC+120]               // convert C to fp16
buffer_store_b16 v120, v158, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+121], s[sgprBeta], v159, v[vgprValuC+121] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v121, v[vgprValuC+121]               // convert C to fp16
buffer_store_b16 v121, v160, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+122], s[sgprBeta], v161, v[vgprValuC+122] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v122, v[vgprValuC+122]               // convert C to fp16
buffer_store_b16 v122, v162, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+123], s[sgprBeta], v163, v[vgprValuC+123] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v123, v[vgprValuC+123]               // convert C to fp16
buffer_store_b16 v123, v164, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+124], s[sgprBeta], v165, v[vgprValuC+124] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v124, v[vgprValuC+124]               // convert C to fp16
buffer_store_b16 v124, v166, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+125], s[sgprBeta], v167, v[vgprValuC+125] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v125, v[vgprValuC+125]               // convert C to fp16
buffer_store_b16 v125, v168, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+126], s[sgprBeta], v169, v[vgprValuC+126] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v126, v[vgprValuC+126]               // convert C to fp16
buffer_store_b16 v126, v170, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+127], s[sgprBeta], v171, v[vgprValuC+127] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v127, v[vgprValuC+127]               // convert C to fp16
buffer_store_b16 v127, v172, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+128], s[sgprBeta], v173, v[vgprValuC+128] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v128, v[vgprValuC+128]               // convert C to fp16
buffer_store_b16 v128, v174, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+129], s[sgprBeta], v175, v[vgprValuC+129] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v129, v[vgprValuC+129]               // convert C to fp16
buffer_store_b16 v129, v176, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+130], s[sgprBeta], v177, v[vgprValuC+130] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v130, v[vgprValuC+130]               // convert C to fp16
buffer_store_b16 v130, v178, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+131], s[sgprBeta], v179, v[vgprValuC+131] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v131, v[vgprValuC+131]               // convert C to fp16
buffer_store_b16 v131, v180, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+132], s[sgprBeta], v181, v[vgprValuC+132] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v132, v[vgprValuC+132]               // convert C to fp16
buffer_store_b16 v132, v182, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+133], s[sgprBeta], v183, v[vgprValuC+133] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v133, v[vgprValuC+133]               // convert C to fp16
buffer_store_b16 v133, v184, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+134], s[sgprBeta], v185, v[vgprValuC+134] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v134, v[vgprValuC+134]               // convert C to fp16
buffer_store_b16 v134, v186, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+135], s[sgprBeta], v187, v[vgprValuC+135] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v135, v[vgprValuC+135]               // convert C to fp16
buffer_store_b16 v135, v188, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+136], s[sgprBeta], v189, v[vgprValuC+136] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v136, v[vgprValuC+136]               // convert C to fp16
buffer_store_b16 v136, v190, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+137], s[sgprBeta], v191, v[vgprValuC+137] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v137, v[vgprValuC+137]               // convert C to fp16
buffer_store_b16 v137, v192, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+138], s[sgprBeta], v193, v[vgprValuC+138] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v138, v[vgprValuC+138]               // convert C to fp16
buffer_store_b16 v138, v194, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+139], s[sgprBeta], v195, v[vgprValuC+139] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v139, v[vgprValuC+139]               // convert C to fp16
buffer_store_b16 v139, v196, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+140], s[sgprBeta], v197, v[vgprValuC+140] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v140, v[vgprValuC+140]               // convert C to fp16
buffer_store_b16 v140, v198, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+141], s[sgprBeta], v199, v[vgprValuC+141] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v141, v[vgprValuC+141]               // convert C to fp16
buffer_store_b16 v141, v200, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+142], s[sgprBeta], v201, v[vgprValuC+142] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v142, v[vgprValuC+142]               // convert C to fp16
buffer_store_b16 v142, v202, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+143], s[sgprBeta], v203, v[vgprValuC+143] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v143, v[vgprValuC+143]               // convert C to fp16
buffer_store_b16 v143, v204, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+144], s[sgprBeta], v205, v[vgprValuC+144] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v144, v[vgprValuC+144]               // convert C to fp16
buffer_store_b16 v144, v206, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+145], s[sgprBeta], v207, v[vgprValuC+145] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v145, v[vgprValuC+145]               // convert C to fp16
buffer_store_b16 v145, v208, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+146], s[sgprBeta], v209, v[vgprValuC+146] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v146, v[vgprValuC+146]               // convert C to fp16
buffer_store_b16 v146, v210, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+147], s[sgprBeta], v211, v[vgprValuC+147] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v147, v[vgprValuC+147]               // convert C to fp16
buffer_store_b16 v147, v212, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+148], s[sgprBeta], v213, v[vgprValuC+148] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v148, v[vgprValuC+148]               // convert C to fp16
buffer_store_b16 v148, v214, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+149], s[sgprBeta], v215, v[vgprValuC+149] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v149, v[vgprValuC+149]               // convert C to fp16
buffer_store_b16 v149, v216, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+150], s[sgprBeta], v217, v[vgprValuC+150] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v150, v[vgprValuC+150]               // convert C to fp16
buffer_store_b16 v150, v218, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+151], s[sgprBeta], v219, v[vgprValuC+151] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v151, v[vgprValuC+151]               // convert C to fp16
buffer_store_b16 v151, v220, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+152], s[sgprBeta], v221, v[vgprValuC+152] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v152, v[vgprValuC+152]               // convert C to fp16
buffer_store_b16 v152, v223, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+153], s[sgprBeta], v224, v[vgprValuC+153] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v153, v[vgprValuC+153]               // convert C to fp16
buffer_store_b16 v153, v225, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+154], s[sgprBeta], v226, v[vgprValuC+154] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v154, v[vgprValuC+154]               // convert C to fp16
buffer_store_b16 v154, v227, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_End_1:
label_KernelEnd:
s_endpgm                                           // Kernel End
label_ASM_End:  /// The end of the kernel
