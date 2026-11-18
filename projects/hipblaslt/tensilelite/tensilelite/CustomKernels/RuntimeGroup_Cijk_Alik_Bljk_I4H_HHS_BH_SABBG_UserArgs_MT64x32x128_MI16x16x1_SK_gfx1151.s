
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
.text
.protected RuntimeGroup_Cijk_Alik_Bljk_I4H_HHS_BH_SABBG_UserArgs_MT64x32x128_MI16x16x1_SK_gfx1151
.globl RuntimeGroup_Cijk_Alik_Bljk_I4H_HHS_BH_SABBG_UserArgs_MT64x32x128_MI16x16x1_SK_gfx1151
.p2align 8
.type RuntimeGroup_Cijk_Alik_Bljk_I4H_HHS_BH_SABBG_UserArgs_MT64x32x128_MI16x16x1_SK_gfx1151,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel RuntimeGroup_Cijk_Alik_Bljk_I4H_HHS_BH_SABBG_UserArgs_MT64x32x128_MI16x16x1_SK_gfx1151
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 168 // vgprs
  .amdhsa_next_free_sgpr 92 // sgprs
  .amdhsa_group_segment_fixed_size 26112 // lds bytes
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
/* Num VGPR   =168 */
/* Num AccVGPR=0 */
/* Num SGPR   =92 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 8 x 2 */
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
  TileProcessingStrategy: StreamK
  WorkAssignment: StaticGrid
amdhsa.version:
  - 1
  - 1
amdhsa.kernels:
  - .name: RuntimeGroup_Cijk_Alik_Bljk_I4H_HHS_BH_SABBG_UserArgs_MT64x32x128_MI16x16x1_SK_gfx1151
    .symbol: 'RuntimeGroup_Cijk_Alik_Bljk_I4H_HHS_BH_SABBG_UserArgs_MT64x32x128_MI16x16x1_SK_gfx1151.kd'
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
      - .name:            AddressFlags
        .size:            8
        .offset:          64
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            ItersPerTile
        .size:            4
        .offset:          72
        .value_kind:      by_value
        .value_type:      u32
      - .name:            MagicNumberItersPerTile
        .size:            4
        .offset:          76
        .value_kind:      by_value
        .value_type:      u32
      - .name:            MagicShiftItersPerTile
        .size:            4
        .offset:          80
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SKItersPerWG
        .size:            4
        .offset:          84
        .value_kind:      by_value
        .value_type:      u32
      - .name:            skGrid
        .size:            4
        .offset:          88
        .value_kind:      by_value
        .value_type:      u32
      - .name:            skTiles
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
      - .name:            AddressWS
        .size:            8
        .offset:          104
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            D
        .size:            8
        .offset:          112
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            C
        .size:            8
        .offset:          120
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            strideD0
        .size:            4
        .offset:          128
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideD1
        .size:            4
        .offset:          132
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC0
        .size:            4
        .offset:          136
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC1
        .size:            4
        .offset:          140
        .value_kind:      by_value
        .value_type:      u32
      - .name:            AddressScaleA
        .size:            8
        .offset:          144
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            AddressScaleB
        .size:            8
        .offset:          152
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            batchOffsetD
        .size:            8
        .offset:          160
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetC
        .size:            8
        .offset:          168
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetA
        .size:            8
        .offset:          176
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetB
        .size:            8
        .offset:          184
        .value_kind:      by_value
        .value_type:      u64
      - .name:            groupSize
        .size:            4
        .offset:          192
        .value_kind:      by_value
        .value_type:      u32
    .group_segment_fixed_size:   26112
    .kernarg_segment_align:      8
    .kernarg_segment_size:       200
    .max_flat_workgroup_size:    128
    .private_segment_fixed_size: 0
    .sgpr_count:                 92
    .sgpr_spill_count:           0
    .vgpr_count:                 168
    .vgpr_spill_count:           0
    .wavefront_size:             32
...
.end_amdgpu_metadata
RuntimeGroup_Cijk_Alik_Bljk_I4H_HHS_BH_SABBG_UserArgs_MT64x32x128_MI16x16x1_SK_gfx1151:
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
/* ValuC range: [0-16), serializedStore enabled */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprBase, 54
.set vgprLocalWriteAddrA, 48
.set vgprLocalWriteAddrB, 49
.set vgprGlobalReadOffsetA, 16
.set vgprGlobalReadOffsetB, 24
.set vgprGlobalReadOffsetScaleA, 32
.set vgprG2LScaleA, 40
.set vgprLocalReadAddrA, 50
.set vgprLocalReadAddrB, 51
.set vgprSerial, 128

/******************************************/
/* VGPR Macro Assignments                 */
/******************************************/
.set vgprValuA_X0_I0_BASE, vgprBase+0
.set vgprValuB_X0_I0_BASE, vgprBase+9
.set vgprG2LA_BASE, vgprBase+26
.set vgprG2LB_BASE, vgprBase+58
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
.set sgprWGM, 6
.set sgprLoopCounterL, 7
.set sgprOrigLoopCounter, 8
.set sgprSrdD, 12
.set sgprSrdC, 16
.set sgprNumWorkGroups0, 9
.set sgprNumWorkGroups1, 10
.set sgprSizesFree, 20
.set sgprSizesSum, 23
.set sgprAddressA, 24
.set sgprAddressB, 26
.set sgprStridesA, 28
.set sgprStridesB, 30
.set sgprAddressFlags, 32
.set sgprItersPerTile, 34
.set sgprMagicNumberItersPerTile, 35
.set sgprMagicShiftItersPerTile, 36
.set sgprSKItersPerWG, 37
.set sgprskGrid, 38
.set sgprskTiles, 39
.set sgprAlpha, 40
.set sgprBeta, 41
.set sgprAddressWS, 42
.set sgprAddressD, 44
.set sgprAddressC, 46
.set sgprStridesD, 48
.set sgprStridesC, 50
.set sgprSrdWS, 52
.set sgprStreamKLocalEnd, 56
.set sgprStreamKLocalStart, 57
.set sgprPersistentIterationEnd, 58
.set sgprPersistentIteration, 59
.set sgprPersistentWorkGroupIndex, 60
.set sgprAddressScaleA, 62
.set sgprStrideScaleA, 61
.set sgprSrdScaleA, 64
.set sgprScaleAGMinus1, 68
.set sgprScaleAGShift, 69
.set sgprScaleAKMask, 70
.set sgprScaleAInc, 71
.set sgprScaleAKCnt, 72
.set sgprScaleAPkMagic, 73

/* StreamK Parallel Reduction Assignments */
.set sgprSkSplit, sgprskTiles+0
.set sgprSkPartialIdx, sgprBeta+0

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
.set MT1, 32
.set DepthU, 128
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

/* Load num of Gemms */
s_load_b32 s11, s[sgprKernArgAddress:sgprKernArgAddress+1], 0

/* Load packed kernel args (StaggerU/GSU) */
s_load_b32 s17, s[sgprKernArgAddress:sgprKernArgAddress+1], 4

/* Load WGM data */
s_load_b32 s[sgprWGM], s[sgprKernArgAddress:sgprKernArgAddress+1], 8

/* Load num of WGs */
s_load_b32 s18, s[sgprKernArgAddress:sgprKernArgAddress+1], 12
s_waitcnt lgkmcnt(0)                               // load args
s_lshr_b32 s16, s11, 0x1e                          // Get arg type
s_and_b32 s11, 0x3fffffff, s11                     // Get nums of gemm
s_cmp_eq_u32 s16, 3                                // Is kernel argType == 3
s_cbranch_scc1 label_Bypass_ArgType3_to_ArgType0_Instance1
s_cmp_eq_u32 s16, 0                                // Is kernel args
s_cbranch_scc0 label_HBMArgs
label_Bypass_ArgType3_to_ArgType0_Instance1:
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], 0x10 // Shift common args
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_b512 s[20:35], s[sgprKernArgAddress:sgprKernArgAddress+1], 0 // 0
s_load_b512 s[36:51], s[sgprKernArgAddress:sgprKernArgAddress+1], 64 // 64
s_load_b32 s[sgprScaleAGMinus1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0xb0
s_load_b64 s[sgprAddressScaleA:sgprAddressScaleA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x80
s_branch label_LoadArgsEnd
label_HBMArgs:

/* Load address of kernel arguments */
s_load_b64 s[sgprKernArgAddress:sgprKernArgAddress+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 16
s_waitcnt lgkmcnt(0)                               // wait for args to load
label_LoadArgsEnd:
s_mov_b32 s[sgprArgType], s16
s_mov_b32 m0, 0x6600                               // LDS clamp at 26112 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id
s_mov_b32 vcc_hi, 0                                // Ensure hi bits are zero

/* remap workgroup to XCCs */
s_lshr_b32 s80, s[sgprWGM], 0x10                   // Get WGMXCC
s_ff1_i32_b32 s80, s80                             // Get log(WGMXCC)
s_lshr_b32 s81, s[sgprWGM], 0x16                   // Get CU_Count
/* remap WGs if WGMXCC > 1 ( log(WGMXCC) > 0 ) */
s_cmp_gt_i32 s80, 0
s_cbranch_scc0 label_skip_WGMXCC
/* only remap WGs in the range */
s_lshr_b32 s77, s18, s80
s_lshl_b32 s77, s77, s80
s_cmp_ge_u32 s[sgprWorkGroup0], s77
s_cbranch_scc1 label_skip_WGMXCC
s_cmp_eq_u32 s81, 0                                // CU_Count == 0 ?
s_cbranch_scc0 label_XCCG_nonzero
s_lshr_b32 s77, s[sgprWorkGroup0], s80
s_bfm_b32 s78, s80, 0
s_and_b32 s78, s[sgprWorkGroup0], s78
s_lshr_b32 s79, s18, s80
s_mul_i32 s78, s78, s79
s_add_u32 s[sgprWorkGroup0], s77, s78
s_branch label_skip_WGMXCC
label_XCCG_nonzero:
/* temp0 = (wg//CU_Count)*CU_Count */
v_cvt_f64_u32 v[0:1], s81                          // s77 = s[sgprWorkGroup0] / s81
v_rcp_f64 v[0:1], v[0:1]                           // s77 = s[sgprWorkGroup0] / s81
v_cvt_f64_u32 v[2:3], s[sgprWorkGroup0]            // s77 = s[sgprWorkGroup0] / s81
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s77 = s[sgprWorkGroup0] / s81
v_cvt_u32_f64 v0, v[0:1]                           // s77 = s[sgprWorkGroup0] / s81
v_mul_lo_u32 v1, v0, s81                           // s77 = s[sgprWorkGroup0] / s81
v_sub_nc_u32 v2, s[sgprWorkGroup0], v1             // s77 = s[sgprWorkGroup0] / s81
v_cmp_ge_u32 vcc_lo, v2, s81                       // s77 = s[sgprWorkGroup0] / s81
s_mov_b32 exec_lo, vcc_lo                          // s77 = s[sgprWorkGroup0] / s81
v_add_nc_u32 v0, v0, 1                             // s77 = s[sgprWorkGroup0] / s81
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v1, v0, s81                           // s77 = s[sgprWorkGroup0] / s81
v_sub_nc_u32 v2, s[sgprWorkGroup0], v1             // s77 = s[sgprWorkGroup0] / s81
v_readfirstlane_b32 s77, v0                        // quotient
v_readfirstlane_b32 s78, v2                        // remainder
s_mul_i32 s77, s77, s81
/* temp1 = (wg%CU_Count)//WGMXCC */
s_lshr_b32 s78, s78, s80
/* temp0 = temp0 + temp1 */
s_add_u32 s77, s77, s78
/* temp1 = (wg%WGMXCC) * ((WGs - (WGs//CU_Count) * CU_Count) if (wg > (WGs//CU_Count) * CU_Count) else CU_Count)//WGMXCC */
v_cvt_f64_u32 v[0:1], s81                          // s78 = s18 / s81
v_rcp_f64 v[0:1], v[0:1]                           // s78 = s18 / s81
v_cvt_f64_u32 v[2:3], s18                          // s78 = s18 / s81
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s78 = s18 / s81
v_cvt_u32_f64 v0, v[0:1]                           // s78 = s18 / s81
v_mul_lo_u32 v1, v0, s81                           // s78 = s18 / s81
v_sub_nc_u32 v2, s18, v1                           // s78 = s18 / s81
v_cmp_ge_u32 vcc_lo, v2, s81                       // s78 = s18 / s81
s_mov_b32 exec_lo, vcc_lo                          // s78 = s18 / s81
v_add_nc_u32 v0, v0, 1                             // s78 = s18 / s81
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s78, v0                        // quotient
s_mul_i32 s78, s78, s81
s_sub_u32 s79, s18, s78
s_cmp_gt_u32 s[sgprWorkGroup0], s78
s_cselect_b32 s78, s79, s81
s_lshr_b32 s78, s78, s80
s_bfm_b32 s79, s80, 0
s_and_b32 s79, s[sgprWorkGroup0], s79
s_mul_i32 s78, s78, s79
/* WorkGroup0 = temp0 + temp1 */
s_add_u32 s[sgprWorkGroup0], s77, s78
label_skip_WGMXCC:  /// skip WGMXCC if no enough WGs to remap
s_cmp_eq_u32 s16, 3
s_cbranch_scc1 label_ArgType3_Routed_To_ArgType0
s_cmp_eq_u32 s16, 0
s_cbranch_scc0 label_MultiGemm
label_ArgType3_Routed_To_ArgType0:
/* init: add vgpr [54...133) to pool */
/* init: add vgpr [0...16) to pool */
/* init: add agpr [0...0) to pool */
s_waitcnt lgkmcnt(0)                               // wait for 128/0 bytes of kern args over preload
s_lshr_b32 s[sgprNumWorkGroups0], s[sgprSizesFree+0], 6 // s[sgprNumWorkGroups0] = s[sgprSizesFree+0] / 64
s_and_b32 s12, 63, s[sgprSizesFree+0]              // s12 = s[sgprSizesFree+0] % 64
s_cmp_gt_u32 s12, 0                                // Check if remainder > 0
s_addc_u32 s[sgprNumWorkGroups0], s[sgprNumWorkGroups0], 0 // Increment if remainder > 0
s_lshr_b32 s[sgprNumWorkGroups1], s[sgprSizesFree+1], 5 // s[sgprNumWorkGroups1] = s[sgprSizesFree+1] / 32
s_and_b32 s12, 31, s[sgprSizesFree+1]              // s12 = s[sgprSizesFree+1] % 32
s_cmp_gt_u32 s12, 0                                // Check if remainder > 0
s_addc_u32 s[sgprNumWorkGroups1], s[sgprNumWorkGroups1], 0 // Increment if remainder > 0
s_branch label_MultiGemmEnd
label_MultiGemm:

/* Check if custom structure pointer is null */
s_and_b32 s52, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s52, 2                                // ArgType == 2 ?
s_cbranch_scc1 label_IsExternalValid               // branch if ArgType == 2
s_mov_b32 s10, 144                                 // KernArgAddressOffset
s_mul_i32 s18, s11, 4
s_mov_b64 s[12:13], s[sgprKernArgAddress:sgprKernArgAddress+1]
s_branch label_IsExternalValidEnd
label_IsExternalValid:
s_mov_b32 s10, 256
s_mov_b32 s18, 0
s_mov_b64 s[12:13], s[sgprKernArgAddress:sgprKernArgAddress+1]
label_IsExternalValidEnd:

/* Grouped Gemm:: prefetch 1 arg load */
s_mov_b32 s9, 1
s_mov_b32 s19, 0
s_load_b128 s[20:23], s[12:13], s18
s_cmpk_eq_u32 s11, 1                               // if gemm_count is 1?
s_cbranch_scc1 label_wgTable_noLoadLoop

/* Grouped Gemm:: accumulate numTiles for each gemm */
/* Grouped Gemm:: loop start */
label_Loop_GemmCount:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s16, s20, 6                             // s16 = s20 / 64
s_and_b32 s14, 63, s20                             // s14 = s20 % 64
s_addc_u32 s16, s16, 0
s_lshr_b32 s17, s21, 5                             // s17 = s21 / 32
s_and_b32 s14, 31, s21                             // s14 = s21 % 32
s_addc_u32 s17, s17, 0
s_mul_i32 s16, s16, s17
s_mul_i32 s16, s16, s22
s_add_u32 s19, s19, s16
s_cmp_lt_u32 s[sgprWorkGroup0], s19
s_cbranch_scc1 label_FOUND
s_add_u32 s18, s18, s10
s_load_b128 s[20:23], s[12:13], s18
s_add_u32 s9, s9, 1
s_cmp_lt_u32 s9, s11
s_cbranch_scc1 label_Loop_GemmCount

/* Grouped Gemm:: noLoadLoop */
label_wgTable_noLoadLoop:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s16, s20, 6                             // s16 = s20 / 64
s_and_b32 s14, 63, s20                             // s14 = s20 % 64
s_addc_u32 s16, s16, 0
s_lshr_b32 s17, s21, 5                             // s17 = s21 / 32
s_and_b32 s14, 31, s21                             // s14 = s21 % 32
s_addc_u32 s17, s17, 0
s_mul_i32 s16, s16, s17
s_mul_i32 s16, s16, s22
s_add_u32 s19, s19, s16

/* Grouped Gemm:: gemmIndex found */
label_FOUND:
s_sub_u32 s13, s9, 1
s_sub_u32 s12, s19, s16
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s12
/* Check if custom structure pointer is null */
s_and_b32 s52, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s52, 2                                // ArgType == 2 ?
s_cbranch_scc1 label_LoadExternalStruct            // branch if ArgType == 2

/* Grouped Gemm: offset argument address to gemm */
/* Grouped Gemm: offset address from wg_table_start to args_start */
s_lshl2_add_u32 s[sgprKernArgAddress], s11, s[sgprKernArgAddress]
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s13, s13, 144                            // KernArgAddressOffset
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s13
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_b512 s[24:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_b256 s[40:47], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
s_load_b128 s[48:51], s[sgprKernArgAddress:sgprKernArgAddress+1], 112 // 112
s_load_b32 s[sgprScaleAGMinus1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0xb0
s_load_b64 s[sgprAddressScaleA:sgprAddressScaleA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x80
s_branch label_LoadExternalStructEnd
label_LoadExternalStruct:
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s13, s13, 256
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s13
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
s_load_b32 s41, s[sgprKernArgAddress:sgprKernArgAddress+1], 136 // 136
s_load_b64 s[sgprAddressScaleA:sgprAddressScaleA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x8c
label_LoadExternalStructEnd:
/* init: add vgpr [54...133) to pool */
/* init: add vgpr [0...16) to pool */
/* init: add agpr [0...0) to pool */
s_waitcnt lgkmcnt(0)                               // wait for 128/0 bytes of kern args over preload
s_lshr_b32 s[sgprNumWorkGroups0], s[sgprSizesFree+0], 6 // s[sgprNumWorkGroups0] = s[sgprSizesFree+0] / 64
s_and_b32 s12, 63, s[sgprSizesFree+0]              // s12 = s[sgprSizesFree+0] % 64
s_cmp_gt_u32 s12, 0                                // Check if remainder > 0
s_addc_u32 s[sgprNumWorkGroups0], s[sgprNumWorkGroups0], 0 // Increment if remainder > 0
s_lshr_b32 s[sgprNumWorkGroups1], s[sgprSizesFree+1], 5 // s[sgprNumWorkGroups1] = s[sgprSizesFree+1] / 32
s_and_b32 s12, 31, s[sgprSizesFree+1]              // s12 = s[sgprSizesFree+1] % 32
s_cmp_gt_u32 s12, 0                                // Check if remainder > 0
s_addc_u32 s[sgprNumWorkGroups1], s[sgprNumWorkGroups1], 0 // Increment if remainder > 0

/* Early stop if N(SizeFreeJ) == 0 */
s_cmp_eq_u32 s[sgprSizeJ], 0
s_cbranch_scc0 label_NoEarlyStop_N0
label_EarlyStop_if_N_is_0:
s_endpgm
label_NoEarlyStop_N0:

label_MultiGemmEnd:
.set sgprSrdA, 76
.set sgprSrdB, 80
.set sgprShadowLimitA, 84
.set sgprShadowLimitB, 86
s_and_b32 s11, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s11, 3                                // ArgType == 3 for General Batched GEMM
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
s_mov_b32 s[sgprPersistentWorkGroupIndex], s[sgprWorkGroup0] // Save mapped persistent rank
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_SK_SplitInit                  // Jump to single kernel init
v_cvt_f32_u32 v0, s[sgprskTiles]                   // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_rcp_iflag_f32 v0, v0                             // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_mov_b32 v1, 0x4f7ffffe                           // 2^32 - 512
v_mul_f32 v0, v1, v0                               // 2^32 / divisor
v_cvt_u32_f32 v0, v0                               // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_mul_lo_u32 v1, v0, s[sgprskTiles]                // refine 2^32 / divisor
v_sub_nc_u32 v1, 0, v1                             // refine 2^32 / divisor
v_mul_hi_u32 v1, v0, v1                            // refine 2^32 / divisor
v_add_nc_u32 v0, v0, v1                            // refine 2^32 / divisor
v_mul_hi_u32 v0, s[sgprPersistentWorkGroupIndex], v0 // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_mul_lo_u32 v1, v0, s[sgprskTiles]                // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_sub_nc_u32 v1, s[sgprPersistentWorkGroupIndex], v1 // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_cmp_ge_u32 vcc_lo, v1, s[sgprskTiles]            // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v0, 1, v0                             // quotient + 1
v_sub_nc_u32 v1, v1, s[sgprskTiles]                // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v1, s[sgprskTiles]            // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v0, 1, v0                             // quotient + 1
v_sub_nc_u32 v1, v1, s[sgprskTiles]                // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s11, v0                        // quotient
v_readfirstlane_b32 s12, v1                        // remainder
s_mul_i32 s13, s[sgprskTiles], s[sgprSKItersPerWG]
s_sub_u32 s13, s[sgprItersPerTile], s13            // extraIters = itersPerTile - SkSplit * skItersPerWG
s_mul_i32 s[sgprPersistentIteration], s12, s[sgprSKItersPerWG] // StreamK starting iteration (case: after extra iters)
s_cmp_lt_u32 s12, s13                              // Check if WG gets an extra iteration
s_cbranch_scc1 label_SK_HasExtra                   // Has extra iter
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s13 // This WG does not have an extra iteration
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIteration], s[sgprSKItersPerWG] // StreamK ending iteration (case: after extra iters)
s_branch label_SK_DoneExtra                        // Done init for parallel reduction
label_SK_HasExtra:
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s12 // This WG has an extra iteration
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIteration], s[sgprSKItersPerWG] // StreamK ending iteration (case: after extra iters)
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], 1 // StreamK ending iteration (case: after extra iters)
label_SK_DoneExtra:
s_mul_i32 s11, s11, s[sgprItersPerTile]            // Tile offset = tilesIdx * itersPerTile
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s11 // Offset to correct tile
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], s11 // Offset to correct tile
s_mov_b32 s[sgprSkPartialIdx], s12                 // Save partial idx for SrdD calculation
s_branch label_SK_InitDone                         // Done init for parallel reduction
label_SK_SplitInit:
s_mul_i32 s[sgprPersistentIteration], s[sgprPersistentWorkGroupIndex], s[sgprItersPerTile] // DP starting iteration (case: DP work to do)
s_mul_i32 s11, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1] // totalTiles = nwg0 * nwg1
s_mul_i32 s11, s11, s[sgprSizesFree+2]             // totalTiles *= batch dim 0
s_mul_i32 s11, s11, s[sgprItersPerTile]            // totalIters = totalTiles * itersPerTile
s_mov_b32 s[sgprPersistentIterationEnd], s11       // DP ending iteration (case: only DP work to do)
s_mul_i32 s11, s[sgprskTiles], s[sgprItersPerTile] // Total SK iters
s_cmp_lt_u32 s11, s[sgprPersistentIterationEnd]    // Check if there are DP tiles to do
s_cbranch_scc1 label_SK_InitDone                   // Done init
s_mul_i32 s11, s[sgprskTiles], s[sgprItersPerTile]
s_mul_i32 s12, s[sgprSKItersPerWG], s[sgprskGrid]
s_sub_u32 s11, s11, s12                            // skTiles * ItersPerTile - SKItersPerWG * skGrid
s_bitcmp1_b32 s[sgprMagicShiftItersPerTile], 29    // USO on? (bit 29 of MagicShiftItersPerTile); off -> historical global first-E mapping
s_cbranch_scc0 label_SK_GlobalExtraIters           // USO off -> historical global mapping
s_cmp_eq_u32 s[sgprskTiles], 0                     // skTiles == 0?
s_cbranch_scc1 label_SK_AssignNoTiles              // no SK tiles -> global mapping
v_cvt_f32_u32 v0, s[sgprskTiles]                   // F = skGrid / skTiles, rem = skGrid % skTiles
v_rcp_iflag_f32 v0, v0                             // F = skGrid / skTiles, rem = skGrid % skTiles
v_mov_b32 v1, 0x4f7ffffe                           // 2^32 - 512
v_mul_f32 v0, v1, v0                               // 2^32 / divisor
v_cvt_u32_f32 v0, v0                               // F = skGrid / skTiles, rem = skGrid % skTiles
v_mul_lo_u32 v1, v0, s[sgprskTiles]                // refine 2^32 / divisor
v_sub_nc_u32 v1, 0, v1                             // refine 2^32 / divisor
v_mul_hi_u32 v1, v0, v1                            // refine 2^32 / divisor
v_add_nc_u32 v0, v0, v1                            // refine 2^32 / divisor
v_mul_hi_u32 v0, s[sgprskGrid], v0                 // F = skGrid / skTiles, rem = skGrid % skTiles
v_mul_lo_u32 v1, v0, s[sgprskTiles]                // F = skGrid / skTiles, rem = skGrid % skTiles
v_sub_nc_u32 v1, s[sgprskGrid], v1                 // F = skGrid / skTiles, rem = skGrid % skTiles
v_cmp_ge_u32 vcc_lo, v1, s[sgprskTiles]            // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v0, 1, v0                             // quotient + 1
v_sub_nc_u32 v1, v1, s[sgprskTiles]                // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v1, s[sgprskTiles]            // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v0, 1, v0                             // quotient + 1
v_sub_nc_u32 v1, v1, s[sgprskTiles]                // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s12, v0                        // quotient
v_readfirstlane_b32 s13, v1                        // remainder
s_cmp_eq_u32 s13, 0                                // skGrid % skTiles == 0?
s_cbranch_scc1 label_SK_PerTileExtraIters          // all-partial -> per-tile extras
s_branch label_SK_GlobalExtraIters                 // ragged -> global mapping
label_SK_AssignNoTiles:
label_SK_GlobalExtraIters:
s_mul_i32 s[sgprPersistentIteration], s[sgprPersistentWorkGroupIndex], s[sgprSKItersPerWG] // StreamK starting iteration (case: after extra iters)
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s11 // Add extra iters
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIteration], s[sgprSKItersPerWG] // StreamK ending iteration (case: after extra iters)
s_add_u32 s13, s[sgprSKItersPerWG], 1              // Spread out extra iterations
s_mul_i32 s12, s[sgprPersistentWorkGroupIndex], s13 // StreamK starting iteration (case: before extra iters)
s_add_u32 s13, s12, s13                            // StreamK ending iteration (case: before extra iters)
s_cmp_lt_u32 s[sgprPersistentWorkGroupIndex], s11  // Check if lane gets an extra iteration
s_cselect_b32 s[sgprPersistentIteration], s12, s[sgprPersistentIteration] // Set start iter
s_cselect_b32 s[sgprPersistentIterationEnd], s13, s[sgprPersistentIterationEnd] // Set end iter
s_branch label_SK_AssignItersDone                  // skip per-tile path
label_SK_PerTileExtraIters:
s_mov_b32 s11, s12                                 // F = skGrid / skTiles
v_cvt_f32_u32 v0, s11                              // q = w/F, s = w%F
v_rcp_iflag_f32 v0, v0                             // q = w/F, s = w%F
v_mov_b32 v1, 0x4f7ffffe                           // 2^32 - 512
v_mul_f32 v0, v1, v0                               // 2^32 / divisor
v_cvt_u32_f32 v0, v0                               // q = w/F, s = w%F
v_mul_lo_u32 v1, v0, s11                           // refine 2^32 / divisor
v_sub_nc_u32 v1, 0, v1                             // refine 2^32 / divisor
v_mul_hi_u32 v1, v0, v1                            // refine 2^32 / divisor
v_add_nc_u32 v0, v0, v1                            // refine 2^32 / divisor
v_mul_hi_u32 v0, s[sgprPersistentWorkGroupIndex], v0 // q = w/F, s = w%F
v_mul_lo_u32 v1, v0, s11                           // q = w/F, s = w%F
v_sub_nc_u32 v1, s[sgprPersistentWorkGroupIndex], v1 // q = w/F, s = w%F
v_cmp_ge_u32 vcc_lo, v1, s11                       // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v0, 1, v0                             // quotient + 1
v_sub_nc_u32 v1, v1, s11                           // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v1, s11                       // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v0, 1, v0                             // quotient + 1
v_sub_nc_u32 v1, v1, s11                           // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s12, v0                        // quotient
v_readfirstlane_b32 s13, v1                        // remainder
s_mul_i32 s[sgprPersistentIteration], s12, s[sgprItersPerTile] // q * ItersPerTile
s_mul_i32 s12, s13, s[sgprSKItersPerWG]            // s * SKItersPerWG
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s12 // q*I + s*W
s_mul_i32 s12, s11, s[sgprSKItersPerWG]            // F * SKItersPerWG
s_sub_u32 s12, s[sgprItersPerTile], s12            // remI = ItersPerTile - F*SKItersPerWG
s_min_u32 s11, s13, s12                            // min(s, remI)
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s11 // start = q*I + s*W + min(s, remI)
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIteration], s[sgprSKItersPerWG] // start + SKItersPerWG
s_cmp_lt_u32 s13, s12                              // s < remI?
s_cselect_b32 s11, 1, 0                            // extra iter within tile
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], s11 // end = start + W + (s < remI)
label_SK_AssignItersDone:
s_mul_i32 s11, s[sgprskTiles], s[sgprItersPerTile] // Total SK iters
s_min_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], s11 // Cap ending iter at total SK iters
label_SK_InitDone:
s_mul_i32 s11, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1] // totalTiles = nwg0 * nwg1
s_mul_i32 s11, s11, s[sgprSizesFree+2]             // totalTiles *= batch dim 0
s_mul_i32 s11, s11, s[sgprItersPerTile]            // totalIters = totalTiles * itersPerTile
s_cmp_lt_u32 s[sgprPersistentIteration], s11       // Make sure there's work to do
s_cbranch_scc1 label_NoBranch_0                    // Only branch on scc0
s_getpc_b64 s[12:13]                               // addr of next instr
s_add_i32 s14, label_KernelEnd, 4                  // target branch offset
s_add_u32 s12, s12, s14                            // add target branch offset
s_addc_u32 s13, s13, 0                             // add high and carry
s_setpc_b64 s[12:13]                               // branch to label_KernelEnd
label_NoBranch_0:

/******************************************/
/* Persistent Loop Start                  */
/******************************************/
label_PersistentLoopStart:

/******************************************/
/* Begin setupNewTile                     */
/******************************************/

/* global read addresses: work-group */
/* graWorkGroup mapping */
/* StreamK calculate tile idx and map to WG */
s_mul_hi_u32 s13, s[sgprPersistentIteration], s[sgprMagicNumberItersPerTile] // s_magic mul, div alg 2
s_lshr_b32 s14, s[sgprMagicShiftItersPerTile], 31  // tmpS = extract abit
s_mul_i32 s12, s[sgprPersistentIteration], s14     // s_magic mul, div alg 2
s_add_u32 s12, s12, s13
s_and_b32 s14, s[sgprMagicShiftItersPerTile], 2147483647 // tmpS = remove abit to final shift
s_lshr_b32 s12, s12, s14                           // sMagicDiv Alg 2
s_mul_i32 s13, s12, s[sgprItersPerTile]            // Tile start iteration
s_add_u32 s14, s13, s[sgprItersPerTile]            // Tile end iteration
s_sub_u32 s[sgprStreamKLocalStart], s[sgprPersistentIteration], s13 // Local iteration start
s_min_u32 s[sgprStreamKLocalEnd], s[sgprPersistentIterationEnd], s14 // 1. (Local) iteration end (SK tile)
s_sub_u32 s[sgprStreamKLocalEnd], s[sgprStreamKLocalEnd], s13 // 2. Local iteration end (SK tile)
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_SK_SplitUpdate                // Jump to single kernel update
s_mov_b32 s13, s[sgprPersistentIterationEnd]       // Parallel reduction, work contained to single partial tile
s_branch label_SK_UpdateDone                       // Done update for parallel reduction
label_SK_SplitUpdate:
s_mul_i32 s15, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1] // totalTiles = nwg0 * nwg1
s_mul_i32 s15, s15, s[sgprSizesFree+2]             // totalTiles *= batch dim 0
s_sub_u32 s15, s15, s[sgprskTiles]                 // dpTiles = totalTiles - skTiles
s_mul_i32 s15, s15, s[sgprItersPerTile]            // dpSectionSize = dpTiles * ItersPerTile
s_mul_i32 s13, s[sgprskGrid], s[sgprItersPerTile]  // DP iterations shift
s_add_u32 s13, s13, s[sgprPersistentIteration]     // Add DP shift
s_cmp_lt_u32 s13, s15                              // Check if still in DP section
s_cbranch_scc1 label_SK_UpdateDone                 // Done update
s_mov_b32 s13, s14                                 // SK iterations shift
s_cmp_le_u32 s15, s[sgprPersistentIteration]       // Check if continuing in SK section
s_cbranch_scc1 label_SK_UpdateDone                 // Done update
s_mov_b32 s16, s15                                 // park dpSectionSize
s_mov_b32 s17, s12                                 // park current tile idx
s_mul_i32 s11, s[sgprskTiles], s[sgprItersPerTile]
s_mul_i32 s12, s[sgprSKItersPerWG], s[sgprskGrid]
s_sub_u32 s11, s11, s12                            // skTiles * ItersPerTile - SKItersPerWG * skGrid
s_bitcmp1_b32 s[sgprMagicShiftItersPerTile], 29    // USO on? (bit 29 of MagicShiftItersPerTile); off -> historical global first-E mapping
s_cbranch_scc0 label_SK_GlobalExtraIters_1         // USO off -> historical global mapping
s_cmp_eq_u32 s[sgprskTiles], 0                     // skTiles == 0?
s_cbranch_scc1 label_SK_AssignNoTiles_1            // no SK tiles -> global mapping
v_cvt_f32_u32 v0, s[sgprskTiles]                   // F = skGrid / skTiles, rem = skGrid % skTiles
v_rcp_iflag_f32 v0, v0                             // F = skGrid / skTiles, rem = skGrid % skTiles
v_mov_b32 v1, 0x4f7ffffe                           // 2^32 - 512
v_mul_f32 v0, v1, v0                               // 2^32 / divisor
v_cvt_u32_f32 v0, v0                               // F = skGrid / skTiles, rem = skGrid % skTiles
v_mul_lo_u32 v1, v0, s[sgprskTiles]                // refine 2^32 / divisor
v_sub_nc_u32 v1, 0, v1                             // refine 2^32 / divisor
v_mul_hi_u32 v1, v0, v1                            // refine 2^32 / divisor
v_add_nc_u32 v0, v0, v1                            // refine 2^32 / divisor
v_mul_hi_u32 v0, s[sgprskGrid], v0                 // F = skGrid / skTiles, rem = skGrid % skTiles
v_mul_lo_u32 v1, v0, s[sgprskTiles]                // F = skGrid / skTiles, rem = skGrid % skTiles
v_sub_nc_u32 v1, s[sgprskGrid], v1                 // F = skGrid / skTiles, rem = skGrid % skTiles
v_cmp_ge_u32 vcc_lo, v1, s[sgprskTiles]            // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v0, 1, v0                             // quotient + 1
v_sub_nc_u32 v1, v1, s[sgprskTiles]                // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v1, s[sgprskTiles]            // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v0, 1, v0                             // quotient + 1
v_sub_nc_u32 v1, v1, s[sgprskTiles]                // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s12, v0                        // quotient
v_readfirstlane_b32 s13, v1                        // remainder
s_cmp_eq_u32 s13, 0                                // skGrid % skTiles == 0?
s_cbranch_scc1 label_SK_PerTileExtraIters_1        // all-partial -> per-tile extras
s_branch label_SK_GlobalExtraIters_1               // ragged -> global mapping
label_SK_AssignNoTiles_1:
label_SK_GlobalExtraIters_1:
s_mul_i32 s[sgprPersistentIteration], s[sgprPersistentWorkGroupIndex], s[sgprSKItersPerWG] // StreamK starting iteration (case: after extra iters)
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s11 // Add extra iters
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIteration], s[sgprSKItersPerWG] // StreamK ending iteration (case: after extra iters)
s_add_u32 s13, s[sgprSKItersPerWG], 1              // Spread out extra iterations
s_mul_i32 s12, s[sgprPersistentWorkGroupIndex], s13 // StreamK starting iteration (case: before extra iters)
s_add_u32 s13, s12, s13                            // StreamK ending iteration (case: before extra iters)
s_cmp_lt_u32 s[sgprPersistentWorkGroupIndex], s11  // Check if lane gets an extra iteration
s_cselect_b32 s[sgprPersistentIteration], s12, s[sgprPersistentIteration] // Set start iter
s_cselect_b32 s[sgprPersistentIterationEnd], s13, s[sgprPersistentIterationEnd] // Set end iter
s_branch label_SK_AssignItersDone_1                // skip per-tile path
label_SK_PerTileExtraIters_1:
s_mov_b32 s11, s12                                 // F = skGrid / skTiles
v_cvt_f32_u32 v0, s11                              // q = w/F, s = w%F
v_rcp_iflag_f32 v0, v0                             // q = w/F, s = w%F
v_mov_b32 v1, 0x4f7ffffe                           // 2^32 - 512
v_mul_f32 v0, v1, v0                               // 2^32 / divisor
v_cvt_u32_f32 v0, v0                               // q = w/F, s = w%F
v_mul_lo_u32 v1, v0, s11                           // refine 2^32 / divisor
v_sub_nc_u32 v1, 0, v1                             // refine 2^32 / divisor
v_mul_hi_u32 v1, v0, v1                            // refine 2^32 / divisor
v_add_nc_u32 v0, v0, v1                            // refine 2^32 / divisor
v_mul_hi_u32 v0, s[sgprPersistentWorkGroupIndex], v0 // q = w/F, s = w%F
v_mul_lo_u32 v1, v0, s11                           // q = w/F, s = w%F
v_sub_nc_u32 v1, s[sgprPersistentWorkGroupIndex], v1 // q = w/F, s = w%F
v_cmp_ge_u32 vcc_lo, v1, s11                       // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v0, 1, v0                             // quotient + 1
v_sub_nc_u32 v1, v1, s11                           // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v1, s11                       // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v0, 1, v0                             // quotient + 1
v_sub_nc_u32 v1, v1, s11                           // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s12, v0                        // quotient
v_readfirstlane_b32 s13, v1                        // remainder
s_mul_i32 s[sgprPersistentIteration], s12, s[sgprItersPerTile] // q * ItersPerTile
s_mul_i32 s12, s13, s[sgprSKItersPerWG]            // s * SKItersPerWG
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s12 // q*I + s*W
s_mul_i32 s12, s11, s[sgprSKItersPerWG]            // F * SKItersPerWG
s_sub_u32 s12, s[sgprItersPerTile], s12            // remI = ItersPerTile - F*SKItersPerWG
s_min_u32 s11, s13, s12                            // min(s, remI)
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s11 // start = q*I + s*W + min(s, remI)
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIteration], s[sgprSKItersPerWG] // start + SKItersPerWG
s_cmp_lt_u32 s13, s12                              // s < remI?
s_cselect_b32 s11, 1, 0                            // extra iter within tile
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], s11 // end = start + W + (s < remI)
label_SK_AssignItersDone_1:
s_mov_b32 s12, s17                                 // restore current tile idx
s_add_u32 s13, s[sgprPersistentIteration], s16     // Offset to start of SK section
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], s16 // Offset to start of SK section
s_mul_i32 s11, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1] // totalTiles = nwg0 * nwg1
s_mul_i32 s11, s11, s[sgprSizesFree+2]             // totalTiles *= batch dim 0
s_mul_i32 s11, s11, s[sgprItersPerTile]            // totalIters = totalTiles * itersPerTile
s_min_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], s11 // Cap ending iter at total SK iters
s_cmp_lt_u32 s[sgprPersistentIteration], s11       // Make sure there's work to do
s_cbranch_scc1 label_NoBranch_2                    // Only branch on scc0
s_getpc_b64 s[16:17]                               // addr of next instr
s_add_i32 s18, label_KernelEnd, 4                  // target branch offset
s_add_u32 s16, s16, s18                            // add target branch offset
s_addc_u32 s17, s17, 0                             // add high and carry
s_setpc_b64 s[16:17]                               // branch to label_KernelEnd
label_NoBranch_2:
label_SK_UpdateDone:
s_mov_b32 s[sgprPersistentIteration], s13          // Store current iteration
/* Map persistent tile index to wg0/1/2 */
s_mul_i32 s13, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1] // Total tiles
v_cvt_f32_u32 v0, s13                              // TileID // nWG0*nWG1
v_rcp_iflag_f32 v0, v0                             // TileID // nWG0*nWG1
v_mov_b32 v1, 0x4f7ffffe                           // 2^32 - 512
v_mul_f32 v0, v1, v0                               // 2^32 / divisor
v_cvt_u32_f32 v0, v0                               // TileID // nWG0*nWG1
v_mul_lo_u32 v1, v0, s13                           // refine 2^32 / divisor
v_sub_nc_u32 v1, 0, v1                             // refine 2^32 / divisor
v_mul_hi_u32 v1, v0, v1                            // refine 2^32 / divisor
v_add_nc_u32 v0, v0, v1                            // refine 2^32 / divisor
v_mul_hi_u32 v0, s12, v0                           // TileID // nWG0*nWG1
v_mul_lo_u32 v1, v0, s13                           // TileID // nWG0*nWG1
v_sub_nc_u32 v1, s12, v1                           // TileID // nWG0*nWG1
v_cmp_ge_u32 vcc_lo, v1, s13                       // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v0, 1, v0                             // quotient + 1
v_sub_nc_u32 v1, v1, s13                           // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v1, s13                       // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v0, 1, v0                             // quotient + 1
v_sub_nc_u32 v1, v1, s13                           // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprWorkGroup2], v0          // quotient
v_readfirstlane_b32 s14, v1                        // remainder
v_cvt_f32_u32 v0, s[sgprNumWorkGroups0]            // TileID // nWG0
v_rcp_iflag_f32 v0, v0                             // TileID // nWG0
v_mov_b32 v1, 0x4f7ffffe                           // 2^32 - 512
v_mul_f32 v0, v1, v0                               // 2^32 / divisor
v_cvt_u32_f32 v0, v0                               // TileID // nWG0
v_mul_lo_u32 v1, v0, s[sgprNumWorkGroups0]         // refine 2^32 / divisor
v_sub_nc_u32 v1, 0, v1                             // refine 2^32 / divisor
v_mul_hi_u32 v1, v0, v1                            // refine 2^32 / divisor
v_add_nc_u32 v0, v0, v1                            // refine 2^32 / divisor
v_mul_hi_u32 v0, s14, v0                           // TileID // nWG0
v_mul_lo_u32 v1, v0, s[sgprNumWorkGroups0]         // TileID // nWG0
v_sub_nc_u32 v1, s14, v1                           // TileID // nWG0
v_cmp_ge_u32 vcc_lo, v1, s[sgprNumWorkGroups0]     // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v0, 1, v0                             // quotient + 1
v_sub_nc_u32 v1, v1, s[sgprNumWorkGroups0]         // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v1, s[sgprNumWorkGroups0]     // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v0, 1, v0                             // quotient + 1
v_sub_nc_u32 v1, v1, s[sgprNumWorkGroups0]         // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprWorkGroup1], v0          // quotient
v_readfirstlane_b32 s[sgprWorkGroup0], v1          // remainder

v_cmp_eq_f32 vcc_lo, s[sgprAlpha], 0.0             // s[Alpha] == 0.0f ?
s_cbranch_vccz label_SKAlphaCheck                  // branch if s[Alpha] != 0
s_cmp_eq_u32 s[sgprStreamKLocalStart], 0           // does wg start tile?
s_cbranch_scc1 label_NoBranch_4                    // Only branch on scc0
s_getpc_b64 s[16:17]                               // addr of next instr
s_add_i32 s18, label_PersistentLoopClose, 4        // target branch offset
s_add_u32 s16, s16, s18                            // add target branch offset
s_addc_u32 s17, s17, 0                             // add high and carry
s_setpc_b64 s[16:17]                               // branch to label_PersistentLoopClose
label_NoBranch_4:
s_mov_b32 s[sgprStreamKLocalEnd], s[sgprItersPerTile] // Skip iterations
label_SKAlphaCheck:
/* WGM Calculation */
s_mov_b32 s11, s[sgprWGM]                          // Restore WGM
s_sext_i32_i16 s11, s11                            // Restore WGM
s_cmp_gt_i32 s11, 1                                // WGM > 1 ?
s_cbranch_scc1 label_WGMPositive                   // branch if WGM > 1
s_cmp_ge_i32 s11, 0                                // WGM >= 0 ?
s_cbranch_scc1 label_WGM                           // branch if WGM >= 0
s_abs_i32 s11, s11                                 // abs(WGM)
v_cvt_f64_u32 v[0:1], s11                          // s14 = s[sgprWorkGroup0] / s11
v_rcp_f64 v[0:1], v[0:1]                           // s14 = s[sgprWorkGroup0] / s11
v_cvt_f64_u32 v[2:3], s[sgprWorkGroup0]            // s14 = s[sgprWorkGroup0] / s11
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s14 = s[sgprWorkGroup0] / s11
v_cvt_u32_f64 v0, v[0:1]                           // s14 = s[sgprWorkGroup0] / s11
v_mul_lo_u32 v1, v0, s11                           // s14 = s[sgprWorkGroup0] / s11
v_sub_nc_u32 v2, s[sgprWorkGroup0], v1             // s14 = s[sgprWorkGroup0] / s11
v_cmp_ge_u32 vcc_lo, v2, s11                       // s14 = s[sgprWorkGroup0] / s11
s_mov_b32 exec_lo, vcc_lo                          // s14 = s[sgprWorkGroup0] / s11
v_add_nc_u32 v0, v0, 1                             // s14 = s[sgprWorkGroup0] / s11
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s14, v0                        // quotient
s_mul_i32 s15, s14, s11                            // quotient * non-magic divisor
s_sub_u32 s15, s[sgprWorkGroup0], s15              // WorkGroup0=remainder
s_mul_i32 s15, s15, s[sgprNumWorkGroups1]          // (wg1 % WGM)*NumWorkGroups1
s_add_u32 s15, s15, s[sgprWorkGroup1]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups1
v_cvt_f64_u32 v[0:1], s11                          // s12 = s[sgprNumWorkGroups0] / s11
v_rcp_f64 v[0:1], v[0:1]                           // s12 = s[sgprNumWorkGroups0] / s11
v_cvt_f64_u32 v[2:3], s[sgprNumWorkGroups0]        // s12 = s[sgprNumWorkGroups0] / s11
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s12 = s[sgprNumWorkGroups0] / s11
v_cvt_u32_f64 v0, v[0:1]                           // s12 = s[sgprNumWorkGroups0] / s11
v_mul_lo_u32 v1, v0, s11                           // s12 = s[sgprNumWorkGroups0] / s11
v_sub_nc_u32 v2, s[sgprNumWorkGroups0], v1         // s12 = s[sgprNumWorkGroups0] / s11
v_cmp_ge_u32 vcc_lo, v2, s11                       // s12 = s[sgprNumWorkGroups0] / s11
s_mov_b32 exec_lo, vcc_lo                          // s12 = s[sgprNumWorkGroups0] / s11
v_add_nc_u32 v0, v0, 1                             // s12 = s[sgprNumWorkGroups0] / s11
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s12, v0                        // quotient
s_mul_i32 s13, s11, s12                            // quotient * non-magic divisor
s_sub_u32 s13, s[sgprNumWorkGroups0], s13          // NumWorkGroups0=remainder
s_cmp_eq_u32 s13, 0                                // remainder == 0 ?
s_cmov_b32 s13, s11                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s14, s12                              // blockId >= numFullBlocks ?
s_cselect_b32 s12, s13, s11
v_cvt_f64_u32 v[0:1], s12                          // s[sgprWorkGroup1] = s15 / s12
v_rcp_f64 v[0:1], v[0:1]                           // s[sgprWorkGroup1] = s15 / s12
v_cvt_f64_u32 v[2:3], s15                          // s[sgprWorkGroup1] = s15 / s12
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s[sgprWorkGroup1] = s15 / s12
v_cvt_u32_f64 v0, v[0:1]                           // s[sgprWorkGroup1] = s15 / s12
v_mul_lo_u32 v1, v0, s12                           // s[sgprWorkGroup1] = s15 / s12
v_sub_nc_u32 v2, s15, v1                           // s[sgprWorkGroup1] = s15 / s12
v_cmp_ge_u32 vcc_lo, v2, s12                       // s[sgprWorkGroup1] = s15 / s12
s_mov_b32 exec_lo, vcc_lo                          // s[sgprWorkGroup1] = s15 / s12
v_add_nc_u32 v0, v0, 1                             // s[sgprWorkGroup1] = s15 / s12
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v1, v0, s12                           // s[sgprWorkGroup1] = s15 / s12
v_sub_nc_u32 v2, s15, v1                           // s[sgprWorkGroup1] = s15 / s12
v_readfirstlane_b32 s[sgprWorkGroup1], v0          // quotient
v_readfirstlane_b32 s[sgprWorkGroup0], v2          // remainder
s_mul_i32 s[sgprWorkGroup0], s[sgprWorkGroup1], s12 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup0], s15, s[sgprWorkGroup0] // WorkGroup0=remainder
s_mul_i32 s14, s14, s11                            // blockId * WGM
s_add_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s14 // wg1 += blockId * WGM
s_branch label_WGM
label_WGMPositive:
s_mov_b32 s11, s11                                 // WGM
v_cvt_f64_u32 v[0:1], s11                          // s14 = s[sgprWorkGroup1] / s11
v_rcp_f64 v[0:1], v[0:1]                           // s14 = s[sgprWorkGroup1] / s11
v_cvt_f64_u32 v[2:3], s[sgprWorkGroup1]            // s14 = s[sgprWorkGroup1] / s11
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s14 = s[sgprWorkGroup1] / s11
v_cvt_u32_f64 v0, v[0:1]                           // s14 = s[sgprWorkGroup1] / s11
v_mul_lo_u32 v1, v0, s11                           // s14 = s[sgprWorkGroup1] / s11
v_sub_nc_u32 v2, s[sgprWorkGroup1], v1             // s14 = s[sgprWorkGroup1] / s11
v_cmp_ge_u32 vcc_lo, v2, s11                       // s14 = s[sgprWorkGroup1] / s11
s_mov_b32 exec_lo, vcc_lo                          // s14 = s[sgprWorkGroup1] / s11
v_add_nc_u32 v0, v0, 1                             // s14 = s[sgprWorkGroup1] / s11
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s14, v0                        // quotient
s_mul_i32 s15, s14, s11                            // quotient * non-magic divisor
s_sub_u32 s15, s[sgprWorkGroup1], s15              // WorkGroup1=remainder
s_mul_i32 s15, s15, s[sgprNumWorkGroups0]          // (wg1 % WGM)*NumWorkGroups0
s_add_u32 s15, s15, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups0
v_cvt_f64_u32 v[0:1], s11                          // s12 = s[sgprNumWorkGroups1] / s11
v_rcp_f64 v[0:1], v[0:1]                           // s12 = s[sgprNumWorkGroups1] / s11
v_cvt_f64_u32 v[2:3], s[sgprNumWorkGroups1]        // s12 = s[sgprNumWorkGroups1] / s11
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s12 = s[sgprNumWorkGroups1] / s11
v_cvt_u32_f64 v0, v[0:1]                           // s12 = s[sgprNumWorkGroups1] / s11
v_mul_lo_u32 v1, v0, s11                           // s12 = s[sgprNumWorkGroups1] / s11
v_sub_nc_u32 v2, s[sgprNumWorkGroups1], v1         // s12 = s[sgprNumWorkGroups1] / s11
v_cmp_ge_u32 vcc_lo, v2, s11                       // s12 = s[sgprNumWorkGroups1] / s11
s_mov_b32 exec_lo, vcc_lo                          // s12 = s[sgprNumWorkGroups1] / s11
v_add_nc_u32 v0, v0, 1                             // s12 = s[sgprNumWorkGroups1] / s11
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s12, v0                        // quotient
s_mul_i32 s13, s11, s12                            // quotient * non-magic divisor
s_sub_u32 s13, s[sgprNumWorkGroups1], s13          // NumWorkGroups1=remainder
s_cmp_eq_u32 s13, 0                                // remainder == 0 ?
s_cmov_b32 s13, s11                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s14, s12                              // blockId >= numFullBlocks ?
s_cselect_b32 s12, s13, s11
v_cvt_f64_u32 v[0:1], s12                          // s[sgprWorkGroup0] = s15 / s12
v_rcp_f64 v[0:1], v[0:1]                           // s[sgprWorkGroup0] = s15 / s12
v_cvt_f64_u32 v[2:3], s15                          // s[sgprWorkGroup0] = s15 / s12
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s[sgprWorkGroup0] = s15 / s12
v_cvt_u32_f64 v0, v[0:1]                           // s[sgprWorkGroup0] = s15 / s12
v_mul_lo_u32 v1, v0, s12                           // s[sgprWorkGroup0] = s15 / s12
v_sub_nc_u32 v2, s15, v1                           // s[sgprWorkGroup0] = s15 / s12
v_cmp_ge_u32 vcc_lo, v2, s12                       // s[sgprWorkGroup0] = s15 / s12
s_mov_b32 exec_lo, vcc_lo                          // s[sgprWorkGroup0] = s15 / s12
v_add_nc_u32 v0, v0, 1                             // s[sgprWorkGroup0] = s15 / s12
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v1, v0, s12                           // s[sgprWorkGroup0] = s15 / s12
v_sub_nc_u32 v2, s15, v1                           // s[sgprWorkGroup0] = s15 / s12
v_readfirstlane_b32 s[sgprWorkGroup0], v0          // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v2          // remainder
s_mul_i32 s[sgprWorkGroup1], s[sgprWorkGroup0], s12 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup1], s15, s[sgprWorkGroup1] // WorkGroup1=remainder
s_mul_i32 s14, s14, s11                            // blockId * WGM
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s14 // wg1 += blockId * WGM
label_WGM:

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
v_and_b32 v4, 3, v4                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(4)
v_lshl_add_u32 v0, v4, 11, v0                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(2048); 7. final local read offset: flrOffset = lrOffset + WOffset
/* lr1J */
v_and_b32 v2, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v1, 7, v1                            // 1. N offset: nOffset = nIdx * nStride(128)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)

/* local read addresses: final offsets a */
v_lshrrev_b32 v2, 5, v[vgprSerial]                 // 2 = Serial / 32
v_lshrrev_b32 v2, 2, v2                            // LSU offset: Get LSU wave_id
s_mov_b32 s11, 128                                 // LSU offset: stride = lsuStride(128) when umlds==True
v_mul_lo_u32 v2, s11, v2                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_nc_u32 v[vgprLocalReadAddrA], v2, v0         // Final Offset: offset = (lro0+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrA], 1, v[vgprLocalReadAddrA] //  (multiple bpe)
v_lshrrev_b32 v3, 8, v[vgprLocalReadAddrA]         // Final Offset: padding 16 per block 256
v_lshl_add_u32 v[vgprLocalReadAddrA], v3, 4, v[vgprLocalReadAddrA] // Final Offset: padding 16 per block 256

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 5, v[vgprSerial]                 // 0 = Serial / 32
v_lshrrev_b32 v0, 2, v0                            // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(128) when umlds==True (dup assign opt.)
v_mul_lo_u32 v0, s11, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_nc_u32 v[vgprLocalReadAddrB], v0, v1         // Final Offset: offset = (lro1+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrB], 1, v[vgprLocalReadAddrB] //  (multiple bpe)
v_lshrrev_b32 v2, 8, v[vgprLocalReadAddrB]         // Final Offset: padding 16 per block 256
v_lshl_add_u32 v[vgprLocalReadAddrB], v2, 4, v[vgprLocalReadAddrB] // Final Offset: padding 16 per block 256

/* local read addresses: declare addresses a */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, 0x4400, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

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
/* LVCB = 32 */
/* v3 = B-unroll = serial%LVCB */
v_lshrrev_b32 v2, 5, v[vgprSerial]                 // 2 = Serial / 32
v_and_b32 v3, 31, v[vgprSerial]                    // 3 = Serial % 32
/* unroll *= glvw */
v_lshlrev_b32 v3, 2, v3                            // v3 = v3 * 4
v_mov_b32 v5, v3                                   // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v4 */
/* lwaUnrollAssignmentB = v5 */

/* local write addresses: first offset a */
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x80, v0     // lwAL**(DepthU_Compute + PAD)
v_add_nc_u32 v[vgprLocalWriteAddrA], v4, v[vgprLocalWriteAddrA] // lwFOA = (lwAA + lwAL*(DepthU+PAD))
v_lshlrev_b32 v[vgprLocalWriteAddrA], 1, v[vgprLocalWriteAddrA] //  (multiple bpe)
v_lshrrev_b32 v6, 8, v[vgprLocalWriteAddrA]        // padding 16 per block 256
v_lshl_add_u32 v[vgprLocalWriteAddrA], v6, 4, v[vgprLocalWriteAddrA] // padding 16 per block 256

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x80, v2     // lwBL**(DepthU_Compute + PAD)
v_add_nc_u32 v[vgprLocalWriteAddrB], v5, v[vgprLocalWriteAddrB] // lwFOB = (lwBB + lwBL*(DepthU+PAD))
v_lshlrev_b32 v[vgprLocalWriteAddrB], 1, v[vgprLocalWriteAddrB] //  (multiple bpe)
v_lshrrev_b32 v6, 8, v[vgprLocalWriteAddrB]        // padding 16 per block 256
v_lshl_add_u32 v[vgprLocalWriteAddrB], v6, 4, v[vgprLocalWriteAddrB] // padding 16 per block 256
v_add_co_u32 v[vgprLocalWriteAddrB], vcc_lo, 0x4400, v[vgprLocalWriteAddrB] // lwFOB = lw1J + lwL*MT1J + LDS_OFFSET_B=17408

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
v_add_co_u32 v7, vcc_lo, 8, v6                     // groA0I_1 += LSPA
v_add_co_u32 v8, vcc_lo, 8, v7                     // groA0I_2 += LSPA
v_add_co_u32 v9, vcc_lo, 8, v8                     // groA0I_3 += LSPA
v_add_co_u32 v10, vcc_lo, 8, v9                    // groA0I_4 += LSPA
v_add_co_u32 v11, vcc_lo, 8, v10                   // groA0I_5 += LSPA
v_add_co_u32 v12, vcc_lo, 8, v11                   // groA0I_6 += LSPA
v_add_co_u32 v13, vcc_lo, 8, v12                   // groA0I_7 += LSPA

/* global read addresses: tile offsets b */
v_mov_b32 v54, v2                                  // groB1J_0
v_add_co_u32 v55, vcc_lo, 4, v54                   // groB1J_1 += LSPB
v_add_co_u32 v56, vcc_lo, 4, v55                   // groB1J_2 += LSPB
v_add_co_u32 v57, vcc_lo, 4, v56                   // groB1J_3 += LSPB
v_add_co_u32 v58, vcc_lo, 4, v57                   // groB1J_4 += LSPB
v_add_co_u32 v59, vcc_lo, 4, v58                   // groB1J_5 += LSPB
v_add_co_u32 v60, vcc_lo, 4, v59                   // groB1J_6 += LSPB
v_add_co_u32 v61, vcc_lo, 4, v60                   // groB1J_7 += LSPB

/* global read addresses: unroll offsets a */
v_mov_b32 v14, v1                                  // groAL_0

/* global read addresses: unroll offsets b */
v_mov_b32 v15, v3                                  // groBL_0

/* global read addresses: addresses a */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s15, s[sgprWorkGroup0], 64            // WorkGroup[01] * MT
s_mul_i32 s14, s[sgprWorkGroup0], 64               // WorkGroup[01] * MT
s_mul_hi_u32 s15, s14, s[sgprStrideA0I]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s14, s14, s[sgprStrideA0I]               // tlu=0, scaled tile-offset by stride
s_mul_i32 s12, s[sgprStreamKLocalStart], 128       // StreamK tile start offset
s_mul_hi_u32 s13, s12, constStrideAL               // StreamK tile start offset
s_mul_i32 s12, s12, constStrideAL                  // StreamK tile start offset
s_add_u32 s14, s14, s12                            // accum GsuOffset term to tilestart
s_addc_u32 s15, s15, s13                           // accum GsuOffset term to tilestart
s_mov_b64 s[sgprShadowLimitA+0:sgprShadowLimitA+0+1], 1 // Init tensor size
s_sub_u32 s12, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s13, constStrideAL, s12               // stride x (size-1)
s_mul_i32 s12, constStrideAL, s12                  // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s12 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s13 // sum tensor size
s_sub_u32 s12, s[sgprSizeI], 1                     // (size-1)
s_mul_hi_u32 s13, s[sgprStrideA0I], s12            // stride x (size-1)
s_mul_i32 s12, s[sgprStrideA0I], s12               // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s12 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s13 // sum tensor size
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s14 // sub tileStart
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s15 // sub tileStart
s_lshr_b64 s[sgprShadowLimitA:sgprShadowLimitA+1], s[sgprShadowLimitA:sgprShadowLimitA+1], 1 // Set limit to use bytes (multiple bpe)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 4 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_and_b32 s11, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s11, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_StridedBatchedGemmLoadA
s_mul_i32 s12, 8, s[sgprWorkGroup2]                // Compute Offset into Pointer Array
s_cmp_eq_u32 s[sgprSizesSum], 0x0                  // Don't dereference Pointer array if SizesSum == 0
s_cbranch_scc1 label_StridedBatchedGemmLoadA_End
s_add_u32 s12, s12, s[sgprAddressA+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s13, s[sgprAddressA+1], 0               // Offsetting to the location [Higher half of address]
s_load_b64 s[sgprSrdA:sgprSrdA+1], s[12:13], 0     // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for pointer-array SRD load before reusing base SGPR
s_load_b64 s[12:13], s[sgprKernArgAddress:sgprKernArgAddress+1], 0xa0 // Load batchOffsetA from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s12        // Add batch offset to A address (low)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s13       // Add batch offset to A address (high)
s_sub_u32 s[sgprSrdA+0], s[sgprSrdA+0], 4          // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0         // pre-pad to make room for possible pointer shift
s_lshr_b64 s[14:15], s[14:15], 1                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdA+0], s14, s[sgprSrdA+0]        // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s15, s[sgprSrdA+1]       // SRD base = Address+ tileStart1
s_branch label_StridedBatchedGemmLoadA_End
label_StridedBatchedGemmLoadA:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s13, s[sgprStrideAK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s12, s[sgprStrideAK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s14, s14, s12                            // accum wg term to tilestart
s_addc_u32 s15, s15, s13                           // accum wg term to tilestart
s_lshr_b64 s[14:15], s[14:15], 1                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdA+0], s[sgprAddressA+0], s14    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprAddressA+1], s15   // SRD base = Address+ tileStart1
label_StridedBatchedGemmLoadA_End:  /// End Computing the Batch Matrix's base address for Strided Batched
s_mov_b32 s[sgprSrdA+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: block-scale A srd */
s_ff1_i32_b32 s[sgprScaleAGShift], s[sgprScaleAGMinus1]
s_lshr_b32 s[sgprScaleAKMask], s[sgprScaleAGMinus1], 7
s_max_u32 s[sgprScaleAKMask], s[sgprScaleAKMask], 1
s_sub_u32 s[sgprScaleAKMask], s[sgprScaleAKMask], 1
s_mov_b32 s[sgprScaleAInc], 128
s_lshr_b32 s[sgprScaleAInc], s[sgprScaleAInc], s[sgprScaleAGShift]
s_max_u32 s[sgprScaleAInc], s[sgprScaleAInc], 1
s_lshl_b32 s[sgprScaleAInc], s[sgprScaleAInc], 1
s_mov_b32 s[sgprScaleAKCnt], 0
s_add_u32 s[sgprStrideScaleA], s[sgprSizeL], s[sgprScaleAGMinus1] // SizeL + G
s_sub_u32 s[sgprStrideScaleA], s[sgprStrideScaleA], 1 // SizeL + G-1
s_lshr_b32 s[sgprStrideScaleA], s[sgprStrideScaleA], s[sgprScaleAGShift] // StrideScaleA = ceil(SizeL/32) scale elements
s_mul_i32 s12, s[sgprWorkGroup0], 64               // scaleA: workgroup row origin
s_mul_i32 s12, s12, s[sgprStrideScaleA]            // scaleA: * row stride
s_mul_i32 s12, s12, 2                              // scaleA: elements -> bytes
s_mul_i32 s13, s[sgprSizeI], s[sgprStrideScaleA]   // scaleA: SizeI * row stride
s_mul_i32 s13, s13, 2                              // scaleA: tensor bytes
s_sub_u32 s[sgprSrdScaleA+2], s13, s12             // scaleA: buffer limit from the workgroup origin
s_add_u32 s[sgprSrdScaleA+0], s[sgprAddressScaleA+0], s12 // scaleA: SRD base lo
s_addc_u32 s[sgprSrdScaleA+1], s[sgprAddressScaleA+1], 0 // scaleA: SRD base hi
s_mov_b32 s[sgprSrdScaleA+3], Srd127_96            // scaleA: set bits 127_96 in SRD
s_mov_b32 s[sgprScaleAPkMagic+0], 0xf0000f         // w4a16: nibble 2j at bit 0, nibble 2j+1 at bit 20
s_mov_b32 s[sgprScaleAPkMagic+1], 0xd480e408       // w4a16: -(1024 + 8) : -(64 + 8)
s_and_b32 s[sgprScaleAKCnt], s[sgprStreamKLocalStart], s[sgprScaleAKMask] // scaleA: position of the first iteration in its group
s_mul_i32 s12, s[sgprStreamKLocalStart], 128       // scaleA: first K of this range
s_lshr_b32 s12, s12, s[sgprScaleAGShift]           // scaleA: groups before this range
s_mul_i32 s13, s12, 2                              // scaleA: skipped bytes
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], s13 // scaleA: skip groups before the split (lower)
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA: skip groups before the split (upper)
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], s13 // scaleA: limit shrinks by the skip

/* global read addresses: addresses b */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s15, s[sgprWorkGroup1], 32            // WorkGroup[01] * MT
s_mul_i32 s14, s[sgprWorkGroup1], 32               // WorkGroup[01] * MT
s_mul_hi_u32 s15, s14, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s14, s14, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_mul_i32 s12, s[sgprStreamKLocalStart], 128       // StreamK tile start offset
s_mul_hi_u32 s13, s12, constStrideBL               // StreamK tile start offset
s_mul_i32 s12, s12, constStrideBL                  // StreamK tile start offset
s_add_u32 s14, s14, s12                            // accum GsuOffset term to tilestart
s_addc_u32 s15, s15, s13                           // accum GsuOffset term to tilestart
s_mov_b64 s[sgprShadowLimitB+0:sgprShadowLimitB+0+1], 1 // Init tensor size
s_sub_u32 s12, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s13, constStrideBL, s12               // stride x (size-1)
s_mul_i32 s12, constStrideBL, s12                  // stride x (size-1)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s12 // sum tensor size
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s13 // sum tensor size
s_sub_u32 s12, s[sgprSizeJ], 1                     // (size-1)
s_mul_hi_u32 s13, s[sgprStrideB1J], s12            // stride x (size-1)
s_mul_i32 s12, s[sgprStrideB1J], s12               // stride x (size-1)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s12 // sum tensor size
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s13 // sum tensor size
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s14 // sub tileStart
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s15 // sub tileStart
s_lshl_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], 1 // Set limit to use bytes (multiple bpe)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 8 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_and_b32 s11, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s11, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_StridedBatchedGemmLoadB
s_mul_i32 s12, 8, s[sgprWorkGroup2]                // Compute Offset into Pointer Array
s_cmp_eq_u32 s[sgprSizesSum], 0x0                  // Don't dereference Pointer array if SizesSum == 0
s_cbranch_scc1 label_StridedBatchedGemmLoadB_End
s_add_u32 s12, s12, s[sgprAddressB+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s13, s[sgprAddressB+1], 0               // Offsetting to the location [Higher half of address]
s_load_b64 s[sgprSrdB:sgprSrdB+1], s[12:13], 0     // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for pointer-array SRD load before reusing base SGPR
s_load_b64 s[12:13], s[sgprKernArgAddress:sgprKernArgAddress+1], 0xa8 // Load batchOffsetB from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s12        // Add batch offset to B address (low)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s13       // Add batch offset to B address (high)
s_sub_u32 s[sgprSrdB+0], s[sgprSrdB+0], 8          // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0         // pre-pad to make room for possible pointer shift
s_lshl_b64 s[14:15], s[14:15], 1                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdB+0], s14, s[sgprSrdB+0]        // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s15, s[sgprSrdB+1]       // SRD base = Address+ tileStart1
s_branch label_StridedBatchedGemmLoadB_End
label_StridedBatchedGemmLoadB:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s13, s[sgprStrideBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s12, s[sgprStrideBK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s14, s14, s12                            // accum wg term to tilestart
s_addc_u32 s15, s15, s13                           // accum wg term to tilestart
s_lshl_b64 s[14:15], s[14:15], 1                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdB+0], s[sgprAddressB+0], s14    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprAddressB+1], s15   // SRD base = Address+ tileStart1
label_StridedBatchedGemmLoadB_End:  /// End Computing the Batch Matrix's base address for Strided Batched
s_mov_b32 s[sgprSrdB+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: final offsets a */
/* ============================================================= */
v_mul_lo_u32 v62, s[sgprStrideA0I], v[6]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+0+0], vcc_lo, v[14], v[62+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+0+0], 0x8, v[vgprGlobalReadOffsetA+0+0] // add prepad for pointer shift
v_lshrrev_b32 v62, s[sgprScaleAGShift], v14        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+0], s[sgprStrideScaleA], v6 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+0], v62, v[vgprGlobalReadOffsetScaleA+0] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+0], 1, v[vgprGlobalReadOffsetScaleA+0] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetA+0], 1, v[vgprGlobalReadOffsetA+0] //  (multiple bpe)
v_mul_lo_u32 v62, s[sgprStrideA0I], v[7]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+1+0], vcc_lo, v[14], v[62+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+1+0], 0x8, v[vgprGlobalReadOffsetA+1+0] // add prepad for pointer shift
v_lshrrev_b32 v62, s[sgprScaleAGShift], v14        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+1], s[sgprStrideScaleA], v7 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+1], v62, v[vgprGlobalReadOffsetScaleA+1] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+1], 1, v[vgprGlobalReadOffsetScaleA+1] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetA+1], 1, v[vgprGlobalReadOffsetA+1] //  (multiple bpe)
v_mul_lo_u32 v62, s[sgprStrideA0I], v[8]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+2+0], vcc_lo, v[14], v[62+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+2+0], 0x8, v[vgprGlobalReadOffsetA+2+0] // add prepad for pointer shift
v_lshrrev_b32 v62, s[sgprScaleAGShift], v14        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+2], s[sgprStrideScaleA], v8 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+2], v62, v[vgprGlobalReadOffsetScaleA+2] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+2], 1, v[vgprGlobalReadOffsetScaleA+2] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetA+2], 1, v[vgprGlobalReadOffsetA+2] //  (multiple bpe)
v_mul_lo_u32 v62, s[sgprStrideA0I], v[9]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+3+0], vcc_lo, v[14], v[62+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+3+0], 0x8, v[vgprGlobalReadOffsetA+3+0] // add prepad for pointer shift
v_lshrrev_b32 v62, s[sgprScaleAGShift], v14        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+3], s[sgprStrideScaleA], v9 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+3], v62, v[vgprGlobalReadOffsetScaleA+3] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+3], 1, v[vgprGlobalReadOffsetScaleA+3] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetA+3], 1, v[vgprGlobalReadOffsetA+3] //  (multiple bpe)
v_mul_lo_u32 v62, s[sgprStrideA0I], v[10]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+4+0], vcc_lo, v[14], v[62+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+4+0], 0x8, v[vgprGlobalReadOffsetA+4+0] // add prepad for pointer shift
v_lshrrev_b32 v62, s[sgprScaleAGShift], v14        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+4], s[sgprStrideScaleA], v10 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+4], v62, v[vgprGlobalReadOffsetScaleA+4] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+4], 1, v[vgprGlobalReadOffsetScaleA+4] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetA+4], 1, v[vgprGlobalReadOffsetA+4] //  (multiple bpe)
v_mul_lo_u32 v62, s[sgprStrideA0I], v[11]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+5+0], vcc_lo, v[14], v[62+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+5+0], 0x8, v[vgprGlobalReadOffsetA+5+0] // add prepad for pointer shift
v_lshrrev_b32 v62, s[sgprScaleAGShift], v14        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+5], s[sgprStrideScaleA], v11 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+5], v62, v[vgprGlobalReadOffsetScaleA+5] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+5], 1, v[vgprGlobalReadOffsetScaleA+5] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetA+5], 1, v[vgprGlobalReadOffsetA+5] //  (multiple bpe)
v_mul_lo_u32 v62, s[sgprStrideA0I], v[12]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+6+0], vcc_lo, v[14], v[62+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+6+0], 0x8, v[vgprGlobalReadOffsetA+6+0] // add prepad for pointer shift
v_lshrrev_b32 v62, s[sgprScaleAGShift], v14        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+6], s[sgprStrideScaleA], v12 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+6], v62, v[vgprGlobalReadOffsetScaleA+6] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+6], 1, v[vgprGlobalReadOffsetScaleA+6] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetA+6], 1, v[vgprGlobalReadOffsetA+6] //  (multiple bpe)
v_mul_lo_u32 v62, s[sgprStrideA0I], v[13]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+7+0], vcc_lo, v[14], v[62+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+7+0], 0x8, v[vgprGlobalReadOffsetA+7+0] // add prepad for pointer shift
v_lshrrev_b32 v62, s[sgprScaleAGShift], v14        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+7], s[sgprStrideScaleA], v13 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+7], v62, v[vgprGlobalReadOffsetScaleA+7] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+7], 1, v[vgprGlobalReadOffsetScaleA+7] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetA+7], 1, v[vgprGlobalReadOffsetA+7] //  (multiple bpe)
/* ============================================================= */

/* global read addresses: final offsets b */
/* ============================================================= */
v_mul_lo_u32 v6, s[sgprStrideB1J], v[54]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+0+0], vcc_lo, v[15], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+0+0], 0x4, v[vgprGlobalReadOffsetB+0+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+0], 1, v[vgprGlobalReadOffsetB+0] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[55]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+1+0], vcc_lo, v[15], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+1+0], 0x4, v[vgprGlobalReadOffsetB+1+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+1], 1, v[vgprGlobalReadOffsetB+1] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[56]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+2+0], vcc_lo, v[15], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+2+0], 0x4, v[vgprGlobalReadOffsetB+2+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+2], 1, v[vgprGlobalReadOffsetB+2] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[57]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+3+0], vcc_lo, v[15], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+3+0], 0x4, v[vgprGlobalReadOffsetB+3+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+3], 1, v[vgprGlobalReadOffsetB+3] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[58]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+4+0], vcc_lo, v[15], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+4+0], 0x4, v[vgprGlobalReadOffsetB+4+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+4], 1, v[vgprGlobalReadOffsetB+4] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[59]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+5+0], vcc_lo, v[15], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+5+0], 0x4, v[vgprGlobalReadOffsetB+5+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+5], 1, v[vgprGlobalReadOffsetB+5] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[60]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+6+0], vcc_lo, v[15], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+6+0], 0x4, v[vgprGlobalReadOffsetB+6+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+6], 1, v[vgprGlobalReadOffsetB+6] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[61]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+7+0], vcc_lo, v[15], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+7+0], 0x4, v[vgprGlobalReadOffsetB+7+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+7], 1, v[vgprGlobalReadOffsetB+7] //  (multiple bpe)
/* ============================================================= */

/* global read addresses: increments a */
.set GlobalReadIncsA, 64

/* global read addresses: increments b */
.set GlobalReadIncsB, 256
/* declare loop num iterations */
s_sub_u32 s[sgprLoopCounterL], s[sgprStreamKLocalEnd], s[sgprStreamKLocalStart] // StreamK loop counter = localEnd - localStart
v_cmp_eq_f32 vcc_lo, s[sgprAlpha], 0.0             // s[Alpha] == 0.0f ?
s_cbranch_vccz label_SKAlphaCheck_1                // branch if s[Alpha] != 0
s_mov_b32 s[sgprLoopCounterL], 0                   // Skip iterations
label_SKAlphaCheck_1:
s_mov_b32 s[sgprOrigLoopCounter], s[sgprLoopCounterL] // copy loop counter
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
buffer_load_b32 v[vgprG2LA+18], v[vgprGlobalReadOffsetA+4], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_4_0
buffer_load_b32 v[vgprG2LA+22], v[vgprGlobalReadOffsetA+5], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_5_0
buffer_load_b32 v[vgprG2LA+26], v[vgprGlobalReadOffsetA+6], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_6_0
buffer_load_b32 v[vgprG2LA+30], v[vgprGlobalReadOffsetA+7], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_7_0

/* global read block-scale A */
buffer_load_d16_b16 v[vgprG2LScaleA+0], v[vgprGlobalReadOffsetScaleA+0], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 0
buffer_load_d16_b16 v[vgprG2LScaleA+1], v[vgprGlobalReadOffsetScaleA+1], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 1
buffer_load_d16_b16 v[vgprG2LScaleA+2], v[vgprGlobalReadOffsetScaleA+2], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 2
buffer_load_d16_b16 v[vgprG2LScaleA+3], v[vgprGlobalReadOffsetScaleA+3], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 3
buffer_load_d16_b16 v[vgprG2LScaleA+4], v[vgprGlobalReadOffsetScaleA+4], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 4
buffer_load_d16_b16 v[vgprG2LScaleA+5], v[vgprGlobalReadOffsetScaleA+5], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 5
buffer_load_d16_b16 v[vgprG2LScaleA+6], v[vgprGlobalReadOffsetScaleA+6], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 6
buffer_load_d16_b16 v[vgprG2LScaleA+7], v[vgprGlobalReadOffsetScaleA+7], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 7
buffer_load_b64 v[vgprG2LB+0:vgprG2LB+0+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b64 v[vgprG2LB+2:vgprG2LB+2+1], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b64 v[vgprG2LB+4:vgprG2LB+4+1], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b64 v[vgprG2LB+6:vgprG2LB+6+1], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0
buffer_load_b64 v[vgprG2LB+8:vgprG2LB+8+1], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_4_0
buffer_load_b64 v[vgprG2LB+10:vgprG2LB+10+1], v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_5_0
buffer_load_b64 v[vgprG2LB+12:vgprG2LB+12+1], v[vgprGlobalReadOffsetB+6], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_6_0
buffer_load_b64 v[vgprG2LB+14:vgprG2LB+14+1], v[vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_7_0

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], GlobalReadIncsA // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], GlobalReadIncsA // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

/* global read inc block-scale A (2 bytes every 1 iters) */
s_add_u32 s[sgprScaleAKCnt], s[sgprScaleAKCnt], 1  // scaleA: one more K iteration done
s_and_b32 s11, s[sgprScaleAKCnt], s[sgprScaleAKMask] // scaleA: SCC = counter has not wrapped
s_cselect_b32 s11, 0, s[sgprScaleAInc]             // scaleA: advance only on a wrap
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], s11 // scaleA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], s11 // scaleA limit -= inc

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], GlobalReadIncsB // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], GlobalReadIncsB // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32

/******************************************/
/* End setupNewTile                       */
/******************************************/
label_ShadowInitStart:
s_and_b32 s11, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s11, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_RegularSrdInitializationD
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_GeneralBatchedGemmSrdInitiationD // Parallel Reduction for General Batched GEMM, Srd initialized to workspace
label_RegularSrdInitializationD:  /// Regular SRD initialization for non-General Batched GEMM for D
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], s[sgprAddressD+0:sgprAddressD+0+1] // init SRD base address
s_branch label_GeneralBatchedGemmSrdInitiationD_End
label_GeneralBatchedGemmSrdInitiationD:  /// Handling General Batched GEMM SRD initialization
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], 0            // init SRD to 0
label_GeneralBatchedGemmSrdInitiationD_End:  /// End of handling General Batched GEMM SRD initialization
s_mov_b32 s[sgprSrdD+2], BufferOOB
s_mov_b32 s[sgprSrdD+3], Srd127_96                 // Set bits 127_96 in post-loop SRD

s_and_b32 s11, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s11, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_RegularSrdInitializationC
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_GeneralBatchedGemmSrdInitiationC // Parallel Reduction for General Batched GEMM, Srd initialized to workspace
label_RegularSrdInitializationC:  /// Regular SRD initialization for non-General Batched GEMM for C
s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], s[sgprAddressC+0:sgprAddressC+0+1] // init SRD base address
s_branch label_GeneralBatchedGemmSrdInitiationC_End
label_GeneralBatchedGemmSrdInitiationC:  /// Handling General Batched GEMM SRD initialization
s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], 0            // init SRD to 0
label_GeneralBatchedGemmSrdInitiationC_End:  /// End of handling General Batched GEMM SRD initialization
s_mov_b32 s[sgprSrdC+2], BufferOOB
s_mov_b32 s[sgprSrdC+3], Srd127_96                 // Set bits 127_96 in post-loop SRD

s_mov_b32 s11, 1
s_mov_b32 s52, 1
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_BPEDone                       // If synchronizer, use regular output BPE
s_cmp_eq_u32 s[sgprskTiles], 1                     // split == 1 ?
s_cbranch_scc1 label_BPEDone                       // If split == 1, use reguler output BPE
s_mov_b32 s11, 1
s_mov_b32 s52, 2
label_BPEDone:

s_mul_i32 s90, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_mul_hi_u32 s89, s90, s[sgprStrideC1J]            // ScaleC s90 by Stride
s_mul_i32 s88, s90, s[sgprStrideC1J]               // ScaleC s90 by Stride
s_lshl_b64 s[88:89], s[88:89], s11                 // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s88        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s89       // add hi to SRD
s_mul_hi_u32 s89, s90, s[sgprStrideD1J]            // ScaleD s90 by Stride
s_mul_i32 s88, s90, s[sgprStrideD1J]               // ScaleD s90 by Stride
s_lshl_b64 s[88:89], s[88:89], s52                 // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s88        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s89       // add hi to SRD

s_and_b32 s53, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s53, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_StridedBatchedGemmLoadC
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_GeneralBatchedGemmLoadC
label_StridedBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s89, s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride
s_mul_i32 s88, s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride
s_lshl_b64 s[88:89], s[88:89], s11                 // scale by bpe
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
s_load_b64 s[88:89], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x98 // Load batchOffsetC from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s88        // Add matrix address to SRD (low)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s89       // Add matrix address to SRD (high)
label_GeneralBatchedGemmLoadC_End:  /// End of label GeneralBatchedGemmLoadC
s_and_b32 s53, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s53, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_StridedBatchedGemmLoadD
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_GeneralBatchedGemmLoadD
label_StridedBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s89, s[sgprWorkGroup2], s[sgprStrideDK] // ScaleD s[sgprWorkGroup2] by Stride
s_mul_i32 s88, s[sgprWorkGroup2], s[sgprStrideDK]  // ScaleD s[sgprWorkGroup2] by Stride
s_lshl_b64 s[88:89], s[88:89], s52                 // scale by bpe
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
s_load_b64 s[88:89], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x90 // Load batchOffsetD from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s88        // Add matrix address to SRD (low)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s89       // Add matrix address to SRD (high)
label_GeneralBatchedGemmLoadD_End:  /// End of label GeneralBatchedGemmLoadD

s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_SK_SplitSrd                   // Skip this block if using single-kernel stream-k fixup
s_cmp_eq_u32 s[sgprskTiles], 1                     // split == 1 ?
s_cbranch_scc1 label_SK_SplitSrd                   // branch if split == 1
// Split Output Buffer offset: Free0 + (Free1-1)*StrideC1J + (Free2-1)*StrideCK * SplitIdx * bpe%s
s_mul_hi_u32 s89, s[sgprSizesFree+0], s[sgprSkPartialIdx] // Free0
s_mul_i32 s88, s[sgprSizesFree+0], s[sgprSkPartialIdx] // Free0
s_sub_u32 s90, s[sgprSizesFree+1], 1               // Free1
s_mul_i32 s90, s90, s[sgprSkPartialIdx]            // Free1
s_mul_hi_u32 s91, s90, s[sgprStrideC1J]            // Free1
s_mul_i32 s90, s90, s[sgprStrideC1J]               // Free1
s_add_u32 s88, s88, s90                            // Free1
s_addc_u32 s89, s89, s91                           // Free1
s_sub_u32 s90, s[sgprSizesFree+2], 1               // Free2
s_mul_i32 s90, s90, s[sgprSkPartialIdx]            // Free2
s_mul_hi_u32 s91, s90, s[sgprStrideCK]             // Free2
s_mul_i32 s90, s90, s[sgprStrideCK]                // Free2
s_add_u32 s88, s88, s90                            // Free2
s_addc_u32 s89, s89, s91                           // Free2
s_lshl_b64 s[88:89], s[88:89], 2                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s88        // add lo GSU offset to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s89       // add hi GSU offset to SRD
label_SK_SplitSrd:

/* initC: remove ValuC vgpr buffer [0...16) from pool */

/* initC: remove acc vgpr buffer [0...0) from pool */

/* initC: remove ValuA/B vgpr buffer [54...79) from pool */
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
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?

/* after InitC, skip to end of prefetch last iter if numIter==0 */

/* label_PrefetchGlobalLastIterEnd */
s_cbranch_scc0 label_NoBranch_6                    // Only branch on scc1
s_getpc_b64 s[52:53]                               // addr of next instr
s_add_i32 s54, label_PrefetchGlobalLastIterEnd, 4  // target branch offset
s_add_u32 s52, s52, s54                            // add target branch offset
s_addc_u32 s53, s53, 0                             // add high and carry
s_setpc_b64 s[52:53]                               // branch to label_PrefetchGlobalLastIterEnd
label_NoBranch_6:
s_waitcnt vmcnt(0)                                 // wait for global read
s_barrier                                          // For stream-k / persistent loop

/* local write a */
v_pack_b32_f16 v132, v[vgprG2LScaleA+0], v[vgprG2LScaleA+0] // scaleA: duplicate fp16 scale
v_xor_b32 v131, 0x88888888, v[vgprG2LA+2+0]        // w4a16: signed nibbles to biased unsigned

/* w4a16: save packed int4 dword 0 */
v_perm_b32 v135, v131, v131, 0xc000c00             // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v136, v131, v131, 0xc010c01             // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v137, v131, v131, 0xc020c02             // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v138, v131, v131, 0xc030c03             // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v135, v135, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v136, v136, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v137, v137, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v138, v138, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v135, v135, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v136, v136, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v137, v137, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v138, v138, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_mul_f16 v[vgprG2LA+0+0], v135, v132           // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_mul_f16 v[vgprG2LA+0+1], v136, v132           // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_mul_f16 v[vgprG2LA+0+2], v137, v132           // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_mul_f16 v[vgprG2LA+0+3], v138, v132           // w4a16: (q - z)*s, packed, fp16 pair 3
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
v_pack_b32_f16 v132, v[vgprG2LScaleA+1], v[vgprG2LScaleA+1] // scaleA: duplicate fp16 scale
v_xor_b32 v131, 0x88888888, v[vgprG2LA+6+0]        // w4a16: signed nibbles to biased unsigned

/* w4a16: save packed int4 dword 0 */
v_perm_b32 v135, v131, v131, 0xc000c00             // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v136, v131, v131, 0xc010c01             // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v137, v131, v131, 0xc020c02             // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v138, v131, v131, 0xc030c03             // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v135, v135, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v136, v136, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v137, v137, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v138, v138, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v135, v135, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v136, v136, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v137, v137, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v138, v138, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_mul_f16 v[vgprG2LA+4+0], v135, v132           // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_mul_f16 v[vgprG2LA+4+1], v136, v132           // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_mul_f16 v[vgprG2LA+4+2], v137, v132           // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_mul_f16 v[vgprG2LA+4+3], v138, v132           // w4a16: (q - z)*s, packed, fp16 pair 3
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:2176 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 2176 sync LDS0
v_pack_b32_f16 v132, v[vgprG2LScaleA+2], v[vgprG2LScaleA+2] // scaleA: duplicate fp16 scale
v_xor_b32 v131, 0x88888888, v[vgprG2LA+10+0]       // w4a16: signed nibbles to biased unsigned

/* w4a16: save packed int4 dword 0 */
v_perm_b32 v135, v131, v131, 0xc000c00             // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v136, v131, v131, 0xc010c01             // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v137, v131, v131, 0xc020c02             // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v138, v131, v131, 0xc030c03             // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v135, v135, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v136, v136, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v137, v137, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v138, v138, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v135, v135, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v136, v136, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v137, v137, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v138, v138, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_mul_f16 v[vgprG2LA+8+0], v135, v132           // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_mul_f16 v[vgprG2LA+8+1], v136, v132           // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_mul_f16 v[vgprG2LA+8+2], v137, v132           // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_mul_f16 v[vgprG2LA+8+3], v138, v132           // w4a16: (q - z)*s, packed, fp16 pair 3
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:4352 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 4352 sync LDS0
v_pack_b32_f16 v132, v[vgprG2LScaleA+3], v[vgprG2LScaleA+3] // scaleA: duplicate fp16 scale
v_xor_b32 v131, 0x88888888, v[vgprG2LA+14+0]       // w4a16: signed nibbles to biased unsigned

/* w4a16: save packed int4 dword 0 */
v_perm_b32 v135, v131, v131, 0xc000c00             // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v136, v131, v131, 0xc010c01             // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v137, v131, v131, 0xc020c02             // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v138, v131, v131, 0xc030c03             // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v135, v135, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v136, v136, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v137, v137, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v138, v138, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v135, v135, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v136, v136, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v137, v137, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v138, v138, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_mul_f16 v[vgprG2LA+12+0], v135, v132          // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_mul_f16 v[vgprG2LA+12+1], v136, v132          // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_mul_f16 v[vgprG2LA+12+2], v137, v132          // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_mul_f16 v[vgprG2LA+12+3], v138, v132          // w4a16: (q - z)*s, packed, fp16 pair 3
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+12:vgprG2LA+12+3] offset:6528 // lwoA_0_0_3_0 = (0*LSCA)*(MT0I+PAD) + (3*LSPA) = 6528 sync LDS0
v_pack_b32_f16 v132, v[vgprG2LScaleA+4], v[vgprG2LScaleA+4] // scaleA: duplicate fp16 scale
v_xor_b32 v131, 0x88888888, v[vgprG2LA+18+0]       // w4a16: signed nibbles to biased unsigned

/* w4a16: save packed int4 dword 0 */
v_perm_b32 v135, v131, v131, 0xc000c00             // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v136, v131, v131, 0xc010c01             // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v137, v131, v131, 0xc020c02             // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v138, v131, v131, 0xc030c03             // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v135, v135, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v136, v136, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v137, v137, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v138, v138, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v135, v135, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v136, v136, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v137, v137, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v138, v138, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_mul_f16 v[vgprG2LA+16+0], v135, v132          // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_mul_f16 v[vgprG2LA+16+1], v136, v132          // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_mul_f16 v[vgprG2LA+16+2], v137, v132          // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_mul_f16 v[vgprG2LA+16+3], v138, v132          // w4a16: (q - z)*s, packed, fp16 pair 3
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+16:vgprG2LA+16+3] offset:8704 // lwoA_0_0_4_0 = (0*LSCA)*(MT0I+PAD) + (4*LSPA) = 8704 sync LDS0
v_pack_b32_f16 v132, v[vgprG2LScaleA+5], v[vgprG2LScaleA+5] // scaleA: duplicate fp16 scale
v_xor_b32 v131, 0x88888888, v[vgprG2LA+22+0]       // w4a16: signed nibbles to biased unsigned

/* w4a16: save packed int4 dword 0 */
v_perm_b32 v135, v131, v131, 0xc000c00             // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v136, v131, v131, 0xc010c01             // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v137, v131, v131, 0xc020c02             // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v138, v131, v131, 0xc030c03             // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v135, v135, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v136, v136, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v137, v137, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v138, v138, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v135, v135, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v136, v136, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v137, v137, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v138, v138, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_mul_f16 v[vgprG2LA+20+0], v135, v132          // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_mul_f16 v[vgprG2LA+20+1], v136, v132          // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_mul_f16 v[vgprG2LA+20+2], v137, v132          // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_mul_f16 v[vgprG2LA+20+3], v138, v132          // w4a16: (q - z)*s, packed, fp16 pair 3
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+20:vgprG2LA+20+3] offset:10880 // lwoA_0_0_5_0 = (0*LSCA)*(MT0I+PAD) + (5*LSPA) = 10880 sync LDS0
v_pack_b32_f16 v132, v[vgprG2LScaleA+6], v[vgprG2LScaleA+6] // scaleA: duplicate fp16 scale
v_xor_b32 v131, 0x88888888, v[vgprG2LA+26+0]       // w4a16: signed nibbles to biased unsigned

/* w4a16: save packed int4 dword 0 */
v_perm_b32 v135, v131, v131, 0xc000c00             // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v136, v131, v131, 0xc010c01             // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v137, v131, v131, 0xc020c02             // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v138, v131, v131, 0xc030c03             // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v135, v135, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v136, v136, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v137, v137, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v138, v138, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v135, v135, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v136, v136, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v137, v137, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v138, v138, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_mul_f16 v[vgprG2LA+24+0], v135, v132          // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_mul_f16 v[vgprG2LA+24+1], v136, v132          // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_mul_f16 v[vgprG2LA+24+2], v137, v132          // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_mul_f16 v[vgprG2LA+24+3], v138, v132          // w4a16: (q - z)*s, packed, fp16 pair 3
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+24:vgprG2LA+24+3] offset:13056 // lwoA_0_0_6_0 = (0*LSCA)*(MT0I+PAD) + (6*LSPA) = 13056 sync LDS0
v_pack_b32_f16 v132, v[vgprG2LScaleA+7], v[vgprG2LScaleA+7] // scaleA: duplicate fp16 scale
v_xor_b32 v131, 0x88888888, v[vgprG2LA+30+0]       // w4a16: signed nibbles to biased unsigned

/* w4a16: save packed int4 dword 0 */
v_perm_b32 v135, v131, v131, 0xc000c00             // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v136, v131, v131, 0xc010c01             // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v137, v131, v131, 0xc020c02             // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v138, v131, v131, 0xc030c03             // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v135, v135, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v136, v136, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v137, v137, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v138, v138, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v135, v135, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v136, v136, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v137, v137, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v138, v138, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_mul_f16 v[vgprG2LA+28+0], v135, v132          // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_mul_f16 v[vgprG2LA+28+1], v136, v132          // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_mul_f16 v[vgprG2LA+28+2], v137, v132          // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_mul_f16 v[vgprG2LA+28+3], v138, v132          // w4a16: (q - z)*s, packed, fp16 pair 3
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+28:vgprG2LA+28+3] offset:15232 // lwoA_0_0_7_0 = (0*LSCA)*(MT0I+PAD) + (7*LSPA) = 15232 sync LDS0

/* local write b */
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+1] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+2:vgprG2LB+2+1] offset:1088 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 1088 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4:vgprG2LB+4+1] offset:2176 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 2176 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+6:vgprG2LB+6+1] offset:3264 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 3264 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8:vgprG2LB+8+1] offset:4352 // lwoB_0_0_4_0 = (0*LSCB)*(MT1J+PAD) + (4*LSPB) = 4352 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+10:vgprG2LB+10+1] offset:5440 // lwoB_0_0_5_0 = (0*LSCB)*(MT1J+PAD) + (5*LSPB) = 5440 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12:vgprG2LB+12+1] offset:6528 // lwoB_0_0_6_0 = (0*LSCB)*(MT1J+PAD) + (6*LSPB) = 6528 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+14:vgprG2LB+14+1] offset:7616 // lwoB_0_0_7_0 = (0*LSCB)*(MT1J+PAD) + (7*LSPB) = 7616 sync LDS0

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
/*  grEndMfmaIndex:15, lwStartMfmaIndex:15, lwEndMfmaIndex:15  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:14 */
/*  mfmaIndex:0  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4352 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4368 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b32 v[vgprG2LA+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:1  */
/* localReadsVacancy: latencyLeft 5 */
buffer_load_b32 v[vgprG2LA+6], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b32 v[vgprG2LA+10], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b32 v[vgprG2LA+14], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_3_0
buffer_load_b32 v[vgprG2LA+18], v[vgprGlobalReadOffsetA+4], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_4_0
buffer_load_b32 v[vgprG2LA+22], v[vgprGlobalReadOffsetA+5], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_5_0
buffer_load_b32 v[vgprG2LA+26], v[vgprGlobalReadOffsetA+6], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_6_0
buffer_load_b32 v[vgprG2LA+30], v[vgprGlobalReadOffsetA+7], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_7_0

/* global read block-scale A */
buffer_load_d16_b16 v[vgprG2LScaleA+0], v[vgprGlobalReadOffsetScaleA+0], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 0
buffer_load_d16_b16 v[vgprG2LScaleA+1], v[vgprGlobalReadOffsetScaleA+1], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 1
buffer_load_d16_b16 v[vgprG2LScaleA+2], v[vgprGlobalReadOffsetScaleA+2], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 2
buffer_load_d16_b16 v[vgprG2LScaleA+3], v[vgprGlobalReadOffsetScaleA+3], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 3
buffer_load_d16_b16 v[vgprG2LScaleA+4], v[vgprGlobalReadOffsetScaleA+4], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 4
buffer_load_d16_b16 v[vgprG2LScaleA+5], v[vgprGlobalReadOffsetScaleA+5], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 5
buffer_load_d16_b16 v[vgprG2LScaleA+6], v[vgprGlobalReadOffsetScaleA+6], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 6
buffer_load_d16_b16 v[vgprG2LScaleA+7], v[vgprGlobalReadOffsetScaleA+7], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 7
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=4 */

/* iter 1 */
/*  grEndMfmaIndex:15, lwStartMfmaIndex:15, lwEndMfmaIndex:15  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:14 */
/*  mfmaIndex:2  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4384 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4400 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b64 v[vgprG2LB+0:vgprG2LB+0+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:3  */
/* localReadsVacancy: latencyLeft 5 */
buffer_load_b64 v[vgprG2LB+2:vgprG2LB+2+1], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=0 readsPerIterB=4 */

/* iter 2 */
/*  grEndMfmaIndex:15, lwStartMfmaIndex:15, lwEndMfmaIndex:15  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:14 */
/*  mfmaIndex:4  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4416 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4432 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b64 v[vgprG2LB+4:vgprG2LB+4+1], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:5  */
/* localReadsVacancy: latencyLeft 5 */
buffer_load_b64 v[vgprG2LB+6:vgprG2LB+6+1], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=0 readsPerIterB=4 */

/* iter 3 */
/*  grEndMfmaIndex:15, lwStartMfmaIndex:15, lwEndMfmaIndex:15  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:14 */
/*  mfmaIndex:6  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4448 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4464 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b64 v[vgprG2LB+8:vgprG2LB+8+1], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_4_0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:7  */
/* localReadsVacancy: latencyLeft 5 */
buffer_load_b64 v[vgprG2LB+10:vgprG2LB+10+1], v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_5_0
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/* numPrefetchIter=0 */
/* dataAtIterA=3 numReadsIterA=4 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=3 numReadsIterB=4 skipReadsIterB=0 readsPerIterB=4 */

/* iter 4 */
/*  grEndMfmaIndex:15, lwStartMfmaIndex:15, lwEndMfmaIndex:15  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:14 */
/*  mfmaIndex:8  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:144 // L -> Reg lro=64 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:144 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4480 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4496 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b64 v[vgprG2LB+12:vgprG2LB+12+1], v[vgprGlobalReadOffsetB+6], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_6_0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:9  */
/* localReadsVacancy: latencyLeft 5 */
buffer_load_b64 v[vgprG2LB+14:vgprG2LB+14+1], v[vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_7_0
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/* numPrefetchIter=0 */
/* dataAtIterA=4 numReadsIterA=5 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=4 numReadsIterB=5 skipReadsIterB=0 readsPerIterB=4 */

/* iter 5 */
/*  grEndMfmaIndex:15, lwStartMfmaIndex:15, lwEndMfmaIndex:15  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:14 */
/*  mfmaIndex:10  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:160 // L -> Reg lro=80 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:176 // L -> Reg lro=80 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:160 // L -> Reg lro=80 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:176 // L -> Reg lro=80 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4512 // L -> Reg lro=80 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4528 // L -> Reg lro=80 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], GlobalReadIncsA // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], GlobalReadIncsA // limit -= inc)
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:11  */
/* localReadsVacancy: latencyLeft 5 */
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/* numPrefetchIter=0 */
/* dataAtIterA=5 numReadsIterA=6 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=5 numReadsIterB=6 skipReadsIterB=0 readsPerIterB=4 */

/* iter 6 */
/*  grEndMfmaIndex:15, lwStartMfmaIndex:15, lwEndMfmaIndex:15  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:14 */
/*  mfmaIndex:12  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:208 // L -> Reg lro=96 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:208 // L -> Reg lro=96 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4544 // L -> Reg lro=96 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4560 // L -> Reg lro=96 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* global read inc block-scale A (2 bytes every 1 iters) */
s_add_u32 s[sgprScaleAKCnt], s[sgprScaleAKCnt], 1  // scaleA: one more K iteration done
s_and_b32 s11, s[sgprScaleAKCnt], s[sgprScaleAKMask] // scaleA: SCC = counter has not wrapped
s_cselect_b32 s11, 0, s[sgprScaleAInc]             // scaleA: advance only on a wrap
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:13  */
/* localReadsVacancy: latencyLeft 5 */
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], s11 // scaleA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], s11 // scaleA limit -= inc
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/* numPrefetchIter=0 */
/* dataAtIterA=6 numReadsIterA=7 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=6 numReadsIterB=7 skipReadsIterB=0 readsPerIterB=4 */

/* iter 7 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:15, lwStartMfmaIndex:15, lwEndMfmaIndex:15  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:14 */
/*  mfmaIndex:14  */
/* schedule remaining localreads for one buffer scheduling */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:224 // L -> Reg lro=112 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:240 // L -> Reg lro=112 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:224 // L -> Reg lro=112 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:240 // L -> Reg lro=112 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4576 // L -> Reg lro=112 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4592 // L -> Reg lro=112 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], GlobalReadIncsB // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], GlobalReadIncsB // limit -= inc)
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:15  */
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32
/* sched write - iter 7 writesPerItem=1 */
s_waitcnt vmcnt(15)                                // wait for global read before writing to local
v_pack_b32_f16 v132, v[vgprG2LScaleA+0], v[vgprG2LScaleA+0] // scaleA: duplicate fp16 scale
v_xor_b32 v131, 0x88888888, v[vgprG2LA+2+0]        // w4a16: signed nibbles to biased unsigned

/* w4a16: save packed int4 dword 0 */
v_perm_b32 v135, v131, v131, 0xc000c00             // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v136, v131, v131, 0xc010c01             // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v137, v131, v131, 0xc020c02             // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v138, v131, v131, 0xc030c03             // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v135, v135, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v136, v136, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v137, v137, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v138, v138, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v135, v135, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v136, v136, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v137, v137, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v138, v138, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_mul_f16 v[vgprG2LA+0+0], v135, v132           // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_mul_f16 v[vgprG2LA+0+1], v136, v132           // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_mul_f16 v[vgprG2LA+0+2], v137, v132           // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_mul_f16 v[vgprG2LA+0+3], v138, v132           // w4a16: (q - z)*s, packed, fp16 pair 3
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
/* sched write - iter 7 writesPerItem=1 */
s_waitcnt vmcnt(14)                                // wait for global read before writing to local
v_pack_b32_f16 v132, v[vgprG2LScaleA+1], v[vgprG2LScaleA+1] // scaleA: duplicate fp16 scale
v_xor_b32 v131, 0x88888888, v[vgprG2LA+6+0]        // w4a16: signed nibbles to biased unsigned

/* w4a16: save packed int4 dword 0 */
v_perm_b32 v135, v131, v131, 0xc000c00             // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v136, v131, v131, 0xc010c01             // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v137, v131, v131, 0xc020c02             // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v138, v131, v131, 0xc030c03             // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v135, v135, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v136, v136, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v137, v137, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v138, v138, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v135, v135, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v136, v136, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v137, v137, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v138, v138, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_mul_f16 v[vgprG2LA+4+0], v135, v132           // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_mul_f16 v[vgprG2LA+4+1], v136, v132           // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_mul_f16 v[vgprG2LA+4+2], v137, v132           // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_mul_f16 v[vgprG2LA+4+3], v138, v132           // w4a16: (q - z)*s, packed, fp16 pair 3
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:2176 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 2176 sync LDS0
/* sched write - iter 7 writesPerItem=1 */
s_waitcnt vmcnt(13)                                // wait for global read before writing to local
v_pack_b32_f16 v132, v[vgprG2LScaleA+2], v[vgprG2LScaleA+2] // scaleA: duplicate fp16 scale
v_xor_b32 v131, 0x88888888, v[vgprG2LA+10+0]       // w4a16: signed nibbles to biased unsigned

/* w4a16: save packed int4 dword 0 */
v_perm_b32 v135, v131, v131, 0xc000c00             // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v136, v131, v131, 0xc010c01             // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v137, v131, v131, 0xc020c02             // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v138, v131, v131, 0xc030c03             // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v135, v135, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v136, v136, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v137, v137, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v138, v138, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v135, v135, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v136, v136, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v137, v137, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v138, v138, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_mul_f16 v[vgprG2LA+8+0], v135, v132           // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_mul_f16 v[vgprG2LA+8+1], v136, v132           // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_mul_f16 v[vgprG2LA+8+2], v137, v132           // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_mul_f16 v[vgprG2LA+8+3], v138, v132           // w4a16: (q - z)*s, packed, fp16 pair 3
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:4352 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 4352 sync LDS0
/* sched write - iter 7 writesPerItem=1 */
s_waitcnt vmcnt(12)                                // wait for global read before writing to local
v_pack_b32_f16 v132, v[vgprG2LScaleA+3], v[vgprG2LScaleA+3] // scaleA: duplicate fp16 scale
v_xor_b32 v131, 0x88888888, v[vgprG2LA+14+0]       // w4a16: signed nibbles to biased unsigned

/* w4a16: save packed int4 dword 0 */
v_perm_b32 v135, v131, v131, 0xc000c00             // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v136, v131, v131, 0xc010c01             // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v137, v131, v131, 0xc020c02             // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v138, v131, v131, 0xc030c03             // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v135, v135, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v136, v136, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v137, v137, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v138, v138, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v135, v135, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v136, v136, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v137, v137, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v138, v138, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_mul_f16 v[vgprG2LA+12+0], v135, v132          // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_mul_f16 v[vgprG2LA+12+1], v136, v132          // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_mul_f16 v[vgprG2LA+12+2], v137, v132          // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_mul_f16 v[vgprG2LA+12+3], v138, v132          // w4a16: (q - z)*s, packed, fp16 pair 3
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+12:vgprG2LA+12+3] offset:6528 // lwoA_0_0_3_0 = (0*LSCA)*(MT0I+PAD) + (3*LSPA) = 6528 sync LDS0
/* sched write - iter 7 writesPerItem=1 */
s_waitcnt vmcnt(11)                                // wait for global read before writing to local
v_pack_b32_f16 v132, v[vgprG2LScaleA+4], v[vgprG2LScaleA+4] // scaleA: duplicate fp16 scale
v_xor_b32 v131, 0x88888888, v[vgprG2LA+18+0]       // w4a16: signed nibbles to biased unsigned

/* w4a16: save packed int4 dword 0 */
v_perm_b32 v135, v131, v131, 0xc000c00             // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v136, v131, v131, 0xc010c01             // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v137, v131, v131, 0xc020c02             // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v138, v131, v131, 0xc030c03             // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v135, v135, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v136, v136, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v137, v137, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v138, v138, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v135, v135, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v136, v136, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v137, v137, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v138, v138, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_mul_f16 v[vgprG2LA+16+0], v135, v132          // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_mul_f16 v[vgprG2LA+16+1], v136, v132          // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_mul_f16 v[vgprG2LA+16+2], v137, v132          // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_mul_f16 v[vgprG2LA+16+3], v138, v132          // w4a16: (q - z)*s, packed, fp16 pair 3
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+16:vgprG2LA+16+3] offset:8704 // lwoA_0_0_4_0 = (0*LSCA)*(MT0I+PAD) + (4*LSPA) = 8704 sync LDS0
/* sched write - iter 7 writesPerItem=1 */
s_waitcnt vmcnt(10)                                // wait for global read before writing to local
v_pack_b32_f16 v132, v[vgprG2LScaleA+5], v[vgprG2LScaleA+5] // scaleA: duplicate fp16 scale
v_xor_b32 v131, 0x88888888, v[vgprG2LA+22+0]       // w4a16: signed nibbles to biased unsigned

/* w4a16: save packed int4 dword 0 */
v_perm_b32 v135, v131, v131, 0xc000c00             // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v136, v131, v131, 0xc010c01             // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v137, v131, v131, 0xc020c02             // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v138, v131, v131, 0xc030c03             // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v135, v135, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v136, v136, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v137, v137, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v138, v138, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v135, v135, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v136, v136, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v137, v137, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v138, v138, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_mul_f16 v[vgprG2LA+20+0], v135, v132          // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_mul_f16 v[vgprG2LA+20+1], v136, v132          // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_mul_f16 v[vgprG2LA+20+2], v137, v132          // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_mul_f16 v[vgprG2LA+20+3], v138, v132          // w4a16: (q - z)*s, packed, fp16 pair 3
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+20:vgprG2LA+20+3] offset:10880 // lwoA_0_0_5_0 = (0*LSCA)*(MT0I+PAD) + (5*LSPA) = 10880 sync LDS0
/* sched write - iter 7 writesPerItem=1 */
s_waitcnt vmcnt(9)                                 // wait for global read before writing to local
v_pack_b32_f16 v132, v[vgprG2LScaleA+6], v[vgprG2LScaleA+6] // scaleA: duplicate fp16 scale
v_xor_b32 v131, 0x88888888, v[vgprG2LA+26+0]       // w4a16: signed nibbles to biased unsigned

/* w4a16: save packed int4 dword 0 */
v_perm_b32 v135, v131, v131, 0xc000c00             // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v136, v131, v131, 0xc010c01             // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v137, v131, v131, 0xc020c02             // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v138, v131, v131, 0xc030c03             // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v135, v135, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v136, v136, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v137, v137, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v138, v138, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v135, v135, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v136, v136, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v137, v137, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v138, v138, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_mul_f16 v[vgprG2LA+24+0], v135, v132          // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_mul_f16 v[vgprG2LA+24+1], v136, v132          // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_mul_f16 v[vgprG2LA+24+2], v137, v132          // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_mul_f16 v[vgprG2LA+24+3], v138, v132          // w4a16: (q - z)*s, packed, fp16 pair 3
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+24:vgprG2LA+24+3] offset:13056 // lwoA_0_0_6_0 = (0*LSCA)*(MT0I+PAD) + (6*LSPA) = 13056 sync LDS0
/* sched write - iter 7 writesPerItem=1 */
s_waitcnt vmcnt(8)                                 // wait for global read before writing to local
v_pack_b32_f16 v132, v[vgprG2LScaleA+7], v[vgprG2LScaleA+7] // scaleA: duplicate fp16 scale
v_xor_b32 v131, 0x88888888, v[vgprG2LA+30+0]       // w4a16: signed nibbles to biased unsigned

/* w4a16: save packed int4 dword 0 */
v_perm_b32 v135, v131, v131, 0xc000c00             // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v136, v131, v131, 0xc010c01             // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v137, v131, v131, 0xc020c02             // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v138, v131, v131, 0xc030c03             // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v135, v135, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v136, v136, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v137, v137, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v138, v138, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v135, v135, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v136, v136, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v137, v137, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v138, v138, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_mul_f16 v[vgprG2LA+28+0], v135, v132          // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_mul_f16 v[vgprG2LA+28+1], v136, v132          // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_mul_f16 v[vgprG2LA+28+2], v137, v132          // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_mul_f16 v[vgprG2LA+28+3], v138, v132          // w4a16: (q - z)*s, packed, fp16 pair 3
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+28:vgprG2LA+28+3] offset:15232 // lwoA_0_0_7_0 = (0*LSCA)*(MT0I+PAD) + (7*LSPA) = 15232 sync LDS0
/* sched write - iter 7 writesPerItem=1 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+1] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0
/* sched write - iter 7 writesPerItem=1 */
s_waitcnt vmcnt(6)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+2:vgprG2LB+2+1] offset:1088 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 1088 sync LDS0
/* sched write - iter 7 writesPerItem=1 */
s_waitcnt vmcnt(5)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4:vgprG2LB+4+1] offset:2176 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 2176 sync LDS0
/* sched write - iter 7 writesPerItem=1 */
s_waitcnt vmcnt(4)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+6:vgprG2LB+6+1] offset:3264 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 3264 sync LDS0
/* sched write - iter 7 writesPerItem=1 */
s_waitcnt vmcnt(3)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8:vgprG2LB+8+1] offset:4352 // lwoB_0_0_4_0 = (0*LSCB)*(MT1J+PAD) + (4*LSPB) = 4352 sync LDS0
/* sched write - iter 7 writesPerItem=1 */
s_waitcnt vmcnt(2)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+10:vgprG2LB+10+1] offset:5440 // lwoB_0_0_5_0 = (0*LSCB)*(MT1J+PAD) + (5*LSPB) = 5440 sync LDS0
/* sched write - iter 7 writesPerItem=1 */
s_waitcnt vmcnt(1)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12:vgprG2LB+12+1] offset:6528 // lwoB_0_0_6_0 = (0*LSCB)*(MT1J+PAD) + (6*LSPB) = 6528 sync LDS0
/* sched write - iter 7 writesPerItem=1 */
s_waitcnt vmcnt(0)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+14:vgprG2LB+14+1] offset:7616 // lwoB_0_0_7_0 = (0*LSCB)*(MT1J+PAD) + (7*LSPB) = 7616 sync LDS0

/* local write swap offsets a */

/* local write swap offsets b */

/* local read swap offsets a */

/* local read swap offsets b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/* numPrefetchIter=0 */
/* dataAtIterA=7 numReadsIterA=8 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=7 numReadsIterB=8 skipReadsIterB=0 readsPerIterB=4 */

/******************************************/
/* Unrolled Loop - End                    */
/******************************************/

/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x1              // counterL==1
s_cbranch_scc0 label_LoopBeginL                    // restart LoopL
label_LoopEndL:

/* Before NLL: Check VGPR.checkin for INT8 LW */

/******************************************/
/* Ord. NoLoadLoop - Begin                */
/******************************************/
s_waitcnt lgkmcnt(0)                               // 4wait for local write
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier                                          // wait for local write done, sync

/* iter 0 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:15, lwEndMfmaIndex:15  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:14 */
/*  mfmaIndex:0  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4352 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4368 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:1  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=4 */

/* iter 1 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:15, lwEndMfmaIndex:15  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:14 */
/*  mfmaIndex:2  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4384 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4400 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:3  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=0 readsPerIterB=4 */

/* iter 2 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:15, lwEndMfmaIndex:15  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:14 */
/*  mfmaIndex:4  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4416 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4432 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:5  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=0 readsPerIterB=4 */

/* iter 3 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:15, lwEndMfmaIndex:15  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:14 */
/*  mfmaIndex:6  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4448 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4464 // L -> Reg lro=48 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:7  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/* numPrefetchIter=0 */
/* dataAtIterA=3 numReadsIterA=4 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=3 numReadsIterB=4 skipReadsIterB=0 readsPerIterB=4 */

/* iter 4 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:15, lwEndMfmaIndex:15  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:14 */
/*  mfmaIndex:8  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:144 // L -> Reg lro=64 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:144 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4480 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4496 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:9  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/* numPrefetchIter=0 */
/* dataAtIterA=4 numReadsIterA=5 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=4 numReadsIterB=5 skipReadsIterB=0 readsPerIterB=4 */

/* iter 5 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:15, lwEndMfmaIndex:15  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:14 */
/*  mfmaIndex:10  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:160 // L -> Reg lro=80 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:176 // L -> Reg lro=80 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:160 // L -> Reg lro=80 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:176 // L -> Reg lro=80 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4512 // L -> Reg lro=80 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4528 // L -> Reg lro=80 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:11  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/* numPrefetchIter=0 */
/* dataAtIterA=5 numReadsIterA=6 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=5 numReadsIterB=6 skipReadsIterB=0 readsPerIterB=4 */

/* iter 6 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:15, lwEndMfmaIndex:15  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:14 */
/*  mfmaIndex:12  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:208 // L -> Reg lro=96 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:208 // L -> Reg lro=96 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4544 // L -> Reg lro=96 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4560 // L -> Reg lro=96 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:13  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/* numPrefetchIter=0 */
/* dataAtIterA=6 numReadsIterA=7 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=6 numReadsIterB=7 skipReadsIterB=0 readsPerIterB=4 */

/* iter 7 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:15, lwEndMfmaIndex:15  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:14 */
/*  mfmaIndex:14  */
/* schedule remaining localreads for one buffer scheduling */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:224 // L -> Reg lro=112 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:240 // L -> Reg lro=112 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:224 // L -> Reg lro=112 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:240 // L -> Reg lro=112 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4576 // L -> Reg lro=112 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4592 // L -> Reg lro=112 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:15  */
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/* numPrefetchIter=0 */
/* dataAtIterA=7 numReadsIterA=8 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=7 numReadsIterB=8 skipReadsIterB=0 readsPerIterB=4 */
label_toPGR1end_OrdNLL:
label_PrefetchGlobalLastIterEnd:

/* Tail: add ValuA/B vgpr buffer [54...79) to pool */

/* Tail: add address/G2L vgpr [79...128) to pool */
label_Summation_End_8:
.set sgprLoopCounterL, UNDEF
.set sgprOrigLoopCounter, UNDEF
.set sgprStrideScaleA, UNDEF
.set sgprSrdScaleA, UNDEF
.set sgprScaleAGShift, UNDEF
.set sgprScaleAKMask, UNDEF
.set sgprScaleAInc, UNDEF
.set sgprScaleAKCnt, UNDEF
.set sgprScaleAPkMagic, UNDEF
.set sgprSrdA, UNDEF
.set sgprSrdB, UNDEF
.set sgprShadowLimitA, UNDEF
.set sgprShadowLimitB, UNDEF
/* load store sgprs */

/* Mapping of Acc register -> C Vgpr register */

/* Multiply MI out register with Alpha -> C Vgpr register */

/* not-LocalSplitU: global write indices */
/* computeStoreVgprs */
v_lshrrev_b32 v58, 5, v[vgprSerial]                // 58 = Serial / 32
v_lshrrev_b32 v59, 2, v58                          // 59 = 58 / 4
v_mul_lo_u32 v59, 0x10, v59                        // wave coordination offset 1
v_and_b32 v55, 31, v[vgprSerial]                   // v55 = v[vgprSerial] % 32
v_lshrrev_b32 v55, 4, v55                          // 55 = 55 / 16
                                                   // thread0 * continuous_output (multiplier is 1, do nothing)
v_add_lshl_u32 v55, v59, v55, 0                    // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v56, v55, s[sgprStrideC1J]            //  offset 1
v_mul_lo_u32 v57, v55, s[sgprStrideD1J]            //  offset 1
v_and_b32 v54, 3, v58                              // v54 = v58 % 4
v_mul_lo_u32 v54, 0x10, v54                        // wave coordination offset 0
v_and_b32 v59, 15, v[vgprSerial]                   // v59 = v[vgprSerial] % 16
v_add_lshl_u32 v54, v59, v54, 0                    // coordination 0 = vwA * (wave_id0 + tid0)
s_mul_i32 s7, 64, s[sgprWorkGroup0]                // wgp0 * MT0
v_add_nc_u32 v54, s7, v54                          // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s7, 32, s[sgprWorkGroup1]                // wgp1 * MT1
v_add_nc_u32 v55, s7, v55                          // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1

/* not-LocalSplitU: global write */

/******************************************/
/* Global Write Elements                  */
/******************************************/
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_GSU                           // Branch to stream-k store code
s_cmp_eq_u32 s[sgprskTiles], 1                     // split == 1 ?
s_cbranch_scc1 label_GSU                           // branch if split == 1
label_GW_B0_MB:
label_GW_B0_FD0_MB:

/* Edge/NonEdge store path check (M): Size % 64 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s64, 63, s[sgprSizeI]                    // s64 = s[sgprSizeI] % 64
s_add_u32 s65, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s65                // wg0 >= nwg0-1 ?
s_cselect_b32 s64, s64, 0                          // set rem
s_cmpk_gt_u32 s64, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_MB_Else         // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 32 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s64, 31, s[sgprSizeJ]                    // s64 = s[sgprSizeJ] % 32
s_add_u32 s65, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s65                // wg1 >= nwg1-1
s_cselect_b32 s64, s64, 0                          // set rem
s_cmpk_gt_u32 s64, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_MB_Then         // jump if edges required
label_GW_B0_FD0_VW1_MB_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=100 */
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
v_add_lshl_u32 v65, v57, v54, 2                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=54, coord0Vgpr=54 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0), (3, 0, 0, 0), (4, 0, 0, 0), (5, 0, 0, 0), (6, 0, 0, 0), (7, 0, 0, 0), (8, 0, 0, 0), (9, 0, 0, 0), (10, 0, 0, 0), (11, 0, 0, 0), (12, 0, 0, 0), (13, 0, 0, 0), (14, 0, 0, 0), (15, 0, 0, 0)] */
v_mov_b32 v[vgprValuC+67], v[vgprValuC+0]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+68], v[vgprValuC+1]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+69], v[vgprValuC+2]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+70], v[vgprValuC+3]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+71], v[vgprValuC+4]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+72], v[vgprValuC+5]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+73], v[vgprValuC+6]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+74], v[vgprValuC+7]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+75], v[vgprValuC+8]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+76], v[vgprValuC+9]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+77], v[vgprValuC+10]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+78], v[vgprValuC+11]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+79], v[vgprValuC+12]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+80], v[vgprValuC+13]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+81], v[vgprValuC+14]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+82], v[vgprValuC+15]         // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b32 v67, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v68, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v69, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v70, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v71, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v72, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v73, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v74, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v75, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v76, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v77, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v78, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v79, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v80, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v81, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v82, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_B0_FD0_VW1_MB_NonEdgeEnd:
label_GW_B0_FD0_VW1_MB_Else:
label_GW_B0_FD0_VW1_MB_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=50 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (1,0,0,0:vw1); (2,0,0,0:vw1); (3,0,0,0:vw1); (4,0,0,0:vw1); (5,0,0,0:vw1); (6,0,0,0:vw1); (7,0,0,0:vw1); (8,0,0,0:vw1); (9,0,0,0:vw1); (10,0,0,0:vw1); (11,0,0,0:vw1); (12,0,0,0:vw1); (13,0,0,0:vw1); (14,0,0,0:vw1); (15,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v60, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v81, v57, v54, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v81, v60, v81, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v82, v57, v54, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v82, v60, v82, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v83, v57, v54, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v83, v60, v83, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v84, v57, v54, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v84, v60, v84, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v85, v57, v54, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v85, v60, v85, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v86, v57, v54, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v86, v60, v86, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v87, v57, v54, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v87, v60, v87, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v88, v57, v54, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v88, v60, v88, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v89, v57, v54, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v89, v60, v89, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v90, v57, v54, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v90, v60, v90, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v91, v57, v54, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v91, v60, v91, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v92, v57, v54, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v92, v60, v92, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v93, v57, v54, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v93, v60, v93, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v94, v57, v54, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v94, v60, v94, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v95, v57, v54, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v95, v60, v95, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v96, v57, v54, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v96, v60, v96, s66                   // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0), (3, 0, 0, 0), (4, 0, 0, 0), (5, 0, 0, 0), (6, 0, 0, 0), (7, 0, 0, 0), (8, 0, 0, 0), (9, 0, 0, 0), (10, 0, 0, 0), (11, 0, 0, 0), (12, 0, 0, 0), (13, 0, 0, 0), (14, 0, 0, 0), (15, 0, 0, 0)] */
v_mov_b32 v[vgprValuC+65], v[vgprValuC+0]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+66], v[vgprValuC+1]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+67], v[vgprValuC+2]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+68], v[vgprValuC+3]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+69], v[vgprValuC+4]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+70], v[vgprValuC+5]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+71], v[vgprValuC+6]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+72], v[vgprValuC+7]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+73], v[vgprValuC+8]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+74], v[vgprValuC+9]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+75], v[vgprValuC+10]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+76], v[vgprValuC+11]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+77], v[vgprValuC+12]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+78], v[vgprValuC+13]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+79], v[vgprValuC+14]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+80], v[vgprValuC+15]         // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b32 v65, v81, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v66, v82, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v67, v83, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v68, v84, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v69, v85, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v70, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v71, v87, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v72, v88, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v73, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v74, v90, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v75, v91, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v76, v92, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v77, v93, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v78, v94, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v79, v95, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v80, v96, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_End:
s_getpc_b64 s[64:65]                               // addr of next instr
s_add_i32 s66, label_KernelEnd, 4                  // target branch offset
s_add_u32 s64, s64, s66                            // add target branch offset
s_addc_u32 s65, s65, 0                             // add high and carry
s_setpc_b64 s[64:65]                               // branch to label_KernelEnd
label_GSU:
s_cmp_eq_u32 s[sgprStreamKLocalStart], 0           // does wg start tile?
s_cbranch_scc1 label_NoBranch_9                    // Only branch on scc0
s_getpc_b64 s[70:71]                               // addr of next instr
s_add_i32 s72, label_SK_Partials_1, 4              // target branch offset
s_add_u32 s70, s70, s72                            // add target branch offset
s_addc_u32 s71, s71, 0                             // add high and carry
s_setpc_b64 s[70:71]                               // branch to label_SK_Partials_1
label_NoBranch_9:
s_cmp_eq_u32 s[sgprStreamKLocalEnd], s[sgprItersPerTile] // does wg finish tile?
s_cbranch_scc1 label_SK_Store                      // Branch if started and finished tile, go to regular store code
s_add_u32 s7, s[sgprPersistentWorkGroupIndex], 1   // input partial tile index
s_mul_hi_u32 s65, s[sgprPersistentIterationEnd], s[sgprMagicNumberItersPerTile] // s_magic mul, div alg 2
s_lshr_b32 s66, s[sgprMagicShiftItersPerTile], 31  // tmpS = extract abit
s_mul_i32 s64, s[sgprPersistentIterationEnd], s66  // s_magic mul, div alg 2
s_add_u32 s64, s64, s65
s_and_b32 s66, s[sgprMagicShiftItersPerTile], 2147483647 // tmpS = remove abit to final shift
s_lshr_b32 s64, s64, s66                           // sMagicDiv Alg 2
s_mul_i32 s64, s64, s[sgprItersPerTile]            // start iteration of partial tile
s_sub_u32 s8, s[sgprPersistentIterationEnd], s64   // calc iterations completed by this WG
label_SK_Fixup:
s_lshl_b32 s64, s7, 2                              // flag offset based on physical partial slot
s_load_b32 s66, s[sgprAddressFlags:sgprAddressFlags+1], s64 glc dlc // get flag
s_waitcnt lgkmcnt(0)                               // wait for flag load
s_cmp_eq_u32 s66, 1                                // check if ready
s_cbranch_scc0 label_SK_Fixup                      // if flag not set, wait and check again
s_barrier                                          // wait for all workgroups before resetting flag
v_readfirstlane_b32 s66, v[vgprSerial]             // Wave 0 updates flags
s_cmp_eq_u32 s66, 0                                // Check for wave 0
s_cbranch_scc0 label_SK_SkipFlagReset              // Skip flag reset
v_mov_b32 v65, s66                                 // move flag value to vgpr
v_mov_b32 v66, 0                                   // zero vaddr offset
s_mov_b64 s[72:73], s[sgprAddressFlags:sgprAddressFlags+1]
s_mov_b32 s74, BufferOOB
s_mov_b32 s75, Srd127_96
buffer_store_b32 v65, v66, s[72:75], s64 offen offset:0 glc dlc // reset flag
s_waitcnt_vscnt null 0                             // wait for data store
label_SK_SkipFlagReset:
label_Fixup_E0:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=48 */
s_mov_b64 s[sgprSrdWS:sgprSrdWS+1], s[sgprAddressWS:sgprAddressWS+1]
s_mov_b32 s[sgprSrdWS+2], BufferOOB
s_mov_b32 s[sgprSrdWS+3], Srd127_96

s_mul_i32 s64, 0x2000, s7                          // Offset to correct partials tile (low word)
s_mul_hi_u32 s11, 0x2000, s7                       // partials tile offset (high word) for 64-bit SRD
s_add_u32 s[sgprSrdWS+0], s[sgprSrdWS+0], s64      // add lo to SRD
s_addc_u32 s[sgprSrdWS+1], s[sgprSrdWS+1], s11     // add hi (offset high word + lo carry) to SRD
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Fixup Batch #0 (d1,d0,vc1,vc0) =       */
/*      (0,0,0,0:vw1); (1,0,0,0:vw1); (2,0,0,0:vw1); (3,0,0,0:vw1); (4,0,0,0:vw1); (5,0,0,0:vw1); (6,0,0,0:vw1); (7,0,0,0:vw1); (8,0,0,0:vw1); (9,0,0,0:vw1); (10,0,0,0:vw1); (11,0,0,0:vw1); (12,0,0,0:vw1); (13,0,0,0:vw1); (14,0,0,0:vw1); (15,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_lshlrev_b32 v66, 2, v[vgprSerial]                // v66 = v[vgprSerial] * 4
s_mov_b32 s64, 0                                   // Init sgpr offset
buffer_load_b32 v83, v66, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v84, v66, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v85, v66, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v86, v66, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v87, v66, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v88, v66, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v89, v66, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v90, v66, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v91, v66, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v92, v66, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v93, v66, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v94, v66, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v95, v66, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v96, v66, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v97, v66, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v98, v66, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
v_mov_b32 v[vgprValuC+67], v[vgprValuC+0]          // copy MI out reg to vreg[0]
v_mov_b32 v[vgprValuC+68], v[vgprValuC+1]          // copy MI out reg to vreg[1]
v_mov_b32 v[vgprValuC+69], v[vgprValuC+2]          // copy MI out reg to vreg[2]
v_mov_b32 v[vgprValuC+70], v[vgprValuC+3]          // copy MI out reg to vreg[3]
v_mov_b32 v[vgprValuC+71], v[vgprValuC+4]          // copy MI out reg to vreg[4]
v_mov_b32 v[vgprValuC+72], v[vgprValuC+5]          // copy MI out reg to vreg[5]
v_mov_b32 v[vgprValuC+73], v[vgprValuC+6]          // copy MI out reg to vreg[6]
v_mov_b32 v[vgprValuC+74], v[vgprValuC+7]          // copy MI out reg to vreg[7]
v_mov_b32 v[vgprValuC+75], v[vgprValuC+8]          // copy MI out reg to vreg[8]
v_mov_b32 v[vgprValuC+76], v[vgprValuC+9]          // copy MI out reg to vreg[9]
v_mov_b32 v[vgprValuC+77], v[vgprValuC+10]         // copy MI out reg to vreg[10]
v_mov_b32 v[vgprValuC+78], v[vgprValuC+11]         // copy MI out reg to vreg[11]
v_mov_b32 v[vgprValuC+79], v[vgprValuC+12]         // copy MI out reg to vreg[12]
v_mov_b32 v[vgprValuC+80], v[vgprValuC+13]         // copy MI out reg to vreg[13]
v_mov_b32 v[vgprValuC+81], v[vgprValuC+14]         // copy MI out reg to vreg[14]
v_mov_b32 v[vgprValuC+82], v[vgprValuC+15]         // copy MI out reg to vreg[15]

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 0 - 1
v_add_f32 v[vgprValuC+67], v[vgprValuC+67], v83    // accum partials

s_waitcnt vmcnt(14)                                // wait C (interleaved) 14 = 16 - 1 - 1
v_add_f32 v[vgprValuC+68], v[vgprValuC+68], v84    // accum partials

s_waitcnt vmcnt(13)                                // wait C (interleaved) 13 = 16 - 2 - 1
v_add_f32 v[vgprValuC+69], v[vgprValuC+69], v85    // accum partials

s_waitcnt vmcnt(12)                                // wait C (interleaved) 12 = 16 - 3 - 1
v_add_f32 v[vgprValuC+70], v[vgprValuC+70], v86    // accum partials

s_waitcnt vmcnt(11)                                // wait C (interleaved) 11 = 16 - 4 - 1
v_add_f32 v[vgprValuC+71], v[vgprValuC+71], v87    // accum partials

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 16 - 5 - 1
v_add_f32 v[vgprValuC+72], v[vgprValuC+72], v88    // accum partials

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 16 - 6 - 1
v_add_f32 v[vgprValuC+73], v[vgprValuC+73], v89    // accum partials

s_waitcnt vmcnt(8)                                 // wait C (interleaved) 8 = 16 - 7 - 1
v_add_f32 v[vgprValuC+74], v[vgprValuC+74], v90    // accum partials

s_waitcnt vmcnt(7)                                 // wait C (interleaved) 7 = 16 - 8 - 1
v_add_f32 v[vgprValuC+75], v[vgprValuC+75], v91    // accum partials

s_waitcnt vmcnt(6)                                 // wait C (interleaved) 6 = 16 - 9 - 1
v_add_f32 v[vgprValuC+76], v[vgprValuC+76], v92    // accum partials

s_waitcnt vmcnt(5)                                 // wait C (interleaved) 5 = 16 - 10 - 1
v_add_f32 v[vgprValuC+77], v[vgprValuC+77], v93    // accum partials

s_waitcnt vmcnt(4)                                 // wait C (interleaved) 4 = 16 - 11 - 1
v_add_f32 v[vgprValuC+78], v[vgprValuC+78], v94    // accum partials

s_waitcnt vmcnt(3)                                 // wait C (interleaved) 3 = 16 - 12 - 1
v_add_f32 v[vgprValuC+79], v[vgprValuC+79], v95    // accum partials

s_waitcnt vmcnt(2)                                 // wait C (interleaved) 2 = 16 - 13 - 1
v_add_f32 v[vgprValuC+80], v[vgprValuC+80], v96    // accum partials

s_waitcnt vmcnt(1)                                 // wait C (interleaved) 1 = 16 - 14 - 1
v_add_f32 v[vgprValuC+81], v[vgprValuC+81], v97    // accum partials

s_waitcnt vmcnt(0)                                 // wait C (interleaved) 0 = 16 - 15 - 1
v_add_f32 v[vgprValuC+82], v[vgprValuC+82], v98    // accum partials
v_mov_b32 v[vgprValuC+0], v[vgprValuC+67]          // copy vreg[0] to MI out reg
v_mov_b32 v[vgprValuC+1], v[vgprValuC+68]          // copy vreg[1] to MI out reg
v_mov_b32 v[vgprValuC+2], v[vgprValuC+69]          // copy vreg[2] to MI out reg
v_mov_b32 v[vgprValuC+3], v[vgprValuC+70]          // copy vreg[3] to MI out reg
v_mov_b32 v[vgprValuC+4], v[vgprValuC+71]          // copy vreg[4] to MI out reg
v_mov_b32 v[vgprValuC+5], v[vgprValuC+72]          // copy vreg[5] to MI out reg
v_mov_b32 v[vgprValuC+6], v[vgprValuC+73]          // copy vreg[6] to MI out reg
v_mov_b32 v[vgprValuC+7], v[vgprValuC+74]          // copy vreg[7] to MI out reg
v_mov_b32 v[vgprValuC+8], v[vgprValuC+75]          // copy vreg[8] to MI out reg
v_mov_b32 v[vgprValuC+9], v[vgprValuC+76]          // copy vreg[9] to MI out reg
v_mov_b32 v[vgprValuC+10], v[vgprValuC+77]         // copy vreg[10] to MI out reg
v_mov_b32 v[vgprValuC+11], v[vgprValuC+78]         // copy vreg[11] to MI out reg
v_mov_b32 v[vgprValuC+12], v[vgprValuC+79]         // copy vreg[12] to MI out reg
v_mov_b32 v[vgprValuC+13], v[vgprValuC+80]         // copy vreg[13] to MI out reg
v_mov_b32 v[vgprValuC+14], v[vgprValuC+81]         // copy vreg[14] to MI out reg
v_mov_b32 v[vgprValuC+15], v[vgprValuC+82]         // copy vreg[15] to MI out reg
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_mul_i32 s11, s[sgprskTiles], s[sgprItersPerTile]
s_mul_i32 s61, s[sgprSKItersPerWG], s[sgprskGrid]
s_sub_u32 s11, s11, s61                            // skTiles * ItersPerTile - SKItersPerWG * skGrid
s_bitcmp1_b32 s[sgprMagicShiftItersPerTile], 29    // USO on? (bit 29 of MagicShiftItersPerTile); off -> historical global peer size
s_cbranch_scc0 label_SK_PeerGlobal                 // USO off -> historical global mapping
s_cmp_eq_u32 s[sgprskTiles], 0                     // skTiles == 0?
s_cbranch_scc1 label_SK_PeerNoTiles                // global peer size
v_cvt_f32_u32 v65, s[sgprskTiles]                  // skGrid % skTiles
v_rcp_iflag_f32 v65, v65                           // skGrid % skTiles
v_mov_b32 v66, 0x4f7ffffe                          // 2^32 - 512
v_mul_f32 v65, v66, v65                            // 2^32 / divisor
v_cvt_u32_f32 v65, v65                             // skGrid % skTiles
v_mul_lo_u32 v66, v65, s[sgprskTiles]              // refine 2^32 / divisor
v_sub_nc_u32 v66, 0, v66                           // refine 2^32 / divisor
v_mul_hi_u32 v66, v65, v66                         // refine 2^32 / divisor
v_add_nc_u32 v65, v65, v66                         // refine 2^32 / divisor
v_mul_hi_u32 v65, s[sgprskGrid], v65               // skGrid % skTiles
v_mul_lo_u32 v66, v65, s[sgprskTiles]              // skGrid % skTiles
v_sub_nc_u32 v66, s[sgprskGrid], v66               // skGrid % skTiles
v_cmp_ge_u32 vcc_lo, v66, s[sgprskTiles]           // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v65, 1, v65                           // quotient + 1
v_sub_nc_u32 v66, v66, s[sgprskTiles]              // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v66, s[sgprskTiles]           // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v65, 1, v65                           // quotient + 1
v_sub_nc_u32 v66, v66, s[sgprskTiles]              // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s61, v65                       // quotient
v_readfirstlane_b32 s61, v66                       // remainder
s_cmp_eq_u32 s61, 0                                // skGrid % skTiles == 0?
s_cbranch_scc0 label_SK_PeerGlobal                 // ragged -> global peer
label_SK_PeerPerTile:
v_cvt_f32_u32 v65, s[sgprskTiles]                  // F = skGrid / skTiles
v_rcp_iflag_f32 v65, v65                           // F = skGrid / skTiles
v_mov_b32 v66, 0x4f7ffffe                          // 2^32 - 512
v_mul_f32 v65, v66, v65                            // 2^32 / divisor
v_cvt_u32_f32 v65, v65                             // F = skGrid / skTiles
v_mul_lo_u32 v66, v65, s[sgprskTiles]              // refine 2^32 / divisor
v_sub_nc_u32 v66, 0, v66                           // refine 2^32 / divisor
v_mul_hi_u32 v66, v65, v66                         // refine 2^32 / divisor
v_add_nc_u32 v65, v65, v66                         // refine 2^32 / divisor
v_mul_hi_u32 v65, s[sgprskGrid], v65               // F = skGrid / skTiles
v_mul_lo_u32 v66, v65, s[sgprskTiles]              // F = skGrid / skTiles
v_sub_nc_u32 v66, s[sgprskGrid], v66               // F = skGrid / skTiles
v_cmp_ge_u32 vcc_lo, v66, s[sgprskTiles]           // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v65, 1, v65                           // quotient + 1
v_sub_nc_u32 v66, v66, s[sgprskTiles]              // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v66, s[sgprskTiles]           // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v65, 1, v65                           // quotient + 1
v_sub_nc_u32 v66, v66, s[sgprskTiles]              // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s61, v65                       // quotient
s_mov_b32 s11, s61                                 // F = skGrid / skTiles
v_cvt_f32_u32 v65, s11                             // s = cta % F
v_rcp_iflag_f32 v65, v65                           // s = cta % F
v_mov_b32 v66, 0x4f7ffffe                          // 2^32 - 512
v_mul_f32 v65, v66, v65                            // 2^32 / divisor
v_cvt_u32_f32 v65, v65                             // s = cta % F
v_mul_lo_u32 v66, v65, s11                         // refine 2^32 / divisor
v_sub_nc_u32 v66, 0, v66                           // refine 2^32 / divisor
v_mul_hi_u32 v66, v65, v66                         // refine 2^32 / divisor
v_add_nc_u32 v65, v65, v66                         // refine 2^32 / divisor
v_mul_hi_u32 v65, s7, v65                          // s = cta % F
v_mul_lo_u32 v66, v65, s11                         // s = cta % F
v_sub_nc_u32 v66, s7, v66                          // s = cta % F
v_cmp_ge_u32 vcc_lo, v66, s11                      // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v65, 1, v65                           // quotient + 1
v_sub_nc_u32 v66, v66, s11                         // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v66, s11                      // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v65, 1, v65                           // quotient + 1
v_sub_nc_u32 v66, v66, s11                         // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s61, v65                       // quotient
v_readfirstlane_b32 s61, v66                       // remainder
s_mul_i32 s11, s11, s[sgprSKItersPerWG]            // F * SKItersPerWG
s_sub_u32 s11, s[sgprItersPerTile], s11            // remI = ItersPerTile - F*W
s_cmp_lt_u32 s61, s11                              // s < remI?
s_cselect_b32 s61, 1, 0                            // extra iter within tile
s_add_u32 s61, s[sgprSKItersPerWG], s61            // chunk = W + (s < remI)
s_branch label_SK_PeerDone                         // skip global peer
label_SK_PeerNoTiles:
label_SK_PeerGlobal:
s_add_u32 s61, s[sgprSKItersPerWG], 1              // Add extra iter
s_cmp_lt_u32 s7, s11                               // Check if next WG had an extra iteration
s_cselect_b32 s61, s61, s[sgprSKItersPerWG]        // Select correct number of iterations for next WG
label_SK_PeerDone:
s_add_u32 s8, s8, s61                              // next partial tile iteration
s_add_u32 s7, s7, 1                                // next partial tile index
s_cmp_lt_u32 s8, s[sgprItersPerTile]               // done loading partial tiles?
s_cbranch_scc1 label_SK_Fixup                      // Branch to continue fixup loop
label_SK_Store:
s_cmpk_eq_u32 s[sgprBeta], 0                       // Beta == 0
s_cbranch_scc0 label_GW_B1_GSU1                    // Branch if Beta is not zero

label_GW_B0_GSU1:
label_GW_B0_FD0_GSU1:

/* Edge/NonEdge store path check (M): Size % 64 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s64, 63, s[sgprSizeI]                    // s64 = s[sgprSizeI] % 64
s_add_u32 s65, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s65                // wg0 >= nwg0-1 ?
s_cselect_b32 s64, s64, 0                          // set rem
s_cmpk_gt_u32 s64, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_GSU1_Else       // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 32 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s64, 31, s[sgprSizeJ]                    // s64 = s[sgprSizeJ] % 32
s_add_u32 s65, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s65                // wg1 >= nwg1-1
s_cselect_b32 s64, s64, 0                          // set rem
s_cmpk_gt_u32 s64, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_GSU1_Then       // jump if edges required
label_GW_B0_FD0_VW1_GSU1_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=100 */
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
v_add_lshl_u32 v65, v57, v54, 1                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=54, coord0Vgpr=54 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0), (3, 0, 0, 0), (4, 0, 0, 0), (5, 0, 0, 0), (6, 0, 0, 0), (7, 0, 0, 0), (8, 0, 0, 0), (9, 0, 0, 0), (10, 0, 0, 0), (11, 0, 0, 0), (12, 0, 0, 0), (13, 0, 0, 0), (14, 0, 0, 0), (15, 0, 0, 0)] */
v_mul_f32 v[vgprValuC+67], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+68], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+69], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+70], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+71], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+72], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+73], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+74], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+75], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+76], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+77], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+78], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+79], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+80], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+81], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+82], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v67.l, v[vgprValuC+67]               // convert C to fp16
buffer_store_b16 v67, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v68.l, v[vgprValuC+68]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v68, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v69.l, v[vgprValuC+69]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v69, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v70.l, v[vgprValuC+70]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v70, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v71.l, v[vgprValuC+71]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v71, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v72.l, v[vgprValuC+72]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v72, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v73.l, v[vgprValuC+73]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v73, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v74.l, v[vgprValuC+74]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v74, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v75.l, v[vgprValuC+75]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v75, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v76.l, v[vgprValuC+76]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v76, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v77.l, v[vgprValuC+77]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v77, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v78.l, v[vgprValuC+78]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v78, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v79.l, v[vgprValuC+79]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v79, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v80.l, v[vgprValuC+80]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v80, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v81.l, v[vgprValuC+81]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v81, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v82.l, v[vgprValuC+82]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v82, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW1_GSU1_NonEdgeEnd:
label_GW_B0_FD0_VW1_GSU1_Else:
label_GW_B0_FD0_VW1_GSU1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=50 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (1,0,0,0:vw1); (2,0,0,0:vw1); (3,0,0,0:vw1); (4,0,0,0:vw1); (5,0,0,0:vw1); (6,0,0,0:vw1); (7,0,0,0:vw1); (8,0,0,0:vw1); (9,0,0,0:vw1); (10,0,0,0:vw1); (11,0,0,0:vw1); (12,0,0,0:vw1); (13,0,0,0:vw1); (14,0,0,0:vw1); (15,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v60, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v81, v57, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v81, v60, v81, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v82, v57, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v82, v60, v82, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v83, v57, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v83, v60, v83, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v84, v57, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v84, v60, v84, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v85, v57, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v85, v60, v85, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v86, v57, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v86, v60, v86, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v87, v57, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v87, v60, v87, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v88, v57, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v88, v60, v88, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v89, v57, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v89, v60, v89, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v90, v57, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v90, v60, v90, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v91, v57, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v91, v60, v91, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v92, v57, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v92, v60, v92, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v93, v57, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v93, v60, v93, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v94, v57, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v94, v60, v94, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v95, v57, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v95, v60, v95, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v96, v57, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v96, v60, v96, s66                   // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0), (3, 0, 0, 0), (4, 0, 0, 0), (5, 0, 0, 0), (6, 0, 0, 0), (7, 0, 0, 0), (8, 0, 0, 0), (9, 0, 0, 0), (10, 0, 0, 0), (11, 0, 0, 0), (12, 0, 0, 0), (13, 0, 0, 0), (14, 0, 0, 0), (15, 0, 0, 0)] */
v_mul_f32 v[vgprValuC+65], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+66], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+67], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+68], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+69], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+70], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+71], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+72], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+73], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+74], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+75], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+76], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+77], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+78], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+79], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+80], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v65.l, v[vgprValuC+65]               // convert C to fp16
buffer_store_b16 v65, v81, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v66.l, v[vgprValuC+66]               // convert C to fp16
buffer_store_b16 v66, v82, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v67.l, v[vgprValuC+67]               // convert C to fp16
buffer_store_b16 v67, v83, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v68.l, v[vgprValuC+68]               // convert C to fp16
buffer_store_b16 v68, v84, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v69.l, v[vgprValuC+69]               // convert C to fp16
buffer_store_b16 v69, v85, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v70.l, v[vgprValuC+70]               // convert C to fp16
buffer_store_b16 v70, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v71.l, v[vgprValuC+71]               // convert C to fp16
buffer_store_b16 v71, v87, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v72.l, v[vgprValuC+72]               // convert C to fp16
buffer_store_b16 v72, v88, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v73.l, v[vgprValuC+73]               // convert C to fp16
buffer_store_b16 v73, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v74.l, v[vgprValuC+74]               // convert C to fp16
buffer_store_b16 v74, v90, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v75.l, v[vgprValuC+75]               // convert C to fp16
buffer_store_b16 v75, v91, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v76.l, v[vgprValuC+76]               // convert C to fp16
buffer_store_b16 v76, v92, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v77.l, v[vgprValuC+77]               // convert C to fp16
buffer_store_b16 v77, v93, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v78.l, v[vgprValuC+78]               // convert C to fp16
buffer_store_b16 v78, v94, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v79.l, v[vgprValuC+79]               // convert C to fp16
buffer_store_b16 v79, v95, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v80.l, v[vgprValuC+80]               // convert C to fp16
buffer_store_b16 v80, v96, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_GSU1:
label_GW_B1_FD0_GSU1:

/* Edge/NonEdge store path check (M): Size % 64 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s64, 63, s[sgprSizeI]                    // s64 = s[sgprSizeI] % 64
s_add_u32 s65, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s65                // wg0 >= nwg0-1 ?
s_cselect_b32 s64, s64, 0                          // set rem
s_cmpk_gt_u32 s64, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW1_GSU1_Else       // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 32 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s64, 31, s[sgprSizeJ]                    // s64 = s[sgprSizeJ] % 32
s_add_u32 s65, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s65                // wg1 >= nwg1-1
s_cselect_b32 s64, s64, 0                          // set rem
s_cmpk_gt_u32 s64, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW1_GSU1_Then       // jump if edges required
label_GW_B1_FD0_VW1_GSU1_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=48 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (1,0,0,0:vw1); (2,0,0,0:vw1); (3,0,0,0:vw1); (4,0,0,0:vw1); (5,0,0,0:vw1); (6,0,0,0:vw1); (7,0,0,0:vw1); (8,0,0,0:vw1); (9,0,0,0:vw1); (10,0,0,0:vw1); (11,0,0,0:vw1); (12,0,0,0:vw1); (13,0,0,0:vw1); (14,0,0,0:vw1); (15,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v66, v56, v54, 1                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=54, coord0Vgpr=54 (multiple bpe)
buffer_load_d16_b16 v83, v66, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v84, v66, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v85, v66, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v86, v66, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v87, v66, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v88, v66, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v89, v66, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v90, v66, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v91, v66, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v92, v66, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v93, v66, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v94, v66, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v95, v66, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v96, v66, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v97, v66, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v98, v66, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v65, v57, v54, 1                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=54, coord0Vgpr=54 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0), (3, 0, 0, 0), (4, 0, 0, 0), (5, 0, 0, 0), (6, 0, 0, 0), (7, 0, 0, 0), (8, 0, 0, 0), (9, 0, 0, 0), (10, 0, 0, 0), (11, 0, 0, 0), (12, 0, 0, 0), (13, 0, 0, 0), (14, 0, 0, 0), (15, 0, 0, 0)] */
v_mul_f32 v[vgprValuC+67], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+68], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+69], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+70], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+71], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+72], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+73], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+74], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+75], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+76], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+77], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+78], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+79], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+80], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+81], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+82], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(15)                                // vlcnt(15) = 16 - 1 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+67], s[sgprBeta], v83, v[vgprValuC+67] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v67.l, v[vgprValuC+67]               // convert C to fp16
buffer_store_b16 v67, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(14)                                // vlcnt(14) = 16 - 2 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+68], s[sgprBeta], v84, v[vgprValuC+68] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v68.l, v[vgprValuC+68]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v68, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(13)                                // vlcnt(13) = 16 - 3 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+69], s[sgprBeta], v85, v[vgprValuC+69] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v69.l, v[vgprValuC+69]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v69, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(12)                                // vlcnt(12) = 16 - 4 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+70], s[sgprBeta], v86, v[vgprValuC+70] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v70.l, v[vgprValuC+70]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v70, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(11)                                // vlcnt(11) = 16 - 5 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+71], s[sgprBeta], v87, v[vgprValuC+71] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v71.l, v[vgprValuC+71]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v71, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(10)                                // vlcnt(10) = 16 - 6 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+72], s[sgprBeta], v88, v[vgprValuC+72] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v72.l, v[vgprValuC+72]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v72, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(9)                                 // vlcnt(9) = 16 - 7 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+73], s[sgprBeta], v89, v[vgprValuC+73] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v73.l, v[vgprValuC+73]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v73, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(8)                                 // vlcnt(8) = 16 - 8 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+74], s[sgprBeta], v90, v[vgprValuC+74] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v74.l, v[vgprValuC+74]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v74, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(7)                                 // vlcnt(7) = 16 - 9 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+75], s[sgprBeta], v91, v[vgprValuC+75] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v75.l, v[vgprValuC+75]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v75, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(6)                                 // vlcnt(6) = 16 - 10 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+76], s[sgprBeta], v92, v[vgprValuC+76] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v76.l, v[vgprValuC+76]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v76, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(5)                                 // vlcnt(5) = 16 - 11 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+77], s[sgprBeta], v93, v[vgprValuC+77] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v77.l, v[vgprValuC+77]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v77, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(4)                                 // vlcnt(4) = 16 - 12 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+78], s[sgprBeta], v94, v[vgprValuC+78] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v78.l, v[vgprValuC+78]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v78, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(3)                                 // vlcnt(3) = 16 - 13 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+79], s[sgprBeta], v95, v[vgprValuC+79] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v79.l, v[vgprValuC+79]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v79, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(2)                                 // vlcnt(2) = 16 - 14 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+80], s[sgprBeta], v96, v[vgprValuC+80] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v80.l, v[vgprValuC+80]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v80, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(1)                                 // vlcnt(1) = 16 - 15 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+81], s[sgprBeta], v97, v[vgprValuC+81] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v81.l, v[vgprValuC+81]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v81, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(0)                                 // vlcnt(0) = 16 - 16 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+82], s[sgprBeta], v98, v[vgprValuC+82] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v82.l, v[vgprValuC+82]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v82, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_FD0_VW1_GSU1_NonEdgeEnd:
label_GW_B1_FD0_VW1_GSU1_Else:
label_GW_B1_FD0_VW1_GSU1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=34 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (1,0,0,0:vw1); (2,0,0,0:vw1); (3,0,0,0:vw1); (4,0,0,0:vw1); (5,0,0,0:vw1); (6,0,0,0:vw1); (7,0,0,0:vw1); (8,0,0,0:vw1); (9,0,0,0:vw1); (10,0,0,0:vw1); (11,0,0,0:vw1); (12,0,0,0:vw1); (13,0,0,0:vw1); (14,0,0,0:vw1); (15,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v60, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v82, v56, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v82, v60, v82, s66                   // LDC clip if OOB. offset
buffer_load_d16_b16 v81, v82, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v82, v57, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v82, v60, v82, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v84, v56, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v84, v60, v84, s66                   // LDC clip if OOB. offset
buffer_load_d16_b16 v83, v84, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v84, v57, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v84, v60, v84, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v86, v56, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v86, v60, v86, s66                   // LDC clip if OOB. offset
buffer_load_d16_b16 v85, v86, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v86, v57, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v86, v60, v86, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v88, v56, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v88, v60, v88, s66                   // LDC clip if OOB. offset
buffer_load_d16_b16 v87, v88, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v88, v57, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v88, v60, v88, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v90, v56, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v90, v60, v90, s66                   // LDC clip if OOB. offset
buffer_load_d16_b16 v89, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v90, v57, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v90, v60, v90, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v92, v56, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v92, v60, v92, s66                   // LDC clip if OOB. offset
buffer_load_d16_b16 v91, v92, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v92, v57, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v92, v60, v92, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v94, v56, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v94, v60, v94, s66                   // LDC clip if OOB. offset
buffer_load_d16_b16 v93, v94, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v94, v57, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v94, v60, v94, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v96, v56, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v96, v60, v96, s66                   // LDC clip if OOB. offset
buffer_load_d16_b16 v95, v96, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v96, v57, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v96, v60, v96, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v98, v56, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v98, v60, v98, s66                   // LDC clip if OOB. offset
buffer_load_d16_b16 v97, v98, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v98, v57, v54, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v98, v60, v98, s66                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v100, v56, v54, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v100, v60, v100, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v99, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v100, v57, v54, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v100, v60, v100, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v102, v56, v54, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v102, v60, v102, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v101, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v102, v57, v54, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v102, v60, v102, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v104, v56, v54, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v104, v60, v104, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v103, v104, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v104, v57, v54, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v104, v60, v104, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v106, v56, v54, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v106, v60, v106, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v105, v106, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v106, v57, v54, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v106, v60, v106, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v108, v56, v54, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v108, v60, v108, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v107, v108, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v108, v57, v54, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v108, v60, v108, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v110, v56, v54, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v110, v60, v110, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v109, v110, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v110, v57, v54, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v110, v60, v110, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
v_add_co_u32 v55, vcc_lo, v55, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v56, v56, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v57, v57, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v54, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v55, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v112, v56, v54, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v112, v60, v112, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v111, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v112, v57, v54, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v112, v60, v112, s66                 // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0), (3, 0, 0, 0), (4, 0, 0, 0), (5, 0, 0, 0), (6, 0, 0, 0), (7, 0, 0, 0), (8, 0, 0, 0), (9, 0, 0, 0), (10, 0, 0, 0), (11, 0, 0, 0), (12, 0, 0, 0), (13, 0, 0, 0), (14, 0, 0, 0), (15, 0, 0, 0)] */
v_mul_f32 v[vgprValuC+65], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+66], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+67], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+68], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+69], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+70], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+71], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+72], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+73], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+74], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+75], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+76], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+77], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+78], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+79], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+80], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fma_mix_f32 v[vgprValuC+65], s[sgprBeta], v81, v[vgprValuC+65] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v65.l, v[vgprValuC+65]               // convert C to fp16
buffer_store_b16 v65, v82, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+66], s[sgprBeta], v83, v[vgprValuC+66] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v66.l, v[vgprValuC+66]               // convert C to fp16
buffer_store_b16 v66, v84, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+67], s[sgprBeta], v85, v[vgprValuC+67] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v67.l, v[vgprValuC+67]               // convert C to fp16
buffer_store_b16 v67, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+68], s[sgprBeta], v87, v[vgprValuC+68] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v68.l, v[vgprValuC+68]               // convert C to fp16
buffer_store_b16 v68, v88, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+69], s[sgprBeta], v89, v[vgprValuC+69] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v69.l, v[vgprValuC+69]               // convert C to fp16
buffer_store_b16 v69, v90, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+70], s[sgprBeta], v91, v[vgprValuC+70] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v70.l, v[vgprValuC+70]               // convert C to fp16
buffer_store_b16 v70, v92, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+71], s[sgprBeta], v93, v[vgprValuC+71] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v71.l, v[vgprValuC+71]               // convert C to fp16
buffer_store_b16 v71, v94, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+72], s[sgprBeta], v95, v[vgprValuC+72] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v72.l, v[vgprValuC+72]               // convert C to fp16
buffer_store_b16 v72, v96, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+73], s[sgprBeta], v97, v[vgprValuC+73] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v73.l, v[vgprValuC+73]               // convert C to fp16
buffer_store_b16 v73, v98, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+74], s[sgprBeta], v99, v[vgprValuC+74] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v74.l, v[vgprValuC+74]               // convert C to fp16
buffer_store_b16 v74, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+75], s[sgprBeta], v101, v[vgprValuC+75] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v75.l, v[vgprValuC+75]               // convert C to fp16
buffer_store_b16 v75, v102, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+76], s[sgprBeta], v103, v[vgprValuC+76] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v76.l, v[vgprValuC+76]               // convert C to fp16
buffer_store_b16 v76, v104, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+77], s[sgprBeta], v105, v[vgprValuC+77] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v77.l, v[vgprValuC+77]               // convert C to fp16
buffer_store_b16 v77, v106, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+78], s[sgprBeta], v107, v[vgprValuC+78] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v78.l, v[vgprValuC+78]               // convert C to fp16
buffer_store_b16 v78, v108, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+79], s[sgprBeta], v109, v[vgprValuC+79] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v79.l, v[vgprValuC+79]               // convert C to fp16
buffer_store_b16 v79, v110, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+80], s[sgprBeta], v111, v[vgprValuC+80] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v80.l, v[vgprValuC+80]               // convert C to fp16
buffer_store_b16 v80, v112, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_SK_Partials_1:
label_GW_Partials_E0:
s_mov_b64 s[sgprSrdWS:sgprSrdWS+1], s[sgprAddressWS:sgprAddressWS+1]
s_mov_b32 s[sgprSrdWS+2], BufferOOB
s_mov_b32 s[sgprSrdWS+3], Srd127_96

s_mul_i32 s7, 0x2000, s[sgprPersistentWorkGroupIndex] // Offset to correct partials tile (low word)
s_mul_hi_u32 s8, 0x2000, s[sgprPersistentWorkGroupIndex] // partials tile offset (high word) for 64-bit SRD
s_add_u32 s[sgprSrdWS+0], s[sgprSrdWS+0], s7       // add lo to SRD
s_addc_u32 s[sgprSrdWS+1], s[sgprSrdWS+1], s8      // add hi (offset high word + lo carry) to SRD

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=100 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Partials Write Batch #0 (d1,d0,vc1,vc0) = */
/*      (0,0,0,0:vw1); (1,0,0,0:vw1); (2,0,0,0:vw1); (3,0,0,0:vw1); (4,0,0,0:vw1); (5,0,0,0:vw1); (6,0,0,0:vw1); (7,0,0,0:vw1); (8,0,0,0:vw1); (9,0,0,0:vw1); (10,0,0,0:vw1); (11,0,0,0:vw1); (12,0,0,0:vw1); (13,0,0,0:vw1); (14,0,0,0:vw1); (15,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v[vgprValuC+67], v[vgprValuC+0]          // copy MI out reg to vreg[0]
v_mov_b32 v[vgprValuC+68], v[vgprValuC+1]          // copy MI out reg to vreg[1]
v_mov_b32 v[vgprValuC+69], v[vgprValuC+2]          // copy MI out reg to vreg[2]
v_mov_b32 v[vgprValuC+70], v[vgprValuC+3]          // copy MI out reg to vreg[3]
v_mov_b32 v[vgprValuC+71], v[vgprValuC+4]          // copy MI out reg to vreg[4]
v_mov_b32 v[vgprValuC+72], v[vgprValuC+5]          // copy MI out reg to vreg[5]
v_mov_b32 v[vgprValuC+73], v[vgprValuC+6]          // copy MI out reg to vreg[6]
v_mov_b32 v[vgprValuC+74], v[vgprValuC+7]          // copy MI out reg to vreg[7]
v_mov_b32 v[vgprValuC+75], v[vgprValuC+8]          // copy MI out reg to vreg[8]
v_mov_b32 v[vgprValuC+76], v[vgprValuC+9]          // copy MI out reg to vreg[9]
v_mov_b32 v[vgprValuC+77], v[vgprValuC+10]         // copy MI out reg to vreg[10]
v_mov_b32 v[vgprValuC+78], v[vgprValuC+11]         // copy MI out reg to vreg[11]
v_mov_b32 v[vgprValuC+79], v[vgprValuC+12]         // copy MI out reg to vreg[12]
v_mov_b32 v[vgprValuC+80], v[vgprValuC+13]         // copy MI out reg to vreg[13]
v_mov_b32 v[vgprValuC+81], v[vgprValuC+14]         // copy MI out reg to vreg[14]
v_mov_b32 v[vgprValuC+82], v[vgprValuC+15]         // copy MI out reg to vreg[15]

/* apply mask, calc new C and issue writes */
v_lshlrev_b32 v65, 2, v[vgprSerial]                // v65 = v[vgprSerial] * 4
s_mov_b32 s8, 0                                    // Init sgpr offset
buffer_store_b32 v67, v65, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v68, v65, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v69, v65, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v70, v65, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v71, v65, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v72, v65, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v73, v65, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v74, v65, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v75, v65, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v76, v65, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v77, v65, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v78, v65, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v79, v65, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v80, v65, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v81, v65, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v82, v65, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_waitcnt_vscnt null 0                             // wait for data store
s_barrier                                          // store all data before setting flag
s_lshl_b32 s8, s[sgprPersistentWorkGroupIndex], 2  // flag offset based on physical partial slot
v_readfirstlane_b32 s7, v[vgprSerial]              // Wave 0 updates flags
s_cmp_eq_u32 s7, 0                                 // Check for wave 0
s_cbranch_scc0 label_SK_SkipFlagSet                // Skip flag set
s_mov_b32 s7, 1                                    // flag data
v_mov_b32 v67, s7                                  // move flag value to vgpr
v_mov_b32 v68, 0                                   // zero vaddr offset
s_mov_b64 s[64:65], s[sgprAddressFlags:sgprAddressFlags+1]
s_mov_b32 s66, BufferOOB
s_mov_b32 s67, Srd127_96
buffer_store_b32 v67, v68, s[64:67], s8 offen offset:0 glc dlc // set flag
s_waitcnt_vscnt null 0                             // wait for data store
label_SK_SkipFlagSet:
s_waitcnt lgkmcnt(0)                               // wait for flag
s_branch label_GW_End_1                            // jump to end
label_GW_End_1:
label_PersistentLoopClose:
s_cmp_ge_u32 s[sgprPersistentIteration], s[sgprPersistentIterationEnd] // Check whether assigned work is exhausted
s_cbranch_scc1 label_StaticGrid_CloseDone
s_getpc_b64 s[64:65]                               // addr of next instr
s_add_i32 s66, label_PersistentLoopStart, 4        // target branch offset
s_abs_i32 s66, s66                                 // abs offset
s_sub_u32 s64, s64, s66                            // sub target branch offset
s_subb_u32 s65, s65, 0                             // sub high and carry
s_setpc_b64 s[64:65]                               // branch to label_PersistentLoopStart
label_StaticGrid_CloseDone:
label_KernelEnd:
s_endpgm                                           // Kernel End
label_ASM_End:  /// The end of the kernel
