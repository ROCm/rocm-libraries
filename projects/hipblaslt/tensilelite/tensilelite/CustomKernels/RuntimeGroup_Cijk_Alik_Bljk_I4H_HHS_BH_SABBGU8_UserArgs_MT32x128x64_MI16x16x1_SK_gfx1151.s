
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
.text
.protected RuntimeGroup_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGU8_UserArgs_MT32x128x64_MI16x16x1_SK_gfx1151
.globl RuntimeGroup_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGU8_UserArgs_MT32x128x64_MI16x16x1_SK_gfx1151
.p2align 8
.type RuntimeGroup_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGU8_UserArgs_MT32x128x64_MI16x16x1_SK_gfx1151,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel RuntimeGroup_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGU8_UserArgs_MT32x128x64_MI16x16x1_SK_gfx1151
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 256 // vgprs
  .amdhsa_next_free_sgpr 92 // sgprs
  .amdhsa_group_segment_fixed_size 32768 // lds bytes
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
/* Num SGPR   =92 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 16 x 4 */
/* SubGroup= 2 x 32 */
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
  TileProcessingStrategy: StreamK
  WorkAssignment: StaticGrid
amdhsa.version:
  - 1
  - 1
amdhsa.kernels:
  - .name: RuntimeGroup_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGU8_UserArgs_MT32x128x64_MI16x16x1_SK_gfx1151
    .symbol: 'RuntimeGroup_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGU8_UserArgs_MT32x128x64_MI16x16x1_SK_gfx1151.kd'
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
    .group_segment_fixed_size:   32768
    .kernarg_segment_align:      8
    .kernarg_segment_size:       200
    .max_flat_workgroup_size:    128
    .private_segment_fixed_size: 0
    .sgpr_count:                 92
    .sgpr_spill_count:           0
    .vgpr_count:                 256
    .vgpr_spill_count:           0
    .wavefront_size:             32
...
.end_amdgpu_metadata
RuntimeGroup_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGU8_UserArgs_MT32x128x64_MI16x16x1_SK_gfx1151:
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
/* ValuC range: [0-64), serializedStore enabled */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprBase, 82
.set vgprLocalWriteAddrA, 78
.set vgprLocalWriteAddrB, 79
.set vgprGlobalReadOffsetA, 64
.set vgprGlobalReadOffsetB, 66
.set vgprGlobalReadOffsetScaleA, 74
.set vgprG2LScaleA, 76
.set vgprLocalReadAddrA, 80
.set vgprLocalReadAddrB, 81
.set vgprSerial, 220

/******************************************/
/* VGPR Macro Assignments                 */
/******************************************/
.set vgprValuA_X0_I0_BASE, vgprBase+0
.set vgprValuB_X0_I0_BASE, vgprBase+33
.set vgprG2LA_BASE, vgprBase+98
.set vgprG2LB_BASE, vgprBase+106
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+16
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X1_I0, vgprValuB_X0_I0_BASE+32
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
.set sgprLSUTailLoopOffset, 7
.set sgprLoopCounterL, 8
.set sgprOrigLoopCounter, 9
.set sgprSrdD, 12
.set sgprSrdC, 16
.set sgprNumWorkGroups0, 10
.set sgprNumWorkGroups1, 11
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

.set MT0, 32
.set MT1, 128
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
s_load_b32 s16, s[sgprKernArgAddress:sgprKernArgAddress+1], 0

/* Load packed kernel args (StaggerU/GSU) */
s_load_b32 s18, s[sgprKernArgAddress:sgprKernArgAddress+1], 4

/* Load WGM data */
s_load_b32 s[sgprWGM], s[sgprKernArgAddress:sgprKernArgAddress+1], 8

/* Load num of WGs */
s_load_b32 s19, s[sgprKernArgAddress:sgprKernArgAddress+1], 12
s_waitcnt lgkmcnt(0)                               // load args
s_lshr_b32 s17, s16, 0x1e                          // Get arg type
s_and_b32 s16, 0x3fffffff, s16                     // Get nums of gemm
s_cmp_eq_u32 s17, 3                                // Is kernel argType == 3
s_cbranch_scc1 label_Bypass_ArgType3_to_ArgType0_Instance1
s_cmp_eq_u32 s17, 0                                // Is kernel args
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
s_mov_b32 s[sgprArgType], s17
s_setprio 3                                        // optimization store
s_mov_b32 m0, 0x8000                               // LDS clamp at 32768 bytes
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
s_lshr_b32 s77, s19, s80
s_lshl_b32 s77, s77, s80
s_cmp_ge_u32 s[sgprWorkGroup0], s77
s_cbranch_scc1 label_skip_WGMXCC
s_cmp_eq_u32 s81, 0                                // CU_Count == 0 ?
s_cbranch_scc0 label_XCCG_nonzero
s_lshr_b32 s77, s[sgprWorkGroup0], s80
s_bfm_b32 s78, s80, 0
s_and_b32 s78, s[sgprWorkGroup0], s78
s_lshr_b32 s79, s19, s80
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
v_cvt_f64_u32 v[0:1], s81                          // s78 = s19 / s81
v_rcp_f64 v[0:1], v[0:1]                           // s78 = s19 / s81
v_cvt_f64_u32 v[2:3], s19                          // s78 = s19 / s81
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s78 = s19 / s81
v_cvt_u32_f64 v0, v[0:1]                           // s78 = s19 / s81
v_mul_lo_u32 v1, v0, s81                           // s78 = s19 / s81
v_sub_nc_u32 v2, s19, v1                           // s78 = s19 / s81
v_cmp_ge_u32 vcc_lo, v2, s81                       // s78 = s19 / s81
s_mov_b32 exec_lo, vcc_lo                          // s78 = s19 / s81
v_add_nc_u32 v0, v0, 1                             // s78 = s19 / s81
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s78, v0                        // quotient
s_mul_i32 s78, s78, s81
s_sub_u32 s79, s19, s78
s_cmp_gt_u32 s[sgprWorkGroup0], s78
s_cselect_b32 s78, s79, s81
s_lshr_b32 s78, s78, s80
s_bfm_b32 s79, s80, 0
s_and_b32 s79, s[sgprWorkGroup0], s79
s_mul_i32 s78, s78, s79
/* WorkGroup0 = temp0 + temp1 */
s_add_u32 s[sgprWorkGroup0], s77, s78
label_skip_WGMXCC:  /// skip WGMXCC if no enough WGs to remap
s_cmp_eq_u32 s17, 3
s_cbranch_scc1 label_ArgType3_Routed_To_ArgType0
s_cmp_eq_u32 s17, 0
s_cbranch_scc0 label_MultiGemm
label_ArgType3_Routed_To_ArgType0:
/* init: add vgpr [82...261) to pool */
/* init: add vgpr [0...64) to pool */
/* init: add agpr [0...0) to pool */
s_waitcnt lgkmcnt(0)                               // wait for 128/0 bytes of kern args over preload
s_lshr_b32 s[sgprNumWorkGroups0], s[sgprSizesFree+0], 5 // s[sgprNumWorkGroups0] = s[sgprSizesFree+0] / 32
s_and_b32 s12, 31, s[sgprSizesFree+0]              // s12 = s[sgprSizesFree+0] % 32
s_cmp_gt_u32 s12, 0                                // Check if remainder > 0
s_addc_u32 s[sgprNumWorkGroups0], s[sgprNumWorkGroups0], 0 // Increment if remainder > 0
s_lshr_b32 s[sgprNumWorkGroups1], s[sgprSizesFree+1], 7 // s[sgprNumWorkGroups1] = s[sgprSizesFree+1] / 128
s_and_b32 s12, 127, s[sgprSizesFree+1]             // s12 = s[sgprSizesFree+1] % 128
s_cmp_gt_u32 s12, 0                                // Check if remainder > 0
s_addc_u32 s[sgprNumWorkGroups1], s[sgprNumWorkGroups1], 0 // Increment if remainder > 0
s_branch label_MultiGemmEnd
label_MultiGemm:

/* Check if custom structure pointer is null */
s_and_b32 s12, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s12, 2                                // ArgType == 2 ?
s_cbranch_scc1 label_IsExternalValid               // branch if ArgType == 2
s_mov_b32 s11, 144                                 // KernArgAddressOffset
s_mul_i32 s82, s16, 4
s_mov_b64 s[76:77], s[sgprKernArgAddress:sgprKernArgAddress+1]
s_branch label_IsExternalValidEnd
label_IsExternalValid:
s_mov_b32 s11, 256
s_mov_b32 s82, 0
s_mov_b64 s[76:77], s[sgprKernArgAddress:sgprKernArgAddress+1]
label_IsExternalValidEnd:

/* Grouped Gemm:: prefetch 1 arg load */
s_mov_b32 s10, 1
s_mov_b32 s83, 0
s_load_b128 s[20:23], s[76:77], s82
s_cmpk_eq_u32 s16, 1                               // if gemm_count is 1?
s_cbranch_scc1 label_wgTable_noLoadLoop

/* Grouped Gemm:: accumulate numTiles for each gemm */
/* Grouped Gemm:: loop start */
label_Loop_GemmCount:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s80, s20, 5                             // s80 = s20 / 32
s_and_b32 s78, 31, s20                             // s78 = s20 % 32
s_addc_u32 s80, s80, 0
s_lshr_b32 s81, s21, 7                             // s81 = s21 / 128
s_and_b32 s78, 127, s21                            // s78 = s21 % 128
s_addc_u32 s81, s81, 0
s_mul_i32 s80, s80, s81
s_mul_i32 s80, s80, s22
s_add_u32 s83, s83, s80
s_cmp_lt_u32 s[sgprWorkGroup0], s83
s_cbranch_scc1 label_FOUND
s_add_u32 s82, s82, s11
s_load_b128 s[20:23], s[76:77], s82
s_add_u32 s10, s10, 1
s_cmp_lt_u32 s10, s16
s_cbranch_scc1 label_Loop_GemmCount

/* Grouped Gemm:: noLoadLoop */
label_wgTable_noLoadLoop:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s80, s20, 5                             // s80 = s20 / 32
s_and_b32 s78, 31, s20                             // s78 = s20 % 32
s_addc_u32 s80, s80, 0
s_lshr_b32 s81, s21, 7                             // s81 = s21 / 128
s_and_b32 s78, 127, s21                            // s78 = s21 % 128
s_addc_u32 s81, s81, 0
s_mul_i32 s80, s80, s81
s_mul_i32 s80, s80, s22
s_add_u32 s83, s83, s80

/* Grouped Gemm:: gemmIndex found */
label_FOUND:
s_sub_u32 s77, s10, 1
s_sub_u32 s76, s83, s80
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s76
/* Check if custom structure pointer is null */
s_and_b32 s12, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s12, 2                                // ArgType == 2 ?
s_cbranch_scc1 label_LoadExternalStruct            // branch if ArgType == 2

/* Grouped Gemm: offset argument address to gemm */
/* Grouped Gemm: offset address from wg_table_start to args_start */
s_lshl2_add_u32 s[sgprKernArgAddress], s16, s[sgprKernArgAddress]
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s77, s77, 144                            // KernArgAddressOffset
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s77
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
s_mul_i32 s77, s77, 256
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s77
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
/* init: add vgpr [82...261) to pool */
/* init: add vgpr [0...64) to pool */
/* init: add agpr [0...0) to pool */
s_waitcnt lgkmcnt(0)                               // wait for 128/0 bytes of kern args over preload
s_lshr_b32 s[sgprNumWorkGroups0], s[sgprSizesFree+0], 5 // s[sgprNumWorkGroups0] = s[sgprSizesFree+0] / 32
s_and_b32 s12, 31, s[sgprSizesFree+0]              // s12 = s[sgprSizesFree+0] % 32
s_cmp_gt_u32 s12, 0                                // Check if remainder > 0
s_addc_u32 s[sgprNumWorkGroups0], s[sgprNumWorkGroups0], 0 // Increment if remainder > 0
s_lshr_b32 s[sgprNumWorkGroups1], s[sgprSizesFree+1], 7 // s[sgprNumWorkGroups1] = s[sgprSizesFree+1] / 128
s_and_b32 s12, 127, s[sgprSizesFree+1]             // s12 = s[sgprSizesFree+1] % 128
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
s_and_b32 s12, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s12, 3                                // ArgType == 3 for General Batched GEMM
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
v_readfirstlane_b32 s12, v0                        // quotient
v_readfirstlane_b32 s13, v1                        // remainder
s_mul_i32 s14, s[sgprskTiles], s[sgprSKItersPerWG]
s_sub_u32 s14, s[sgprItersPerTile], s14            // extraIters = itersPerTile - SkSplit * skItersPerWG
s_mul_i32 s[sgprPersistentIteration], s13, s[sgprSKItersPerWG] // StreamK starting iteration (case: after extra iters)
s_cmp_lt_u32 s13, s14                              // Check if WG gets an extra iteration
s_cbranch_scc1 label_SK_HasExtra                   // Has extra iter
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s14 // This WG does not have an extra iteration
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIteration], s[sgprSKItersPerWG] // StreamK ending iteration (case: after extra iters)
s_branch label_SK_DoneExtra                        // Done init for parallel reduction
label_SK_HasExtra:
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s13 // This WG has an extra iteration
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIteration], s[sgprSKItersPerWG] // StreamK ending iteration (case: after extra iters)
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], 1 // StreamK ending iteration (case: after extra iters)
label_SK_DoneExtra:
s_mul_i32 s12, s12, s[sgprItersPerTile]            // Tile offset = tilesIdx * itersPerTile
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s12 // Offset to correct tile
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], s12 // Offset to correct tile
s_mov_b32 s[sgprSkPartialIdx], s13                 // Save partial idx for SrdD calculation
s_branch label_SK_InitDone                         // Done init for parallel reduction
label_SK_SplitInit:
s_mul_i32 s[sgprPersistentIteration], s[sgprPersistentWorkGroupIndex], s[sgprItersPerTile] // DP starting iteration (case: DP work to do)
s_mul_i32 s12, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1] // totalTiles = nwg0 * nwg1
s_mul_i32 s12, s12, s[sgprSizesFree+2]             // totalTiles *= batch dim 0
s_mul_i32 s12, s12, s[sgprItersPerTile]            // totalIters = totalTiles * itersPerTile
s_mov_b32 s[sgprPersistentIterationEnd], s12       // DP ending iteration (case: only DP work to do)
s_mul_i32 s12, s[sgprskTiles], s[sgprItersPerTile] // Total SK iters
s_cmp_lt_u32 s12, s[sgprPersistentIterationEnd]    // Check if there are DP tiles to do
s_cbranch_scc1 label_SK_InitDone                   // Done init
s_mul_i32 s12, s[sgprskTiles], s[sgprItersPerTile]
s_mul_i32 s13, s[sgprSKItersPerWG], s[sgprskGrid]
s_sub_u32 s12, s12, s13                            // skTiles * ItersPerTile - SKItersPerWG * skGrid
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
v_readfirstlane_b32 s13, v0                        // quotient
v_readfirstlane_b32 s14, v1                        // remainder
s_cmp_eq_u32 s14, 0                                // skGrid % skTiles == 0?
s_cbranch_scc1 label_SK_PerTileExtraIters          // all-partial -> per-tile extras
s_branch label_SK_GlobalExtraIters                 // ragged -> global mapping
label_SK_AssignNoTiles:
label_SK_GlobalExtraIters:
s_mul_i32 s[sgprPersistentIteration], s[sgprPersistentWorkGroupIndex], s[sgprSKItersPerWG] // StreamK starting iteration (case: after extra iters)
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s12 // Add extra iters
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIteration], s[sgprSKItersPerWG] // StreamK ending iteration (case: after extra iters)
s_add_u32 s14, s[sgprSKItersPerWG], 1              // Spread out extra iterations
s_mul_i32 s13, s[sgprPersistentWorkGroupIndex], s14 // StreamK starting iteration (case: before extra iters)
s_add_u32 s14, s13, s14                            // StreamK ending iteration (case: before extra iters)
s_cmp_lt_u32 s[sgprPersistentWorkGroupIndex], s12  // Check if lane gets an extra iteration
s_cselect_b32 s[sgprPersistentIteration], s13, s[sgprPersistentIteration] // Set start iter
s_cselect_b32 s[sgprPersistentIterationEnd], s14, s[sgprPersistentIterationEnd] // Set end iter
s_branch label_SK_AssignItersDone                  // skip per-tile path
label_SK_PerTileExtraIters:
s_mov_b32 s12, s13                                 // F = skGrid / skTiles
v_cvt_f32_u32 v0, s12                              // q = w/F, s = w%F
v_rcp_iflag_f32 v0, v0                             // q = w/F, s = w%F
v_mov_b32 v1, 0x4f7ffffe                           // 2^32 - 512
v_mul_f32 v0, v1, v0                               // 2^32 / divisor
v_cvt_u32_f32 v0, v0                               // q = w/F, s = w%F
v_mul_lo_u32 v1, v0, s12                           // refine 2^32 / divisor
v_sub_nc_u32 v1, 0, v1                             // refine 2^32 / divisor
v_mul_hi_u32 v1, v0, v1                            // refine 2^32 / divisor
v_add_nc_u32 v0, v0, v1                            // refine 2^32 / divisor
v_mul_hi_u32 v0, s[sgprPersistentWorkGroupIndex], v0 // q = w/F, s = w%F
v_mul_lo_u32 v1, v0, s12                           // q = w/F, s = w%F
v_sub_nc_u32 v1, s[sgprPersistentWorkGroupIndex], v1 // q = w/F, s = w%F
v_cmp_ge_u32 vcc_lo, v1, s12                       // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v0, 1, v0                             // quotient + 1
v_sub_nc_u32 v1, v1, s12                           // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v1, s12                       // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v0, 1, v0                             // quotient + 1
v_sub_nc_u32 v1, v1, s12                           // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s13, v0                        // quotient
v_readfirstlane_b32 s14, v1                        // remainder
s_mul_i32 s[sgprPersistentIteration], s13, s[sgprItersPerTile] // q * ItersPerTile
s_mul_i32 s13, s14, s[sgprSKItersPerWG]            // s * SKItersPerWG
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s13 // q*I + s*W
s_mul_i32 s13, s12, s[sgprSKItersPerWG]            // F * SKItersPerWG
s_sub_u32 s13, s[sgprItersPerTile], s13            // remI = ItersPerTile - F*SKItersPerWG
s_min_u32 s12, s14, s13                            // min(s, remI)
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s12 // start = q*I + s*W + min(s, remI)
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIteration], s[sgprSKItersPerWG] // start + SKItersPerWG
s_cmp_lt_u32 s14, s13                              // s < remI?
s_cselect_b32 s12, 1, 0                            // extra iter within tile
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], s12 // end = start + W + (s < remI)
label_SK_AssignItersDone:
s_mul_i32 s12, s[sgprskTiles], s[sgprItersPerTile] // Total SK iters
s_min_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], s12 // Cap ending iter at total SK iters
label_SK_InitDone:
s_mul_i32 s12, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1] // totalTiles = nwg0 * nwg1
s_mul_i32 s12, s12, s[sgprSizesFree+2]             // totalTiles *= batch dim 0
s_mul_i32 s12, s12, s[sgprItersPerTile]            // totalIters = totalTiles * itersPerTile
s_cmp_lt_u32 s[sgprPersistentIteration], s12       // Make sure there's work to do
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
s_mul_i32 s12, s[sgprskTiles], s[sgprItersPerTile]
s_mul_i32 s13, s[sgprSKItersPerWG], s[sgprskGrid]
s_sub_u32 s12, s12, s13                            // skTiles * ItersPerTile - SKItersPerWG * skGrid
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
v_readfirstlane_b32 s13, v0                        // quotient
v_readfirstlane_b32 s14, v1                        // remainder
s_cmp_eq_u32 s14, 0                                // skGrid % skTiles == 0?
s_cbranch_scc1 label_SK_PerTileExtraIters_1        // all-partial -> per-tile extras
s_branch label_SK_GlobalExtraIters_1               // ragged -> global mapping
label_SK_AssignNoTiles_1:
label_SK_GlobalExtraIters_1:
s_mul_i32 s[sgprPersistentIteration], s[sgprPersistentWorkGroupIndex], s[sgprSKItersPerWG] // StreamK starting iteration (case: after extra iters)
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s12 // Add extra iters
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIteration], s[sgprSKItersPerWG] // StreamK ending iteration (case: after extra iters)
s_add_u32 s14, s[sgprSKItersPerWG], 1              // Spread out extra iterations
s_mul_i32 s13, s[sgprPersistentWorkGroupIndex], s14 // StreamK starting iteration (case: before extra iters)
s_add_u32 s14, s13, s14                            // StreamK ending iteration (case: before extra iters)
s_cmp_lt_u32 s[sgprPersistentWorkGroupIndex], s12  // Check if lane gets an extra iteration
s_cselect_b32 s[sgprPersistentIteration], s13, s[sgprPersistentIteration] // Set start iter
s_cselect_b32 s[sgprPersistentIterationEnd], s14, s[sgprPersistentIterationEnd] // Set end iter
s_branch label_SK_AssignItersDone_1                // skip per-tile path
label_SK_PerTileExtraIters_1:
s_mov_b32 s12, s13                                 // F = skGrid / skTiles
v_cvt_f32_u32 v0, s12                              // q = w/F, s = w%F
v_rcp_iflag_f32 v0, v0                             // q = w/F, s = w%F
v_mov_b32 v1, 0x4f7ffffe                           // 2^32 - 512
v_mul_f32 v0, v1, v0                               // 2^32 / divisor
v_cvt_u32_f32 v0, v0                               // q = w/F, s = w%F
v_mul_lo_u32 v1, v0, s12                           // refine 2^32 / divisor
v_sub_nc_u32 v1, 0, v1                             // refine 2^32 / divisor
v_mul_hi_u32 v1, v0, v1                            // refine 2^32 / divisor
v_add_nc_u32 v0, v0, v1                            // refine 2^32 / divisor
v_mul_hi_u32 v0, s[sgprPersistentWorkGroupIndex], v0 // q = w/F, s = w%F
v_mul_lo_u32 v1, v0, s12                           // q = w/F, s = w%F
v_sub_nc_u32 v1, s[sgprPersistentWorkGroupIndex], v1 // q = w/F, s = w%F
v_cmp_ge_u32 vcc_lo, v1, s12                       // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v0, 1, v0                             // quotient + 1
v_sub_nc_u32 v1, v1, s12                           // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v1, s12                       // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v0, 1, v0                             // quotient + 1
v_sub_nc_u32 v1, v1, s12                           // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s13, v0                        // quotient
v_readfirstlane_b32 s14, v1                        // remainder
s_mul_i32 s[sgprPersistentIteration], s13, s[sgprItersPerTile] // q * ItersPerTile
s_mul_i32 s13, s14, s[sgprSKItersPerWG]            // s * SKItersPerWG
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s13 // q*I + s*W
s_mul_i32 s13, s12, s[sgprSKItersPerWG]            // F * SKItersPerWG
s_sub_u32 s13, s[sgprItersPerTile], s13            // remI = ItersPerTile - F*SKItersPerWG
s_min_u32 s12, s14, s13                            // min(s, remI)
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s12 // start = q*I + s*W + min(s, remI)
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIteration], s[sgprSKItersPerWG] // start + SKItersPerWG
s_cmp_lt_u32 s14, s13                              // s < remI?
s_cselect_b32 s12, 1, 0                            // extra iter within tile
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], s12 // end = start + W + (s < remI)
label_SK_AssignItersDone_1:
s_mov_b32 s12, s17                                 // restore current tile idx
s_add_u32 s13, s[sgprPersistentIteration], s16     // Offset to start of SK section
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], s16 // Offset to start of SK section
s_mul_i32 s16, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1] // totalTiles = nwg0 * nwg1
s_mul_i32 s16, s16, s[sgprSizesFree+2]             // totalTiles *= batch dim 0
s_mul_i32 s16, s16, s[sgprItersPerTile]            // totalIters = totalTiles * itersPerTile
s_min_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], s16 // Cap ending iter at total SK iters
s_cmp_lt_u32 s[sgprPersistentIteration], s16       // Make sure there's work to do
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
s_mov_b32 s12, s[sgprWGM]                          // Restore WGM
s_sext_i32_i16 s12, s12                            // Restore WGM
s_cmp_gt_i32 s12, 1                                // WGM > 1 ?
s_cbranch_scc1 label_WGMPositive                   // branch if WGM > 1
s_cmp_ge_i32 s12, 0                                // WGM >= 0 ?
s_cbranch_scc1 label_WGM                           // branch if WGM >= 0
s_abs_i32 s12, s12                                 // abs(WGM)
v_cvt_f64_u32 v[0:1], s12                          // s13 = s[sgprWorkGroup0] / s12
v_rcp_f64 v[0:1], v[0:1]                           // s13 = s[sgprWorkGroup0] / s12
v_cvt_f64_u32 v[2:3], s[sgprWorkGroup0]            // s13 = s[sgprWorkGroup0] / s12
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s13 = s[sgprWorkGroup0] / s12
v_cvt_u32_f64 v0, v[0:1]                           // s13 = s[sgprWorkGroup0] / s12
v_mul_lo_u32 v1, v0, s12                           // s13 = s[sgprWorkGroup0] / s12
v_sub_nc_u32 v2, s[sgprWorkGroup0], v1             // s13 = s[sgprWorkGroup0] / s12
v_cmp_ge_u32 vcc_lo, v2, s12                       // s13 = s[sgprWorkGroup0] / s12
s_mov_b32 exec_lo, vcc_lo                          // s13 = s[sgprWorkGroup0] / s12
v_add_nc_u32 v0, v0, 1                             // s13 = s[sgprWorkGroup0] / s12
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s13, v0                        // quotient
s_mul_i32 s16, s13, s12                            // quotient * non-magic divisor
s_sub_u32 s16, s[sgprWorkGroup0], s16              // WorkGroup0=remainder
s_mul_i32 s16, s16, s[sgprNumWorkGroups1]          // (wg1 % WGM)*NumWorkGroups1
s_add_u32 s16, s16, s[sgprWorkGroup1]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups1
v_cvt_f64_u32 v[0:1], s12                          // s14 = s[sgprNumWorkGroups0] / s12
v_rcp_f64 v[0:1], v[0:1]                           // s14 = s[sgprNumWorkGroups0] / s12
v_cvt_f64_u32 v[2:3], s[sgprNumWorkGroups0]        // s14 = s[sgprNumWorkGroups0] / s12
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s14 = s[sgprNumWorkGroups0] / s12
v_cvt_u32_f64 v0, v[0:1]                           // s14 = s[sgprNumWorkGroups0] / s12
v_mul_lo_u32 v1, v0, s12                           // s14 = s[sgprNumWorkGroups0] / s12
v_sub_nc_u32 v2, s[sgprNumWorkGroups0], v1         // s14 = s[sgprNumWorkGroups0] / s12
v_cmp_ge_u32 vcc_lo, v2, s12                       // s14 = s[sgprNumWorkGroups0] / s12
s_mov_b32 exec_lo, vcc_lo                          // s14 = s[sgprNumWorkGroups0] / s12
v_add_nc_u32 v0, v0, 1                             // s14 = s[sgprNumWorkGroups0] / s12
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s14, v0                        // quotient
s_mul_i32 s15, s12, s14                            // quotient * non-magic divisor
s_sub_u32 s15, s[sgprNumWorkGroups0], s15          // NumWorkGroups0=remainder
s_cmp_eq_u32 s15, 0                                // remainder == 0 ?
s_cmov_b32 s15, s12                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s13, s14                              // blockId >= numFullBlocks ?
s_cselect_b32 s14, s15, s12
v_cvt_f64_u32 v[0:1], s14                          // s[sgprWorkGroup1] = s16 / s14
v_rcp_f64 v[0:1], v[0:1]                           // s[sgprWorkGroup1] = s16 / s14
v_cvt_f64_u32 v[2:3], s16                          // s[sgprWorkGroup1] = s16 / s14
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s[sgprWorkGroup1] = s16 / s14
v_cvt_u32_f64 v0, v[0:1]                           // s[sgprWorkGroup1] = s16 / s14
v_mul_lo_u32 v1, v0, s14                           // s[sgprWorkGroup1] = s16 / s14
v_sub_nc_u32 v2, s16, v1                           // s[sgprWorkGroup1] = s16 / s14
v_cmp_ge_u32 vcc_lo, v2, s14                       // s[sgprWorkGroup1] = s16 / s14
s_mov_b32 exec_lo, vcc_lo                          // s[sgprWorkGroup1] = s16 / s14
v_add_nc_u32 v0, v0, 1                             // s[sgprWorkGroup1] = s16 / s14
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v1, v0, s14                           // s[sgprWorkGroup1] = s16 / s14
v_sub_nc_u32 v2, s16, v1                           // s[sgprWorkGroup1] = s16 / s14
v_readfirstlane_b32 s[sgprWorkGroup1], v0          // quotient
v_readfirstlane_b32 s[sgprWorkGroup0], v2          // remainder
s_mul_i32 s[sgprWorkGroup0], s[sgprWorkGroup1], s14 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup0], s16, s[sgprWorkGroup0] // WorkGroup0=remainder
s_mul_i32 s13, s13, s12                            // blockId * WGM
s_add_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s13 // wg1 += blockId * WGM
s_branch label_WGM
label_WGMPositive:
s_mov_b32 s12, s12                                 // WGM
v_cvt_f64_u32 v[0:1], s12                          // s13 = s[sgprWorkGroup1] / s12
v_rcp_f64 v[0:1], v[0:1]                           // s13 = s[sgprWorkGroup1] / s12
v_cvt_f64_u32 v[2:3], s[sgprWorkGroup1]            // s13 = s[sgprWorkGroup1] / s12
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s13 = s[sgprWorkGroup1] / s12
v_cvt_u32_f64 v0, v[0:1]                           // s13 = s[sgprWorkGroup1] / s12
v_mul_lo_u32 v1, v0, s12                           // s13 = s[sgprWorkGroup1] / s12
v_sub_nc_u32 v2, s[sgprWorkGroup1], v1             // s13 = s[sgprWorkGroup1] / s12
v_cmp_ge_u32 vcc_lo, v2, s12                       // s13 = s[sgprWorkGroup1] / s12
s_mov_b32 exec_lo, vcc_lo                          // s13 = s[sgprWorkGroup1] / s12
v_add_nc_u32 v0, v0, 1                             // s13 = s[sgprWorkGroup1] / s12
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s13, v0                        // quotient
s_mul_i32 s16, s13, s12                            // quotient * non-magic divisor
s_sub_u32 s16, s[sgprWorkGroup1], s16              // WorkGroup1=remainder
s_mul_i32 s16, s16, s[sgprNumWorkGroups0]          // (wg1 % WGM)*NumWorkGroups0
s_add_u32 s16, s16, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups0
v_cvt_f64_u32 v[0:1], s12                          // s14 = s[sgprNumWorkGroups1] / s12
v_rcp_f64 v[0:1], v[0:1]                           // s14 = s[sgprNumWorkGroups1] / s12
v_cvt_f64_u32 v[2:3], s[sgprNumWorkGroups1]        // s14 = s[sgprNumWorkGroups1] / s12
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s14 = s[sgprNumWorkGroups1] / s12
v_cvt_u32_f64 v0, v[0:1]                           // s14 = s[sgprNumWorkGroups1] / s12
v_mul_lo_u32 v1, v0, s12                           // s14 = s[sgprNumWorkGroups1] / s12
v_sub_nc_u32 v2, s[sgprNumWorkGroups1], v1         // s14 = s[sgprNumWorkGroups1] / s12
v_cmp_ge_u32 vcc_lo, v2, s12                       // s14 = s[sgprNumWorkGroups1] / s12
s_mov_b32 exec_lo, vcc_lo                          // s14 = s[sgprNumWorkGroups1] / s12
v_add_nc_u32 v0, v0, 1                             // s14 = s[sgprNumWorkGroups1] / s12
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s14, v0                        // quotient
s_mul_i32 s15, s12, s14                            // quotient * non-magic divisor
s_sub_u32 s15, s[sgprNumWorkGroups1], s15          // NumWorkGroups1=remainder
s_cmp_eq_u32 s15, 0                                // remainder == 0 ?
s_cmov_b32 s15, s12                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s13, s14                              // blockId >= numFullBlocks ?
s_cselect_b32 s14, s15, s12
v_cvt_f64_u32 v[0:1], s14                          // s[sgprWorkGroup0] = s16 / s14
v_rcp_f64 v[0:1], v[0:1]                           // s[sgprWorkGroup0] = s16 / s14
v_cvt_f64_u32 v[2:3], s16                          // s[sgprWorkGroup0] = s16 / s14
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s[sgprWorkGroup0] = s16 / s14
v_cvt_u32_f64 v0, v[0:1]                           // s[sgprWorkGroup0] = s16 / s14
v_mul_lo_u32 v1, v0, s14                           // s[sgprWorkGroup0] = s16 / s14
v_sub_nc_u32 v2, s16, v1                           // s[sgprWorkGroup0] = s16 / s14
v_cmp_ge_u32 vcc_lo, v2, s14                       // s[sgprWorkGroup0] = s16 / s14
s_mov_b32 exec_lo, vcc_lo                          // s[sgprWorkGroup0] = s16 / s14
v_add_nc_u32 v0, v0, 1                             // s[sgprWorkGroup0] = s16 / s14
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v1, v0, s14                           // s[sgprWorkGroup0] = s16 / s14
v_sub_nc_u32 v2, s16, v1                           // s[sgprWorkGroup0] = s16 / s14
v_readfirstlane_b32 s[sgprWorkGroup0], v0          // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v2          // remainder
s_mul_i32 s[sgprWorkGroup1], s[sgprWorkGroup0], s14 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup1], s16, s[sgprWorkGroup1] // WorkGroup1=remainder
s_mul_i32 s13, s13, s12                            // blockId * WGM
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s13 // wg1 += blockId * WGM
label_WGM:

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
/* lr1J */
v_and_b32 v2, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v1, 6, v1                            // 1. N offset: nOffset = nIdx * nStride(64)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v3, 5, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(32)
v_and_b32 v3, 1, v3                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v1, v3, 10, v1                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(1024); 7. final local read offset: flrOffset = lrOffset + WOffset

/* local read addresses: final offsets a */
v_lshrrev_b32 v2, 5, v[vgprSerial]                 // 2 = Serial / 32
v_lshrrev_b32 v2, 1, v2                            // LSU offset: Get LSU wave_id
s_mov_b32 s12, 32                                  // LSU offset: stride = lsuStride(32) when umlds==True
v_mul_lo_u32 v2, s12, v2                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_nc_u32 v[vgprLocalReadAddrA], v2, v0         // Final Offset: offset = (lro0+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrA], 1, v[vgprLocalReadAddrA] //  (multiple bpe)
v_lshrrev_b32 v3, 7, v[vgprLocalReadAddrA]         // Final Offset: padding 16 per block 128
v_lshl_add_u32 v[vgprLocalReadAddrA], v3, 4, v[vgprLocalReadAddrA] // Final Offset: padding 16 per block 128

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 5, v[vgprSerial]                 // 0 = Serial / 32
v_lshrrev_b32 v0, 1, v0                            // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(32) when umlds==True (dup assign opt.)
v_mul_lo_u32 v0, s12, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_nc_u32 v[vgprLocalReadAddrB], v0, v1         // Final Offset: offset = (lro1+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrB], 1, v[vgprLocalReadAddrB] //  (multiple bpe)
v_lshrrev_b32 v2, 7, v[vgprLocalReadAddrB]         // Final Offset: padding 16 per block 128
v_lshl_add_u32 v[vgprLocalReadAddrB], v2, 4, v[vgprLocalReadAddrB] // Final Offset: padding 16 per block 128

/* local read addresses: declare addresses a */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, 0x1200, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

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
v_add_co_u32 v[vgprLocalWriteAddrB], vcc_lo, 0x1200, v[vgprLocalWriteAddrB] // lwFOB = lw1J + lwL*MT1J + LDS_OFFSET_B=4608

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

/* global read addresses: tile offsets b */
v_mov_b32 v8, v2                                   // groB1J_0
v_add_co_u32 v9, vcc_lo, 16, v8                    // groB1J_1 += LSPB
v_add_co_u32 v10, vcc_lo, 16, v9                   // groB1J_2 += LSPB
v_add_co_u32 v11, vcc_lo, 16, v10                  // groB1J_3 += LSPB
v_add_co_u32 v12, vcc_lo, 16, v11                  // groB1J_4 += LSPB
v_add_co_u32 v13, vcc_lo, 16, v12                  // groB1J_5 += LSPB
v_add_co_u32 v14, vcc_lo, 16, v13                  // groB1J_6 += LSPB
v_add_co_u32 v15, vcc_lo, 16, v14                  // groB1J_7 += LSPB

/* global read addresses: unroll offsets a */
v_mov_b32 v16, v1                                  // groAL_0

/* global read addresses: unroll offsets b */
v_mov_b32 v17, v3                                  // groBL_0

/* global read addresses: addresses a */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s15, s[sgprWorkGroup0], 32            // WorkGroup[01] * MT
s_mul_i32 s14, s[sgprWorkGroup0], 32               // WorkGroup[01] * MT
s_mul_hi_u32 s15, s14, s[sgprStrideA0I]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s14, s14, s[sgprStrideA0I]               // tlu=0, scaled tile-offset by stride
s_mul_i32 s12, s[sgprStreamKLocalStart], 64        // StreamK tile start offset
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
s_and_b32 s16, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s16, 3                                // ArgType == 3 for General Batched GEMM
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
s_lshr_b32 s[sgprScaleAKMask], s[sgprScaleAGMinus1], 6
s_max_u32 s[sgprScaleAKMask], s[sgprScaleAKMask], 1
s_sub_u32 s[sgprScaleAKMask], s[sgprScaleAKMask], 1
s_mov_b32 s[sgprScaleAInc], 64
s_lshr_b32 s[sgprScaleAInc], s[sgprScaleAInc], s[sgprScaleAGShift]
s_max_u32 s[sgprScaleAInc], s[sgprScaleAInc], 1
s_lshl_b32 s[sgprScaleAInc], s[sgprScaleAInc], 1
s_mov_b32 s[sgprScaleAKCnt], 0
s_add_u32 s[sgprStrideScaleA], s[sgprSizeL], s[sgprScaleAGMinus1] // SizeL + G
s_sub_u32 s[sgprStrideScaleA], s[sgprStrideScaleA], 1 // SizeL + G-1
s_lshr_b32 s[sgprStrideScaleA], s[sgprStrideScaleA], s[sgprScaleAGShift] // StrideScaleA = ceil(SizeL/32) scale elements
s_mul_i32 s12, s[sgprWorkGroup0], 32               // scaleA: workgroup row origin
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
s_mul_i32 s12, s[sgprStreamKLocalStart], 64        // scaleA: first K of this range
s_lshr_b32 s12, s12, s[sgprScaleAGShift]           // scaleA: groups before this range
s_mul_i32 s13, s12, 2                              // scaleA: skipped bytes
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], s13 // scaleA: skip groups before the split (lower)
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA: skip groups before the split (upper)
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], s13 // scaleA: limit shrinks by the skip

/* global read addresses: addresses b */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s15, s[sgprWorkGroup1], 128           // WorkGroup[01] * MT
s_mul_i32 s14, s[sgprWorkGroup1], 128              // WorkGroup[01] * MT
s_mul_hi_u32 s15, s14, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s14, s14, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_mul_i32 s12, s[sgprStreamKLocalStart], 64        // StreamK tile start offset
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
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_and_b32 s16, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s16, 3                                // ArgType == 3 for General Batched GEMM
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
s_sub_u32 s[sgprSrdB+0], s[sgprSrdB+0], 16         // pre-pad to make room for possible pointer shift
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
v_mul_lo_u32 v18, s[sgprStrideA0I], v[6]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+0+0], vcc_lo, v[16], v[18+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+0+0], 0x8, v[vgprGlobalReadOffsetA+0+0] // add prepad for pointer shift
v_lshrrev_b32 v18, s[sgprScaleAGShift], v16        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+0], s[sgprStrideScaleA], v6 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+0], v18, v[vgprGlobalReadOffsetScaleA+0] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+0], 1, v[vgprGlobalReadOffsetScaleA+0] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetA+0], 1, v[vgprGlobalReadOffsetA+0] //  (multiple bpe)
v_mul_lo_u32 v18, s[sgprStrideA0I], v[7]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+1+0], vcc_lo, v[16], v[18+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+1+0], 0x8, v[vgprGlobalReadOffsetA+1+0] // add prepad for pointer shift
v_lshrrev_b32 v18, s[sgprScaleAGShift], v16        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+1], s[sgprStrideScaleA], v7 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+1], v18, v[vgprGlobalReadOffsetScaleA+1] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+1], 1, v[vgprGlobalReadOffsetScaleA+1] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetA+1], 1, v[vgprGlobalReadOffsetA+1] //  (multiple bpe)
/* ============================================================= */

/* global read addresses: final offsets b */
/* ============================================================= */
v_mul_lo_u32 v18, s[sgprStrideB1J], v[8]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+0+0], vcc_lo, v[17], v[18+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+0+0], 0x8, v[vgprGlobalReadOffsetB+0+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+0], 1, v[vgprGlobalReadOffsetB+0] //  (multiple bpe)
v_mul_lo_u32 v18, s[sgprStrideB1J], v[9]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+1+0], vcc_lo, v[17], v[18+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+1+0], 0x8, v[vgprGlobalReadOffsetB+1+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+1], 1, v[vgprGlobalReadOffsetB+1] //  (multiple bpe)
v_mul_lo_u32 v18, s[sgprStrideB1J], v[10]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+2+0], vcc_lo, v[17], v[18+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+2+0], 0x8, v[vgprGlobalReadOffsetB+2+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+2], 1, v[vgprGlobalReadOffsetB+2] //  (multiple bpe)
v_mul_lo_u32 v18, s[sgprStrideB1J], v[11]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+3+0], vcc_lo, v[17], v[18+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+3+0], 0x8, v[vgprGlobalReadOffsetB+3+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+3], 1, v[vgprGlobalReadOffsetB+3] //  (multiple bpe)
v_mul_lo_u32 v18, s[sgprStrideB1J], v[12]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+4+0], vcc_lo, v[17], v[18+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+4+0], 0x8, v[vgprGlobalReadOffsetB+4+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+4], 1, v[vgprGlobalReadOffsetB+4] //  (multiple bpe)
v_mul_lo_u32 v18, s[sgprStrideB1J], v[13]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+5+0], vcc_lo, v[17], v[18+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+5+0], 0x8, v[vgprGlobalReadOffsetB+5+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+5], 1, v[vgprGlobalReadOffsetB+5] //  (multiple bpe)
v_mul_lo_u32 v18, s[sgprStrideB1J], v[14]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+6+0], vcc_lo, v[17], v[18+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+6+0], 0x8, v[vgprGlobalReadOffsetB+6+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+6], 1, v[vgprGlobalReadOffsetB+6] //  (multiple bpe)
v_mul_lo_u32 v18, s[sgprStrideB1J], v[15]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+7+0], vcc_lo, v[17], v[18+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+7+0], 0x8, v[vgprGlobalReadOffsetB+7+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+7], 1, v[vgprGlobalReadOffsetB+7] //  (multiple bpe)
/* ============================================================= */

/* global read addresses: increments a */
.set GlobalReadIncsA, 32

/* global read addresses: increments b */
.set GlobalReadIncsB, 128
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
s_setprio 0                                        // optimization store
s_cbranch_scc1 label_ShadowInitStart               // skip to ShadowInitStart iter b/c numIter==0
buffer_load_b32 v[vgprG2LA+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b32 v[vgprG2LA+6], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_1_0

/* global read block-scale A */
buffer_load_d16_b16 v[vgprG2LScaleA+0], v[vgprGlobalReadOffsetScaleA+0], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 0
buffer_load_d16_b16 v[vgprG2LScaleA+1], v[vgprGlobalReadOffsetScaleA+1], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 1
buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b128 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b128 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b128 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0
buffer_load_b128 v[vgprG2LB+16:vgprG2LB+16+3], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_4_0
buffer_load_b128 v[vgprG2LB+20:vgprG2LB+20+3], v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_5_0
buffer_load_b128 v[vgprG2LB+24:vgprG2LB+24+3], v[vgprGlobalReadOffsetB+6], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_6_0
buffer_load_b128 v[vgprG2LB+28:vgprG2LB+28+3], v[vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_7_0

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], GlobalReadIncsA // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], GlobalReadIncsA // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

/* global read inc block-scale A (2 bytes every 1 iters) */
s_add_u32 s[sgprScaleAKCnt], s[sgprScaleAKCnt], 1  // scaleA: one more K iteration done
s_and_b32 s12, s[sgprScaleAKCnt], s[sgprScaleAKMask] // scaleA: SCC = counter has not wrapped
s_cselect_b32 s12, 0, s[sgprScaleAInc]             // scaleA: advance only on a wrap
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], s12 // scaleA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], s12 // scaleA limit -= inc

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
s_and_b32 s52, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s52, 3                                // ArgType == 3 for General Batched GEMM
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

s_and_b32 s52, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s52, 3                                // ArgType == 3 for General Batched GEMM
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

s_mov_b32 s52, 1
s_mov_b32 s53, 1
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_BPEDone                       // If synchronizer, use regular output BPE
s_cmp_eq_u32 s[sgprskTiles], 1                     // split == 1 ?
s_cbranch_scc1 label_BPEDone                       // If split == 1, use reguler output BPE
s_mov_b32 s52, 1
s_mov_b32 s53, 2
label_BPEDone:

s_mul_i32 s90, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_mul_hi_u32 s89, s90, s[sgprStrideC1J]            // ScaleC s90 by Stride
s_mul_i32 s88, s90, s[sgprStrideC1J]               // ScaleC s90 by Stride
s_lshl_b64 s[88:89], s[88:89], s52                 // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s88        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s89       // add hi to SRD
s_mul_hi_u32 s89, s90, s[sgprStrideD1J]            // ScaleD s90 by Stride
s_mul_i32 s88, s90, s[sgprStrideD1J]               // ScaleD s90 by Stride
s_lshl_b64 s[88:89], s[88:89], s53                 // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s88        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s89       // add hi to SRD

s_and_b32 s54, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s54, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_StridedBatchedGemmLoadC
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_GeneralBatchedGemmLoadC
label_StridedBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s89, s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride
s_mul_i32 s88, s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride
s_lshl_b64 s[88:89], s[88:89], s52                 // scale by bpe
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
s_and_b32 s54, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s54, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_StridedBatchedGemmLoadD
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_GeneralBatchedGemmLoadD
label_StridedBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s89, s[sgprWorkGroup2], s[sgprStrideDK] // ScaleD s[sgprWorkGroup2] by Stride
s_mul_i32 s88, s[sgprWorkGroup2], s[sgprStrideDK]  // ScaleD s[sgprWorkGroup2] by Stride
s_lshl_b64 s[88:89], s[88:89], s53                 // scale by bpe
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

/* initC: remove ValuC vgpr buffer [0...64) from pool */

/* initC: remove acc vgpr buffer [0...0) from pool */

/* initC: remove ValuA/B vgpr buffer [82...179) from pool */
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
v_pack_b32_f16 v224, v[vgprG2LScaleA+0], v[vgprG2LScaleA+0] // scaleA: duplicate fp16 scale

/* w4a16: save packed int4 dword 0 */
v_perm_b32 v227, v[vgprG2LA+2+0], v[vgprG2LA+2+0], 0xc000c00 // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v228, v[vgprG2LA+2+0], v[vgprG2LA+2+0], 0xc010c01 // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v229, v[vgprG2LA+2+0], v[vgprG2LA+2+0], 0xc020c02 // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v230, v[vgprG2LA+2+0], v[vgprG2LA+2+0], 0xc030c03 // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v227, v227, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v228, v228, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v229, v229, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v230, v230, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v227, v227, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v228, v228, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v229, v229, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v230, v230, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_mul_f16 v[vgprG2LA+0+0], v227, v224           // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_mul_f16 v[vgprG2LA+0+1], v228, v224           // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_mul_f16 v[vgprG2LA+0+2], v229, v224           // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_mul_f16 v[vgprG2LA+0+3], v230, v224           // w4a16: (q - z)*s, packed, fp16 pair 3
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
v_pack_b32_f16 v224, v[vgprG2LScaleA+1], v[vgprG2LScaleA+1] // scaleA: duplicate fp16 scale

/* w4a16: save packed int4 dword 0 */
v_perm_b32 v227, v[vgprG2LA+6+0], v[vgprG2LA+6+0], 0xc000c00 // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v228, v[vgprG2LA+6+0], v[vgprG2LA+6+0], 0xc010c01 // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v229, v[vgprG2LA+6+0], v[vgprG2LA+6+0], 0xc020c02 // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v230, v[vgprG2LA+6+0], v[vgprG2LA+6+0], 0xc030c03 // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v227, v227, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v228, v228, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v229, v229, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v230, v230, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v227, v227, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v228, v228, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v229, v229, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v230, v230, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_mul_f16 v[vgprG2LA+4+0], v227, v224           // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_mul_f16 v[vgprG2LA+4+1], v228, v224           // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_mul_f16 v[vgprG2LA+4+2], v229, v224           // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_mul_f16 v[vgprG2LA+4+3], v230, v224           // w4a16: (q - z)*s, packed, fp16 pair 3
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:2304 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 2304 sync LDS0

/* local write b */
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4:vgprG2LB+4+3] offset:2304 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 2304 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8:vgprG2LB+8+3] offset:4608 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 4608 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12:vgprG2LB+12+3] offset:6912 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 6912 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+16:vgprG2LB+16+3] offset:9216 // lwoB_0_0_4_0 = (0*LSCB)*(MT1J+PAD) + (4*LSPB) = 9216 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+20:vgprG2LB+20+3] offset:11520 // lwoB_0_0_5_0 = (0*LSCB)*(MT1J+PAD) + (5*LSPB) = 11520 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+24:vgprG2LB+24+3] offset:13824 // lwoB_0_0_6_0 = (0*LSCB)*(MT1J+PAD) + (6*LSPB) = 13824 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+28:vgprG2LB+28+3] offset:16128 // lwoB_0_0_7_0 = (0*LSCB)*(MT1J+PAD) + (7*LSPB) = 16128 sync LDS0

/* local write swap a */

/* local write swap b */
s_cmp_eq_u32 s[sgprLoopCounterL], 0x1              // PGR=2 but only 1 loop
s_cbranch_scc1 label_skipPGR2_1                    // PGR=2 but only 1 loop
buffer_load_b32 v[vgprG2LA+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b32 v[vgprG2LA+6], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_1_0

/* global read block-scale A */
buffer_load_d16_b16 v[vgprG2LScaleA+0], v[vgprGlobalReadOffsetScaleA+0], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 0
buffer_load_d16_b16 v[vgprG2LScaleA+1], v[vgprGlobalReadOffsetScaleA+1], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 1
buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b128 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b128 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b128 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0
buffer_load_b128 v[vgprG2LB+16:vgprG2LB+16+3], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_4_0
buffer_load_b128 v[vgprG2LB+20:vgprG2LB+20+3], v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_5_0
buffer_load_b128 v[vgprG2LB+24:vgprG2LB+24+3], v[vgprGlobalReadOffsetB+6], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_6_0
buffer_load_b128 v[vgprG2LB+28:vgprG2LB+28+3], v[vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_7_0
s_branch label_skipPGR2_2                          // jump to PGR=2 label
label_skipPGR2_1:
label_skipPGR2_2:
s_waitcnt lgkmcnt(0)                               // 0prefetch wait for local write
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier                                          // LW to PLR, sync

/* local read prefetch a */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:2304 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:2320 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read prefetch b */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4608 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4624 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:9216 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:9232 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:13824 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:13840 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read inc a */
/* N/A, lro->16 */
/* localReadDoCntA 1 localReadDoCntMXSA 0 localReadDoCntB 1 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read inc b */
/* N/A, lro->16 */
/* localReadDoCntA 1 localReadDoCntMXSA 0 localReadDoCntB 1 localReadDoCntMXSB 0 localReadDoCntM 0 */

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

/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */

/* iter 0 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:5, lwStartMfmaIndex:5, lwEndMfmaIndex:5  */
/*  numMfmaForLR:7, syncPlrMfmaIndex:8 , sync1LdsMfmaIndex:4 */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6 for iteration == 0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:1  */
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], GlobalReadIncsA // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], GlobalReadIncsA // limit -= inc)
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:2  */
ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:3  */
ds_load_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA+0] offset:2336 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA+0] offset:2352 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0

/* global read inc block-scale A (2 bytes every 1 iters) */
s_add_u32 s[sgprScaleAKCnt], s[sgprScaleAKCnt], 1  // scaleA: one more K iteration done
s_and_b32 s52, s[sgprScaleAKCnt], s[sgprScaleAKMask] // scaleA: SCC = counter has not wrapped
s_cselect_b32 s52, 0, s[sgprScaleAInc]             // scaleA: advance only on a wrap
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:4  */
/* schedule remaining localreads for one buffer scheduling */
ds_load_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB+0] offset:4640 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[vgprLocalReadAddrB+0] offset:4656 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprLocalReadAddrB+0] offset:9248 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+20:vgprValuB_X1_I0+20+3], v[vgprLocalReadAddrB+0] offset:9264 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprLocalReadAddrB+0] offset:13856 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+28:vgprValuB_X1_I0+28+3], v[vgprLocalReadAddrB+0] offset:13872 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], s52 // scaleA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], s52 // scaleA limit -= inc
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:5  */

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], GlobalReadIncsB // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], GlobalReadIncsB // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32
s_setprio 3                                        // store optimization
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(9)                                 // wait for global read before writing to local
v_pack_b32_f16 v224, v[vgprG2LScaleA+0], v[vgprG2LScaleA+0] // scaleA: duplicate fp16 scale

/* w4a16: save packed int4 dword 0 */
v_perm_b32 v227, v[vgprG2LA+2+0], v[vgprG2LA+2+0], 0xc000c00 // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v228, v[vgprG2LA+2+0], v[vgprG2LA+2+0], 0xc010c01 // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v229, v[vgprG2LA+2+0], v[vgprG2LA+2+0], 0xc020c02 // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v230, v[vgprG2LA+2+0], v[vgprG2LA+2+0], 0xc030c03 // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v227, v227, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v228, v228, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v229, v229, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v230, v230, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v227, v227, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v228, v228, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v229, v229, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v230, v230, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_mul_f16 v[vgprG2LA+0+0], v227, v224           // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_mul_f16 v[vgprG2LA+0+1], v228, v224           // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_mul_f16 v[vgprG2LA+0+2], v229, v224           // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_mul_f16 v[vgprG2LA+0+3], v230, v224           // w4a16: (q - z)*s, packed, fp16 pair 3
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
buffer_load_b32 v[vgprG2LA+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(9)                                 // wait for global read before writing to local
v_pack_b32_f16 v224, v[vgprG2LScaleA+1], v[vgprG2LScaleA+1] // scaleA: duplicate fp16 scale

/* w4a16: save packed int4 dword 0 */
v_perm_b32 v227, v[vgprG2LA+6+0], v[vgprG2LA+6+0], 0xc000c00 // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v228, v[vgprG2LA+6+0], v[vgprG2LA+6+0], 0xc010c01 // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v229, v[vgprG2LA+6+0], v[vgprG2LA+6+0], 0xc020c02 // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v230, v[vgprG2LA+6+0], v[vgprG2LA+6+0], 0xc030c03 // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v227, v227, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v228, v228, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v229, v229, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v230, v230, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v227, v227, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v228, v228, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v229, v229, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v230, v230, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_mul_f16 v[vgprG2LA+4+0], v227, v224           // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_mul_f16 v[vgprG2LA+4+1], v228, v224           // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_mul_f16 v[vgprG2LA+4+2], v229, v224           // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_mul_f16 v[vgprG2LA+4+3], v230, v224           // w4a16: (q - z)*s, packed, fp16 pair 3
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:2304 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 2304 sync LDS0
buffer_load_b32 v[vgprG2LA+6], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_1_0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(9)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0

/* global read block-scale A */
buffer_load_d16_b16 v[vgprG2LScaleA+0], v[vgprGlobalReadOffsetScaleA+0], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 0
buffer_load_d16_b16 v[vgprG2LScaleA+1], v[vgprGlobalReadOffsetScaleA+1], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 1
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(10)                                // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4:vgprG2LB+4+3] offset:2304 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 2304 sync LDS0
buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(10)                                // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8:vgprG2LB+8+3] offset:4608 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 4608 sync LDS0
buffer_load_b128 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(10)                                // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12:vgprG2LB+12+3] offset:6912 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 6912 sync LDS0
buffer_load_b128 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(10)                                // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+16:vgprG2LB+16+3] offset:9216 // lwoB_0_0_4_0 = (0*LSCB)*(MT1J+PAD) + (4*LSPB) = 9216 sync LDS0
buffer_load_b128 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(10)                                // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+20:vgprG2LB+20+3] offset:11520 // lwoB_0_0_5_0 = (0*LSCB)*(MT1J+PAD) + (5*LSPB) = 11520 sync LDS0
buffer_load_b128 v[vgprG2LB+16:vgprG2LB+16+3], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_4_0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(10)                                // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+24:vgprG2LB+24+3] offset:13824 // lwoB_0_0_6_0 = (0*LSCB)*(MT1J+PAD) + (6*LSPB) = 13824 sync LDS0
buffer_load_b128 v[vgprG2LB+20:vgprG2LB+20+3], v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_5_0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(10)                                // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+28:vgprG2LB+28+3] offset:16128 // lwoB_0_0_7_0 = (0*LSCB)*(MT1J+PAD) + (7*LSPB) = 16128 sync LDS0
buffer_load_b128 v[vgprG2LB+24:vgprG2LB+24+3], v[vgprGlobalReadOffsetB+6], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_6_0
buffer_load_b128 v[vgprG2LB+28:vgprG2LB+28+3], v[vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_7_0

/* local write swap offsets a */

/* local write swap offsets b */
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:6  */
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:7  */

/* local read swap offsets a */

/* local read swap offsets b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_wmma_f32_16x16x16_f16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
s_setprio 0                                        // store optimization
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=4 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=8 */

/* iter 1 */
/*  grEndMfmaIndex:5, lwStartMfmaIndex:5, lwEndMfmaIndex:5  */
/*  numMfmaForLR:7, syncPlrMfmaIndex:8 , sync1LdsMfmaIndex:4 */
/*  mfmaIndex:8  */
s_waitcnt lgkmcnt(0)                               // 3wait for local write
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier                                          // PGR, and wait until LW done to sync
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:9  */
s_setprio 3                                        // store optimization
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:10  */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:11  */
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:2304 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:2320 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:12  */
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4608 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4624 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X1_I0+16+0+0:vgprValuB_X1_I0+16+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:13  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:9216 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:9232 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X1_I0+16+0+0:vgprValuB_X1_I0+16+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:14  */
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:13824 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X1_I0+24+0+0:vgprValuB_X1_I0+24+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:15  */
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:13840 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_f16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X1_I0+24+0+0:vgprValuB_X1_I0+24+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
s_setprio 0                                        // store optimization

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

/* iter 0 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:5, lwStartMfmaIndex:5, lwEndMfmaIndex:5  */
/*  numMfmaForLR:7, syncPlrMfmaIndex:8 , sync1LdsMfmaIndex:4 */
/*  mfmaIndex:0  */

/* Global Read IncA */

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], GlobalReadIncsA // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], GlobalReadIncsA // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

/* global read inc block-scale A (2 bytes every 1 iters) */
s_add_u32 s[sgprScaleAKCnt], s[sgprScaleAKCnt], 1  // scaleA: one more K iteration done
s_and_b32 s52, s[sgprScaleAKCnt], s[sgprScaleAKMask] // scaleA: SCC = counter has not wrapped
s_cselect_b32 s52, 0, s[sgprScaleAInc]             // scaleA: advance only on a wrap
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], s52 // scaleA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], s52 // scaleA limit -= inc

/* Global Read IncB */

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], GlobalReadIncsB // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], GlobalReadIncsB // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6 for iteration == 0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:1  */
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:2  */
ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:3  */
ds_load_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA+0] offset:2336 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA+0] offset:2352 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:4  */
/* schedule remaining localreads for one buffer scheduling */
ds_load_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB+0] offset:4640 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[vgprLocalReadAddrB+0] offset:4656 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprLocalReadAddrB+0] offset:9248 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+20:vgprValuB_X1_I0+20+3], v[vgprLocalReadAddrB+0] offset:9264 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprLocalReadAddrB+0] offset:13856 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+28:vgprValuB_X1_I0+28+3], v[vgprLocalReadAddrB+0] offset:13872 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:5  */
s_setprio 3                                        // store optimization
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(9)                                 // wait for global read before writing to local
v_pack_b32_f16 v224, v[vgprG2LScaleA+0], v[vgprG2LScaleA+0] // scaleA: duplicate fp16 scale

/* w4a16: save packed int4 dword 0 */
v_perm_b32 v227, v[vgprG2LA+2+0], v[vgprG2LA+2+0], 0xc000c00 // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v228, v[vgprG2LA+2+0], v[vgprG2LA+2+0], 0xc010c01 // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v229, v[vgprG2LA+2+0], v[vgprG2LA+2+0], 0xc020c02 // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v230, v[vgprG2LA+2+0], v[vgprG2LA+2+0], 0xc030c03 // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v227, v227, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v228, v228, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v229, v229, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v230, v230, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v227, v227, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v228, v228, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v229, v229, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v230, v230, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_mul_f16 v[vgprG2LA+0+0], v227, v224           // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_mul_f16 v[vgprG2LA+0+1], v228, v224           // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_mul_f16 v[vgprG2LA+0+2], v229, v224           // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_mul_f16 v[vgprG2LA+0+3], v230, v224           // w4a16: (q - z)*s, packed, fp16 pair 3
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(8)                                 // wait for global read before writing to local
v_pack_b32_f16 v224, v[vgprG2LScaleA+1], v[vgprG2LScaleA+1] // scaleA: duplicate fp16 scale

/* w4a16: save packed int4 dword 0 */
v_perm_b32 v227, v[vgprG2LA+6+0], v[vgprG2LA+6+0], 0xc000c00 // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v228, v[vgprG2LA+6+0], v[vgprG2LA+6+0], 0xc010c01 // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v229, v[vgprG2LA+6+0], v[vgprG2LA+6+0], 0xc020c02 // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v230, v[vgprG2LA+6+0], v[vgprG2LA+6+0], 0xc030c03 // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v227, v227, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v228, v228, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v229, v229, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v230, v230, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v227, v227, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v228, v228, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v229, v229, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v230, v230, s[sgprScaleAPkMagic+1]    // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_mul_f16 v[vgprG2LA+4+0], v227, v224           // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_mul_f16 v[vgprG2LA+4+1], v228, v224           // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_mul_f16 v[vgprG2LA+4+2], v229, v224           // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_mul_f16 v[vgprG2LA+4+3], v230, v224           // w4a16: (q - z)*s, packed, fp16 pair 3
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:2304 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 2304 sync LDS0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(6)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4:vgprG2LB+4+3] offset:2304 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 2304 sync LDS0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(5)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8:vgprG2LB+8+3] offset:4608 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 4608 sync LDS0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(4)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12:vgprG2LB+12+3] offset:6912 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 6912 sync LDS0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(3)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+16:vgprG2LB+16+3] offset:9216 // lwoB_0_0_4_0 = (0*LSCB)*(MT1J+PAD) + (4*LSPB) = 9216 sync LDS0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(2)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+20:vgprG2LB+20+3] offset:11520 // lwoB_0_0_5_0 = (0*LSCB)*(MT1J+PAD) + (5*LSPB) = 11520 sync LDS0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(1)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+24:vgprG2LB+24+3] offset:13824 // lwoB_0_0_6_0 = (0*LSCB)*(MT1J+PAD) + (6*LSPB) = 13824 sync LDS0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(0)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+28:vgprG2LB+28+3] offset:16128 // lwoB_0_0_7_0 = (0*LSCB)*(MT1J+PAD) + (7*LSPB) = 16128 sync LDS0

/* local write swap offsets a */

/* local write swap offsets b */
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:6  */
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:7  */

/* local read swap offsets a */

/* local read swap offsets b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_wmma_f32_16x16x16_f16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
s_setprio 0                                        // store optimization
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=4 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=8 */

/* iter 1 */
/*  grEndMfmaIndex:5, lwStartMfmaIndex:5, lwEndMfmaIndex:5  */
/*  numMfmaForLR:7, syncPlrMfmaIndex:8 , sync1LdsMfmaIndex:4 */
/*  mfmaIndex:8  */
s_waitcnt lgkmcnt(0)                               // 3wait for local write
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier                                          // noLoadLoop sync
s_waitcnt lgkmcnt(10)                              // wait for prior local read local write old=0, new=10 newLW=10 newLR=0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:9  */
s_setprio 3                                        // store optimization
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:10  */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:11  */
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:2304 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:2320 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:12  */
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4608 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4624 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X1_I0+16+0+0:vgprValuB_X1_I0+16+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:13  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:9216 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:9232 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X1_I0+16+0+0:vgprValuB_X1_I0+16+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:14  */
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:13824 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X1_I0+24+0+0:vgprValuB_X1_I0+24+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:15  */
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:13840 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_f16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X1_I0+24+0+0:vgprValuB_X1_I0+24+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
s_setprio 0                                        // store optimization
/* numPrefetchIter=1 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=4 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=8 */
label_toPGR1:

/******************************************/
/* Ord. NoLoadLoop - Begin                */
/******************************************/

/* iter 0 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:5, lwEndMfmaIndex:5  */
/*  numMfmaForLR:7, syncPlrMfmaIndex:8 , sync1LdsMfmaIndex:4 */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6 for iteration == 0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:1  */
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:2  */
ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:3  */
ds_load_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA+0] offset:2336 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA+0] offset:2352 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:4  */
/* schedule remaining localreads for one buffer scheduling */
ds_load_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB+0] offset:4640 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[vgprLocalReadAddrB+0] offset:4656 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprLocalReadAddrB+0] offset:9248 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+20:vgprValuB_X1_I0+20+3], v[vgprLocalReadAddrB+0] offset:9264 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprLocalReadAddrB+0] offset:13856 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+28:vgprValuB_X1_I0+28+3], v[vgprLocalReadAddrB+0] offset:13872 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:5  */
s_setprio 3                                        // store optimization
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:6  */
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:7  */
v_wmma_f32_16x16x16_f16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
s_setprio 0                                        // store optimization
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=4 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=8 */

/* iter 1 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:5, lwEndMfmaIndex:5  */
/*  numMfmaForLR:7, syncPlrMfmaIndex:8 , sync1LdsMfmaIndex:4 */
/*  mfmaIndex:8  */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:9  */
s_setprio 3                                        // store optimization
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:10  */
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:11  */
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:12  */
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X1_I0+16+0+0:vgprValuB_X1_I0+16+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:13  */
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X1_I0+16+0+0:vgprValuB_X1_I0+16+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:14  */
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X1_I0+24+0+0:vgprValuB_X1_I0+24+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:15  */
v_wmma_f32_16x16x16_f16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X1_I0+24+0+0:vgprValuB_X1_I0+24+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=8 */
label_toPGR1end_OrdNLL:
label_PrefetchGlobalLastIterEnd:

/* Tail: add ValuA/B vgpr buffer [82...179) to pool */

/* Tail: add address/G2L vgpr [179...220) to pool */
label_Summation_End_8:
s_setprio 0                                        // optimization store
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

/* LocalSplitU: local write and read */

/******************************************/
/* LocalSplitU Reduction                  */
/******************************************/
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier
label_localSplitULocalWriteAndRead:
/* computeStoreVgprs */
v_lshrrev_b32 v88, 5, v[vgprSerial]                // 88 = Serial / 32
v_lshrrev_b32 v89, 0, v88                          // 89 = 88 / 1
v_and_b32 v89, 1, v89                              // v89 = v89 % 2
v_mul_lo_u32 v89, 0x10, v89                        // wave coordination offset 1
v_and_b32 v83, 31, v[vgprSerial]                   // v83 = v[vgprSerial] % 32
v_lshrrev_b32 v83, 4, v83                          // 83 = 83 / 16
                                                   // thread0 * continuous_output (multiplier is 1, do nothing)
v_add_lshl_u32 v87, v89, v83, 0                    // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v84, v87, s[sgprStrideC1J]            //  offset 1
v_mul_lo_u32 v85, v87, s[sgprStrideD1J]            //  offset 1
v_and_b32 v82, 0, v88                              // v82 = v88 % 1
v_mul_lo_u32 v82, 0x10, v82                        // wave coordination offset 0
v_and_b32 v89, 15, v[vgprSerial]                   // v89 = v[vgprSerial] % 16
v_add_lshl_u32 v86, v89, v82, 0                    // coordination 0 = vwA * (wave_id0 + tid0)
s_mul_i32 s8, 32, s[sgprWorkGroup0]                // wgp0 * MT0
v_add_nc_u32 v82, s8, v86                          // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s8, 128, s[sgprWorkGroup1]               // wgp1 * MT1
v_add_nc_u32 v83, s8, v87                          // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1
.set vgprLsuReduction, 88
/* Size of vgprLsuReduction is 32 */
v_lshrrev_b32 v121, 5, v[vgprSerial]               // 121 = Serial / 32
v_lshrrev_b32 v120, 1, v121                        // Get LSU wave ID
v_and_b32 v121, 0x1, v121                          // Get wave ID

/* LocalSplitU: local write 1/1 */
label_localSplitULocalWriteAndRead_1:
s_mov_b32 s8, 0x2000                               // dataPerWave (8192)
v_and_b32 v125, 0x1f, v[vgprSerial]                // initial addr
v_mul_lo_u32 v122, s8, v121                        // tmp = waveId * dataPerWave
v_lshl_add_u32 v125, v125, 2, v122                 // addr = initial addr + tmp
s_mov_b32 s8, 0x4000                               // ldsStride = waveNum * dataPerWave (16384)
v_mul_lo_u32 v122, s8, v120                        // tmp = (waveNum * dataPerWave) * lsu_id
v_add_nc_u32 v125, v122, v125                      // addr += tmp
v_mov_b32 v124, v[vgprValuC+0]                     // copy acc[0] to vreg[124], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:0                   // arch[0]
v_mov_b32 v124, v[vgprValuC+1]                     // copy acc[1] to vreg[124], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:128                 // arch[1]
v_mov_b32 v124, v[vgprValuC+2]                     // copy acc[2] to vreg[124], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:256                 // arch[2]
v_mov_b32 v124, v[vgprValuC+3]                     // copy acc[3] to vreg[124], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:384                 // arch[3]
v_mov_b32 v124, v[vgprValuC+4]                     // copy acc[4] to vreg[124], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:512                 // arch[4]
v_mov_b32 v124, v[vgprValuC+5]                     // copy acc[5] to vreg[124], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:640                 // arch[5]
v_mov_b32 v124, v[vgprValuC+6]                     // copy acc[6] to vreg[124], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:768                 // arch[6]
v_mov_b32 v124, v[vgprValuC+7]                     // copy acc[7] to vreg[124], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:896                 // arch[7]
v_mov_b32 v124, v[vgprValuC+16]                    // copy acc[16] to vreg[124], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:1024                // arch[8]
v_mov_b32 v124, v[vgprValuC+17]                    // copy acc[17] to vreg[124], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:1152                // arch[9]
v_mov_b32 v124, v[vgprValuC+18]                    // copy acc[18] to vreg[124], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:1280                // arch[10]
v_mov_b32 v124, v[vgprValuC+19]                    // copy acc[19] to vreg[124], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:1408                // arch[11]
v_mov_b32 v124, v[vgprValuC+20]                    // copy acc[20] to vreg[124], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:1536                // arch[12]
v_mov_b32 v124, v[vgprValuC+21]                    // copy acc[21] to vreg[124], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:1664                // arch[13]
v_mov_b32 v124, v[vgprValuC+22]                    // copy acc[22] to vreg[124], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:1792                // arch[14]
v_mov_b32 v124, v[vgprValuC+23]                    // copy acc[23] to vreg[124], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:1920                // arch[15]
v_mov_b32 v124, v[vgprValuC+32]                    // copy acc[32] to vreg[124], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:2048                // arch[16]
v_mov_b32 v124, v[vgprValuC+33]                    // copy acc[33] to vreg[124], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:2176                // arch[17]
v_mov_b32 v124, v[vgprValuC+34]                    // copy acc[34] to vreg[124], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:2304                // arch[18]
v_mov_b32 v124, v[vgprValuC+35]                    // copy acc[35] to vreg[124], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:2432                // arch[19]
v_mov_b32 v124, v[vgprValuC+36]                    // copy acc[36] to vreg[124], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:2560                // arch[20]
v_mov_b32 v124, v[vgprValuC+37]                    // copy acc[37] to vreg[124], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:2688                // arch[21]
v_mov_b32 v124, v[vgprValuC+38]                    // copy acc[38] to vreg[124], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:2816                // arch[22]
v_mov_b32 v124, v[vgprValuC+39]                    // copy acc[39] to vreg[124], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:2944                // arch[23]
v_mov_b32 v124, v[vgprValuC+48]                    // copy acc[48] to vreg[124], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:3072                // arch[24]
v_mov_b32 v124, v[vgprValuC+49]                    // copy acc[49] to vreg[124], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:3200                // arch[25]
v_mov_b32 v124, v[vgprValuC+50]                    // copy acc[50] to vreg[124], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:3328                // arch[26]
v_mov_b32 v124, v[vgprValuC+51]                    // copy acc[51] to vreg[124], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:3456                // arch[27]
v_mov_b32 v124, v[vgprValuC+52]                    // copy acc[52] to vreg[124], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:3584                // arch[28]
v_mov_b32 v124, v[vgprValuC+53]                    // copy acc[53] to vreg[124], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:3712                // arch[29]
v_mov_b32 v124, v[vgprValuC+54]                    // copy acc[54] to vreg[124], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:3840                // arch[30]
v_mov_b32 v124, v[vgprValuC+55]                    // copy acc[55] to vreg[124], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:3968                // arch[31]
v_mov_b32 v124, v[vgprValuC+8]                     // copy acc[8] to vreg[124], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:4096                // arch[32]
v_mov_b32 v124, v[vgprValuC+9]                     // copy acc[9] to vreg[124], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:4224                // arch[33]
v_mov_b32 v124, v[vgprValuC+10]                    // copy acc[10] to vreg[124], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:4352                // arch[34]
v_mov_b32 v124, v[vgprValuC+11]                    // copy acc[11] to vreg[124], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:4480                // arch[35]
v_mov_b32 v124, v[vgprValuC+12]                    // copy acc[12] to vreg[124], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:4608                // arch[36]
v_mov_b32 v124, v[vgprValuC+13]                    // copy acc[13] to vreg[124], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:4736                // arch[37]
v_mov_b32 v124, v[vgprValuC+14]                    // copy acc[14] to vreg[124], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:4864                // arch[38]
v_mov_b32 v124, v[vgprValuC+15]                    // copy acc[15] to vreg[124], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:4992                // arch[39]
v_mov_b32 v124, v[vgprValuC+24]                    // copy acc[24] to vreg[124], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:5120                // arch[40]
v_mov_b32 v124, v[vgprValuC+25]                    // copy acc[25] to vreg[124], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:5248                // arch[41]
v_mov_b32 v124, v[vgprValuC+26]                    // copy acc[26] to vreg[124], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:5376                // arch[42]
v_mov_b32 v124, v[vgprValuC+27]                    // copy acc[27] to vreg[124], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:5504                // arch[43]
v_mov_b32 v124, v[vgprValuC+28]                    // copy acc[28] to vreg[124], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:5632                // arch[44]
v_mov_b32 v124, v[vgprValuC+29]                    // copy acc[29] to vreg[124], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:5760                // arch[45]
v_mov_b32 v124, v[vgprValuC+30]                    // copy acc[30] to vreg[124], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:5888                // arch[46]
v_mov_b32 v124, v[vgprValuC+31]                    // copy acc[31] to vreg[124], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:6016                // arch[47]
v_mov_b32 v124, v[vgprValuC+40]                    // copy acc[40] to vreg[124], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:6144                // arch[48]
v_mov_b32 v124, v[vgprValuC+41]                    // copy acc[41] to vreg[124], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:6272                // arch[49]
v_mov_b32 v124, v[vgprValuC+42]                    // copy acc[42] to vreg[124], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:6400                // arch[50]
v_mov_b32 v124, v[vgprValuC+43]                    // copy acc[43] to vreg[124], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:6528                // arch[51]
v_mov_b32 v124, v[vgprValuC+44]                    // copy acc[44] to vreg[124], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:6656                // arch[52]
v_mov_b32 v124, v[vgprValuC+45]                    // copy acc[45] to vreg[124], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:6784                // arch[53]
v_mov_b32 v124, v[vgprValuC+46]                    // copy acc[46] to vreg[124], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:6912                // arch[54]
v_mov_b32 v124, v[vgprValuC+47]                    // copy acc[47] to vreg[124], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:7040                // arch[55]
v_mov_b32 v124, v[vgprValuC+56]                    // copy acc[56] to vreg[124], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:7168                // arch[56]
v_mov_b32 v124, v[vgprValuC+57]                    // copy acc[57] to vreg[124], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:7296                // arch[57]
v_mov_b32 v124, v[vgprValuC+58]                    // copy acc[58] to vreg[124], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:7424                // arch[58]
v_mov_b32 v124, v[vgprValuC+59]                    // copy acc[59] to vreg[124], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:7552                // arch[59]
v_mov_b32 v124, v[vgprValuC+60]                    // copy acc[60] to vreg[124], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:7680                // arch[60]
v_mov_b32 v124, v[vgprValuC+61]                    // copy acc[61] to vreg[124], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:7808                // arch[61]
v_mov_b32 v124, v[vgprValuC+62]                    // copy acc[62] to vreg[124], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:7936                // arch[62]
v_mov_b32 v124, v[vgprValuC+63]                    // copy acc[63] to vreg[124], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v125, v124 offset:8064                // arch[63]

/* LocalSplitU: local read 1/1 */
v_and_b32 v125, 0x1f, v[vgprSerial]                // initial addr
s_mov_b32 s8, 0x2000                               // wave offset (8192)
v_mul_lo_u32 v122, s8, v121                        // wave offset = wave_id * wave offset
v_lshl_add_u32 v125, v125, 2, v122                 // addr = initial addr + wave offset
s_mov_b32 s8, 0x1000                               // LSU Process Offset 4096
v_mul_lo_u32 v122, s8, v120                        // lsu offset = lsu_id * LSU Process Offset
v_add_nc_u32 v125, v125, v122                      // addr += lsu offset
s_waitcnt lgkmcnt(0)                               // wait for all writes
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier                                          // post-lsu local write
label_localSplitULocalRead_1:
ds_load_b32 v[vgprLsuReduction+0], v125 offset:0   // r=0 i=0, from acc[0]
ds_load_b32 v128, v125 offset:16384                // r=1 i=0, from acc[0]
ds_load_b32 v[vgprLsuReduction+1], v125 offset:128 // r=0 i=1, from acc[1]
ds_load_b32 v129, v125 offset:16512                // r=1 i=1, from acc[1]
ds_load_b32 v[vgprLsuReduction+2], v125 offset:256 // r=0 i=2, from acc[2]
ds_load_b32 v130, v125 offset:16640                // r=1 i=2, from acc[2]
ds_load_b32 v[vgprLsuReduction+3], v125 offset:384 // r=0 i=3, from acc[3]
ds_load_b32 v131, v125 offset:16768                // r=1 i=3, from acc[3]
ds_load_b32 v[vgprLsuReduction+4], v125 offset:512 // r=0 i=4, from acc[4]
ds_load_b32 v132, v125 offset:16896                // r=1 i=4, from acc[4]
ds_load_b32 v[vgprLsuReduction+5], v125 offset:640 // r=0 i=5, from acc[5]
ds_load_b32 v133, v125 offset:17024                // r=1 i=5, from acc[5]
ds_load_b32 v[vgprLsuReduction+6], v125 offset:768 // r=0 i=6, from acc[6]
ds_load_b32 v134, v125 offset:17152                // r=1 i=6, from acc[6]
ds_load_b32 v[vgprLsuReduction+7], v125 offset:896 // r=0 i=7, from acc[7]
ds_load_b32 v135, v125 offset:17280                // r=1 i=7, from acc[7]
ds_load_b32 v[vgprLsuReduction+8], v125 offset:1024 // r=0 i=8, from acc[16]
ds_load_b32 v136, v125 offset:17408                // r=1 i=8, from acc[16]
ds_load_b32 v[vgprLsuReduction+9], v125 offset:1152 // r=0 i=9, from acc[17]
ds_load_b32 v137, v125 offset:17536                // r=1 i=9, from acc[17]
ds_load_b32 v[vgprLsuReduction+10], v125 offset:1280 // r=0 i=10, from acc[18]
ds_load_b32 v138, v125 offset:17664                // r=1 i=10, from acc[18]
ds_load_b32 v[vgprLsuReduction+11], v125 offset:1408 // r=0 i=11, from acc[19]
ds_load_b32 v139, v125 offset:17792                // r=1 i=11, from acc[19]
ds_load_b32 v[vgprLsuReduction+12], v125 offset:1536 // r=0 i=12, from acc[20]
ds_load_b32 v140, v125 offset:17920                // r=1 i=12, from acc[20]
ds_load_b32 v[vgprLsuReduction+13], v125 offset:1664 // r=0 i=13, from acc[21]
ds_load_b32 v141, v125 offset:18048                // r=1 i=13, from acc[21]
ds_load_b32 v[vgprLsuReduction+14], v125 offset:1792 // r=0 i=14, from acc[22]
ds_load_b32 v142, v125 offset:18176                // r=1 i=14, from acc[22]
ds_load_b32 v[vgprLsuReduction+15], v125 offset:1920 // r=0 i=15, from acc[23]
ds_load_b32 v143, v125 offset:18304                // r=1 i=15, from acc[23]
ds_load_b32 v[vgprLsuReduction+16], v125 offset:2048 // r=0 i=16, from acc[32]
ds_load_b32 v144, v125 offset:18432                // r=1 i=16, from acc[32]
ds_load_b32 v[vgprLsuReduction+17], v125 offset:2176 // r=0 i=17, from acc[33]
ds_load_b32 v145, v125 offset:18560                // r=1 i=17, from acc[33]
ds_load_b32 v[vgprLsuReduction+18], v125 offset:2304 // r=0 i=18, from acc[34]
ds_load_b32 v146, v125 offset:18688                // r=1 i=18, from acc[34]
ds_load_b32 v[vgprLsuReduction+19], v125 offset:2432 // r=0 i=19, from acc[35]
ds_load_b32 v147, v125 offset:18816                // r=1 i=19, from acc[35]
ds_load_b32 v[vgprLsuReduction+20], v125 offset:2560 // r=0 i=20, from acc[36]
ds_load_b32 v148, v125 offset:18944                // r=1 i=20, from acc[36]
ds_load_b32 v[vgprLsuReduction+21], v125 offset:2688 // r=0 i=21, from acc[37]
ds_load_b32 v149, v125 offset:19072                // r=1 i=21, from acc[37]
ds_load_b32 v[vgprLsuReduction+22], v125 offset:2816 // r=0 i=22, from acc[38]
ds_load_b32 v150, v125 offset:19200                // r=1 i=22, from acc[38]
ds_load_b32 v[vgprLsuReduction+23], v125 offset:2944 // r=0 i=23, from acc[39]
ds_load_b32 v151, v125 offset:19328                // r=1 i=23, from acc[39]
ds_load_b32 v[vgprLsuReduction+24], v125 offset:3072 // r=0 i=24, from acc[48]
ds_load_b32 v152, v125 offset:19456                // r=1 i=24, from acc[48]
ds_load_b32 v[vgprLsuReduction+25], v125 offset:3200 // r=0 i=25, from acc[49]
ds_load_b32 v153, v125 offset:19584                // r=1 i=25, from acc[49]
ds_load_b32 v[vgprLsuReduction+26], v125 offset:3328 // r=0 i=26, from acc[50]
ds_load_b32 v154, v125 offset:19712                // r=1 i=26, from acc[50]
ds_load_b32 v[vgprLsuReduction+27], v125 offset:3456 // r=0 i=27, from acc[51]
ds_load_b32 v155, v125 offset:19840                // r=1 i=27, from acc[51]
ds_load_b32 v[vgprLsuReduction+28], v125 offset:3584 // r=0 i=28, from acc[52]
ds_load_b32 v156, v125 offset:19968                // r=1 i=28, from acc[52]
ds_load_b32 v[vgprLsuReduction+29], v125 offset:3712 // r=0 i=29, from acc[53]
ds_load_b32 v157, v125 offset:20096                // r=1 i=29, from acc[53]
ds_load_b32 v[vgprLsuReduction+30], v125 offset:3840 // r=0 i=30, from acc[54]
ds_load_b32 v158, v125 offset:20224                // r=1 i=30, from acc[54]
ds_load_b32 v[vgprLsuReduction+31], v125 offset:3968 // r=0 i=31, from acc[55]
ds_load_b32 v159, v125 offset:20352                // r=1 i=31, from acc[55]
s_waitcnt lgkmcnt(62)                              // wait count is (64-2)
v_add_f32 v[vgprLsuReduction+0], v128, v[vgprLsuReduction+0]
s_waitcnt lgkmcnt(60)                              // wait count is (64-4)
v_add_f32 v[vgprLsuReduction+1], v129, v[vgprLsuReduction+1]
s_waitcnt lgkmcnt(58)                              // wait count is (64-6)
v_add_f32 v[vgprLsuReduction+2], v130, v[vgprLsuReduction+2]
s_waitcnt lgkmcnt(56)                              // wait count is (64-8)
v_add_f32 v[vgprLsuReduction+3], v131, v[vgprLsuReduction+3]
s_waitcnt lgkmcnt(54)                              // wait count is (64-10)
v_add_f32 v[vgprLsuReduction+4], v132, v[vgprLsuReduction+4]
s_waitcnt lgkmcnt(52)                              // wait count is (64-12)
v_add_f32 v[vgprLsuReduction+5], v133, v[vgprLsuReduction+5]
s_waitcnt lgkmcnt(50)                              // wait count is (64-14)
v_add_f32 v[vgprLsuReduction+6], v134, v[vgprLsuReduction+6]
s_waitcnt lgkmcnt(48)                              // wait count is (64-16)
v_add_f32 v[vgprLsuReduction+7], v135, v[vgprLsuReduction+7]
s_waitcnt lgkmcnt(46)                              // wait count is (64-18)
v_add_f32 v[vgprLsuReduction+8], v136, v[vgprLsuReduction+8]
s_waitcnt lgkmcnt(44)                              // wait count is (64-20)
v_add_f32 v[vgprLsuReduction+9], v137, v[vgprLsuReduction+9]
s_waitcnt lgkmcnt(42)                              // wait count is (64-22)
v_add_f32 v[vgprLsuReduction+10], v138, v[vgprLsuReduction+10]
s_waitcnt lgkmcnt(40)                              // wait count is (64-24)
v_add_f32 v[vgprLsuReduction+11], v139, v[vgprLsuReduction+11]
s_waitcnt lgkmcnt(38)                              // wait count is (64-26)
v_add_f32 v[vgprLsuReduction+12], v140, v[vgprLsuReduction+12]
s_waitcnt lgkmcnt(36)                              // wait count is (64-28)
v_add_f32 v[vgprLsuReduction+13], v141, v[vgprLsuReduction+13]
s_waitcnt lgkmcnt(34)                              // wait count is (64-30)
v_add_f32 v[vgprLsuReduction+14], v142, v[vgprLsuReduction+14]
s_waitcnt lgkmcnt(32)                              // wait count is (64-32)
v_add_f32 v[vgprLsuReduction+15], v143, v[vgprLsuReduction+15]
s_waitcnt lgkmcnt(30)                              // wait count is (64-34)
v_add_f32 v[vgprLsuReduction+16], v144, v[vgprLsuReduction+16]
s_waitcnt lgkmcnt(28)                              // wait count is (64-36)
v_add_f32 v[vgprLsuReduction+17], v145, v[vgprLsuReduction+17]
s_waitcnt lgkmcnt(26)                              // wait count is (64-38)
v_add_f32 v[vgprLsuReduction+18], v146, v[vgprLsuReduction+18]
s_waitcnt lgkmcnt(24)                              // wait count is (64-40)
v_add_f32 v[vgprLsuReduction+19], v147, v[vgprLsuReduction+19]
s_waitcnt lgkmcnt(22)                              // wait count is (64-42)
v_add_f32 v[vgprLsuReduction+20], v148, v[vgprLsuReduction+20]
s_waitcnt lgkmcnt(20)                              // wait count is (64-44)
v_add_f32 v[vgprLsuReduction+21], v149, v[vgprLsuReduction+21]
s_waitcnt lgkmcnt(18)                              // wait count is (64-46)
v_add_f32 v[vgprLsuReduction+22], v150, v[vgprLsuReduction+22]
s_waitcnt lgkmcnt(16)                              // wait count is (64-48)
v_add_f32 v[vgprLsuReduction+23], v151, v[vgprLsuReduction+23]
s_waitcnt lgkmcnt(14)                              // wait count is (64-50)
v_add_f32 v[vgprLsuReduction+24], v152, v[vgprLsuReduction+24]
s_waitcnt lgkmcnt(12)                              // wait count is (64-52)
v_add_f32 v[vgprLsuReduction+25], v153, v[vgprLsuReduction+25]
s_waitcnt lgkmcnt(10)                              // wait count is (64-54)
v_add_f32 v[vgprLsuReduction+26], v154, v[vgprLsuReduction+26]
s_waitcnt lgkmcnt(8)                               // wait count is (64-56)
v_add_f32 v[vgprLsuReduction+27], v155, v[vgprLsuReduction+27]
s_waitcnt lgkmcnt(6)                               // wait count is (64-58)
v_add_f32 v[vgprLsuReduction+28], v156, v[vgprLsuReduction+28]
s_waitcnt lgkmcnt(4)                               // wait count is (64-60)
v_add_f32 v[vgprLsuReduction+29], v157, v[vgprLsuReduction+29]
s_waitcnt lgkmcnt(2)                               // wait count is (64-62)
v_add_f32 v[vgprLsuReduction+30], v158, v[vgprLsuReduction+30]
s_waitcnt lgkmcnt(0)                               // wait count is (64-64)
v_add_f32 v[vgprLsuReduction+31], v159, v[vgprLsuReduction+31]
.set vgprValuC, 88

/* LocalSplitU: global write indices */
v_lshrrev_b32 v120, 5, v[vgprSerial]               // 120 = Serial / 32
v_lshrrev_b32 v120, 1, v120                        // 120 = 120 / 2
s_mov_b32 s8, 0x10                                 // a valid offset
v_mul_lo_u32 v122, v120, s8                        // wave LSU offset
v_add_nc_u32 v82, v122, v82                        // coord0 += LSU offset0
/* valid offset coord1 is zero. */

/* LocalSplitU: global write */

/******************************************/
/* Global Write Elements                  */
/******************************************/
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_GSU                           // Branch to stream-k store code
s_cmp_eq_u32 s[sgprskTiles], 1                     // split == 1 ?
s_cbranch_scc1 label_GSU                           // branch if split == 1
label_GW_B0_MB:
label_GW_B0_FD0_MB:

/* Edge/NonEdge store path check (M): Size % 32 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s64, 31, s[sgprSizeI]                    // s64 = s[sgprSizeI] % 32
s_add_u32 s65, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s65                // wg0 >= nwg0-1 ?
s_cselect_b32 s64, s64, 0                          // set rem
s_cmpk_gt_u32 s64, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_MB_Else_1       // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 128 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s64, 127, s[sgprSizeJ]                   // s64 = s[sgprSizeJ] % 128
s_add_u32 s65, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s65                // wg1 >= nwg1-1
s_cselect_b32 s64, s64, 0                          // set rem
s_cmpk_gt_u32 s64, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_MB_Then_1       // jump if edges required
label_GW_B0_FD0_VW1_MB_NonEdge_1:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=128 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (1,0,0,0:vw1); (2,0,0,0:vw1); (3,0,0,0:vw1); (4,0,0,0:vw1); (5,0,0,0:vw1); (6,0,0,0:vw1); (7,0,0,0:vw1); (8,0,0,0:vw1); (9,0,0,0:vw1); (10,0,0,0:vw1); (11,0,0,0:vw1); (12,0,0,0:vw1); (13,0,0,0:vw1); (14,0,0,0:vw1); (15,0,0,0:vw1); (16,0,0,0:vw1); (17,0,0,0:vw1); (18,0,0,0:vw1); (19,0,0,0:vw1); (20,0,0,0:vw1); (21,0,0,0:vw1); (22,0,0,0:vw1); (23,0,0,0:vw1); (24,0,0,0:vw1); (25,0,0,0:vw1); (26,0,0,0:vw1); (27,0,0,0:vw1); (28,0,0,0:vw1); (29,0,0,0:vw1); (30,0,0,0:vw1); (31,0,0,0:vw1) */
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
v_add_lshl_u32 v86, v85, v82, 2                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=82, coord0Vgpr=82 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0), (3, 0, 0, 0), (4, 0, 0, 0), (5, 0, 0, 0), (6, 0, 0, 0), (7, 0, 0, 0), (8, 0, 0, 0), (9, 0, 0, 0), (10, 0, 0, 0), (11, 0, 0, 0), (12, 0, 0, 0), (13, 0, 0, 0), (14, 0, 0, 0), (15, 0, 0, 0), (16, 0, 0, 0), (17, 0, 0, 0), (18, 0, 0, 0), (19, 0, 0, 0), (20, 0, 0, 0), (21, 0, 0, 0), (22, 0, 0, 0), (23, 0, 0, 0), (24, 0, 0, 0), (25, 0, 0, 0), (26, 0, 0, 0), (27, 0, 0, 0), (28, 0, 0, 0), (29, 0, 0, 0), (30, 0, 0, 0), (31, 0, 0, 0)] */

/* apply mask, calc new C and issue writes */
buffer_store_b32 v88, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v89, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v90, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v91, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v92, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v93, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v94, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v95, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 72                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b32 v96, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v97, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v98, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v99, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v100, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v101, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v102, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v103, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 72                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b32 v104, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v105, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v106, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v107, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v108, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v109, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v110, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v111, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 72                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b32 v112, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v113, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v114, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v115, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v116, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v117, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v118, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v119, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_B0_FD0_VW1_MB_NonEdgeEnd_1:
label_GW_B0_FD0_VW1_MB_Else_1:
label_GW_B0_FD0_VW1_MB_Then_1:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=64 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (1,0,0,0:vw1); (2,0,0,0:vw1); (3,0,0,0:vw1); (4,0,0,0:vw1); (5,0,0,0:vw1); (6,0,0,0:vw1); (7,0,0,0:vw1); (8,0,0,0:vw1); (9,0,0,0:vw1); (10,0,0,0:vw1); (11,0,0,0:vw1); (12,0,0,0:vw1); (13,0,0,0:vw1); (14,0,0,0:vw1); (15,0,0,0:vw1); (16,0,0,0:vw1); (17,0,0,0:vw1); (18,0,0,0:vw1); (19,0,0,0:vw1); (20,0,0,0:vw1); (21,0,0,0:vw1); (22,0,0,0:vw1); (23,0,0,0:vw1); (24,0,0,0:vw1); (25,0,0,0:vw1); (26,0,0,0:vw1); (27,0,0,0:vw1); (28,0,0,0:vw1); (29,0,0,0:vw1); (30,0,0,0:vw1); (31,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v122, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v86, v85, v82, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v86, v122, v86, s66                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v87, v85, v82, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v87, v122, v87, s66                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v127, v85, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v127, v122, v127, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v128, v85, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v128, v122, v128, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v129, v85, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v129, v122, v129, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v130, v85, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v122, v130, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v131, v85, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v131, v122, v131, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v132, v85, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v122, v132, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 18                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v133, v85, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v133, v122, v133, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v134, v85, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v122, v134, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v135, v85, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v135, v122, v135, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v136, v85, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v122, v136, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v137, v85, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v137, v122, v137, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v138, v85, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v122, v138, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v139, v85, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v139, v122, v139, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v140, v85, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v122, v140, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 18                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v141, v85, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v141, v122, v141, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v142, v85, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v122, v142, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v143, v85, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v143, v122, v143, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v144, v85, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v144, v122, v144, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v145, v85, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v145, v122, v145, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v146, v85, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v146, v122, v146, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v147, v85, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v147, v122, v147, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v148, v85, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v148, v122, v148, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 18                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v149, v85, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v149, v122, v149, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v150, v85, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v150, v122, v150, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v151, v85, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v151, v122, v151, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v152, v85, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v152, v122, v152, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v153, v85, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v153, v122, v153, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v154, v85, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v122, v154, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v155, v85, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v155, v122, v155, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v156, v85, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v122, v156, s66                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0), (3, 0, 0, 0), (4, 0, 0, 0), (5, 0, 0, 0), (6, 0, 0, 0), (7, 0, 0, 0), (8, 0, 0, 0), (9, 0, 0, 0), (10, 0, 0, 0), (11, 0, 0, 0), (12, 0, 0, 0), (13, 0, 0, 0), (14, 0, 0, 0), (15, 0, 0, 0), (16, 0, 0, 0), (17, 0, 0, 0), (18, 0, 0, 0), (19, 0, 0, 0), (20, 0, 0, 0), (21, 0, 0, 0), (22, 0, 0, 0), (23, 0, 0, 0), (24, 0, 0, 0), (25, 0, 0, 0), (26, 0, 0, 0), (27, 0, 0, 0), (28, 0, 0, 0), (29, 0, 0, 0), (30, 0, 0, 0), (31, 0, 0, 0)] */

/* apply mask, calc new C and issue writes */
buffer_store_b32 v88, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v89, v87, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v90, v127, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v91, v128, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v92, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v93, v130, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v94, v131, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v95, v132, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v96, v133, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v97, v134, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v98, v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v99, v136, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v100, v137, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v101, v138, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v102, v139, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v103, v140, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v104, v141, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v105, v142, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v106, v143, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v107, v144, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v108, v145, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v109, v146, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v110, v147, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v111, v148, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v112, v149, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v113, v150, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v114, v151, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v115, v152, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v116, v153, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v117, v154, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v118, v155, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v119, v156, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
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
s_add_u32 s8, s[sgprPersistentWorkGroupIndex], 1   // input partial tile index
s_mul_hi_u32 s65, s[sgprPersistentIterationEnd], s[sgprMagicNumberItersPerTile] // s_magic mul, div alg 2
s_lshr_b32 s66, s[sgprMagicShiftItersPerTile], 31  // tmpS = extract abit
s_mul_i32 s64, s[sgprPersistentIterationEnd], s66  // s_magic mul, div alg 2
s_add_u32 s64, s64, s65
s_and_b32 s66, s[sgprMagicShiftItersPerTile], 2147483647 // tmpS = remove abit to final shift
s_lshr_b32 s64, s64, s66                           // sMagicDiv Alg 2
s_mul_i32 s64, s64, s[sgprItersPerTile]            // start iteration of partial tile
s_sub_u32 s9, s[sgprPersistentIterationEnd], s64   // calc iterations completed by this WG
label_SK_Fixup:
s_lshl_b32 s64, s8, 2                              // flag offset based on physical partial slot
s_load_b32 s66, s[sgprAddressFlags:sgprAddressFlags+1], s64 glc dlc // get flag
s_waitcnt lgkmcnt(0)                               // wait for flag load
s_cmp_eq_u32 s66, 1                                // check if ready
s_cbranch_scc0 label_SK_Fixup                      // if flag not set, wait and check again
s_barrier                                          // wait for all workgroups before resetting flag
v_readfirstlane_b32 s66, v[vgprSerial]             // Wave 0 updates flags
s_cmp_eq_u32 s66, 0                                // Check for wave 0
s_cbranch_scc0 label_SK_SkipFlagReset              // Skip flag reset
v_mov_b32 v86, s66                                 // move flag value to vgpr
v_mov_b32 v87, 0                                   // zero vaddr offset
s_mov_b64 s[72:73], s[sgprAddressFlags:sgprAddressFlags+1]
s_mov_b32 s74, BufferOOB
s_mov_b32 s75, Srd127_96
buffer_store_b32 v86, v87, s[72:75], s64 offen offset:0 glc dlc // reset flag
s_waitcnt_vscnt null 0                             // wait for data store
label_SK_SkipFlagReset:
label_Fixup_E0:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=62 */
s_mov_b64 s[sgprSrdWS:sgprSrdWS+1], s[sgprAddressWS:sgprAddressWS+1]
s_mov_b32 s[sgprSrdWS+2], BufferOOB
s_mov_b32 s[sgprSrdWS+3], Srd127_96

s_mul_i32 s64, 0x4000, s8                          // Offset to correct partials tile (low word)
s_mul_hi_u32 s61, 0x4000, s8                       // partials tile offset (high word) for 64-bit SRD
s_add_u32 s[sgprSrdWS+0], s[sgprSrdWS+0], s64      // add lo to SRD
s_addc_u32 s[sgprSrdWS+1], s[sgprSrdWS+1], s61     // add hi (offset high word + lo carry) to SRD
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Fixup Batch #0 (d1,d0,vc1,vc0) =       */
/*      (0,0,0,0:vw1); (1,0,0,0:vw1); (2,0,0,0:vw1); (3,0,0,0:vw1); (4,0,0,0:vw1); (5,0,0,0:vw1); (6,0,0,0:vw1); (7,0,0,0:vw1); (8,0,0,0:vw1); (9,0,0,0:vw1); (10,0,0,0:vw1); (11,0,0,0:vw1); (12,0,0,0:vw1); (13,0,0,0:vw1); (14,0,0,0:vw1); (15,0,0,0:vw1); (16,0,0,0:vw1); (17,0,0,0:vw1); (18,0,0,0:vw1); (19,0,0,0:vw1); (20,0,0,0:vw1); (21,0,0,0:vw1); (22,0,0,0:vw1); (23,0,0,0:vw1); (24,0,0,0:vw1); (25,0,0,0:vw1); (26,0,0,0:vw1); (27,0,0,0:vw1); (28,0,0,0:vw1); (29,0,0,0:vw1); (30,0,0,0:vw1); (31,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_lshlrev_b32 v87, 2, v[vgprSerial]                // v87 = v[vgprSerial] * 4
s_mov_b32 s64, 0                                   // Init sgpr offset
buffer_load_b32 v127, v87, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v128, v87, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v129, v87, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v130, v87, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v131, v87, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v132, v87, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v133, v87, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v134, v87, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v135, v87, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v136, v87, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v137, v87, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v138, v87, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v139, v87, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v140, v87, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v141, v87, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v142, v87, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v143, v87, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v144, v87, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v145, v87, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v146, v87, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v147, v87, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v148, v87, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v149, v87, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v150, v87, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v151, v87, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v152, v87, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v153, v87, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v154, v87, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v155, v87, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v156, v87, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v157, v87, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 512                            // Inc sgpr offset
buffer_load_b32 v158, v87, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(31)                                // wait C (interleaved) 31 = 32 - 0 - 1
v_add_f32 v[vgprValuC+0], v[vgprValuC+0], v127     // accum partials

s_waitcnt vmcnt(30)                                // wait C (interleaved) 30 = 32 - 1 - 1
v_add_f32 v[vgprValuC+1], v[vgprValuC+1], v128     // accum partials

s_waitcnt vmcnt(29)                                // wait C (interleaved) 29 = 32 - 2 - 1
v_add_f32 v[vgprValuC+2], v[vgprValuC+2], v129     // accum partials

s_waitcnt vmcnt(28)                                // wait C (interleaved) 28 = 32 - 3 - 1
v_add_f32 v[vgprValuC+3], v[vgprValuC+3], v130     // accum partials

s_waitcnt vmcnt(27)                                // wait C (interleaved) 27 = 32 - 4 - 1
v_add_f32 v[vgprValuC+4], v[vgprValuC+4], v131     // accum partials

s_waitcnt vmcnt(26)                                // wait C (interleaved) 26 = 32 - 5 - 1
v_add_f32 v[vgprValuC+5], v[vgprValuC+5], v132     // accum partials

s_waitcnt vmcnt(25)                                // wait C (interleaved) 25 = 32 - 6 - 1
v_add_f32 v[vgprValuC+6], v[vgprValuC+6], v133     // accum partials

s_waitcnt vmcnt(24)                                // wait C (interleaved) 24 = 32 - 7 - 1
v_add_f32 v[vgprValuC+7], v[vgprValuC+7], v134     // accum partials

s_waitcnt vmcnt(23)                                // wait C (interleaved) 23 = 32 - 8 - 1
v_add_f32 v[vgprValuC+8], v[vgprValuC+8], v135     // accum partials

s_waitcnt vmcnt(22)                                // wait C (interleaved) 22 = 32 - 9 - 1
v_add_f32 v[vgprValuC+9], v[vgprValuC+9], v136     // accum partials

s_waitcnt vmcnt(21)                                // wait C (interleaved) 21 = 32 - 10 - 1
v_add_f32 v[vgprValuC+10], v[vgprValuC+10], v137   // accum partials

s_waitcnt vmcnt(20)                                // wait C (interleaved) 20 = 32 - 11 - 1
v_add_f32 v[vgprValuC+11], v[vgprValuC+11], v138   // accum partials

s_waitcnt vmcnt(19)                                // wait C (interleaved) 19 = 32 - 12 - 1
v_add_f32 v[vgprValuC+12], v[vgprValuC+12], v139   // accum partials

s_waitcnt vmcnt(18)                                // wait C (interleaved) 18 = 32 - 13 - 1
v_add_f32 v[vgprValuC+13], v[vgprValuC+13], v140   // accum partials

s_waitcnt vmcnt(17)                                // wait C (interleaved) 17 = 32 - 14 - 1
v_add_f32 v[vgprValuC+14], v[vgprValuC+14], v141   // accum partials

s_waitcnt vmcnt(16)                                // wait C (interleaved) 16 = 32 - 15 - 1
v_add_f32 v[vgprValuC+15], v[vgprValuC+15], v142   // accum partials

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 32 - 16 - 1
v_add_f32 v[vgprValuC+16], v[vgprValuC+16], v143   // accum partials

s_waitcnt vmcnt(14)                                // wait C (interleaved) 14 = 32 - 17 - 1
v_add_f32 v[vgprValuC+17], v[vgprValuC+17], v144   // accum partials

s_waitcnt vmcnt(13)                                // wait C (interleaved) 13 = 32 - 18 - 1
v_add_f32 v[vgprValuC+18], v[vgprValuC+18], v145   // accum partials

s_waitcnt vmcnt(12)                                // wait C (interleaved) 12 = 32 - 19 - 1
v_add_f32 v[vgprValuC+19], v[vgprValuC+19], v146   // accum partials

s_waitcnt vmcnt(11)                                // wait C (interleaved) 11 = 32 - 20 - 1
v_add_f32 v[vgprValuC+20], v[vgprValuC+20], v147   // accum partials

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 32 - 21 - 1
v_add_f32 v[vgprValuC+21], v[vgprValuC+21], v148   // accum partials

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 32 - 22 - 1
v_add_f32 v[vgprValuC+22], v[vgprValuC+22], v149   // accum partials

s_waitcnt vmcnt(8)                                 // wait C (interleaved) 8 = 32 - 23 - 1
v_add_f32 v[vgprValuC+23], v[vgprValuC+23], v150   // accum partials

s_waitcnt vmcnt(7)                                 // wait C (interleaved) 7 = 32 - 24 - 1
v_add_f32 v[vgprValuC+24], v[vgprValuC+24], v151   // accum partials

s_waitcnt vmcnt(6)                                 // wait C (interleaved) 6 = 32 - 25 - 1
v_add_f32 v[vgprValuC+25], v[vgprValuC+25], v152   // accum partials

s_waitcnt vmcnt(5)                                 // wait C (interleaved) 5 = 32 - 26 - 1
v_add_f32 v[vgprValuC+26], v[vgprValuC+26], v153   // accum partials

s_waitcnt vmcnt(4)                                 // wait C (interleaved) 4 = 32 - 27 - 1
v_add_f32 v[vgprValuC+27], v[vgprValuC+27], v154   // accum partials

s_waitcnt vmcnt(3)                                 // wait C (interleaved) 3 = 32 - 28 - 1
v_add_f32 v[vgprValuC+28], v[vgprValuC+28], v155   // accum partials

s_waitcnt vmcnt(2)                                 // wait C (interleaved) 2 = 32 - 29 - 1
v_add_f32 v[vgprValuC+29], v[vgprValuC+29], v156   // accum partials

s_waitcnt vmcnt(1)                                 // wait C (interleaved) 1 = 32 - 30 - 1
v_add_f32 v[vgprValuC+30], v[vgprValuC+30], v157   // accum partials

s_waitcnt vmcnt(0)                                 // wait C (interleaved) 0 = 32 - 31 - 1
v_add_f32 v[vgprValuC+31], v[vgprValuC+31], v158   // accum partials
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_mul_i32 s61, s[sgprskTiles], s[sgprItersPerTile]
s_mul_i32 s64, s[sgprSKItersPerWG], s[sgprskGrid]
s_sub_u32 s61, s61, s64                            // skTiles * ItersPerTile - SKItersPerWG * skGrid
s_bitcmp1_b32 s[sgprMagicShiftItersPerTile], 29    // USO on? (bit 29 of MagicShiftItersPerTile); off -> historical global peer size
s_cbranch_scc0 label_SK_PeerGlobal                 // USO off -> historical global mapping
s_cmp_eq_u32 s[sgprskTiles], 0                     // skTiles == 0?
s_cbranch_scc1 label_SK_PeerNoTiles                // global peer size
v_cvt_f32_u32 v86, s[sgprskTiles]                  // skGrid % skTiles
v_rcp_iflag_f32 v86, v86                           // skGrid % skTiles
v_mov_b32 v87, 0x4f7ffffe                          // 2^32 - 512
v_mul_f32 v86, v87, v86                            // 2^32 / divisor
v_cvt_u32_f32 v86, v86                             // skGrid % skTiles
v_mul_lo_u32 v87, v86, s[sgprskTiles]              // refine 2^32 / divisor
v_sub_nc_u32 v87, 0, v87                           // refine 2^32 / divisor
v_mul_hi_u32 v87, v86, v87                         // refine 2^32 / divisor
v_add_nc_u32 v86, v86, v87                         // refine 2^32 / divisor
v_mul_hi_u32 v86, s[sgprskGrid], v86               // skGrid % skTiles
v_mul_lo_u32 v87, v86, s[sgprskTiles]              // skGrid % skTiles
v_sub_nc_u32 v87, s[sgprskGrid], v87               // skGrid % skTiles
v_cmp_ge_u32 vcc_lo, v87, s[sgprskTiles]           // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v86, 1, v86                           // quotient + 1
v_sub_nc_u32 v87, v87, s[sgprskTiles]              // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v87, s[sgprskTiles]           // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v86, 1, v86                           // quotient + 1
v_sub_nc_u32 v87, v87, s[sgprskTiles]              // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s64, v86                       // quotient
v_readfirstlane_b32 s64, v87                       // remainder
s_cmp_eq_u32 s64, 0                                // skGrid % skTiles == 0?
s_cbranch_scc0 label_SK_PeerGlobal                 // ragged -> global peer
label_SK_PeerPerTile:
v_cvt_f32_u32 v86, s[sgprskTiles]                  // F = skGrid / skTiles
v_rcp_iflag_f32 v86, v86                           // F = skGrid / skTiles
v_mov_b32 v87, 0x4f7ffffe                          // 2^32 - 512
v_mul_f32 v86, v87, v86                            // 2^32 / divisor
v_cvt_u32_f32 v86, v86                             // F = skGrid / skTiles
v_mul_lo_u32 v87, v86, s[sgprskTiles]              // refine 2^32 / divisor
v_sub_nc_u32 v87, 0, v87                           // refine 2^32 / divisor
v_mul_hi_u32 v87, v86, v87                         // refine 2^32 / divisor
v_add_nc_u32 v86, v86, v87                         // refine 2^32 / divisor
v_mul_hi_u32 v86, s[sgprskGrid], v86               // F = skGrid / skTiles
v_mul_lo_u32 v87, v86, s[sgprskTiles]              // F = skGrid / skTiles
v_sub_nc_u32 v87, s[sgprskGrid], v87               // F = skGrid / skTiles
v_cmp_ge_u32 vcc_lo, v87, s[sgprskTiles]           // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v86, 1, v86                           // quotient + 1
v_sub_nc_u32 v87, v87, s[sgprskTiles]              // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v87, s[sgprskTiles]           // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v86, 1, v86                           // quotient + 1
v_sub_nc_u32 v87, v87, s[sgprskTiles]              // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s64, v86                       // quotient
s_mov_b32 s61, s64                                 // F = skGrid / skTiles
v_cvt_f32_u32 v86, s61                             // s = cta % F
v_rcp_iflag_f32 v86, v86                           // s = cta % F
v_mov_b32 v87, 0x4f7ffffe                          // 2^32 - 512
v_mul_f32 v86, v87, v86                            // 2^32 / divisor
v_cvt_u32_f32 v86, v86                             // s = cta % F
v_mul_lo_u32 v87, v86, s61                         // refine 2^32 / divisor
v_sub_nc_u32 v87, 0, v87                           // refine 2^32 / divisor
v_mul_hi_u32 v87, v86, v87                         // refine 2^32 / divisor
v_add_nc_u32 v86, v86, v87                         // refine 2^32 / divisor
v_mul_hi_u32 v86, s8, v86                          // s = cta % F
v_mul_lo_u32 v87, v86, s61                         // s = cta % F
v_sub_nc_u32 v87, s8, v87                          // s = cta % F
v_cmp_ge_u32 vcc_lo, v87, s61                      // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v86, 1, v86                           // quotient + 1
v_sub_nc_u32 v87, v87, s61                         // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v87, s61                      // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v86, 1, v86                           // quotient + 1
v_sub_nc_u32 v87, v87, s61                         // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s64, v86                       // quotient
v_readfirstlane_b32 s64, v87                       // remainder
s_mul_i32 s61, s61, s[sgprSKItersPerWG]            // F * SKItersPerWG
s_sub_u32 s61, s[sgprItersPerTile], s61            // remI = ItersPerTile - F*W
s_cmp_lt_u32 s64, s61                              // s < remI?
s_cselect_b32 s64, 1, 0                            // extra iter within tile
s_add_u32 s64, s[sgprSKItersPerWG], s64            // chunk = W + (s < remI)
s_branch label_SK_PeerDone                         // skip global peer
label_SK_PeerNoTiles:
label_SK_PeerGlobal:
s_add_u32 s64, s[sgprSKItersPerWG], 1              // Add extra iter
s_cmp_lt_u32 s8, s61                               // Check if next WG had an extra iteration
s_cselect_b32 s64, s64, s[sgprSKItersPerWG]        // Select correct number of iterations for next WG
label_SK_PeerDone:
s_add_u32 s9, s9, s64                              // next partial tile iteration
s_add_u32 s8, s8, 1                                // next partial tile index
s_cmp_lt_u32 s9, s[sgprItersPerTile]               // done loading partial tiles?
s_cbranch_scc1 label_SK_Fixup                      // Branch to continue fixup loop
label_SK_Store:
s_cmpk_eq_u32 s[sgprBeta], 0                       // Beta == 0
s_cbranch_scc0 label_GW_B1_GSU1                    // Branch if Beta is not zero

label_GW_B0_GSU1:
label_GW_B0_FD0_GSU1:

/* Edge/NonEdge store path check (M): Size % 32 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s64, 31, s[sgprSizeI]                    // s64 = s[sgprSizeI] % 32
s_add_u32 s65, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s65                // wg0 >= nwg0-1 ?
s_cselect_b32 s64, s64, 0                          // set rem
s_cmpk_gt_u32 s64, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_GSU1_Else_1     // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 128 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s64, 127, s[sgprSizeJ]                   // s64 = s[sgprSizeJ] % 128
s_add_u32 s65, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s65                // wg1 >= nwg1-1
s_cselect_b32 s64, s64, 0                          // set rem
s_cmpk_gt_u32 s64, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_GSU1_Then_1     // jump if edges required
label_GW_B0_FD0_VW1_GSU1_NonEdge_1:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=128 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (1,0,0,0:vw1); (2,0,0,0:vw1); (3,0,0,0:vw1); (4,0,0,0:vw1); (5,0,0,0:vw1); (6,0,0,0:vw1); (7,0,0,0:vw1); (8,0,0,0:vw1); (9,0,0,0:vw1); (10,0,0,0:vw1); (11,0,0,0:vw1); (12,0,0,0:vw1); (13,0,0,0:vw1); (14,0,0,0:vw1); (15,0,0,0:vw1); (16,0,0,0:vw1); (17,0,0,0:vw1); (18,0,0,0:vw1); (19,0,0,0:vw1); (20,0,0,0:vw1); (21,0,0,0:vw1); (22,0,0,0:vw1); (23,0,0,0:vw1); (24,0,0,0:vw1); (25,0,0,0:vw1); (26,0,0,0:vw1); (27,0,0,0:vw1); (28,0,0,0:vw1); (29,0,0,0:vw1); (30,0,0,0:vw1); (31,0,0,0:vw1) */
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
v_add_lshl_u32 v86, v85, v82, 1                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=82, coord0Vgpr=82 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0), (3, 0, 0, 0), (4, 0, 0, 0), (5, 0, 0, 0), (6, 0, 0, 0), (7, 0, 0, 0), (8, 0, 0, 0), (9, 0, 0, 0), (10, 0, 0, 0), (11, 0, 0, 0), (12, 0, 0, 0), (13, 0, 0, 0), (14, 0, 0, 0), (15, 0, 0, 0), (16, 0, 0, 0), (17, 0, 0, 0), (18, 0, 0, 0), (19, 0, 0, 0), (20, 0, 0, 0), (21, 0, 0, 0), (22, 0, 0, 0), (23, 0, 0, 0), (24, 0, 0, 0), (25, 0, 0, 0), (26, 0, 0, 0), (27, 0, 0, 0), (28, 0, 0, 0), (29, 0, 0, 0), (30, 0, 0, 0), (31, 0, 0, 0)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha (pk)
v_mul_f32 v[vgprValuC+0+1], s[sgprAlpha], v[vgprValuC+0+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2+1], s[sgprAlpha], v[vgprValuC+2+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // *= alpha (pk)
v_mul_f32 v[vgprValuC+4+1], s[sgprAlpha], v[vgprValuC+4+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+6], s[sgprAlpha], v[vgprValuC+6] // *= alpha (pk)
v_mul_f32 v[vgprValuC+6+1], s[sgprAlpha], v[vgprValuC+6+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // *= alpha (pk)
v_mul_f32 v[vgprValuC+8+1], s[sgprAlpha], v[vgprValuC+8+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+10], s[sgprAlpha], v[vgprValuC+10] // *= alpha (pk)
v_mul_f32 v[vgprValuC+10+1], s[sgprAlpha], v[vgprValuC+10+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+12], s[sgprAlpha], v[vgprValuC+12] // *= alpha (pk)
v_mul_f32 v[vgprValuC+12+1], s[sgprAlpha], v[vgprValuC+12+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+14], s[sgprAlpha], v[vgprValuC+14] // *= alpha (pk)
v_mul_f32 v[vgprValuC+14+1], s[sgprAlpha], v[vgprValuC+14+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+16], s[sgprAlpha], v[vgprValuC+16] // *= alpha (pk)
v_mul_f32 v[vgprValuC+16+1], s[sgprAlpha], v[vgprValuC+16+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+18], s[sgprAlpha], v[vgprValuC+18] // *= alpha (pk)
v_mul_f32 v[vgprValuC+18+1], s[sgprAlpha], v[vgprValuC+18+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+20], s[sgprAlpha], v[vgprValuC+20] // *= alpha (pk)
v_mul_f32 v[vgprValuC+20+1], s[sgprAlpha], v[vgprValuC+20+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+22], s[sgprAlpha], v[vgprValuC+22] // *= alpha (pk)
v_mul_f32 v[vgprValuC+22+1], s[sgprAlpha], v[vgprValuC+22+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+24], s[sgprAlpha], v[vgprValuC+24] // *= alpha (pk)
v_mul_f32 v[vgprValuC+24+1], s[sgprAlpha], v[vgprValuC+24+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+26] // *= alpha (pk)
v_mul_f32 v[vgprValuC+26+1], s[sgprAlpha], v[vgprValuC+26+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+28], s[sgprAlpha], v[vgprValuC+28] // *= alpha (pk)
v_mul_f32 v[vgprValuC+28+1], s[sgprAlpha], v[vgprValuC+28+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+30], s[sgprAlpha], v[vgprValuC+30] // *= alpha (pk)
v_mul_f32 v[vgprValuC+30+1], s[sgprAlpha], v[vgprValuC+30+1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v88.l, v[vgprValuC+0]                // convert C to fp16
buffer_store_b16 v88, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v89.l, v[vgprValuC+1]                // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v89, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v90.l, v[vgprValuC+2]                // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v90, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v91.l, v[vgprValuC+3]                // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v91, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v92.l, v[vgprValuC+4]                // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v92, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v93.l, v[vgprValuC+5]                // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v93, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v94.l, v[vgprValuC+6]                // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v94, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v95.l, v[vgprValuC+7]                // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v95, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v96.l, v[vgprValuC+8]                // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 36                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b16 v96, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v97.l, v[vgprValuC+9]                // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v97, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v98.l, v[vgprValuC+10]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v98, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v99.l, v[vgprValuC+11]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v99, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v100.l, v[vgprValuC+12]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v100, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v101.l, v[vgprValuC+13]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v101, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v102.l, v[vgprValuC+14]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v102, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v103.l, v[vgprValuC+15]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v103, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v104.l, v[vgprValuC+16]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 36                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b16 v104, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v105.l, v[vgprValuC+17]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v105, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v106.l, v[vgprValuC+18]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v106, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v107.l, v[vgprValuC+19]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v107, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v108.l, v[vgprValuC+20]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v108, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v109.l, v[vgprValuC+21]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v109, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v110.l, v[vgprValuC+22]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v110, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v111.l, v[vgprValuC+23]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v111, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v112.l, v[vgprValuC+24]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 36                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b16 v112, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v113.l, v[vgprValuC+25]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v113, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v114.l, v[vgprValuC+26]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v114, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v115.l, v[vgprValuC+27]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v115, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v116.l, v[vgprValuC+28]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v116, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v117.l, v[vgprValuC+29]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v117, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v118.l, v[vgprValuC+30]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v118, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v119.l, v[vgprValuC+31]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v119, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW1_GSU1_NonEdgeEnd_1:
label_GW_B0_FD0_VW1_GSU1_Else_1:
label_GW_B0_FD0_VW1_GSU1_Then_1:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=64 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (1,0,0,0:vw1); (2,0,0,0:vw1); (3,0,0,0:vw1); (4,0,0,0:vw1); (5,0,0,0:vw1); (6,0,0,0:vw1); (7,0,0,0:vw1); (8,0,0,0:vw1); (9,0,0,0:vw1); (10,0,0,0:vw1); (11,0,0,0:vw1); (12,0,0,0:vw1); (13,0,0,0:vw1); (14,0,0,0:vw1); (15,0,0,0:vw1); (16,0,0,0:vw1); (17,0,0,0:vw1); (18,0,0,0:vw1); (19,0,0,0:vw1); (20,0,0,0:vw1); (21,0,0,0:vw1); (22,0,0,0:vw1); (23,0,0,0:vw1); (24,0,0,0:vw1); (25,0,0,0:vw1); (26,0,0,0:vw1); (27,0,0,0:vw1); (28,0,0,0:vw1); (29,0,0,0:vw1); (30,0,0,0:vw1); (31,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v122, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v86, v85, v82, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v86, v122, v86, s66                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v87, v85, v82, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v87, v122, v87, s66                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v127, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v127, v122, v127, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v128, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v128, v122, v128, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v129, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v129, v122, v129, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v130, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v122, v130, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v131, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v131, v122, v131, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v132, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v122, v132, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 18                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v133, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v133, v122, v133, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v134, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v122, v134, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v135, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v135, v122, v135, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v136, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v122, v136, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v137, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v137, v122, v137, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v138, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v122, v138, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v139, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v139, v122, v139, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v140, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v122, v140, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 18                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v141, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v141, v122, v141, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v142, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v122, v142, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v143, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v143, v122, v143, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v144, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v144, v122, v144, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v145, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v145, v122, v145, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v146, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v146, v122, v146, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v147, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v147, v122, v147, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v148, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v148, v122, v148, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 18                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v149, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v149, v122, v149, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v150, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v150, v122, v150, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v151, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v151, v122, v151, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v152, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v152, v122, v152, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v153, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v153, v122, v153, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v154, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v122, v154, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v155, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v155, v122, v155, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v156, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v122, v156, s66                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0), (3, 0, 0, 0), (4, 0, 0, 0), (5, 0, 0, 0), (6, 0, 0, 0), (7, 0, 0, 0), (8, 0, 0, 0), (9, 0, 0, 0), (10, 0, 0, 0), (11, 0, 0, 0), (12, 0, 0, 0), (13, 0, 0, 0), (14, 0, 0, 0), (15, 0, 0, 0), (16, 0, 0, 0), (17, 0, 0, 0), (18, 0, 0, 0), (19, 0, 0, 0), (20, 0, 0, 0), (21, 0, 0, 0), (22, 0, 0, 0), (23, 0, 0, 0), (24, 0, 0, 0), (25, 0, 0, 0), (26, 0, 0, 0), (27, 0, 0, 0), (28, 0, 0, 0), (29, 0, 0, 0), (30, 0, 0, 0), (31, 0, 0, 0)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha (pk)
v_mul_f32 v[vgprValuC+0+1], s[sgprAlpha], v[vgprValuC+0+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2+1], s[sgprAlpha], v[vgprValuC+2+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // *= alpha (pk)
v_mul_f32 v[vgprValuC+4+1], s[sgprAlpha], v[vgprValuC+4+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+6], s[sgprAlpha], v[vgprValuC+6] // *= alpha (pk)
v_mul_f32 v[vgprValuC+6+1], s[sgprAlpha], v[vgprValuC+6+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // *= alpha (pk)
v_mul_f32 v[vgprValuC+8+1], s[sgprAlpha], v[vgprValuC+8+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+10], s[sgprAlpha], v[vgprValuC+10] // *= alpha (pk)
v_mul_f32 v[vgprValuC+10+1], s[sgprAlpha], v[vgprValuC+10+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+12], s[sgprAlpha], v[vgprValuC+12] // *= alpha (pk)
v_mul_f32 v[vgprValuC+12+1], s[sgprAlpha], v[vgprValuC+12+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+14], s[sgprAlpha], v[vgprValuC+14] // *= alpha (pk)
v_mul_f32 v[vgprValuC+14+1], s[sgprAlpha], v[vgprValuC+14+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+16], s[sgprAlpha], v[vgprValuC+16] // *= alpha (pk)
v_mul_f32 v[vgprValuC+16+1], s[sgprAlpha], v[vgprValuC+16+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+18], s[sgprAlpha], v[vgprValuC+18] // *= alpha (pk)
v_mul_f32 v[vgprValuC+18+1], s[sgprAlpha], v[vgprValuC+18+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+20], s[sgprAlpha], v[vgprValuC+20] // *= alpha (pk)
v_mul_f32 v[vgprValuC+20+1], s[sgprAlpha], v[vgprValuC+20+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+22], s[sgprAlpha], v[vgprValuC+22] // *= alpha (pk)
v_mul_f32 v[vgprValuC+22+1], s[sgprAlpha], v[vgprValuC+22+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+24], s[sgprAlpha], v[vgprValuC+24] // *= alpha (pk)
v_mul_f32 v[vgprValuC+24+1], s[sgprAlpha], v[vgprValuC+24+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+26] // *= alpha (pk)
v_mul_f32 v[vgprValuC+26+1], s[sgprAlpha], v[vgprValuC+26+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+28], s[sgprAlpha], v[vgprValuC+28] // *= alpha (pk)
v_mul_f32 v[vgprValuC+28+1], s[sgprAlpha], v[vgprValuC+28+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+30], s[sgprAlpha], v[vgprValuC+30] // *= alpha (pk)
v_mul_f32 v[vgprValuC+30+1], s[sgprAlpha], v[vgprValuC+30+1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v88.l, v[vgprValuC+0]                // convert C to fp16
buffer_store_b16 v88, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v89.l, v[vgprValuC+1]                // convert C to fp16
buffer_store_b16 v89, v87, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v90.l, v[vgprValuC+2]                // convert C to fp16
buffer_store_b16 v90, v127, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v91.l, v[vgprValuC+3]                // convert C to fp16
buffer_store_b16 v91, v128, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v92.l, v[vgprValuC+4]                // convert C to fp16
buffer_store_b16 v92, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v93.l, v[vgprValuC+5]                // convert C to fp16
buffer_store_b16 v93, v130, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v94.l, v[vgprValuC+6]                // convert C to fp16
buffer_store_b16 v94, v131, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v95.l, v[vgprValuC+7]                // convert C to fp16
buffer_store_b16 v95, v132, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v96.l, v[vgprValuC+8]                // convert C to fp16
buffer_store_b16 v96, v133, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v97.l, v[vgprValuC+9]                // convert C to fp16
buffer_store_b16 v97, v134, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v98.l, v[vgprValuC+10]               // convert C to fp16
buffer_store_b16 v98, v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v99.l, v[vgprValuC+11]               // convert C to fp16
buffer_store_b16 v99, v136, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v100.l, v[vgprValuC+12]              // convert C to fp16
buffer_store_b16 v100, v137, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v101.l, v[vgprValuC+13]              // convert C to fp16
buffer_store_b16 v101, v138, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v102.l, v[vgprValuC+14]              // convert C to fp16
buffer_store_b16 v102, v139, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v103.l, v[vgprValuC+15]              // convert C to fp16
buffer_store_b16 v103, v140, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v104.l, v[vgprValuC+16]              // convert C to fp16
buffer_store_b16 v104, v141, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v105.l, v[vgprValuC+17]              // convert C to fp16
buffer_store_b16 v105, v142, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v106.l, v[vgprValuC+18]              // convert C to fp16
buffer_store_b16 v106, v143, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v107.l, v[vgprValuC+19]              // convert C to fp16
buffer_store_b16 v107, v144, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v108.l, v[vgprValuC+20]              // convert C to fp16
buffer_store_b16 v108, v145, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v109.l, v[vgprValuC+21]              // convert C to fp16
buffer_store_b16 v109, v146, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v110.l, v[vgprValuC+22]              // convert C to fp16
buffer_store_b16 v110, v147, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v111.l, v[vgprValuC+23]              // convert C to fp16
buffer_store_b16 v111, v148, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v112.l, v[vgprValuC+24]              // convert C to fp16
buffer_store_b16 v112, v149, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v113.l, v[vgprValuC+25]              // convert C to fp16
buffer_store_b16 v113, v150, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v114.l, v[vgprValuC+26]              // convert C to fp16
buffer_store_b16 v114, v151, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v115.l, v[vgprValuC+27]              // convert C to fp16
buffer_store_b16 v115, v152, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v116.l, v[vgprValuC+28]              // convert C to fp16
buffer_store_b16 v116, v153, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v117.l, v[vgprValuC+29]              // convert C to fp16
buffer_store_b16 v117, v154, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v118.l, v[vgprValuC+30]              // convert C to fp16
buffer_store_b16 v118, v155, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v119.l, v[vgprValuC+31]              // convert C to fp16
buffer_store_b16 v119, v156, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_GSU1:
label_GW_B1_FD0_GSU1:

/* Edge/NonEdge store path check (M): Size % 32 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s64, 31, s[sgprSizeI]                    // s64 = s[sgprSizeI] % 32
s_add_u32 s65, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s65                // wg0 >= nwg0-1 ?
s_cselect_b32 s64, s64, 0                          // set rem
s_cmpk_gt_u32 s64, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW1_GSU1_Else_1     // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 128 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s64, 127, s[sgprSizeJ]                   // s64 = s[sgprSizeJ] % 128
s_add_u32 s65, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s65                // wg1 >= nwg1-1
s_cselect_b32 s64, s64, 0                          // set rem
s_cmpk_gt_u32 s64, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW1_GSU1_Then_1     // jump if edges required
label_GW_B1_FD0_VW1_GSU1_NonEdge_1:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=62 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (1,0,0,0:vw1); (2,0,0,0:vw1); (3,0,0,0:vw1); (4,0,0,0:vw1); (5,0,0,0:vw1); (6,0,0,0:vw1); (7,0,0,0:vw1); (8,0,0,0:vw1); (9,0,0,0:vw1); (10,0,0,0:vw1); (11,0,0,0:vw1); (12,0,0,0:vw1); (13,0,0,0:vw1); (14,0,0,0:vw1); (15,0,0,0:vw1); (16,0,0,0:vw1); (17,0,0,0:vw1); (18,0,0,0:vw1); (19,0,0,0:vw1); (20,0,0,0:vw1); (21,0,0,0:vw1); (22,0,0,0:vw1); (23,0,0,0:vw1); (24,0,0,0:vw1); (25,0,0,0:vw1); (26,0,0,0:vw1); (27,0,0,0:vw1); (28,0,0,0:vw1); (29,0,0,0:vw1); (30,0,0,0:vw1); (31,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v87, v84, v82, 1                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=82, coord0Vgpr=82 (multiple bpe)
buffer_load_d16_b16 v127, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v128, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v129, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v130, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v131, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v132, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v133, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v134, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 36                 // scale StrideC *= numRows(18) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_load_d16_b16 v135, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v136, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v137, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v138, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v139, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v140, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v141, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v142, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 36                 // scale StrideC *= numRows(18) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_load_d16_b16 v143, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v144, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v145, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v146, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v147, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v148, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v149, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v150, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 36                 // scale StrideC *= numRows(18) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_load_d16_b16 v151, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v152, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v153, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v154, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v155, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v156, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v157, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v158, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v86, v85, v82, 1                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=82, coord0Vgpr=82 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0), (3, 0, 0, 0), (4, 0, 0, 0), (5, 0, 0, 0), (6, 0, 0, 0), (7, 0, 0, 0), (8, 0, 0, 0), (9, 0, 0, 0), (10, 0, 0, 0), (11, 0, 0, 0), (12, 0, 0, 0), (13, 0, 0, 0), (14, 0, 0, 0), (15, 0, 0, 0), (16, 0, 0, 0), (17, 0, 0, 0), (18, 0, 0, 0), (19, 0, 0, 0), (20, 0, 0, 0), (21, 0, 0, 0), (22, 0, 0, 0), (23, 0, 0, 0), (24, 0, 0, 0), (25, 0, 0, 0), (26, 0, 0, 0), (27, 0, 0, 0), (28, 0, 0, 0), (29, 0, 0, 0), (30, 0, 0, 0), (31, 0, 0, 0)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha (pk)
v_mul_f32 v[vgprValuC+0+1], s[sgprAlpha], v[vgprValuC+0+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2+1], s[sgprAlpha], v[vgprValuC+2+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // *= alpha (pk)
v_mul_f32 v[vgprValuC+4+1], s[sgprAlpha], v[vgprValuC+4+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+6], s[sgprAlpha], v[vgprValuC+6] // *= alpha (pk)
v_mul_f32 v[vgprValuC+6+1], s[sgprAlpha], v[vgprValuC+6+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // *= alpha (pk)
v_mul_f32 v[vgprValuC+8+1], s[sgprAlpha], v[vgprValuC+8+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+10], s[sgprAlpha], v[vgprValuC+10] // *= alpha (pk)
v_mul_f32 v[vgprValuC+10+1], s[sgprAlpha], v[vgprValuC+10+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+12], s[sgprAlpha], v[vgprValuC+12] // *= alpha (pk)
v_mul_f32 v[vgprValuC+12+1], s[sgprAlpha], v[vgprValuC+12+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+14], s[sgprAlpha], v[vgprValuC+14] // *= alpha (pk)
v_mul_f32 v[vgprValuC+14+1], s[sgprAlpha], v[vgprValuC+14+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+16], s[sgprAlpha], v[vgprValuC+16] // *= alpha (pk)
v_mul_f32 v[vgprValuC+16+1], s[sgprAlpha], v[vgprValuC+16+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+18], s[sgprAlpha], v[vgprValuC+18] // *= alpha (pk)
v_mul_f32 v[vgprValuC+18+1], s[sgprAlpha], v[vgprValuC+18+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+20], s[sgprAlpha], v[vgprValuC+20] // *= alpha (pk)
v_mul_f32 v[vgprValuC+20+1], s[sgprAlpha], v[vgprValuC+20+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+22], s[sgprAlpha], v[vgprValuC+22] // *= alpha (pk)
v_mul_f32 v[vgprValuC+22+1], s[sgprAlpha], v[vgprValuC+22+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+24], s[sgprAlpha], v[vgprValuC+24] // *= alpha (pk)
v_mul_f32 v[vgprValuC+24+1], s[sgprAlpha], v[vgprValuC+24+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+26] // *= alpha (pk)
v_mul_f32 v[vgprValuC+26+1], s[sgprAlpha], v[vgprValuC+26+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+28], s[sgprAlpha], v[vgprValuC+28] // *= alpha (pk)
v_mul_f32 v[vgprValuC+28+1], s[sgprAlpha], v[vgprValuC+28+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+30], s[sgprAlpha], v[vgprValuC+30] // *= alpha (pk)
v_mul_f32 v[vgprValuC+30+1], s[sgprAlpha], v[vgprValuC+30+1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(31)                                // vlcnt(31) = 32 - 1 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+0], s[sgprBeta], v127, v[vgprValuC+0] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v88.l, v[vgprValuC+0]                // convert C to fp16
buffer_store_b16 v88, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(30)                                // vlcnt(30) = 32 - 2 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+1], s[sgprBeta], v128, v[vgprValuC+1] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v89.l, v[vgprValuC+1]                // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v89, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(29)                                // vlcnt(29) = 32 - 3 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+2], s[sgprBeta], v129, v[vgprValuC+2] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v90.l, v[vgprValuC+2]                // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v90, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(28)                                // vlcnt(28) = 32 - 4 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+3], s[sgprBeta], v130, v[vgprValuC+3] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v91.l, v[vgprValuC+3]                // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v91, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(27)                                // vlcnt(27) = 32 - 5 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+4], s[sgprBeta], v131, v[vgprValuC+4] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v92.l, v[vgprValuC+4]                // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v92, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(26)                                // vlcnt(26) = 32 - 6 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+5], s[sgprBeta], v132, v[vgprValuC+5] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v93.l, v[vgprValuC+5]                // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v93, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(25)                                // vlcnt(25) = 32 - 7 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+6], s[sgprBeta], v133, v[vgprValuC+6] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v94.l, v[vgprValuC+6]                // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v94, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(24)                                // vlcnt(24) = 32 - 8 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+7], s[sgprBeta], v134, v[vgprValuC+7] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v95.l, v[vgprValuC+7]                // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v95, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(23)                                // vlcnt(23) = 32 - 9 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+8], s[sgprBeta], v135, v[vgprValuC+8] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v96.l, v[vgprValuC+8]                // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 36                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b16 v96, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(22)                                // vlcnt(22) = 32 - 10 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+9], s[sgprBeta], v136, v[vgprValuC+9] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v97.l, v[vgprValuC+9]                // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v97, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(21)                                // vlcnt(21) = 32 - 11 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+10], s[sgprBeta], v137, v[vgprValuC+10] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v98.l, v[vgprValuC+10]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v98, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(20)                                // vlcnt(20) = 32 - 12 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+11], s[sgprBeta], v138, v[vgprValuC+11] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v99.l, v[vgprValuC+11]               // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v99, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(19)                                // vlcnt(19) = 32 - 13 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+12], s[sgprBeta], v139, v[vgprValuC+12] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v100.l, v[vgprValuC+12]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v100, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(18)                                // vlcnt(18) = 32 - 14 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+13], s[sgprBeta], v140, v[vgprValuC+13] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v101.l, v[vgprValuC+13]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v101, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(17)                                // vlcnt(17) = 32 - 15 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+14], s[sgprBeta], v141, v[vgprValuC+14] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v102.l, v[vgprValuC+14]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v102, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(16)                                // vlcnt(16) = 32 - 16 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+15], s[sgprBeta], v142, v[vgprValuC+15] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v103.l, v[vgprValuC+15]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v103, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(15) = 32 - 17 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+16], s[sgprBeta], v143, v[vgprValuC+16] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v104.l, v[vgprValuC+16]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 36                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b16 v104, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(14)                                // vlcnt(14) = 32 - 18 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+17], s[sgprBeta], v144, v[vgprValuC+17] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v105.l, v[vgprValuC+17]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v105, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(13)                                // vlcnt(13) = 32 - 19 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+18], s[sgprBeta], v145, v[vgprValuC+18] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v106.l, v[vgprValuC+18]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v106, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(12)                                // vlcnt(12) = 32 - 20 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+19], s[sgprBeta], v146, v[vgprValuC+19] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v107.l, v[vgprValuC+19]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v107, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(11)                                // vlcnt(11) = 32 - 21 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+20], s[sgprBeta], v147, v[vgprValuC+20] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v108.l, v[vgprValuC+20]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v108, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(10)                                // vlcnt(10) = 32 - 22 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+21], s[sgprBeta], v148, v[vgprValuC+21] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v109.l, v[vgprValuC+21]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v109, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(9)                                 // vlcnt(9) = 32 - 23 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+22], s[sgprBeta], v149, v[vgprValuC+22] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v110.l, v[vgprValuC+22]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v110, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(8)                                 // vlcnt(8) = 32 - 24 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+23], s[sgprBeta], v150, v[vgprValuC+23] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v111.l, v[vgprValuC+23]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v111, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(7)                                 // vlcnt(7) = 32 - 25 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+24], s[sgprBeta], v151, v[vgprValuC+24] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v112.l, v[vgprValuC+24]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 36                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b16 v112, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(6)                                 // vlcnt(6) = 32 - 26 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+25], s[sgprBeta], v152, v[vgprValuC+25] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v113.l, v[vgprValuC+25]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v113, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(5)                                 // vlcnt(5) = 32 - 27 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+26], s[sgprBeta], v153, v[vgprValuC+26] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v114.l, v[vgprValuC+26]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v114, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(4)                                 // vlcnt(4) = 32 - 28 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+27], s[sgprBeta], v154, v[vgprValuC+27] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v115.l, v[vgprValuC+27]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v115, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(3)                                 // vlcnt(3) = 32 - 29 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+28], s[sgprBeta], v155, v[vgprValuC+28] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v116.l, v[vgprValuC+28]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v116, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(2)                                 // vlcnt(2) = 32 - 30 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+29], s[sgprBeta], v156, v[vgprValuC+29] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v117.l, v[vgprValuC+29]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v117, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(1)                                 // vlcnt(1) = 32 - 31 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+30], s[sgprBeta], v157, v[vgprValuC+30] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v118.l, v[vgprValuC+30]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v118, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(0)                                 // vlcnt(0) = 32 - 32 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+31], s[sgprBeta], v158, v[vgprValuC+31] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v119.l, v[vgprValuC+31]              // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v119, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_FD0_VW1_GSU1_NonEdgeEnd_1:
label_GW_B1_FD0_VW1_GSU1_Else_1:
label_GW_B1_FD0_VW1_GSU1_Then_1:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=42 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (1,0,0,0:vw1); (2,0,0,0:vw1); (3,0,0,0:vw1); (4,0,0,0:vw1); (5,0,0,0:vw1); (6,0,0,0:vw1); (7,0,0,0:vw1); (8,0,0,0:vw1); (9,0,0,0:vw1); (10,0,0,0:vw1); (11,0,0,0:vw1); (12,0,0,0:vw1); (13,0,0,0:vw1); (14,0,0,0:vw1); (15,0,0,0:vw1); (16,0,0,0:vw1); (17,0,0,0:vw1); (18,0,0,0:vw1); (19,0,0,0:vw1); (20,0,0,0:vw1); (21,0,0,0:vw1); (22,0,0,0:vw1); (23,0,0,0:vw1); (24,0,0,0:vw1); (25,0,0,0:vw1); (26,0,0,0:vw1); (27,0,0,0:vw1); (28,0,0,0:vw1); (29,0,0,0:vw1); (30,0,0,0:vw1); (31,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v122, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v87, v84, v82, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v87, v122, v87, s66                  // LDC clip if OOB. offset
buffer_load_d16_b16 v86, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v87, v85, v82, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v87, v122, v87, s66                  // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v128, v84, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v128, v122, v128, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v127, v128, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v128, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v128, v122, v128, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v130, v84, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v122, v130, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v129, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v130, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v122, v130, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v132, v84, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v122, v132, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v131, v132, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v132, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v122, v132, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v134, v84, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v122, v134, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v133, v134, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v134, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v122, v134, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v136, v84, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v122, v136, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v135, v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v136, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v122, v136, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v138, v84, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v122, v138, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v137, v138, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v138, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v122, v138, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v140, v84, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v122, v140, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v139, v140, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v140, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v122, v140, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 18                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v142, v84, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v122, v142, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v141, v142, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v142, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v122, v142, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v144, v84, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v144, v122, v144, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v143, v144, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v144, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v144, v122, v144, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v146, v84, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v146, v122, v146, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v145, v146, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v146, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v146, v122, v146, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v148, v84, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v148, v122, v148, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v147, v148, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v148, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v148, v122, v148, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v150, v84, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v150, v122, v150, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v149, v150, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v150, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v150, v122, v150, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v152, v84, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v152, v122, v152, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v151, v152, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v152, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v152, v122, v152, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v154, v84, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v122, v154, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v153, v154, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v154, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v122, v154, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v156, v84, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v122, v156, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v155, v156, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v156, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v122, v156, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 18                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v158, v84, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v122, v158, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v157, v158, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v158, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v122, v158, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v160, v84, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v160, v122, v160, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v159, v160, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v160, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v160, v122, v160, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v162, v84, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v162, v122, v162, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v161, v162, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v162, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v162, v122, v162, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v164, v84, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v164, v122, v164, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v163, v164, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v164, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v164, v122, v164, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v166, v84, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v166, v122, v166, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v165, v166, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v166, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v166, v122, v166, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v168, v84, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v168, v122, v168, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v167, v168, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v168, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v168, v122, v168, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v170, v84, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v170, v122, v170, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v169, v170, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v170, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v170, v122, v170, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v172, v84, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v172, v122, v172, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v171, v172, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v172, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v172, v122, v172, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 18                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v174, v84, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v174, v122, v174, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v173, v174, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v174, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v174, v122, v174, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v176, v84, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v176, v122, v176, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v175, v176, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v176, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v176, v122, v176, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v178, v84, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v178, v122, v178, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v177, v178, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v178, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v178, v122, v178, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v180, v84, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v180, v122, v180, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v179, v180, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v180, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v180, v122, v180, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v182, v84, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v182, v122, v182, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v181, v182, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v182, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v182, v122, v182, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v184, v84, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v184, v122, v184, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v183, v184, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v184, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v184, v122, v184, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v186, v84, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v186, v122, v186, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v185, v186, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v186, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v186, v122, v186, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
v_add_co_u32 v83, vcc_lo, v83, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v84, v84, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v85, v85, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v83, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v188, v84, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v188, v122, v188, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v187, v188, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v188, v85, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v188, v122, v188, s66                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0), (3, 0, 0, 0), (4, 0, 0, 0), (5, 0, 0, 0), (6, 0, 0, 0), (7, 0, 0, 0), (8, 0, 0, 0), (9, 0, 0, 0), (10, 0, 0, 0), (11, 0, 0, 0), (12, 0, 0, 0), (13, 0, 0, 0), (14, 0, 0, 0), (15, 0, 0, 0), (16, 0, 0, 0), (17, 0, 0, 0), (18, 0, 0, 0), (19, 0, 0, 0), (20, 0, 0, 0), (21, 0, 0, 0), (22, 0, 0, 0), (23, 0, 0, 0), (24, 0, 0, 0), (25, 0, 0, 0), (26, 0, 0, 0), (27, 0, 0, 0), (28, 0, 0, 0), (29, 0, 0, 0), (30, 0, 0, 0), (31, 0, 0, 0)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha (pk)
v_mul_f32 v[vgprValuC+0+1], s[sgprAlpha], v[vgprValuC+0+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2+1], s[sgprAlpha], v[vgprValuC+2+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // *= alpha (pk)
v_mul_f32 v[vgprValuC+4+1], s[sgprAlpha], v[vgprValuC+4+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+6], s[sgprAlpha], v[vgprValuC+6] // *= alpha (pk)
v_mul_f32 v[vgprValuC+6+1], s[sgprAlpha], v[vgprValuC+6+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // *= alpha (pk)
v_mul_f32 v[vgprValuC+8+1], s[sgprAlpha], v[vgprValuC+8+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+10], s[sgprAlpha], v[vgprValuC+10] // *= alpha (pk)
v_mul_f32 v[vgprValuC+10+1], s[sgprAlpha], v[vgprValuC+10+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+12], s[sgprAlpha], v[vgprValuC+12] // *= alpha (pk)
v_mul_f32 v[vgprValuC+12+1], s[sgprAlpha], v[vgprValuC+12+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+14], s[sgprAlpha], v[vgprValuC+14] // *= alpha (pk)
v_mul_f32 v[vgprValuC+14+1], s[sgprAlpha], v[vgprValuC+14+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+16], s[sgprAlpha], v[vgprValuC+16] // *= alpha (pk)
v_mul_f32 v[vgprValuC+16+1], s[sgprAlpha], v[vgprValuC+16+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+18], s[sgprAlpha], v[vgprValuC+18] // *= alpha (pk)
v_mul_f32 v[vgprValuC+18+1], s[sgprAlpha], v[vgprValuC+18+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+20], s[sgprAlpha], v[vgprValuC+20] // *= alpha (pk)
v_mul_f32 v[vgprValuC+20+1], s[sgprAlpha], v[vgprValuC+20+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+22], s[sgprAlpha], v[vgprValuC+22] // *= alpha (pk)
v_mul_f32 v[vgprValuC+22+1], s[sgprAlpha], v[vgprValuC+22+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+24], s[sgprAlpha], v[vgprValuC+24] // *= alpha (pk)
v_mul_f32 v[vgprValuC+24+1], s[sgprAlpha], v[vgprValuC+24+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+26] // *= alpha (pk)
v_mul_f32 v[vgprValuC+26+1], s[sgprAlpha], v[vgprValuC+26+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+28], s[sgprAlpha], v[vgprValuC+28] // *= alpha (pk)
v_mul_f32 v[vgprValuC+28+1], s[sgprAlpha], v[vgprValuC+28+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+30], s[sgprAlpha], v[vgprValuC+30] // *= alpha (pk)
v_mul_f32 v[vgprValuC+30+1], s[sgprAlpha], v[vgprValuC+30+1] // *= alpha (pk)
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fma_mix_f32 v[vgprValuC+0], s[sgprBeta], v86, v[vgprValuC+0] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v88.l, v[vgprValuC+0]                // convert C to fp16
buffer_store_b16 v88, v87, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+1], s[sgprBeta], v127, v[vgprValuC+1] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v89.l, v[vgprValuC+1]                // convert C to fp16
buffer_store_b16 v89, v128, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+2], s[sgprBeta], v129, v[vgprValuC+2] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v90.l, v[vgprValuC+2]                // convert C to fp16
buffer_store_b16 v90, v130, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+3], s[sgprBeta], v131, v[vgprValuC+3] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v91.l, v[vgprValuC+3]                // convert C to fp16
buffer_store_b16 v91, v132, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+4], s[sgprBeta], v133, v[vgprValuC+4] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v92.l, v[vgprValuC+4]                // convert C to fp16
buffer_store_b16 v92, v134, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+5], s[sgprBeta], v135, v[vgprValuC+5] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v93.l, v[vgprValuC+5]                // convert C to fp16
buffer_store_b16 v93, v136, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+6], s[sgprBeta], v137, v[vgprValuC+6] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v94.l, v[vgprValuC+6]                // convert C to fp16
buffer_store_b16 v94, v138, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+7], s[sgprBeta], v139, v[vgprValuC+7] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v95.l, v[vgprValuC+7]                // convert C to fp16
buffer_store_b16 v95, v140, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+8], s[sgprBeta], v141, v[vgprValuC+8] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v96.l, v[vgprValuC+8]                // convert C to fp16
buffer_store_b16 v96, v142, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+9], s[sgprBeta], v143, v[vgprValuC+9] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v97.l, v[vgprValuC+9]                // convert C to fp16
buffer_store_b16 v97, v144, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+10], s[sgprBeta], v145, v[vgprValuC+10] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v98.l, v[vgprValuC+10]               // convert C to fp16
buffer_store_b16 v98, v146, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+11], s[sgprBeta], v147, v[vgprValuC+11] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v99.l, v[vgprValuC+11]               // convert C to fp16
buffer_store_b16 v99, v148, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+12], s[sgprBeta], v149, v[vgprValuC+12] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v100.l, v[vgprValuC+12]              // convert C to fp16
buffer_store_b16 v100, v150, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+13], s[sgprBeta], v151, v[vgprValuC+13] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v101.l, v[vgprValuC+13]              // convert C to fp16
buffer_store_b16 v101, v152, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+14], s[sgprBeta], v153, v[vgprValuC+14] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v102.l, v[vgprValuC+14]              // convert C to fp16
buffer_store_b16 v102, v154, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+15], s[sgprBeta], v155, v[vgprValuC+15] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v103.l, v[vgprValuC+15]              // convert C to fp16
buffer_store_b16 v103, v156, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+16], s[sgprBeta], v157, v[vgprValuC+16] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v104.l, v[vgprValuC+16]              // convert C to fp16
buffer_store_b16 v104, v158, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+17], s[sgprBeta], v159, v[vgprValuC+17] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v105.l, v[vgprValuC+17]              // convert C to fp16
buffer_store_b16 v105, v160, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+18], s[sgprBeta], v161, v[vgprValuC+18] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v106.l, v[vgprValuC+18]              // convert C to fp16
buffer_store_b16 v106, v162, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+19], s[sgprBeta], v163, v[vgprValuC+19] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v107.l, v[vgprValuC+19]              // convert C to fp16
buffer_store_b16 v107, v164, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+20], s[sgprBeta], v165, v[vgprValuC+20] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v108.l, v[vgprValuC+20]              // convert C to fp16
buffer_store_b16 v108, v166, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+21], s[sgprBeta], v167, v[vgprValuC+21] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v109.l, v[vgprValuC+21]              // convert C to fp16
buffer_store_b16 v109, v168, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+22], s[sgprBeta], v169, v[vgprValuC+22] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v110.l, v[vgprValuC+22]              // convert C to fp16
buffer_store_b16 v110, v170, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+23], s[sgprBeta], v171, v[vgprValuC+23] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v111.l, v[vgprValuC+23]              // convert C to fp16
buffer_store_b16 v111, v172, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+24], s[sgprBeta], v173, v[vgprValuC+24] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v112.l, v[vgprValuC+24]              // convert C to fp16
buffer_store_b16 v112, v174, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+25], s[sgprBeta], v175, v[vgprValuC+25] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v113.l, v[vgprValuC+25]              // convert C to fp16
buffer_store_b16 v113, v176, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+26], s[sgprBeta], v177, v[vgprValuC+26] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v114.l, v[vgprValuC+26]              // convert C to fp16
buffer_store_b16 v114, v178, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+27], s[sgprBeta], v179, v[vgprValuC+27] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v115.l, v[vgprValuC+27]              // convert C to fp16
buffer_store_b16 v115, v180, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+28], s[sgprBeta], v181, v[vgprValuC+28] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v116.l, v[vgprValuC+28]              // convert C to fp16
buffer_store_b16 v116, v182, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+29], s[sgprBeta], v183, v[vgprValuC+29] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v117.l, v[vgprValuC+29]              // convert C to fp16
buffer_store_b16 v117, v184, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+30], s[sgprBeta], v185, v[vgprValuC+30] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v118.l, v[vgprValuC+30]              // convert C to fp16
buffer_store_b16 v118, v186, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+31], s[sgprBeta], v187, v[vgprValuC+31] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v119.l, v[vgprValuC+31]              // convert C to fp16
buffer_store_b16 v119, v188, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_SK_Partials_1:
label_GW_Partials_E0:
s_mov_b64 s[sgprSrdWS:sgprSrdWS+1], s[sgprAddressWS:sgprAddressWS+1]
s_mov_b32 s[sgprSrdWS+2], BufferOOB
s_mov_b32 s[sgprSrdWS+3], Srd127_96

s_mul_i32 s8, 0x4000, s[sgprPersistentWorkGroupIndex] // Offset to correct partials tile (low word)
s_mul_hi_u32 s9, 0x4000, s[sgprPersistentWorkGroupIndex] // partials tile offset (high word) for 64-bit SRD
s_add_u32 s[sgprSrdWS+0], s[sgprSrdWS+0], s8       // add lo to SRD
s_addc_u32 s[sgprSrdWS+1], s[sgprSrdWS+1], s9      // add hi (offset high word + lo carry) to SRD

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=128 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Partials Write Batch #0 (d1,d0,vc1,vc0) = */
/*      (0,0,0,0:vw1); (1,0,0,0:vw1); (2,0,0,0:vw1); (3,0,0,0:vw1); (4,0,0,0:vw1); (5,0,0,0:vw1); (6,0,0,0:vw1); (7,0,0,0:vw1); (8,0,0,0:vw1); (9,0,0,0:vw1); (10,0,0,0:vw1); (11,0,0,0:vw1); (12,0,0,0:vw1); (13,0,0,0:vw1); (14,0,0,0:vw1); (15,0,0,0:vw1); (16,0,0,0:vw1); (17,0,0,0:vw1); (18,0,0,0:vw1); (19,0,0,0:vw1); (20,0,0,0:vw1); (21,0,0,0:vw1); (22,0,0,0:vw1); (23,0,0,0:vw1); (24,0,0,0:vw1); (25,0,0,0:vw1); (26,0,0,0:vw1); (27,0,0,0:vw1); (28,0,0,0:vw1); (29,0,0,0:vw1); (30,0,0,0:vw1); (31,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* apply mask, calc new C and issue writes */
v_lshlrev_b32 v86, 2, v[vgprSerial]                // v86 = v[vgprSerial] * 4
s_mov_b32 s8, 0                                    // Init sgpr offset
buffer_store_b32 v88, v86, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v89, v86, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v90, v86, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v91, v86, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v92, v86, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v93, v86, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v94, v86, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v95, v86, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v96, v86, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v97, v86, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v98, v86, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v99, v86, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v100, v86, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v101, v86, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v102, v86, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v103, v86, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v104, v86, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v105, v86, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v106, v86, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v107, v86, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v108, v86, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v109, v86, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v110, v86, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v111, v86, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v112, v86, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v113, v86, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v114, v86, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v115, v86, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v116, v86, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v117, v86, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v118, v86, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v119, v86, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_waitcnt_vscnt null 0                             // wait for data store
s_barrier                                          // store all data before setting flag
s_lshl_b32 s8, s[sgprPersistentWorkGroupIndex], 2  // flag offset based on physical partial slot
v_readfirstlane_b32 s9, v[vgprSerial]              // Wave 0 updates flags
s_cmp_eq_u32 s9, 0                                 // Check for wave 0
s_cbranch_scc0 label_SK_SkipFlagSet                // Skip flag set
s_mov_b32 s9, 1                                    // flag data
v_mov_b32 v127, s9                                 // move flag value to vgpr
v_mov_b32 v128, 0                                  // zero vaddr offset
s_mov_b64 s[64:65], s[sgprAddressFlags:sgprAddressFlags+1]
s_mov_b32 s66, BufferOOB
s_mov_b32 s67, Srd127_96
buffer_store_b32 v127, v128, s[64:67], s8 offen offset:0 glc dlc // set flag
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
