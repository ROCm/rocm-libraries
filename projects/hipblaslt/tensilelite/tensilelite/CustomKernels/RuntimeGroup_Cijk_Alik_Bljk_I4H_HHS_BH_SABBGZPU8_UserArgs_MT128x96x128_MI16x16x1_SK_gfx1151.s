
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
.text
.protected RuntimeGroup_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGZPU8_UserArgs_MT128x96x128_MI16x16x1_SK_gfx1151
.globl RuntimeGroup_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGZPU8_UserArgs_MT128x96x128_MI16x16x1_SK_gfx1151
.p2align 8
.type RuntimeGroup_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGZPU8_UserArgs_MT128x96x128_MI16x16x1_SK_gfx1151,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel RuntimeGroup_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGZPU8_UserArgs_MT128x96x128_MI16x16x1_SK_gfx1151
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 256 // vgprs
  .amdhsa_next_free_sgpr 96 // sgprs
  .amdhsa_group_segment_fixed_size 60928 // lds bytes
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
/* Num SGPR   =96 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 16 x 3 */
/* SubGroup= 8 x 32 */
/* VectorWidthA=1 */
/* VectorWidthB=1 */
/* GlobalReadVectorWidthA=16, GlobalReadVectorWidthB=4 */
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
  - .name: RuntimeGroup_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGZPU8_UserArgs_MT128x96x128_MI16x16x1_SK_gfx1151
    .symbol: 'RuntimeGroup_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGZPU8_UserArgs_MT128x96x128_MI16x16x1_SK_gfx1151.kd'
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
      - .name:            AddressScaleZeroA
        .size:            8
        .offset:          160
        .value_kind:      global_buffer
        .value_type:      void
        .address_space:   generic
      - .name:            batchOffsetD
        .size:            8
        .offset:          168
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetC
        .size:            8
        .offset:          176
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetA
        .size:            8
        .offset:          184
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetB
        .size:            8
        .offset:          192
        .value_kind:      by_value
        .value_type:      u64
      - .name:            groupSize
        .size:            4
        .offset:          200
        .value_kind:      by_value
        .value_type:      u32
    .group_segment_fixed_size:   60928
    .kernarg_segment_align:      8
    .kernarg_segment_size:       208
    .max_flat_workgroup_size:    256
    .private_segment_fixed_size: 0
    .sgpr_count:                 96
    .sgpr_spill_count:           0
    .vgpr_count:                 256
    .vgpr_spill_count:           0
    .wavefront_size:             32
...
.end_amdgpu_metadata
RuntimeGroup_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGZPU8_UserArgs_MT128x96x128_MI16x16x1_SK_gfx1151:
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
/* ValuC range: [0-48), serializedStore enabled */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprBase, 90
.set vgprLocalWriteAddrA, 84
.set vgprLocalWriteAddrB, 85
.set vgprGlobalReadOffsetA, 48
.set vgprGlobalReadOffsetB, 52
.set vgprGlobalReadOffsetScaleA, 64
.set vgprG2LScaleA, 68
.set vgprG2LScaleZeroA, 72
.set vgprGlobalReadOffsetScaleZeroA, 76
.set vgprGlobalReadByteOffsetScaleZeroA, 80
.set vgprLocalReadAddrA, 86
.set vgprLocalReadAddrB, 87
.set vgprSerial, 188

/******************************************/
/* VGPR Macro Assignments                 */
/******************************************/
.set vgprValuA_X0_I0_BASE, vgprBase+0
.set vgprValuB_X0_I0_BASE, vgprBase+17
.set vgprG2LA_BASE, vgprBase+42
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
.set sgprAddressScaleZeroA, 74
.set sgprSrdScaleZeroA, 76

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

.set MT0, 128
.set MT1, 96
.set DepthU, 128
/* Number of elements to shift-left SRD */
.set SrdShiftLeftA, 16
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
s_load_b32 s[sgprScaleAGMinus1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0xb8
s_load_b64 s[sgprAddressScaleA:sgprAddressScaleA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x80
s_load_b64 s[sgprAddressScaleZeroA:sgprAddressScaleZeroA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x90
s_branch label_LoadArgsEnd
label_HBMArgs:

/* Load address of kernel arguments */
s_load_b64 s[sgprKernArgAddress:sgprKernArgAddress+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 16
s_waitcnt lgkmcnt(0)                               // wait for args to load
label_LoadArgsEnd:
s_mov_b32 s[sgprArgType], s16
s_mov_b32 m0, 0xee00                               // LDS clamp at 60928 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id
s_mov_b32 vcc_hi, 0                                // Ensure hi bits are zero

/* remap workgroup to XCCs */
s_lshr_b32 s84, s[sgprWGM], 0x10                   // Get WGMXCC
s_ff1_i32_b32 s84, s84                             // Get log(WGMXCC)
s_lshr_b32 s85, s[sgprWGM], 0x16                   // Get CU_Count
/* remap WGs if WGMXCC > 1 ( log(WGMXCC) > 0 ) */
s_cmp_gt_i32 s84, 0
s_cbranch_scc0 label_skip_WGMXCC
/* only remap WGs in the range */
s_lshr_b32 s81, s18, s84
s_lshl_b32 s81, s81, s84
s_cmp_ge_u32 s[sgprWorkGroup0], s81
s_cbranch_scc1 label_skip_WGMXCC
s_cmp_eq_u32 s85, 0                                // CU_Count == 0 ?
s_cbranch_scc0 label_XCCG_nonzero
s_lshr_b32 s81, s[sgprWorkGroup0], s84
s_bfm_b32 s82, s84, 0
s_and_b32 s82, s[sgprWorkGroup0], s82
s_lshr_b32 s83, s18, s84
s_mul_i32 s82, s82, s83
s_add_u32 s[sgprWorkGroup0], s81, s82
s_branch label_skip_WGMXCC
label_XCCG_nonzero:
/* temp0 = (wg//CU_Count)*CU_Count */
v_cvt_f64_u32 v[0:1], s85                          // s81 = s[sgprWorkGroup0] / s85
v_rcp_f64 v[0:1], v[0:1]                           // s81 = s[sgprWorkGroup0] / s85
v_cvt_f64_u32 v[2:3], s[sgprWorkGroup0]            // s81 = s[sgprWorkGroup0] / s85
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s81 = s[sgprWorkGroup0] / s85
v_cvt_u32_f64 v0, v[0:1]                           // s81 = s[sgprWorkGroup0] / s85
v_mul_lo_u32 v1, v0, s85                           // s81 = s[sgprWorkGroup0] / s85
v_sub_nc_u32 v2, s[sgprWorkGroup0], v1             // s81 = s[sgprWorkGroup0] / s85
v_cmp_ge_u32 vcc_lo, v2, s85                       // s81 = s[sgprWorkGroup0] / s85
s_mov_b32 exec_lo, vcc_lo                          // s81 = s[sgprWorkGroup0] / s85
v_add_nc_u32 v0, v0, 1                             // s81 = s[sgprWorkGroup0] / s85
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v1, v0, s85                           // s81 = s[sgprWorkGroup0] / s85
v_sub_nc_u32 v2, s[sgprWorkGroup0], v1             // s81 = s[sgprWorkGroup0] / s85
v_readfirstlane_b32 s81, v0                        // quotient
v_readfirstlane_b32 s82, v2                        // remainder
s_mul_i32 s81, s81, s85
/* temp1 = (wg%CU_Count)//WGMXCC */
s_lshr_b32 s82, s82, s84
/* temp0 = temp0 + temp1 */
s_add_u32 s81, s81, s82
/* temp1 = (wg%WGMXCC) * ((WGs - (WGs//CU_Count) * CU_Count) if (wg > (WGs//CU_Count) * CU_Count) else CU_Count)//WGMXCC */
v_cvt_f64_u32 v[0:1], s85                          // s82 = s18 / s85
v_rcp_f64 v[0:1], v[0:1]                           // s82 = s18 / s85
v_cvt_f64_u32 v[2:3], s18                          // s82 = s18 / s85
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s82 = s18 / s85
v_cvt_u32_f64 v0, v[0:1]                           // s82 = s18 / s85
v_mul_lo_u32 v1, v0, s85                           // s82 = s18 / s85
v_sub_nc_u32 v2, s18, v1                           // s82 = s18 / s85
v_cmp_ge_u32 vcc_lo, v2, s85                       // s82 = s18 / s85
s_mov_b32 exec_lo, vcc_lo                          // s82 = s18 / s85
v_add_nc_u32 v0, v0, 1                             // s82 = s18 / s85
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s82, v0                        // quotient
s_mul_i32 s82, s82, s85
s_sub_u32 s83, s18, s82
s_cmp_gt_u32 s[sgprWorkGroup0], s82
s_cselect_b32 s82, s83, s85
s_lshr_b32 s82, s82, s84
s_bfm_b32 s83, s84, 0
s_and_b32 s83, s[sgprWorkGroup0], s83
s_mul_i32 s82, s82, s83
/* WorkGroup0 = temp0 + temp1 */
s_add_u32 s[sgprWorkGroup0], s81, s82
label_skip_WGMXCC:  /// skip WGMXCC if no enough WGs to remap
s_cmp_eq_u32 s16, 3
s_cbranch_scc1 label_ArgType3_Routed_To_ArgType0
s_cmp_eq_u32 s16, 0
s_cbranch_scc0 label_MultiGemm
label_ArgType3_Routed_To_ArgType0:
/* init: add vgpr [90...221) to pool */
/* init: add vgpr [0...48) to pool */
/* init: add agpr [0...0) to pool */
s_waitcnt lgkmcnt(0)                               // wait for 128/0 bytes of kern args over preload
s_lshr_b32 s[sgprNumWorkGroups0], s[sgprSizesFree+0], 7 // s[sgprNumWorkGroups0] = s[sgprSizesFree+0] / 128
s_and_b32 s12, 127, s[sgprSizesFree+0]             // s12 = s[sgprSizesFree+0] % 128
s_cmp_gt_u32 s12, 0                                // Check if remainder > 0
s_addc_u32 s[sgprNumWorkGroups0], s[sgprNumWorkGroups0], 0 // Increment if remainder > 0
s_mul_hi_u32 s15, s[sgprSizesFree+1], 89478486     // STATIC_DIV: divisor=96
s_mul_i32 s14, s[sgprSizesFree+1], 89478486        // tmp = dividend * magic
s_lshr_b64 s[14:15], s[14:15], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s[sgprNumWorkGroups1], s14               // quotient
s_mul_i32 s14, s[sgprNumWorkGroups1], 96           // quotient*divisor
s_sub_u32 s12, s[sgprSizesFree+1], s14             // rReg = dividend - quotient*divisor
s_cmp_gt_u32 s12, 0                                // Check if remainder > 0
s_addc_u32 s[sgprNumWorkGroups1], s[sgprNumWorkGroups1], 0 // Increment if remainder > 0
s_branch label_MultiGemmEnd
label_MultiGemm:

/* Check if custom structure pointer is null */
s_and_b32 s52, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s52, 2                                // ArgType == 2 ?
s_cbranch_scc1 label_IsExternalValid               // branch if ArgType == 2
s_mov_b32 s10, 152                                 // KernArgAddressOffset
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
s_lshr_b32 s16, s20, 7                             // s16 = s20 / 128
s_and_b32 s14, 127, s20                            // s14 = s20 % 128
s_addc_u32 s16, s16, 0
s_mul_hi_u32 s15, s21, 89478486                    // STATIC_DIV: divisor=96
s_mul_i32 s14, s21, 89478486                       // tmp = dividend * magic
s_lshr_b64 s[14:15], s[14:15], 33                  // tmp0 = quotient
s_mul_i32 s15, s14, 96                             // tmp1 = quotient * divisor
s_cmp_lg_u32 s15, s21                              // if (quotient * divisor != dividend), result+=1
s_addc_u32 s17, s14, 0                             // if (quotient * divisor != dividend), result+=1
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
s_lshr_b32 s16, s20, 7                             // s16 = s20 / 128
s_and_b32 s14, 127, s20                            // s14 = s20 % 128
s_addc_u32 s16, s16, 0
s_mul_hi_u32 s15, s21, 89478486                    // STATIC_DIV: divisor=96
s_mul_i32 s14, s21, 89478486                       // tmp = dividend * magic
s_lshr_b64 s[14:15], s[14:15], 33                  // tmp0 = quotient
s_mul_i32 s15, s14, 96                             // tmp1 = quotient * divisor
s_cmp_lg_u32 s15, s21                              // if (quotient * divisor != dividend), result+=1
s_addc_u32 s17, s14, 0                             // if (quotient * divisor != dividend), result+=1
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
s_mul_i32 s13, s13, 152                            // KernArgAddressOffset
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s13
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_b512 s[24:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_b256 s[40:47], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
s_load_b128 s[48:51], s[sgprKernArgAddress:sgprKernArgAddress+1], 112 // 112
s_load_b32 s[sgprScaleAGMinus1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0xb8
s_load_b64 s[sgprAddressScaleA:sgprAddressScaleA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x80
s_load_b64 s[sgprAddressScaleZeroA:sgprAddressScaleZeroA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x90
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
s_load_b64 s[sgprAddressScaleZeroA:sgprAddressScaleZeroA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x9c
label_LoadExternalStructEnd:
/* init: add vgpr [90...221) to pool */
/* init: add vgpr [0...48) to pool */
/* init: add agpr [0...0) to pool */
s_waitcnt lgkmcnt(0)                               // wait for 128/0 bytes of kern args over preload
s_lshr_b32 s[sgprNumWorkGroups0], s[sgprSizesFree+0], 7 // s[sgprNumWorkGroups0] = s[sgprSizesFree+0] / 128
s_and_b32 s12, 127, s[sgprSizesFree+0]             // s12 = s[sgprSizesFree+0] % 128
s_cmp_gt_u32 s12, 0                                // Check if remainder > 0
s_addc_u32 s[sgprNumWorkGroups0], s[sgprNumWorkGroups0], 0 // Increment if remainder > 0
s_mul_hi_u32 s15, s[sgprSizesFree+1], 89478486     // STATIC_DIV: divisor=96
s_mul_i32 s14, s[sgprSizesFree+1], 89478486        // tmp = dividend * magic
s_lshr_b64 s[14:15], s[14:15], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s[sgprNumWorkGroups1], s14               // quotient
s_mul_i32 s14, s[sgprNumWorkGroups1], 96           // quotient*divisor
s_sub_u32 s12, s[sgprSizesFree+1], s14             // rReg = dividend - quotient*divisor
s_cmp_gt_u32 s12, 0                                // Check if remainder > 0
s_addc_u32 s[sgprNumWorkGroups1], s[sgprNumWorkGroups1], 0 // Increment if remainder > 0

/* Early stop if N(SizeFreeJ) == 0 */
s_cmp_eq_u32 s[sgprSizeJ], 0
s_cbranch_scc0 label_NoEarlyStop_N0
label_EarlyStop_if_N_is_0:
s_endpgm
label_NoEarlyStop_N0:

label_MultiGemmEnd:
.set sgprSrdA, 80
.set sgprSrdB, 84
.set sgprShadowLimitA, 88
.set sgprShadowLimitB, 90
s_and_b32 s11, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s11, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_Skip_Address_Prepad_For_Pointer_Array
s_sub_u32 s[sgprAddressA+0], s[sgprAddressA+0], 8  // pre-pad to make room for possible pointer shift
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
v_lshrrev_b32 v3, 7, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(128)
v_and_b32 v3, 1, v3                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v1, v3, 11, v1                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(2048); 7. final local read offset: flrOffset = lrOffset + WOffset

/* local read addresses: final offsets a */
v_lshrrev_b32 v2, 5, v[vgprSerial]                 // 2 = Serial / 32
v_lshrrev_b32 v2, 3, v2                            // LSU offset: Get LSU wave_id
s_mov_b32 s11, 128                                 // LSU offset: stride = lsuStride(128) when umlds==True
v_mul_lo_u32 v2, s11, v2                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_nc_u32 v[vgprLocalReadAddrA], v2, v0         // Final Offset: offset = (lro0+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrA], 1, v[vgprLocalReadAddrA] //  (multiple bpe)
v_lshrrev_b32 v3, 8, v[vgprLocalReadAddrA]         // Final Offset: padding 16 per block 256
v_lshl_add_u32 v[vgprLocalReadAddrA], v3, 4, v[vgprLocalReadAddrA] // Final Offset: padding 16 per block 256

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 5, v[vgprSerial]                 // 0 = Serial / 32
v_lshrrev_b32 v0, 3, v0                            // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(128) when umlds==True (dup assign opt.)
v_mul_lo_u32 v0, s11, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_nc_u32 v[vgprLocalReadAddrB], v0, v1         // Final Offset: offset = (lro1+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrB], 1, v[vgprLocalReadAddrB] //  (multiple bpe)
v_lshrrev_b32 v2, 8, v[vgprLocalReadAddrB]         // Final Offset: padding 16 per block 256
v_lshl_add_u32 v[vgprLocalReadAddrB], v2, 4, v[vgprLocalReadAddrB] // Final Offset: padding 16 per block 256

/* local read addresses: declare addresses a */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, 0x8800, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 8 */
/* v1 = A-unroll = serial%LVCA */
v_lshrrev_b32 v0, 3, v[vgprSerial]                 // 0 = Serial / 8
v_and_b32 v1, 7, v[vgprSerial]                     // 1 = Serial % 8
/* unroll *= glvw */
v_lshlrev_b32 v1, 4, v1                            // v1 = v1 * 16
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
v_add_co_u32 v[vgprLocalWriteAddrB], vcc_lo, 0x8800, v[vgprLocalWriteAddrB] // lwFOB = lw1J + lwL*MT1J + LDS_OFFSET_B=34816

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
v_add_co_u32 v7, vcc_lo, 32, v6                    // groA0I_1 += LSPA
v_add_co_u32 v8, vcc_lo, 32, v7                    // groA0I_2 += LSPA
v_add_co_u32 v9, vcc_lo, 32, v8                    // groA0I_3 += LSPA

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

/* global read addresses: unroll offsets a */
v_mov_b32 v22, v1                                  // groAL_0

/* global read addresses: unroll offsets b */
v_mov_b32 v23, v3                                  // groBL_0

/* global read addresses: addresses a */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s15, s[sgprWorkGroup0], 128           // WorkGroup[01] * MT
s_mul_i32 s14, s[sgprWorkGroup0], 128              // WorkGroup[01] * MT
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
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 8 // extend limit for pre-pad
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
s_load_b64 s[12:13], s[sgprKernArgAddress:sgprKernArgAddress+1], 0xa8 // Load batchOffsetA from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s12        // Add batch offset to A address (low)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s13       // Add batch offset to A address (high)
s_sub_u32 s[sgprSrdA+0], s[sgprSrdA+0], 8          // pre-pad to make room for possible pointer shift
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
s_mul_i32 s12, s[sgprWorkGroup0], 128              // scaleA: workgroup row origin
s_mul_i32 s12, s12, s[sgprStrideScaleA]            // scaleA: * row stride
s_mul_i32 s12, s12, 2                              // scaleA: elements -> bytes
s_mul_i32 s13, s[sgprSizeI], s[sgprStrideScaleA]   // scaleA: SizeI * row stride
s_mul_i32 s13, s13, 2                              // scaleA: tensor bytes
s_sub_u32 s[sgprSrdScaleA+2], s13, s12             // scaleA: buffer limit from the workgroup origin
s_add_u32 s[sgprSrdScaleA+0], s[sgprAddressScaleA+0], s12 // scaleA: SRD base lo
s_addc_u32 s[sgprSrdScaleA+1], s[sgprAddressScaleA+1], 0 // scaleA: SRD base hi
s_mov_b32 s[sgprSrdScaleA+3], Srd127_96            // scaleA: set bits 127_96 in SRD
s_mov_b32 s[sgprScaleAPkMagic+0], 0xf0000f         // w4a16: nibble 2j at bit 0, nibble 2j+1 at bit 20

/* global read addresses: block-scale A zero-point srd */
s_mul_i32 s12, s[sgprWorkGroup0], 128              // scaleZeroA: workgroup row origin
s_lshr_b32 s12, s12, 1                             // scaleZeroA: aligned row origin to bytes
s_mul_i32 s12, s12, s[sgprStrideScaleA]            // scaleZeroA: * kGroups
s_add_u32 s13, s[sgprSizeI], 7                     // scaleZeroA: SizeI + 7
s_lshr_b32 s13, s13, 3                             // scaleZeroA: ceil(SizeI/8) words
s_lshl_b32 s13, s13, 2                             // scaleZeroA: words to bytes
s_mul_i32 s13, s13, s[sgprStrideScaleA]            // scaleZeroA: total bytes
s_sub_u32 s[sgprSrdScaleZeroA+2], s13, s12         // scaleZeroA: buffer limit from the workgroup origin
s_add_u32 s[sgprSrdScaleZeroA+0], s[sgprAddressScaleZeroA+0], s12 // scaleZeroA: SRD base lo
s_addc_u32 s[sgprSrdScaleZeroA+1], s[sgprAddressScaleZeroA+1], 0 // scaleZeroA: SRD base hi
s_mov_b32 s[sgprSrdScaleZeroA+3], Srd127_96        // scaleZeroA: set bits 127_96 in SRD
s_and_b32 s[sgprScaleAKCnt], s[sgprStreamKLocalStart], s[sgprScaleAKMask] // scaleA: position of the first iteration in its group
s_mul_i32 s12, s[sgprStreamKLocalStart], 128       // scaleA: first K of this range
s_lshr_b32 s12, s12, s[sgprScaleAGShift]           // scaleA: groups before this range
s_mul_i32 s13, s12, 2                              // scaleA: skipped bytes
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], s13 // scaleA: skip groups before the split (lower)
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA: skip groups before the split (upper)
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], s13 // scaleA: limit shrinks by the skip
s_lshl_b32 s13, s12, 2                             // scaleZeroA: skipped bytes
s_add_u32 s[sgprSrdScaleZeroA+0], s[sgprSrdScaleZeroA+0], s13 // scaleZeroA: skip groups before the split (lower)
s_addc_u32 s[sgprSrdScaleZeroA+1], s[sgprSrdScaleZeroA+1], 0 // scaleZeroA: skip groups before the split (upper)
s_sub_u32 s[sgprSrdScaleZeroA+2], s[sgprSrdScaleZeroA+2], s13 // scaleZeroA: limit shrinks by the skip

/* global read addresses: addresses b */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s15, s[sgprWorkGroup1], 96            // WorkGroup[01] * MT
s_mul_i32 s14, s[sgprWorkGroup1], 96               // WorkGroup[01] * MT
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
s_load_b64 s[12:13], s[sgprKernArgAddress:sgprKernArgAddress+1], 0xb0 // Load batchOffsetB from kernel args
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
v_mul_lo_u32 v24, s[sgprStrideA0I], v[6]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+0+0], vcc_lo, v[22], v[24+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+0+0], 0x10, v[vgprGlobalReadOffsetA+0+0] // add prepad for pointer shift
v_lshrrev_b32 v24, s[sgprScaleAGShift], v22        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+0], s[sgprStrideScaleA], v6 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+0], v24, v[vgprGlobalReadOffsetScaleA+0] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+0], 1, v[vgprGlobalReadOffsetScaleA+0] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetScaleZeroA+0], 3, v6 // scaleZeroA: row / 8
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleZeroA+0], s[sgprStrideScaleA], v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: (row/8) * kGroups
v_add_nc_u32 v[vgprGlobalReadOffsetScaleZeroA+0], v24, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: + kGroup -> word offset
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleZeroA+0], 3, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: word to nibble index
v_and_b32 v24, 7, v6                               // scaleZeroA: nibble = row & 7
v_or_b32 v[vgprGlobalReadOffsetScaleZeroA+0], v24, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: (word << 3) | nibble
v_lshrrev_b32 v[vgprGlobalReadByteOffsetScaleZeroA+0], 1, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: save loop-invariant byte offset
v_lshrrev_b32 v[vgprGlobalReadOffsetA+0], 1, v[vgprGlobalReadOffsetA+0] //  (multiple bpe)
v_mul_lo_u32 v24, s[sgprStrideA0I], v[7]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+1+0], vcc_lo, v[22], v[24+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+1+0], 0x10, v[vgprGlobalReadOffsetA+1+0] // add prepad for pointer shift
v_lshrrev_b32 v24, s[sgprScaleAGShift], v22        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+1], s[sgprStrideScaleA], v7 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+1], v24, v[vgprGlobalReadOffsetScaleA+1] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+1], 1, v[vgprGlobalReadOffsetScaleA+1] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetScaleZeroA+1], 3, v7 // scaleZeroA: row / 8
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleZeroA+1], s[sgprStrideScaleA], v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: (row/8) * kGroups
v_add_nc_u32 v[vgprGlobalReadOffsetScaleZeroA+1], v24, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: + kGroup -> word offset
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleZeroA+1], 3, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: word to nibble index
v_and_b32 v24, 7, v7                               // scaleZeroA: nibble = row & 7
v_or_b32 v[vgprGlobalReadOffsetScaleZeroA+1], v24, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: (word << 3) | nibble
v_lshrrev_b32 v[vgprGlobalReadByteOffsetScaleZeroA+1], 1, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: save loop-invariant byte offset
v_lshrrev_b32 v[vgprGlobalReadOffsetA+1], 1, v[vgprGlobalReadOffsetA+1] //  (multiple bpe)
v_mul_lo_u32 v24, s[sgprStrideA0I], v[8]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+2+0], vcc_lo, v[22], v[24+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+2+0], 0x10, v[vgprGlobalReadOffsetA+2+0] // add prepad for pointer shift
v_lshrrev_b32 v24, s[sgprScaleAGShift], v22        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+2], s[sgprStrideScaleA], v8 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+2], v24, v[vgprGlobalReadOffsetScaleA+2] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+2], 1, v[vgprGlobalReadOffsetScaleA+2] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetScaleZeroA+2], 3, v8 // scaleZeroA: row / 8
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleZeroA+2], s[sgprStrideScaleA], v[vgprGlobalReadOffsetScaleZeroA+2] // scaleZeroA: (row/8) * kGroups
v_add_nc_u32 v[vgprGlobalReadOffsetScaleZeroA+2], v24, v[vgprGlobalReadOffsetScaleZeroA+2] // scaleZeroA: + kGroup -> word offset
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleZeroA+2], 3, v[vgprGlobalReadOffsetScaleZeroA+2] // scaleZeroA: word to nibble index
v_and_b32 v24, 7, v8                               // scaleZeroA: nibble = row & 7
v_or_b32 v[vgprGlobalReadOffsetScaleZeroA+2], v24, v[vgprGlobalReadOffsetScaleZeroA+2] // scaleZeroA: (word << 3) | nibble
v_lshrrev_b32 v[vgprGlobalReadByteOffsetScaleZeroA+2], 1, v[vgprGlobalReadOffsetScaleZeroA+2] // scaleZeroA: save loop-invariant byte offset
v_lshrrev_b32 v[vgprGlobalReadOffsetA+2], 1, v[vgprGlobalReadOffsetA+2] //  (multiple bpe)
v_mul_lo_u32 v24, s[sgprStrideA0I], v[9]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+3+0], vcc_lo, v[22], v[24+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+3+0], 0x10, v[vgprGlobalReadOffsetA+3+0] // add prepad for pointer shift
v_lshrrev_b32 v24, s[sgprScaleAGShift], v22        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+3], s[sgprStrideScaleA], v9 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+3], v24, v[vgprGlobalReadOffsetScaleA+3] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+3], 1, v[vgprGlobalReadOffsetScaleA+3] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetScaleZeroA+3], 3, v9 // scaleZeroA: row / 8
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleZeroA+3], s[sgprStrideScaleA], v[vgprGlobalReadOffsetScaleZeroA+3] // scaleZeroA: (row/8) * kGroups
v_add_nc_u32 v[vgprGlobalReadOffsetScaleZeroA+3], v24, v[vgprGlobalReadOffsetScaleZeroA+3] // scaleZeroA: + kGroup -> word offset
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleZeroA+3], 3, v[vgprGlobalReadOffsetScaleZeroA+3] // scaleZeroA: word to nibble index
v_and_b32 v24, 7, v9                               // scaleZeroA: nibble = row & 7
v_or_b32 v[vgprGlobalReadOffsetScaleZeroA+3], v24, v[vgprGlobalReadOffsetScaleZeroA+3] // scaleZeroA: (word << 3) | nibble
v_lshrrev_b32 v[vgprGlobalReadByteOffsetScaleZeroA+3], 1, v[vgprGlobalReadOffsetScaleZeroA+3] // scaleZeroA: save loop-invariant byte offset
v_lshrrev_b32 v[vgprGlobalReadOffsetA+3], 1, v[vgprGlobalReadOffsetA+3] //  (multiple bpe)
/* ============================================================= */

/* global read addresses: final offsets b */
/* ============================================================= */
v_mul_lo_u32 v6, s[sgprStrideB1J], v[10]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+0+0], vcc_lo, v[23], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+0+0], 0x4, v[vgprGlobalReadOffsetB+0+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+0], 1, v[vgprGlobalReadOffsetB+0] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[11]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+1+0], vcc_lo, v[23], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+1+0], 0x4, v[vgprGlobalReadOffsetB+1+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+1], 1, v[vgprGlobalReadOffsetB+1] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[12]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+2+0], vcc_lo, v[23], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+2+0], 0x4, v[vgprGlobalReadOffsetB+2+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+2], 1, v[vgprGlobalReadOffsetB+2] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[13]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+3+0], vcc_lo, v[23], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+3+0], 0x4, v[vgprGlobalReadOffsetB+3+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+3], 1, v[vgprGlobalReadOffsetB+3] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[14]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+4+0], vcc_lo, v[23], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+4+0], 0x4, v[vgprGlobalReadOffsetB+4+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+4], 1, v[vgprGlobalReadOffsetB+4] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[15]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+5+0], vcc_lo, v[23], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+5+0], 0x4, v[vgprGlobalReadOffsetB+5+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+5], 1, v[vgprGlobalReadOffsetB+5] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[16]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+6+0], vcc_lo, v[23], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+6+0], 0x4, v[vgprGlobalReadOffsetB+6+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+6], 1, v[vgprGlobalReadOffsetB+6] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[17]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+7+0], vcc_lo, v[23], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+7+0], 0x4, v[vgprGlobalReadOffsetB+7+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+7], 1, v[vgprGlobalReadOffsetB+7] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[18]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+8+0], vcc_lo, v[23], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+8+0], 0x4, v[vgprGlobalReadOffsetB+8+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+8], 1, v[vgprGlobalReadOffsetB+8] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[19]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+9+0], vcc_lo, v[23], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+9+0], 0x4, v[vgprGlobalReadOffsetB+9+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+9], 1, v[vgprGlobalReadOffsetB+9] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[20]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+10+0], vcc_lo, v[23], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+10+0], 0x4, v[vgprGlobalReadOffsetB+10+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+10], 1, v[vgprGlobalReadOffsetB+10] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[21]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+11+0], vcc_lo, v[23], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+11+0], 0x4, v[vgprGlobalReadOffsetB+11+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+11], 1, v[vgprGlobalReadOffsetB+11] //  (multiple bpe)
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
buffer_load_b64 v[vgprG2LA+6:vgprG2LA+6+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b64 v[vgprG2LA+14:vgprG2LA+14+1], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b64 v[vgprG2LA+22:vgprG2LA+22+1], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b64 v[vgprG2LA+30:vgprG2LA+30+1], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_3_0

/* global read block-scale A */
buffer_load_d16_b16 v[vgprG2LScaleA+0], v[vgprGlobalReadOffsetScaleA+0], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 0
buffer_load_d16_b16 v[vgprG2LScaleA+1], v[vgprGlobalReadOffsetScaleA+1], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 1
buffer_load_d16_b16 v[vgprG2LScaleA+2], v[vgprGlobalReadOffsetScaleA+2], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 2
buffer_load_d16_b16 v[vgprG2LScaleA+3], v[vgprGlobalReadOffsetScaleA+3], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 3
buffer_load_d16_u8 v[vgprG2LScaleZeroA+0], v[vgprGlobalReadByteOffsetScaleZeroA+0], s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 0
buffer_load_d16_u8 v[vgprG2LScaleZeroA+1], v[vgprGlobalReadByteOffsetScaleZeroA+1], s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 1
buffer_load_d16_u8 v[vgprG2LScaleZeroA+2], v[vgprGlobalReadByteOffsetScaleZeroA+2], s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 2
buffer_load_d16_u8 v[vgprG2LScaleZeroA+3], v[vgprGlobalReadByteOffsetScaleZeroA+3], s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 3
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
s_lshl_b32 s13, s12, 1                             // scaleZeroA: advance only on a wrap
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], s12 // scaleA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], s12 // scaleA limit -= inc
s_add_u32 s[sgprSrdScaleZeroA+0], s[sgprSrdScaleZeroA+0], s13 // scaleZeroA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleZeroA+1], s[sgprSrdScaleZeroA+1], 0 // scaleZeroA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleZeroA+2], s[sgprSrdScaleZeroA+2], s13 // scaleZeroA limit -= inc

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

s_mul_i32 s94, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_mul_hi_u32 s93, s94, s[sgprStrideC1J]            // ScaleC s94 by Stride
s_mul_i32 s92, s94, s[sgprStrideC1J]               // ScaleC s94 by Stride
s_lshl_b64 s[92:93], s[92:93], s11                 // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s92        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s93       // add hi to SRD
s_mul_hi_u32 s93, s94, s[sgprStrideD1J]            // ScaleD s94 by Stride
s_mul_i32 s92, s94, s[sgprStrideD1J]               // ScaleD s94 by Stride
s_lshl_b64 s[92:93], s[92:93], s52                 // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s92        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s93       // add hi to SRD

s_and_b32 s53, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s53, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_StridedBatchedGemmLoadC
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_GeneralBatchedGemmLoadC
label_StridedBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s93, s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride
s_mul_i32 s92, s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride
s_lshl_b64 s[92:93], s[92:93], s11                 // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s92        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s93       // add hi to SRD
s_branch label_GeneralBatchedGemmLoadC_End
label_GeneralBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for General Batched GEMM
s_mul_i32 s92, 8, s[sgprWorkGroup2]                // Compute stride in bytes into Pointer Array
s_add_u32 s92, s92, s[sgprAddressC+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s93, s[sgprAddressC+1], 0               // Offsetting to the location [Higher half of address]
s_load_b64 s[92:93], s[92:93], 0                   // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for the Matrix Address Load from the Pointer Array
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s92        // Offsetting within the Batch Matrix [Lower half of address]
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s93       // Offsetting within the Batch Matrix [Higher half of address]
s_load_b64 s[92:93], s[sgprKernArgAddress:sgprKernArgAddress+1], 0xa0 // Load batchOffsetC from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s92        // Add matrix address to SRD (low)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s93       // Add matrix address to SRD (high)
label_GeneralBatchedGemmLoadC_End:  /// End of label GeneralBatchedGemmLoadC
s_and_b32 s53, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s53, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_StridedBatchedGemmLoadD
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_GeneralBatchedGemmLoadD
label_StridedBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s93, s[sgprWorkGroup2], s[sgprStrideDK] // ScaleD s[sgprWorkGroup2] by Stride
s_mul_i32 s92, s[sgprWorkGroup2], s[sgprStrideDK]  // ScaleD s[sgprWorkGroup2] by Stride
s_lshl_b64 s[92:93], s[92:93], s52                 // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s92        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s93       // add hi to SRD
s_branch label_GeneralBatchedGemmLoadD_End
label_GeneralBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for General Batched GEMM
s_mul_i32 s92, 8, s[sgprWorkGroup2]                // Compute stride in bytes into Pointer Array
s_add_u32 s92, s92, s[sgprAddressD+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s93, s[sgprAddressD+1], 0               // Offsetting to the location [Higher half of address]
s_load_b64 s[92:93], s[92:93], 0                   // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for the Matrix Address Load from the Pointer Array
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s92        // Offsetting within the Batch Matrix [Lower half of address]
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s93       // Offsetting within the Batch Matrix [Higher half of address]
s_load_b64 s[92:93], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x98 // Load batchOffsetD from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s92        // Add matrix address to SRD (low)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s93       // Add matrix address to SRD (high)
label_GeneralBatchedGemmLoadD_End:  /// End of label GeneralBatchedGemmLoadD

s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_SK_SplitSrd                   // Skip this block if using single-kernel stream-k fixup
s_cmp_eq_u32 s[sgprskTiles], 1                     // split == 1 ?
s_cbranch_scc1 label_SK_SplitSrd                   // branch if split == 1
// Split Output Buffer offset: Free0 + (Free1-1)*StrideC1J + (Free2-1)*StrideCK * SplitIdx * bpe%s
s_mul_hi_u32 s93, s[sgprSizesFree+0], s[sgprSkPartialIdx] // Free0
s_mul_i32 s92, s[sgprSizesFree+0], s[sgprSkPartialIdx] // Free0
s_sub_u32 s94, s[sgprSizesFree+1], 1               // Free1
s_mul_i32 s94, s94, s[sgprSkPartialIdx]            // Free1
s_mul_hi_u32 s95, s94, s[sgprStrideC1J]            // Free1
s_mul_i32 s94, s94, s[sgprStrideC1J]               // Free1
s_add_u32 s92, s92, s94                            // Free1
s_addc_u32 s93, s93, s95                           // Free1
s_sub_u32 s94, s[sgprSizesFree+2], 1               // Free2
s_mul_i32 s94, s94, s[sgprSkPartialIdx]            // Free2
s_mul_hi_u32 s95, s94, s[sgprStrideCK]             // Free2
s_mul_i32 s94, s94, s[sgprStrideCK]                // Free2
s_add_u32 s92, s92, s94                            // Free2
s_addc_u32 s93, s93, s95                           // Free2
s_lshl_b64 s[92:93], s[92:93], 2                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s92        // add lo GSU offset to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s93       // add hi GSU offset to SRD
label_SK_SplitSrd:

/* initC: remove ValuC vgpr buffer [0...48) from pool */

/* initC: remove acc vgpr buffer [0...0) from pool */

/* initC: remove ValuA/B vgpr buffer [90...131) from pool */
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
v_pack_b32_f16 v192, v[vgprG2LScaleA+0], v[vgprG2LScaleA+0] // scaleA: duplicate fp16 scale
v_and_b32 v189, 1, v[vgprGlobalReadOffsetScaleZeroA+0]
v_lshlrev_b32 v189, 2, v189
v_bfe_u32 v189, v[vgprG2LScaleZeroA+0], v189, 4
v_or_b32 v193, 0xd400e400, v189                    // scaleZeroA: -(1024 + c)
v_lshl_or_b32 v193, v189, 20, v193                 // scaleZeroA: -(64 + c) : -(1024 + c)

/* w4a16: save packed int4 dword 0 */
v_perm_b32 v195, v[vgprG2LA+6+0], v[vgprG2LA+6+0], 0xc000c00 // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v196, v[vgprG2LA+6+0], v[vgprG2LA+6+0], 0xc010c01 // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v197, v[vgprG2LA+6+0], v[vgprG2LA+6+0], 0xc020c02 // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v198, v[vgprG2LA+6+0], v[vgprG2LA+6+0], 0xc030c03 // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v195, v195, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v196, v196, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v197, v197, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v198, v198, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v195, v195, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v196, v196, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v197, v197, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v198, v198, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_fma_f16 v[vgprG2LA+0+0], v195, v192, 0        // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_fma_f16 v[vgprG2LA+0+1], v196, v192, 0        // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_fma_f16 v[vgprG2LA+0+2], v197, v192, 0        // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_fma_f16 v[vgprG2LA+0+3], v198, v192, 0        // w4a16: (q - z)*s, packed, fp16 pair 3

/* w4a16: save packed int4 dword 1 */
v_perm_b32 v195, v[vgprG2LA+6+1], v[vgprG2LA+6+1], 0xc000c00 // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v196, v[vgprG2LA+6+1], v[vgprG2LA+6+1], 0xc010c01 // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v197, v[vgprG2LA+6+1], v[vgprG2LA+6+1], 0xc020c02 // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v198, v[vgprG2LA+6+1], v[vgprG2LA+6+1], 0xc030c03 // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v195, v195, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v196, v196, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v197, v197, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v198, v198, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v195, v195, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v196, v196, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v197, v197, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v198, v198, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_fma_f16 v[vgprG2LA+0+4], v195, v192, 0        // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_fma_f16 v[vgprG2LA+0+5], v196, v192, 0        // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_fma_f16 v[vgprG2LA+0+6], v197, v192, 0        // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_fma_f16 v[vgprG2LA+0+7], v198, v192, 0        // w4a16: (q - z)*s, packed, fp16 pair 3
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:16 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
v_pack_b32_f16 v192, v[vgprG2LScaleA+1], v[vgprG2LScaleA+1] // scaleA: duplicate fp16 scale
v_and_b32 v189, 1, v[vgprGlobalReadOffsetScaleZeroA+1]
v_lshlrev_b32 v189, 2, v189
v_bfe_u32 v189, v[vgprG2LScaleZeroA+1], v189, 4
v_or_b32 v193, 0xd400e400, v189                    // scaleZeroA: -(1024 + c)
v_lshl_or_b32 v193, v189, 20, v193                 // scaleZeroA: -(64 + c) : -(1024 + c)

/* w4a16: save packed int4 dword 0 */
v_perm_b32 v195, v[vgprG2LA+14+0], v[vgprG2LA+14+0], 0xc000c00 // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v196, v[vgprG2LA+14+0], v[vgprG2LA+14+0], 0xc010c01 // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v197, v[vgprG2LA+14+0], v[vgprG2LA+14+0], 0xc020c02 // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v198, v[vgprG2LA+14+0], v[vgprG2LA+14+0], 0xc030c03 // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v195, v195, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v196, v196, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v197, v197, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v198, v198, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v195, v195, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v196, v196, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v197, v197, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v198, v198, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_fma_f16 v[vgprG2LA+8+0], v195, v192, 0        // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_fma_f16 v[vgprG2LA+8+1], v196, v192, 0        // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_fma_f16 v[vgprG2LA+8+2], v197, v192, 0        // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_fma_f16 v[vgprG2LA+8+3], v198, v192, 0        // w4a16: (q - z)*s, packed, fp16 pair 3

/* w4a16: save packed int4 dword 1 */
v_perm_b32 v195, v[vgprG2LA+14+1], v[vgprG2LA+14+1], 0xc000c00 // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v196, v[vgprG2LA+14+1], v[vgprG2LA+14+1], 0xc010c01 // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v197, v[vgprG2LA+14+1], v[vgprG2LA+14+1], 0xc020c02 // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v198, v[vgprG2LA+14+1], v[vgprG2LA+14+1], 0xc030c03 // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v195, v195, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v196, v196, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v197, v197, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v198, v198, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v195, v195, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v196, v196, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v197, v197, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v198, v198, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_fma_f16 v[vgprG2LA+8+4], v195, v192, 0        // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_fma_f16 v[vgprG2LA+8+5], v196, v192, 0        // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_fma_f16 v[vgprG2LA+8+6], v197, v192, 0        // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_fma_f16 v[vgprG2LA+8+7], v198, v192, 0        // w4a16: (q - z)*s, packed, fp16 pair 3
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:8704 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 8704 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+12:vgprG2LA+12+3] offset:8720 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 8704 sync LDS0
v_pack_b32_f16 v192, v[vgprG2LScaleA+2], v[vgprG2LScaleA+2] // scaleA: duplicate fp16 scale
v_and_b32 v189, 1, v[vgprGlobalReadOffsetScaleZeroA+2]
v_lshlrev_b32 v189, 2, v189
v_bfe_u32 v189, v[vgprG2LScaleZeroA+2], v189, 4
v_or_b32 v193, 0xd400e400, v189                    // scaleZeroA: -(1024 + c)
v_lshl_or_b32 v193, v189, 20, v193                 // scaleZeroA: -(64 + c) : -(1024 + c)

/* w4a16: save packed int4 dword 0 */
v_perm_b32 v195, v[vgprG2LA+22+0], v[vgprG2LA+22+0], 0xc000c00 // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v196, v[vgprG2LA+22+0], v[vgprG2LA+22+0], 0xc010c01 // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v197, v[vgprG2LA+22+0], v[vgprG2LA+22+0], 0xc020c02 // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v198, v[vgprG2LA+22+0], v[vgprG2LA+22+0], 0xc030c03 // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v195, v195, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v196, v196, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v197, v197, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v198, v198, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v195, v195, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v196, v196, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v197, v197, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v198, v198, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_fma_f16 v[vgprG2LA+16+0], v195, v192, 0       // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_fma_f16 v[vgprG2LA+16+1], v196, v192, 0       // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_fma_f16 v[vgprG2LA+16+2], v197, v192, 0       // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_fma_f16 v[vgprG2LA+16+3], v198, v192, 0       // w4a16: (q - z)*s, packed, fp16 pair 3

/* w4a16: save packed int4 dword 1 */
v_perm_b32 v195, v[vgprG2LA+22+1], v[vgprG2LA+22+1], 0xc000c00 // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v196, v[vgprG2LA+22+1], v[vgprG2LA+22+1], 0xc010c01 // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v197, v[vgprG2LA+22+1], v[vgprG2LA+22+1], 0xc020c02 // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v198, v[vgprG2LA+22+1], v[vgprG2LA+22+1], 0xc030c03 // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v195, v195, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v196, v196, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v197, v197, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v198, v198, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v195, v195, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v196, v196, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v197, v197, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v198, v198, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_fma_f16 v[vgprG2LA+16+4], v195, v192, 0       // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_fma_f16 v[vgprG2LA+16+5], v196, v192, 0       // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_fma_f16 v[vgprG2LA+16+6], v197, v192, 0       // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_fma_f16 v[vgprG2LA+16+7], v198, v192, 0       // w4a16: (q - z)*s, packed, fp16 pair 3
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+16:vgprG2LA+16+3] offset:17408 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 17408 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+20:vgprG2LA+20+3] offset:17424 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 17408 sync LDS0
v_pack_b32_f16 v192, v[vgprG2LScaleA+3], v[vgprG2LScaleA+3] // scaleA: duplicate fp16 scale
v_and_b32 v189, 1, v[vgprGlobalReadOffsetScaleZeroA+3]
v_lshlrev_b32 v189, 2, v189
v_bfe_u32 v189, v[vgprG2LScaleZeroA+3], v189, 4
v_or_b32 v193, 0xd400e400, v189                    // scaleZeroA: -(1024 + c)
v_lshl_or_b32 v193, v189, 20, v193                 // scaleZeroA: -(64 + c) : -(1024 + c)

/* w4a16: save packed int4 dword 0 */
v_perm_b32 v195, v[vgprG2LA+30+0], v[vgprG2LA+30+0], 0xc000c00 // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v196, v[vgprG2LA+30+0], v[vgprG2LA+30+0], 0xc010c01 // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v197, v[vgprG2LA+30+0], v[vgprG2LA+30+0], 0xc020c02 // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v198, v[vgprG2LA+30+0], v[vgprG2LA+30+0], 0xc030c03 // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v195, v195, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v196, v196, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v197, v197, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v198, v198, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v195, v195, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v196, v196, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v197, v197, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v198, v198, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_fma_f16 v[vgprG2LA+24+0], v195, v192, 0       // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_fma_f16 v[vgprG2LA+24+1], v196, v192, 0       // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_fma_f16 v[vgprG2LA+24+2], v197, v192, 0       // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_fma_f16 v[vgprG2LA+24+3], v198, v192, 0       // w4a16: (q - z)*s, packed, fp16 pair 3

/* w4a16: save packed int4 dword 1 */
v_perm_b32 v195, v[vgprG2LA+30+1], v[vgprG2LA+30+1], 0xc000c00 // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v196, v[vgprG2LA+30+1], v[vgprG2LA+30+1], 0xc010c01 // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v197, v[vgprG2LA+30+1], v[vgprG2LA+30+1], 0xc020c02 // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v198, v[vgprG2LA+30+1], v[vgprG2LA+30+1], 0xc030c03 // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v195, v195, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v196, v196, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v197, v197, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v198, v198, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v195, v195, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v196, v196, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v197, v197, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v198, v198, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_fma_f16 v[vgprG2LA+24+4], v195, v192, 0       // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_fma_f16 v[vgprG2LA+24+5], v196, v192, 0       // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_fma_f16 v[vgprG2LA+24+6], v197, v192, 0       // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_fma_f16 v[vgprG2LA+24+7], v198, v192, 0       // w4a16: (q - z)*s, packed, fp16 pair 3
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+24:vgprG2LA+24+3] offset:26112 // lwoA_0_0_3_0 = (0*LSCA)*(MT0I+PAD) + (3*LSPA) = 26112 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+28:vgprG2LA+28+3] offset:26128 // lwoA_0_0_3_0 = (0*LSCA)*(MT0I+PAD) + (3*LSPA) = 26112 sync LDS0

/* local write b */
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+1] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+2:vgprG2LB+2+1] offset:2176 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 2176 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4:vgprG2LB+4+1] offset:4352 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 4352 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+6:vgprG2LB+6+1] offset:6528 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 6528 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8:vgprG2LB+8+1] offset:8704 // lwoB_0_0_4_0 = (0*LSCB)*(MT1J+PAD) + (4*LSPB) = 8704 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+10:vgprG2LB+10+1] offset:10880 // lwoB_0_0_5_0 = (0*LSCB)*(MT1J+PAD) + (5*LSPB) = 10880 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12:vgprG2LB+12+1] offset:13056 // lwoB_0_0_6_0 = (0*LSCB)*(MT1J+PAD) + (6*LSPB) = 13056 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+14:vgprG2LB+14+1] offset:15232 // lwoB_0_0_7_0 = (0*LSCB)*(MT1J+PAD) + (7*LSPB) = 15232 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+16:vgprG2LB+16+1] offset:17408 // lwoB_0_0_8_0 = (0*LSCB)*(MT1J+PAD) + (8*LSPB) = 17408 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+18:vgprG2LB+18+1] offset:19584 // lwoB_0_0_9_0 = (0*LSCB)*(MT1J+PAD) + (9*LSPB) = 19584 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+20:vgprG2LB+20+1] offset:21760 // lwoB_0_0_10_0 = (0*LSCB)*(MT1J+PAD) + (10*LSPB) = 21760 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+22:vgprG2LB+22+1] offset:23936 // lwoB_0_0_11_0 = (0*LSCB)*(MT1J+PAD) + (11*LSPB) = 23936 sync LDS0

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
/*  grEndMfmaIndex:16, lwStartMfmaIndex:47, lwEndMfmaIndex:47  */
/*  numMfmaForLR:5, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:46 */
/*  mfmaIndex:0  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:17408 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:17424 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:8704 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:8720 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b64 v[vgprG2LA+6:vgprG2LA+6+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:1  */
buffer_load_b64 v[vgprG2LA+14:vgprG2LA+14+1], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_1_0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:2  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:17408 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:17424 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b64 v[vgprG2LA+22:vgprG2LA+22+1], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_2_0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:3  */
/* localReadsVacancy: latencyLeft 5 */
buffer_load_b64 v[vgprG2LA+30:vgprG2LA+30+1], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_3_0
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:4  */
/* localReadsVacancy: latencyLeft 5 */

/* global read block-scale A */
buffer_load_d16_b16 v[vgprG2LScaleA+0], v[vgprGlobalReadOffsetScaleA+0], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 0
buffer_load_d16_b16 v[vgprG2LScaleA+1], v[vgprGlobalReadOffsetScaleA+1], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 1
buffer_load_d16_b16 v[vgprG2LScaleA+2], v[vgprGlobalReadOffsetScaleA+2], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 2
buffer_load_d16_b16 v[vgprG2LScaleA+3], v[vgprGlobalReadOffsetScaleA+3], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 3
buffer_load_d16_u8 v[vgprG2LScaleZeroA+0], v[vgprGlobalReadByteOffsetScaleZeroA+0], s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 0
buffer_load_d16_u8 v[vgprG2LScaleZeroA+1], v[vgprGlobalReadByteOffsetScaleZeroA+1], s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 1
buffer_load_d16_u8 v[vgprG2LScaleZeroA+2], v[vgprGlobalReadByteOffsetScaleZeroA+2], s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 2
buffer_load_d16_u8 v[vgprG2LScaleZeroA+3], v[vgprGlobalReadByteOffsetScaleZeroA+3], s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 3
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:5  */
/* localReadsVacancy: latencyLeft 5 */
buffer_load_b64 v[vgprG2LB+0:vgprG2LB+0+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=6 */

/* iter 1 */
/*  grEndMfmaIndex:16, lwStartMfmaIndex:47, lwEndMfmaIndex:47  */
/*  numMfmaForLR:5, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:46 */
/*  mfmaIndex:6  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:17440 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:17456 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:8736 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:8752 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b64 v[vgprG2LB+2:vgprG2LB+2+1], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:7  */
buffer_load_b64 v[vgprG2LB+4:vgprG2LB+4+1], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:8  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:17440 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:17456 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b64 v[vgprG2LB+6:vgprG2LB+6+1], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:9  */
/* localReadsVacancy: latencyLeft 5 */
buffer_load_b64 v[vgprG2LB+8:vgprG2LB+8+1], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_4_0
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:10  */
/* localReadsVacancy: latencyLeft 5 */
buffer_load_b64 v[vgprG2LB+10:vgprG2LB+10+1], v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_5_0
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:11  */
/* localReadsVacancy: latencyLeft 5 */
buffer_load_b64 v[vgprG2LB+12:vgprG2LB+12+1], v[vgprGlobalReadOffsetB+6], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_6_0
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=0 readsPerIterB=6 */

/* iter 2 */
/*  grEndMfmaIndex:16, lwStartMfmaIndex:47, lwEndMfmaIndex:47  */
/*  numMfmaForLR:5, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:46 */
/*  mfmaIndex:12  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:17472 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:17488 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:8768 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:8784 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b64 v[vgprG2LB+14:vgprG2LB+14+1], v[vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_7_0
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:13  */
buffer_load_b64 v[vgprG2LB+16:vgprG2LB+16+1], v[vgprGlobalReadOffsetB+8], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_8_0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:14  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:17472 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:17488 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b64 v[vgprG2LB+18:vgprG2LB+18+1], v[vgprGlobalReadOffsetB+9], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_9_0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:15  */
/* localReadsVacancy: latencyLeft 5 */
buffer_load_b64 v[vgprG2LB+20:vgprG2LB+20+1], v[vgprGlobalReadOffsetB+10], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_10_0
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:16  */
/* localReadsVacancy: latencyLeft 5 */
buffer_load_b64 v[vgprG2LB+22:vgprG2LB+22+1], v[vgprGlobalReadOffsetB+11], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_11_0
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:17  */
/* localReadsVacancy: latencyLeft 5 */

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], GlobalReadIncsA // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], GlobalReadIncsA // limit -= inc)
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=0 readsPerIterB=6 */

/* iter 3 */
/*  grEndMfmaIndex:16, lwStartMfmaIndex:47, lwEndMfmaIndex:47  */
/*  numMfmaForLR:5, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:46 */
/*  mfmaIndex:18  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:17504 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:17520 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:8800 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:8816 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:19  */

/* global read inc block-scale A (2 bytes every 1 iters) */
s_add_u32 s[sgprScaleAKCnt], s[sgprScaleAKCnt], 1  // scaleA: one more K iteration done
s_and_b32 s52, s[sgprScaleAKCnt], s[sgprScaleAKMask] // scaleA: SCC = counter has not wrapped
s_cselect_b32 s52, 0, s[sgprScaleAInc]             // scaleA: advance only on a wrap
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:20  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:17504 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:17520 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_lshl_b32 s53, s52, 1                             // scaleZeroA: advance only on a wrap
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], s52 // scaleA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA SRD += inc(upper)
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:21  */
/* localReadsVacancy: latencyLeft 5 */
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], s52 // scaleA limit -= inc
s_add_u32 s[sgprSrdScaleZeroA+0], s[sgprSrdScaleZeroA+0], s53 // scaleZeroA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleZeroA+1], s[sgprSrdScaleZeroA+1], 0 // scaleZeroA SRD += inc(upper)
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:22  */
/* localReadsVacancy: latencyLeft 5 */
s_sub_u32 s[sgprSrdScaleZeroA+2], s[sgprSrdScaleZeroA+2], s53 // scaleZeroA limit -= inc

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], GlobalReadIncsB // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0x0       // gra SRD += inc(upper)
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:23  */
/* localReadsVacancy: latencyLeft 5 */
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], GlobalReadIncsB // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/* numPrefetchIter=0 */
/* dataAtIterA=3 numReadsIterA=4 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=3 numReadsIterB=4 skipReadsIterB=0 readsPerIterB=6 */

/* iter 4 */
/*  grEndMfmaIndex:16, lwStartMfmaIndex:47, lwEndMfmaIndex:47  */
/*  numMfmaForLR:5, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:46 */
/*  mfmaIndex:24  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:144 // L -> Reg lro=64 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:144 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:17536 // L -> Reg lro=64 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:17552 // L -> Reg lro=64 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:8832 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:8848 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:25  */
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:26  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:17536 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:17552 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:27  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:28  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:29  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/* numPrefetchIter=0 */
/* dataAtIterA=4 numReadsIterA=5 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=4 numReadsIterB=5 skipReadsIterB=0 readsPerIterB=6 */

/* iter 5 */
/*  grEndMfmaIndex:16, lwStartMfmaIndex:47, lwEndMfmaIndex:47  */
/*  numMfmaForLR:5, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:46 */
/*  mfmaIndex:30  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:160 // L -> Reg lro=80 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:176 // L -> Reg lro=80 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:160 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:176 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:17568 // L -> Reg lro=80 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:17584 // L -> Reg lro=80 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:8864 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:8880 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:31  */
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:32  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:17568 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:17584 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:33  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:34  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:35  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/* numPrefetchIter=0 */
/* dataAtIterA=5 numReadsIterA=6 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=5 numReadsIterB=6 skipReadsIterB=0 readsPerIterB=6 */

/* iter 6 */
/*  grEndMfmaIndex:16, lwStartMfmaIndex:47, lwEndMfmaIndex:47  */
/*  numMfmaForLR:5, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:46 */
/*  mfmaIndex:36  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:208 // L -> Reg lro=96 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:208 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:17600 // L -> Reg lro=96 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:17616 // L -> Reg lro=96 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:8896 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:8912 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:37  */
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:38  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:17600 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:17616 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:39  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:40  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:41  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/* numPrefetchIter=0 */
/* dataAtIterA=6 numReadsIterA=7 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=6 numReadsIterB=7 skipReadsIterB=0 readsPerIterB=6 */

/* iter 7 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:16, lwStartMfmaIndex:47, lwEndMfmaIndex:47  */
/*  numMfmaForLR:5, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:46 */
/*  mfmaIndex:42  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:224 // L -> Reg lro=112 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:240 // L -> Reg lro=112 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:224 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:240 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:17632 // L -> Reg lro=112 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:17648 // L -> Reg lro=112 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:8928 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:8944 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:43  */
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:44  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:17632 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:17648 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:45  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:46  */
/* schedule remaining localreads for one buffer scheduling */
/* localReadsVacancy: latencyLeft 5 */
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:47  */
/* sched write - iter 7 writesPerItem=1 */
s_waitcnt vmcnt(15)                                // wait for global read before writing to local
v_pack_b32_f16 v192, v[vgprG2LScaleA+0], v[vgprG2LScaleA+0] // scaleA: duplicate fp16 scale
v_and_b32 v189, 1, v[vgprGlobalReadOffsetScaleZeroA+0]
v_lshlrev_b32 v189, 2, v189
v_bfe_u32 v189, v[vgprG2LScaleZeroA+0], v189, 4
v_or_b32 v193, 0xd400e400, v189                    // scaleZeroA: -(1024 + c)
v_lshl_or_b32 v193, v189, 20, v193                 // scaleZeroA: -(64 + c) : -(1024 + c)

/* w4a16: save packed int4 dword 0 */
v_perm_b32 v195, v[vgprG2LA+6+0], v[vgprG2LA+6+0], 0xc000c00 // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v196, v[vgprG2LA+6+0], v[vgprG2LA+6+0], 0xc010c01 // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v197, v[vgprG2LA+6+0], v[vgprG2LA+6+0], 0xc020c02 // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v198, v[vgprG2LA+6+0], v[vgprG2LA+6+0], 0xc030c03 // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v195, v195, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v196, v196, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v197, v197, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v198, v198, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v195, v195, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v196, v196, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v197, v197, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v198, v198, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_fma_f16 v[vgprG2LA+0+0], v195, v192, 0        // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_fma_f16 v[vgprG2LA+0+1], v196, v192, 0        // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_fma_f16 v[vgprG2LA+0+2], v197, v192, 0        // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_fma_f16 v[vgprG2LA+0+3], v198, v192, 0        // w4a16: (q - z)*s, packed, fp16 pair 3

/* w4a16: save packed int4 dword 1 */
v_perm_b32 v195, v[vgprG2LA+6+1], v[vgprG2LA+6+1], 0xc000c00 // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v196, v[vgprG2LA+6+1], v[vgprG2LA+6+1], 0xc010c01 // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v197, v[vgprG2LA+6+1], v[vgprG2LA+6+1], 0xc020c02 // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v198, v[vgprG2LA+6+1], v[vgprG2LA+6+1], 0xc030c03 // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v195, v195, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v196, v196, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v197, v197, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v198, v198, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v195, v195, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v196, v196, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v197, v197, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v198, v198, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_fma_f16 v[vgprG2LA+0+4], v195, v192, 0        // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_fma_f16 v[vgprG2LA+0+5], v196, v192, 0        // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_fma_f16 v[vgprG2LA+0+6], v197, v192, 0        // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_fma_f16 v[vgprG2LA+0+7], v198, v192, 0        // w4a16: (q - z)*s, packed, fp16 pair 3
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:16 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
/* sched write - iter 7 writesPerItem=1 */
s_waitcnt vmcnt(14)                                // wait for global read before writing to local
v_pack_b32_f16 v192, v[vgprG2LScaleA+1], v[vgprG2LScaleA+1] // scaleA: duplicate fp16 scale
v_and_b32 v189, 1, v[vgprGlobalReadOffsetScaleZeroA+1]
v_lshlrev_b32 v189, 2, v189
v_bfe_u32 v189, v[vgprG2LScaleZeroA+1], v189, 4
v_or_b32 v193, 0xd400e400, v189                    // scaleZeroA: -(1024 + c)
v_lshl_or_b32 v193, v189, 20, v193                 // scaleZeroA: -(64 + c) : -(1024 + c)

/* w4a16: save packed int4 dword 0 */
v_perm_b32 v195, v[vgprG2LA+14+0], v[vgprG2LA+14+0], 0xc000c00 // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v196, v[vgprG2LA+14+0], v[vgprG2LA+14+0], 0xc010c01 // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v197, v[vgprG2LA+14+0], v[vgprG2LA+14+0], 0xc020c02 // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v198, v[vgprG2LA+14+0], v[vgprG2LA+14+0], 0xc030c03 // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v195, v195, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v196, v196, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v197, v197, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v198, v198, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v195, v195, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v196, v196, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v197, v197, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v198, v198, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_fma_f16 v[vgprG2LA+8+0], v195, v192, 0        // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_fma_f16 v[vgprG2LA+8+1], v196, v192, 0        // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_fma_f16 v[vgprG2LA+8+2], v197, v192, 0        // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_fma_f16 v[vgprG2LA+8+3], v198, v192, 0        // w4a16: (q - z)*s, packed, fp16 pair 3

/* w4a16: save packed int4 dword 1 */
v_perm_b32 v195, v[vgprG2LA+14+1], v[vgprG2LA+14+1], 0xc000c00 // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v196, v[vgprG2LA+14+1], v[vgprG2LA+14+1], 0xc010c01 // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v197, v[vgprG2LA+14+1], v[vgprG2LA+14+1], 0xc020c02 // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v198, v[vgprG2LA+14+1], v[vgprG2LA+14+1], 0xc030c03 // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v195, v195, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v196, v196, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v197, v197, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v198, v198, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v195, v195, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v196, v196, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v197, v197, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v198, v198, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_fma_f16 v[vgprG2LA+8+4], v195, v192, 0        // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_fma_f16 v[vgprG2LA+8+5], v196, v192, 0        // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_fma_f16 v[vgprG2LA+8+6], v197, v192, 0        // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_fma_f16 v[vgprG2LA+8+7], v198, v192, 0        // w4a16: (q - z)*s, packed, fp16 pair 3
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:8704 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 8704 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+12:vgprG2LA+12+3] offset:8720 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 8704 sync LDS0
/* sched write - iter 7 writesPerItem=1 */
s_waitcnt vmcnt(13)                                // wait for global read before writing to local
v_pack_b32_f16 v192, v[vgprG2LScaleA+2], v[vgprG2LScaleA+2] // scaleA: duplicate fp16 scale
v_and_b32 v189, 1, v[vgprGlobalReadOffsetScaleZeroA+2]
v_lshlrev_b32 v189, 2, v189
v_bfe_u32 v189, v[vgprG2LScaleZeroA+2], v189, 4
v_or_b32 v193, 0xd400e400, v189                    // scaleZeroA: -(1024 + c)
v_lshl_or_b32 v193, v189, 20, v193                 // scaleZeroA: -(64 + c) : -(1024 + c)

/* w4a16: save packed int4 dword 0 */
v_perm_b32 v195, v[vgprG2LA+22+0], v[vgprG2LA+22+0], 0xc000c00 // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v196, v[vgprG2LA+22+0], v[vgprG2LA+22+0], 0xc010c01 // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v197, v[vgprG2LA+22+0], v[vgprG2LA+22+0], 0xc020c02 // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v198, v[vgprG2LA+22+0], v[vgprG2LA+22+0], 0xc030c03 // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v195, v195, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v196, v196, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v197, v197, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v198, v198, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v195, v195, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v196, v196, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v197, v197, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v198, v198, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_fma_f16 v[vgprG2LA+16+0], v195, v192, 0       // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_fma_f16 v[vgprG2LA+16+1], v196, v192, 0       // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_fma_f16 v[vgprG2LA+16+2], v197, v192, 0       // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_fma_f16 v[vgprG2LA+16+3], v198, v192, 0       // w4a16: (q - z)*s, packed, fp16 pair 3

/* w4a16: save packed int4 dword 1 */
v_perm_b32 v195, v[vgprG2LA+22+1], v[vgprG2LA+22+1], 0xc000c00 // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v196, v[vgprG2LA+22+1], v[vgprG2LA+22+1], 0xc010c01 // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v197, v[vgprG2LA+22+1], v[vgprG2LA+22+1], 0xc020c02 // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v198, v[vgprG2LA+22+1], v[vgprG2LA+22+1], 0xc030c03 // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v195, v195, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v196, v196, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v197, v197, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v198, v198, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v195, v195, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v196, v196, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v197, v197, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v198, v198, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_fma_f16 v[vgprG2LA+16+4], v195, v192, 0       // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_fma_f16 v[vgprG2LA+16+5], v196, v192, 0       // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_fma_f16 v[vgprG2LA+16+6], v197, v192, 0       // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_fma_f16 v[vgprG2LA+16+7], v198, v192, 0       // w4a16: (q - z)*s, packed, fp16 pair 3
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+16:vgprG2LA+16+3] offset:17408 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 17408 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+20:vgprG2LA+20+3] offset:17424 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 17408 sync LDS0
/* sched write - iter 7 writesPerItem=1 */
s_waitcnt vmcnt(12)                                // wait for global read before writing to local
v_pack_b32_f16 v192, v[vgprG2LScaleA+3], v[vgprG2LScaleA+3] // scaleA: duplicate fp16 scale
v_and_b32 v189, 1, v[vgprGlobalReadOffsetScaleZeroA+3]
v_lshlrev_b32 v189, 2, v189
v_bfe_u32 v189, v[vgprG2LScaleZeroA+3], v189, 4
v_or_b32 v193, 0xd400e400, v189                    // scaleZeroA: -(1024 + c)
v_lshl_or_b32 v193, v189, 20, v193                 // scaleZeroA: -(64 + c) : -(1024 + c)

/* w4a16: save packed int4 dword 0 */
v_perm_b32 v195, v[vgprG2LA+30+0], v[vgprG2LA+30+0], 0xc000c00 // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v196, v[vgprG2LA+30+0], v[vgprG2LA+30+0], 0xc010c01 // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v197, v[vgprG2LA+30+0], v[vgprG2LA+30+0], 0xc020c02 // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v198, v[vgprG2LA+30+0], v[vgprG2LA+30+0], 0xc030c03 // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v195, v195, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v196, v196, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v197, v197, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v198, v198, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v195, v195, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v196, v196, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v197, v197, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v198, v198, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_fma_f16 v[vgprG2LA+24+0], v195, v192, 0       // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_fma_f16 v[vgprG2LA+24+1], v196, v192, 0       // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_fma_f16 v[vgprG2LA+24+2], v197, v192, 0       // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_fma_f16 v[vgprG2LA+24+3], v198, v192, 0       // w4a16: (q - z)*s, packed, fp16 pair 3

/* w4a16: save packed int4 dword 1 */
v_perm_b32 v195, v[vgprG2LA+30+1], v[vgprG2LA+30+1], 0xc000c00 // w4a16: byte 0 (int4 #0, #1) into bytes 0 and 2
v_perm_b32 v196, v[vgprG2LA+30+1], v[vgprG2LA+30+1], 0xc010c01 // w4a16: byte 1 (int4 #2, #3) into bytes 0 and 2
v_perm_b32 v197, v[vgprG2LA+30+1], v[vgprG2LA+30+1], 0xc020c02 // w4a16: byte 2 (int4 #4, #5) into bytes 0 and 2
v_perm_b32 v198, v[vgprG2LA+30+1], v[vgprG2LA+30+1], 0xc030c03 // w4a16: byte 3 (int4 #6, #7) into bytes 0 and 2
v_and_or_b32 v195, v195, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #0, #1 -> fp16 64+q : 1024+q
v_and_or_b32 v196, v196, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #2, #3 -> fp16 64+q : 1024+q
v_and_or_b32 v197, v197, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #4, #5 -> fp16 64+q : 1024+q
v_and_or_b32 v198, v198, s[sgprScaleAPkMagic+0], 0x54006400 // w4a16: int4 #6, #7 -> fp16 64+q : 1024+q
v_pk_add_f16 v195, v195, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v196, v196, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v197, v197, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_add_f16 v198, v198, v193                      // w4a16: q - z (exact: both sides are integers near the magics)
v_pk_fma_f16 v[vgprG2LA+24+4], v195, v192, 0       // w4a16: (q - z)*s, packed, fp16 pair 0
v_pk_fma_f16 v[vgprG2LA+24+5], v196, v192, 0       // w4a16: (q - z)*s, packed, fp16 pair 1
v_pk_fma_f16 v[vgprG2LA+24+6], v197, v192, 0       // w4a16: (q - z)*s, packed, fp16 pair 2
v_pk_fma_f16 v[vgprG2LA+24+7], v198, v192, 0       // w4a16: (q - z)*s, packed, fp16 pair 3
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+24:vgprG2LA+24+3] offset:26112 // lwoA_0_0_3_0 = (0*LSCA)*(MT0I+PAD) + (3*LSPA) = 26112 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+28:vgprG2LA+28+3] offset:26128 // lwoA_0_0_3_0 = (0*LSCA)*(MT0I+PAD) + (3*LSPA) = 26112 sync LDS0
/* sched write - iter 7 writesPerItem=1 */
s_waitcnt vmcnt(11)                                // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+1] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0
/* sched write - iter 7 writesPerItem=1 */
s_waitcnt vmcnt(10)                                // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+2:vgprG2LB+2+1] offset:2176 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 2176 sync LDS0
/* sched write - iter 7 writesPerItem=1 */
s_waitcnt vmcnt(9)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4:vgprG2LB+4+1] offset:4352 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 4352 sync LDS0
/* sched write - iter 7 writesPerItem=1 */
s_waitcnt vmcnt(8)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+6:vgprG2LB+6+1] offset:6528 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 6528 sync LDS0
/* sched write - iter 7 writesPerItem=1 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8:vgprG2LB+8+1] offset:8704 // lwoB_0_0_4_0 = (0*LSCB)*(MT1J+PAD) + (4*LSPB) = 8704 sync LDS0
/* sched write - iter 7 writesPerItem=1 */
s_waitcnt vmcnt(6)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+10:vgprG2LB+10+1] offset:10880 // lwoB_0_0_5_0 = (0*LSCB)*(MT1J+PAD) + (5*LSPB) = 10880 sync LDS0
/* sched write - iter 7 writesPerItem=1 */
s_waitcnt vmcnt(5)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12:vgprG2LB+12+1] offset:13056 // lwoB_0_0_6_0 = (0*LSCB)*(MT1J+PAD) + (6*LSPB) = 13056 sync LDS0
/* sched write - iter 7 writesPerItem=1 */
s_waitcnt vmcnt(4)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+14:vgprG2LB+14+1] offset:15232 // lwoB_0_0_7_0 = (0*LSCB)*(MT1J+PAD) + (7*LSPB) = 15232 sync LDS0
/* sched write - iter 7 writesPerItem=1 */
s_waitcnt vmcnt(3)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+16:vgprG2LB+16+1] offset:17408 // lwoB_0_0_8_0 = (0*LSCB)*(MT1J+PAD) + (8*LSPB) = 17408 sync LDS0
/* sched write - iter 7 writesPerItem=1 */
s_waitcnt vmcnt(2)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+18:vgprG2LB+18+1] offset:19584 // lwoB_0_0_9_0 = (0*LSCB)*(MT1J+PAD) + (9*LSPB) = 19584 sync LDS0
/* sched write - iter 7 writesPerItem=1 */
s_waitcnt vmcnt(1)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+20:vgprG2LB+20+1] offset:21760 // lwoB_0_0_10_0 = (0*LSCB)*(MT1J+PAD) + (10*LSPB) = 21760 sync LDS0
/* sched write - iter 7 writesPerItem=1 */
s_waitcnt vmcnt(0)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+22:vgprG2LB+22+1] offset:23936 // lwoB_0_0_11_0 = (0*LSCB)*(MT1J+PAD) + (11*LSPB) = 23936 sync LDS0

/* local write swap offsets a */

/* local write swap offsets b */

/* local read swap offsets a */

/* local read swap offsets b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/* numPrefetchIter=0 */
/* dataAtIterA=7 numReadsIterA=8 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=7 numReadsIterB=8 skipReadsIterB=0 readsPerIterB=6 */

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
/*  grEndMfmaIndex:0, lwStartMfmaIndex:47, lwEndMfmaIndex:47  */
/*  numMfmaForLR:5, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:46 */
/*  mfmaIndex:0  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:17408 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:17424 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:8704 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:8720 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:1  */
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:2  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:17408 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:17424 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:3  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:4  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:5  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=6 */

/* iter 1 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:47, lwEndMfmaIndex:47  */
/*  numMfmaForLR:5, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:46 */
/*  mfmaIndex:6  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:17440 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:17456 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:8736 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:8752 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:7  */
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:8  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:17440 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:17456 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:9  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:10  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:11  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=0 readsPerIterB=6 */

/* iter 2 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:47, lwEndMfmaIndex:47  */
/*  numMfmaForLR:5, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:46 */
/*  mfmaIndex:12  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:17472 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:17488 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:8768 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:8784 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:13  */
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:14  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:17472 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:17488 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:15  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:16  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:17  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=0 readsPerIterB=6 */

/* iter 3 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:47, lwEndMfmaIndex:47  */
/*  numMfmaForLR:5, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:46 */
/*  mfmaIndex:18  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:17504 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:17520 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:8800 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:8816 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:19  */
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:20  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:17504 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:17520 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:21  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:22  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:23  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/* numPrefetchIter=0 */
/* dataAtIterA=3 numReadsIterA=4 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=3 numReadsIterB=4 skipReadsIterB=0 readsPerIterB=6 */

/* iter 4 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:47, lwEndMfmaIndex:47  */
/*  numMfmaForLR:5, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:46 */
/*  mfmaIndex:24  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:144 // L -> Reg lro=64 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:144 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:17536 // L -> Reg lro=64 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:17552 // L -> Reg lro=64 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:8832 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:8848 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:25  */
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:26  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:17536 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:17552 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:27  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:28  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:29  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/* numPrefetchIter=0 */
/* dataAtIterA=4 numReadsIterA=5 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=4 numReadsIterB=5 skipReadsIterB=0 readsPerIterB=6 */

/* iter 5 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:47, lwEndMfmaIndex:47  */
/*  numMfmaForLR:5, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:46 */
/*  mfmaIndex:30  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:160 // L -> Reg lro=80 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:176 // L -> Reg lro=80 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:160 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:176 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:17568 // L -> Reg lro=80 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:17584 // L -> Reg lro=80 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:8864 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:8880 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:31  */
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:32  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:17568 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:17584 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:33  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:34  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:35  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/* numPrefetchIter=0 */
/* dataAtIterA=5 numReadsIterA=6 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=5 numReadsIterB=6 skipReadsIterB=0 readsPerIterB=6 */

/* iter 6 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:47, lwEndMfmaIndex:47  */
/*  numMfmaForLR:5, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:46 */
/*  mfmaIndex:36  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:208 // L -> Reg lro=96 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:208 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:17600 // L -> Reg lro=96 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:17616 // L -> Reg lro=96 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:8896 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:8912 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:37  */
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:38  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:17600 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:17616 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:39  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:40  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:41  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/* numPrefetchIter=0 */
/* dataAtIterA=6 numReadsIterA=7 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=6 numReadsIterB=7 skipReadsIterB=0 readsPerIterB=6 */

/* iter 7 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:47, lwEndMfmaIndex:47  */
/*  numMfmaForLR:5, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:46 */
/*  mfmaIndex:42  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:224 // L -> Reg lro=112 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:240 // L -> Reg lro=112 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:224 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:240 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:17632 // L -> Reg lro=112 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:17648 // L -> Reg lro=112 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:8928 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:8944 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:43  */
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:44  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:17632 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:17648 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:45  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:46  */
/* schedule remaining localreads for one buffer scheduling */
/* localReadsVacancy: latencyLeft 5 */
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:47  */
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/* numPrefetchIter=0 */
/* dataAtIterA=7 numReadsIterA=8 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=7 numReadsIterB=8 skipReadsIterB=0 readsPerIterB=6 */
label_toPGR1end_OrdNLL:
label_PrefetchGlobalLastIterEnd:

/* Tail: add ValuA/B vgpr buffer [90...131) to pool */

/* Tail: add address/G2L vgpr [131...188) to pool */
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
.set sgprSrdScaleZeroA, UNDEF
.set sgprSrdA, UNDEF
.set sgprSrdB, UNDEF
.set sgprShadowLimitA, UNDEF
.set sgprShadowLimitB, UNDEF
/* load store sgprs */

/* Mapping of Acc register -> C Vgpr register */

/* Multiply MI out register with Alpha -> C Vgpr register */

/* not-LocalSplitU: global write indices */
/* computeStoreVgprs */
v_lshrrev_b32 v94, 5, v[vgprSerial]                // 94 = Serial / 32
v_lshrrev_b32 v95, 2, v94                          // 95 = 94 / 4
v_mul_lo_u32 v95, 0x10, v95                        // wave coordination offset 1
v_and_b32 v91, 31, v[vgprSerial]                   // v91 = v[vgprSerial] % 32
v_lshrrev_b32 v91, 4, v91                          // 91 = 91 / 16
                                                   // thread0 * continuous_output (multiplier is 1, do nothing)
v_add_lshl_u32 v91, v95, v91, 0                    // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v92, v91, s[sgprStrideC1J]            //  offset 1
v_mul_lo_u32 v93, v91, s[sgprStrideD1J]            //  offset 1
v_and_b32 v90, 3, v94                              // v90 = v94 % 4
v_mul_lo_u32 v90, 0x10, v90                        // wave coordination offset 0
v_and_b32 v95, 15, v[vgprSerial]                   // v95 = v[vgprSerial] % 16
v_add_lshl_u32 v90, v95, v90, 0                    // coordination 0 = vwA * (wave_id0 + tid0)
s_mul_i32 s7, 128, s[sgprWorkGroup0]               // wgp0 * MT0
v_add_nc_u32 v90, s7, v90                          // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s7, 96, s[sgprWorkGroup1]                // wgp1 * MT1
v_add_nc_u32 v91, s7, v91                          // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1

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

/* Edge/NonEdge store path check (M): Size % 128 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s64, 127, s[sgprSizeI]                   // s64 = s[sgprSizeI] % 128
s_add_u32 s65, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s65                // wg0 >= nwg0-1 ?
s_cselect_b32 s64, s64, 0                          // set rem
s_cmpk_gt_u32 s64, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_MB_Else         // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 96 > 0 -> Edge store; else -> NonEdge store */
s_mul_hi_u32 s67, s[sgprSizeJ], 89478486           // STATIC_DIV: divisor=96
s_mul_i32 s66, s[sgprSizeJ], 89478486              // tmp = dividend * magic
s_lshr_b64 s[66:67], s[66:67], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s65, s66                                 // quotient
s_mul_i32 s66, s65, 96                             // quotient*divisor
s_sub_u32 s64, s[sgprSizeJ], s66                   // rReg = dividend - quotient*divisor
s_add_u32 s65, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s65                // wg1 >= nwg1-1
s_cselect_b32 s64, s64, 0                          // set rem
s_cmpk_gt_u32 s64, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_MB_Then         // jump if edges required
label_GW_B0_FD0_VW1_MB_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=152 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1) */
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
v_add_lshl_u32 v101, v93, v90, 2                   // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=90, coord0Vgpr=90 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (19, 0, 0, 0), (19, 1, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0), (22, 0, 0, 0), (22, 1, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0)] */
v_mov_b32 v[vgprValuC+103], v[vgprValuC+0]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+104], v[vgprValuC+8]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+105], v[vgprValuC+1]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+106], v[vgprValuC+9]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+107], v[vgprValuC+2]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+108], v[vgprValuC+10]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+109], v[vgprValuC+3]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+110], v[vgprValuC+11]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+111], v[vgprValuC+4]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+112], v[vgprValuC+12]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+113], v[vgprValuC+5]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+114], v[vgprValuC+13]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+115], v[vgprValuC+6]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+116], v[vgprValuC+14]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+117], v[vgprValuC+7]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+118], v[vgprValuC+15]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+119], v[vgprValuC+16]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+120], v[vgprValuC+24]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+121], v[vgprValuC+17]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+122], v[vgprValuC+25]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+123], v[vgprValuC+18]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+124], v[vgprValuC+26]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+125], v[vgprValuC+19]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+126], v[vgprValuC+27]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+127], v[vgprValuC+20]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+128], v[vgprValuC+28]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+129], v[vgprValuC+21]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+130], v[vgprValuC+29]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+131], v[vgprValuC+22]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+132], v[vgprValuC+30]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+133], v[vgprValuC+23]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+134], v[vgprValuC+31]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+135], v[vgprValuC+32]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+136], v[vgprValuC+40]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+137], v[vgprValuC+33]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+138], v[vgprValuC+41]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+139], v[vgprValuC+34]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+140], v[vgprValuC+42]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+141], v[vgprValuC+35]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+142], v[vgprValuC+43]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+143], v[vgprValuC+36]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+144], v[vgprValuC+44]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+145], v[vgprValuC+37]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+146], v[vgprValuC+45]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+147], v[vgprValuC+38]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+148], v[vgprValuC+46]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+149], v[vgprValuC+39]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+150], v[vgprValuC+47]        // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b32 v103, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v104, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v105, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v106, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v107, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v108, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v109, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v110, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v111, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v112, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v113, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v114, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v115, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v116, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v117, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v118, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 72                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b32 v119, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v120, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v121, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v122, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v123, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v124, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v125, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v126, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v127, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v128, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v129, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v130, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v131, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v132, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v133, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v134, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 72                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b32 v135, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v136, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v137, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v138, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v139, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v140, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v141, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v142, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v143, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v144, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v145, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v146, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v147, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v148, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v149, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v150, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_B0_FD0_VW1_MB_NonEdgeEnd:
label_GW_B0_FD0_VW1_MB_Else:
label_GW_B0_FD0_VW1_MB_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=76 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v96, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v149, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v149, v96, v149, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v150, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v150, v96, v150, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v151, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v151, v96, v151, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v152, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v152, v96, v152, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v153, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v153, v96, v153, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v154, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v96, v154, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v155, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v155, v96, v155, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v156, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v96, v156, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v157, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v157, v96, v157, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v158, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v96, v158, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v159, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v159, v96, v159, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v160, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v160, v96, v160, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v161, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v161, v96, v161, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v162, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v162, v96, v162, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v163, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v163, v96, v163, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v164, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v164, v96, v164, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 18                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v165, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v165, v96, v165, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v166, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v166, v96, v166, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v167, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v167, v96, v167, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v168, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v168, v96, v168, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v169, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v169, v96, v169, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v170, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v170, v96, v170, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v171, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v171, v96, v171, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v172, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v172, v96, v172, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v173, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v173, v96, v173, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v174, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v174, v96, v174, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v175, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v175, v96, v175, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v176, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v176, v96, v176, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v177, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v177, v96, v177, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v178, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v178, v96, v178, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v179, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v179, v96, v179, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v180, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v180, v96, v180, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 18                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v181, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v181, v96, v181, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v182, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v182, v96, v182, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v183, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v183, v96, v183, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v184, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v184, v96, v184, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v185, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v185, v96, v185, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v186, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v186, v96, v186, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v187, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v187, v96, v187, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v189, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v189, v96, v189, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v190, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v190, v96, v190, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v191, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v191, v96, v191, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v192, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v192, v96, v192, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v193, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v193, v96, v193, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v194, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v194, v96, v194, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v195, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v195, v96, v195, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v196, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v196, v96, v196, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v197, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v197, v96, v197, s66                 // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (19, 0, 0, 0), (19, 1, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0), (22, 0, 0, 0), (22, 1, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0)] */
v_mov_b32 v[vgprValuC+101], v[vgprValuC+0]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+102], v[vgprValuC+8]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+103], v[vgprValuC+1]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+104], v[vgprValuC+9]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+105], v[vgprValuC+2]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+106], v[vgprValuC+10]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+107], v[vgprValuC+3]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+108], v[vgprValuC+11]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+109], v[vgprValuC+4]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+110], v[vgprValuC+12]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+111], v[vgprValuC+5]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+112], v[vgprValuC+13]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+113], v[vgprValuC+6]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+114], v[vgprValuC+14]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+115], v[vgprValuC+7]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+116], v[vgprValuC+15]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+117], v[vgprValuC+16]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+118], v[vgprValuC+24]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+119], v[vgprValuC+17]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+120], v[vgprValuC+25]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+121], v[vgprValuC+18]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+122], v[vgprValuC+26]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+123], v[vgprValuC+19]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+124], v[vgprValuC+27]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+125], v[vgprValuC+20]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+126], v[vgprValuC+28]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+127], v[vgprValuC+21]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+128], v[vgprValuC+29]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+129], v[vgprValuC+22]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+130], v[vgprValuC+30]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+131], v[vgprValuC+23]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+132], v[vgprValuC+31]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+133], v[vgprValuC+32]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+134], v[vgprValuC+40]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+135], v[vgprValuC+33]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+136], v[vgprValuC+41]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+137], v[vgprValuC+34]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+138], v[vgprValuC+42]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+139], v[vgprValuC+35]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+140], v[vgprValuC+43]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+141], v[vgprValuC+36]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+142], v[vgprValuC+44]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+143], v[vgprValuC+37]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+144], v[vgprValuC+45]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+145], v[vgprValuC+38]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+146], v[vgprValuC+46]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+147], v[vgprValuC+39]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+148], v[vgprValuC+47]        // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b32 v101, v149, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v102, v150, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v103, v151, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v104, v152, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v105, v153, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v106, v154, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v107, v155, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v108, v156, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v109, v157, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v110, v158, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v111, v159, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v112, v160, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v113, v161, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v114, v162, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v115, v163, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v116, v164, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v117, v165, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v118, v166, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v119, v167, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v120, v168, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v121, v169, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v122, v170, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v123, v171, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v124, v172, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v125, v173, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v126, v174, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v127, v175, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v128, v176, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v129, v177, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v130, v178, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v131, v179, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v132, v180, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v133, v181, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v134, v182, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v135, v183, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v136, v184, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v137, v185, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v138, v186, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v139, v187, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v140, v189, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v141, v190, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v142, v191, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v143, v192, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v144, v193, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v145, v194, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v146, v195, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v147, v196, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v148, v197, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
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
v_mov_b32 v101, s66                                // move flag value to vgpr
v_mov_b32 v102, 0                                  // zero vaddr offset
s_mov_b64 s[76:77], s[sgprAddressFlags:sgprAddressFlags+1]
s_mov_b32 s78, BufferOOB
s_mov_b32 s79, Srd127_96
buffer_store_b32 v101, v102, s[76:79], s64 offen offset:0 glc dlc // reset flag
s_waitcnt_vscnt null 0                             // wait for data store
label_SK_SkipFlagReset:
label_Fixup_E0:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=74 */
s_mov_b64 s[sgprSrdWS:sgprSrdWS+1], s[sgprAddressWS:sgprAddressWS+1]
s_mov_b32 s[sgprSrdWS+2], BufferOOB
s_mov_b32 s[sgprSrdWS+3], Srd127_96

s_mul_i32 s64, 0xc000, s7                          // Offset to correct partials tile (low word)
s_mul_hi_u32 s11, 0xc000, s7                       // partials tile offset (high word) for 64-bit SRD
s_add_u32 s[sgprSrdWS+0], s[sgprSrdWS+0], s64      // add lo to SRD
s_addc_u32 s[sgprSrdWS+1], s[sgprSrdWS+1], s11     // add hi (offset high word + lo carry) to SRD
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Fixup Batch #0 (d1,d0,vc1,vc0) =       */
/*      (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_lshlrev_b32 v102, 2, v[vgprSerial]               // v102 = v[vgprSerial] * 4
s_mov_b32 s64, 0                                   // Init sgpr offset
buffer_load_b32 v151, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v152, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v153, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v154, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v155, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v156, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v157, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v158, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v159, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v160, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v161, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v162, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v163, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v164, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v165, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v166, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v167, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v168, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v169, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v170, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v171, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v172, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v173, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v174, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v175, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v176, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v177, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v178, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v179, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v180, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v181, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v182, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v183, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v184, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v185, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v186, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v187, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v189, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v190, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v191, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v192, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v193, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v194, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v195, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v196, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v197, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v198, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_b32 v199, v102, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
v_mov_b32 v[vgprValuC+103], v[vgprValuC+0]         // copy MI out reg to vreg[0]
v_mov_b32 v[vgprValuC+104], v[vgprValuC+8]         // copy MI out reg to vreg[1]
v_mov_b32 v[vgprValuC+105], v[vgprValuC+1]         // copy MI out reg to vreg[2]
v_mov_b32 v[vgprValuC+106], v[vgprValuC+9]         // copy MI out reg to vreg[3]
v_mov_b32 v[vgprValuC+107], v[vgprValuC+2]         // copy MI out reg to vreg[4]
v_mov_b32 v[vgprValuC+108], v[vgprValuC+10]        // copy MI out reg to vreg[5]
v_mov_b32 v[vgprValuC+109], v[vgprValuC+3]         // copy MI out reg to vreg[6]
v_mov_b32 v[vgprValuC+110], v[vgprValuC+11]        // copy MI out reg to vreg[7]
v_mov_b32 v[vgprValuC+111], v[vgprValuC+4]         // copy MI out reg to vreg[8]
v_mov_b32 v[vgprValuC+112], v[vgprValuC+12]        // copy MI out reg to vreg[9]
v_mov_b32 v[vgprValuC+113], v[vgprValuC+5]         // copy MI out reg to vreg[10]
v_mov_b32 v[vgprValuC+114], v[vgprValuC+13]        // copy MI out reg to vreg[11]
v_mov_b32 v[vgprValuC+115], v[vgprValuC+6]         // copy MI out reg to vreg[12]
v_mov_b32 v[vgprValuC+116], v[vgprValuC+14]        // copy MI out reg to vreg[13]
v_mov_b32 v[vgprValuC+117], v[vgprValuC+7]         // copy MI out reg to vreg[14]
v_mov_b32 v[vgprValuC+118], v[vgprValuC+15]        // copy MI out reg to vreg[15]
v_mov_b32 v[vgprValuC+119], v[vgprValuC+16]        // copy MI out reg to vreg[16]
v_mov_b32 v[vgprValuC+120], v[vgprValuC+24]        // copy MI out reg to vreg[17]
v_mov_b32 v[vgprValuC+121], v[vgprValuC+17]        // copy MI out reg to vreg[18]
v_mov_b32 v[vgprValuC+122], v[vgprValuC+25]        // copy MI out reg to vreg[19]
v_mov_b32 v[vgprValuC+123], v[vgprValuC+18]        // copy MI out reg to vreg[20]
v_mov_b32 v[vgprValuC+124], v[vgprValuC+26]        // copy MI out reg to vreg[21]
v_mov_b32 v[vgprValuC+125], v[vgprValuC+19]        // copy MI out reg to vreg[22]
v_mov_b32 v[vgprValuC+126], v[vgprValuC+27]        // copy MI out reg to vreg[23]
v_mov_b32 v[vgprValuC+127], v[vgprValuC+20]        // copy MI out reg to vreg[24]
v_mov_b32 v[vgprValuC+128], v[vgprValuC+28]        // copy MI out reg to vreg[25]
v_mov_b32 v[vgprValuC+129], v[vgprValuC+21]        // copy MI out reg to vreg[26]
v_mov_b32 v[vgprValuC+130], v[vgprValuC+29]        // copy MI out reg to vreg[27]
v_mov_b32 v[vgprValuC+131], v[vgprValuC+22]        // copy MI out reg to vreg[28]
v_mov_b32 v[vgprValuC+132], v[vgprValuC+30]        // copy MI out reg to vreg[29]
v_mov_b32 v[vgprValuC+133], v[vgprValuC+23]        // copy MI out reg to vreg[30]
v_mov_b32 v[vgprValuC+134], v[vgprValuC+31]        // copy MI out reg to vreg[31]
v_mov_b32 v[vgprValuC+135], v[vgprValuC+32]        // copy MI out reg to vreg[32]
v_mov_b32 v[vgprValuC+136], v[vgprValuC+40]        // copy MI out reg to vreg[33]
v_mov_b32 v[vgprValuC+137], v[vgprValuC+33]        // copy MI out reg to vreg[34]
v_mov_b32 v[vgprValuC+138], v[vgprValuC+41]        // copy MI out reg to vreg[35]
v_mov_b32 v[vgprValuC+139], v[vgprValuC+34]        // copy MI out reg to vreg[36]
v_mov_b32 v[vgprValuC+140], v[vgprValuC+42]        // copy MI out reg to vreg[37]
v_mov_b32 v[vgprValuC+141], v[vgprValuC+35]        // copy MI out reg to vreg[38]
v_mov_b32 v[vgprValuC+142], v[vgprValuC+43]        // copy MI out reg to vreg[39]
v_mov_b32 v[vgprValuC+143], v[vgprValuC+36]        // copy MI out reg to vreg[40]
v_mov_b32 v[vgprValuC+144], v[vgprValuC+44]        // copy MI out reg to vreg[41]
v_mov_b32 v[vgprValuC+145], v[vgprValuC+37]        // copy MI out reg to vreg[42]
v_mov_b32 v[vgprValuC+146], v[vgprValuC+45]        // copy MI out reg to vreg[43]
v_mov_b32 v[vgprValuC+147], v[vgprValuC+38]        // copy MI out reg to vreg[44]
v_mov_b32 v[vgprValuC+148], v[vgprValuC+46]        // copy MI out reg to vreg[45]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+39]        // copy MI out reg to vreg[46]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+47]        // copy MI out reg to vreg[47]

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 0 - 1
v_add_f32 v[vgprValuC+103], v[vgprValuC+103], v151 // accum partials

s_waitcnt vmcnt(46)                                // wait C (interleaved) 46 = 48 - 1 - 1
v_add_f32 v[vgprValuC+104], v[vgprValuC+104], v152 // accum partials

s_waitcnt vmcnt(45)                                // wait C (interleaved) 45 = 48 - 2 - 1
v_add_f32 v[vgprValuC+105], v[vgprValuC+105], v153 // accum partials

s_waitcnt vmcnt(44)                                // wait C (interleaved) 44 = 48 - 3 - 1
v_add_f32 v[vgprValuC+106], v[vgprValuC+106], v154 // accum partials

s_waitcnt vmcnt(43)                                // wait C (interleaved) 43 = 48 - 4 - 1
v_add_f32 v[vgprValuC+107], v[vgprValuC+107], v155 // accum partials

s_waitcnt vmcnt(42)                                // wait C (interleaved) 42 = 48 - 5 - 1
v_add_f32 v[vgprValuC+108], v[vgprValuC+108], v156 // accum partials

s_waitcnt vmcnt(41)                                // wait C (interleaved) 41 = 48 - 6 - 1
v_add_f32 v[vgprValuC+109], v[vgprValuC+109], v157 // accum partials

s_waitcnt vmcnt(40)                                // wait C (interleaved) 40 = 48 - 7 - 1
v_add_f32 v[vgprValuC+110], v[vgprValuC+110], v158 // accum partials

s_waitcnt vmcnt(39)                                // wait C (interleaved) 39 = 48 - 8 - 1
v_add_f32 v[vgprValuC+111], v[vgprValuC+111], v159 // accum partials

s_waitcnt vmcnt(38)                                // wait C (interleaved) 38 = 48 - 9 - 1
v_add_f32 v[vgprValuC+112], v[vgprValuC+112], v160 // accum partials

s_waitcnt vmcnt(37)                                // wait C (interleaved) 37 = 48 - 10 - 1
v_add_f32 v[vgprValuC+113], v[vgprValuC+113], v161 // accum partials

s_waitcnt vmcnt(36)                                // wait C (interleaved) 36 = 48 - 11 - 1
v_add_f32 v[vgprValuC+114], v[vgprValuC+114], v162 // accum partials

s_waitcnt vmcnt(35)                                // wait C (interleaved) 35 = 48 - 12 - 1
v_add_f32 v[vgprValuC+115], v[vgprValuC+115], v163 // accum partials

s_waitcnt vmcnt(34)                                // wait C (interleaved) 34 = 48 - 13 - 1
v_add_f32 v[vgprValuC+116], v[vgprValuC+116], v164 // accum partials

s_waitcnt vmcnt(33)                                // wait C (interleaved) 33 = 48 - 14 - 1
v_add_f32 v[vgprValuC+117], v[vgprValuC+117], v165 // accum partials

s_waitcnt vmcnt(32)                                // wait C (interleaved) 32 = 48 - 15 - 1
v_add_f32 v[vgprValuC+118], v[vgprValuC+118], v166 // accum partials

s_waitcnt vmcnt(31)                                // wait C (interleaved) 31 = 48 - 16 - 1
v_add_f32 v[vgprValuC+119], v[vgprValuC+119], v167 // accum partials

s_waitcnt vmcnt(30)                                // wait C (interleaved) 30 = 48 - 17 - 1
v_add_f32 v[vgprValuC+120], v[vgprValuC+120], v168 // accum partials

s_waitcnt vmcnt(29)                                // wait C (interleaved) 29 = 48 - 18 - 1
v_add_f32 v[vgprValuC+121], v[vgprValuC+121], v169 // accum partials

s_waitcnt vmcnt(28)                                // wait C (interleaved) 28 = 48 - 19 - 1
v_add_f32 v[vgprValuC+122], v[vgprValuC+122], v170 // accum partials

s_waitcnt vmcnt(27)                                // wait C (interleaved) 27 = 48 - 20 - 1
v_add_f32 v[vgprValuC+123], v[vgprValuC+123], v171 // accum partials

s_waitcnt vmcnt(26)                                // wait C (interleaved) 26 = 48 - 21 - 1
v_add_f32 v[vgprValuC+124], v[vgprValuC+124], v172 // accum partials

s_waitcnt vmcnt(25)                                // wait C (interleaved) 25 = 48 - 22 - 1
v_add_f32 v[vgprValuC+125], v[vgprValuC+125], v173 // accum partials

s_waitcnt vmcnt(24)                                // wait C (interleaved) 24 = 48 - 23 - 1
v_add_f32 v[vgprValuC+126], v[vgprValuC+126], v174 // accum partials

s_waitcnt vmcnt(23)                                // wait C (interleaved) 23 = 48 - 24 - 1
v_add_f32 v[vgprValuC+127], v[vgprValuC+127], v175 // accum partials

s_waitcnt vmcnt(22)                                // wait C (interleaved) 22 = 48 - 25 - 1
v_add_f32 v[vgprValuC+128], v[vgprValuC+128], v176 // accum partials

s_waitcnt vmcnt(21)                                // wait C (interleaved) 21 = 48 - 26 - 1
v_add_f32 v[vgprValuC+129], v[vgprValuC+129], v177 // accum partials

s_waitcnt vmcnt(20)                                // wait C (interleaved) 20 = 48 - 27 - 1
v_add_f32 v[vgprValuC+130], v[vgprValuC+130], v178 // accum partials

s_waitcnt vmcnt(19)                                // wait C (interleaved) 19 = 48 - 28 - 1
v_add_f32 v[vgprValuC+131], v[vgprValuC+131], v179 // accum partials

s_waitcnt vmcnt(18)                                // wait C (interleaved) 18 = 48 - 29 - 1
v_add_f32 v[vgprValuC+132], v[vgprValuC+132], v180 // accum partials

s_waitcnt vmcnt(17)                                // wait C (interleaved) 17 = 48 - 30 - 1
v_add_f32 v[vgprValuC+133], v[vgprValuC+133], v181 // accum partials

s_waitcnt vmcnt(16)                                // wait C (interleaved) 16 = 48 - 31 - 1
v_add_f32 v[vgprValuC+134], v[vgprValuC+134], v182 // accum partials

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 48 - 32 - 1
v_add_f32 v[vgprValuC+135], v[vgprValuC+135], v183 // accum partials

s_waitcnt vmcnt(14)                                // wait C (interleaved) 14 = 48 - 33 - 1
v_add_f32 v[vgprValuC+136], v[vgprValuC+136], v184 // accum partials

s_waitcnt vmcnt(13)                                // wait C (interleaved) 13 = 48 - 34 - 1
v_add_f32 v[vgprValuC+137], v[vgprValuC+137], v185 // accum partials

s_waitcnt vmcnt(12)                                // wait C (interleaved) 12 = 48 - 35 - 1
v_add_f32 v[vgprValuC+138], v[vgprValuC+138], v186 // accum partials

s_waitcnt vmcnt(11)                                // wait C (interleaved) 11 = 48 - 36 - 1
v_add_f32 v[vgprValuC+139], v[vgprValuC+139], v187 // accum partials

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 48 - 37 - 1
v_add_f32 v[vgprValuC+140], v[vgprValuC+140], v189 // accum partials

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 48 - 38 - 1
v_add_f32 v[vgprValuC+141], v[vgprValuC+141], v190 // accum partials

s_waitcnt vmcnt(8)                                 // wait C (interleaved) 8 = 48 - 39 - 1
v_add_f32 v[vgprValuC+142], v[vgprValuC+142], v191 // accum partials

s_waitcnt vmcnt(7)                                 // wait C (interleaved) 7 = 48 - 40 - 1
v_add_f32 v[vgprValuC+143], v[vgprValuC+143], v192 // accum partials

s_waitcnt vmcnt(6)                                 // wait C (interleaved) 6 = 48 - 41 - 1
v_add_f32 v[vgprValuC+144], v[vgprValuC+144], v193 // accum partials

s_waitcnt vmcnt(5)                                 // wait C (interleaved) 5 = 48 - 42 - 1
v_add_f32 v[vgprValuC+145], v[vgprValuC+145], v194 // accum partials

s_waitcnt vmcnt(4)                                 // wait C (interleaved) 4 = 48 - 43 - 1
v_add_f32 v[vgprValuC+146], v[vgprValuC+146], v195 // accum partials

s_waitcnt vmcnt(3)                                 // wait C (interleaved) 3 = 48 - 44 - 1
v_add_f32 v[vgprValuC+147], v[vgprValuC+147], v196 // accum partials

s_waitcnt vmcnt(2)                                 // wait C (interleaved) 2 = 48 - 45 - 1
v_add_f32 v[vgprValuC+148], v[vgprValuC+148], v197 // accum partials

s_waitcnt vmcnt(1)                                 // wait C (interleaved) 1 = 48 - 46 - 1
v_add_f32 v[vgprValuC+149], v[vgprValuC+149], v198 // accum partials

s_waitcnt vmcnt(0)                                 // wait C (interleaved) 0 = 48 - 47 - 1
v_add_f32 v[vgprValuC+150], v[vgprValuC+150], v199 // accum partials
v_mov_b32 v[vgprValuC+0], v[vgprValuC+103]         // copy vreg[0] to MI out reg
v_mov_b32 v[vgprValuC+8], v[vgprValuC+104]         // copy vreg[1] to MI out reg
v_mov_b32 v[vgprValuC+1], v[vgprValuC+105]         // copy vreg[2] to MI out reg
v_mov_b32 v[vgprValuC+9], v[vgprValuC+106]         // copy vreg[3] to MI out reg
v_mov_b32 v[vgprValuC+2], v[vgprValuC+107]         // copy vreg[4] to MI out reg
v_mov_b32 v[vgprValuC+10], v[vgprValuC+108]        // copy vreg[5] to MI out reg
v_mov_b32 v[vgprValuC+3], v[vgprValuC+109]         // copy vreg[6] to MI out reg
v_mov_b32 v[vgprValuC+11], v[vgprValuC+110]        // copy vreg[7] to MI out reg
v_mov_b32 v[vgprValuC+4], v[vgprValuC+111]         // copy vreg[8] to MI out reg
v_mov_b32 v[vgprValuC+12], v[vgprValuC+112]        // copy vreg[9] to MI out reg
v_mov_b32 v[vgprValuC+5], v[vgprValuC+113]         // copy vreg[10] to MI out reg
v_mov_b32 v[vgprValuC+13], v[vgprValuC+114]        // copy vreg[11] to MI out reg
v_mov_b32 v[vgprValuC+6], v[vgprValuC+115]         // copy vreg[12] to MI out reg
v_mov_b32 v[vgprValuC+14], v[vgprValuC+116]        // copy vreg[13] to MI out reg
v_mov_b32 v[vgprValuC+7], v[vgprValuC+117]         // copy vreg[14] to MI out reg
v_mov_b32 v[vgprValuC+15], v[vgprValuC+118]        // copy vreg[15] to MI out reg
v_mov_b32 v[vgprValuC+16], v[vgprValuC+119]        // copy vreg[16] to MI out reg
v_mov_b32 v[vgprValuC+24], v[vgprValuC+120]        // copy vreg[17] to MI out reg
v_mov_b32 v[vgprValuC+17], v[vgprValuC+121]        // copy vreg[18] to MI out reg
v_mov_b32 v[vgprValuC+25], v[vgprValuC+122]        // copy vreg[19] to MI out reg
v_mov_b32 v[vgprValuC+18], v[vgprValuC+123]        // copy vreg[20] to MI out reg
v_mov_b32 v[vgprValuC+26], v[vgprValuC+124]        // copy vreg[21] to MI out reg
v_mov_b32 v[vgprValuC+19], v[vgprValuC+125]        // copy vreg[22] to MI out reg
v_mov_b32 v[vgprValuC+27], v[vgprValuC+126]        // copy vreg[23] to MI out reg
v_mov_b32 v[vgprValuC+20], v[vgprValuC+127]        // copy vreg[24] to MI out reg
v_mov_b32 v[vgprValuC+28], v[vgprValuC+128]        // copy vreg[25] to MI out reg
v_mov_b32 v[vgprValuC+21], v[vgprValuC+129]        // copy vreg[26] to MI out reg
v_mov_b32 v[vgprValuC+29], v[vgprValuC+130]        // copy vreg[27] to MI out reg
v_mov_b32 v[vgprValuC+22], v[vgprValuC+131]        // copy vreg[28] to MI out reg
v_mov_b32 v[vgprValuC+30], v[vgprValuC+132]        // copy vreg[29] to MI out reg
v_mov_b32 v[vgprValuC+23], v[vgprValuC+133]        // copy vreg[30] to MI out reg
v_mov_b32 v[vgprValuC+31], v[vgprValuC+134]        // copy vreg[31] to MI out reg
v_mov_b32 v[vgprValuC+32], v[vgprValuC+135]        // copy vreg[32] to MI out reg
v_mov_b32 v[vgprValuC+40], v[vgprValuC+136]        // copy vreg[33] to MI out reg
v_mov_b32 v[vgprValuC+33], v[vgprValuC+137]        // copy vreg[34] to MI out reg
v_mov_b32 v[vgprValuC+41], v[vgprValuC+138]        // copy vreg[35] to MI out reg
v_mov_b32 v[vgprValuC+34], v[vgprValuC+139]        // copy vreg[36] to MI out reg
v_mov_b32 v[vgprValuC+42], v[vgprValuC+140]        // copy vreg[37] to MI out reg
v_mov_b32 v[vgprValuC+35], v[vgprValuC+141]        // copy vreg[38] to MI out reg
v_mov_b32 v[vgprValuC+43], v[vgprValuC+142]        // copy vreg[39] to MI out reg
v_mov_b32 v[vgprValuC+36], v[vgprValuC+143]        // copy vreg[40] to MI out reg
v_mov_b32 v[vgprValuC+44], v[vgprValuC+144]        // copy vreg[41] to MI out reg
v_mov_b32 v[vgprValuC+37], v[vgprValuC+145]        // copy vreg[42] to MI out reg
v_mov_b32 v[vgprValuC+45], v[vgprValuC+146]        // copy vreg[43] to MI out reg
v_mov_b32 v[vgprValuC+38], v[vgprValuC+147]        // copy vreg[44] to MI out reg
v_mov_b32 v[vgprValuC+46], v[vgprValuC+148]        // copy vreg[45] to MI out reg
v_mov_b32 v[vgprValuC+39], v[vgprValuC+149]        // copy vreg[46] to MI out reg
v_mov_b32 v[vgprValuC+47], v[vgprValuC+150]        // copy vreg[47] to MI out reg
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_mul_i32 s11, s[sgprskTiles], s[sgprItersPerTile]
s_mul_i32 s61, s[sgprSKItersPerWG], s[sgprskGrid]
s_sub_u32 s11, s11, s61                            // skTiles * ItersPerTile - SKItersPerWG * skGrid
s_bitcmp1_b32 s[sgprMagicShiftItersPerTile], 29    // USO on? (bit 29 of MagicShiftItersPerTile); off -> historical global peer size
s_cbranch_scc0 label_SK_PeerGlobal                 // USO off -> historical global mapping
s_cmp_eq_u32 s[sgprskTiles], 0                     // skTiles == 0?
s_cbranch_scc1 label_SK_PeerNoTiles                // global peer size
v_cvt_f32_u32 v101, s[sgprskTiles]                 // skGrid % skTiles
v_rcp_iflag_f32 v101, v101                         // skGrid % skTiles
v_mov_b32 v102, 0x4f7ffffe                         // 2^32 - 512
v_mul_f32 v101, v102, v101                         // 2^32 / divisor
v_cvt_u32_f32 v101, v101                           // skGrid % skTiles
v_mul_lo_u32 v102, v101, s[sgprskTiles]            // refine 2^32 / divisor
v_sub_nc_u32 v102, 0, v102                         // refine 2^32 / divisor
v_mul_hi_u32 v102, v101, v102                      // refine 2^32 / divisor
v_add_nc_u32 v101, v101, v102                      // refine 2^32 / divisor
v_mul_hi_u32 v101, s[sgprskGrid], v101             // skGrid % skTiles
v_mul_lo_u32 v102, v101, s[sgprskTiles]            // skGrid % skTiles
v_sub_nc_u32 v102, s[sgprskGrid], v102             // skGrid % skTiles
v_cmp_ge_u32 vcc_lo, v102, s[sgprskTiles]          // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v101, 1, v101                         // quotient + 1
v_sub_nc_u32 v102, v102, s[sgprskTiles]            // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v102, s[sgprskTiles]          // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v101, 1, v101                         // quotient + 1
v_sub_nc_u32 v102, v102, s[sgprskTiles]            // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s61, v101                      // quotient
v_readfirstlane_b32 s61, v102                      // remainder
s_cmp_eq_u32 s61, 0                                // skGrid % skTiles == 0?
s_cbranch_scc0 label_SK_PeerGlobal                 // ragged -> global peer
label_SK_PeerPerTile:
v_cvt_f32_u32 v101, s[sgprskTiles]                 // F = skGrid / skTiles
v_rcp_iflag_f32 v101, v101                         // F = skGrid / skTiles
v_mov_b32 v102, 0x4f7ffffe                         // 2^32 - 512
v_mul_f32 v101, v102, v101                         // 2^32 / divisor
v_cvt_u32_f32 v101, v101                           // F = skGrid / skTiles
v_mul_lo_u32 v102, v101, s[sgprskTiles]            // refine 2^32 / divisor
v_sub_nc_u32 v102, 0, v102                         // refine 2^32 / divisor
v_mul_hi_u32 v102, v101, v102                      // refine 2^32 / divisor
v_add_nc_u32 v101, v101, v102                      // refine 2^32 / divisor
v_mul_hi_u32 v101, s[sgprskGrid], v101             // F = skGrid / skTiles
v_mul_lo_u32 v102, v101, s[sgprskTiles]            // F = skGrid / skTiles
v_sub_nc_u32 v102, s[sgprskGrid], v102             // F = skGrid / skTiles
v_cmp_ge_u32 vcc_lo, v102, s[sgprskTiles]          // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v101, 1, v101                         // quotient + 1
v_sub_nc_u32 v102, v102, s[sgprskTiles]            // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v102, s[sgprskTiles]          // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v101, 1, v101                         // quotient + 1
v_sub_nc_u32 v102, v102, s[sgprskTiles]            // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s61, v101                      // quotient
s_mov_b32 s11, s61                                 // F = skGrid / skTiles
v_cvt_f32_u32 v101, s11                            // s = cta % F
v_rcp_iflag_f32 v101, v101                         // s = cta % F
v_mov_b32 v102, 0x4f7ffffe                         // 2^32 - 512
v_mul_f32 v101, v102, v101                         // 2^32 / divisor
v_cvt_u32_f32 v101, v101                           // s = cta % F
v_mul_lo_u32 v102, v101, s11                       // refine 2^32 / divisor
v_sub_nc_u32 v102, 0, v102                         // refine 2^32 / divisor
v_mul_hi_u32 v102, v101, v102                      // refine 2^32 / divisor
v_add_nc_u32 v101, v101, v102                      // refine 2^32 / divisor
v_mul_hi_u32 v101, s7, v101                        // s = cta % F
v_mul_lo_u32 v102, v101, s11                       // s = cta % F
v_sub_nc_u32 v102, s7, v102                        // s = cta % F
v_cmp_ge_u32 vcc_lo, v102, s11                     // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v101, 1, v101                         // quotient + 1
v_sub_nc_u32 v102, v102, s11                       // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v102, s11                     // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v101, 1, v101                         // quotient + 1
v_sub_nc_u32 v102, v102, s11                       // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s61, v101                      // quotient
v_readfirstlane_b32 s61, v102                      // remainder
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

/* Edge/NonEdge store path check (M): Size % 128 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s64, 127, s[sgprSizeI]                   // s64 = s[sgprSizeI] % 128
s_add_u32 s65, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s65                // wg0 >= nwg0-1 ?
s_cselect_b32 s64, s64, 0                          // set rem
s_cmpk_gt_u32 s64, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_GSU1_Else       // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 96 > 0 -> Edge store; else -> NonEdge store */
s_mul_hi_u32 s67, s[sgprSizeJ], 89478486           // STATIC_DIV: divisor=96
s_mul_i32 s66, s[sgprSizeJ], 89478486              // tmp = dividend * magic
s_lshr_b64 s[66:67], s[66:67], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s65, s66                                 // quotient
s_mul_i32 s66, s65, 96                             // quotient*divisor
s_sub_u32 s64, s[sgprSizeJ], s66                   // rReg = dividend - quotient*divisor
s_add_u32 s65, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s65                // wg1 >= nwg1-1
s_cselect_b32 s64, s64, 0                          // set rem
s_cmpk_gt_u32 s64, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_GSU1_Then       // jump if edges required
label_GW_B0_FD0_VW1_GSU1_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=152 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1) */
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
v_add_lshl_u32 v101, v93, v90, 1                   // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=90, coord0Vgpr=90 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (19, 0, 0, 0), (19, 1, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0), (22, 0, 0, 0), (22, 1, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+103], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+104], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+105], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+106], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+107], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+108], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+109], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+110], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+111], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+112], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+128], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+129], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+130], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+131], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+132], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+133], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+134], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+135], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v103.l, v[vgprValuC+103]             // convert C to fp16
buffer_store_b16 v103, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v104.l, v[vgprValuC+104]             // convert C to fp16
buffer_store_b16 v104, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cvt_f16_f32 v105.l, v[vgprValuC+105]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v105, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v106.l, v[vgprValuC+106]             // convert C to fp16
buffer_store_b16 v106, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cvt_f16_f32 v107.l, v[vgprValuC+107]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v107, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v108.l, v[vgprValuC+108]             // convert C to fp16
buffer_store_b16 v108, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cvt_f16_f32 v109.l, v[vgprValuC+109]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v109, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v110.l, v[vgprValuC+110]             // convert C to fp16
buffer_store_b16 v110, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cvt_f16_f32 v111.l, v[vgprValuC+111]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v111, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v112.l, v[vgprValuC+112]             // convert C to fp16
buffer_store_b16 v112, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cvt_f16_f32 v113.l, v[vgprValuC+113]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v113, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v114.l, v[vgprValuC+114]             // convert C to fp16
buffer_store_b16 v114, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cvt_f16_f32 v115.l, v[vgprValuC+115]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v115, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v116.l, v[vgprValuC+116]             // convert C to fp16
buffer_store_b16 v116, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cvt_f16_f32 v117.l, v[vgprValuC+117]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v117, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v118.l, v[vgprValuC+118]             // convert C to fp16
buffer_store_b16 v118, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cvt_f16_f32 v119.l, v[vgprValuC+119]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 36                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b16 v119, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v120.l, v[vgprValuC+120]             // convert C to fp16
buffer_store_b16 v120, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cvt_f16_f32 v121.l, v[vgprValuC+121]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v121, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v122.l, v[vgprValuC+122]             // convert C to fp16
buffer_store_b16 v122, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cvt_f16_f32 v123.l, v[vgprValuC+123]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v123, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v124.l, v[vgprValuC+124]             // convert C to fp16
buffer_store_b16 v124, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cvt_f16_f32 v125.l, v[vgprValuC+125]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v125, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v126.l, v[vgprValuC+126]             // convert C to fp16
buffer_store_b16 v126, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cvt_f16_f32 v127.l, v[vgprValuC+127]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v127, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v128.l, v[vgprValuC+128]             // convert C to fp16
buffer_store_b16 v128, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cvt_f16_f32 v129.l, v[vgprValuC+129]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v129, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v130.l, v[vgprValuC+130]             // convert C to fp16
buffer_store_b16 v130, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cvt_f16_f32 v131.l, v[vgprValuC+131]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v131, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v132.l, v[vgprValuC+132]             // convert C to fp16
buffer_store_b16 v132, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cvt_f16_f32 v133.l, v[vgprValuC+133]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v133, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v134.l, v[vgprValuC+134]             // convert C to fp16
buffer_store_b16 v134, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cvt_f16_f32 v135.l, v[vgprValuC+135]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 36                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b16 v135, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v136.l, v[vgprValuC+136]             // convert C to fp16
buffer_store_b16 v136, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cvt_f16_f32 v137.l, v[vgprValuC+137]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v137, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v138.l, v[vgprValuC+138]             // convert C to fp16
buffer_store_b16 v138, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cvt_f16_f32 v139.l, v[vgprValuC+139]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v139, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v140.l, v[vgprValuC+140]             // convert C to fp16
buffer_store_b16 v140, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cvt_f16_f32 v141.l, v[vgprValuC+141]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v141, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v142.l, v[vgprValuC+142]             // convert C to fp16
buffer_store_b16 v142, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cvt_f16_f32 v143.l, v[vgprValuC+143]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v143, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v144.l, v[vgprValuC+144]             // convert C to fp16
buffer_store_b16 v144, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cvt_f16_f32 v145.l, v[vgprValuC+145]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v145, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v146.l, v[vgprValuC+146]             // convert C to fp16
buffer_store_b16 v146, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cvt_f16_f32 v147.l, v[vgprValuC+147]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v147, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v148.l, v[vgprValuC+148]             // convert C to fp16
buffer_store_b16 v148, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cvt_f16_f32 v149.l, v[vgprValuC+149]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v149, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v150.l, v[vgprValuC+150]             // convert C to fp16
buffer_store_b16 v150, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW1_GSU1_NonEdgeEnd:
label_GW_B0_FD0_VW1_GSU1_Else:
label_GW_B0_FD0_VW1_GSU1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=76 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v96, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v149, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v149, v96, v149, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v150, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v150, v96, v150, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v151, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v151, v96, v151, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v152, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v152, v96, v152, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v153, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v153, v96, v153, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v154, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v96, v154, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v155, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v155, v96, v155, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v156, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v96, v156, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v157, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v157, v96, v157, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v158, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v96, v158, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v159, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v159, v96, v159, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v160, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v160, v96, v160, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v161, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v161, v96, v161, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v162, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v162, v96, v162, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v163, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v163, v96, v163, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v164, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v164, v96, v164, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 18                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v165, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v165, v96, v165, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v166, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v166, v96, v166, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v167, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v167, v96, v167, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v168, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v168, v96, v168, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v169, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v169, v96, v169, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v170, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v170, v96, v170, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v171, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v171, v96, v171, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v172, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v172, v96, v172, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v173, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v173, v96, v173, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v174, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v174, v96, v174, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v175, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v175, v96, v175, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v176, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v176, v96, v176, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v177, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v177, v96, v177, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v178, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v178, v96, v178, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v179, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v179, v96, v179, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v180, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v180, v96, v180, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 18                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v181, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v181, v96, v181, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v182, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v182, v96, v182, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v183, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v183, v96, v183, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v184, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v184, v96, v184, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v185, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v185, v96, v185, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v186, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v186, v96, v186, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v187, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v187, v96, v187, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v189, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v189, v96, v189, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v190, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v190, v96, v190, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v191, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v191, v96, v191, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v192, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v192, v96, v192, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v193, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v193, v96, v193, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v194, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v194, v96, v194, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v195, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v195, v96, v195, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v196, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v196, v96, v196, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v197, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v197, v96, v197, s66                 // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (19, 0, 0, 0), (19, 1, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0), (22, 0, 0, 0), (22, 1, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+102], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+103], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+104], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+105], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+106], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+107], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+108], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+109], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+110], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+111], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+112], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+128], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+129], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+130], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+131], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+132], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+133], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+134], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+135], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v101.l, v[vgprValuC+101]             // convert C to fp16
buffer_store_b16 v101, v149, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v102.l, v[vgprValuC+102]             // convert C to fp16
buffer_store_b16 v102, v150, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v103.l, v[vgprValuC+103]             // convert C to fp16
buffer_store_b16 v103, v151, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v104.l, v[vgprValuC+104]             // convert C to fp16
buffer_store_b16 v104, v152, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v105.l, v[vgprValuC+105]             // convert C to fp16
buffer_store_b16 v105, v153, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v106.l, v[vgprValuC+106]             // convert C to fp16
buffer_store_b16 v106, v154, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v107.l, v[vgprValuC+107]             // convert C to fp16
buffer_store_b16 v107, v155, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v108.l, v[vgprValuC+108]             // convert C to fp16
buffer_store_b16 v108, v156, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v109.l, v[vgprValuC+109]             // convert C to fp16
buffer_store_b16 v109, v157, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v110.l, v[vgprValuC+110]             // convert C to fp16
buffer_store_b16 v110, v158, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v111.l, v[vgprValuC+111]             // convert C to fp16
buffer_store_b16 v111, v159, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v112.l, v[vgprValuC+112]             // convert C to fp16
buffer_store_b16 v112, v160, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v113.l, v[vgprValuC+113]             // convert C to fp16
buffer_store_b16 v113, v161, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v114.l, v[vgprValuC+114]             // convert C to fp16
buffer_store_b16 v114, v162, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v115.l, v[vgprValuC+115]             // convert C to fp16
buffer_store_b16 v115, v163, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v116.l, v[vgprValuC+116]             // convert C to fp16
buffer_store_b16 v116, v164, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v117.l, v[vgprValuC+117]             // convert C to fp16
buffer_store_b16 v117, v165, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v118.l, v[vgprValuC+118]             // convert C to fp16
buffer_store_b16 v118, v166, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v119.l, v[vgprValuC+119]             // convert C to fp16
buffer_store_b16 v119, v167, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v120.l, v[vgprValuC+120]             // convert C to fp16
buffer_store_b16 v120, v168, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v121.l, v[vgprValuC+121]             // convert C to fp16
buffer_store_b16 v121, v169, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v122.l, v[vgprValuC+122]             // convert C to fp16
buffer_store_b16 v122, v170, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v123.l, v[vgprValuC+123]             // convert C to fp16
buffer_store_b16 v123, v171, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v124.l, v[vgprValuC+124]             // convert C to fp16
buffer_store_b16 v124, v172, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v125.l, v[vgprValuC+125]             // convert C to fp16
buffer_store_b16 v125, v173, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v126.l, v[vgprValuC+126]             // convert C to fp16
buffer_store_b16 v126, v174, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v127.l, v[vgprValuC+127]             // convert C to fp16
buffer_store_b16 v127, v175, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v128.l, v[vgprValuC+128]             // convert C to fp16
buffer_store_b16 v128, v176, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v129.l, v[vgprValuC+129]             // convert C to fp16
buffer_store_b16 v129, v177, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v130.l, v[vgprValuC+130]             // convert C to fp16
buffer_store_b16 v130, v178, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v131.l, v[vgprValuC+131]             // convert C to fp16
buffer_store_b16 v131, v179, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v132.l, v[vgprValuC+132]             // convert C to fp16
buffer_store_b16 v132, v180, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v133.l, v[vgprValuC+133]             // convert C to fp16
buffer_store_b16 v133, v181, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v134.l, v[vgprValuC+134]             // convert C to fp16
buffer_store_b16 v134, v182, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v135.l, v[vgprValuC+135]             // convert C to fp16
buffer_store_b16 v135, v183, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v136.l, v[vgprValuC+136]             // convert C to fp16
buffer_store_b16 v136, v184, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v137.l, v[vgprValuC+137]             // convert C to fp16
buffer_store_b16 v137, v185, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v138.l, v[vgprValuC+138]             // convert C to fp16
buffer_store_b16 v138, v186, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v139.l, v[vgprValuC+139]             // convert C to fp16
buffer_store_b16 v139, v187, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v140.l, v[vgprValuC+140]             // convert C to fp16
buffer_store_b16 v140, v189, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v141.l, v[vgprValuC+141]             // convert C to fp16
buffer_store_b16 v141, v190, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v142.l, v[vgprValuC+142]             // convert C to fp16
buffer_store_b16 v142, v191, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v143.l, v[vgprValuC+143]             // convert C to fp16
buffer_store_b16 v143, v192, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v144.l, v[vgprValuC+144]             // convert C to fp16
buffer_store_b16 v144, v193, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v145.l, v[vgprValuC+145]             // convert C to fp16
buffer_store_b16 v145, v194, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v146.l, v[vgprValuC+146]             // convert C to fp16
buffer_store_b16 v146, v195, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v147.l, v[vgprValuC+147]             // convert C to fp16
buffer_store_b16 v147, v196, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v148.l, v[vgprValuC+148]             // convert C to fp16
buffer_store_b16 v148, v197, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_GSU1:
label_GW_B1_FD0_GSU1:

/* Edge/NonEdge store path check (M): Size % 128 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s64, 127, s[sgprSizeI]                   // s64 = s[sgprSizeI] % 128
s_add_u32 s65, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s65                // wg0 >= nwg0-1 ?
s_cselect_b32 s64, s64, 0                          // set rem
s_cmpk_gt_u32 s64, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW1_GSU1_Else       // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 96 > 0 -> Edge store; else -> NonEdge store */
s_mul_hi_u32 s67, s[sgprSizeJ], 89478486           // STATIC_DIV: divisor=96
s_mul_i32 s66, s[sgprSizeJ], 89478486              // tmp = dividend * magic
s_lshr_b64 s[66:67], s[66:67], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s65, s66                                 // quotient
s_mul_i32 s66, s65, 96                             // quotient*divisor
s_sub_u32 s64, s[sgprSizeJ], s66                   // rReg = dividend - quotient*divisor
s_add_u32 s65, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s65                // wg1 >= nwg1-1
s_cselect_b32 s64, s64, 0                          // set rem
s_cmpk_gt_u32 s64, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW1_GSU1_Then       // jump if edges required
label_GW_B1_FD0_VW1_GSU1_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=74 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v102, v92, v90, 1                   // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=90, coord0Vgpr=90 (multiple bpe)
buffer_load_d16_b16 v151, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
buffer_load_d16_b16 v152, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v153, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
buffer_load_d16_b16 v154, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v155, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
buffer_load_d16_b16 v156, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v157, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
buffer_load_d16_b16 v158, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v159, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
buffer_load_d16_b16 v160, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v161, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
buffer_load_d16_b16 v162, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v163, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
buffer_load_d16_b16 v164, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v165, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
buffer_load_d16_b16 v166, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 36                 // scale StrideC *= numRows(18) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_load_d16_b16 v167, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
buffer_load_d16_b16 v168, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v169, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(9,0,1,0) */
buffer_load_d16_b16 v170, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v171, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(10,0,1,0) */
buffer_load_d16_b16 v172, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v173, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(11,0,1,0) */
buffer_load_d16_b16 v174, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v175, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
buffer_load_d16_b16 v176, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v177, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(13,0,1,0) */
buffer_load_d16_b16 v178, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v179, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(14,0,1,0) */
buffer_load_d16_b16 v180, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v181, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(15,0,1,0) */
buffer_load_d16_b16 v182, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 36                 // scale StrideC *= numRows(18) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_load_d16_b16 v183, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
buffer_load_d16_b16 v184, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v185, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(17,0,1,0) */
buffer_load_d16_b16 v186, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v187, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(18,0,1,0) */
buffer_load_d16_b16 v189, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v190, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(19,0,1,0) */
buffer_load_d16_b16 v191, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v192, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
buffer_load_d16_b16 v193, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v194, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(21,0,1,0) */
buffer_load_d16_b16 v195, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v196, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(22,0,1,0) */
buffer_load_d16_b16 v197, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v198, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(23,0,1,0) */
buffer_load_d16_b16 v199, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
v_add_lshl_u32 v101, v93, v90, 1                   // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=90, coord0Vgpr=90 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (19, 0, 0, 0), (19, 1, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0), (22, 0, 0, 0), (22, 1, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+103], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+104], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+105], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+106], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+107], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+108], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+109], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+110], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+111], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+112], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+128], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+129], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+130], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+131], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+132], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+133], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+134], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+135], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(47)                                // vlcnt(47) = 48 - 1 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+103], s[sgprBeta], v151, v[vgprValuC+103] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v103.l, v[vgprValuC+103]             // convert C to fp16
buffer_store_b16 v103, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(46)                                // vlcnt(46) = 48 - 2 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+104], s[sgprBeta], v152, v[vgprValuC+104] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v104.l, v[vgprValuC+104]             // convert C to fp16
buffer_store_b16 v104, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(45)                                // vlcnt(45) = 48 - 3 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+105], s[sgprBeta], v153, v[vgprValuC+105] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v105.l, v[vgprValuC+105]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v105, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(44)                                // vlcnt(44) = 48 - 4 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+106], s[sgprBeta], v154, v[vgprValuC+106] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v106.l, v[vgprValuC+106]             // convert C to fp16
buffer_store_b16 v106, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(43)                                // vlcnt(43) = 48 - 5 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+107], s[sgprBeta], v155, v[vgprValuC+107] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v107.l, v[vgprValuC+107]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v107, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(42)                                // vlcnt(42) = 48 - 6 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+108], s[sgprBeta], v156, v[vgprValuC+108] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v108.l, v[vgprValuC+108]             // convert C to fp16
buffer_store_b16 v108, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(41)                                // vlcnt(41) = 48 - 7 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+109], s[sgprBeta], v157, v[vgprValuC+109] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v109.l, v[vgprValuC+109]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v109, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(40)                                // vlcnt(40) = 48 - 8 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+110], s[sgprBeta], v158, v[vgprValuC+110] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v110.l, v[vgprValuC+110]             // convert C to fp16
buffer_store_b16 v110, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(39)                                // vlcnt(39) = 48 - 9 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+111], s[sgprBeta], v159, v[vgprValuC+111] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v111.l, v[vgprValuC+111]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v111, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(38)                                // vlcnt(38) = 48 - 10 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+112], s[sgprBeta], v160, v[vgprValuC+112] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v112.l, v[vgprValuC+112]             // convert C to fp16
buffer_store_b16 v112, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(37)                                // vlcnt(37) = 48 - 11 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+113], s[sgprBeta], v161, v[vgprValuC+113] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v113.l, v[vgprValuC+113]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v113, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(36)                                // vlcnt(36) = 48 - 12 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+114], s[sgprBeta], v162, v[vgprValuC+114] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v114.l, v[vgprValuC+114]             // convert C to fp16
buffer_store_b16 v114, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(35)                                // vlcnt(35) = 48 - 13 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+115], s[sgprBeta], v163, v[vgprValuC+115] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v115.l, v[vgprValuC+115]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v115, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(34)                                // vlcnt(34) = 48 - 14 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+116], s[sgprBeta], v164, v[vgprValuC+116] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v116.l, v[vgprValuC+116]             // convert C to fp16
buffer_store_b16 v116, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(33)                                // vlcnt(33) = 48 - 15 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+117], s[sgprBeta], v165, v[vgprValuC+117] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v117.l, v[vgprValuC+117]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v117, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(32)                                // vlcnt(32) = 48 - 16 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+118], s[sgprBeta], v166, v[vgprValuC+118] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v118.l, v[vgprValuC+118]             // convert C to fp16
buffer_store_b16 v118, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(31)                                // vlcnt(31) = 48 - 17 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+119], s[sgprBeta], v167, v[vgprValuC+119] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v119.l, v[vgprValuC+119]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 36                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b16 v119, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(30)                                // vlcnt(30) = 48 - 18 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+120], s[sgprBeta], v168, v[vgprValuC+120] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v120.l, v[vgprValuC+120]             // convert C to fp16
buffer_store_b16 v120, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(29)                                // vlcnt(29) = 48 - 19 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+121], s[sgprBeta], v169, v[vgprValuC+121] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v121.l, v[vgprValuC+121]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v121, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(28)                                // vlcnt(28) = 48 - 20 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+122], s[sgprBeta], v170, v[vgprValuC+122] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v122.l, v[vgprValuC+122]             // convert C to fp16
buffer_store_b16 v122, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(27)                                // vlcnt(27) = 48 - 21 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+123], s[sgprBeta], v171, v[vgprValuC+123] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v123.l, v[vgprValuC+123]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v123, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(26)                                // vlcnt(26) = 48 - 22 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+124], s[sgprBeta], v172, v[vgprValuC+124] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v124.l, v[vgprValuC+124]             // convert C to fp16
buffer_store_b16 v124, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(25)                                // vlcnt(25) = 48 - 23 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+125], s[sgprBeta], v173, v[vgprValuC+125] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v125.l, v[vgprValuC+125]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v125, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(24)                                // vlcnt(24) = 48 - 24 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+126], s[sgprBeta], v174, v[vgprValuC+126] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v126.l, v[vgprValuC+126]             // convert C to fp16
buffer_store_b16 v126, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(23)                                // vlcnt(23) = 48 - 25 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+127], s[sgprBeta], v175, v[vgprValuC+127] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v127.l, v[vgprValuC+127]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v127, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(22)                                // vlcnt(22) = 48 - 26 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+128], s[sgprBeta], v176, v[vgprValuC+128] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v128.l, v[vgprValuC+128]             // convert C to fp16
buffer_store_b16 v128, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(21)                                // vlcnt(21) = 48 - 27 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+129], s[sgprBeta], v177, v[vgprValuC+129] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v129.l, v[vgprValuC+129]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v129, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(20)                                // vlcnt(20) = 48 - 28 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+130], s[sgprBeta], v178, v[vgprValuC+130] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v130.l, v[vgprValuC+130]             // convert C to fp16
buffer_store_b16 v130, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(19)                                // vlcnt(19) = 48 - 29 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+131], s[sgprBeta], v179, v[vgprValuC+131] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v131.l, v[vgprValuC+131]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v131, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(18)                                // vlcnt(18) = 48 - 30 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+132], s[sgprBeta], v180, v[vgprValuC+132] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v132.l, v[vgprValuC+132]             // convert C to fp16
buffer_store_b16 v132, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(17)                                // vlcnt(17) = 48 - 31 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+133], s[sgprBeta], v181, v[vgprValuC+133] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v133.l, v[vgprValuC+133]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v133, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(16)                                // vlcnt(16) = 48 - 32 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+134], s[sgprBeta], v182, v[vgprValuC+134] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v134.l, v[vgprValuC+134]             // convert C to fp16
buffer_store_b16 v134, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(15)                                // vlcnt(15) = 48 - 33 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+135], s[sgprBeta], v183, v[vgprValuC+135] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v135.l, v[vgprValuC+135]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 36                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b16 v135, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(14)                                // vlcnt(14) = 48 - 34 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+136], s[sgprBeta], v184, v[vgprValuC+136] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v136.l, v[vgprValuC+136]             // convert C to fp16
buffer_store_b16 v136, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(13)                                // vlcnt(13) = 48 - 35 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+137], s[sgprBeta], v185, v[vgprValuC+137] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v137.l, v[vgprValuC+137]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v137, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(12)                                // vlcnt(12) = 48 - 36 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+138], s[sgprBeta], v186, v[vgprValuC+138] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v138.l, v[vgprValuC+138]             // convert C to fp16
buffer_store_b16 v138, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(11)                                // vlcnt(11) = 48 - 37 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+139], s[sgprBeta], v187, v[vgprValuC+139] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v139.l, v[vgprValuC+139]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v139, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(10)                                // vlcnt(10) = 48 - 38 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+140], s[sgprBeta], v189, v[vgprValuC+140] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v140.l, v[vgprValuC+140]             // convert C to fp16
buffer_store_b16 v140, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(9)                                 // vlcnt(9) = 48 - 39 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+141], s[sgprBeta], v190, v[vgprValuC+141] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v141.l, v[vgprValuC+141]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v141, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(8)                                 // vlcnt(8) = 48 - 40 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+142], s[sgprBeta], v191, v[vgprValuC+142] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v142.l, v[vgprValuC+142]             // convert C to fp16
buffer_store_b16 v142, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(7)                                 // vlcnt(7) = 48 - 41 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+143], s[sgprBeta], v192, v[vgprValuC+143] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v143.l, v[vgprValuC+143]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v143, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(6)                                 // vlcnt(6) = 48 - 42 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+144], s[sgprBeta], v193, v[vgprValuC+144] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v144.l, v[vgprValuC+144]             // convert C to fp16
buffer_store_b16 v144, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(5)                                 // vlcnt(5) = 48 - 43 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+145], s[sgprBeta], v194, v[vgprValuC+145] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v145.l, v[vgprValuC+145]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v145, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(4)                                 // vlcnt(4) = 48 - 44 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+146], s[sgprBeta], v195, v[vgprValuC+146] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v146.l, v[vgprValuC+146]             // convert C to fp16
buffer_store_b16 v146, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(3)                                 // vlcnt(3) = 48 - 45 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+147], s[sgprBeta], v196, v[vgprValuC+147] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v147.l, v[vgprValuC+147]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v147, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(2)                                 // vlcnt(2) = 48 - 46 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+148], s[sgprBeta], v197, v[vgprValuC+148] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v148.l, v[vgprValuC+148]             // convert C to fp16
buffer_store_b16 v148, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(1)                                 // vlcnt(1) = 48 - 47 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+149], s[sgprBeta], v198, v[vgprValuC+149] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v149.l, v[vgprValuC+149]             // convert C to fp16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v149, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(0)                                 // vlcnt(0) = 48 - 48 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+150], s[sgprBeta], v199, v[vgprValuC+150] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v150.l, v[vgprValuC+150]             // convert C to fp16
buffer_store_b16 v150, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_FD0_VW1_GSU1_NonEdgeEnd:
label_GW_B1_FD0_VW1_GSU1_Else:
label_GW_B1_FD0_VW1_GSU1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=50 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v96, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v150, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v150, v96, v150, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v149, v150, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v150, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v150, v96, v150, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v152, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v152, v96, v152, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v151, v152, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v152, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v152, v96, v152, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v154, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v96, v154, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v153, v154, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v154, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v96, v154, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v156, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v96, v156, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v155, v156, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v156, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v96, v156, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v158, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v96, v158, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v157, v158, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v158, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v96, v158, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v160, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v160, v96, v160, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v159, v160, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v160, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v160, v96, v160, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v162, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v162, v96, v162, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v161, v162, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v162, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v162, v96, v162, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v164, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v164, v96, v164, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v163, v164, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v164, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v164, v96, v164, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v166, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v166, v96, v166, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v165, v166, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v166, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v166, v96, v166, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v168, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v168, v96, v168, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v167, v168, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v168, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v168, v96, v168, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v170, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v170, v96, v170, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v169, v170, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v170, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v170, v96, v170, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v172, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v172, v96, v172, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v171, v172, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v172, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v172, v96, v172, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v174, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v174, v96, v174, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v173, v174, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v174, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v174, v96, v174, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v176, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v176, v96, v176, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v175, v176, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v176, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v176, v96, v176, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v178, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v178, v96, v178, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v177, v178, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v178, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v178, v96, v178, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v180, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v180, v96, v180, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v179, v180, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v180, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v180, v96, v180, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 18                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v182, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v182, v96, v182, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v181, v182, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v182, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v182, v96, v182, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v184, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v184, v96, v184, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v183, v184, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v184, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v184, v96, v184, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v186, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v186, v96, v186, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v185, v186, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v186, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v186, v96, v186, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v189, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v189, v96, v189, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v187, v189, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v189, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v189, v96, v189, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v191, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v191, v96, v191, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v190, v191, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v191, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v191, v96, v191, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v193, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v193, v96, v193, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v192, v193, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v193, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v193, v96, v193, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v195, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v195, v96, v195, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v194, v195, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v195, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v195, v96, v195, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v197, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v197, v96, v197, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v196, v197, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v197, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v197, v96, v197, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v199, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v199, v96, v199, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v198, v199, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v199, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v199, v96, v199, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v201, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v201, v96, v201, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v200, v201, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v201, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v201, v96, v201, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v203, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v203, v96, v203, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v202, v203, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v203, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v203, v96, v203, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v205, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v205, v96, v205, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v204, v205, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v205, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v205, v96, v205, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v207, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v207, v96, v207, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v206, v207, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v207, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v207, v96, v207, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v209, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v209, v96, v209, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v208, v209, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v209, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v209, v96, v209, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v211, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v211, v96, v211, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v210, v211, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v211, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v211, v96, v211, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v213, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v213, v96, v213, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v212, v213, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v213, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v213, v96, v213, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 18                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v215, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v215, v96, v215, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v214, v215, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v215, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v215, v96, v215, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v217, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v217, v96, v217, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v216, v217, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v217, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v217, v96, v217, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v219, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v219, v96, v219, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v218, v219, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v219, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v219, v96, v219, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v221, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v221, v96, v221, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v220, v221, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v221, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v221, v96, v221, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v223, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v223, v96, v223, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v222, v223, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v223, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v223, v96, v223, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v225, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v225, v96, v225, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v224, v225, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v225, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v225, v96, v225, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v227, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v227, v96, v227, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v226, v227, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v227, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v227, v96, v227, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v229, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v229, v96, v229, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v228, v229, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v229, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v229, v96, v229, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v231, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v231, v96, v231, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v230, v231, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v231, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v231, v96, v231, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v233, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v233, v96, v233, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v232, v233, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v233, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v233, v96, v233, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v235, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v235, v96, v235, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v234, v235, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v235, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v235, v96, v235, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v237, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v237, v96, v237, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v236, v237, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v237, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v237, v96, v237, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v239, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v239, v96, v239, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v238, v239, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v239, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v239, v96, v239, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v241, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v241, v96, v241, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v240, v241, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v241, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v241, v96, v241, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s64                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v243, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v243, v96, v243, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v242, v243, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v243, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v243, v96, v243, s66                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s66, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v245, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v245, v96, v245, s66                 // LDC clip if OOB. offset
buffer_load_d16_b16 v244, v245, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v245, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v245, v96, v245, s66                 // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (19, 0, 0, 0), (19, 1, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0), (22, 0, 0, 0), (22, 1, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+102], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+103], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+104], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+105], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+106], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+107], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+108], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+109], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+110], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+111], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+112], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+128], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+129], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+130], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+131], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+132], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+133], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+134], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+135], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v149, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v101.l, v[vgprValuC+101]             // convert C to fp16
buffer_store_b16 v101, v150, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+102], s[sgprBeta], v151, v[vgprValuC+102] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v102.l, v[vgprValuC+102]             // convert C to fp16
buffer_store_b16 v102, v152, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+103], s[sgprBeta], v153, v[vgprValuC+103] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v103.l, v[vgprValuC+103]             // convert C to fp16
buffer_store_b16 v103, v154, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+104], s[sgprBeta], v155, v[vgprValuC+104] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v104.l, v[vgprValuC+104]             // convert C to fp16
buffer_store_b16 v104, v156, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+105], s[sgprBeta], v157, v[vgprValuC+105] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v105.l, v[vgprValuC+105]             // convert C to fp16
buffer_store_b16 v105, v158, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+106], s[sgprBeta], v159, v[vgprValuC+106] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v106.l, v[vgprValuC+106]             // convert C to fp16
buffer_store_b16 v106, v160, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+107], s[sgprBeta], v161, v[vgprValuC+107] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v107.l, v[vgprValuC+107]             // convert C to fp16
buffer_store_b16 v107, v162, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+108], s[sgprBeta], v163, v[vgprValuC+108] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v108.l, v[vgprValuC+108]             // convert C to fp16
buffer_store_b16 v108, v164, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+109], s[sgprBeta], v165, v[vgprValuC+109] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v109.l, v[vgprValuC+109]             // convert C to fp16
buffer_store_b16 v109, v166, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+110], s[sgprBeta], v167, v[vgprValuC+110] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v110.l, v[vgprValuC+110]             // convert C to fp16
buffer_store_b16 v110, v168, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+111], s[sgprBeta], v169, v[vgprValuC+111] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v111.l, v[vgprValuC+111]             // convert C to fp16
buffer_store_b16 v111, v170, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+112], s[sgprBeta], v171, v[vgprValuC+112] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v112.l, v[vgprValuC+112]             // convert C to fp16
buffer_store_b16 v112, v172, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+113], s[sgprBeta], v173, v[vgprValuC+113] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v113.l, v[vgprValuC+113]             // convert C to fp16
buffer_store_b16 v113, v174, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+114], s[sgprBeta], v175, v[vgprValuC+114] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v114.l, v[vgprValuC+114]             // convert C to fp16
buffer_store_b16 v114, v176, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+115], s[sgprBeta], v177, v[vgprValuC+115] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v115.l, v[vgprValuC+115]             // convert C to fp16
buffer_store_b16 v115, v178, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+116], s[sgprBeta], v179, v[vgprValuC+116] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v116.l, v[vgprValuC+116]             // convert C to fp16
buffer_store_b16 v116, v180, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+117], s[sgprBeta], v181, v[vgprValuC+117] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v117.l, v[vgprValuC+117]             // convert C to fp16
buffer_store_b16 v117, v182, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+118], s[sgprBeta], v183, v[vgprValuC+118] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v118.l, v[vgprValuC+118]             // convert C to fp16
buffer_store_b16 v118, v184, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+119], s[sgprBeta], v185, v[vgprValuC+119] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v119.l, v[vgprValuC+119]             // convert C to fp16
buffer_store_b16 v119, v186, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+120], s[sgprBeta], v187, v[vgprValuC+120] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v120.l, v[vgprValuC+120]             // convert C to fp16
buffer_store_b16 v120, v189, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+121], s[sgprBeta], v190, v[vgprValuC+121] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v121.l, v[vgprValuC+121]             // convert C to fp16
buffer_store_b16 v121, v191, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+122], s[sgprBeta], v192, v[vgprValuC+122] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v122.l, v[vgprValuC+122]             // convert C to fp16
buffer_store_b16 v122, v193, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+123], s[sgprBeta], v194, v[vgprValuC+123] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v123.l, v[vgprValuC+123]             // convert C to fp16
buffer_store_b16 v123, v195, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+124], s[sgprBeta], v196, v[vgprValuC+124] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v124.l, v[vgprValuC+124]             // convert C to fp16
buffer_store_b16 v124, v197, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+125], s[sgprBeta], v198, v[vgprValuC+125] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v125.l, v[vgprValuC+125]             // convert C to fp16
buffer_store_b16 v125, v199, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+126], s[sgprBeta], v200, v[vgprValuC+126] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v126.l, v[vgprValuC+126]             // convert C to fp16
buffer_store_b16 v126, v201, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+127], s[sgprBeta], v202, v[vgprValuC+127] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v127.l, v[vgprValuC+127]             // convert C to fp16
buffer_store_b16 v127, v203, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+128], s[sgprBeta], v204, v[vgprValuC+128] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v128.l, v[vgprValuC+128]             // convert C to fp16
buffer_store_b16 v128, v205, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+129], s[sgprBeta], v206, v[vgprValuC+129] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v129.l, v[vgprValuC+129]             // convert C to fp16
buffer_store_b16 v129, v207, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+130], s[sgprBeta], v208, v[vgprValuC+130] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v130.l, v[vgprValuC+130]             // convert C to fp16
buffer_store_b16 v130, v209, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+131], s[sgprBeta], v210, v[vgprValuC+131] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v131.l, v[vgprValuC+131]             // convert C to fp16
buffer_store_b16 v131, v211, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+132], s[sgprBeta], v212, v[vgprValuC+132] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v132.l, v[vgprValuC+132]             // convert C to fp16
buffer_store_b16 v132, v213, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+133], s[sgprBeta], v214, v[vgprValuC+133] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v133.l, v[vgprValuC+133]             // convert C to fp16
buffer_store_b16 v133, v215, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+134], s[sgprBeta], v216, v[vgprValuC+134] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v134.l, v[vgprValuC+134]             // convert C to fp16
buffer_store_b16 v134, v217, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+135], s[sgprBeta], v218, v[vgprValuC+135] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v135.l, v[vgprValuC+135]             // convert C to fp16
buffer_store_b16 v135, v219, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+136], s[sgprBeta], v220, v[vgprValuC+136] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v136.l, v[vgprValuC+136]             // convert C to fp16
buffer_store_b16 v136, v221, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+137], s[sgprBeta], v222, v[vgprValuC+137] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v137.l, v[vgprValuC+137]             // convert C to fp16
buffer_store_b16 v137, v223, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+138], s[sgprBeta], v224, v[vgprValuC+138] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v138.l, v[vgprValuC+138]             // convert C to fp16
buffer_store_b16 v138, v225, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+139], s[sgprBeta], v226, v[vgprValuC+139] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v139.l, v[vgprValuC+139]             // convert C to fp16
buffer_store_b16 v139, v227, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+140], s[sgprBeta], v228, v[vgprValuC+140] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v140.l, v[vgprValuC+140]             // convert C to fp16
buffer_store_b16 v140, v229, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+141], s[sgprBeta], v230, v[vgprValuC+141] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v141.l, v[vgprValuC+141]             // convert C to fp16
buffer_store_b16 v141, v231, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+142], s[sgprBeta], v232, v[vgprValuC+142] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v142.l, v[vgprValuC+142]             // convert C to fp16
buffer_store_b16 v142, v233, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+143], s[sgprBeta], v234, v[vgprValuC+143] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v143.l, v[vgprValuC+143]             // convert C to fp16
buffer_store_b16 v143, v235, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+144], s[sgprBeta], v236, v[vgprValuC+144] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v144.l, v[vgprValuC+144]             // convert C to fp16
buffer_store_b16 v144, v237, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+145], s[sgprBeta], v238, v[vgprValuC+145] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v145.l, v[vgprValuC+145]             // convert C to fp16
buffer_store_b16 v145, v239, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+146], s[sgprBeta], v240, v[vgprValuC+146] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v146.l, v[vgprValuC+146]             // convert C to fp16
buffer_store_b16 v146, v241, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+147], s[sgprBeta], v242, v[vgprValuC+147] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v147.l, v[vgprValuC+147]             // convert C to fp16
buffer_store_b16 v147, v243, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+148], s[sgprBeta], v244, v[vgprValuC+148] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v148.l, v[vgprValuC+148]             // convert C to fp16
buffer_store_b16 v148, v245, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_SK_Partials_1:
label_GW_Partials_E0:
s_mov_b64 s[sgprSrdWS:sgprSrdWS+1], s[sgprAddressWS:sgprAddressWS+1]
s_mov_b32 s[sgprSrdWS+2], BufferOOB
s_mov_b32 s[sgprSrdWS+3], Srd127_96

s_mul_i32 s7, 0xc000, s[sgprPersistentWorkGroupIndex] // Offset to correct partials tile (low word)
s_mul_hi_u32 s8, 0xc000, s[sgprPersistentWorkGroupIndex] // partials tile offset (high word) for 64-bit SRD
s_add_u32 s[sgprSrdWS+0], s[sgprSrdWS+0], s7       // add lo to SRD
s_addc_u32 s[sgprSrdWS+1], s[sgprSrdWS+1], s8      // add hi (offset high word + lo carry) to SRD

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=152 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Partials Write Batch #0 (d1,d0,vc1,vc0) = */
/*      (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v[vgprValuC+103], v[vgprValuC+0]         // copy MI out reg to vreg[0]
v_mov_b32 v[vgprValuC+104], v[vgprValuC+8]         // copy MI out reg to vreg[1]
v_mov_b32 v[vgprValuC+105], v[vgprValuC+1]         // copy MI out reg to vreg[2]
v_mov_b32 v[vgprValuC+106], v[vgprValuC+9]         // copy MI out reg to vreg[3]
v_mov_b32 v[vgprValuC+107], v[vgprValuC+2]         // copy MI out reg to vreg[4]
v_mov_b32 v[vgprValuC+108], v[vgprValuC+10]        // copy MI out reg to vreg[5]
v_mov_b32 v[vgprValuC+109], v[vgprValuC+3]         // copy MI out reg to vreg[6]
v_mov_b32 v[vgprValuC+110], v[vgprValuC+11]        // copy MI out reg to vreg[7]
v_mov_b32 v[vgprValuC+111], v[vgprValuC+4]         // copy MI out reg to vreg[8]
v_mov_b32 v[vgprValuC+112], v[vgprValuC+12]        // copy MI out reg to vreg[9]
v_mov_b32 v[vgprValuC+113], v[vgprValuC+5]         // copy MI out reg to vreg[10]
v_mov_b32 v[vgprValuC+114], v[vgprValuC+13]        // copy MI out reg to vreg[11]
v_mov_b32 v[vgprValuC+115], v[vgprValuC+6]         // copy MI out reg to vreg[12]
v_mov_b32 v[vgprValuC+116], v[vgprValuC+14]        // copy MI out reg to vreg[13]
v_mov_b32 v[vgprValuC+117], v[vgprValuC+7]         // copy MI out reg to vreg[14]
v_mov_b32 v[vgprValuC+118], v[vgprValuC+15]        // copy MI out reg to vreg[15]
v_mov_b32 v[vgprValuC+119], v[vgprValuC+16]        // copy MI out reg to vreg[16]
v_mov_b32 v[vgprValuC+120], v[vgprValuC+24]        // copy MI out reg to vreg[17]
v_mov_b32 v[vgprValuC+121], v[vgprValuC+17]        // copy MI out reg to vreg[18]
v_mov_b32 v[vgprValuC+122], v[vgprValuC+25]        // copy MI out reg to vreg[19]
v_mov_b32 v[vgprValuC+123], v[vgprValuC+18]        // copy MI out reg to vreg[20]
v_mov_b32 v[vgprValuC+124], v[vgprValuC+26]        // copy MI out reg to vreg[21]
v_mov_b32 v[vgprValuC+125], v[vgprValuC+19]        // copy MI out reg to vreg[22]
v_mov_b32 v[vgprValuC+126], v[vgprValuC+27]        // copy MI out reg to vreg[23]
v_mov_b32 v[vgprValuC+127], v[vgprValuC+20]        // copy MI out reg to vreg[24]
v_mov_b32 v[vgprValuC+128], v[vgprValuC+28]        // copy MI out reg to vreg[25]
v_mov_b32 v[vgprValuC+129], v[vgprValuC+21]        // copy MI out reg to vreg[26]
v_mov_b32 v[vgprValuC+130], v[vgprValuC+29]        // copy MI out reg to vreg[27]
v_mov_b32 v[vgprValuC+131], v[vgprValuC+22]        // copy MI out reg to vreg[28]
v_mov_b32 v[vgprValuC+132], v[vgprValuC+30]        // copy MI out reg to vreg[29]
v_mov_b32 v[vgprValuC+133], v[vgprValuC+23]        // copy MI out reg to vreg[30]
v_mov_b32 v[vgprValuC+134], v[vgprValuC+31]        // copy MI out reg to vreg[31]
v_mov_b32 v[vgprValuC+135], v[vgprValuC+32]        // copy MI out reg to vreg[32]
v_mov_b32 v[vgprValuC+136], v[vgprValuC+40]        // copy MI out reg to vreg[33]
v_mov_b32 v[vgprValuC+137], v[vgprValuC+33]        // copy MI out reg to vreg[34]
v_mov_b32 v[vgprValuC+138], v[vgprValuC+41]        // copy MI out reg to vreg[35]
v_mov_b32 v[vgprValuC+139], v[vgprValuC+34]        // copy MI out reg to vreg[36]
v_mov_b32 v[vgprValuC+140], v[vgprValuC+42]        // copy MI out reg to vreg[37]
v_mov_b32 v[vgprValuC+141], v[vgprValuC+35]        // copy MI out reg to vreg[38]
v_mov_b32 v[vgprValuC+142], v[vgprValuC+43]        // copy MI out reg to vreg[39]
v_mov_b32 v[vgprValuC+143], v[vgprValuC+36]        // copy MI out reg to vreg[40]
v_mov_b32 v[vgprValuC+144], v[vgprValuC+44]        // copy MI out reg to vreg[41]
v_mov_b32 v[vgprValuC+145], v[vgprValuC+37]        // copy MI out reg to vreg[42]
v_mov_b32 v[vgprValuC+146], v[vgprValuC+45]        // copy MI out reg to vreg[43]
v_mov_b32 v[vgprValuC+147], v[vgprValuC+38]        // copy MI out reg to vreg[44]
v_mov_b32 v[vgprValuC+148], v[vgprValuC+46]        // copy MI out reg to vreg[45]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+39]        // copy MI out reg to vreg[46]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+47]        // copy MI out reg to vreg[47]

/* apply mask, calc new C and issue writes */
v_lshlrev_b32 v101, 2, v[vgprSerial]               // v101 = v[vgprSerial] * 4
s_mov_b32 s8, 0                                    // Init sgpr offset
buffer_store_b32 v103, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v104, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v105, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v106, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v107, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v108, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v109, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v110, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v111, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v112, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v113, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v114, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v115, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v116, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v117, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v118, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v119, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v120, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v121, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v122, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v123, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v124, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v125, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v126, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v127, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v128, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v129, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v130, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v131, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v132, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v133, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v134, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v135, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v136, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v137, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v138, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v139, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v140, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v141, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v142, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v143, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v144, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v145, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v146, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v147, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v148, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v149, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v150, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_waitcnt_vscnt null 0                             // wait for data store
s_barrier                                          // store all data before setting flag
s_lshl_b32 s8, s[sgprPersistentWorkGroupIndex], 2  // flag offset based on physical partial slot
v_readfirstlane_b32 s7, v[vgprSerial]              // Wave 0 updates flags
s_cmp_eq_u32 s7, 0                                 // Check for wave 0
s_cbranch_scc0 label_SK_SkipFlagSet                // Skip flag set
s_mov_b32 s7, 1                                    // flag data
v_mov_b32 v103, s7                                 // move flag value to vgpr
v_mov_b32 v104, 0                                  // zero vaddr offset
s_mov_b64 s[64:65], s[sgprAddressFlags:sgprAddressFlags+1]
s_mov_b32 s66, BufferOOB
s_mov_b32 s67, Srd127_96
buffer_store_b32 v103, v104, s[64:67], s8 offen offset:0 glc dlc // set flag
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
