
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
.text
.protected RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBGU8_UserArgs_MT96x192x64_MI16x16x1_SK_gfx1151
.globl RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBGU8_UserArgs_MT96x192x64_MI16x16x1_SK_gfx1151
.p2align 8
.type RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBGU8_UserArgs_MT96x192x64_MI16x16x1_SK_gfx1151,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBGU8_UserArgs_MT96x192x64_MI16x16x1_SK_gfx1151
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 256 // vgprs
  .amdhsa_next_free_sgpr 88 // sgprs
  .amdhsa_group_segment_fixed_size 41472 // lds bytes
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
/* Num SGPR   =88 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 24 x 3 */
/* SubGroup= 4 x 64 */
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
  - .name: RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBGU8_UserArgs_MT96x192x64_MI16x16x1_SK_gfx1151
    .symbol: 'RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBGU8_UserArgs_MT96x192x64_MI16x16x1_SK_gfx1151.kd'
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
      - .name:            AddressFlags
        .size:            8
        .offset:          64
        .value_kind:      global_buffer
        .value_type:      bf16
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
        .value_type:      bf16
        .address_space:   generic
      - .name:            C
        .size:            8
        .offset:          120
        .value_kind:      global_buffer
        .value_type:      bf16
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
    .group_segment_fixed_size:   41472
    .kernarg_segment_align:      8
    .kernarg_segment_size:       200
    .max_flat_workgroup_size:    256
    .private_segment_fixed_size: 0
    .sgpr_count:                 88
    .sgpr_spill_count:           0
    .vgpr_count:                 256
    .vgpr_spill_count:           0
    .wavefront_size:             32
...
.end_amdgpu_metadata
RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBGU8_UserArgs_MT96x192x64_MI16x16x1_SK_gfx1151:
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
/* ValuC range: [0-72), serializedStore enabled */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprBase, 90
.set vgprLocalWriteAddrA, 87
.set vgprLocalWriteAddrB, 88
.set vgprGlobalReadOffsetA, 72
.set vgprGlobalReadOffsetB, 75
.set vgprGlobalReadOffsetScaleA, 81
.set vgprG2LScaleA, 84
.set vgprLocalReadAddrA, 224
.set vgprLocalReadAddrB, 89
.set vgprSerial, 225

/******************************************/
/* VGPR Macro Assignments                 */
/******************************************/
.set vgprValuA_X0_I0_BASE, vgprBase+0
.set vgprValuB_X0_I0_BASE, vgprBase+49
.set vgprG2LA_BASE, vgprBase+98
.set vgprG2LB_BASE, vgprBase+110
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+24
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X1_I0, vgprValuB_X0_I0_BASE+24
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
.set sgprNumWorkGroups0, 9
.set sgprNumWorkGroups1, 10
.set sgprSizesFree, 12
.set sgprSizesSum, 15
.set sgprAddressA, 16
.set sgprAddressB, 18
.set sgprStridesA, 20
.set sgprStridesB, 22
.set sgprAddressFlags, 24
.set sgprItersPerTile, 26
.set sgprMagicNumberItersPerTile, 27
.set sgprMagicShiftItersPerTile, 28
.set sgprSKItersPerWG, 29
.set sgprskGrid, 30
.set sgprskTiles, 31
.set sgprAlpha, 32
.set sgprBeta, 33
.set sgprAddressWS, 34
.set sgprAddressD, 36
.set sgprAddressC, 38
.set sgprStridesD, 40
.set sgprStridesC, 42
.set sgprSrdWS, 44
.set sgprStreamKLocalEnd, 48
.set sgprStreamKLocalStart, 49
.set sgprPersistentIterationEnd, 50
.set sgprPersistentIteration, 51
.set sgprPersistentWorkGroupIndex, 52
.set sgprAddressScaleA, 54
.set sgprStrideScaleA, 53
.set sgprSrdScaleA, 56
.set sgprScaleAGMinus1, 60
.set sgprScaleAGShift, 61
.set sgprScaleAKMask, 62
.set sgprScaleAInc, 63
.set sgprScaleAKCnt, 64
.set sgprScaleALiftMagic, 65

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

.set MT0, 96
.set MT1, 192
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
s_load_b32 s11, s[sgprKernArgAddress:sgprKernArgAddress+1], 0

/* Load packed kernel args (StaggerU/GSU) */
s_load_b32 s45, s[sgprKernArgAddress:sgprKernArgAddress+1], 4

/* Load WGM data */
s_load_b32 s[sgprWGM], s[sgprKernArgAddress:sgprKernArgAddress+1], 8

/* Load num of WGs */
s_load_b32 s46, s[sgprKernArgAddress:sgprKernArgAddress+1], 12
s_waitcnt lgkmcnt(0)                               // load args
s_lshr_b32 s44, s11, 0x1e                          // Get arg type
s_and_b32 s11, 0x3fffffff, s11                     // Get nums of gemm
s_cmp_eq_u32 s44, 3                                // Is kernel argType == 3
s_cbranch_scc1 label_Bypass_ArgType3_to_ArgType0_Instance1
s_cmp_eq_u32 s44, 0                                // Is kernel args
s_cbranch_scc0 label_HBMArgs
label_Bypass_ArgType3_to_ArgType0_Instance1:
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], 0x10 // Shift common args
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_b512 s[12:27], s[sgprKernArgAddress:sgprKernArgAddress+1], 0 // 0
s_load_b512 s[28:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 64 // 64
s_load_b32 s[sgprScaleAGMinus1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0xb0
s_load_b64 s[sgprAddressScaleA:sgprAddressScaleA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x80
s_branch label_LoadArgsEnd
label_HBMArgs:

/* Load address of kernel arguments */
s_load_b64 s[sgprKernArgAddress:sgprKernArgAddress+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 16
s_waitcnt lgkmcnt(0)                               // wait for args to load
label_LoadArgsEnd:
s_mov_b32 s[sgprArgType], s44
s_mov_b32 m0, 0xa200                               // LDS clamp at 41472 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id
v_mov_b32 v[vgprG2LScaleA+0], 0                    // scaleA: low half of the f32 scale
v_mov_b32 v[vgprG2LScaleA+1], 0                    // scaleA: low half of the f32 scale
v_mov_b32 v[vgprG2LScaleA+2], 0                    // scaleA: low half of the f32 scale
s_mov_b32 vcc_hi, 0                                // Ensure hi bits are zero

/* remap workgroup to XCCs */
s_lshr_b32 s80, s[sgprWGM], 0x10                   // Get WGMXCC
s_ff1_i32_b32 s80, s80                             // Get log(WGMXCC)
s_lshr_b32 s81, s[sgprWGM], 0x16                   // Get CU_Count
/* remap WGs if WGMXCC > 1 ( log(WGMXCC) > 0 ) */
s_cmp_gt_i32 s80, 0
s_cbranch_scc0 label_skip_WGMXCC
/* only remap WGs in the range */
s_lshr_b32 s77, s46, s80
s_lshl_b32 s77, s77, s80
s_cmp_ge_u32 s[sgprWorkGroup0], s77
s_cbranch_scc1 label_skip_WGMXCC
s_cmp_eq_u32 s81, 0                                // CU_Count == 0 ?
s_cbranch_scc0 label_XCCG_nonzero
s_lshr_b32 s77, s[sgprWorkGroup0], s80
s_bfm_b32 s78, s80, 0
s_and_b32 s78, s[sgprWorkGroup0], s78
s_lshr_b32 s79, s46, s80
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
v_cvt_f64_u32 v[0:1], s81                          // s78 = s46 / s81
v_rcp_f64 v[0:1], v[0:1]                           // s78 = s46 / s81
v_cvt_f64_u32 v[2:3], s46                          // s78 = s46 / s81
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s78 = s46 / s81
v_cvt_u32_f64 v0, v[0:1]                           // s78 = s46 / s81
v_mul_lo_u32 v1, v0, s81                           // s78 = s46 / s81
v_sub_nc_u32 v2, s46, v1                           // s78 = s46 / s81
v_cmp_ge_u32 vcc_lo, v2, s81                       // s78 = s46 / s81
s_mov_b32 exec_lo, vcc_lo                          // s78 = s46 / s81
v_add_nc_u32 v0, v0, 1                             // s78 = s46 / s81
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s78, v0                        // quotient
s_mul_i32 s78, s78, s81
s_sub_u32 s79, s46, s78
s_cmp_gt_u32 s[sgprWorkGroup0], s78
s_cselect_b32 s78, s79, s81
s_lshr_b32 s78, s78, s80
s_bfm_b32 s79, s80, 0
s_and_b32 s79, s[sgprWorkGroup0], s79
s_mul_i32 s78, s78, s79
/* WorkGroup0 = temp0 + temp1 */
s_add_u32 s[sgprWorkGroup0], s77, s78
label_skip_WGMXCC:  /// skip WGMXCC if no enough WGs to remap
s_cmp_eq_u32 s44, 3
s_cbranch_scc1 label_ArgType3_Routed_To_ArgType0
s_cmp_eq_u32 s44, 0
s_cbranch_scc0 label_MultiGemm
label_ArgType3_Routed_To_ArgType0:
/* init: add vgpr [90...277) to pool */
/* init: add vgpr [0...72) to pool */
/* init: add agpr [0...0) to pool */
s_waitcnt lgkmcnt(0)                               // wait for 128/0 bytes of kern args over preload
s_mul_hi_u32 s77, s[sgprSizesFree+0], 89478486     // STATIC_DIV: divisor=96
s_mul_i32 s76, s[sgprSizesFree+0], 89478486        // tmp = dividend * magic
s_lshr_b64 s[76:77], s[76:77], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s[sgprNumWorkGroups0], s76               // quotient
s_mul_i32 s76, s[sgprNumWorkGroups0], 96           // quotient*divisor
s_sub_u32 s45, s[sgprSizesFree+0], s76             // rReg = dividend - quotient*divisor
s_cmp_gt_u32 s45, 0                                // Check if remainder > 0
s_addc_u32 s[sgprNumWorkGroups0], s[sgprNumWorkGroups0], 0 // Increment if remainder > 0
s_mul_hi_u32 s77, s[sgprSizesFree+1], 44739243     // STATIC_DIV: divisor=192
s_mul_i32 s76, s[sgprSizesFree+1], 44739243        // tmp = dividend * magic
s_lshr_b64 s[76:77], s[76:77], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s[sgprNumWorkGroups1], s76               // quotient
s_mul_i32 s76, s[sgprNumWorkGroups1], 192          // quotient*divisor
s_sub_u32 s45, s[sgprSizesFree+1], s76             // rReg = dividend - quotient*divisor
s_cmp_gt_u32 s45, 0                                // Check if remainder > 0
s_addc_u32 s[sgprNumWorkGroups1], s[sgprNumWorkGroups1], 0 // Increment if remainder > 0
s_branch label_MultiGemmEnd
label_MultiGemm:

/* Check if custom structure pointer is null */
s_and_b32 s44, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s44, 2                                // ArgType == 2 ?
s_cbranch_scc1 label_IsExternalValid               // branch if ArgType == 2
s_mov_b32 s10, 144                                 // KernArgAddressOffset
s_mul_i32 s82, s11, 4
s_mov_b64 s[76:77], s[sgprKernArgAddress:sgprKernArgAddress+1]
s_branch label_IsExternalValidEnd
label_IsExternalValid:
s_mov_b32 s10, 256
s_mov_b32 s82, 0
s_mov_b64 s[76:77], s[sgprKernArgAddress:sgprKernArgAddress+1]
label_IsExternalValidEnd:

/* Grouped Gemm:: prefetch 1 arg load */
s_mov_b32 s9, 1
s_mov_b32 s83, 0
s_load_b128 s[12:15], s[76:77], s82
s_cmpk_eq_u32 s11, 1                               // if gemm_count is 1?
s_cbranch_scc1 label_wgTable_noLoadLoop

/* Grouped Gemm:: accumulate numTiles for each gemm */
/* Grouped Gemm:: loop start */
label_Loop_GemmCount:
s_waitcnt lgkmcnt(0)
s_mul_hi_u32 s79, s12, 89478486                    // STATIC_DIV: divisor=96
s_mul_i32 s78, s12, 89478486                       // tmp = dividend * magic
s_lshr_b64 s[78:79], s[78:79], 33                  // tmp0 = quotient
s_mul_i32 s79, s78, 96                             // tmp1 = quotient * divisor
s_cmp_lg_u32 s79, s12                              // if (quotient * divisor != dividend), result+=1
s_addc_u32 s80, s78, 0                             // if (quotient * divisor != dividend), result+=1
s_mul_hi_u32 s79, s13, 44739243                    // STATIC_DIV: divisor=192
s_mul_i32 s78, s13, 44739243                       // tmp = dividend * magic
s_lshr_b64 s[78:79], s[78:79], 33                  // tmp0 = quotient
s_mul_i32 s79, s78, 192                            // tmp1 = quotient * divisor
s_cmp_lg_u32 s79, s13                              // if (quotient * divisor != dividend), result+=1
s_addc_u32 s81, s78, 0                             // if (quotient * divisor != dividend), result+=1
s_mul_i32 s80, s80, s81
s_mul_i32 s80, s80, s14
s_add_u32 s83, s83, s80
s_cmp_lt_u32 s[sgprWorkGroup0], s83
s_cbranch_scc1 label_FOUND
s_add_u32 s82, s82, s10
s_load_b128 s[12:15], s[76:77], s82
s_add_u32 s9, s9, 1
s_cmp_lt_u32 s9, s11
s_cbranch_scc1 label_Loop_GemmCount

/* Grouped Gemm:: noLoadLoop */
label_wgTable_noLoadLoop:
s_waitcnt lgkmcnt(0)
s_mul_hi_u32 s79, s12, 89478486                    // STATIC_DIV: divisor=96
s_mul_i32 s78, s12, 89478486                       // tmp = dividend * magic
s_lshr_b64 s[78:79], s[78:79], 33                  // tmp0 = quotient
s_mul_i32 s79, s78, 96                             // tmp1 = quotient * divisor
s_cmp_lg_u32 s79, s12                              // if (quotient * divisor != dividend), result+=1
s_addc_u32 s80, s78, 0                             // if (quotient * divisor != dividend), result+=1
s_mul_hi_u32 s79, s13, 44739243                    // STATIC_DIV: divisor=192
s_mul_i32 s78, s13, 44739243                       // tmp = dividend * magic
s_lshr_b64 s[78:79], s[78:79], 33                  // tmp0 = quotient
s_mul_i32 s79, s78, 192                            // tmp1 = quotient * divisor
s_cmp_lg_u32 s79, s13                              // if (quotient * divisor != dividend), result+=1
s_addc_u32 s81, s78, 0                             // if (quotient * divisor != dividend), result+=1
s_mul_i32 s80, s80, s81
s_mul_i32 s80, s80, s14
s_add_u32 s83, s83, s80

/* Grouped Gemm:: gemmIndex found */
label_FOUND:
s_sub_u32 s77, s9, 1
s_sub_u32 s76, s83, s80
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s76
/* Check if custom structure pointer is null */
s_and_b32 s44, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s44, 2                                // ArgType == 2 ?
s_cbranch_scc1 label_LoadExternalStruct            // branch if ArgType == 2

/* Grouped Gemm: offset argument address to gemm */
/* Grouped Gemm: offset address from wg_table_start to args_start */
s_lshl2_add_u32 s[sgprKernArgAddress], s11, s[sgprKernArgAddress]
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s77, s77, 144                            // KernArgAddressOffset
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s77
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_b512 s[16:31], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_b256 s[32:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
s_load_b128 s[40:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 112 // 112
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
s_load_b32 s33, s[sgprKernArgAddress:sgprKernArgAddress+1], 136 // 136
s_load_b64 s[sgprAddressScaleA:sgprAddressScaleA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x8c
label_LoadExternalStructEnd:
/* init: add vgpr [90...277) to pool */
/* init: add vgpr [0...72) to pool */
/* init: add agpr [0...0) to pool */
s_waitcnt lgkmcnt(0)                               // wait for 128/0 bytes of kern args over preload
s_mul_hi_u32 s77, s[sgprSizesFree+0], 89478486     // STATIC_DIV: divisor=96
s_mul_i32 s76, s[sgprSizesFree+0], 89478486        // tmp = dividend * magic
s_lshr_b64 s[76:77], s[76:77], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s[sgprNumWorkGroups0], s76               // quotient
s_mul_i32 s76, s[sgprNumWorkGroups0], 96           // quotient*divisor
s_sub_u32 s45, s[sgprSizesFree+0], s76             // rReg = dividend - quotient*divisor
s_cmp_gt_u32 s45, 0                                // Check if remainder > 0
s_addc_u32 s[sgprNumWorkGroups0], s[sgprNumWorkGroups0], 0 // Increment if remainder > 0
s_mul_hi_u32 s77, s[sgprSizesFree+1], 44739243     // STATIC_DIV: divisor=192
s_mul_i32 s76, s[sgprSizesFree+1], 44739243        // tmp = dividend * magic
s_lshr_b64 s[76:77], s[76:77], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s[sgprNumWorkGroups1], s76               // quotient
s_mul_i32 s76, s[sgprNumWorkGroups1], 192          // quotient*divisor
s_sub_u32 s45, s[sgprSizesFree+1], s76             // rReg = dividend - quotient*divisor
s_cmp_gt_u32 s45, 0                                // Check if remainder > 0
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
v_readfirstlane_b32 s11, v0                        // quotient
v_readfirstlane_b32 s44, v1                        // remainder
s_mul_i32 s45, s[sgprskTiles], s[sgprSKItersPerWG]
s_sub_u32 s45, s[sgprItersPerTile], s45            // extraIters = itersPerTile - SkSplit * skItersPerWG
s_mul_i32 s[sgprPersistentIteration], s44, s[sgprSKItersPerWG] // StreamK starting iteration (case: after extra iters)
s_cmp_lt_u32 s44, s45                              // Check if WG gets an extra iteration
s_cbranch_scc1 label_SK_HasExtra                   // Has extra iter
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s45 // This WG does not have an extra iteration
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIteration], s[sgprSKItersPerWG] // StreamK ending iteration (case: after extra iters)
s_branch label_SK_DoneExtra                        // Done init for parallel reduction
label_SK_HasExtra:
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s44 // This WG has an extra iteration
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIteration], s[sgprSKItersPerWG] // StreamK ending iteration (case: after extra iters)
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], 1 // StreamK ending iteration (case: after extra iters)
label_SK_DoneExtra:
s_mul_i32 s11, s11, s[sgprItersPerTile]            // Tile offset = tilesIdx * itersPerTile
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s11 // Offset to correct tile
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], s11 // Offset to correct tile
s_mov_b32 s[sgprSkPartialIdx], s44                 // Save partial idx for SrdD calculation
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
s_mul_i32 s44, s[sgprSKItersPerWG], s[sgprskGrid]
s_sub_u32 s11, s11, s44                            // skTiles * ItersPerTile - SKItersPerWG * skGrid
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
v_readfirstlane_b32 s44, v0                        // quotient
v_readfirstlane_b32 s45, v1                        // remainder
s_cmp_eq_u32 s45, 0                                // skGrid % skTiles == 0?
s_cbranch_scc1 label_SK_PerTileExtraIters          // all-partial -> per-tile extras
s_branch label_SK_GlobalExtraIters                 // ragged -> global mapping
label_SK_AssignNoTiles:
label_SK_GlobalExtraIters:
s_mul_i32 s[sgprPersistentIteration], s[sgprPersistentWorkGroupIndex], s[sgprSKItersPerWG] // StreamK starting iteration (case: after extra iters)
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s11 // Add extra iters
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIteration], s[sgprSKItersPerWG] // StreamK ending iteration (case: after extra iters)
s_add_u32 s45, s[sgprSKItersPerWG], 1              // Spread out extra iterations
s_mul_i32 s44, s[sgprPersistentWorkGroupIndex], s45 // StreamK starting iteration (case: before extra iters)
s_add_u32 s45, s44, s45                            // StreamK ending iteration (case: before extra iters)
s_cmp_lt_u32 s[sgprPersistentWorkGroupIndex], s11  // Check if lane gets an extra iteration
s_cselect_b32 s[sgprPersistentIteration], s44, s[sgprPersistentIteration] // Set start iter
s_cselect_b32 s[sgprPersistentIterationEnd], s45, s[sgprPersistentIterationEnd] // Set end iter
s_branch label_SK_AssignItersDone                  // skip per-tile path
label_SK_PerTileExtraIters:
s_mov_b32 s11, s44                                 // F = skGrid / skTiles
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
v_readfirstlane_b32 s44, v0                        // quotient
v_readfirstlane_b32 s45, v1                        // remainder
s_mul_i32 s[sgprPersistentIteration], s44, s[sgprItersPerTile] // q * ItersPerTile
s_mul_i32 s44, s45, s[sgprSKItersPerWG]            // s * SKItersPerWG
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s44 // q*I + s*W
s_mul_i32 s44, s11, s[sgprSKItersPerWG]            // F * SKItersPerWG
s_sub_u32 s44, s[sgprItersPerTile], s44            // remI = ItersPerTile - F*SKItersPerWG
s_min_u32 s11, s45, s44                            // min(s, remI)
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s11 // start = q*I + s*W + min(s, remI)
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIteration], s[sgprSKItersPerWG] // start + SKItersPerWG
s_cmp_lt_u32 s45, s44                              // s < remI?
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
s_getpc_b64 s[44:45]                               // addr of next instr
s_add_i32 s46, label_KernelEnd, 4                  // target branch offset
s_add_u32 s44, s44, s46                            // add target branch offset
s_addc_u32 s45, s45, 0                             // add high and carry
s_setpc_b64 s[44:45]                               // branch to label_KernelEnd
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
s_mul_hi_u32 s45, s[sgprPersistentIteration], s[sgprMagicNumberItersPerTile] // s_magic mul, div alg 2
s_lshr_b32 s46, s[sgprMagicShiftItersPerTile], 31  // tmpS = extract abit
s_mul_i32 s44, s[sgprPersistentIteration], s46     // s_magic mul, div alg 2
s_add_u32 s44, s44, s45
s_and_b32 s46, s[sgprMagicShiftItersPerTile], 2147483647 // tmpS = remove abit to final shift
s_lshr_b32 s44, s44, s46                           // sMagicDiv Alg 2
s_mul_i32 s45, s44, s[sgprItersPerTile]            // Tile start iteration
s_add_u32 s46, s45, s[sgprItersPerTile]            // Tile end iteration
s_sub_u32 s[sgprStreamKLocalStart], s[sgprPersistentIteration], s45 // Local iteration start
s_min_u32 s[sgprStreamKLocalEnd], s[sgprPersistentIterationEnd], s46 // 1. (Local) iteration end (SK tile)
s_sub_u32 s[sgprStreamKLocalEnd], s[sgprStreamKLocalEnd], s45 // 2. Local iteration end (SK tile)
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_SK_SplitUpdate                // Jump to single kernel update
s_mov_b32 s45, s[sgprPersistentIterationEnd]       // Parallel reduction, work contained to single partial tile
s_branch label_SK_UpdateDone                       // Done update for parallel reduction
label_SK_SplitUpdate:
s_mul_i32 s47, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1] // totalTiles = nwg0 * nwg1
s_mul_i32 s47, s47, s[sgprSizesFree+2]             // totalTiles *= batch dim 0
s_sub_u32 s47, s47, s[sgprskTiles]                 // dpTiles = totalTiles - skTiles
s_mul_i32 s47, s47, s[sgprItersPerTile]            // dpSectionSize = dpTiles * ItersPerTile
s_mul_i32 s45, s[sgprskGrid], s[sgprItersPerTile]  // DP iterations shift
s_add_u32 s45, s45, s[sgprPersistentIteration]     // Add DP shift
s_cmp_lt_u32 s45, s47                              // Check if still in DP section
s_cbranch_scc1 label_SK_UpdateDone                 // Done update
s_mov_b32 s45, s46                                 // SK iterations shift
s_cmp_le_u32 s47, s[sgprPersistentIteration]       // Check if continuing in SK section
s_cbranch_scc1 label_SK_UpdateDone                 // Done update
s_mov_b32 s76, s47                                 // park dpSectionSize
s_mov_b32 s77, s44                                 // park current tile idx
s_mul_i32 s11, s[sgprskTiles], s[sgprItersPerTile]
s_mul_i32 s44, s[sgprSKItersPerWG], s[sgprskGrid]
s_sub_u32 s11, s11, s44                            // skTiles * ItersPerTile - SKItersPerWG * skGrid
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
v_readfirstlane_b32 s44, v0                        // quotient
v_readfirstlane_b32 s45, v1                        // remainder
s_cmp_eq_u32 s45, 0                                // skGrid % skTiles == 0?
s_cbranch_scc1 label_SK_PerTileExtraIters_1        // all-partial -> per-tile extras
s_branch label_SK_GlobalExtraIters_1               // ragged -> global mapping
label_SK_AssignNoTiles_1:
label_SK_GlobalExtraIters_1:
s_mul_i32 s[sgprPersistentIteration], s[sgprPersistentWorkGroupIndex], s[sgprSKItersPerWG] // StreamK starting iteration (case: after extra iters)
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s11 // Add extra iters
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIteration], s[sgprSKItersPerWG] // StreamK ending iteration (case: after extra iters)
s_add_u32 s45, s[sgprSKItersPerWG], 1              // Spread out extra iterations
s_mul_i32 s44, s[sgprPersistentWorkGroupIndex], s45 // StreamK starting iteration (case: before extra iters)
s_add_u32 s45, s44, s45                            // StreamK ending iteration (case: before extra iters)
s_cmp_lt_u32 s[sgprPersistentWorkGroupIndex], s11  // Check if lane gets an extra iteration
s_cselect_b32 s[sgprPersistentIteration], s44, s[sgprPersistentIteration] // Set start iter
s_cselect_b32 s[sgprPersistentIterationEnd], s45, s[sgprPersistentIterationEnd] // Set end iter
s_branch label_SK_AssignItersDone_1                // skip per-tile path
label_SK_PerTileExtraIters_1:
s_mov_b32 s11, s44                                 // F = skGrid / skTiles
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
v_readfirstlane_b32 s44, v0                        // quotient
v_readfirstlane_b32 s45, v1                        // remainder
s_mul_i32 s[sgprPersistentIteration], s44, s[sgprItersPerTile] // q * ItersPerTile
s_mul_i32 s44, s45, s[sgprSKItersPerWG]            // s * SKItersPerWG
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s44 // q*I + s*W
s_mul_i32 s44, s11, s[sgprSKItersPerWG]            // F * SKItersPerWG
s_sub_u32 s44, s[sgprItersPerTile], s44            // remI = ItersPerTile - F*SKItersPerWG
s_min_u32 s11, s45, s44                            // min(s, remI)
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s11 // start = q*I + s*W + min(s, remI)
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIteration], s[sgprSKItersPerWG] // start + SKItersPerWG
s_cmp_lt_u32 s45, s44                              // s < remI?
s_cselect_b32 s11, 1, 0                            // extra iter within tile
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], s11 // end = start + W + (s < remI)
label_SK_AssignItersDone_1:
s_mov_b32 s44, s77                                 // restore current tile idx
s_add_u32 s45, s[sgprPersistentIteration], s76     // Offset to start of SK section
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], s76 // Offset to start of SK section
s_mul_i32 s11, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1] // totalTiles = nwg0 * nwg1
s_mul_i32 s11, s11, s[sgprSizesFree+2]             // totalTiles *= batch dim 0
s_mul_i32 s11, s11, s[sgprItersPerTile]            // totalIters = totalTiles * itersPerTile
s_min_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], s11 // Cap ending iter at total SK iters
s_cmp_lt_u32 s[sgprPersistentIteration], s11       // Make sure there's work to do
s_cbranch_scc1 label_NoBranch_2                    // Only branch on scc0
s_getpc_b64 s[76:77]                               // addr of next instr
s_add_i32 s78, label_KernelEnd, 4                  // target branch offset
s_add_u32 s76, s76, s78                            // add target branch offset
s_addc_u32 s77, s77, 0                             // add high and carry
s_setpc_b64 s[76:77]                               // branch to label_KernelEnd
label_NoBranch_2:
label_SK_UpdateDone:
s_mov_b32 s[sgprPersistentIteration], s45          // Store current iteration
/* Map persistent tile index to wg0/1/2 */
s_mul_i32 s45, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1] // Total tiles
v_cvt_f32_u32 v0, s45                              // TileID // nWG0*nWG1
v_rcp_iflag_f32 v0, v0                             // TileID // nWG0*nWG1
v_mov_b32 v1, 0x4f7ffffe                           // 2^32 - 512
v_mul_f32 v0, v1, v0                               // 2^32 / divisor
v_cvt_u32_f32 v0, v0                               // TileID // nWG0*nWG1
v_mul_lo_u32 v1, v0, s45                           // refine 2^32 / divisor
v_sub_nc_u32 v1, 0, v1                             // refine 2^32 / divisor
v_mul_hi_u32 v1, v0, v1                            // refine 2^32 / divisor
v_add_nc_u32 v0, v0, v1                            // refine 2^32 / divisor
v_mul_hi_u32 v0, s44, v0                           // TileID // nWG0*nWG1
v_mul_lo_u32 v1, v0, s45                           // TileID // nWG0*nWG1
v_sub_nc_u32 v1, s44, v1                           // TileID // nWG0*nWG1
v_cmp_ge_u32 vcc_lo, v1, s45                       // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v0, 1, v0                             // quotient + 1
v_sub_nc_u32 v1, v1, s45                           // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v1, s45                       // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v0, 1, v0                             // quotient + 1
v_sub_nc_u32 v1, v1, s45                           // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprWorkGroup2], v0          // quotient
v_readfirstlane_b32 s46, v1                        // remainder
v_cvt_f32_u32 v0, s[sgprNumWorkGroups0]            // TileID // nWG0
v_rcp_iflag_f32 v0, v0                             // TileID // nWG0
v_mov_b32 v1, 0x4f7ffffe                           // 2^32 - 512
v_mul_f32 v0, v1, v0                               // 2^32 / divisor
v_cvt_u32_f32 v0, v0                               // TileID // nWG0
v_mul_lo_u32 v1, v0, s[sgprNumWorkGroups0]         // refine 2^32 / divisor
v_sub_nc_u32 v1, 0, v1                             // refine 2^32 / divisor
v_mul_hi_u32 v1, v0, v1                            // refine 2^32 / divisor
v_add_nc_u32 v0, v0, v1                            // refine 2^32 / divisor
v_mul_hi_u32 v0, s46, v0                           // TileID // nWG0
v_mul_lo_u32 v1, v0, s[sgprNumWorkGroups0]         // TileID // nWG0
v_sub_nc_u32 v1, s46, v1                           // TileID // nWG0
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
s_getpc_b64 s[76:77]                               // addr of next instr
s_add_i32 s78, label_PersistentLoopClose, 4        // target branch offset
s_add_u32 s76, s76, s78                            // add target branch offset
s_addc_u32 s77, s77, 0                             // add high and carry
s_setpc_b64 s[76:77]                               // branch to label_PersistentLoopClose
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
v_cvt_f64_u32 v[0:1], s11                          // s46 = s[sgprWorkGroup0] / s11
v_rcp_f64 v[0:1], v[0:1]                           // s46 = s[sgprWorkGroup0] / s11
v_cvt_f64_u32 v[2:3], s[sgprWorkGroup0]            // s46 = s[sgprWorkGroup0] / s11
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s46 = s[sgprWorkGroup0] / s11
v_cvt_u32_f64 v0, v[0:1]                           // s46 = s[sgprWorkGroup0] / s11
v_mul_lo_u32 v1, v0, s11                           // s46 = s[sgprWorkGroup0] / s11
v_sub_nc_u32 v2, s[sgprWorkGroup0], v1             // s46 = s[sgprWorkGroup0] / s11
v_cmp_ge_u32 vcc_lo, v2, s11                       // s46 = s[sgprWorkGroup0] / s11
s_mov_b32 exec_lo, vcc_lo                          // s46 = s[sgprWorkGroup0] / s11
v_add_nc_u32 v0, v0, 1                             // s46 = s[sgprWorkGroup0] / s11
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s46, v0                        // quotient
s_mul_i32 s47, s46, s11                            // quotient * non-magic divisor
s_sub_u32 s47, s[sgprWorkGroup0], s47              // WorkGroup0=remainder
s_mul_i32 s47, s47, s[sgprNumWorkGroups1]          // (wg1 % WGM)*NumWorkGroups1
s_add_u32 s47, s47, s[sgprWorkGroup1]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups1
v_cvt_f64_u32 v[0:1], s11                          // s44 = s[sgprNumWorkGroups0] / s11
v_rcp_f64 v[0:1], v[0:1]                           // s44 = s[sgprNumWorkGroups0] / s11
v_cvt_f64_u32 v[2:3], s[sgprNumWorkGroups0]        // s44 = s[sgprNumWorkGroups0] / s11
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s44 = s[sgprNumWorkGroups0] / s11
v_cvt_u32_f64 v0, v[0:1]                           // s44 = s[sgprNumWorkGroups0] / s11
v_mul_lo_u32 v1, v0, s11                           // s44 = s[sgprNumWorkGroups0] / s11
v_sub_nc_u32 v2, s[sgprNumWorkGroups0], v1         // s44 = s[sgprNumWorkGroups0] / s11
v_cmp_ge_u32 vcc_lo, v2, s11                       // s44 = s[sgprNumWorkGroups0] / s11
s_mov_b32 exec_lo, vcc_lo                          // s44 = s[sgprNumWorkGroups0] / s11
v_add_nc_u32 v0, v0, 1                             // s44 = s[sgprNumWorkGroups0] / s11
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s44, v0                        // quotient
s_mul_i32 s45, s11, s44                            // quotient * non-magic divisor
s_sub_u32 s45, s[sgprNumWorkGroups0], s45          // NumWorkGroups0=remainder
s_cmp_eq_u32 s45, 0                                // remainder == 0 ?
s_cmov_b32 s45, s11                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s46, s44                              // blockId >= numFullBlocks ?
s_cselect_b32 s44, s45, s11
v_cvt_f64_u32 v[0:1], s44                          // s[sgprWorkGroup1] = s47 / s44
v_rcp_f64 v[0:1], v[0:1]                           // s[sgprWorkGroup1] = s47 / s44
v_cvt_f64_u32 v[2:3], s47                          // s[sgprWorkGroup1] = s47 / s44
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s[sgprWorkGroup1] = s47 / s44
v_cvt_u32_f64 v0, v[0:1]                           // s[sgprWorkGroup1] = s47 / s44
v_mul_lo_u32 v1, v0, s44                           // s[sgprWorkGroup1] = s47 / s44
v_sub_nc_u32 v2, s47, v1                           // s[sgprWorkGroup1] = s47 / s44
v_cmp_ge_u32 vcc_lo, v2, s44                       // s[sgprWorkGroup1] = s47 / s44
s_mov_b32 exec_lo, vcc_lo                          // s[sgprWorkGroup1] = s47 / s44
v_add_nc_u32 v0, v0, 1                             // s[sgprWorkGroup1] = s47 / s44
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v1, v0, s44                           // s[sgprWorkGroup1] = s47 / s44
v_sub_nc_u32 v2, s47, v1                           // s[sgprWorkGroup1] = s47 / s44
v_readfirstlane_b32 s[sgprWorkGroup1], v0          // quotient
v_readfirstlane_b32 s[sgprWorkGroup0], v2          // remainder
s_mul_i32 s[sgprWorkGroup0], s[sgprWorkGroup1], s44 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup0], s47, s[sgprWorkGroup0] // WorkGroup0=remainder
s_mul_i32 s46, s46, s11                            // blockId * WGM
s_add_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s46 // wg1 += blockId * WGM
s_branch label_WGM
label_WGMPositive:
s_mov_b32 s11, s11                                 // WGM
v_cvt_f64_u32 v[0:1], s11                          // s46 = s[sgprWorkGroup1] / s11
v_rcp_f64 v[0:1], v[0:1]                           // s46 = s[sgprWorkGroup1] / s11
v_cvt_f64_u32 v[2:3], s[sgprWorkGroup1]            // s46 = s[sgprWorkGroup1] / s11
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s46 = s[sgprWorkGroup1] / s11
v_cvt_u32_f64 v0, v[0:1]                           // s46 = s[sgprWorkGroup1] / s11
v_mul_lo_u32 v1, v0, s11                           // s46 = s[sgprWorkGroup1] / s11
v_sub_nc_u32 v2, s[sgprWorkGroup1], v1             // s46 = s[sgprWorkGroup1] / s11
v_cmp_ge_u32 vcc_lo, v2, s11                       // s46 = s[sgprWorkGroup1] / s11
s_mov_b32 exec_lo, vcc_lo                          // s46 = s[sgprWorkGroup1] / s11
v_add_nc_u32 v0, v0, 1                             // s46 = s[sgprWorkGroup1] / s11
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s46, v0                        // quotient
s_mul_i32 s47, s46, s11                            // quotient * non-magic divisor
s_sub_u32 s47, s[sgprWorkGroup1], s47              // WorkGroup1=remainder
s_mul_i32 s47, s47, s[sgprNumWorkGroups0]          // (wg1 % WGM)*NumWorkGroups0
s_add_u32 s47, s47, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups0
v_cvt_f64_u32 v[0:1], s11                          // s44 = s[sgprNumWorkGroups1] / s11
v_rcp_f64 v[0:1], v[0:1]                           // s44 = s[sgprNumWorkGroups1] / s11
v_cvt_f64_u32 v[2:3], s[sgprNumWorkGroups1]        // s44 = s[sgprNumWorkGroups1] / s11
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s44 = s[sgprNumWorkGroups1] / s11
v_cvt_u32_f64 v0, v[0:1]                           // s44 = s[sgprNumWorkGroups1] / s11
v_mul_lo_u32 v1, v0, s11                           // s44 = s[sgprNumWorkGroups1] / s11
v_sub_nc_u32 v2, s[sgprNumWorkGroups1], v1         // s44 = s[sgprNumWorkGroups1] / s11
v_cmp_ge_u32 vcc_lo, v2, s11                       // s44 = s[sgprNumWorkGroups1] / s11
s_mov_b32 exec_lo, vcc_lo                          // s44 = s[sgprNumWorkGroups1] / s11
v_add_nc_u32 v0, v0, 1                             // s44 = s[sgprNumWorkGroups1] / s11
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s44, v0                        // quotient
s_mul_i32 s45, s11, s44                            // quotient * non-magic divisor
s_sub_u32 s45, s[sgprNumWorkGroups1], s45          // NumWorkGroups1=remainder
s_cmp_eq_u32 s45, 0                                // remainder == 0 ?
s_cmov_b32 s45, s11                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s46, s44                              // blockId >= numFullBlocks ?
s_cselect_b32 s44, s45, s11
v_cvt_f64_u32 v[0:1], s44                          // s[sgprWorkGroup0] = s47 / s44
v_rcp_f64 v[0:1], v[0:1]                           // s[sgprWorkGroup0] = s47 / s44
v_cvt_f64_u32 v[2:3], s47                          // s[sgprWorkGroup0] = s47 / s44
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s[sgprWorkGroup0] = s47 / s44
v_cvt_u32_f64 v0, v[0:1]                           // s[sgprWorkGroup0] = s47 / s44
v_mul_lo_u32 v1, v0, s44                           // s[sgprWorkGroup0] = s47 / s44
v_sub_nc_u32 v2, s47, v1                           // s[sgprWorkGroup0] = s47 / s44
v_cmp_ge_u32 vcc_lo, v2, s44                       // s[sgprWorkGroup0] = s47 / s44
s_mov_b32 exec_lo, vcc_lo                          // s[sgprWorkGroup0] = s47 / s44
v_add_nc_u32 v0, v0, 1                             // s[sgprWorkGroup0] = s47 / s44
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v1, v0, s44                           // s[sgprWorkGroup0] = s47 / s44
v_sub_nc_u32 v2, s47, v1                           // s[sgprWorkGroup0] = s47 / s44
v_readfirstlane_b32 s[sgprWorkGroup0], v0          // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v2          // remainder
s_mul_i32 s[sgprWorkGroup1], s[sgprWorkGroup0], s44 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup1], s47, s[sgprWorkGroup1] // WorkGroup1=remainder
s_mul_i32 s46, s46, s11                            // blockId * WGM
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s46 // wg1 += blockId * WGM
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
v_and_b32 v3, 3, v3                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(4)
v_lshl_add_u32 v1, v3, 10, v1                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(1024); 7. final local read offset: flrOffset = lrOffset + WOffset

/* local read addresses: final offsets a */
v_lshrrev_b32 v2, 5, v[vgprSerial]                 // 2 = Serial / 32
v_lshrrev_b32 v2, 3, v2                            // LSU offset: Get LSU wave_id
s_mov_b32 s11, 64                                  // LSU offset: stride = lsuStride(64) when umlds==True
v_mul_lo_u32 v2, s11, v2                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_nc_u32 v[vgprLocalReadAddrA], v2, v0         // Final Offset: offset = (lro0+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrA], 1, v[vgprLocalReadAddrA] //  (multiple bpe)
v_lshrrev_b32 v3, 7, v[vgprLocalReadAddrA]         // Final Offset: padding 16 per block 128
v_lshl_add_u32 v[vgprLocalReadAddrA], v3, 4, v[vgprLocalReadAddrA] // Final Offset: padding 16 per block 128

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 5, v[vgprSerial]                 // 0 = Serial / 32
v_lshrrev_b32 v0, 3, v0                            // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(64) when umlds==True (dup assign opt.)
v_mul_lo_u32 v0, s11, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_nc_u32 v[vgprLocalReadAddrB], v0, v1         // Final Offset: offset = (lro1+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrB], 1, v[vgprLocalReadAddrB] //  (multiple bpe)
v_lshrrev_b32 v2, 7, v[vgprLocalReadAddrB]         // Final Offset: padding 16 per block 128
v_lshl_add_u32 v[vgprLocalReadAddrB], v2, 4, v[vgprLocalReadAddrB] // Final Offset: padding 16 per block 128

/* local read addresses: declare addresses a */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, 0x3600, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 8 */
/* v1 = A-unroll = serial%LVCA */
v_and_b32 v4, 31, v[vgprSerial]                    // v4 = v[vgprSerial] % 32
v_lshrrev_b32 v0, 3, v4                            // 0 = 4 / 8
v_and_b32 v1, 7, v4                                // 1 = 4 % 8
v_lshrrev_b32 v4, 0x5, v[vgprSerial]               // WaveID
v_mov_b32 v5, 24                                   // Global Read Wave: add back to cloumn index
v_mul_lo_u32 v0, v5, v0                            // Global Read Wave: add back to cloumn index
v_add_nc_u32 v0, v4, v0                            // Global Read Wave: add back to cloumn index
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
v_add_co_u32 v[vgprLocalWriteAddrB], vcc_lo, 0x3600, v[vgprLocalWriteAddrB] // lwFOB = lw1J + lwL*MT1J + LDS_OFFSET_B=13824

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

/* global read addresses: tile offsets b */
v_mov_b32 v9, v2                                   // groB1J_0
v_add_co_u32 v10, vcc_lo, 32, v9                   // groB1J_1 += LSPB
v_add_co_u32 v11, vcc_lo, 32, v10                  // groB1J_2 += LSPB
v_add_co_u32 v12, vcc_lo, 32, v11                  // groB1J_3 += LSPB
v_add_co_u32 v13, vcc_lo, 32, v12                  // groB1J_4 += LSPB
v_add_co_u32 v14, vcc_lo, 32, v13                  // groB1J_5 += LSPB

/* global read addresses: unroll offsets a */
v_mov_b32 v15, v1                                  // groAL_0

/* global read addresses: unroll offsets b */
v_mov_b32 v16, v3                                  // groBL_0

/* global read addresses: addresses a */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s47, s[sgprWorkGroup0], 96            // WorkGroup[01] * MT
s_mul_i32 s46, s[sgprWorkGroup0], 96               // WorkGroup[01] * MT
s_mul_hi_u32 s47, s46, s[sgprStrideA0I]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s46, s46, s[sgprStrideA0I]               // tlu=0, scaled tile-offset by stride
s_mul_i32 s44, s[sgprStreamKLocalStart], 64        // StreamK tile start offset
s_mul_hi_u32 s45, s44, constStrideAL               // StreamK tile start offset
s_mul_i32 s44, s44, constStrideAL                  // StreamK tile start offset
s_add_u32 s46, s46, s44                            // accum GsuOffset term to tilestart
s_addc_u32 s47, s47, s45                           // accum GsuOffset term to tilestart
s_mov_b64 s[sgprShadowLimitA+0:sgprShadowLimitA+0+1], 1 // Init tensor size
s_sub_u32 s44, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s45, constStrideAL, s44               // stride x (size-1)
s_mul_i32 s44, constStrideAL, s44                  // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s44 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s45 // sum tensor size
s_sub_u32 s44, s[sgprSizeI], 1                     // (size-1)
s_mul_hi_u32 s45, s[sgprStrideA0I], s44            // stride x (size-1)
s_mul_i32 s44, s[sgprStrideA0I], s44               // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s44 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s45 // sum tensor size
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s46 // sub tileStart
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s47 // sub tileStart
s_lshr_b64 s[sgprShadowLimitA:sgprShadowLimitA+1], s[sgprShadowLimitA:sgprShadowLimitA+1], 1 // Set limit to use bytes (multiple bpe)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 4 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_and_b32 s11, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s11, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_StridedBatchedGemmLoadA
s_mul_i32 s44, 8, s[sgprWorkGroup2]                // Compute Offset into Pointer Array
s_cmp_eq_u32 s[sgprSizesSum], 0x0                  // Don't dereference Pointer array if SizesSum == 0
s_cbranch_scc1 label_StridedBatchedGemmLoadA_End
s_add_u32 s44, s44, s[sgprAddressA+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s45, s[sgprAddressA+1], 0               // Offsetting to the location [Higher half of address]
s_load_b64 s[sgprSrdA:sgprSrdA+1], s[44:45], 0     // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for pointer-array SRD load before reusing base SGPR
s_load_b64 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 0xa0 // Load batchOffsetA from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s44        // Add batch offset to A address (low)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s45       // Add batch offset to A address (high)
s_sub_u32 s[sgprSrdA+0], s[sgprSrdA+0], 4          // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0         // pre-pad to make room for possible pointer shift
s_lshr_b64 s[46:47], s[46:47], 1                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdA+0], s46, s[sgprSrdA+0]        // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s47, s[sgprSrdA+1]       // SRD base = Address+ tileStart1
s_branch label_StridedBatchedGemmLoadA_End
label_StridedBatchedGemmLoadA:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s45, s[sgprStrideAK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s44, s[sgprStrideAK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s46, s46, s44                            // accum wg term to tilestart
s_addc_u32 s47, s47, s45                           // accum wg term to tilestart
s_lshr_b64 s[46:47], s[46:47], 1                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdA+0], s[sgprAddressA+0], s46    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprAddressA+1], s47   // SRD base = Address+ tileStart1
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
s_mul_i32 s44, s[sgprWorkGroup0], 96               // scaleA: workgroup row origin
s_mul_i32 s44, s44, s[sgprStrideScaleA]            // scaleA: * row stride
s_mul_i32 s44, s44, 2                              // scaleA: elements -> bytes
s_mul_i32 s45, s[sgprSizeI], s[sgprStrideScaleA]   // scaleA: SizeI * row stride
s_mul_i32 s45, s45, 2                              // scaleA: tensor bytes
s_sub_u32 s[sgprSrdScaleA+2], s45, s44             // scaleA: buffer limit from the workgroup origin
s_add_u32 s[sgprSrdScaleA+0], s[sgprAddressScaleA+0], s44 // scaleA: SRD base lo
s_addc_u32 s[sgprSrdScaleA+1], s[sgprAddressScaleA+1], 0 // scaleA: SRD base hi
s_mov_b32 s[sgprSrdScaleA+3], Srd127_96            // scaleA: set bits 127_96 in SRD
s_mov_b32 s[sgprScaleALiftMagic+0], 0x4b000000     // w4a16: 2^23, lsb of nibble 0 at 1.0
s_mov_b32 s[sgprScaleALiftMagic+1], 0x49000000     // w4a16: 2^19, lsb of nibble 1 at 1.0
s_mov_b32 s[sgprScaleALiftMagic+2], 0x47000000     // w4a16: 2^15, lsb of nibble 2 at 1.0
s_mov_b32 s[sgprScaleALiftMagic+3], 0x45000000     // w4a16: 2^11, lsb of nibble 3 at 1.0
s_mov_b32 s[sgprScaleALiftMagic+4], 0x43000000     // w4a16: 2^7, lsb of nibble 4 at 1.0
s_mov_b32 s[sgprScaleALiftMagic+5], 0x8000         // w4a16: bf16 rounding bias
s_mov_b32 s[sgprScaleALiftMagic+6], 0x4b000008     // w4a16: 2^23 + 8
s_mov_b32 s[sgprScaleALiftMagic+7], 0x49000080     // w4a16: 2^19 + 8
s_mov_b32 s[sgprScaleALiftMagic+8], 0x47000800     // w4a16: 2^15 + 8
s_mov_b32 s[sgprScaleALiftMagic+9], 0x45008000     // w4a16: 2^11 + 8
s_mov_b32 s[sgprScaleALiftMagic+10], 0x43080000    // w4a16: 2^7 + 8
s_and_b32 s[sgprScaleAKCnt], s[sgprStreamKLocalStart], s[sgprScaleAKMask] // scaleA: position of the first iteration in its group
s_mul_i32 s44, s[sgprStreamKLocalStart], 64        // scaleA: first K of this range
s_lshr_b32 s44, s44, s[sgprScaleAGShift]           // scaleA: groups before this range
s_mul_i32 s45, s44, 2                              // scaleA: skipped bytes
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], s45 // scaleA: skip groups before the split (lower)
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA: skip groups before the split (upper)
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], s45 // scaleA: limit shrinks by the skip

/* global read addresses: addresses b */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s47, s[sgprWorkGroup1], 192           // WorkGroup[01] * MT
s_mul_i32 s46, s[sgprWorkGroup1], 192              // WorkGroup[01] * MT
s_mul_hi_u32 s47, s46, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s46, s46, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_mul_i32 s44, s[sgprStreamKLocalStart], 64        // StreamK tile start offset
s_mul_hi_u32 s45, s44, constStrideBL               // StreamK tile start offset
s_mul_i32 s44, s44, constStrideBL                  // StreamK tile start offset
s_add_u32 s46, s46, s44                            // accum GsuOffset term to tilestart
s_addc_u32 s47, s47, s45                           // accum GsuOffset term to tilestart
s_mov_b64 s[sgprShadowLimitB+0:sgprShadowLimitB+0+1], 1 // Init tensor size
s_sub_u32 s44, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s45, constStrideBL, s44               // stride x (size-1)
s_mul_i32 s44, constStrideBL, s44                  // stride x (size-1)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s44 // sum tensor size
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s45 // sum tensor size
s_sub_u32 s44, s[sgprSizeJ], 1                     // (size-1)
s_mul_hi_u32 s45, s[sgprStrideB1J], s44            // stride x (size-1)
s_mul_i32 s44, s[sgprStrideB1J], s44               // stride x (size-1)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s44 // sum tensor size
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s45 // sum tensor size
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s46 // sub tileStart
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s47 // sub tileStart
s_lshl_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], 1 // Set limit to use bytes (multiple bpe)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_and_b32 s11, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s11, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_StridedBatchedGemmLoadB
s_mul_i32 s44, 8, s[sgprWorkGroup2]                // Compute Offset into Pointer Array
s_cmp_eq_u32 s[sgprSizesSum], 0x0                  // Don't dereference Pointer array if SizesSum == 0
s_cbranch_scc1 label_StridedBatchedGemmLoadB_End
s_add_u32 s44, s44, s[sgprAddressB+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s45, s[sgprAddressB+1], 0               // Offsetting to the location [Higher half of address]
s_load_b64 s[sgprSrdB:sgprSrdB+1], s[44:45], 0     // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for pointer-array SRD load before reusing base SGPR
s_load_b64 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 0xa8 // Load batchOffsetB from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s44        // Add batch offset to B address (low)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s45       // Add batch offset to B address (high)
s_sub_u32 s[sgprSrdB+0], s[sgprSrdB+0], 16         // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0         // pre-pad to make room for possible pointer shift
s_lshl_b64 s[46:47], s[46:47], 1                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdB+0], s46, s[sgprSrdB+0]        // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s47, s[sgprSrdB+1]       // SRD base = Address+ tileStart1
s_branch label_StridedBatchedGemmLoadB_End
label_StridedBatchedGemmLoadB:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s45, s[sgprStrideBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s44, s[sgprStrideBK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s46, s46, s44                            // accum wg term to tilestart
s_addc_u32 s47, s47, s45                           // accum wg term to tilestart
s_lshl_b64 s[46:47], s[46:47], 1                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdB+0], s[sgprAddressB+0], s46    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprAddressB+1], s47   // SRD base = Address+ tileStart1
label_StridedBatchedGemmLoadB_End:  /// End Computing the Batch Matrix's base address for Strided Batched
s_mov_b32 s[sgprSrdB+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: final offsets a */
/* ============================================================= */
v_mul_lo_u32 v17, s[sgprStrideA0I], v[6]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+0+0], vcc_lo, v[15], v[17+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+0+0], 0x8, v[vgprGlobalReadOffsetA+0+0] // add prepad for pointer shift
v_lshrrev_b32 v17, s[sgprScaleAGShift], v15        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+0], s[sgprStrideScaleA], v6 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+0], v17, v[vgprGlobalReadOffsetScaleA+0] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+0], 1, v[vgprGlobalReadOffsetScaleA+0] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetA+0], 1, v[vgprGlobalReadOffsetA+0] //  (multiple bpe)
v_mul_lo_u32 v17, s[sgprStrideA0I], v[7]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+1+0], vcc_lo, v[15], v[17+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+1+0], 0x8, v[vgprGlobalReadOffsetA+1+0] // add prepad for pointer shift
v_lshrrev_b32 v17, s[sgprScaleAGShift], v15        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+1], s[sgprStrideScaleA], v7 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+1], v17, v[vgprGlobalReadOffsetScaleA+1] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+1], 1, v[vgprGlobalReadOffsetScaleA+1] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetA+1], 1, v[vgprGlobalReadOffsetA+1] //  (multiple bpe)
v_mul_lo_u32 v17, s[sgprStrideA0I], v[8]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+2+0], vcc_lo, v[15], v[17+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+2+0], 0x8, v[vgprGlobalReadOffsetA+2+0] // add prepad for pointer shift
v_lshrrev_b32 v17, s[sgprScaleAGShift], v15        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+2], s[sgprStrideScaleA], v8 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+2], v17, v[vgprGlobalReadOffsetScaleA+2] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+2], 1, v[vgprGlobalReadOffsetScaleA+2] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetA+2], 1, v[vgprGlobalReadOffsetA+2] //  (multiple bpe)
/* ============================================================= */

/* global read addresses: final offsets b */
/* ============================================================= */
v_mul_lo_u32 v6, s[sgprStrideB1J], v[9]            // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+0+0], vcc_lo, v[16], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+0+0], 0x8, v[vgprGlobalReadOffsetB+0+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+0], 1, v[vgprGlobalReadOffsetB+0] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[10]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+1+0], vcc_lo, v[16], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+1+0], 0x8, v[vgprGlobalReadOffsetB+1+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+1], 1, v[vgprGlobalReadOffsetB+1] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[11]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+2+0], vcc_lo, v[16], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+2+0], 0x8, v[vgprGlobalReadOffsetB+2+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+2], 1, v[vgprGlobalReadOffsetB+2] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[12]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+3+0], vcc_lo, v[16], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+3+0], 0x8, v[vgprGlobalReadOffsetB+3+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+3], 1, v[vgprGlobalReadOffsetB+3] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[13]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+4+0], vcc_lo, v[16], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+4+0], 0x8, v[vgprGlobalReadOffsetB+4+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+4], 1, v[vgprGlobalReadOffsetB+4] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[14]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+5+0], vcc_lo, v[16], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+5+0], 0x8, v[vgprGlobalReadOffsetB+5+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+5], 1, v[vgprGlobalReadOffsetB+5] //  (multiple bpe)
/* ============================================================= */

/* global read addresses: increments a */
.set GlobalReadIncsA, 32

/* global read addresses: increments b */
.set GlobalReadIncsB, 128
/* declare loop num iterations */

/* initC: remove ValuC vgpr buffer [0...72) from pool */

/* initC: remove acc vgpr buffer [0...0) from pool */

/* initC: remove ValuA/B vgpr buffer [90...187) from pool */
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

/* after InitC, skip to end of prefetch last iter if numIter==0 */
s_cbranch_scc0 label_NoBranch_6                    // Only branch on scc1
s_getpc_b64 s[44:45]                               // addr of next instr
s_add_i32 s46, label_PrefetchGlobalLastIterEnd, 4  // target branch offset
s_add_u32 s44, s44, s46                            // add target branch offset
s_addc_u32 s45, s45, 0                             // add high and carry
s_setpc_b64 s[44:45]                               // branch to label_PrefetchGlobalLastIterEnd
label_NoBranch_6:
buffer_load_b32 v[vgprG2LA+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b32 v[vgprG2LA+6], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b32 v[vgprG2LA+10], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_2_0

/* global read block-scale A */
buffer_load_d16_hi_b16 v[vgprG2LScaleA+0], v[vgprGlobalReadOffsetScaleA+0], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 0 as f32
buffer_load_d16_hi_b16 v[vgprG2LScaleA+1], v[vgprGlobalReadOffsetScaleA+1], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 1 as f32
buffer_load_d16_hi_b16 v[vgprG2LScaleA+2], v[vgprGlobalReadOffsetScaleA+2], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 2 as f32
buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b128 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b128 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b128 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0
buffer_load_b128 v[vgprG2LB+16:vgprG2LB+16+3], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_4_0
buffer_load_b128 v[vgprG2LB+20:vgprG2LB+20+3], v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_5_0

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
s_waitcnt vmcnt(0)                                 // wait for global read
s_barrier                                          // For stream-k / persistent loop

/* local write a */
v_mov_b32 v226, v[vgprG2LScaleA+0]                 // scaleA: copy on the other VGPR bank
v_mov_b32 v227, v[vgprG2LA+2+0]                    // w4a16: save packed int4 dword 0
v_and_or_b32 v228, v227, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v229, v227, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v231, v227, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v230, v227, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v228, s[sgprScaleALiftMagic+6], v228 :: v_dual_subrev_f32 v229, s[sgprScaleALiftMagic+7], v229 // w4a16: n - c, exact
v_dual_subrev_f32 v231, s[sgprScaleALiftMagic+8], v231 :: v_dual_subrev_f32 v230, s[sgprScaleALiftMagic+9], v230 // w4a16: n - c, exact
v_dual_mul_f32 v228, v84, v228 :: v_dual_mul_f32 v229, v226, v229 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v231, v84, v231 :: v_dual_add_nc_u32 v228, s[sgprScaleALiftMagic+5], v228 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v230, v226, v230 :: v_dual_add_nc_u32 v229, s[sgprScaleALiftMagic+5], v229 // w4a16: (n - c) * s :: round
v_and_or_b32 v232, v227, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v227, 20, v227                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v233, v227, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v235, v227, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v234, v227, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v232, s[sgprScaleALiftMagic+10], v232 :: v_dual_add_nc_u32 v231, s[sgprScaleALiftMagic+5], v231 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v233, s[sgprScaleALiftMagic+6], v233 :: v_dual_add_nc_u32 v230, s[sgprScaleALiftMagic+5], v230 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v235, s[sgprScaleALiftMagic+7], v235 :: v_dual_subrev_f32 v234, s[sgprScaleALiftMagic+8], v234 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+0+0], v229, v228, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+1], v230, v231, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v232, v84, v232 :: v_dual_mul_f32 v233, v226, v233 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v235, v84, v235 :: v_dual_add_nc_u32 v232, s[sgprScaleALiftMagic+5], v232 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v234, v226, v234 :: v_dual_add_nc_u32 v233, s[sgprScaleALiftMagic+5], v233 // w4a16: (n - c) * s :: round
v_add_nc_u32 v235, s[sgprScaleALiftMagic+5], v235  // w4a16: round
v_add_nc_u32 v234, s[sgprScaleALiftMagic+5], v234  // w4a16: round
v_perm_b32 v[vgprG2LA+0+2], v233, v232, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+3], v234, v235, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
v_mov_b32 v226, v[vgprG2LScaleA+1]                 // scaleA: copy on the other VGPR bank
v_mov_b32 v227, v[vgprG2LA+6+0]                    // w4a16: save packed int4 dword 0
v_and_or_b32 v228, v227, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v229, v227, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v231, v227, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v230, v227, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v228, s[sgprScaleALiftMagic+6], v228 :: v_dual_subrev_f32 v229, s[sgprScaleALiftMagic+7], v229 // w4a16: n - c, exact
v_dual_subrev_f32 v231, s[sgprScaleALiftMagic+8], v231 :: v_dual_subrev_f32 v230, s[sgprScaleALiftMagic+9], v230 // w4a16: n - c, exact
v_dual_mul_f32 v228, v85, v228 :: v_dual_mul_f32 v229, v226, v229 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v231, v85, v231 :: v_dual_add_nc_u32 v228, s[sgprScaleALiftMagic+5], v228 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v230, v226, v230 :: v_dual_add_nc_u32 v229, s[sgprScaleALiftMagic+5], v229 // w4a16: (n - c) * s :: round
v_and_or_b32 v232, v227, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v227, 20, v227                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v233, v227, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v235, v227, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v234, v227, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v232, s[sgprScaleALiftMagic+10], v232 :: v_dual_add_nc_u32 v231, s[sgprScaleALiftMagic+5], v231 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v233, s[sgprScaleALiftMagic+6], v233 :: v_dual_add_nc_u32 v230, s[sgprScaleALiftMagic+5], v230 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v235, s[sgprScaleALiftMagic+7], v235 :: v_dual_subrev_f32 v234, s[sgprScaleALiftMagic+8], v234 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+4+0], v229, v228, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+4+1], v230, v231, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v232, v85, v232 :: v_dual_mul_f32 v233, v226, v233 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v235, v85, v235 :: v_dual_add_nc_u32 v232, s[sgprScaleALiftMagic+5], v232 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v234, v226, v234 :: v_dual_add_nc_u32 v233, s[sgprScaleALiftMagic+5], v233 // w4a16: (n - c) * s :: round
v_add_nc_u32 v235, s[sgprScaleALiftMagic+5], v235  // w4a16: round
v_add_nc_u32 v234, s[sgprScaleALiftMagic+5], v234  // w4a16: round
v_perm_b32 v[vgprG2LA+4+2], v233, v232, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+4+3], v234, v235, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:1152 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 1152 sync LDS0
v_mov_b32 v227, v[vgprG2LScaleA+2]                 // scaleA: copy on the other VGPR bank
v_mov_b32 v226, v[vgprG2LA+10+0]                   // w4a16: save packed int4 dword 0
v_and_or_b32 v228, v226, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v229, v226, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v231, v226, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v230, v226, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v228, s[sgprScaleALiftMagic+6], v228 :: v_dual_subrev_f32 v229, s[sgprScaleALiftMagic+7], v229 // w4a16: n - c, exact
v_dual_subrev_f32 v231, s[sgprScaleALiftMagic+8], v231 :: v_dual_subrev_f32 v230, s[sgprScaleALiftMagic+9], v230 // w4a16: n - c, exact
v_dual_mul_f32 v228, v86, v228 :: v_dual_mul_f32 v229, v227, v229 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v231, v86, v231 :: v_dual_add_nc_u32 v228, s[sgprScaleALiftMagic+5], v228 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v230, v227, v230 :: v_dual_add_nc_u32 v229, s[sgprScaleALiftMagic+5], v229 // w4a16: (n - c) * s :: round
v_and_or_b32 v232, v226, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v226, 20, v226                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v233, v226, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v235, v226, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v234, v226, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v232, s[sgprScaleALiftMagic+10], v232 :: v_dual_add_nc_u32 v231, s[sgprScaleALiftMagic+5], v231 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v233, s[sgprScaleALiftMagic+6], v233 :: v_dual_add_nc_u32 v230, s[sgprScaleALiftMagic+5], v230 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v235, s[sgprScaleALiftMagic+7], v235 :: v_dual_subrev_f32 v234, s[sgprScaleALiftMagic+8], v234 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+8+0], v229, v228, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+1], v230, v231, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v232, v86, v232 :: v_dual_mul_f32 v233, v227, v233 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v235, v86, v235 :: v_dual_add_nc_u32 v232, s[sgprScaleALiftMagic+5], v232 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v234, v227, v234 :: v_dual_add_nc_u32 v233, s[sgprScaleALiftMagic+5], v233 // w4a16: (n - c) * s :: round
v_add_nc_u32 v235, s[sgprScaleALiftMagic+5], v235  // w4a16: round
v_add_nc_u32 v234, s[sgprScaleALiftMagic+5], v234  // w4a16: round
v_perm_b32 v[vgprG2LA+8+2], v233, v232, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+3], v234, v235, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:2304 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 2304 sync LDS0

/* local write b */
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4:vgprG2LB+4+3] offset:4608 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 4608 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8:vgprG2LB+8+3] offset:9216 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 9216 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12:vgprG2LB+12+3] offset:13824 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 13824 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+16:vgprG2LB+16+3] offset:18432 // lwoB_0_0_4_0 = (0*LSCB)*(MT1J+PAD) + (4*LSPB) = 18432 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+20:vgprG2LB+20+3] offset:23040 // lwoB_0_0_5_0 = (0*LSCB)*(MT1J+PAD) + (5*LSPB) = 23040 sync LDS0

/* local write swap a */

/* local write swap b */
s_cmp_eq_u32 s[sgprLoopCounterL], 0x1              // PGR=2 but only 1 loop
s_cbranch_scc1 label_skipPGR2_1                    // PGR=2 but only 1 loop
buffer_load_b32 v[vgprG2LA+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b32 v[vgprG2LA+6], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b32 v[vgprG2LA+10], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_2_0

/* global read block-scale A */
buffer_load_d16_hi_b16 v[vgprG2LScaleA+0], v[vgprGlobalReadOffsetScaleA+0], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 0 as f32
buffer_load_d16_hi_b16 v[vgprG2LScaleA+1], v[vgprGlobalReadOffsetScaleA+1], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 1 as f32
buffer_load_d16_hi_b16 v[vgprG2LScaleA+2], v[vgprGlobalReadOffsetScaleA+2], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 2 as f32
buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b128 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b128 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b128 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0
buffer_load_b128 v[vgprG2LB+16:vgprG2LB+16+3], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_4_0
buffer_load_b128 v[vgprG2LB+20:vgprG2LB+20+3], v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_5_0
s_branch label_skipPGR2_2                          // jump to PGR=2 label
label_skipPGR2_1:
label_skipPGR2_2:
s_waitcnt lgkmcnt(0)                               // 0prefetch wait for local write
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier                                          // LW to PLR, sync

/* local read prefetch a */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4608 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4624 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA+0] offset:9216 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA+0] offset:9232 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read prefetch b */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:9216 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:9232 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:18432 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:18448 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

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

/* iter 0 */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:22, lwEndMfmaIndex:24  */
/*  numMfmaForLR:8, syncPlrMfmaIndex:27 , sync1LdsMfmaIndex:21 */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=4 for iteration == 0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:1  */
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], GlobalReadIncsA // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], GlobalReadIncsA // limit -= inc)
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:2  */
ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:3  */
ds_load_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA+0] offset:4640 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA+0] offset:4656 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0

/* global read inc block-scale A (2 bytes every 1 iters) */
s_add_u32 s[sgprScaleAKCnt], s[sgprScaleAKCnt], 1  // scaleA: one more K iteration done
s_and_b32 s11, s[sgprScaleAKCnt], s[sgprScaleAKMask] // scaleA: SCC = counter has not wrapped
s_cselect_b32 s11, 0, s[sgprScaleAInc]             // scaleA: advance only on a wrap
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:4  */
ds_load_b128 v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], v[vgprLocalReadAddrA+0] offset:9248 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+20:vgprValuA_X1_I0+20+3], v[vgprLocalReadAddrA+0] offset:9264 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], s11 // scaleA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], s11 // scaleA limit -= inc
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:5  */
ds_load_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB+0] offset:9248 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], GlobalReadIncsB // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], GlobalReadIncsB // limit -= inc)
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:6  */
ds_load_b128 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[vgprLocalReadAddrB+0] offset:9264 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32
v_wmma_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:7  */
ds_load_b128 v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprLocalReadAddrB+0] offset:18464 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/*  mfmaIndex:8  */
ds_load_b128 v[vgprValuB_X1_I0+20:vgprValuB_X1_I0+20+3], v[vgprLocalReadAddrB+0] offset:18480 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
/* localReadsVacancy: latencyLeft 1 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=6 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=6 */

/* iter 1 */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:22, lwEndMfmaIndex:24  */
/*  numMfmaForLR:8, syncPlrMfmaIndex:27 , sync1LdsMfmaIndex:21 */
/*  mfmaIndex:9  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // wait for prior local read local write old=0, new=2 newLW=0 newLR=2
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:10  */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:11  */
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4672 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4688 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:12  */
ds_load_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA+0] offset:9280 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:13  */
ds_load_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA+0] offset:9296 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:14  */
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:9280 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:15  */
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:9296 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X1_I0+16+0+0:vgprValuB_X1_I0+16+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:16  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:18496 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X1_I0+16+0+0:vgprValuB_X1_I0+16+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/*  mfmaIndex:17  */
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:18512 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
/* localReadsVacancy: latencyLeft 1 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X1_I0+16+0+0:vgprValuB_X1_I0+16+0+0+7], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=6 */
/* dataAtIterB=0 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=6 */

/* iter 2 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:22, lwEndMfmaIndex:24  */
/*  numMfmaForLR:8, syncPlrMfmaIndex:27 , sync1LdsMfmaIndex:21 */
/*  mfmaIndex:18  */
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // wait for prior local read local write old=0, new=2 newLW=0 newLR=2
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:19  */
ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:20  */
ds_load_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA+0] offset:4704 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA+0] offset:4720 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:21  */
/* schedule remaining localreads for one buffer scheduling */
ds_load_b128 v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], v[vgprLocalReadAddrA+0] offset:9312 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+20:vgprValuA_X1_I0+20+3], v[vgprLocalReadAddrA+0] offset:9328 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB+0] offset:9312 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[vgprLocalReadAddrB+0] offset:9328 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprLocalReadAddrB+0] offset:18528 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+20:vgprValuB_X1_I0+20+3], v[vgprLocalReadAddrB+0] offset:18544 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:22  */
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(8)                                 // wait for global read before writing to local
v_mov_b32 v226, v[vgprG2LScaleA+0]                 // scaleA: copy on the other VGPR bank
v_mov_b32 v227, v[vgprG2LA+2+0]                    // w4a16: save packed int4 dword 0
v_and_or_b32 v228, v227, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v229, v227, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v231, v227, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v230, v227, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v228, s[sgprScaleALiftMagic+6], v228 :: v_dual_subrev_f32 v229, s[sgprScaleALiftMagic+7], v229 // w4a16: n - c, exact
v_dual_subrev_f32 v231, s[sgprScaleALiftMagic+8], v231 :: v_dual_subrev_f32 v230, s[sgprScaleALiftMagic+9], v230 // w4a16: n - c, exact
v_dual_mul_f32 v228, v84, v228 :: v_dual_mul_f32 v229, v226, v229 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v231, v84, v231 :: v_dual_add_nc_u32 v228, s[sgprScaleALiftMagic+5], v228 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v230, v226, v230 :: v_dual_add_nc_u32 v229, s[sgprScaleALiftMagic+5], v229 // w4a16: (n - c) * s :: round
v_and_or_b32 v232, v227, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v227, 20, v227                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v233, v227, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v235, v227, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v234, v227, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v232, s[sgprScaleALiftMagic+10], v232 :: v_dual_add_nc_u32 v231, s[sgprScaleALiftMagic+5], v231 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v233, s[sgprScaleALiftMagic+6], v233 :: v_dual_add_nc_u32 v230, s[sgprScaleALiftMagic+5], v230 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v235, s[sgprScaleALiftMagic+7], v235 :: v_dual_subrev_f32 v234, s[sgprScaleALiftMagic+8], v234 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+0+0], v229, v228, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+1], v230, v231, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v232, v84, v232 :: v_dual_mul_f32 v233, v226, v233 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v235, v84, v235 :: v_dual_add_nc_u32 v232, s[sgprScaleALiftMagic+5], v232 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v234, v226, v234 :: v_dual_add_nc_u32 v233, s[sgprScaleALiftMagic+5], v233 // w4a16: (n - c) * s :: round
v_add_nc_u32 v235, s[sgprScaleALiftMagic+5], v235  // w4a16: round
v_add_nc_u32 v234, s[sgprScaleALiftMagic+5], v234  // w4a16: round
v_perm_b32 v[vgprG2LA+0+2], v233, v232, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+3], v234, v235, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
buffer_load_b32 v[vgprG2LA+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(8)                                 // wait for global read before writing to local
v_mov_b32 v226, v[vgprG2LScaleA+1]                 // scaleA: copy on the other VGPR bank
v_mov_b32 v227, v[vgprG2LA+6+0]                    // w4a16: save packed int4 dword 0
v_and_or_b32 v228, v227, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v229, v227, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v231, v227, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v230, v227, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v228, s[sgprScaleALiftMagic+6], v228 :: v_dual_subrev_f32 v229, s[sgprScaleALiftMagic+7], v229 // w4a16: n - c, exact
v_dual_subrev_f32 v231, s[sgprScaleALiftMagic+8], v231 :: v_dual_subrev_f32 v230, s[sgprScaleALiftMagic+9], v230 // w4a16: n - c, exact
v_dual_mul_f32 v228, v85, v228 :: v_dual_mul_f32 v229, v226, v229 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v231, v85, v231 :: v_dual_add_nc_u32 v228, s[sgprScaleALiftMagic+5], v228 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v230, v226, v230 :: v_dual_add_nc_u32 v229, s[sgprScaleALiftMagic+5], v229 // w4a16: (n - c) * s :: round
v_and_or_b32 v232, v227, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v227, 20, v227                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v233, v227, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v235, v227, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v234, v227, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v232, s[sgprScaleALiftMagic+10], v232 :: v_dual_add_nc_u32 v231, s[sgprScaleALiftMagic+5], v231 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v233, s[sgprScaleALiftMagic+6], v233 :: v_dual_add_nc_u32 v230, s[sgprScaleALiftMagic+5], v230 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v235, s[sgprScaleALiftMagic+7], v235 :: v_dual_subrev_f32 v234, s[sgprScaleALiftMagic+8], v234 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+4+0], v229, v228, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+4+1], v230, v231, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v232, v85, v232 :: v_dual_mul_f32 v233, v226, v233 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v235, v85, v235 :: v_dual_add_nc_u32 v232, s[sgprScaleALiftMagic+5], v232 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v234, v226, v234 :: v_dual_add_nc_u32 v233, s[sgprScaleALiftMagic+5], v233 // w4a16: (n - c) * s :: round
v_add_nc_u32 v235, s[sgprScaleALiftMagic+5], v235  // w4a16: round
v_add_nc_u32 v234, s[sgprScaleALiftMagic+5], v234  // w4a16: round
v_perm_b32 v[vgprG2LA+4+2], v233, v232, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+4+3], v234, v235, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:1152 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 1152 sync LDS0
buffer_load_b32 v[vgprG2LA+6], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_1_0
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(8)                                 // wait for global read before writing to local
v_mov_b32 v227, v[vgprG2LScaleA+2]                 // scaleA: copy on the other VGPR bank
v_mov_b32 v226, v[vgprG2LA+10+0]                   // w4a16: save packed int4 dword 0
v_and_or_b32 v228, v226, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v229, v226, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v231, v226, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v230, v226, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v228, s[sgprScaleALiftMagic+6], v228 :: v_dual_subrev_f32 v229, s[sgprScaleALiftMagic+7], v229 // w4a16: n - c, exact
v_dual_subrev_f32 v231, s[sgprScaleALiftMagic+8], v231 :: v_dual_subrev_f32 v230, s[sgprScaleALiftMagic+9], v230 // w4a16: n - c, exact
v_dual_mul_f32 v228, v86, v228 :: v_dual_mul_f32 v229, v227, v229 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v231, v86, v231 :: v_dual_add_nc_u32 v228, s[sgprScaleALiftMagic+5], v228 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v230, v227, v230 :: v_dual_add_nc_u32 v229, s[sgprScaleALiftMagic+5], v229 // w4a16: (n - c) * s :: round
v_and_or_b32 v232, v226, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v226, 20, v226                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v233, v226, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v235, v226, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v234, v226, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v232, s[sgprScaleALiftMagic+10], v232 :: v_dual_add_nc_u32 v231, s[sgprScaleALiftMagic+5], v231 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v233, s[sgprScaleALiftMagic+6], v233 :: v_dual_add_nc_u32 v230, s[sgprScaleALiftMagic+5], v230 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v235, s[sgprScaleALiftMagic+7], v235 :: v_dual_subrev_f32 v234, s[sgprScaleALiftMagic+8], v234 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+8+0], v229, v228, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+1], v230, v231, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v232, v86, v232 :: v_dual_mul_f32 v233, v227, v233 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v235, v86, v235 :: v_dual_add_nc_u32 v232, s[sgprScaleALiftMagic+5], v232 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v234, v227, v234 :: v_dual_add_nc_u32 v233, s[sgprScaleALiftMagic+5], v233 // w4a16: (n - c) * s :: round
v_add_nc_u32 v235, s[sgprScaleALiftMagic+5], v235  // w4a16: round
v_add_nc_u32 v234, s[sgprScaleALiftMagic+5], v234  // w4a16: round
v_perm_b32 v[vgprG2LA+8+2], v233, v232, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+3], v234, v235, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:2304 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 2304 sync LDS0
buffer_load_b32 v[vgprG2LA+10], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_2_0
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(8)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0

/* global read block-scale A */
buffer_load_d16_hi_b16 v[vgprG2LScaleA+0], v[vgprGlobalReadOffsetScaleA+0], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 0 as f32
buffer_load_d16_hi_b16 v[vgprG2LScaleA+1], v[vgprGlobalReadOffsetScaleA+1], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 1 as f32
buffer_load_d16_hi_b16 v[vgprG2LScaleA+2], v[vgprGlobalReadOffsetScaleA+2], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 2 as f32
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:23  */
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(10)                                // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4:vgprG2LB+4+3] offset:4608 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 4608 sync LDS0
buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(10)                                // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8:vgprG2LB+8+3] offset:9216 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 9216 sync LDS0
buffer_load_b128 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(10)                                // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12:vgprG2LB+12+3] offset:13824 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 13824 sync LDS0
buffer_load_b128 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(10)                                // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+16:vgprG2LB+16+3] offset:18432 // lwoB_0_0_4_0 = (0*LSCB)*(MT1J+PAD) + (4*LSPB) = 18432 sync LDS0
buffer_load_b128 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:24  */
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(10)                                // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+20:vgprG2LB+20+3] offset:23040 // lwoB_0_0_5_0 = (0*LSCB)*(MT1J+PAD) + (5*LSPB) = 23040 sync LDS0
buffer_load_b128 v[vgprG2LB+16:vgprG2LB+16+3], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_4_0
buffer_load_b128 v[vgprG2LB+20:vgprG2LB+20+3], v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_5_0

/* local write swap offsets a */

/* local write swap offsets b */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:25  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/*  mfmaIndex:26  */

/* local read swap offsets a */

/* local read swap offsets b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=3 skipReadsIterA=1 readsPerIterA=6 */
/* dataAtIterB=1 numReadsIterB=3 skipReadsIterB=1 readsPerIterB=6 */

/* iter 3 */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:22, lwEndMfmaIndex:24  */
/*  numMfmaForLR:8, syncPlrMfmaIndex:27 , sync1LdsMfmaIndex:21 */
/*  mfmaIndex:27  */
s_waitcnt lgkmcnt(0)                               // 3wait for local write
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier                                          // PGR, and wait until LW done to sync
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:28  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:29  */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:30  */
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4608 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4624 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:31  */
ds_load_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA+0] offset:9216 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA+0] offset:9232 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:32  */
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:9216 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:33  */
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:9232 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X1_I0+16+0+0:vgprValuB_X1_I0+16+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:34  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:18432 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X1_I0+16+0+0:vgprValuB_X1_I0+16+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/*  mfmaIndex:35  */
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:18448 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X1_I0+16+0+0:vgprValuB_X1_I0+16+0+0+7], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]

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

/* iter 0 */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:22, lwEndMfmaIndex:24  */
/*  numMfmaForLR:8, syncPlrMfmaIndex:27 , sync1LdsMfmaIndex:21 */
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
s_and_b32 s11, s[sgprScaleAKCnt], s[sgprScaleAKMask] // scaleA: SCC = counter has not wrapped
s_cselect_b32 s11, 0, s[sgprScaleAInc]             // scaleA: advance only on a wrap
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], s11 // scaleA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], s11 // scaleA limit -= inc

/* Global Read IncB */

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], GlobalReadIncsB // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], GlobalReadIncsB // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=4 for iteration == 0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:1  */
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:2  */
ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:3  */
ds_load_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA+0] offset:4640 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA+0] offset:4656 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:4  */
ds_load_b128 v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], v[vgprLocalReadAddrA+0] offset:9248 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+20:vgprValuA_X1_I0+20+3], v[vgprLocalReadAddrA+0] offset:9264 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:5  */
ds_load_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB+0] offset:9248 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:6  */
ds_load_b128 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[vgprLocalReadAddrB+0] offset:9264 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:7  */
ds_load_b128 v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprLocalReadAddrB+0] offset:18464 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/*  mfmaIndex:8  */
ds_load_b128 v[vgprValuB_X1_I0+20:vgprValuB_X1_I0+20+3], v[vgprLocalReadAddrB+0] offset:18480 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
/* localReadsVacancy: latencyLeft 1 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=6 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=6 */

/* iter 1 */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:22, lwEndMfmaIndex:24  */
/*  numMfmaForLR:8, syncPlrMfmaIndex:27 , sync1LdsMfmaIndex:21 */
/*  mfmaIndex:9  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // wait for prior local read local write old=0, new=2 newLW=0 newLR=2
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:10  */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:11  */
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4672 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4688 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:12  */
ds_load_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA+0] offset:9280 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:13  */
ds_load_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA+0] offset:9296 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:14  */
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:9280 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:15  */
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:9296 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X1_I0+16+0+0:vgprValuB_X1_I0+16+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:16  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:18496 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X1_I0+16+0+0:vgprValuB_X1_I0+16+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/*  mfmaIndex:17  */
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:18512 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
/* localReadsVacancy: latencyLeft 1 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X1_I0+16+0+0:vgprValuB_X1_I0+16+0+0+7], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=6 */
/* dataAtIterB=0 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=6 */

/* iter 2 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:22, lwEndMfmaIndex:24  */
/*  numMfmaForLR:8, syncPlrMfmaIndex:27 , sync1LdsMfmaIndex:21 */
/*  mfmaIndex:18  */
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // wait for prior local read local write old=0, new=2 newLW=0 newLR=2
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:19  */
ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:20  */
ds_load_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA+0] offset:4704 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA+0] offset:4720 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:21  */
/* schedule remaining localreads for one buffer scheduling */
ds_load_b128 v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], v[vgprLocalReadAddrA+0] offset:9312 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+20:vgprValuA_X1_I0+20+3], v[vgprLocalReadAddrA+0] offset:9328 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB+0] offset:9312 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[vgprLocalReadAddrB+0] offset:9328 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprLocalReadAddrB+0] offset:18528 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+20:vgprValuB_X1_I0+20+3], v[vgprLocalReadAddrB+0] offset:18544 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:22  */
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(8)                                 // wait for global read before writing to local
v_mov_b32 v226, v[vgprG2LScaleA+0]                 // scaleA: copy on the other VGPR bank
v_mov_b32 v227, v[vgprG2LA+2+0]                    // w4a16: save packed int4 dword 0
v_and_or_b32 v228, v227, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v229, v227, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v231, v227, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v230, v227, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v228, s[sgprScaleALiftMagic+6], v228 :: v_dual_subrev_f32 v229, s[sgprScaleALiftMagic+7], v229 // w4a16: n - c, exact
v_dual_subrev_f32 v231, s[sgprScaleALiftMagic+8], v231 :: v_dual_subrev_f32 v230, s[sgprScaleALiftMagic+9], v230 // w4a16: n - c, exact
v_dual_mul_f32 v228, v84, v228 :: v_dual_mul_f32 v229, v226, v229 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v231, v84, v231 :: v_dual_add_nc_u32 v228, s[sgprScaleALiftMagic+5], v228 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v230, v226, v230 :: v_dual_add_nc_u32 v229, s[sgprScaleALiftMagic+5], v229 // w4a16: (n - c) * s :: round
v_and_or_b32 v232, v227, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v227, 20, v227                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v233, v227, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v235, v227, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v234, v227, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v232, s[sgprScaleALiftMagic+10], v232 :: v_dual_add_nc_u32 v231, s[sgprScaleALiftMagic+5], v231 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v233, s[sgprScaleALiftMagic+6], v233 :: v_dual_add_nc_u32 v230, s[sgprScaleALiftMagic+5], v230 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v235, s[sgprScaleALiftMagic+7], v235 :: v_dual_subrev_f32 v234, s[sgprScaleALiftMagic+8], v234 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+0+0], v229, v228, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+1], v230, v231, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v232, v84, v232 :: v_dual_mul_f32 v233, v226, v233 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v235, v84, v235 :: v_dual_add_nc_u32 v232, s[sgprScaleALiftMagic+5], v232 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v234, v226, v234 :: v_dual_add_nc_u32 v233, s[sgprScaleALiftMagic+5], v233 // w4a16: (n - c) * s :: round
v_add_nc_u32 v235, s[sgprScaleALiftMagic+5], v235  // w4a16: round
v_add_nc_u32 v234, s[sgprScaleALiftMagic+5], v234  // w4a16: round
v_perm_b32 v[vgprG2LA+0+2], v233, v232, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+3], v234, v235, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
v_mov_b32 v226, v[vgprG2LScaleA+1]                 // scaleA: copy on the other VGPR bank
v_mov_b32 v227, v[vgprG2LA+6+0]                    // w4a16: save packed int4 dword 0
v_and_or_b32 v228, v227, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v229, v227, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v231, v227, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v230, v227, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v228, s[sgprScaleALiftMagic+6], v228 :: v_dual_subrev_f32 v229, s[sgprScaleALiftMagic+7], v229 // w4a16: n - c, exact
v_dual_subrev_f32 v231, s[sgprScaleALiftMagic+8], v231 :: v_dual_subrev_f32 v230, s[sgprScaleALiftMagic+9], v230 // w4a16: n - c, exact
v_dual_mul_f32 v228, v85, v228 :: v_dual_mul_f32 v229, v226, v229 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v231, v85, v231 :: v_dual_add_nc_u32 v228, s[sgprScaleALiftMagic+5], v228 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v230, v226, v230 :: v_dual_add_nc_u32 v229, s[sgprScaleALiftMagic+5], v229 // w4a16: (n - c) * s :: round
v_and_or_b32 v232, v227, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v227, 20, v227                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v233, v227, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v235, v227, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v234, v227, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v232, s[sgprScaleALiftMagic+10], v232 :: v_dual_add_nc_u32 v231, s[sgprScaleALiftMagic+5], v231 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v233, s[sgprScaleALiftMagic+6], v233 :: v_dual_add_nc_u32 v230, s[sgprScaleALiftMagic+5], v230 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v235, s[sgprScaleALiftMagic+7], v235 :: v_dual_subrev_f32 v234, s[sgprScaleALiftMagic+8], v234 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+4+0], v229, v228, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+4+1], v230, v231, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v232, v85, v232 :: v_dual_mul_f32 v233, v226, v233 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v235, v85, v235 :: v_dual_add_nc_u32 v232, s[sgprScaleALiftMagic+5], v232 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v234, v226, v234 :: v_dual_add_nc_u32 v233, s[sgprScaleALiftMagic+5], v233 // w4a16: (n - c) * s :: round
v_add_nc_u32 v235, s[sgprScaleALiftMagic+5], v235  // w4a16: round
v_add_nc_u32 v234, s[sgprScaleALiftMagic+5], v234  // w4a16: round
v_perm_b32 v[vgprG2LA+4+2], v233, v232, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+4+3], v234, v235, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:1152 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 1152 sync LDS0
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(6)                                 // wait for global read before writing to local
v_mov_b32 v227, v[vgprG2LScaleA+2]                 // scaleA: copy on the other VGPR bank
v_mov_b32 v226, v[vgprG2LA+10+0]                   // w4a16: save packed int4 dword 0
v_and_or_b32 v228, v226, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v229, v226, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v231, v226, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v230, v226, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v228, s[sgprScaleALiftMagic+6], v228 :: v_dual_subrev_f32 v229, s[sgprScaleALiftMagic+7], v229 // w4a16: n - c, exact
v_dual_subrev_f32 v231, s[sgprScaleALiftMagic+8], v231 :: v_dual_subrev_f32 v230, s[sgprScaleALiftMagic+9], v230 // w4a16: n - c, exact
v_dual_mul_f32 v228, v86, v228 :: v_dual_mul_f32 v229, v227, v229 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v231, v86, v231 :: v_dual_add_nc_u32 v228, s[sgprScaleALiftMagic+5], v228 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v230, v227, v230 :: v_dual_add_nc_u32 v229, s[sgprScaleALiftMagic+5], v229 // w4a16: (n - c) * s :: round
v_and_or_b32 v232, v226, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v226, 20, v226                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v233, v226, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v235, v226, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v234, v226, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v232, s[sgprScaleALiftMagic+10], v232 :: v_dual_add_nc_u32 v231, s[sgprScaleALiftMagic+5], v231 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v233, s[sgprScaleALiftMagic+6], v233 :: v_dual_add_nc_u32 v230, s[sgprScaleALiftMagic+5], v230 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v235, s[sgprScaleALiftMagic+7], v235 :: v_dual_subrev_f32 v234, s[sgprScaleALiftMagic+8], v234 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+8+0], v229, v228, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+1], v230, v231, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v232, v86, v232 :: v_dual_mul_f32 v233, v227, v233 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v235, v86, v235 :: v_dual_add_nc_u32 v232, s[sgprScaleALiftMagic+5], v232 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v234, v227, v234 :: v_dual_add_nc_u32 v233, s[sgprScaleALiftMagic+5], v233 // w4a16: (n - c) * s :: round
v_add_nc_u32 v235, s[sgprScaleALiftMagic+5], v235  // w4a16: round
v_add_nc_u32 v234, s[sgprScaleALiftMagic+5], v234  // w4a16: round
v_perm_b32 v[vgprG2LA+8+2], v233, v232, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+3], v234, v235, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:2304 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 2304 sync LDS0
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(5)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:23  */
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(4)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4:vgprG2LB+4+3] offset:4608 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 4608 sync LDS0
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(3)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8:vgprG2LB+8+3] offset:9216 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 9216 sync LDS0
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(2)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12:vgprG2LB+12+3] offset:13824 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 13824 sync LDS0
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(1)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+16:vgprG2LB+16+3] offset:18432 // lwoB_0_0_4_0 = (0*LSCB)*(MT1J+PAD) + (4*LSPB) = 18432 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:24  */
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(0)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+20:vgprG2LB+20+3] offset:23040 // lwoB_0_0_5_0 = (0*LSCB)*(MT1J+PAD) + (5*LSPB) = 23040 sync LDS0

/* local write swap offsets a */

/* local write swap offsets b */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:25  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/*  mfmaIndex:26  */

/* local read swap offsets a */

/* local read swap offsets b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=3 skipReadsIterA=1 readsPerIterA=6 */
/* dataAtIterB=1 numReadsIterB=3 skipReadsIterB=1 readsPerIterB=6 */

/* iter 3 */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:22, lwEndMfmaIndex:24  */
/*  numMfmaForLR:8, syncPlrMfmaIndex:27 , sync1LdsMfmaIndex:21 */
/*  mfmaIndex:27  */
s_waitcnt lgkmcnt(0)                               // 3wait for local write
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier                                          // noLoadLoop sync
s_waitcnt lgkmcnt(9)                               // wait for prior local read local write old=0, new=9 newLW=9 newLR=0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:28  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:29  */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:30  */
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4608 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4624 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:31  */
ds_load_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA+0] offset:9216 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA+0] offset:9232 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:32  */
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:9216 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:33  */
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:9232 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X1_I0+16+0+0:vgprValuB_X1_I0+16+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:34  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:18432 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X1_I0+16+0+0:vgprValuB_X1_I0+16+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/*  mfmaIndex:35  */
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:18448 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X1_I0+16+0+0:vgprValuB_X1_I0+16+0+0+7], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
/* numPrefetchIter=1 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=1 readsPerIterA=6 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=1 readsPerIterB=6 */
label_toPGR1:

/******************************************/
/* Ord. NoLoadLoop - Begin                */
/******************************************/

/* iter 0 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:24, lwEndMfmaIndex:24  */
/*  numMfmaForLR:8, syncPlrMfmaIndex:27 , sync1LdsMfmaIndex:23 */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=4 for iteration == 0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:1  */
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:2  */
ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:3  */
ds_load_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA+0] offset:4640 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA+0] offset:4656 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:4  */
ds_load_b128 v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], v[vgprLocalReadAddrA+0] offset:9248 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+20:vgprValuA_X1_I0+20+3], v[vgprLocalReadAddrA+0] offset:9264 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:5  */
ds_load_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB+0] offset:9248 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:6  */
ds_load_b128 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[vgprLocalReadAddrB+0] offset:9264 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:7  */
ds_load_b128 v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprLocalReadAddrB+0] offset:18464 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/*  mfmaIndex:8  */
ds_load_b128 v[vgprValuB_X1_I0+20:vgprValuB_X1_I0+20+3], v[vgprLocalReadAddrB+0] offset:18480 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
/* localReadsVacancy: latencyLeft 1 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=6 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=6 */

/* iter 1 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:24, lwEndMfmaIndex:24  */
/*  numMfmaForLR:8, syncPlrMfmaIndex:27 , sync1LdsMfmaIndex:23 */
/*  mfmaIndex:9  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // wait for prior local read local write old=0, new=2 newLW=0 newLR=2
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:10  */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:11  */
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4672 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4688 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:12  */
ds_load_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA+0] offset:9280 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:13  */
ds_load_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA+0] offset:9296 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:14  */
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:9280 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:15  */
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:9296 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X1_I0+16+0+0:vgprValuB_X1_I0+16+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:16  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:18496 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X1_I0+16+0+0:vgprValuB_X1_I0+16+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/*  mfmaIndex:17  */
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:18512 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
/* localReadsVacancy: latencyLeft 1 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X1_I0+16+0+0:vgprValuB_X1_I0+16+0+0+7], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=6 */
/* dataAtIterB=0 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=6 */

/* iter 2 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:24, lwEndMfmaIndex:24  */
/*  numMfmaForLR:8, syncPlrMfmaIndex:27 , sync1LdsMfmaIndex:23 */
/*  mfmaIndex:18  */
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // wait for prior local read local write old=0, new=2 newLW=0 newLR=2
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:19  */
ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:20  */
ds_load_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA+0] offset:4704 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA+0] offset:4720 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:21  */
ds_load_b128 v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], v[vgprLocalReadAddrA+0] offset:9312 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:22  */
ds_load_b128 v[vgprValuA_X1_I0+20:vgprValuA_X1_I0+20+3], v[vgprLocalReadAddrA+0] offset:9328 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:23  */
/* schedule remaining localreads for one buffer scheduling */
ds_load_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB+0] offset:9312 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[vgprLocalReadAddrB+0] offset:9328 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprLocalReadAddrB+0] offset:18528 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+20:vgprValuB_X1_I0+20+3], v[vgprLocalReadAddrB+0] offset:18544 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:24  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:25  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/*  mfmaIndex:26  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=3 skipReadsIterA=1 readsPerIterA=6 */
/* dataAtIterB=1 numReadsIterB=3 skipReadsIterB=1 readsPerIterB=6 */

/* iter 3 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:24, lwEndMfmaIndex:24  */
/*  numMfmaForLR:8, syncPlrMfmaIndex:27 , sync1LdsMfmaIndex:23 */
/*  mfmaIndex:27  */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:28  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:29  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:30  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:31  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:32  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:33  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X1_I0+16+0+0:vgprValuB_X1_I0+16+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:34  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X1_I0+16+0+0:vgprValuB_X1_I0+16+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/*  mfmaIndex:35  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X1_I0+16+0+0:vgprValuB_X1_I0+16+0+0+7], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=0 readsPerIterA=6 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=0 readsPerIterB=6 */
label_toPGR1end_OrdNLL:
label_PrefetchGlobalLastIterEnd:

/* Tail: add ValuA/B vgpr buffer [90...187) to pool */

/* Tail: add address/G2L vgpr [187...224) to pool */
label_Summation_End_8:
.set sgprLoopCounterL, UNDEF
.set sgprOrigLoopCounter, UNDEF
.set sgprStrideScaleA, UNDEF
.set sgprSrdScaleA, UNDEF
.set sgprScaleAGShift, UNDEF
.set sgprScaleAKMask, UNDEF
.set sgprScaleAInc, UNDEF
.set sgprScaleAKCnt, UNDEF
.set sgprScaleALiftMagic, UNDEF
.set sgprSrdA, UNDEF
.set sgprSrdB, UNDEF
.set sgprShadowLimitA, UNDEF
.set sgprShadowLimitB, UNDEF
/* load store sgprs */
.set sgprSrdC, 64
.set sgprSrdD, 56

/* Mapping of Acc register -> C Vgpr register */

/* Multiply MI out register with Alpha -> C Vgpr register */
s_and_b32 s7, s[sgprArgType], 0xff                 // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s7, 3                                 // ArgType == 3 for General Batched GEMM
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

s_and_b32 s7, s[sgprArgType], 0xff                 // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s7, 3                                 // ArgType == 3 for General Batched GEMM
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

s_mov_b32 s7, 1
s_mov_b32 s8, 1
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_BPEDone                       // If synchronizer, use regular output BPE
s_cmp_eq_u32 s[sgprskTiles], 1                     // split == 1 ?
s_cbranch_scc1 label_BPEDone                       // If split == 1, use reguler output BPE
s_mov_b32 s7, 1
s_mov_b32 s8, 2
label_BPEDone:

s_mul_i32 s46, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_mul_hi_u32 s45, s46, s[sgprStrideC1J]            // ScaleC s46 by Stride
s_mul_i32 s44, s46, s[sgprStrideC1J]               // ScaleC s46 by Stride
s_lshl_b64 s[44:45], s[44:45], s7                  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s44        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s45       // add hi to SRD
s_mul_hi_u32 s45, s46, s[sgprStrideD1J]            // ScaleD s46 by Stride
s_mul_i32 s44, s46, s[sgprStrideD1J]               // ScaleD s46 by Stride
s_lshl_b64 s[44:45], s[44:45], s8                  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s44        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s45       // add hi to SRD

s_and_b32 s11, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s11, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_StridedBatchedGemmLoadC
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_GeneralBatchedGemmLoadC
label_StridedBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s45, s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride
s_mul_i32 s44, s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride
s_lshl_b64 s[44:45], s[44:45], s7                  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s44        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s45       // add hi to SRD
s_branch label_GeneralBatchedGemmLoadC_End
label_GeneralBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for General Batched GEMM
s_mul_i32 s44, 8, s[sgprWorkGroup2]                // Compute stride in bytes into Pointer Array
s_add_u32 s44, s44, s[sgprAddressC+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s45, s[sgprAddressC+1], 0               // Offsetting to the location [Higher half of address]
s_load_b64 s[44:45], s[44:45], 0                   // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for the Matrix Address Load from the Pointer Array
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s44        // Offsetting within the Batch Matrix [Lower half of address]
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s45       // Offsetting within the Batch Matrix [Higher half of address]
s_load_b64 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x98 // Load batchOffsetC from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s44        // Add matrix address to SRD (low)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s45       // Add matrix address to SRD (high)
label_GeneralBatchedGemmLoadC_End:  /// End of label GeneralBatchedGemmLoadC
s_and_b32 s11, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s11, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_StridedBatchedGemmLoadD
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_GeneralBatchedGemmLoadD
label_StridedBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s45, s[sgprWorkGroup2], s[sgprStrideDK] // ScaleD s[sgprWorkGroup2] by Stride
s_mul_i32 s44, s[sgprWorkGroup2], s[sgprStrideDK]  // ScaleD s[sgprWorkGroup2] by Stride
s_lshl_b64 s[44:45], s[44:45], s8                  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s44        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s45       // add hi to SRD
s_branch label_GeneralBatchedGemmLoadD_End
label_GeneralBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for General Batched GEMM
s_mul_i32 s44, 8, s[sgprWorkGroup2]                // Compute stride in bytes into Pointer Array
s_add_u32 s44, s44, s[sgprAddressD+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s45, s[sgprAddressD+1], 0               // Offsetting to the location [Higher half of address]
s_load_b64 s[44:45], s[44:45], 0                   // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for the Matrix Address Load from the Pointer Array
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s44        // Offsetting within the Batch Matrix [Lower half of address]
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s45       // Offsetting within the Batch Matrix [Higher half of address]
s_load_b64 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x90 // Load batchOffsetD from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s44        // Add matrix address to SRD (low)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s45       // Add matrix address to SRD (high)
label_GeneralBatchedGemmLoadD_End:  /// End of label GeneralBatchedGemmLoadD

s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_SK_SplitSrd                   // Skip this block if using single-kernel stream-k fixup
s_cmp_eq_u32 s[sgprskTiles], 1                     // split == 1 ?
s_cbranch_scc1 label_SK_SplitSrd                   // branch if split == 1
// Split Output Buffer offset: Free0 + (Free1-1)*StrideC1J + (Free2-1)*StrideCK * SplitIdx * bpe%s
s_mul_hi_u32 s45, s[sgprSizesFree+0], s[sgprSkPartialIdx] // Free0
s_mul_i32 s44, s[sgprSizesFree+0], s[sgprSkPartialIdx] // Free0
s_sub_u32 s46, s[sgprSizesFree+1], 1               // Free1
s_mul_i32 s46, s46, s[sgprSkPartialIdx]            // Free1
s_mul_hi_u32 s47, s46, s[sgprStrideC1J]            // Free1
s_mul_i32 s46, s46, s[sgprStrideC1J]               // Free1
s_add_u32 s44, s44, s46                            // Free1
s_addc_u32 s45, s45, s47                           // Free1
s_sub_u32 s46, s[sgprSizesFree+2], 1               // Free2
s_mul_i32 s46, s46, s[sgprSkPartialIdx]            // Free2
s_mul_hi_u32 s47, s46, s[sgprStrideCK]             // Free2
s_mul_i32 s46, s46, s[sgprStrideCK]                // Free2
s_add_u32 s44, s44, s46                            // Free2
s_addc_u32 s45, s45, s47                           // Free2
s_lshl_b64 s[44:45], s[44:45], 2                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s44        // add lo GSU offset to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s45       // add hi GSU offset to SRD
label_SK_SplitSrd:

/* not-LocalSplitU: global write indices */
/* computeStoreVgprs */
v_lshrrev_b32 v94, 5, v[vgprSerial]                // 94 = Serial / 32
v_lshrrev_b32 v95, 1, v94                          // 95 = 94 / 2
v_mul_lo_u32 v95, 0x10, v95                        // wave coordination offset 1
v_and_b32 v91, 31, v[vgprSerial]                   // v91 = v[vgprSerial] % 32
v_lshrrev_b32 v91, 4, v91                          // 91 = 91 / 16
                                                   // thread0 * continuous_output (multiplier is 1, do nothing)
v_add_lshl_u32 v91, v95, v91, 0                    // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v92, v91, s[sgprStrideC1J]            //  offset 1
v_mul_lo_u32 v93, v91, s[sgprStrideD1J]            //  offset 1
v_and_b32 v90, 1, v94                              // v90 = v94 % 2
v_mul_lo_u32 v90, 0x10, v90                        // wave coordination offset 0
v_and_b32 v95, 15, v[vgprSerial]                   // v95 = v[vgprSerial] % 16
v_add_lshl_u32 v90, v95, v90, 0                    // coordination 0 = vwA * (wave_id0 + tid0)
s_mul_i32 s7, 96, s[sgprWorkGroup0]                // wgp0 * MT0
v_add_nc_u32 v90, s7, v90                          // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s7, 192, s[sgprWorkGroup1]               // wgp1 * MT1
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

/* Edge/NonEdge store path check (M): Size % 96 > 0 -> Edge store; else -> NonEdge store */
s_mul_hi_u32 s71, s[sgprSizeI], 89478486           // STATIC_DIV: divisor=96
s_mul_i32 s70, s[sgprSizeI], 89478486              // tmp = dividend * magic
s_lshr_b64 s[70:71], s[70:71], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s69, s70                                 // quotient
s_mul_i32 s70, s69, 96                             // quotient*divisor
s_sub_u32 s68, s[sgprSizeI], s70                   // rReg = dividend - quotient*divisor
s_add_u32 s69, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s69                // wg0 >= nwg0-1 ?
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_MB_Else         // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 192 > 0 -> Edge store; else -> NonEdge store */
s_mul_hi_u32 s71, s[sgprSizeJ], 44739243           // STATIC_DIV: divisor=192
s_mul_i32 s70, s[sgprSizeJ], 44739243              // tmp = dividend * magic
s_lshr_b64 s[70:71], s[70:71], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s69, s70                                 // quotient
s_mul_i32 s70, s69, 192                            // quotient*divisor
s_sub_u32 s68, s[sgprSizeJ], s70                   // rReg = dividend - quotient*divisor
s_add_u32 s69, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s69                // wg1 >= nwg1-1
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_MB_Then         // jump if edges required
label_GW_B0_FD0_VW1_MB_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=150 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (8,2,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (9,2,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (10,2,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (11,2,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (12,2,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (13,2,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (14,2,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (15,2,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (16,2,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (17,2,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (18,2,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (19,2,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (20,2,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (21,2,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (22,2,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1); (23,2,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
/* (d1,vc1,d0,vc0)=(4,0,2,0) */
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
/* (d1,vc1,d0,vc0)=(5,0,2,0) */
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
/* (d1,vc1,d0,vc0)=(6,0,2,0) */
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
/* (d1,vc1,d0,vc0)=(7,0,2,0) */
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
/* (d1,vc1,d0,vc0)=(8,0,2,0) */
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
/* (d1,vc1,d0,vc0)=(9,0,1,0) */
/* (d1,vc1,d0,vc0)=(9,0,2,0) */
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
/* (d1,vc1,d0,vc0)=(10,0,1,0) */
/* (d1,vc1,d0,vc0)=(10,0,2,0) */
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
/* (d1,vc1,d0,vc0)=(11,0,1,0) */
/* (d1,vc1,d0,vc0)=(11,0,2,0) */
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
/* (d1,vc1,d0,vc0)=(12,0,2,0) */
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
/* (d1,vc1,d0,vc0)=(13,0,1,0) */
/* (d1,vc1,d0,vc0)=(13,0,2,0) */
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
/* (d1,vc1,d0,vc0)=(14,0,1,0) */
/* (d1,vc1,d0,vc0)=(14,0,2,0) */
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
/* (d1,vc1,d0,vc0)=(15,0,1,0) */
/* (d1,vc1,d0,vc0)=(15,0,2,0) */
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
/* (d1,vc1,d0,vc0)=(16,0,2,0) */
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
/* (d1,vc1,d0,vc0)=(17,0,1,0) */
/* (d1,vc1,d0,vc0)=(17,0,2,0) */
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
/* (d1,vc1,d0,vc0)=(18,0,1,0) */
/* (d1,vc1,d0,vc0)=(18,0,2,0) */
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
/* (d1,vc1,d0,vc0)=(19,0,1,0) */
/* (d1,vc1,d0,vc0)=(19,0,2,0) */
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
/* (d1,vc1,d0,vc0)=(20,0,2,0) */
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
/* (d1,vc1,d0,vc0)=(21,0,1,0) */
/* (d1,vc1,d0,vc0)=(21,0,2,0) */
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
/* (d1,vc1,d0,vc0)=(22,0,1,0) */
/* (d1,vc1,d0,vc0)=(22,0,2,0) */
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
/* (d1,vc1,d0,vc0)=(23,0,1,0) */
/* (d1,vc1,d0,vc0)=(23,0,2,0) */
v_add_lshl_u32 v101, v93, v90, 2                   // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=90, coord0Vgpr=90 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (5, 2, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (7, 2, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (8, 2, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (9, 2, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (10, 2, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (11, 2, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (12, 2, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (13, 2, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (14, 2, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (15, 2, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (16, 2, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (17, 2, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (18, 2, 0, 0), (19, 0, 0, 0), (19, 1, 0, 0), (19, 2, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0), (20, 2, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0), (21, 2, 0, 0), (22, 0, 0, 0), (22, 1, 0, 0), (22, 2, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0), (23, 2, 0, 0)] */
v_mov_b32 v[vgprValuC+103], v[vgprValuC+0]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+104], v[vgprValuC+8]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+105], v[vgprValuC+16]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+106], v[vgprValuC+1]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+107], v[vgprValuC+9]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+108], v[vgprValuC+17]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+109], v[vgprValuC+2]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+110], v[vgprValuC+10]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+111], v[vgprValuC+18]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+112], v[vgprValuC+3]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+113], v[vgprValuC+11]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+114], v[vgprValuC+19]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+115], v[vgprValuC+4]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+116], v[vgprValuC+12]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+117], v[vgprValuC+20]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+118], v[vgprValuC+5]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+119], v[vgprValuC+13]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+120], v[vgprValuC+21]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+121], v[vgprValuC+6]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+122], v[vgprValuC+14]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+123], v[vgprValuC+22]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+124], v[vgprValuC+7]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+125], v[vgprValuC+15]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+126], v[vgprValuC+23]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+127], v[vgprValuC+24]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+128], v[vgprValuC+32]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+129], v[vgprValuC+40]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+130], v[vgprValuC+25]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+131], v[vgprValuC+33]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+132], v[vgprValuC+41]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+133], v[vgprValuC+26]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+134], v[vgprValuC+34]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+135], v[vgprValuC+42]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+136], v[vgprValuC+27]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+137], v[vgprValuC+35]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+138], v[vgprValuC+43]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+139], v[vgprValuC+28]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+140], v[vgprValuC+36]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+141], v[vgprValuC+44]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+142], v[vgprValuC+29]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+143], v[vgprValuC+37]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+144], v[vgprValuC+45]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+145], v[vgprValuC+30]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+146], v[vgprValuC+38]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+147], v[vgprValuC+46]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+148], v[vgprValuC+31]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+149], v[vgprValuC+39]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+150], v[vgprValuC+47]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+151], v[vgprValuC+48]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+152], v[vgprValuC+56]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+153], v[vgprValuC+64]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+154], v[vgprValuC+49]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+155], v[vgprValuC+57]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+156], v[vgprValuC+65]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+157], v[vgprValuC+50]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+158], v[vgprValuC+58]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+159], v[vgprValuC+66]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+160], v[vgprValuC+51]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+161], v[vgprValuC+59]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+162], v[vgprValuC+67]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+163], v[vgprValuC+52]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+164], v[vgprValuC+60]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+165], v[vgprValuC+68]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+166], v[vgprValuC+53]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+167], v[vgprValuC+61]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+168], v[vgprValuC+69]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+169], v[vgprValuC+54]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+170], v[vgprValuC+62]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+171], v[vgprValuC+70]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+172], v[vgprValuC+55]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+173], v[vgprValuC+63]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+174], v[vgprValuC+71]        // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b32 v103, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v104, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b32 v105, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v106, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v107, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b32 v108, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v109, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v110, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b32 v111, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v112, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v113, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b32 v114, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v115, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v116, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b32 v117, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v118, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v119, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b32 v120, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v121, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v122, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b32 v123, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v124, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v125, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b32 v126, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 200                // scale StrideD *= numRows(50) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(50): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(50): gra SRD += inc(upper)
buffer_store_b32 v127, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v128, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b32 v129, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v130, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v131, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b32 v132, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v133, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v134, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b32 v135, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v136, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v137, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b32 v138, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v139, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v140, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b32 v141, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v142, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v143, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b32 v144, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v145, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v146, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b32 v147, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v148, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v149, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b32 v150, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 200                // scale StrideD *= numRows(50) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(50): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(50): gra SRD += inc(upper)
buffer_store_b32 v151, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v152, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b32 v153, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v154, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v155, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b32 v156, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v157, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v158, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b32 v159, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v160, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v161, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b32 v162, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v163, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v164, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b32 v165, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v166, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v167, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b32 v168, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v169, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v170, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b32 v171, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v172, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v173, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b32 v174, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_B0_FD0_VW1_MB_NonEdgeEnd:
label_GW_B0_FD0_VW1_MB_Else:
label_GW_B0_FD0_VW1_MB_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=76 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (8,2,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (9,2,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (10,2,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (11,2,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (12,2,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (13,2,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (14,2,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (15,2,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (16,2,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (17,2,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (18,2,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (19,2,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (20,2,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (21,2,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (22,2,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1); (23,2,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v96, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v173, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v173, v96, v173, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v174, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v174, v96, v174, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v175, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v175, v96, v175, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v176, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v176, v96, v176, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v177, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v177, v96, v177, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v178, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v178, v96, v178, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v179, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v179, v96, v179, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v180, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v180, v96, v180, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v181, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v181, v96, v181, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v182, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v182, v96, v182, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v183, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v183, v96, v183, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v184, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v184, v96, v184, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v185, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v185, v96, v185, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v186, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v186, v96, v186, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v187, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v187, v96, v187, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v188, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v188, v96, v188, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v189, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v189, v96, v189, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v190, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v190, v96, v190, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v191, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v191, v96, v191, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v192, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v192, v96, v192, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v193, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v193, v96, v193, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v194, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v194, v96, v194, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v195, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v195, v96, v195, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v196, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v196, v96, v196, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 50                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 50                // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 50                // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v197, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v197, v96, v197, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v198, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v198, v96, v198, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v199, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v199, v96, v199, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v200, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v200, v96, v200, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v201, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v201, v96, v201, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v202, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v202, v96, v202, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v203, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v203, v96, v203, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v204, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v204, v96, v204, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v205, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v205, v96, v205, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v206, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v206, v96, v206, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v207, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v207, v96, v207, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v208, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v208, v96, v208, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v209, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v209, v96, v209, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v210, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v210, v96, v210, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v211, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v211, v96, v211, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v212, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v212, v96, v212, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v213, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v213, v96, v213, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v214, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v214, v96, v214, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v215, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v215, v96, v215, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v216, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v216, v96, v216, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v217, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v217, v96, v217, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v218, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v218, v96, v218, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v219, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v219, v96, v219, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v220, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v220, v96, v220, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 50                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 50                // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 50                // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v221, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v221, v96, v221, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v222, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v222, v96, v222, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v223, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v223, v96, v223, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v226, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v226, v96, v226, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v227, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v227, v96, v227, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v228, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v228, v96, v228, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v229, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v229, v96, v229, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v230, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v230, v96, v230, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v231, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v231, v96, v231, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v232, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v232, v96, v232, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v233, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v233, v96, v233, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v234, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v234, v96, v234, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v235, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v235, v96, v235, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v236, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v236, v96, v236, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v237, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v237, v96, v237, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v238, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v238, v96, v238, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v239, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v239, v96, v239, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v240, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v240, v96, v240, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v241, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v241, v96, v241, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v242, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v242, v96, v242, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v243, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v243, v96, v243, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v244, v93, v90, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v244, v96, v244, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v245, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v245, v96, v245, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v246, v93, v94, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v246, v96, v246, s70                 // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (5, 2, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (7, 2, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (8, 2, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (9, 2, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (10, 2, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (11, 2, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (12, 2, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (13, 2, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (14, 2, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (15, 2, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (16, 2, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (17, 2, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (18, 2, 0, 0), (19, 0, 0, 0), (19, 1, 0, 0), (19, 2, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0), (20, 2, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0), (21, 2, 0, 0), (22, 0, 0, 0), (22, 1, 0, 0), (22, 2, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0), (23, 2, 0, 0)] */
v_mov_b32 v[vgprValuC+101], v[vgprValuC+0]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+102], v[vgprValuC+8]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+103], v[vgprValuC+16]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+104], v[vgprValuC+1]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+105], v[vgprValuC+9]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+106], v[vgprValuC+17]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+107], v[vgprValuC+2]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+108], v[vgprValuC+10]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+109], v[vgprValuC+18]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+110], v[vgprValuC+3]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+111], v[vgprValuC+11]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+112], v[vgprValuC+19]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+113], v[vgprValuC+4]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+114], v[vgprValuC+12]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+115], v[vgprValuC+20]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+116], v[vgprValuC+5]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+117], v[vgprValuC+13]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+118], v[vgprValuC+21]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+119], v[vgprValuC+6]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+120], v[vgprValuC+14]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+121], v[vgprValuC+22]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+122], v[vgprValuC+7]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+123], v[vgprValuC+15]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+124], v[vgprValuC+23]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+125], v[vgprValuC+24]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+126], v[vgprValuC+32]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+127], v[vgprValuC+40]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+128], v[vgprValuC+25]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+129], v[vgprValuC+33]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+130], v[vgprValuC+41]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+131], v[vgprValuC+26]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+132], v[vgprValuC+34]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+133], v[vgprValuC+42]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+134], v[vgprValuC+27]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+135], v[vgprValuC+35]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+136], v[vgprValuC+43]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+137], v[vgprValuC+28]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+138], v[vgprValuC+36]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+139], v[vgprValuC+44]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+140], v[vgprValuC+29]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+141], v[vgprValuC+37]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+142], v[vgprValuC+45]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+143], v[vgprValuC+30]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+144], v[vgprValuC+38]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+145], v[vgprValuC+46]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+146], v[vgprValuC+31]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+147], v[vgprValuC+39]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+148], v[vgprValuC+47]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+149], v[vgprValuC+48]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+150], v[vgprValuC+56]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+151], v[vgprValuC+64]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+152], v[vgprValuC+49]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+153], v[vgprValuC+57]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+154], v[vgprValuC+65]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+155], v[vgprValuC+50]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+156], v[vgprValuC+58]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+157], v[vgprValuC+66]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+158], v[vgprValuC+51]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+159], v[vgprValuC+59]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+160], v[vgprValuC+67]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+161], v[vgprValuC+52]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+162], v[vgprValuC+60]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+163], v[vgprValuC+68]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+164], v[vgprValuC+53]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+165], v[vgprValuC+61]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+166], v[vgprValuC+69]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+167], v[vgprValuC+54]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+168], v[vgprValuC+62]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+169], v[vgprValuC+70]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+170], v[vgprValuC+55]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+171], v[vgprValuC+63]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+172], v[vgprValuC+71]        // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b32 v101, v173, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v102, v174, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v103, v175, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v104, v176, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v105, v177, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v106, v178, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v107, v179, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v108, v180, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v109, v181, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v110, v182, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v111, v183, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v112, v184, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v113, v185, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v114, v186, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v115, v187, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v116, v188, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v117, v189, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v118, v190, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v119, v191, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v120, v192, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v121, v193, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v122, v194, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v123, v195, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v124, v196, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v125, v197, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v126, v198, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v127, v199, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v128, v200, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v129, v201, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v130, v202, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v131, v203, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v132, v204, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v133, v205, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v134, v206, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v135, v207, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v136, v208, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v137, v209, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v138, v210, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v139, v211, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v140, v212, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v141, v213, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v142, v214, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v143, v215, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v144, v216, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v145, v217, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v146, v218, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v147, v219, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v148, v220, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v149, v221, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v150, v222, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v151, v223, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v152, v226, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v153, v227, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v154, v228, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v155, v229, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v156, v230, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v157, v231, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v158, v232, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v159, v233, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v160, v234, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v161, v235, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v162, v236, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v163, v237, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v164, v238, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v165, v239, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v166, v240, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v167, v241, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v168, v242, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v169, v243, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v170, v244, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v171, v245, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v172, v246, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_End:
s_getpc_b64 s[68:69]                               // addr of next instr
s_add_i32 s70, label_KernelEnd, 4                  // target branch offset
s_add_u32 s68, s68, s70                            // add target branch offset
s_addc_u32 s69, s69, 0                             // add high and carry
s_setpc_b64 s[68:69]                               // branch to label_KernelEnd
label_GSU:
s_cmp_eq_u32 s[sgprStreamKLocalStart], 0           // does wg start tile?
s_cbranch_scc1 label_NoBranch_9                    // Only branch on scc0
s_getpc_b64 s[72:73]                               // addr of next instr
s_add_i32 s74, label_SK_Partials_1, 4              // target branch offset
s_add_u32 s72, s72, s74                            // add target branch offset
s_addc_u32 s73, s73, 0                             // add high and carry
s_setpc_b64 s[72:73]                               // branch to label_SK_Partials_1
label_NoBranch_9:
s_cmp_eq_u32 s[sgprStreamKLocalEnd], s[sgprItersPerTile] // does wg finish tile?
s_cbranch_scc1 label_SK_Store                      // Branch if started and finished tile, go to regular store code
s_add_u32 s7, s[sgprPersistentWorkGroupIndex], 1   // input partial tile index
s_mul_hi_u32 s69, s[sgprPersistentIterationEnd], s[sgprMagicNumberItersPerTile] // s_magic mul, div alg 2
s_lshr_b32 s70, s[sgprMagicShiftItersPerTile], 31  // tmpS = extract abit
s_mul_i32 s68, s[sgprPersistentIterationEnd], s70  // s_magic mul, div alg 2
s_add_u32 s68, s68, s69
s_and_b32 s70, s[sgprMagicShiftItersPerTile], 2147483647 // tmpS = remove abit to final shift
s_lshr_b32 s68, s68, s70                           // sMagicDiv Alg 2
s_mul_i32 s68, s68, s[sgprItersPerTile]            // start iteration of partial tile
s_sub_u32 s8, s[sgprPersistentIterationEnd], s68   // calc iterations completed by this WG
label_SK_Fixup:
s_lshl_b32 s68, s7, 2                              // flag offset based on physical partial slot
s_load_b32 s70, s[sgprAddressFlags:sgprAddressFlags+1], s68 glc dlc // get flag
s_waitcnt lgkmcnt(0)                               // wait for flag load
s_cmp_eq_u32 s70, 1                                // check if ready
s_cbranch_scc0 label_SK_Fixup                      // if flag not set, wait and check again
s_barrier                                          // wait for all workgroups before resetting flag
v_readfirstlane_b32 s70, v[vgprSerial]             // Wave 0 updates flags
s_cmp_eq_u32 s70, 0                                // Check for wave 0
s_cbranch_scc0 label_SK_SkipFlagReset              // Skip flag reset
v_mov_b32 v101, s70                                // move flag value to vgpr
v_mov_b32 v102, 0                                  // zero vaddr offset
s_mov_b64 s[72:73], s[sgprAddressFlags:sgprAddressFlags+1]
s_mov_b32 s74, BufferOOB
s_mov_b32 s75, Srd127_96
buffer_store_b32 v101, v102, s[72:75], s68 offen offset:0 glc dlc // reset flag
s_waitcnt_vscnt null 0                             // wait for data store
label_SK_SkipFlagReset:
label_Fixup_E0:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=74 */
s_mov_b64 s[sgprSrdWS:sgprSrdWS+1], s[sgprAddressWS:sgprAddressWS+1]
s_mov_b32 s[sgprSrdWS+2], BufferOOB
s_mov_b32 s[sgprSrdWS+3], Srd127_96

s_mul_i32 s62, 0x12000, s7                         // Offset to correct partials tile (low word)
s_mul_hi_u32 s11, 0x12000, s7                      // partials tile offset (high word) for 64-bit SRD
s_add_u32 s[sgprSrdWS+0], s[sgprSrdWS+0], s62      // add lo to SRD
s_addc_u32 s[sgprSrdWS+1], s[sgprSrdWS+1], s11     // add hi (offset high word + lo carry) to SRD
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Fixup Batch #0 (d1,d0,vc1,vc0) =       */
/*      (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (8,2,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (9,2,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (10,2,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (11,2,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (12,2,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (13,2,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (14,2,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (15,2,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (16,2,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (17,2,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (18,2,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (19,2,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (20,2,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (21,2,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (22,2,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1); (23,2,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_lshlrev_b32 v102, 2, v[vgprSerial]               // v102 = v[vgprSerial] * 4
s_mov_b32 s62, 0                                   // Init sgpr offset
buffer_load_b32 v175, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v176, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v177, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v178, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v179, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v180, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v181, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v182, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v183, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v184, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v185, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v186, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v187, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v188, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v189, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v190, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v191, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v192, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v193, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v194, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v195, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v196, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v197, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v198, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v199, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v200, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v201, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v202, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v203, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v204, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v205, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v206, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v207, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v208, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v209, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v210, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v211, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v212, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v213, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v214, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v215, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v216, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v217, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v218, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v219, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v220, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v221, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v222, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v223, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v226, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v227, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v228, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v229, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v230, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v231, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v232, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v233, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v234, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v235, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v236, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v237, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v238, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v239, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v240, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v241, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v242, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v243, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v244, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v245, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v246, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v247, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v248, v102, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
v_mov_b32 v[vgprValuC+103], v[vgprValuC+0]         // copy MI out reg to vreg[0]
v_mov_b32 v[vgprValuC+104], v[vgprValuC+8]         // copy MI out reg to vreg[1]
v_mov_b32 v[vgprValuC+105], v[vgprValuC+16]        // copy MI out reg to vreg[2]
v_mov_b32 v[vgprValuC+106], v[vgprValuC+1]         // copy MI out reg to vreg[3]
v_mov_b32 v[vgprValuC+107], v[vgprValuC+9]         // copy MI out reg to vreg[4]
v_mov_b32 v[vgprValuC+108], v[vgprValuC+17]        // copy MI out reg to vreg[5]
v_mov_b32 v[vgprValuC+109], v[vgprValuC+2]         // copy MI out reg to vreg[6]
v_mov_b32 v[vgprValuC+110], v[vgprValuC+10]        // copy MI out reg to vreg[7]
v_mov_b32 v[vgprValuC+111], v[vgprValuC+18]        // copy MI out reg to vreg[8]
v_mov_b32 v[vgprValuC+112], v[vgprValuC+3]         // copy MI out reg to vreg[9]
v_mov_b32 v[vgprValuC+113], v[vgprValuC+11]        // copy MI out reg to vreg[10]
v_mov_b32 v[vgprValuC+114], v[vgprValuC+19]        // copy MI out reg to vreg[11]
v_mov_b32 v[vgprValuC+115], v[vgprValuC+4]         // copy MI out reg to vreg[12]
v_mov_b32 v[vgprValuC+116], v[vgprValuC+12]        // copy MI out reg to vreg[13]
v_mov_b32 v[vgprValuC+117], v[vgprValuC+20]        // copy MI out reg to vreg[14]
v_mov_b32 v[vgprValuC+118], v[vgprValuC+5]         // copy MI out reg to vreg[15]
v_mov_b32 v[vgprValuC+119], v[vgprValuC+13]        // copy MI out reg to vreg[16]
v_mov_b32 v[vgprValuC+120], v[vgprValuC+21]        // copy MI out reg to vreg[17]
v_mov_b32 v[vgprValuC+121], v[vgprValuC+6]         // copy MI out reg to vreg[18]
v_mov_b32 v[vgprValuC+122], v[vgprValuC+14]        // copy MI out reg to vreg[19]
v_mov_b32 v[vgprValuC+123], v[vgprValuC+22]        // copy MI out reg to vreg[20]
v_mov_b32 v[vgprValuC+124], v[vgprValuC+7]         // copy MI out reg to vreg[21]
v_mov_b32 v[vgprValuC+125], v[vgprValuC+15]        // copy MI out reg to vreg[22]
v_mov_b32 v[vgprValuC+126], v[vgprValuC+23]        // copy MI out reg to vreg[23]
v_mov_b32 v[vgprValuC+127], v[vgprValuC+24]        // copy MI out reg to vreg[24]
v_mov_b32 v[vgprValuC+128], v[vgprValuC+32]        // copy MI out reg to vreg[25]
v_mov_b32 v[vgprValuC+129], v[vgprValuC+40]        // copy MI out reg to vreg[26]
v_mov_b32 v[vgprValuC+130], v[vgprValuC+25]        // copy MI out reg to vreg[27]
v_mov_b32 v[vgprValuC+131], v[vgprValuC+33]        // copy MI out reg to vreg[28]
v_mov_b32 v[vgprValuC+132], v[vgprValuC+41]        // copy MI out reg to vreg[29]
v_mov_b32 v[vgprValuC+133], v[vgprValuC+26]        // copy MI out reg to vreg[30]
v_mov_b32 v[vgprValuC+134], v[vgprValuC+34]        // copy MI out reg to vreg[31]
v_mov_b32 v[vgprValuC+135], v[vgprValuC+42]        // copy MI out reg to vreg[32]
v_mov_b32 v[vgprValuC+136], v[vgprValuC+27]        // copy MI out reg to vreg[33]
v_mov_b32 v[vgprValuC+137], v[vgprValuC+35]        // copy MI out reg to vreg[34]
v_mov_b32 v[vgprValuC+138], v[vgprValuC+43]        // copy MI out reg to vreg[35]
v_mov_b32 v[vgprValuC+139], v[vgprValuC+28]        // copy MI out reg to vreg[36]
v_mov_b32 v[vgprValuC+140], v[vgprValuC+36]        // copy MI out reg to vreg[37]
v_mov_b32 v[vgprValuC+141], v[vgprValuC+44]        // copy MI out reg to vreg[38]
v_mov_b32 v[vgprValuC+142], v[vgprValuC+29]        // copy MI out reg to vreg[39]
v_mov_b32 v[vgprValuC+143], v[vgprValuC+37]        // copy MI out reg to vreg[40]
v_mov_b32 v[vgprValuC+144], v[vgprValuC+45]        // copy MI out reg to vreg[41]
v_mov_b32 v[vgprValuC+145], v[vgprValuC+30]        // copy MI out reg to vreg[42]
v_mov_b32 v[vgprValuC+146], v[vgprValuC+38]        // copy MI out reg to vreg[43]
v_mov_b32 v[vgprValuC+147], v[vgprValuC+46]        // copy MI out reg to vreg[44]
v_mov_b32 v[vgprValuC+148], v[vgprValuC+31]        // copy MI out reg to vreg[45]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+39]        // copy MI out reg to vreg[46]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+47]        // copy MI out reg to vreg[47]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+48]        // copy MI out reg to vreg[48]
v_mov_b32 v[vgprValuC+152], v[vgprValuC+56]        // copy MI out reg to vreg[49]
v_mov_b32 v[vgprValuC+153], v[vgprValuC+64]        // copy MI out reg to vreg[50]
v_mov_b32 v[vgprValuC+154], v[vgprValuC+49]        // copy MI out reg to vreg[51]
v_mov_b32 v[vgprValuC+155], v[vgprValuC+57]        // copy MI out reg to vreg[52]
v_mov_b32 v[vgprValuC+156], v[vgprValuC+65]        // copy MI out reg to vreg[53]
v_mov_b32 v[vgprValuC+157], v[vgprValuC+50]        // copy MI out reg to vreg[54]
v_mov_b32 v[vgprValuC+158], v[vgprValuC+58]        // copy MI out reg to vreg[55]
v_mov_b32 v[vgprValuC+159], v[vgprValuC+66]        // copy MI out reg to vreg[56]
v_mov_b32 v[vgprValuC+160], v[vgprValuC+51]        // copy MI out reg to vreg[57]
v_mov_b32 v[vgprValuC+161], v[vgprValuC+59]        // copy MI out reg to vreg[58]
v_mov_b32 v[vgprValuC+162], v[vgprValuC+67]        // copy MI out reg to vreg[59]
v_mov_b32 v[vgprValuC+163], v[vgprValuC+52]        // copy MI out reg to vreg[60]
v_mov_b32 v[vgprValuC+164], v[vgprValuC+60]        // copy MI out reg to vreg[61]
v_mov_b32 v[vgprValuC+165], v[vgprValuC+68]        // copy MI out reg to vreg[62]
v_mov_b32 v[vgprValuC+166], v[vgprValuC+53]        // copy MI out reg to vreg[63]
v_mov_b32 v[vgprValuC+167], v[vgprValuC+61]        // copy MI out reg to vreg[64]
v_mov_b32 v[vgprValuC+168], v[vgprValuC+69]        // copy MI out reg to vreg[65]
v_mov_b32 v[vgprValuC+169], v[vgprValuC+54]        // copy MI out reg to vreg[66]
v_mov_b32 v[vgprValuC+170], v[vgprValuC+62]        // copy MI out reg to vreg[67]
v_mov_b32 v[vgprValuC+171], v[vgprValuC+70]        // copy MI out reg to vreg[68]
v_mov_b32 v[vgprValuC+172], v[vgprValuC+55]        // copy MI out reg to vreg[69]
v_mov_b32 v[vgprValuC+173], v[vgprValuC+63]        // copy MI out reg to vreg[70]
v_mov_b32 v[vgprValuC+174], v[vgprValuC+71]        // copy MI out reg to vreg[71]

/* apply mask, calc new C and issue writes */
v_mov_b32 v98, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v99, 0x7fff0000                          // fp32 Nan
v_mov_b32 v100, 0x7fff                             // rounding bias for bfloat16

s_waitcnt vmcnt(63)                                // wait C (interleaved) 71 = 72 - 0 - 1
v_add_f32 v[vgprValuC+103], v[vgprValuC+103], v175 // accum partials

s_waitcnt vmcnt(63)                                // wait C (interleaved) 70 = 72 - 1 - 1
v_add_f32 v[vgprValuC+104], v[vgprValuC+104], v176 // accum partials

s_waitcnt vmcnt(63)                                // wait C (interleaved) 69 = 72 - 2 - 1
v_add_f32 v[vgprValuC+105], v[vgprValuC+105], v177 // accum partials

s_waitcnt vmcnt(63)                                // wait C (interleaved) 68 = 72 - 3 - 1
v_add_f32 v[vgprValuC+106], v[vgprValuC+106], v178 // accum partials

s_waitcnt vmcnt(63)                                // wait C (interleaved) 67 = 72 - 4 - 1
v_add_f32 v[vgprValuC+107], v[vgprValuC+107], v179 // accum partials

s_waitcnt vmcnt(63)                                // wait C (interleaved) 66 = 72 - 5 - 1
v_add_f32 v[vgprValuC+108], v[vgprValuC+108], v180 // accum partials

s_waitcnt vmcnt(63)                                // wait C (interleaved) 65 = 72 - 6 - 1
v_add_f32 v[vgprValuC+109], v[vgprValuC+109], v181 // accum partials

s_waitcnt vmcnt(63)                                // wait C (interleaved) 64 = 72 - 7 - 1
v_add_f32 v[vgprValuC+110], v[vgprValuC+110], v182 // accum partials

s_waitcnt vmcnt(63)                                // wait C (interleaved) 63 = 72 - 8 - 1
v_add_f32 v[vgprValuC+111], v[vgprValuC+111], v183 // accum partials

s_waitcnt vmcnt(62)                                // wait C (interleaved) 62 = 72 - 9 - 1
v_add_f32 v[vgprValuC+112], v[vgprValuC+112], v184 // accum partials

s_waitcnt vmcnt(61)                                // wait C (interleaved) 61 = 72 - 10 - 1
v_add_f32 v[vgprValuC+113], v[vgprValuC+113], v185 // accum partials

s_waitcnt vmcnt(60)                                // wait C (interleaved) 60 = 72 - 11 - 1
v_add_f32 v[vgprValuC+114], v[vgprValuC+114], v186 // accum partials

s_waitcnt vmcnt(59)                                // wait C (interleaved) 59 = 72 - 12 - 1
v_add_f32 v[vgprValuC+115], v[vgprValuC+115], v187 // accum partials

s_waitcnt vmcnt(58)                                // wait C (interleaved) 58 = 72 - 13 - 1
v_add_f32 v[vgprValuC+116], v[vgprValuC+116], v188 // accum partials

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 72 - 14 - 1
v_add_f32 v[vgprValuC+117], v[vgprValuC+117], v189 // accum partials

s_waitcnt vmcnt(56)                                // wait C (interleaved) 56 = 72 - 15 - 1
v_add_f32 v[vgprValuC+118], v[vgprValuC+118], v190 // accum partials

s_waitcnt vmcnt(55)                                // wait C (interleaved) 55 = 72 - 16 - 1
v_add_f32 v[vgprValuC+119], v[vgprValuC+119], v191 // accum partials

s_waitcnt vmcnt(54)                                // wait C (interleaved) 54 = 72 - 17 - 1
v_add_f32 v[vgprValuC+120], v[vgprValuC+120], v192 // accum partials

s_waitcnt vmcnt(53)                                // wait C (interleaved) 53 = 72 - 18 - 1
v_add_f32 v[vgprValuC+121], v[vgprValuC+121], v193 // accum partials

s_waitcnt vmcnt(52)                                // wait C (interleaved) 52 = 72 - 19 - 1
v_add_f32 v[vgprValuC+122], v[vgprValuC+122], v194 // accum partials

s_waitcnt vmcnt(51)                                // wait C (interleaved) 51 = 72 - 20 - 1
v_add_f32 v[vgprValuC+123], v[vgprValuC+123], v195 // accum partials

s_waitcnt vmcnt(50)                                // wait C (interleaved) 50 = 72 - 21 - 1
v_add_f32 v[vgprValuC+124], v[vgprValuC+124], v196 // accum partials

s_waitcnt vmcnt(49)                                // wait C (interleaved) 49 = 72 - 22 - 1
v_add_f32 v[vgprValuC+125], v[vgprValuC+125], v197 // accum partials

s_waitcnt vmcnt(48)                                // wait C (interleaved) 48 = 72 - 23 - 1
v_add_f32 v[vgprValuC+126], v[vgprValuC+126], v198 // accum partials

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 72 - 24 - 1
v_add_f32 v[vgprValuC+127], v[vgprValuC+127], v199 // accum partials

s_waitcnt vmcnt(46)                                // wait C (interleaved) 46 = 72 - 25 - 1
v_add_f32 v[vgprValuC+128], v[vgprValuC+128], v200 // accum partials

s_waitcnt vmcnt(45)                                // wait C (interleaved) 45 = 72 - 26 - 1
v_add_f32 v[vgprValuC+129], v[vgprValuC+129], v201 // accum partials

s_waitcnt vmcnt(44)                                // wait C (interleaved) 44 = 72 - 27 - 1
v_add_f32 v[vgprValuC+130], v[vgprValuC+130], v202 // accum partials

s_waitcnt vmcnt(43)                                // wait C (interleaved) 43 = 72 - 28 - 1
v_add_f32 v[vgprValuC+131], v[vgprValuC+131], v203 // accum partials

s_waitcnt vmcnt(42)                                // wait C (interleaved) 42 = 72 - 29 - 1
v_add_f32 v[vgprValuC+132], v[vgprValuC+132], v204 // accum partials

s_waitcnt vmcnt(41)                                // wait C (interleaved) 41 = 72 - 30 - 1
v_add_f32 v[vgprValuC+133], v[vgprValuC+133], v205 // accum partials

s_waitcnt vmcnt(40)                                // wait C (interleaved) 40 = 72 - 31 - 1
v_add_f32 v[vgprValuC+134], v[vgprValuC+134], v206 // accum partials

s_waitcnt vmcnt(39)                                // wait C (interleaved) 39 = 72 - 32 - 1
v_add_f32 v[vgprValuC+135], v[vgprValuC+135], v207 // accum partials

s_waitcnt vmcnt(38)                                // wait C (interleaved) 38 = 72 - 33 - 1
v_add_f32 v[vgprValuC+136], v[vgprValuC+136], v208 // accum partials

s_waitcnt vmcnt(37)                                // wait C (interleaved) 37 = 72 - 34 - 1
v_add_f32 v[vgprValuC+137], v[vgprValuC+137], v209 // accum partials

s_waitcnt vmcnt(36)                                // wait C (interleaved) 36 = 72 - 35 - 1
v_add_f32 v[vgprValuC+138], v[vgprValuC+138], v210 // accum partials

s_waitcnt vmcnt(35)                                // wait C (interleaved) 35 = 72 - 36 - 1
v_add_f32 v[vgprValuC+139], v[vgprValuC+139], v211 // accum partials

s_waitcnt vmcnt(34)                                // wait C (interleaved) 34 = 72 - 37 - 1
v_add_f32 v[vgprValuC+140], v[vgprValuC+140], v212 // accum partials

s_waitcnt vmcnt(33)                                // wait C (interleaved) 33 = 72 - 38 - 1
v_add_f32 v[vgprValuC+141], v[vgprValuC+141], v213 // accum partials

s_waitcnt vmcnt(32)                                // wait C (interleaved) 32 = 72 - 39 - 1
v_add_f32 v[vgprValuC+142], v[vgprValuC+142], v214 // accum partials

s_waitcnt vmcnt(31)                                // wait C (interleaved) 31 = 72 - 40 - 1
v_add_f32 v[vgprValuC+143], v[vgprValuC+143], v215 // accum partials

s_waitcnt vmcnt(30)                                // wait C (interleaved) 30 = 72 - 41 - 1
v_add_f32 v[vgprValuC+144], v[vgprValuC+144], v216 // accum partials

s_waitcnt vmcnt(29)                                // wait C (interleaved) 29 = 72 - 42 - 1
v_add_f32 v[vgprValuC+145], v[vgprValuC+145], v217 // accum partials

s_waitcnt vmcnt(28)                                // wait C (interleaved) 28 = 72 - 43 - 1
v_add_f32 v[vgprValuC+146], v[vgprValuC+146], v218 // accum partials

s_waitcnt vmcnt(27)                                // wait C (interleaved) 27 = 72 - 44 - 1
v_add_f32 v[vgprValuC+147], v[vgprValuC+147], v219 // accum partials

s_waitcnt vmcnt(26)                                // wait C (interleaved) 26 = 72 - 45 - 1
v_add_f32 v[vgprValuC+148], v[vgprValuC+148], v220 // accum partials

s_waitcnt vmcnt(25)                                // wait C (interleaved) 25 = 72 - 46 - 1
v_add_f32 v[vgprValuC+149], v[vgprValuC+149], v221 // accum partials

s_waitcnt vmcnt(24)                                // wait C (interleaved) 24 = 72 - 47 - 1
v_add_f32 v[vgprValuC+150], v[vgprValuC+150], v222 // accum partials

s_waitcnt vmcnt(23)                                // wait C (interleaved) 23 = 72 - 48 - 1
v_add_f32 v[vgprValuC+151], v[vgprValuC+151], v223 // accum partials

s_waitcnt vmcnt(22)                                // wait C (interleaved) 22 = 72 - 49 - 1
v_add_f32 v[vgprValuC+152], v[vgprValuC+152], v226 // accum partials

s_waitcnt vmcnt(21)                                // wait C (interleaved) 21 = 72 - 50 - 1
v_add_f32 v[vgprValuC+153], v[vgprValuC+153], v227 // accum partials

s_waitcnt vmcnt(20)                                // wait C (interleaved) 20 = 72 - 51 - 1
v_add_f32 v[vgprValuC+154], v[vgprValuC+154], v228 // accum partials

s_waitcnt vmcnt(19)                                // wait C (interleaved) 19 = 72 - 52 - 1
v_add_f32 v[vgprValuC+155], v[vgprValuC+155], v229 // accum partials

s_waitcnt vmcnt(18)                                // wait C (interleaved) 18 = 72 - 53 - 1
v_add_f32 v[vgprValuC+156], v[vgprValuC+156], v230 // accum partials

s_waitcnt vmcnt(17)                                // wait C (interleaved) 17 = 72 - 54 - 1
v_add_f32 v[vgprValuC+157], v[vgprValuC+157], v231 // accum partials

s_waitcnt vmcnt(16)                                // wait C (interleaved) 16 = 72 - 55 - 1
v_add_f32 v[vgprValuC+158], v[vgprValuC+158], v232 // accum partials

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 72 - 56 - 1
v_add_f32 v[vgprValuC+159], v[vgprValuC+159], v233 // accum partials

s_waitcnt vmcnt(14)                                // wait C (interleaved) 14 = 72 - 57 - 1
v_add_f32 v[vgprValuC+160], v[vgprValuC+160], v234 // accum partials

s_waitcnt vmcnt(13)                                // wait C (interleaved) 13 = 72 - 58 - 1
v_add_f32 v[vgprValuC+161], v[vgprValuC+161], v235 // accum partials

s_waitcnt vmcnt(12)                                // wait C (interleaved) 12 = 72 - 59 - 1
v_add_f32 v[vgprValuC+162], v[vgprValuC+162], v236 // accum partials

s_waitcnt vmcnt(11)                                // wait C (interleaved) 11 = 72 - 60 - 1
v_add_f32 v[vgprValuC+163], v[vgprValuC+163], v237 // accum partials

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 72 - 61 - 1
v_add_f32 v[vgprValuC+164], v[vgprValuC+164], v238 // accum partials

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 72 - 62 - 1
v_add_f32 v[vgprValuC+165], v[vgprValuC+165], v239 // accum partials

s_waitcnt vmcnt(8)                                 // wait C (interleaved) 8 = 72 - 63 - 1
v_add_f32 v[vgprValuC+166], v[vgprValuC+166], v240 // accum partials

s_waitcnt vmcnt(7)                                 // wait C (interleaved) 7 = 72 - 64 - 1
v_add_f32 v[vgprValuC+167], v[vgprValuC+167], v241 // accum partials

s_waitcnt vmcnt(6)                                 // wait C (interleaved) 6 = 72 - 65 - 1
v_add_f32 v[vgprValuC+168], v[vgprValuC+168], v242 // accum partials

s_waitcnt vmcnt(5)                                 // wait C (interleaved) 5 = 72 - 66 - 1
v_add_f32 v[vgprValuC+169], v[vgprValuC+169], v243 // accum partials

s_waitcnt vmcnt(4)                                 // wait C (interleaved) 4 = 72 - 67 - 1
v_add_f32 v[vgprValuC+170], v[vgprValuC+170], v244 // accum partials

s_waitcnt vmcnt(3)                                 // wait C (interleaved) 3 = 72 - 68 - 1
v_add_f32 v[vgprValuC+171], v[vgprValuC+171], v245 // accum partials

s_waitcnt vmcnt(2)                                 // wait C (interleaved) 2 = 72 - 69 - 1
v_add_f32 v[vgprValuC+172], v[vgprValuC+172], v246 // accum partials

s_waitcnt vmcnt(1)                                 // wait C (interleaved) 1 = 72 - 70 - 1
v_add_f32 v[vgprValuC+173], v[vgprValuC+173], v247 // accum partials

s_waitcnt vmcnt(0)                                 // wait C (interleaved) 0 = 72 - 71 - 1
v_add_f32 v[vgprValuC+174], v[vgprValuC+174], v248 // accum partials
v_mov_b32 v[vgprValuC+0], v[vgprValuC+103]         // copy vreg[0] to MI out reg
v_mov_b32 v[vgprValuC+8], v[vgprValuC+104]         // copy vreg[1] to MI out reg
v_mov_b32 v[vgprValuC+16], v[vgprValuC+105]        // copy vreg[2] to MI out reg
v_mov_b32 v[vgprValuC+1], v[vgprValuC+106]         // copy vreg[3] to MI out reg
v_mov_b32 v[vgprValuC+9], v[vgprValuC+107]         // copy vreg[4] to MI out reg
v_mov_b32 v[vgprValuC+17], v[vgprValuC+108]        // copy vreg[5] to MI out reg
v_mov_b32 v[vgprValuC+2], v[vgprValuC+109]         // copy vreg[6] to MI out reg
v_mov_b32 v[vgprValuC+10], v[vgprValuC+110]        // copy vreg[7] to MI out reg
v_mov_b32 v[vgprValuC+18], v[vgprValuC+111]        // copy vreg[8] to MI out reg
v_mov_b32 v[vgprValuC+3], v[vgprValuC+112]         // copy vreg[9] to MI out reg
v_mov_b32 v[vgprValuC+11], v[vgprValuC+113]        // copy vreg[10] to MI out reg
v_mov_b32 v[vgprValuC+19], v[vgprValuC+114]        // copy vreg[11] to MI out reg
v_mov_b32 v[vgprValuC+4], v[vgprValuC+115]         // copy vreg[12] to MI out reg
v_mov_b32 v[vgprValuC+12], v[vgprValuC+116]        // copy vreg[13] to MI out reg
v_mov_b32 v[vgprValuC+20], v[vgprValuC+117]        // copy vreg[14] to MI out reg
v_mov_b32 v[vgprValuC+5], v[vgprValuC+118]         // copy vreg[15] to MI out reg
v_mov_b32 v[vgprValuC+13], v[vgprValuC+119]        // copy vreg[16] to MI out reg
v_mov_b32 v[vgprValuC+21], v[vgprValuC+120]        // copy vreg[17] to MI out reg
v_mov_b32 v[vgprValuC+6], v[vgprValuC+121]         // copy vreg[18] to MI out reg
v_mov_b32 v[vgprValuC+14], v[vgprValuC+122]        // copy vreg[19] to MI out reg
v_mov_b32 v[vgprValuC+22], v[vgprValuC+123]        // copy vreg[20] to MI out reg
v_mov_b32 v[vgprValuC+7], v[vgprValuC+124]         // copy vreg[21] to MI out reg
v_mov_b32 v[vgprValuC+15], v[vgprValuC+125]        // copy vreg[22] to MI out reg
v_mov_b32 v[vgprValuC+23], v[vgprValuC+126]        // copy vreg[23] to MI out reg
v_mov_b32 v[vgprValuC+24], v[vgprValuC+127]        // copy vreg[24] to MI out reg
v_mov_b32 v[vgprValuC+32], v[vgprValuC+128]        // copy vreg[25] to MI out reg
v_mov_b32 v[vgprValuC+40], v[vgprValuC+129]        // copy vreg[26] to MI out reg
v_mov_b32 v[vgprValuC+25], v[vgprValuC+130]        // copy vreg[27] to MI out reg
v_mov_b32 v[vgprValuC+33], v[vgprValuC+131]        // copy vreg[28] to MI out reg
v_mov_b32 v[vgprValuC+41], v[vgprValuC+132]        // copy vreg[29] to MI out reg
v_mov_b32 v[vgprValuC+26], v[vgprValuC+133]        // copy vreg[30] to MI out reg
v_mov_b32 v[vgprValuC+34], v[vgprValuC+134]        // copy vreg[31] to MI out reg
v_mov_b32 v[vgprValuC+42], v[vgprValuC+135]        // copy vreg[32] to MI out reg
v_mov_b32 v[vgprValuC+27], v[vgprValuC+136]        // copy vreg[33] to MI out reg
v_mov_b32 v[vgprValuC+35], v[vgprValuC+137]        // copy vreg[34] to MI out reg
v_mov_b32 v[vgprValuC+43], v[vgprValuC+138]        // copy vreg[35] to MI out reg
v_mov_b32 v[vgprValuC+28], v[vgprValuC+139]        // copy vreg[36] to MI out reg
v_mov_b32 v[vgprValuC+36], v[vgprValuC+140]        // copy vreg[37] to MI out reg
v_mov_b32 v[vgprValuC+44], v[vgprValuC+141]        // copy vreg[38] to MI out reg
v_mov_b32 v[vgprValuC+29], v[vgprValuC+142]        // copy vreg[39] to MI out reg
v_mov_b32 v[vgprValuC+37], v[vgprValuC+143]        // copy vreg[40] to MI out reg
v_mov_b32 v[vgprValuC+45], v[vgprValuC+144]        // copy vreg[41] to MI out reg
v_mov_b32 v[vgprValuC+30], v[vgprValuC+145]        // copy vreg[42] to MI out reg
v_mov_b32 v[vgprValuC+38], v[vgprValuC+146]        // copy vreg[43] to MI out reg
v_mov_b32 v[vgprValuC+46], v[vgprValuC+147]        // copy vreg[44] to MI out reg
v_mov_b32 v[vgprValuC+31], v[vgprValuC+148]        // copy vreg[45] to MI out reg
v_mov_b32 v[vgprValuC+39], v[vgprValuC+149]        // copy vreg[46] to MI out reg
v_mov_b32 v[vgprValuC+47], v[vgprValuC+150]        // copy vreg[47] to MI out reg
v_mov_b32 v[vgprValuC+48], v[vgprValuC+151]        // copy vreg[48] to MI out reg
v_mov_b32 v[vgprValuC+56], v[vgprValuC+152]        // copy vreg[49] to MI out reg
v_mov_b32 v[vgprValuC+64], v[vgprValuC+153]        // copy vreg[50] to MI out reg
v_mov_b32 v[vgprValuC+49], v[vgprValuC+154]        // copy vreg[51] to MI out reg
v_mov_b32 v[vgprValuC+57], v[vgprValuC+155]        // copy vreg[52] to MI out reg
v_mov_b32 v[vgprValuC+65], v[vgprValuC+156]        // copy vreg[53] to MI out reg
v_mov_b32 v[vgprValuC+50], v[vgprValuC+157]        // copy vreg[54] to MI out reg
v_mov_b32 v[vgprValuC+58], v[vgprValuC+158]        // copy vreg[55] to MI out reg
v_mov_b32 v[vgprValuC+66], v[vgprValuC+159]        // copy vreg[56] to MI out reg
v_mov_b32 v[vgprValuC+51], v[vgprValuC+160]        // copy vreg[57] to MI out reg
v_mov_b32 v[vgprValuC+59], v[vgprValuC+161]        // copy vreg[58] to MI out reg
v_mov_b32 v[vgprValuC+67], v[vgprValuC+162]        // copy vreg[59] to MI out reg
v_mov_b32 v[vgprValuC+52], v[vgprValuC+163]        // copy vreg[60] to MI out reg
v_mov_b32 v[vgprValuC+60], v[vgprValuC+164]        // copy vreg[61] to MI out reg
v_mov_b32 v[vgprValuC+68], v[vgprValuC+165]        // copy vreg[62] to MI out reg
v_mov_b32 v[vgprValuC+53], v[vgprValuC+166]        // copy vreg[63] to MI out reg
v_mov_b32 v[vgprValuC+61], v[vgprValuC+167]        // copy vreg[64] to MI out reg
v_mov_b32 v[vgprValuC+69], v[vgprValuC+168]        // copy vreg[65] to MI out reg
v_mov_b32 v[vgprValuC+54], v[vgprValuC+169]        // copy vreg[66] to MI out reg
v_mov_b32 v[vgprValuC+62], v[vgprValuC+170]        // copy vreg[67] to MI out reg
v_mov_b32 v[vgprValuC+70], v[vgprValuC+171]        // copy vreg[68] to MI out reg
v_mov_b32 v[vgprValuC+55], v[vgprValuC+172]        // copy vreg[69] to MI out reg
v_mov_b32 v[vgprValuC+63], v[vgprValuC+173]        // copy vreg[70] to MI out reg
v_mov_b32 v[vgprValuC+71], v[vgprValuC+174]        // copy vreg[71] to MI out reg
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_mul_i32 s11, s[sgprskTiles], s[sgprItersPerTile]
s_mul_i32 s53, s[sgprSKItersPerWG], s[sgprskGrid]
s_sub_u32 s11, s11, s53                            // skTiles * ItersPerTile - SKItersPerWG * skGrid
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
v_readfirstlane_b32 s53, v101                      // quotient
v_readfirstlane_b32 s53, v102                      // remainder
s_cmp_eq_u32 s53, 0                                // skGrid % skTiles == 0?
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
v_readfirstlane_b32 s53, v101                      // quotient
s_mov_b32 s11, s53                                 // F = skGrid / skTiles
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
v_readfirstlane_b32 s53, v101                      // quotient
v_readfirstlane_b32 s53, v102                      // remainder
s_mul_i32 s11, s11, s[sgprSKItersPerWG]            // F * SKItersPerWG
s_sub_u32 s11, s[sgprItersPerTile], s11            // remI = ItersPerTile - F*W
s_cmp_lt_u32 s53, s11                              // s < remI?
s_cselect_b32 s53, 1, 0                            // extra iter within tile
s_add_u32 s53, s[sgprSKItersPerWG], s53            // chunk = W + (s < remI)
s_branch label_SK_PeerDone                         // skip global peer
label_SK_PeerNoTiles:
label_SK_PeerGlobal:
s_add_u32 s53, s[sgprSKItersPerWG], 1              // Add extra iter
s_cmp_lt_u32 s7, s11                               // Check if next WG had an extra iteration
s_cselect_b32 s53, s53, s[sgprSKItersPerWG]        // Select correct number of iterations for next WG
label_SK_PeerDone:
s_add_u32 s8, s8, s53                              // next partial tile iteration
s_add_u32 s7, s7, 1                                // next partial tile index
s_cmp_lt_u32 s8, s[sgprItersPerTile]               // done loading partial tiles?
s_cbranch_scc1 label_SK_Fixup                      // Branch to continue fixup loop
label_SK_Store:
s_cmpk_eq_u32 s[sgprBeta], 0                       // Beta == 0
s_cbranch_scc0 label_GW_B1_GSU1                    // Branch if Beta is not zero

label_GW_B0_GSU1:
label_GW_B0_FD0_GSU1:

/* Edge/NonEdge store path check (M): Size % 96 > 0 -> Edge store; else -> NonEdge store */
s_mul_hi_u32 s71, s[sgprSizeI], 89478486           // STATIC_DIV: divisor=96
s_mul_i32 s70, s[sgprSizeI], 89478486              // tmp = dividend * magic
s_lshr_b64 s[70:71], s[70:71], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s69, s70                                 // quotient
s_mul_i32 s70, s69, 96                             // quotient*divisor
s_sub_u32 s68, s[sgprSizeI], s70                   // rReg = dividend - quotient*divisor
s_add_u32 s69, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s69                // wg0 >= nwg0-1 ?
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_GSU1_Else       // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 192 > 0 -> Edge store; else -> NonEdge store */
s_mul_hi_u32 s71, s[sgprSizeJ], 44739243           // STATIC_DIV: divisor=192
s_mul_i32 s70, s[sgprSizeJ], 44739243              // tmp = dividend * magic
s_lshr_b64 s[70:71], s[70:71], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s69, s70                                 // quotient
s_mul_i32 s70, s69, 192                            // quotient*divisor
s_sub_u32 s68, s[sgprSizeJ], s70                   // rReg = dividend - quotient*divisor
s_add_u32 s69, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s69                // wg1 >= nwg1-1
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_GSU1_Then       // jump if edges required
label_GW_B0_FD0_VW1_GSU1_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=150 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (8,2,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (9,2,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (10,2,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (11,2,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (12,2,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (13,2,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (14,2,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (15,2,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (16,2,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (17,2,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (18,2,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (19,2,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (20,2,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (21,2,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (22,2,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1); (23,2,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
/* (d1,vc1,d0,vc0)=(4,0,2,0) */
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
/* (d1,vc1,d0,vc0)=(5,0,2,0) */
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
/* (d1,vc1,d0,vc0)=(6,0,2,0) */
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
/* (d1,vc1,d0,vc0)=(7,0,2,0) */
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
/* (d1,vc1,d0,vc0)=(8,0,2,0) */
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
/* (d1,vc1,d0,vc0)=(9,0,1,0) */
/* (d1,vc1,d0,vc0)=(9,0,2,0) */
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
/* (d1,vc1,d0,vc0)=(10,0,1,0) */
/* (d1,vc1,d0,vc0)=(10,0,2,0) */
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
/* (d1,vc1,d0,vc0)=(11,0,1,0) */
/* (d1,vc1,d0,vc0)=(11,0,2,0) */
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
/* (d1,vc1,d0,vc0)=(12,0,2,0) */
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
/* (d1,vc1,d0,vc0)=(13,0,1,0) */
/* (d1,vc1,d0,vc0)=(13,0,2,0) */
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
/* (d1,vc1,d0,vc0)=(14,0,1,0) */
/* (d1,vc1,d0,vc0)=(14,0,2,0) */
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
/* (d1,vc1,d0,vc0)=(15,0,1,0) */
/* (d1,vc1,d0,vc0)=(15,0,2,0) */
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
/* (d1,vc1,d0,vc0)=(16,0,2,0) */
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
/* (d1,vc1,d0,vc0)=(17,0,1,0) */
/* (d1,vc1,d0,vc0)=(17,0,2,0) */
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
/* (d1,vc1,d0,vc0)=(18,0,1,0) */
/* (d1,vc1,d0,vc0)=(18,0,2,0) */
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
/* (d1,vc1,d0,vc0)=(19,0,1,0) */
/* (d1,vc1,d0,vc0)=(19,0,2,0) */
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
/* (d1,vc1,d0,vc0)=(20,0,2,0) */
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
/* (d1,vc1,d0,vc0)=(21,0,1,0) */
/* (d1,vc1,d0,vc0)=(21,0,2,0) */
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
/* (d1,vc1,d0,vc0)=(22,0,1,0) */
/* (d1,vc1,d0,vc0)=(22,0,2,0) */
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
/* (d1,vc1,d0,vc0)=(23,0,1,0) */
/* (d1,vc1,d0,vc0)=(23,0,2,0) */
v_add_lshl_u32 v101, v93, v90, 1                   // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=90, coord0Vgpr=90 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (5, 2, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (7, 2, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (8, 2, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (9, 2, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (10, 2, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (11, 2, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (12, 2, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (13, 2, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (14, 2, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (15, 2, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (16, 2, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (17, 2, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (18, 2, 0, 0), (19, 0, 0, 0), (19, 1, 0, 0), (19, 2, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0), (20, 2, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0), (21, 2, 0, 0), (22, 0, 0, 0), (22, 1, 0, 0), (22, 2, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0), (23, 2, 0, 0)] */
v_mul_f32 v[vgprValuC+103], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+104], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+105], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+106], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+107], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+108], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+109], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+110], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+111], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+112], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+128], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+129], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+130], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+131], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+132], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+133], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+134], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+135], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+48] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+152], s[sgprAlpha], v[vgprValuC+56] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+64] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+154], s[sgprAlpha], v[vgprValuC+49] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+57] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+65] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+50] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+158], s[sgprAlpha], v[vgprValuC+58] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+66] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+160], s[sgprAlpha], v[vgprValuC+51] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+59] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+162], s[sgprAlpha], v[vgprValuC+67] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+163], s[sgprAlpha], v[vgprValuC+52] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+164], s[sgprAlpha], v[vgprValuC+60] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+68] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+166], s[sgprAlpha], v[vgprValuC+53] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+167], s[sgprAlpha], v[vgprValuC+61] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+168], s[sgprAlpha], v[vgprValuC+69] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+169], s[sgprAlpha], v[vgprValuC+54] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+170], s[sgprAlpha], v[vgprValuC+62] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+171], s[sgprAlpha], v[vgprValuC+70] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+172], s[sgprAlpha], v[vgprValuC+55] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+173], s[sgprAlpha], v[vgprValuC+63] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+174], s[sgprAlpha], v[vgprValuC+71] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v98, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v99, 0x7fff0000                          // fp32 Nan
v_mov_b32 v100, 0x7fff                             // rounding bias for bfloat16
v_cmp_u_f32 s8, v[vgprValuC+103], v[vgprValuC+103] // check Nan
v_bfe_u32 v97, v[vgprValuC+103], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+103], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+103], v97, v99, s8
v_lshrrev_b32 v103, 16, v[vgprValuC+103]           // convert C to bf16
buffer_store_b16 v103, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+104], v[vgprValuC+104] // check Nan
v_bfe_u32 v97, v[vgprValuC+104], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+104], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+104], v97, v99, s8
v_lshrrev_b32 v104, 16, v[vgprValuC+104]           // convert C to bf16
buffer_store_b16 v104, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+105], v[vgprValuC+105] // check Nan
v_bfe_u32 v97, v[vgprValuC+105], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+105], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+105], v97, v99, s8
v_lshrrev_b32 v105, 16, v[vgprValuC+105]           // convert C to bf16
buffer_store_b16 v105, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+106], v[vgprValuC+106] // check Nan
v_bfe_u32 v97, v[vgprValuC+106], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+106], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+106], v97, v99, s8
v_lshrrev_b32 v106, 16, v[vgprValuC+106]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v106, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+107], v[vgprValuC+107] // check Nan
v_bfe_u32 v97, v[vgprValuC+107], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+107], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+107], v97, v99, s8
v_lshrrev_b32 v107, 16, v[vgprValuC+107]           // convert C to bf16
buffer_store_b16 v107, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+108], v[vgprValuC+108] // check Nan
v_bfe_u32 v97, v[vgprValuC+108], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+108], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+108], v97, v99, s8
v_lshrrev_b32 v108, 16, v[vgprValuC+108]           // convert C to bf16
buffer_store_b16 v108, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+109], v[vgprValuC+109] // check Nan
v_bfe_u32 v97, v[vgprValuC+109], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+109], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+109], v97, v99, s8
v_lshrrev_b32 v109, 16, v[vgprValuC+109]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v109, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+110], v[vgprValuC+110] // check Nan
v_bfe_u32 v97, v[vgprValuC+110], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+110], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+110], v97, v99, s8
v_lshrrev_b32 v110, 16, v[vgprValuC+110]           // convert C to bf16
buffer_store_b16 v110, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+111], v[vgprValuC+111] // check Nan
v_bfe_u32 v97, v[vgprValuC+111], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+111], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+111], v97, v99, s8
v_lshrrev_b32 v111, 16, v[vgprValuC+111]           // convert C to bf16
buffer_store_b16 v111, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+112], v[vgprValuC+112] // check Nan
v_bfe_u32 v97, v[vgprValuC+112], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+112], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+112], v97, v99, s8
v_lshrrev_b32 v112, 16, v[vgprValuC+112]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v112, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+113], v[vgprValuC+113] // check Nan
v_bfe_u32 v97, v[vgprValuC+113], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+113], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+113], v97, v99, s8
v_lshrrev_b32 v113, 16, v[vgprValuC+113]           // convert C to bf16
buffer_store_b16 v113, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+114], v[vgprValuC+114] // check Nan
v_bfe_u32 v97, v[vgprValuC+114], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+114], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+114], v97, v99, s8
v_lshrrev_b32 v114, 16, v[vgprValuC+114]           // convert C to bf16
buffer_store_b16 v114, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+115], v[vgprValuC+115] // check Nan
v_bfe_u32 v97, v[vgprValuC+115], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+115], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+115], v97, v99, s8
v_lshrrev_b32 v115, 16, v[vgprValuC+115]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v115, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+116], v[vgprValuC+116] // check Nan
v_bfe_u32 v97, v[vgprValuC+116], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+116], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+116], v97, v99, s8
v_lshrrev_b32 v116, 16, v[vgprValuC+116]           // convert C to bf16
buffer_store_b16 v116, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+117], v[vgprValuC+117] // check Nan
v_bfe_u32 v97, v[vgprValuC+117], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+117], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+117], v97, v99, s8
v_lshrrev_b32 v117, 16, v[vgprValuC+117]           // convert C to bf16
buffer_store_b16 v117, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+118], v[vgprValuC+118] // check Nan
v_bfe_u32 v97, v[vgprValuC+118], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+118], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+118], v97, v99, s8
v_lshrrev_b32 v118, 16, v[vgprValuC+118]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v118, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+119], v[vgprValuC+119] // check Nan
v_bfe_u32 v97, v[vgprValuC+119], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+119], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+119], v97, v99, s8
v_lshrrev_b32 v119, 16, v[vgprValuC+119]           // convert C to bf16
buffer_store_b16 v119, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+120], v[vgprValuC+120] // check Nan
v_bfe_u32 v97, v[vgprValuC+120], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+120], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+120], v97, v99, s8
v_lshrrev_b32 v120, 16, v[vgprValuC+120]           // convert C to bf16
buffer_store_b16 v120, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+121], v[vgprValuC+121] // check Nan
v_bfe_u32 v97, v[vgprValuC+121], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+121], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+121], v97, v99, s8
v_lshrrev_b32 v121, 16, v[vgprValuC+121]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v121, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+122], v[vgprValuC+122] // check Nan
v_bfe_u32 v97, v[vgprValuC+122], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+122], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+122], v97, v99, s8
v_lshrrev_b32 v122, 16, v[vgprValuC+122]           // convert C to bf16
buffer_store_b16 v122, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+123], v[vgprValuC+123] // check Nan
v_bfe_u32 v97, v[vgprValuC+123], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+123], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+123], v97, v99, s8
v_lshrrev_b32 v123, 16, v[vgprValuC+123]           // convert C to bf16
buffer_store_b16 v123, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+124], v[vgprValuC+124] // check Nan
v_bfe_u32 v97, v[vgprValuC+124], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+124], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+124], v97, v99, s8
v_lshrrev_b32 v124, 16, v[vgprValuC+124]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v124, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+125], v[vgprValuC+125] // check Nan
v_bfe_u32 v97, v[vgprValuC+125], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+125], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+125], v97, v99, s8
v_lshrrev_b32 v125, 16, v[vgprValuC+125]           // convert C to bf16
buffer_store_b16 v125, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+126], v[vgprValuC+126] // check Nan
v_bfe_u32 v97, v[vgprValuC+126], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+126], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+126], v97, v99, s8
v_lshrrev_b32 v126, 16, v[vgprValuC+126]           // convert C to bf16
buffer_store_b16 v126, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+127], v[vgprValuC+127] // check Nan
v_bfe_u32 v97, v[vgprValuC+127], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+127], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+127], v97, v99, s8
v_lshrrev_b32 v127, 16, v[vgprValuC+127]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 100                // scale StrideD *= numRows(50) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(50): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(50): gra SRD += inc(upper)
buffer_store_b16 v127, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+128], v[vgprValuC+128] // check Nan
v_bfe_u32 v97, v[vgprValuC+128], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+128], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+128], v97, v99, s8
v_lshrrev_b32 v128, 16, v[vgprValuC+128]           // convert C to bf16
buffer_store_b16 v128, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+129], v[vgprValuC+129] // check Nan
v_bfe_u32 v97, v[vgprValuC+129], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+129], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+129], v97, v99, s8
v_lshrrev_b32 v129, 16, v[vgprValuC+129]           // convert C to bf16
buffer_store_b16 v129, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+130], v[vgprValuC+130] // check Nan
v_bfe_u32 v97, v[vgprValuC+130], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+130], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+130], v97, v99, s8
v_lshrrev_b32 v130, 16, v[vgprValuC+130]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v130, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+131], v[vgprValuC+131] // check Nan
v_bfe_u32 v97, v[vgprValuC+131], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+131], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+131], v97, v99, s8
v_lshrrev_b32 v131, 16, v[vgprValuC+131]           // convert C to bf16
buffer_store_b16 v131, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+132], v[vgprValuC+132] // check Nan
v_bfe_u32 v97, v[vgprValuC+132], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+132], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+132], v97, v99, s8
v_lshrrev_b32 v132, 16, v[vgprValuC+132]           // convert C to bf16
buffer_store_b16 v132, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+133], v[vgprValuC+133] // check Nan
v_bfe_u32 v97, v[vgprValuC+133], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+133], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+133], v97, v99, s8
v_lshrrev_b32 v133, 16, v[vgprValuC+133]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v133, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+134], v[vgprValuC+134] // check Nan
v_bfe_u32 v97, v[vgprValuC+134], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+134], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+134], v97, v99, s8
v_lshrrev_b32 v134, 16, v[vgprValuC+134]           // convert C to bf16
buffer_store_b16 v134, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+135], v[vgprValuC+135] // check Nan
v_bfe_u32 v97, v[vgprValuC+135], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+135], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+135], v97, v99, s8
v_lshrrev_b32 v135, 16, v[vgprValuC+135]           // convert C to bf16
buffer_store_b16 v135, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+136], v[vgprValuC+136] // check Nan
v_bfe_u32 v97, v[vgprValuC+136], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+136], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+136], v97, v99, s8
v_lshrrev_b32 v136, 16, v[vgprValuC+136]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v136, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+137], v[vgprValuC+137] // check Nan
v_bfe_u32 v97, v[vgprValuC+137], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+137], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+137], v97, v99, s8
v_lshrrev_b32 v137, 16, v[vgprValuC+137]           // convert C to bf16
buffer_store_b16 v137, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+138], v[vgprValuC+138] // check Nan
v_bfe_u32 v97, v[vgprValuC+138], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+138], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+138], v97, v99, s8
v_lshrrev_b32 v138, 16, v[vgprValuC+138]           // convert C to bf16
buffer_store_b16 v138, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+139], v[vgprValuC+139] // check Nan
v_bfe_u32 v97, v[vgprValuC+139], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+139], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+139], v97, v99, s8
v_lshrrev_b32 v139, 16, v[vgprValuC+139]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v139, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+140], v[vgprValuC+140] // check Nan
v_bfe_u32 v97, v[vgprValuC+140], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+140], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+140], v97, v99, s8
v_lshrrev_b32 v140, 16, v[vgprValuC+140]           // convert C to bf16
buffer_store_b16 v140, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+141], v[vgprValuC+141] // check Nan
v_bfe_u32 v97, v[vgprValuC+141], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+141], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+141], v97, v99, s8
v_lshrrev_b32 v141, 16, v[vgprValuC+141]           // convert C to bf16
buffer_store_b16 v141, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+142], v[vgprValuC+142] // check Nan
v_bfe_u32 v97, v[vgprValuC+142], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+142], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+142], v97, v99, s8
v_lshrrev_b32 v142, 16, v[vgprValuC+142]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v142, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+143], v[vgprValuC+143] // check Nan
v_bfe_u32 v97, v[vgprValuC+143], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+143], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+143], v97, v99, s8
v_lshrrev_b32 v143, 16, v[vgprValuC+143]           // convert C to bf16
buffer_store_b16 v143, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+144], v[vgprValuC+144] // check Nan
v_bfe_u32 v97, v[vgprValuC+144], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+144], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+144], v97, v99, s8
v_lshrrev_b32 v144, 16, v[vgprValuC+144]           // convert C to bf16
buffer_store_b16 v144, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+145], v[vgprValuC+145] // check Nan
v_bfe_u32 v97, v[vgprValuC+145], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+145], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+145], v97, v99, s8
v_lshrrev_b32 v145, 16, v[vgprValuC+145]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v145, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+146], v[vgprValuC+146] // check Nan
v_bfe_u32 v97, v[vgprValuC+146], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+146], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+146], v97, v99, s8
v_lshrrev_b32 v146, 16, v[vgprValuC+146]           // convert C to bf16
buffer_store_b16 v146, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+147], v[vgprValuC+147] // check Nan
v_bfe_u32 v97, v[vgprValuC+147], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+147], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+147], v97, v99, s8
v_lshrrev_b32 v147, 16, v[vgprValuC+147]           // convert C to bf16
buffer_store_b16 v147, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+148], v[vgprValuC+148] // check Nan
v_bfe_u32 v97, v[vgprValuC+148], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+148], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+148], v97, v99, s8
v_lshrrev_b32 v148, 16, v[vgprValuC+148]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v148, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+149], v[vgprValuC+149] // check Nan
v_bfe_u32 v97, v[vgprValuC+149], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+149], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+149], v97, v99, s8
v_lshrrev_b32 v149, 16, v[vgprValuC+149]           // convert C to bf16
buffer_store_b16 v149, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+150], v[vgprValuC+150] // check Nan
v_bfe_u32 v97, v[vgprValuC+150], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+150], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+150], v97, v99, s8
v_lshrrev_b32 v150, 16, v[vgprValuC+150]           // convert C to bf16
buffer_store_b16 v150, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+151], v[vgprValuC+151] // check Nan
v_bfe_u32 v97, v[vgprValuC+151], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+151], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+151], v97, v99, s8
v_lshrrev_b32 v151, 16, v[vgprValuC+151]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 100                // scale StrideD *= numRows(50) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(50): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(50): gra SRD += inc(upper)
buffer_store_b16 v151, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+152], v[vgprValuC+152] // check Nan
v_bfe_u32 v97, v[vgprValuC+152], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+152], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+152], v97, v99, s8
v_lshrrev_b32 v152, 16, v[vgprValuC+152]           // convert C to bf16
buffer_store_b16 v152, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+153], v[vgprValuC+153] // check Nan
v_bfe_u32 v97, v[vgprValuC+153], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+153], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+153], v97, v99, s8
v_lshrrev_b32 v153, 16, v[vgprValuC+153]           // convert C to bf16
buffer_store_b16 v153, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+154], v[vgprValuC+154] // check Nan
v_bfe_u32 v97, v[vgprValuC+154], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+154], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+154], v97, v99, s8
v_lshrrev_b32 v154, 16, v[vgprValuC+154]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v154, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+155], v[vgprValuC+155] // check Nan
v_bfe_u32 v97, v[vgprValuC+155], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+155], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+155], v97, v99, s8
v_lshrrev_b32 v155, 16, v[vgprValuC+155]           // convert C to bf16
buffer_store_b16 v155, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+156], v[vgprValuC+156] // check Nan
v_bfe_u32 v97, v[vgprValuC+156], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+156], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+156], v97, v99, s8
v_lshrrev_b32 v156, 16, v[vgprValuC+156]           // convert C to bf16
buffer_store_b16 v156, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+157], v[vgprValuC+157] // check Nan
v_bfe_u32 v97, v[vgprValuC+157], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+157], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+157], v97, v99, s8
v_lshrrev_b32 v157, 16, v[vgprValuC+157]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v157, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+158], v[vgprValuC+158] // check Nan
v_bfe_u32 v97, v[vgprValuC+158], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+158], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+158], v97, v99, s8
v_lshrrev_b32 v158, 16, v[vgprValuC+158]           // convert C to bf16
buffer_store_b16 v158, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+159], v[vgprValuC+159] // check Nan
v_bfe_u32 v97, v[vgprValuC+159], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+159], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+159], v97, v99, s8
v_lshrrev_b32 v159, 16, v[vgprValuC+159]           // convert C to bf16
buffer_store_b16 v159, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+160], v[vgprValuC+160] // check Nan
v_bfe_u32 v97, v[vgprValuC+160], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+160], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+160], v97, v99, s8
v_lshrrev_b32 v160, 16, v[vgprValuC+160]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v160, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+161], v[vgprValuC+161] // check Nan
v_bfe_u32 v97, v[vgprValuC+161], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+161], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+161], v97, v99, s8
v_lshrrev_b32 v161, 16, v[vgprValuC+161]           // convert C to bf16
buffer_store_b16 v161, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+162], v[vgprValuC+162] // check Nan
v_bfe_u32 v97, v[vgprValuC+162], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+162], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+162], v97, v99, s8
v_lshrrev_b32 v162, 16, v[vgprValuC+162]           // convert C to bf16
buffer_store_b16 v162, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+163], v[vgprValuC+163] // check Nan
v_bfe_u32 v97, v[vgprValuC+163], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+163], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+163], v97, v99, s8
v_lshrrev_b32 v163, 16, v[vgprValuC+163]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v163, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+164], v[vgprValuC+164] // check Nan
v_bfe_u32 v97, v[vgprValuC+164], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+164], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+164], v97, v99, s8
v_lshrrev_b32 v164, 16, v[vgprValuC+164]           // convert C to bf16
buffer_store_b16 v164, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+165], v[vgprValuC+165] // check Nan
v_bfe_u32 v97, v[vgprValuC+165], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+165], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+165], v97, v99, s8
v_lshrrev_b32 v165, 16, v[vgprValuC+165]           // convert C to bf16
buffer_store_b16 v165, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+166], v[vgprValuC+166] // check Nan
v_bfe_u32 v97, v[vgprValuC+166], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+166], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+166], v97, v99, s8
v_lshrrev_b32 v166, 16, v[vgprValuC+166]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v166, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+167], v[vgprValuC+167] // check Nan
v_bfe_u32 v97, v[vgprValuC+167], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+167], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+167], v97, v99, s8
v_lshrrev_b32 v167, 16, v[vgprValuC+167]           // convert C to bf16
buffer_store_b16 v167, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+168], v[vgprValuC+168] // check Nan
v_bfe_u32 v97, v[vgprValuC+168], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+168], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+168], v97, v99, s8
v_lshrrev_b32 v168, 16, v[vgprValuC+168]           // convert C to bf16
buffer_store_b16 v168, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+169], v[vgprValuC+169] // check Nan
v_bfe_u32 v97, v[vgprValuC+169], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+169], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+169], v97, v99, s8
v_lshrrev_b32 v169, 16, v[vgprValuC+169]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v169, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+170], v[vgprValuC+170] // check Nan
v_bfe_u32 v97, v[vgprValuC+170], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+170], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+170], v97, v99, s8
v_lshrrev_b32 v170, 16, v[vgprValuC+170]           // convert C to bf16
buffer_store_b16 v170, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+171], v[vgprValuC+171] // check Nan
v_bfe_u32 v97, v[vgprValuC+171], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+171], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+171], v97, v99, s8
v_lshrrev_b32 v171, 16, v[vgprValuC+171]           // convert C to bf16
buffer_store_b16 v171, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+172], v[vgprValuC+172] // check Nan
v_bfe_u32 v97, v[vgprValuC+172], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+172], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+172], v97, v99, s8
v_lshrrev_b32 v172, 16, v[vgprValuC+172]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v172, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+173], v[vgprValuC+173] // check Nan
v_bfe_u32 v97, v[vgprValuC+173], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+173], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+173], v97, v99, s8
v_lshrrev_b32 v173, 16, v[vgprValuC+173]           // convert C to bf16
buffer_store_b16 v173, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+174], v[vgprValuC+174] // check Nan
v_bfe_u32 v97, v[vgprValuC+174], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+174], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+174], v97, v99, s8
v_lshrrev_b32 v174, 16, v[vgprValuC+174]           // convert C to bf16
buffer_store_b16 v174, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW1_GSU1_NonEdgeEnd:
label_GW_B0_FD0_VW1_GSU1_Else:
label_GW_B0_FD0_VW1_GSU1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=76 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (8,2,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (9,2,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (10,2,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (11,2,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (12,2,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (13,2,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (14,2,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (15,2,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (16,2,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (17,2,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (18,2,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (19,2,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (20,2,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (21,2,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (22,2,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1); (23,2,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v96, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v173, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v173, v96, v173, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v174, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v174, v96, v174, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v175, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v175, v96, v175, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v176, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v176, v96, v176, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v177, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v177, v96, v177, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v178, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v178, v96, v178, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v179, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v179, v96, v179, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v180, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v180, v96, v180, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v181, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v181, v96, v181, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v182, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v182, v96, v182, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v183, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v183, v96, v183, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v184, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v184, v96, v184, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v185, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v185, v96, v185, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v186, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v186, v96, v186, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v187, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v187, v96, v187, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v188, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v188, v96, v188, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v189, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v189, v96, v189, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v190, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v190, v96, v190, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v191, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v191, v96, v191, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v192, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v192, v96, v192, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v193, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v193, v96, v193, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v194, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v194, v96, v194, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v195, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v195, v96, v195, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v196, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v196, v96, v196, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 50                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 50                // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 50                // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v197, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v197, v96, v197, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v198, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v198, v96, v198, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v199, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v199, v96, v199, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v200, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v200, v96, v200, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v201, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v201, v96, v201, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v202, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v202, v96, v202, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v203, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v203, v96, v203, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v204, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v204, v96, v204, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v205, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v205, v96, v205, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v206, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v206, v96, v206, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v207, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v207, v96, v207, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v208, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v208, v96, v208, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v209, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v209, v96, v209, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v210, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v210, v96, v210, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v211, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v211, v96, v211, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v212, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v212, v96, v212, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v213, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v213, v96, v213, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v214, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v214, v96, v214, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v215, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v215, v96, v215, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v216, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v216, v96, v216, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v217, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v217, v96, v217, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v218, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v218, v96, v218, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v219, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v219, v96, v219, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v220, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v220, v96, v220, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 50                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 50                // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 50                // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v221, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v221, v96, v221, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v222, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v222, v96, v222, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v223, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v223, v96, v223, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v226, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v226, v96, v226, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v227, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v227, v96, v227, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v228, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v228, v96, v228, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v229, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v229, v96, v229, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v230, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v230, v96, v230, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v231, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v231, v96, v231, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v232, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v232, v96, v232, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v233, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v233, v96, v233, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v234, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v234, v96, v234, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v235, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v235, v96, v235, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v236, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v236, v96, v236, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v237, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v237, v96, v237, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v238, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v238, v96, v238, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v239, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v239, v96, v239, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v240, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v240, v96, v240, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v241, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v241, v96, v241, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v242, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v242, v96, v242, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v243, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v243, v96, v243, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v244, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v244, v96, v244, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v245, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v245, v96, v245, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v246, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v246, v96, v246, s70                 // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (5, 2, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (7, 2, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (8, 2, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (9, 2, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (10, 2, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (11, 2, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (12, 2, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (13, 2, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (14, 2, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (15, 2, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (16, 2, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (17, 2, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (18, 2, 0, 0), (19, 0, 0, 0), (19, 1, 0, 0), (19, 2, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0), (20, 2, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0), (21, 2, 0, 0), (22, 0, 0, 0), (22, 1, 0, 0), (22, 2, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0), (23, 2, 0, 0)] */
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+102], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+103], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+104], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+105], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+106], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+107], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+108], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+109], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+110], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+111], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+112], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+128], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+129], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+130], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+131], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+132], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+133], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+134], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+135], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+48] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+56] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+64] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+152], s[sgprAlpha], v[vgprValuC+49] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+57] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+154], s[sgprAlpha], v[vgprValuC+65] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+50] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+58] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+66] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+158], s[sgprAlpha], v[vgprValuC+51] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+59] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+160], s[sgprAlpha], v[vgprValuC+67] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+52] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+162], s[sgprAlpha], v[vgprValuC+60] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+163], s[sgprAlpha], v[vgprValuC+68] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+164], s[sgprAlpha], v[vgprValuC+53] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+61] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+166], s[sgprAlpha], v[vgprValuC+69] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+167], s[sgprAlpha], v[vgprValuC+54] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+168], s[sgprAlpha], v[vgprValuC+62] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+169], s[sgprAlpha], v[vgprValuC+70] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+170], s[sgprAlpha], v[vgprValuC+55] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+171], s[sgprAlpha], v[vgprValuC+63] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+172], s[sgprAlpha], v[vgprValuC+71] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v98, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v99, 0x7fff0000                          // fp32 Nan
v_mov_b32 v100, 0x7fff                             // rounding bias for bfloat16
v_cmp_u_f32 s68, v[vgprValuC+101], v[vgprValuC+101] // check Nan
v_bfe_u32 v97, v[vgprValuC+101], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+101], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+101], v97, v99, s68
v_lshrrev_b32 v101, 16, v[vgprValuC+101]           // convert C to bf16
buffer_store_b16 v101, v173, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+102], v[vgprValuC+102] // check Nan
v_bfe_u32 v97, v[vgprValuC+102], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+102], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+102], v97, v99, s68
v_lshrrev_b32 v102, 16, v[vgprValuC+102]           // convert C to bf16
buffer_store_b16 v102, v174, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+103], v[vgprValuC+103] // check Nan
v_bfe_u32 v97, v[vgprValuC+103], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+103], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+103], v97, v99, s68
v_lshrrev_b32 v103, 16, v[vgprValuC+103]           // convert C to bf16
buffer_store_b16 v103, v175, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+104], v[vgprValuC+104] // check Nan
v_bfe_u32 v97, v[vgprValuC+104], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+104], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+104], v97, v99, s68
v_lshrrev_b32 v104, 16, v[vgprValuC+104]           // convert C to bf16
buffer_store_b16 v104, v176, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+105], v[vgprValuC+105] // check Nan
v_bfe_u32 v97, v[vgprValuC+105], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+105], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+105], v97, v99, s68
v_lshrrev_b32 v105, 16, v[vgprValuC+105]           // convert C to bf16
buffer_store_b16 v105, v177, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+106], v[vgprValuC+106] // check Nan
v_bfe_u32 v97, v[vgprValuC+106], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+106], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+106], v97, v99, s68
v_lshrrev_b32 v106, 16, v[vgprValuC+106]           // convert C to bf16
buffer_store_b16 v106, v178, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+107], v[vgprValuC+107] // check Nan
v_bfe_u32 v97, v[vgprValuC+107], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+107], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+107], v97, v99, s68
v_lshrrev_b32 v107, 16, v[vgprValuC+107]           // convert C to bf16
buffer_store_b16 v107, v179, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+108], v[vgprValuC+108] // check Nan
v_bfe_u32 v97, v[vgprValuC+108], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+108], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+108], v97, v99, s68
v_lshrrev_b32 v108, 16, v[vgprValuC+108]           // convert C to bf16
buffer_store_b16 v108, v180, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+109], v[vgprValuC+109] // check Nan
v_bfe_u32 v97, v[vgprValuC+109], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+109], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+109], v97, v99, s68
v_lshrrev_b32 v109, 16, v[vgprValuC+109]           // convert C to bf16
buffer_store_b16 v109, v181, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+110], v[vgprValuC+110] // check Nan
v_bfe_u32 v97, v[vgprValuC+110], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+110], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+110], v97, v99, s68
v_lshrrev_b32 v110, 16, v[vgprValuC+110]           // convert C to bf16
buffer_store_b16 v110, v182, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+111], v[vgprValuC+111] // check Nan
v_bfe_u32 v97, v[vgprValuC+111], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+111], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+111], v97, v99, s68
v_lshrrev_b32 v111, 16, v[vgprValuC+111]           // convert C to bf16
buffer_store_b16 v111, v183, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+112], v[vgprValuC+112] // check Nan
v_bfe_u32 v97, v[vgprValuC+112], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+112], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+112], v97, v99, s68
v_lshrrev_b32 v112, 16, v[vgprValuC+112]           // convert C to bf16
buffer_store_b16 v112, v184, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+113], v[vgprValuC+113] // check Nan
v_bfe_u32 v97, v[vgprValuC+113], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+113], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+113], v97, v99, s68
v_lshrrev_b32 v113, 16, v[vgprValuC+113]           // convert C to bf16
buffer_store_b16 v113, v185, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+114], v[vgprValuC+114] // check Nan
v_bfe_u32 v97, v[vgprValuC+114], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+114], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+114], v97, v99, s68
v_lshrrev_b32 v114, 16, v[vgprValuC+114]           // convert C to bf16
buffer_store_b16 v114, v186, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+115], v[vgprValuC+115] // check Nan
v_bfe_u32 v97, v[vgprValuC+115], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+115], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+115], v97, v99, s68
v_lshrrev_b32 v115, 16, v[vgprValuC+115]           // convert C to bf16
buffer_store_b16 v115, v187, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+116], v[vgprValuC+116] // check Nan
v_bfe_u32 v97, v[vgprValuC+116], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+116], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+116], v97, v99, s68
v_lshrrev_b32 v116, 16, v[vgprValuC+116]           // convert C to bf16
buffer_store_b16 v116, v188, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+117], v[vgprValuC+117] // check Nan
v_bfe_u32 v97, v[vgprValuC+117], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+117], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+117], v97, v99, s68
v_lshrrev_b32 v117, 16, v[vgprValuC+117]           // convert C to bf16
buffer_store_b16 v117, v189, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+118], v[vgprValuC+118] // check Nan
v_bfe_u32 v97, v[vgprValuC+118], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+118], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+118], v97, v99, s68
v_lshrrev_b32 v118, 16, v[vgprValuC+118]           // convert C to bf16
buffer_store_b16 v118, v190, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+119], v[vgprValuC+119] // check Nan
v_bfe_u32 v97, v[vgprValuC+119], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+119], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+119], v97, v99, s68
v_lshrrev_b32 v119, 16, v[vgprValuC+119]           // convert C to bf16
buffer_store_b16 v119, v191, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+120], v[vgprValuC+120] // check Nan
v_bfe_u32 v97, v[vgprValuC+120], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+120], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+120], v97, v99, s68
v_lshrrev_b32 v120, 16, v[vgprValuC+120]           // convert C to bf16
buffer_store_b16 v120, v192, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+121], v[vgprValuC+121] // check Nan
v_bfe_u32 v97, v[vgprValuC+121], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+121], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+121], v97, v99, s68
v_lshrrev_b32 v121, 16, v[vgprValuC+121]           // convert C to bf16
buffer_store_b16 v121, v193, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+122], v[vgprValuC+122] // check Nan
v_bfe_u32 v97, v[vgprValuC+122], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+122], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+122], v97, v99, s68
v_lshrrev_b32 v122, 16, v[vgprValuC+122]           // convert C to bf16
buffer_store_b16 v122, v194, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+123], v[vgprValuC+123] // check Nan
v_bfe_u32 v97, v[vgprValuC+123], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+123], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+123], v97, v99, s68
v_lshrrev_b32 v123, 16, v[vgprValuC+123]           // convert C to bf16
buffer_store_b16 v123, v195, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+124], v[vgprValuC+124] // check Nan
v_bfe_u32 v97, v[vgprValuC+124], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+124], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+124], v97, v99, s68
v_lshrrev_b32 v124, 16, v[vgprValuC+124]           // convert C to bf16
buffer_store_b16 v124, v196, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+125], v[vgprValuC+125] // check Nan
v_bfe_u32 v97, v[vgprValuC+125], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+125], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+125], v97, v99, s68
v_lshrrev_b32 v125, 16, v[vgprValuC+125]           // convert C to bf16
buffer_store_b16 v125, v197, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+126], v[vgprValuC+126] // check Nan
v_bfe_u32 v97, v[vgprValuC+126], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+126], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+126], v97, v99, s68
v_lshrrev_b32 v126, 16, v[vgprValuC+126]           // convert C to bf16
buffer_store_b16 v126, v198, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+127], v[vgprValuC+127] // check Nan
v_bfe_u32 v97, v[vgprValuC+127], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+127], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+127], v97, v99, s68
v_lshrrev_b32 v127, 16, v[vgprValuC+127]           // convert C to bf16
buffer_store_b16 v127, v199, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+128], v[vgprValuC+128] // check Nan
v_bfe_u32 v97, v[vgprValuC+128], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+128], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+128], v97, v99, s68
v_lshrrev_b32 v128, 16, v[vgprValuC+128]           // convert C to bf16
buffer_store_b16 v128, v200, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+129], v[vgprValuC+129] // check Nan
v_bfe_u32 v97, v[vgprValuC+129], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+129], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+129], v97, v99, s68
v_lshrrev_b32 v129, 16, v[vgprValuC+129]           // convert C to bf16
buffer_store_b16 v129, v201, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+130], v[vgprValuC+130] // check Nan
v_bfe_u32 v97, v[vgprValuC+130], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+130], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+130], v97, v99, s68
v_lshrrev_b32 v130, 16, v[vgprValuC+130]           // convert C to bf16
buffer_store_b16 v130, v202, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+131], v[vgprValuC+131] // check Nan
v_bfe_u32 v97, v[vgprValuC+131], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+131], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+131], v97, v99, s68
v_lshrrev_b32 v131, 16, v[vgprValuC+131]           // convert C to bf16
buffer_store_b16 v131, v203, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+132], v[vgprValuC+132] // check Nan
v_bfe_u32 v97, v[vgprValuC+132], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+132], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+132], v97, v99, s68
v_lshrrev_b32 v132, 16, v[vgprValuC+132]           // convert C to bf16
buffer_store_b16 v132, v204, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+133], v[vgprValuC+133] // check Nan
v_bfe_u32 v97, v[vgprValuC+133], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+133], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+133], v97, v99, s68
v_lshrrev_b32 v133, 16, v[vgprValuC+133]           // convert C to bf16
buffer_store_b16 v133, v205, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+134], v[vgprValuC+134] // check Nan
v_bfe_u32 v97, v[vgprValuC+134], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+134], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+134], v97, v99, s68
v_lshrrev_b32 v134, 16, v[vgprValuC+134]           // convert C to bf16
buffer_store_b16 v134, v206, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+135], v[vgprValuC+135] // check Nan
v_bfe_u32 v97, v[vgprValuC+135], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+135], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+135], v97, v99, s68
v_lshrrev_b32 v135, 16, v[vgprValuC+135]           // convert C to bf16
buffer_store_b16 v135, v207, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+136], v[vgprValuC+136] // check Nan
v_bfe_u32 v97, v[vgprValuC+136], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+136], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+136], v97, v99, s68
v_lshrrev_b32 v136, 16, v[vgprValuC+136]           // convert C to bf16
buffer_store_b16 v136, v208, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+137], v[vgprValuC+137] // check Nan
v_bfe_u32 v97, v[vgprValuC+137], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+137], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+137], v97, v99, s68
v_lshrrev_b32 v137, 16, v[vgprValuC+137]           // convert C to bf16
buffer_store_b16 v137, v209, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+138], v[vgprValuC+138] // check Nan
v_bfe_u32 v97, v[vgprValuC+138], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+138], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+138], v97, v99, s68
v_lshrrev_b32 v138, 16, v[vgprValuC+138]           // convert C to bf16
buffer_store_b16 v138, v210, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+139], v[vgprValuC+139] // check Nan
v_bfe_u32 v97, v[vgprValuC+139], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+139], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+139], v97, v99, s68
v_lshrrev_b32 v139, 16, v[vgprValuC+139]           // convert C to bf16
buffer_store_b16 v139, v211, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+140], v[vgprValuC+140] // check Nan
v_bfe_u32 v97, v[vgprValuC+140], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+140], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+140], v97, v99, s68
v_lshrrev_b32 v140, 16, v[vgprValuC+140]           // convert C to bf16
buffer_store_b16 v140, v212, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+141], v[vgprValuC+141] // check Nan
v_bfe_u32 v97, v[vgprValuC+141], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+141], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+141], v97, v99, s68
v_lshrrev_b32 v141, 16, v[vgprValuC+141]           // convert C to bf16
buffer_store_b16 v141, v213, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+142], v[vgprValuC+142] // check Nan
v_bfe_u32 v97, v[vgprValuC+142], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+142], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+142], v97, v99, s68
v_lshrrev_b32 v142, 16, v[vgprValuC+142]           // convert C to bf16
buffer_store_b16 v142, v214, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+143], v[vgprValuC+143] // check Nan
v_bfe_u32 v97, v[vgprValuC+143], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+143], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+143], v97, v99, s68
v_lshrrev_b32 v143, 16, v[vgprValuC+143]           // convert C to bf16
buffer_store_b16 v143, v215, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+144], v[vgprValuC+144] // check Nan
v_bfe_u32 v97, v[vgprValuC+144], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+144], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+144], v97, v99, s68
v_lshrrev_b32 v144, 16, v[vgprValuC+144]           // convert C to bf16
buffer_store_b16 v144, v216, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+145], v[vgprValuC+145] // check Nan
v_bfe_u32 v97, v[vgprValuC+145], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+145], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+145], v97, v99, s68
v_lshrrev_b32 v145, 16, v[vgprValuC+145]           // convert C to bf16
buffer_store_b16 v145, v217, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+146], v[vgprValuC+146] // check Nan
v_bfe_u32 v97, v[vgprValuC+146], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+146], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+146], v97, v99, s68
v_lshrrev_b32 v146, 16, v[vgprValuC+146]           // convert C to bf16
buffer_store_b16 v146, v218, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+147], v[vgprValuC+147] // check Nan
v_bfe_u32 v97, v[vgprValuC+147], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+147], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+147], v97, v99, s68
v_lshrrev_b32 v147, 16, v[vgprValuC+147]           // convert C to bf16
buffer_store_b16 v147, v219, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+148], v[vgprValuC+148] // check Nan
v_bfe_u32 v97, v[vgprValuC+148], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+148], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+148], v97, v99, s68
v_lshrrev_b32 v148, 16, v[vgprValuC+148]           // convert C to bf16
buffer_store_b16 v148, v220, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+149], v[vgprValuC+149] // check Nan
v_bfe_u32 v97, v[vgprValuC+149], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+149], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+149], v97, v99, s68
v_lshrrev_b32 v149, 16, v[vgprValuC+149]           // convert C to bf16
buffer_store_b16 v149, v221, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+150], v[vgprValuC+150] // check Nan
v_bfe_u32 v97, v[vgprValuC+150], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+150], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+150], v97, v99, s68
v_lshrrev_b32 v150, 16, v[vgprValuC+150]           // convert C to bf16
buffer_store_b16 v150, v222, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+151], v[vgprValuC+151] // check Nan
v_bfe_u32 v97, v[vgprValuC+151], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+151], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+151], v97, v99, s68
v_lshrrev_b32 v151, 16, v[vgprValuC+151]           // convert C to bf16
buffer_store_b16 v151, v223, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+152], v[vgprValuC+152] // check Nan
v_bfe_u32 v97, v[vgprValuC+152], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+152], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+152], v97, v99, s68
v_lshrrev_b32 v152, 16, v[vgprValuC+152]           // convert C to bf16
buffer_store_b16 v152, v226, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+153], v[vgprValuC+153] // check Nan
v_bfe_u32 v97, v[vgprValuC+153], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+153], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+153], v97, v99, s68
v_lshrrev_b32 v153, 16, v[vgprValuC+153]           // convert C to bf16
buffer_store_b16 v153, v227, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+154], v[vgprValuC+154] // check Nan
v_bfe_u32 v97, v[vgprValuC+154], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+154], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+154], v97, v99, s68
v_lshrrev_b32 v154, 16, v[vgprValuC+154]           // convert C to bf16
buffer_store_b16 v154, v228, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+155], v[vgprValuC+155] // check Nan
v_bfe_u32 v97, v[vgprValuC+155], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+155], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+155], v97, v99, s68
v_lshrrev_b32 v155, 16, v[vgprValuC+155]           // convert C to bf16
buffer_store_b16 v155, v229, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+156], v[vgprValuC+156] // check Nan
v_bfe_u32 v97, v[vgprValuC+156], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+156], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+156], v97, v99, s68
v_lshrrev_b32 v156, 16, v[vgprValuC+156]           // convert C to bf16
buffer_store_b16 v156, v230, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+157], v[vgprValuC+157] // check Nan
v_bfe_u32 v97, v[vgprValuC+157], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+157], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+157], v97, v99, s68
v_lshrrev_b32 v157, 16, v[vgprValuC+157]           // convert C to bf16
buffer_store_b16 v157, v231, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+158], v[vgprValuC+158] // check Nan
v_bfe_u32 v97, v[vgprValuC+158], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+158], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+158], v97, v99, s68
v_lshrrev_b32 v158, 16, v[vgprValuC+158]           // convert C to bf16
buffer_store_b16 v158, v232, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+159], v[vgprValuC+159] // check Nan
v_bfe_u32 v97, v[vgprValuC+159], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+159], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+159], v97, v99, s68
v_lshrrev_b32 v159, 16, v[vgprValuC+159]           // convert C to bf16
buffer_store_b16 v159, v233, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+160], v[vgprValuC+160] // check Nan
v_bfe_u32 v97, v[vgprValuC+160], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+160], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+160], v97, v99, s68
v_lshrrev_b32 v160, 16, v[vgprValuC+160]           // convert C to bf16
buffer_store_b16 v160, v234, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+161], v[vgprValuC+161] // check Nan
v_bfe_u32 v97, v[vgprValuC+161], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+161], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+161], v97, v99, s68
v_lshrrev_b32 v161, 16, v[vgprValuC+161]           // convert C to bf16
buffer_store_b16 v161, v235, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+162], v[vgprValuC+162] // check Nan
v_bfe_u32 v97, v[vgprValuC+162], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+162], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+162], v97, v99, s68
v_lshrrev_b32 v162, 16, v[vgprValuC+162]           // convert C to bf16
buffer_store_b16 v162, v236, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+163], v[vgprValuC+163] // check Nan
v_bfe_u32 v97, v[vgprValuC+163], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+163], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+163], v97, v99, s68
v_lshrrev_b32 v163, 16, v[vgprValuC+163]           // convert C to bf16
buffer_store_b16 v163, v237, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+164], v[vgprValuC+164] // check Nan
v_bfe_u32 v97, v[vgprValuC+164], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+164], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+164], v97, v99, s68
v_lshrrev_b32 v164, 16, v[vgprValuC+164]           // convert C to bf16
buffer_store_b16 v164, v238, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+165], v[vgprValuC+165] // check Nan
v_bfe_u32 v97, v[vgprValuC+165], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+165], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+165], v97, v99, s68
v_lshrrev_b32 v165, 16, v[vgprValuC+165]           // convert C to bf16
buffer_store_b16 v165, v239, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+166], v[vgprValuC+166] // check Nan
v_bfe_u32 v97, v[vgprValuC+166], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+166], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+166], v97, v99, s68
v_lshrrev_b32 v166, 16, v[vgprValuC+166]           // convert C to bf16
buffer_store_b16 v166, v240, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+167], v[vgprValuC+167] // check Nan
v_bfe_u32 v97, v[vgprValuC+167], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+167], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+167], v97, v99, s68
v_lshrrev_b32 v167, 16, v[vgprValuC+167]           // convert C to bf16
buffer_store_b16 v167, v241, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+168], v[vgprValuC+168] // check Nan
v_bfe_u32 v97, v[vgprValuC+168], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+168], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+168], v97, v99, s68
v_lshrrev_b32 v168, 16, v[vgprValuC+168]           // convert C to bf16
buffer_store_b16 v168, v242, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+169], v[vgprValuC+169] // check Nan
v_bfe_u32 v97, v[vgprValuC+169], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+169], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+169], v97, v99, s68
v_lshrrev_b32 v169, 16, v[vgprValuC+169]           // convert C to bf16
buffer_store_b16 v169, v243, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+170], v[vgprValuC+170] // check Nan
v_bfe_u32 v97, v[vgprValuC+170], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+170], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+170], v97, v99, s68
v_lshrrev_b32 v170, 16, v[vgprValuC+170]           // convert C to bf16
buffer_store_b16 v170, v244, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+171], v[vgprValuC+171] // check Nan
v_bfe_u32 v97, v[vgprValuC+171], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+171], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+171], v97, v99, s68
v_lshrrev_b32 v171, 16, v[vgprValuC+171]           // convert C to bf16
buffer_store_b16 v171, v245, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+172], v[vgprValuC+172] // check Nan
v_bfe_u32 v97, v[vgprValuC+172], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+172], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+172], v97, v99, s68
v_lshrrev_b32 v172, 16, v[vgprValuC+172]           // convert C to bf16
buffer_store_b16 v172, v246, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_GSU1:
label_GW_B1_FD0_GSU1:

/* Edge/NonEdge store path check (M): Size % 96 > 0 -> Edge store; else -> NonEdge store */
s_mul_hi_u32 s71, s[sgprSizeI], 89478486           // STATIC_DIV: divisor=96
s_mul_i32 s70, s[sgprSizeI], 89478486              // tmp = dividend * magic
s_lshr_b64 s[70:71], s[70:71], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s69, s70                                 // quotient
s_mul_i32 s70, s69, 96                             // quotient*divisor
s_sub_u32 s68, s[sgprSizeI], s70                   // rReg = dividend - quotient*divisor
s_add_u32 s69, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s69                // wg0 >= nwg0-1 ?
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW1_GSU1_Else       // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 192 > 0 -> Edge store; else -> NonEdge store */
s_mul_hi_u32 s71, s[sgprSizeJ], 44739243           // STATIC_DIV: divisor=192
s_mul_i32 s70, s[sgprSizeJ], 44739243              // tmp = dividend * magic
s_lshr_b64 s[70:71], s[70:71], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s69, s70                                 // quotient
s_mul_i32 s70, s69, 192                            // quotient*divisor
s_sub_u32 s68, s[sgprSizeJ], s70                   // rReg = dividend - quotient*divisor
s_add_u32 s69, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s69                // wg1 >= nwg1-1
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW1_GSU1_Then       // jump if edges required
label_GW_B1_FD0_VW1_GSU1_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=74 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (8,2,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (9,2,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (10,2,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (11,2,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (12,2,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (13,2,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (14,2,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (15,2,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (16,2,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (17,2,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (18,2,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (19,2,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (20,2,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (21,2,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (22,2,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1); (23,2,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v102, v92, v90, 1                   // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=90, coord0Vgpr=90 (multiple bpe)
buffer_load_d16_b16 v175, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
buffer_load_d16_b16 v176, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
buffer_load_d16_b16 v177, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v178, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
buffer_load_d16_b16 v179, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
buffer_load_d16_b16 v180, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v181, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
buffer_load_d16_b16 v182, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
buffer_load_d16_b16 v183, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v184, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
buffer_load_d16_b16 v185, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
buffer_load_d16_b16 v186, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v187, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
buffer_load_d16_b16 v188, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(4,0,2,0) */
buffer_load_d16_b16 v189, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v190, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
buffer_load_d16_b16 v191, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(5,0,2,0) */
buffer_load_d16_b16 v192, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v193, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
buffer_load_d16_b16 v194, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(6,0,2,0) */
buffer_load_d16_b16 v195, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v196, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
buffer_load_d16_b16 v197, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(7,0,2,0) */
buffer_load_d16_b16 v198, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 100                // scale StrideC *= numRows(50) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(50): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(50): gra SRD += inc(upper)
buffer_load_d16_b16 v199, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
buffer_load_d16_b16 v200, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(8,0,2,0) */
buffer_load_d16_b16 v201, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v202, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(9,0,1,0) */
buffer_load_d16_b16 v203, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(9,0,2,0) */
buffer_load_d16_b16 v204, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v205, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(10,0,1,0) */
buffer_load_d16_b16 v206, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(10,0,2,0) */
buffer_load_d16_b16 v207, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v208, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(11,0,1,0) */
buffer_load_d16_b16 v209, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(11,0,2,0) */
buffer_load_d16_b16 v210, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v211, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
buffer_load_d16_b16 v212, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(12,0,2,0) */
buffer_load_d16_b16 v213, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v214, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(13,0,1,0) */
buffer_load_d16_b16 v215, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(13,0,2,0) */
buffer_load_d16_b16 v216, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v217, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(14,0,1,0) */
buffer_load_d16_b16 v218, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(14,0,2,0) */
buffer_load_d16_b16 v219, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v220, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(15,0,1,0) */
buffer_load_d16_b16 v221, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(15,0,2,0) */
buffer_load_d16_b16 v222, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 100                // scale StrideC *= numRows(50) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(50): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(50): gra SRD += inc(upper)
buffer_load_d16_b16 v223, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
buffer_load_d16_b16 v226, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(16,0,2,0) */
buffer_load_d16_b16 v227, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v228, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(17,0,1,0) */
buffer_load_d16_b16 v229, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(17,0,2,0) */
buffer_load_d16_b16 v230, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v231, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(18,0,1,0) */
buffer_load_d16_b16 v232, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(18,0,2,0) */
buffer_load_d16_b16 v233, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v234, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(19,0,1,0) */
buffer_load_d16_b16 v235, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(19,0,2,0) */
buffer_load_d16_b16 v236, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v237, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
buffer_load_d16_b16 v238, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(20,0,2,0) */
buffer_load_d16_b16 v239, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v240, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(21,0,1,0) */
buffer_load_d16_b16 v241, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(21,0,2,0) */
buffer_load_d16_b16 v242, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v243, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(22,0,1,0) */
buffer_load_d16_b16 v244, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(22,0,2,0) */
buffer_load_d16_b16 v245, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v246, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(23,0,1,0) */
buffer_load_d16_b16 v247, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(23,0,2,0) */
buffer_load_d16_b16 v248, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
v_add_lshl_u32 v101, v93, v90, 1                   // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=90, coord0Vgpr=90 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (5, 2, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (7, 2, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (8, 2, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (9, 2, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (10, 2, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (11, 2, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (12, 2, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (13, 2, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (14, 2, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (15, 2, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (16, 2, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (17, 2, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (18, 2, 0, 0), (19, 0, 0, 0), (19, 1, 0, 0), (19, 2, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0), (20, 2, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0), (21, 2, 0, 0), (22, 0, 0, 0), (22, 1, 0, 0), (22, 2, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0), (23, 2, 0, 0)] */
v_mul_f32 v[vgprValuC+103], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+104], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+105], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+106], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+107], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+108], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+109], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+110], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+111], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+112], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+128], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+129], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+130], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+131], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+132], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+133], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+134], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+135], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+48] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+152], s[sgprAlpha], v[vgprValuC+56] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+64] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+154], s[sgprAlpha], v[vgprValuC+49] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+57] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+65] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+50] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+158], s[sgprAlpha], v[vgprValuC+58] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+66] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+160], s[sgprAlpha], v[vgprValuC+51] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+59] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+162], s[sgprAlpha], v[vgprValuC+67] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+163], s[sgprAlpha], v[vgprValuC+52] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+164], s[sgprAlpha], v[vgprValuC+60] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+68] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+166], s[sgprAlpha], v[vgprValuC+53] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+167], s[sgprAlpha], v[vgprValuC+61] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+168], s[sgprAlpha], v[vgprValuC+69] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+169], s[sgprAlpha], v[vgprValuC+54] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+170], s[sgprAlpha], v[vgprValuC+62] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+171], s[sgprAlpha], v[vgprValuC+70] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+172], s[sgprAlpha], v[vgprValuC+55] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+173], s[sgprAlpha], v[vgprValuC+63] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+174], s[sgprAlpha], v[vgprValuC+71] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v98, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v99, 0x7fff0000                          // fp32 Nan
v_mov_b32 v100, 0x7fff                             // rounding bias for bfloat16

s_waitcnt vmcnt(63)                                // vlcnt(71) = 72 - 1 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v175                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+103], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+103], v[vgprValuC+103] // check Nan
v_bfe_u32 v97, v[vgprValuC+103], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+103], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+103], v97, v99, s8
v_lshrrev_b32 v103, 16, v[vgprValuC+103]           // convert C to bf16
buffer_store_b16 v103, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(63)                                // vlcnt(70) = 72 - 2 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v176                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+104], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+104], v[vgprValuC+104] // check Nan
v_bfe_u32 v97, v[vgprValuC+104], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+104], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+104], v97, v99, s8
v_lshrrev_b32 v104, 16, v[vgprValuC+104]           // convert C to bf16
buffer_store_b16 v104, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(63)                                // vlcnt(69) = 72 - 3 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v177                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+105], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+105], v[vgprValuC+105] // check Nan
v_bfe_u32 v97, v[vgprValuC+105], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+105], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+105], v97, v99, s8
v_lshrrev_b32 v105, 16, v[vgprValuC+105]           // convert C to bf16
buffer_store_b16 v105, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(63)                                // vlcnt(68) = 72 - 4 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v178                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+106], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+106], v[vgprValuC+106] // check Nan
v_bfe_u32 v97, v[vgprValuC+106], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+106], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+106], v97, v99, s8
v_lshrrev_b32 v106, 16, v[vgprValuC+106]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v106, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(63)                                // vlcnt(67) = 72 - 5 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v179                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+107], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+107], v[vgprValuC+107] // check Nan
v_bfe_u32 v97, v[vgprValuC+107], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+107], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+107], v97, v99, s8
v_lshrrev_b32 v107, 16, v[vgprValuC+107]           // convert C to bf16
buffer_store_b16 v107, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(63)                                // vlcnt(66) = 72 - 6 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v180                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+108], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+108], v[vgprValuC+108] // check Nan
v_bfe_u32 v97, v[vgprValuC+108], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+108], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+108], v97, v99, s8
v_lshrrev_b32 v108, 16, v[vgprValuC+108]           // convert C to bf16
buffer_store_b16 v108, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(63)                                // vlcnt(65) = 72 - 7 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v181                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+109], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+109], v[vgprValuC+109] // check Nan
v_bfe_u32 v97, v[vgprValuC+109], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+109], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+109], v97, v99, s8
v_lshrrev_b32 v109, 16, v[vgprValuC+109]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v109, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(63)                                // vlcnt(64) = 72 - 8 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v182                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+110], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+110], v[vgprValuC+110] // check Nan
v_bfe_u32 v97, v[vgprValuC+110], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+110], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+110], v97, v99, s8
v_lshrrev_b32 v110, 16, v[vgprValuC+110]           // convert C to bf16
buffer_store_b16 v110, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(63)                                // vlcnt(63) = 72 - 9 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v183                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+111], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+111], v[vgprValuC+111] // check Nan
v_bfe_u32 v97, v[vgprValuC+111], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+111], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+111], v97, v99, s8
v_lshrrev_b32 v111, 16, v[vgprValuC+111]           // convert C to bf16
buffer_store_b16 v111, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(62)                                // vlcnt(62) = 72 - 10 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v184                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+112], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+112], v[vgprValuC+112] // check Nan
v_bfe_u32 v97, v[vgprValuC+112], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+112], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+112], v97, v99, s8
v_lshrrev_b32 v112, 16, v[vgprValuC+112]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v112, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(61)                                // vlcnt(61) = 72 - 11 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v185                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+113], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+113], v[vgprValuC+113] // check Nan
v_bfe_u32 v97, v[vgprValuC+113], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+113], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+113], v97, v99, s8
v_lshrrev_b32 v113, 16, v[vgprValuC+113]           // convert C to bf16
buffer_store_b16 v113, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(60)                                // vlcnt(60) = 72 - 12 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v186                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+114], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+114], v[vgprValuC+114] // check Nan
v_bfe_u32 v97, v[vgprValuC+114], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+114], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+114], v97, v99, s8
v_lshrrev_b32 v114, 16, v[vgprValuC+114]           // convert C to bf16
buffer_store_b16 v114, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(59)                                // vlcnt(59) = 72 - 13 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v187                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+115], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+115], v[vgprValuC+115] // check Nan
v_bfe_u32 v97, v[vgprValuC+115], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+115], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+115], v97, v99, s8
v_lshrrev_b32 v115, 16, v[vgprValuC+115]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v115, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(58)                                // vlcnt(58) = 72 - 14 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v188                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+116], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+116], v[vgprValuC+116] // check Nan
v_bfe_u32 v97, v[vgprValuC+116], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+116], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+116], v97, v99, s8
v_lshrrev_b32 v116, 16, v[vgprValuC+116]           // convert C to bf16
buffer_store_b16 v116, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(57)                                // vlcnt(57) = 72 - 15 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v189                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+117], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+117], v[vgprValuC+117] // check Nan
v_bfe_u32 v97, v[vgprValuC+117], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+117], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+117], v97, v99, s8
v_lshrrev_b32 v117, 16, v[vgprValuC+117]           // convert C to bf16
buffer_store_b16 v117, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(56)                                // vlcnt(56) = 72 - 16 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v190                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+118], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+118], v[vgprValuC+118] // check Nan
v_bfe_u32 v97, v[vgprValuC+118], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+118], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+118], v97, v99, s8
v_lshrrev_b32 v118, 16, v[vgprValuC+118]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v118, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(55)                                // vlcnt(55) = 72 - 17 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v191                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+119], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+119], v[vgprValuC+119] // check Nan
v_bfe_u32 v97, v[vgprValuC+119], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+119], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+119], v97, v99, s8
v_lshrrev_b32 v119, 16, v[vgprValuC+119]           // convert C to bf16
buffer_store_b16 v119, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(54)                                // vlcnt(54) = 72 - 18 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v192                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+120], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+120], v[vgprValuC+120] // check Nan
v_bfe_u32 v97, v[vgprValuC+120], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+120], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+120], v97, v99, s8
v_lshrrev_b32 v120, 16, v[vgprValuC+120]           // convert C to bf16
buffer_store_b16 v120, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(53)                                // vlcnt(53) = 72 - 19 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v193                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+121], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+121], v[vgprValuC+121] // check Nan
v_bfe_u32 v97, v[vgprValuC+121], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+121], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+121], v97, v99, s8
v_lshrrev_b32 v121, 16, v[vgprValuC+121]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v121, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(52)                                // vlcnt(52) = 72 - 20 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v194                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+122], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+122], v[vgprValuC+122] // check Nan
v_bfe_u32 v97, v[vgprValuC+122], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+122], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+122], v97, v99, s8
v_lshrrev_b32 v122, 16, v[vgprValuC+122]           // convert C to bf16
buffer_store_b16 v122, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(51)                                // vlcnt(51) = 72 - 21 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v195                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+123], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+123], v[vgprValuC+123] // check Nan
v_bfe_u32 v97, v[vgprValuC+123], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+123], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+123], v97, v99, s8
v_lshrrev_b32 v123, 16, v[vgprValuC+123]           // convert C to bf16
buffer_store_b16 v123, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(50)                                // vlcnt(50) = 72 - 22 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v196                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+124], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+124], v[vgprValuC+124] // check Nan
v_bfe_u32 v97, v[vgprValuC+124], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+124], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+124], v97, v99, s8
v_lshrrev_b32 v124, 16, v[vgprValuC+124]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v124, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(49)                                // vlcnt(49) = 72 - 23 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v197                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+125], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+125], v[vgprValuC+125] // check Nan
v_bfe_u32 v97, v[vgprValuC+125], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+125], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+125], v97, v99, s8
v_lshrrev_b32 v125, 16, v[vgprValuC+125]           // convert C to bf16
buffer_store_b16 v125, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(48)                                // vlcnt(48) = 72 - 24 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v198                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+126], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+126], v[vgprValuC+126] // check Nan
v_bfe_u32 v97, v[vgprValuC+126], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+126], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+126], v97, v99, s8
v_lshrrev_b32 v126, 16, v[vgprValuC+126]           // convert C to bf16
buffer_store_b16 v126, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(47)                                // vlcnt(47) = 72 - 25 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v199                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+127], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+127], v[vgprValuC+127] // check Nan
v_bfe_u32 v97, v[vgprValuC+127], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+127], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+127], v97, v99, s8
v_lshrrev_b32 v127, 16, v[vgprValuC+127]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 100                // scale StrideD *= numRows(50) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(50): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(50): gra SRD += inc(upper)
buffer_store_b16 v127, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(46)                                // vlcnt(46) = 72 - 26 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v200                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+128], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+128], v[vgprValuC+128] // check Nan
v_bfe_u32 v97, v[vgprValuC+128], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+128], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+128], v97, v99, s8
v_lshrrev_b32 v128, 16, v[vgprValuC+128]           // convert C to bf16
buffer_store_b16 v128, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(45)                                // vlcnt(45) = 72 - 27 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v201                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+129], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+129], v[vgprValuC+129] // check Nan
v_bfe_u32 v97, v[vgprValuC+129], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+129], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+129], v97, v99, s8
v_lshrrev_b32 v129, 16, v[vgprValuC+129]           // convert C to bf16
buffer_store_b16 v129, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(44)                                // vlcnt(44) = 72 - 28 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v202                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+130], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+130], v[vgprValuC+130] // check Nan
v_bfe_u32 v97, v[vgprValuC+130], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+130], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+130], v97, v99, s8
v_lshrrev_b32 v130, 16, v[vgprValuC+130]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v130, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(43)                                // vlcnt(43) = 72 - 29 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v203                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+131], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+131], v[vgprValuC+131] // check Nan
v_bfe_u32 v97, v[vgprValuC+131], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+131], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+131], v97, v99, s8
v_lshrrev_b32 v131, 16, v[vgprValuC+131]           // convert C to bf16
buffer_store_b16 v131, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(42)                                // vlcnt(42) = 72 - 30 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v204                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+132], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+132], v[vgprValuC+132] // check Nan
v_bfe_u32 v97, v[vgprValuC+132], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+132], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+132], v97, v99, s8
v_lshrrev_b32 v132, 16, v[vgprValuC+132]           // convert C to bf16
buffer_store_b16 v132, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(41)                                // vlcnt(41) = 72 - 31 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v205                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+133], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+133], v[vgprValuC+133] // check Nan
v_bfe_u32 v97, v[vgprValuC+133], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+133], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+133], v97, v99, s8
v_lshrrev_b32 v133, 16, v[vgprValuC+133]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v133, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(40)                                // vlcnt(40) = 72 - 32 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v206                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+134], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+134], v[vgprValuC+134] // check Nan
v_bfe_u32 v97, v[vgprValuC+134], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+134], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+134], v97, v99, s8
v_lshrrev_b32 v134, 16, v[vgprValuC+134]           // convert C to bf16
buffer_store_b16 v134, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(39)                                // vlcnt(39) = 72 - 33 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v207                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+135], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+135], v[vgprValuC+135] // check Nan
v_bfe_u32 v97, v[vgprValuC+135], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+135], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+135], v97, v99, s8
v_lshrrev_b32 v135, 16, v[vgprValuC+135]           // convert C to bf16
buffer_store_b16 v135, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(38)                                // vlcnt(38) = 72 - 34 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v208                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+136], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+136], v[vgprValuC+136] // check Nan
v_bfe_u32 v97, v[vgprValuC+136], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+136], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+136], v97, v99, s8
v_lshrrev_b32 v136, 16, v[vgprValuC+136]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v136, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(37)                                // vlcnt(37) = 72 - 35 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v209                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+137], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+137], v[vgprValuC+137] // check Nan
v_bfe_u32 v97, v[vgprValuC+137], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+137], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+137], v97, v99, s8
v_lshrrev_b32 v137, 16, v[vgprValuC+137]           // convert C to bf16
buffer_store_b16 v137, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(36)                                // vlcnt(36) = 72 - 36 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v210                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+138], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+138], v[vgprValuC+138] // check Nan
v_bfe_u32 v97, v[vgprValuC+138], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+138], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+138], v97, v99, s8
v_lshrrev_b32 v138, 16, v[vgprValuC+138]           // convert C to bf16
buffer_store_b16 v138, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(35)                                // vlcnt(35) = 72 - 37 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v211                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+139], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+139], v[vgprValuC+139] // check Nan
v_bfe_u32 v97, v[vgprValuC+139], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+139], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+139], v97, v99, s8
v_lshrrev_b32 v139, 16, v[vgprValuC+139]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v139, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(34)                                // vlcnt(34) = 72 - 38 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v212                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+140], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+140], v[vgprValuC+140] // check Nan
v_bfe_u32 v97, v[vgprValuC+140], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+140], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+140], v97, v99, s8
v_lshrrev_b32 v140, 16, v[vgprValuC+140]           // convert C to bf16
buffer_store_b16 v140, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(33)                                // vlcnt(33) = 72 - 39 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v213                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+141], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+141], v[vgprValuC+141] // check Nan
v_bfe_u32 v97, v[vgprValuC+141], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+141], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+141], v97, v99, s8
v_lshrrev_b32 v141, 16, v[vgprValuC+141]           // convert C to bf16
buffer_store_b16 v141, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(32)                                // vlcnt(32) = 72 - 40 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v214                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+142], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+142], v[vgprValuC+142] // check Nan
v_bfe_u32 v97, v[vgprValuC+142], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+142], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+142], v97, v99, s8
v_lshrrev_b32 v142, 16, v[vgprValuC+142]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v142, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(31)                                // vlcnt(31) = 72 - 41 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v215                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+143], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+143], v[vgprValuC+143] // check Nan
v_bfe_u32 v97, v[vgprValuC+143], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+143], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+143], v97, v99, s8
v_lshrrev_b32 v143, 16, v[vgprValuC+143]           // convert C to bf16
buffer_store_b16 v143, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(30)                                // vlcnt(30) = 72 - 42 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v216                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+144], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+144], v[vgprValuC+144] // check Nan
v_bfe_u32 v97, v[vgprValuC+144], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+144], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+144], v97, v99, s8
v_lshrrev_b32 v144, 16, v[vgprValuC+144]           // convert C to bf16
buffer_store_b16 v144, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(29)                                // vlcnt(29) = 72 - 43 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v217                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+145], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+145], v[vgprValuC+145] // check Nan
v_bfe_u32 v97, v[vgprValuC+145], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+145], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+145], v97, v99, s8
v_lshrrev_b32 v145, 16, v[vgprValuC+145]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v145, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(28)                                // vlcnt(28) = 72 - 44 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v218                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+146], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+146], v[vgprValuC+146] // check Nan
v_bfe_u32 v97, v[vgprValuC+146], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+146], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+146], v97, v99, s8
v_lshrrev_b32 v146, 16, v[vgprValuC+146]           // convert C to bf16
buffer_store_b16 v146, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(27)                                // vlcnt(27) = 72 - 45 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v219                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+147], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+147], v[vgprValuC+147] // check Nan
v_bfe_u32 v97, v[vgprValuC+147], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+147], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+147], v97, v99, s8
v_lshrrev_b32 v147, 16, v[vgprValuC+147]           // convert C to bf16
buffer_store_b16 v147, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(26)                                // vlcnt(26) = 72 - 46 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v220                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+148], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+148], v[vgprValuC+148] // check Nan
v_bfe_u32 v97, v[vgprValuC+148], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+148], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+148], v97, v99, s8
v_lshrrev_b32 v148, 16, v[vgprValuC+148]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v148, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(25)                                // vlcnt(25) = 72 - 47 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v221                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+149], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+149], v[vgprValuC+149] // check Nan
v_bfe_u32 v97, v[vgprValuC+149], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+149], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+149], v97, v99, s8
v_lshrrev_b32 v149, 16, v[vgprValuC+149]           // convert C to bf16
buffer_store_b16 v149, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(24)                                // vlcnt(24) = 72 - 48 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v222                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+150], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+150], v[vgprValuC+150] // check Nan
v_bfe_u32 v97, v[vgprValuC+150], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+150], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+150], v97, v99, s8
v_lshrrev_b32 v150, 16, v[vgprValuC+150]           // convert C to bf16
buffer_store_b16 v150, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(23)                                // vlcnt(23) = 72 - 49 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v223                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+151], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+151], v[vgprValuC+151] // check Nan
v_bfe_u32 v97, v[vgprValuC+151], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+151], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+151], v97, v99, s8
v_lshrrev_b32 v151, 16, v[vgprValuC+151]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 100                // scale StrideD *= numRows(50) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(50): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(50): gra SRD += inc(upper)
buffer_store_b16 v151, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(22)                                // vlcnt(22) = 72 - 50 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v226                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+152], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+152], v[vgprValuC+152] // check Nan
v_bfe_u32 v97, v[vgprValuC+152], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+152], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+152], v97, v99, s8
v_lshrrev_b32 v152, 16, v[vgprValuC+152]           // convert C to bf16
buffer_store_b16 v152, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(21)                                // vlcnt(21) = 72 - 51 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v227                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+153], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+153], v[vgprValuC+153] // check Nan
v_bfe_u32 v97, v[vgprValuC+153], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+153], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+153], v97, v99, s8
v_lshrrev_b32 v153, 16, v[vgprValuC+153]           // convert C to bf16
buffer_store_b16 v153, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(20)                                // vlcnt(20) = 72 - 52 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v228                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+154], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+154], v[vgprValuC+154] // check Nan
v_bfe_u32 v97, v[vgprValuC+154], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+154], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+154], v97, v99, s8
v_lshrrev_b32 v154, 16, v[vgprValuC+154]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v154, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(19)                                // vlcnt(19) = 72 - 53 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v229                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+155], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+155], v[vgprValuC+155] // check Nan
v_bfe_u32 v97, v[vgprValuC+155], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+155], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+155], v97, v99, s8
v_lshrrev_b32 v155, 16, v[vgprValuC+155]           // convert C to bf16
buffer_store_b16 v155, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(18)                                // vlcnt(18) = 72 - 54 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v230                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+156], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+156], v[vgprValuC+156] // check Nan
v_bfe_u32 v97, v[vgprValuC+156], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+156], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+156], v97, v99, s8
v_lshrrev_b32 v156, 16, v[vgprValuC+156]           // convert C to bf16
buffer_store_b16 v156, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(17)                                // vlcnt(17) = 72 - 55 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v231                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+157], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+157], v[vgprValuC+157] // check Nan
v_bfe_u32 v97, v[vgprValuC+157], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+157], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+157], v97, v99, s8
v_lshrrev_b32 v157, 16, v[vgprValuC+157]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v157, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(16)                                // vlcnt(16) = 72 - 56 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v232                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+158], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+158], v[vgprValuC+158] // check Nan
v_bfe_u32 v97, v[vgprValuC+158], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+158], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+158], v97, v99, s8
v_lshrrev_b32 v158, 16, v[vgprValuC+158]           // convert C to bf16
buffer_store_b16 v158, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(15)                                // vlcnt(15) = 72 - 57 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v233                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+159], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+159], v[vgprValuC+159] // check Nan
v_bfe_u32 v97, v[vgprValuC+159], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+159], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+159], v97, v99, s8
v_lshrrev_b32 v159, 16, v[vgprValuC+159]           // convert C to bf16
buffer_store_b16 v159, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(14)                                // vlcnt(14) = 72 - 58 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v234                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+160], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+160], v[vgprValuC+160] // check Nan
v_bfe_u32 v97, v[vgprValuC+160], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+160], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+160], v97, v99, s8
v_lshrrev_b32 v160, 16, v[vgprValuC+160]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v160, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(13)                                // vlcnt(13) = 72 - 59 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v235                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+161], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+161], v[vgprValuC+161] // check Nan
v_bfe_u32 v97, v[vgprValuC+161], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+161], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+161], v97, v99, s8
v_lshrrev_b32 v161, 16, v[vgprValuC+161]           // convert C to bf16
buffer_store_b16 v161, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(12)                                // vlcnt(12) = 72 - 60 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v236                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+162], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+162], v[vgprValuC+162] // check Nan
v_bfe_u32 v97, v[vgprValuC+162], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+162], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+162], v97, v99, s8
v_lshrrev_b32 v162, 16, v[vgprValuC+162]           // convert C to bf16
buffer_store_b16 v162, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(11)                                // vlcnt(11) = 72 - 61 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v237                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+163], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+163], v[vgprValuC+163] // check Nan
v_bfe_u32 v97, v[vgprValuC+163], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+163], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+163], v97, v99, s8
v_lshrrev_b32 v163, 16, v[vgprValuC+163]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v163, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(10)                                // vlcnt(10) = 72 - 62 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v238                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+164], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+164], v[vgprValuC+164] // check Nan
v_bfe_u32 v97, v[vgprValuC+164], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+164], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+164], v97, v99, s8
v_lshrrev_b32 v164, 16, v[vgprValuC+164]           // convert C to bf16
buffer_store_b16 v164, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(9)                                 // vlcnt(9) = 72 - 63 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v239                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+165], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+165], v[vgprValuC+165] // check Nan
v_bfe_u32 v97, v[vgprValuC+165], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+165], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+165], v97, v99, s8
v_lshrrev_b32 v165, 16, v[vgprValuC+165]           // convert C to bf16
buffer_store_b16 v165, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(8)                                 // vlcnt(8) = 72 - 64 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v240                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+166], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+166], v[vgprValuC+166] // check Nan
v_bfe_u32 v97, v[vgprValuC+166], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+166], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+166], v97, v99, s8
v_lshrrev_b32 v166, 16, v[vgprValuC+166]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v166, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(7)                                 // vlcnt(7) = 72 - 65 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v241                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+167], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+167], v[vgprValuC+167] // check Nan
v_bfe_u32 v97, v[vgprValuC+167], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+167], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+167], v97, v99, s8
v_lshrrev_b32 v167, 16, v[vgprValuC+167]           // convert C to bf16
buffer_store_b16 v167, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(6)                                 // vlcnt(6) = 72 - 66 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v242                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+168], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+168], v[vgprValuC+168] // check Nan
v_bfe_u32 v97, v[vgprValuC+168], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+168], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+168], v97, v99, s8
v_lshrrev_b32 v168, 16, v[vgprValuC+168]           // convert C to bf16
buffer_store_b16 v168, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(5)                                 // vlcnt(5) = 72 - 67 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v243                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+169], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+169], v[vgprValuC+169] // check Nan
v_bfe_u32 v97, v[vgprValuC+169], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+169], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+169], v97, v99, s8
v_lshrrev_b32 v169, 16, v[vgprValuC+169]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v169, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(4)                                 // vlcnt(4) = 72 - 68 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v244                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+170], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+170], v[vgprValuC+170] // check Nan
v_bfe_u32 v97, v[vgprValuC+170], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+170], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+170], v97, v99, s8
v_lshrrev_b32 v170, 16, v[vgprValuC+170]           // convert C to bf16
buffer_store_b16 v170, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(3)                                 // vlcnt(3) = 72 - 69 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v245                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+171], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+171], v[vgprValuC+171] // check Nan
v_bfe_u32 v97, v[vgprValuC+171], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+171], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+171], v97, v99, s8
v_lshrrev_b32 v171, 16, v[vgprValuC+171]           // convert C to bf16
buffer_store_b16 v171, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(2)                                 // vlcnt(2) = 72 - 70 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v246                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+172], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+172], v[vgprValuC+172] // check Nan
v_bfe_u32 v97, v[vgprValuC+172], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+172], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+172], v97, v99, s8
v_lshrrev_b32 v172, 16, v[vgprValuC+172]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v172, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(1)                                 // vlcnt(1) = 72 - 71 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v247                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+173], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+173], v[vgprValuC+173] // check Nan
v_bfe_u32 v97, v[vgprValuC+173], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+173], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+173], v97, v99, s8
v_lshrrev_b32 v173, 16, v[vgprValuC+173]           // convert C to bf16
buffer_store_b16 v173, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(0)                                 // vlcnt(0) = 72 - 72 (beta) (interleaved)
v_lshlrev_b32 v94, 16, v248                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+174], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+174], v[vgprValuC+174] // check Nan
v_bfe_u32 v97, v[vgprValuC+174], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+174], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+174], v97, v99, s8
v_lshrrev_b32 v174, 16, v[vgprValuC+174]           // convert C to bf16
buffer_store_b16 v174, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_FD0_VW1_GSU1_NonEdgeEnd:
label_GW_B1_FD0_VW1_GSU1_Else:
label_GW_B1_FD0_VW1_GSU1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=50 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (8,2,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (9,2,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (10,2,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (11,2,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (12,2,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (13,2,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (14,2,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (15,2,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v96, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v152, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v152, v96, v152, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v151, v152, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v152, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v152, v96, v152, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v154, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v96, v154, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v153, v154, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v154, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v96, v154, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v156, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v96, v156, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v155, v156, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v156, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v96, v156, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v158, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v96, v158, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v157, v158, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v158, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v96, v158, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v160, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v160, v96, v160, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v159, v160, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v160, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v160, v96, v160, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v162, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v162, v96, v162, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v161, v162, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v162, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v162, v96, v162, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v164, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v164, v96, v164, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v163, v164, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v164, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v164, v96, v164, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v166, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v166, v96, v166, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v165, v166, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v166, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v166, v96, v166, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v168, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v168, v96, v168, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v167, v168, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v168, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v168, v96, v168, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v170, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v170, v96, v170, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v169, v170, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v170, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v170, v96, v170, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v172, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v172, v96, v172, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v171, v172, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v172, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v172, v96, v172, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v174, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v174, v96, v174, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v173, v174, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v174, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v174, v96, v174, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v176, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v176, v96, v176, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v175, v176, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v176, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v176, v96, v176, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v178, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v178, v96, v178, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v177, v178, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v178, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v178, v96, v178, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v180, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v180, v96, v180, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v179, v180, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v180, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v180, v96, v180, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v182, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v182, v96, v182, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v181, v182, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v182, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v182, v96, v182, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v184, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v184, v96, v184, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v183, v184, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v184, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v184, v96, v184, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v186, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v186, v96, v186, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v185, v186, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v186, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v186, v96, v186, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v188, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v188, v96, v188, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v187, v188, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v188, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v188, v96, v188, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v190, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v190, v96, v190, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v189, v190, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v190, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v190, v96, v190, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v192, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v192, v96, v192, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v191, v192, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v192, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v192, v96, v192, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v194, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v194, v96, v194, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v193, v194, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v194, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v194, v96, v194, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v196, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v196, v96, v196, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v195, v196, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v196, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v196, v96, v196, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v198, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v198, v96, v198, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v197, v198, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v198, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v198, v96, v198, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 50                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 50                // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 50                // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v200, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v200, v96, v200, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v199, v200, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v200, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v200, v96, v200, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v202, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v202, v96, v202, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v201, v202, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v202, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v202, v96, v202, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v204, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v204, v96, v204, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v203, v204, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v204, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v204, v96, v204, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v206, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v206, v96, v206, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v205, v206, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v206, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v206, v96, v206, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v208, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v208, v96, v208, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v207, v208, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v208, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v208, v96, v208, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v210, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v210, v96, v210, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v209, v210, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v210, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v210, v96, v210, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v212, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v212, v96, v212, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v211, v212, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v212, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v212, v96, v212, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v214, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v214, v96, v214, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v213, v214, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v214, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v214, v96, v214, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v216, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v216, v96, v216, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v215, v216, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v216, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v216, v96, v216, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v218, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v218, v96, v218, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v217, v218, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v218, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v218, v96, v218, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v220, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v220, v96, v220, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v219, v220, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v220, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v220, v96, v220, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v222, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v222, v96, v222, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v221, v222, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v222, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v222, v96, v222, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v226, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v226, v96, v226, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v223, v226, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v226, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v226, v96, v226, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v228, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v228, v96, v228, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v227, v228, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v228, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v228, v96, v228, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v230, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v230, v96, v230, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v229, v230, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v230, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v230, v96, v230, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v232, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v232, v96, v232, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v231, v232, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v232, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v232, v96, v232, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v234, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v234, v96, v234, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v233, v234, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v234, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v234, v96, v234, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v236, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v236, v96, v236, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v235, v236, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v236, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v236, v96, v236, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v238, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v238, v96, v238, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v237, v238, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v238, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v238, v96, v238, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v240, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v240, v96, v240, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v239, v240, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v240, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v240, v96, v240, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v242, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v242, v96, v242, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v241, v242, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v242, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v242, v96, v242, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v244, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v244, v96, v244, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v243, v244, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v244, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v244, v96, v244, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v246, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v246, v96, v246, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v245, v246, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v246, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v246, v96, v246, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v248, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v248, v96, v248, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v247, v248, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v248, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v248, v96, v248, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 50                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 50                // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 50                // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v250, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v250, v96, v250, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v249, v250, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v250, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v250, v96, v250, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v252, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v252, v96, v252, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v251, v252, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v252, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v252, v96, v252, s70                 // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (5, 2, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (7, 2, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (8, 2, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (9, 2, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (10, 2, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (11, 2, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (12, 2, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (13, 2, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (14, 2, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (15, 2, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+102], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+103], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+104], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+105], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+106], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+107], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+108], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+109], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+110], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+111], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+112], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+128], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+129], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+130], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+131], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+132], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+133], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+134], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+135], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+48] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+56] // Multiply MI out reg with alpha
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_mov_b32 v98, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v99, 0x7fff0000                          // fp32 Nan
v_mov_b32 v100, 0x7fff                             // rounding bias for bfloat16
v_lshlrev_b32 v94, 16, v151                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+101], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+101], v[vgprValuC+101] // check Nan
v_bfe_u32 v97, v[vgprValuC+101], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+101], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+101], v97, v99, s68
v_lshrrev_b32 v101, 16, v[vgprValuC+101]           // convert C to bf16
buffer_store_b16 v101, v152, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v153                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+102], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+102], v[vgprValuC+102] // check Nan
v_bfe_u32 v97, v[vgprValuC+102], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+102], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+102], v97, v99, s68
v_lshrrev_b32 v102, 16, v[vgprValuC+102]           // convert C to bf16
buffer_store_b16 v102, v154, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v155                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+103], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+103], v[vgprValuC+103] // check Nan
v_bfe_u32 v97, v[vgprValuC+103], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+103], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+103], v97, v99, s68
v_lshrrev_b32 v103, 16, v[vgprValuC+103]           // convert C to bf16
buffer_store_b16 v103, v156, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v157                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+104], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+104], v[vgprValuC+104] // check Nan
v_bfe_u32 v97, v[vgprValuC+104], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+104], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+104], v97, v99, s68
v_lshrrev_b32 v104, 16, v[vgprValuC+104]           // convert C to bf16
buffer_store_b16 v104, v158, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v159                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+105], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+105], v[vgprValuC+105] // check Nan
v_bfe_u32 v97, v[vgprValuC+105], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+105], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+105], v97, v99, s68
v_lshrrev_b32 v105, 16, v[vgprValuC+105]           // convert C to bf16
buffer_store_b16 v105, v160, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v161                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+106], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+106], v[vgprValuC+106] // check Nan
v_bfe_u32 v97, v[vgprValuC+106], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+106], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+106], v97, v99, s68
v_lshrrev_b32 v106, 16, v[vgprValuC+106]           // convert C to bf16
buffer_store_b16 v106, v162, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v163                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+107], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+107], v[vgprValuC+107] // check Nan
v_bfe_u32 v97, v[vgprValuC+107], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+107], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+107], v97, v99, s68
v_lshrrev_b32 v107, 16, v[vgprValuC+107]           // convert C to bf16
buffer_store_b16 v107, v164, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v165                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+108], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+108], v[vgprValuC+108] // check Nan
v_bfe_u32 v97, v[vgprValuC+108], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+108], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+108], v97, v99, s68
v_lshrrev_b32 v108, 16, v[vgprValuC+108]           // convert C to bf16
buffer_store_b16 v108, v166, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v167                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+109], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+109], v[vgprValuC+109] // check Nan
v_bfe_u32 v97, v[vgprValuC+109], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+109], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+109], v97, v99, s68
v_lshrrev_b32 v109, 16, v[vgprValuC+109]           // convert C to bf16
buffer_store_b16 v109, v168, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v169                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+110], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+110], v[vgprValuC+110] // check Nan
v_bfe_u32 v97, v[vgprValuC+110], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+110], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+110], v97, v99, s68
v_lshrrev_b32 v110, 16, v[vgprValuC+110]           // convert C to bf16
buffer_store_b16 v110, v170, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v171                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+111], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+111], v[vgprValuC+111] // check Nan
v_bfe_u32 v97, v[vgprValuC+111], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+111], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+111], v97, v99, s68
v_lshrrev_b32 v111, 16, v[vgprValuC+111]           // convert C to bf16
buffer_store_b16 v111, v172, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v173                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+112], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+112], v[vgprValuC+112] // check Nan
v_bfe_u32 v97, v[vgprValuC+112], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+112], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+112], v97, v99, s68
v_lshrrev_b32 v112, 16, v[vgprValuC+112]           // convert C to bf16
buffer_store_b16 v112, v174, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v175                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+113], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+113], v[vgprValuC+113] // check Nan
v_bfe_u32 v97, v[vgprValuC+113], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+113], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+113], v97, v99, s68
v_lshrrev_b32 v113, 16, v[vgprValuC+113]           // convert C to bf16
buffer_store_b16 v113, v176, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v177                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+114], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+114], v[vgprValuC+114] // check Nan
v_bfe_u32 v97, v[vgprValuC+114], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+114], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+114], v97, v99, s68
v_lshrrev_b32 v114, 16, v[vgprValuC+114]           // convert C to bf16
buffer_store_b16 v114, v178, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v179                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+115], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+115], v[vgprValuC+115] // check Nan
v_bfe_u32 v97, v[vgprValuC+115], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+115], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+115], v97, v99, s68
v_lshrrev_b32 v115, 16, v[vgprValuC+115]           // convert C to bf16
buffer_store_b16 v115, v180, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v181                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+116], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+116], v[vgprValuC+116] // check Nan
v_bfe_u32 v97, v[vgprValuC+116], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+116], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+116], v97, v99, s68
v_lshrrev_b32 v116, 16, v[vgprValuC+116]           // convert C to bf16
buffer_store_b16 v116, v182, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v183                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+117], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+117], v[vgprValuC+117] // check Nan
v_bfe_u32 v97, v[vgprValuC+117], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+117], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+117], v97, v99, s68
v_lshrrev_b32 v117, 16, v[vgprValuC+117]           // convert C to bf16
buffer_store_b16 v117, v184, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v185                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+118], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+118], v[vgprValuC+118] // check Nan
v_bfe_u32 v97, v[vgprValuC+118], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+118], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+118], v97, v99, s68
v_lshrrev_b32 v118, 16, v[vgprValuC+118]           // convert C to bf16
buffer_store_b16 v118, v186, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v187                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+119], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+119], v[vgprValuC+119] // check Nan
v_bfe_u32 v97, v[vgprValuC+119], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+119], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+119], v97, v99, s68
v_lshrrev_b32 v119, 16, v[vgprValuC+119]           // convert C to bf16
buffer_store_b16 v119, v188, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v189                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+120], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+120], v[vgprValuC+120] // check Nan
v_bfe_u32 v97, v[vgprValuC+120], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+120], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+120], v97, v99, s68
v_lshrrev_b32 v120, 16, v[vgprValuC+120]           // convert C to bf16
buffer_store_b16 v120, v190, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v191                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+121], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+121], v[vgprValuC+121] // check Nan
v_bfe_u32 v97, v[vgprValuC+121], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+121], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+121], v97, v99, s68
v_lshrrev_b32 v121, 16, v[vgprValuC+121]           // convert C to bf16
buffer_store_b16 v121, v192, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v193                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+122], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+122], v[vgprValuC+122] // check Nan
v_bfe_u32 v97, v[vgprValuC+122], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+122], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+122], v97, v99, s68
v_lshrrev_b32 v122, 16, v[vgprValuC+122]           // convert C to bf16
buffer_store_b16 v122, v194, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v195                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+123], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+123], v[vgprValuC+123] // check Nan
v_bfe_u32 v97, v[vgprValuC+123], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+123], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+123], v97, v99, s68
v_lshrrev_b32 v123, 16, v[vgprValuC+123]           // convert C to bf16
buffer_store_b16 v123, v196, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v197                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+124], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+124], v[vgprValuC+124] // check Nan
v_bfe_u32 v97, v[vgprValuC+124], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+124], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+124], v97, v99, s68
v_lshrrev_b32 v124, 16, v[vgprValuC+124]           // convert C to bf16
buffer_store_b16 v124, v198, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v199                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+125], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+125], v[vgprValuC+125] // check Nan
v_bfe_u32 v97, v[vgprValuC+125], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+125], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+125], v97, v99, s68
v_lshrrev_b32 v125, 16, v[vgprValuC+125]           // convert C to bf16
buffer_store_b16 v125, v200, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v201                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+126], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+126], v[vgprValuC+126] // check Nan
v_bfe_u32 v97, v[vgprValuC+126], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+126], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+126], v97, v99, s68
v_lshrrev_b32 v126, 16, v[vgprValuC+126]           // convert C to bf16
buffer_store_b16 v126, v202, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v203                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+127], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+127], v[vgprValuC+127] // check Nan
v_bfe_u32 v97, v[vgprValuC+127], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+127], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+127], v97, v99, s68
v_lshrrev_b32 v127, 16, v[vgprValuC+127]           // convert C to bf16
buffer_store_b16 v127, v204, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v205                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+128], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+128], v[vgprValuC+128] // check Nan
v_bfe_u32 v97, v[vgprValuC+128], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+128], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+128], v97, v99, s68
v_lshrrev_b32 v128, 16, v[vgprValuC+128]           // convert C to bf16
buffer_store_b16 v128, v206, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v207                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+129], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+129], v[vgprValuC+129] // check Nan
v_bfe_u32 v97, v[vgprValuC+129], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+129], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+129], v97, v99, s68
v_lshrrev_b32 v129, 16, v[vgprValuC+129]           // convert C to bf16
buffer_store_b16 v129, v208, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v209                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+130], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+130], v[vgprValuC+130] // check Nan
v_bfe_u32 v97, v[vgprValuC+130], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+130], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+130], v97, v99, s68
v_lshrrev_b32 v130, 16, v[vgprValuC+130]           // convert C to bf16
buffer_store_b16 v130, v210, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v211                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+131], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+131], v[vgprValuC+131] // check Nan
v_bfe_u32 v97, v[vgprValuC+131], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+131], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+131], v97, v99, s68
v_lshrrev_b32 v131, 16, v[vgprValuC+131]           // convert C to bf16
buffer_store_b16 v131, v212, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v213                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+132], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+132], v[vgprValuC+132] // check Nan
v_bfe_u32 v97, v[vgprValuC+132], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+132], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+132], v97, v99, s68
v_lshrrev_b32 v132, 16, v[vgprValuC+132]           // convert C to bf16
buffer_store_b16 v132, v214, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v215                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+133], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+133], v[vgprValuC+133] // check Nan
v_bfe_u32 v97, v[vgprValuC+133], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+133], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+133], v97, v99, s68
v_lshrrev_b32 v133, 16, v[vgprValuC+133]           // convert C to bf16
buffer_store_b16 v133, v216, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v217                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+134], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+134], v[vgprValuC+134] // check Nan
v_bfe_u32 v97, v[vgprValuC+134], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+134], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+134], v97, v99, s68
v_lshrrev_b32 v134, 16, v[vgprValuC+134]           // convert C to bf16
buffer_store_b16 v134, v218, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v219                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+135], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+135], v[vgprValuC+135] // check Nan
v_bfe_u32 v97, v[vgprValuC+135], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+135], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+135], v97, v99, s68
v_lshrrev_b32 v135, 16, v[vgprValuC+135]           // convert C to bf16
buffer_store_b16 v135, v220, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v221                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+136], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+136], v[vgprValuC+136] // check Nan
v_bfe_u32 v97, v[vgprValuC+136], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+136], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+136], v97, v99, s68
v_lshrrev_b32 v136, 16, v[vgprValuC+136]           // convert C to bf16
buffer_store_b16 v136, v222, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v223                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+137], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+137], v[vgprValuC+137] // check Nan
v_bfe_u32 v97, v[vgprValuC+137], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+137], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+137], v97, v99, s68
v_lshrrev_b32 v137, 16, v[vgprValuC+137]           // convert C to bf16
buffer_store_b16 v137, v226, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v227                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+138], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+138], v[vgprValuC+138] // check Nan
v_bfe_u32 v97, v[vgprValuC+138], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+138], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+138], v97, v99, s68
v_lshrrev_b32 v138, 16, v[vgprValuC+138]           // convert C to bf16
buffer_store_b16 v138, v228, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v229                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+139], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+139], v[vgprValuC+139] // check Nan
v_bfe_u32 v97, v[vgprValuC+139], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+139], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+139], v97, v99, s68
v_lshrrev_b32 v139, 16, v[vgprValuC+139]           // convert C to bf16
buffer_store_b16 v139, v230, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v231                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+140], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+140], v[vgprValuC+140] // check Nan
v_bfe_u32 v97, v[vgprValuC+140], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+140], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+140], v97, v99, s68
v_lshrrev_b32 v140, 16, v[vgprValuC+140]           // convert C to bf16
buffer_store_b16 v140, v232, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v233                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+141], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+141], v[vgprValuC+141] // check Nan
v_bfe_u32 v97, v[vgprValuC+141], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+141], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+141], v97, v99, s68
v_lshrrev_b32 v141, 16, v[vgprValuC+141]           // convert C to bf16
buffer_store_b16 v141, v234, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v235                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+142], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+142], v[vgprValuC+142] // check Nan
v_bfe_u32 v97, v[vgprValuC+142], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+142], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+142], v97, v99, s68
v_lshrrev_b32 v142, 16, v[vgprValuC+142]           // convert C to bf16
buffer_store_b16 v142, v236, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v237                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+143], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+143], v[vgprValuC+143] // check Nan
v_bfe_u32 v97, v[vgprValuC+143], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+143], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+143], v97, v99, s68
v_lshrrev_b32 v143, 16, v[vgprValuC+143]           // convert C to bf16
buffer_store_b16 v143, v238, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v239                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+144], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+144], v[vgprValuC+144] // check Nan
v_bfe_u32 v97, v[vgprValuC+144], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+144], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+144], v97, v99, s68
v_lshrrev_b32 v144, 16, v[vgprValuC+144]           // convert C to bf16
buffer_store_b16 v144, v240, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v241                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+145], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+145], v[vgprValuC+145] // check Nan
v_bfe_u32 v97, v[vgprValuC+145], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+145], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+145], v97, v99, s68
v_lshrrev_b32 v145, 16, v[vgprValuC+145]           // convert C to bf16
buffer_store_b16 v145, v242, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v243                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+146], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+146], v[vgprValuC+146] // check Nan
v_bfe_u32 v97, v[vgprValuC+146], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+146], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+146], v97, v99, s68
v_lshrrev_b32 v146, 16, v[vgprValuC+146]           // convert C to bf16
buffer_store_b16 v146, v244, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v245                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+147], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+147], v[vgprValuC+147] // check Nan
v_bfe_u32 v97, v[vgprValuC+147], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+147], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+147], v97, v99, s68
v_lshrrev_b32 v147, 16, v[vgprValuC+147]           // convert C to bf16
buffer_store_b16 v147, v246, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v247                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+148], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+148], v[vgprValuC+148] // check Nan
v_bfe_u32 v97, v[vgprValuC+148], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+148], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+148], v97, v99, s68
v_lshrrev_b32 v148, 16, v[vgprValuC+148]           // convert C to bf16
buffer_store_b16 v148, v248, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v249                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+149], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+149], v[vgprValuC+149] // check Nan
v_bfe_u32 v97, v[vgprValuC+149], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+149], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+149], v97, v99, s68
v_lshrrev_b32 v149, 16, v[vgprValuC+149]           // convert C to bf16
buffer_store_b16 v149, v250, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v251                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+150], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+150], v[vgprValuC+150] // check Nan
v_bfe_u32 v97, v[vgprValuC+150], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+150], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+150], v97, v99, s68
v_lshrrev_b32 v150, 16, v[vgprValuC+150]           // convert C to bf16
buffer_store_b16 v150, v252, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (16,2,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (17,2,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (18,2,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (19,2,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (20,2,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (21,2,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (22,2,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1); (23,2,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v96, BufferOOB
/* (d1,vc1,d0,vc0)=(16,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v124, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v124, v96, v124, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v123, v124, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v124, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v124, v96, v124, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v126, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v126, v96, v126, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v125, v126, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v126, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v126, v96, v126, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v128, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v128, v96, v128, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v127, v128, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v128, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v128, v96, v128, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v130, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v96, v130, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v129, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v130, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v96, v130, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v132, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v96, v132, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v131, v132, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v132, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v96, v132, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v134, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v96, v134, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v133, v134, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v134, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v96, v134, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v136, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v96, v136, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v135, v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v136, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v96, v136, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v138, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v96, v138, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v137, v138, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v138, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v96, v138, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v140, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v96, v140, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v139, v140, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v140, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v96, v140, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v142, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v96, v142, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v141, v142, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v142, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v96, v142, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v144, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v144, v96, v144, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v143, v144, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v144, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v144, v96, v144, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v146, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v146, v96, v146, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v145, v146, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v146, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v146, v96, v146, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v148, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v148, v96, v148, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v147, v148, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v148, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v148, v96, v148, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v150, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v150, v96, v150, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v149, v150, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v150, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v150, v96, v150, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v152, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v152, v96, v152, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v151, v152, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v152, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v152, v96, v152, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v154, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v96, v154, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v153, v154, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v154, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v96, v154, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v156, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v96, v156, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v155, v156, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v156, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v96, v156, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v158, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v96, v158, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v157, v158, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v158, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v96, v158, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v160, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v160, v96, v160, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v159, v160, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v160, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v160, v96, v160, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
v_add_co_u32 v91, vcc_lo, v91, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v92, v92, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v93, v93, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v90, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v162, v92, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v162, v96, v162, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v161, v162, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v162, v93, v90, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v162, v96, v162, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,1,0) */
v_add_co_u32 v94, vcc_lo, v90, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v164, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v164, v96, v164, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v163, v164, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v164, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v164, v96, v164, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,2,0) */
v_add_co_u32 v94, vcc_lo, v90, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v94, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v91, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v166, v92, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v166, v96, v166, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v165, v166, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v166, v93, v94, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v166, v96, v166, s70                 // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(16, 2, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (17, 2, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (18, 2, 0, 0), (19, 0, 0, 0), (19, 1, 0, 0), (19, 2, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0), (20, 2, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0), (21, 2, 0, 0), (22, 0, 0, 0), (22, 1, 0, 0), (22, 2, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0), (23, 2, 0, 0)] */
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+64] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+102], s[sgprAlpha], v[vgprValuC+49] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+103], s[sgprAlpha], v[vgprValuC+57] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+104], s[sgprAlpha], v[vgprValuC+65] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+105], s[sgprAlpha], v[vgprValuC+50] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+106], s[sgprAlpha], v[vgprValuC+58] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+107], s[sgprAlpha], v[vgprValuC+66] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+108], s[sgprAlpha], v[vgprValuC+51] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+109], s[sgprAlpha], v[vgprValuC+59] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+110], s[sgprAlpha], v[vgprValuC+67] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+111], s[sgprAlpha], v[vgprValuC+52] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+112], s[sgprAlpha], v[vgprValuC+60] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+68] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+53] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+61] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+69] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+54] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+62] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+70] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+55] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+63] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+71] // Multiply MI out reg with alpha
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_mov_b32 v98, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v99, 0x7fff0000                          // fp32 Nan
v_mov_b32 v100, 0x7fff                             // rounding bias for bfloat16
v_lshlrev_b32 v94, 16, v123                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+101], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+101], v[vgprValuC+101] // check Nan
v_bfe_u32 v97, v[vgprValuC+101], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+101], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+101], v97, v99, s68
v_lshrrev_b32 v101, 16, v[vgprValuC+101]           // convert C to bf16
buffer_store_b16 v101, v124, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v125                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+102], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+102], v[vgprValuC+102] // check Nan
v_bfe_u32 v97, v[vgprValuC+102], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+102], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+102], v97, v99, s68
v_lshrrev_b32 v102, 16, v[vgprValuC+102]           // convert C to bf16
buffer_store_b16 v102, v126, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v127                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+103], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+103], v[vgprValuC+103] // check Nan
v_bfe_u32 v97, v[vgprValuC+103], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+103], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+103], v97, v99, s68
v_lshrrev_b32 v103, 16, v[vgprValuC+103]           // convert C to bf16
buffer_store_b16 v103, v128, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v129                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+104], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+104], v[vgprValuC+104] // check Nan
v_bfe_u32 v97, v[vgprValuC+104], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+104], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+104], v97, v99, s68
v_lshrrev_b32 v104, 16, v[vgprValuC+104]           // convert C to bf16
buffer_store_b16 v104, v130, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v131                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+105], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+105], v[vgprValuC+105] // check Nan
v_bfe_u32 v97, v[vgprValuC+105], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+105], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+105], v97, v99, s68
v_lshrrev_b32 v105, 16, v[vgprValuC+105]           // convert C to bf16
buffer_store_b16 v105, v132, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v133                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+106], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+106], v[vgprValuC+106] // check Nan
v_bfe_u32 v97, v[vgprValuC+106], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+106], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+106], v97, v99, s68
v_lshrrev_b32 v106, 16, v[vgprValuC+106]           // convert C to bf16
buffer_store_b16 v106, v134, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v135                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+107], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+107], v[vgprValuC+107] // check Nan
v_bfe_u32 v97, v[vgprValuC+107], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+107], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+107], v97, v99, s68
v_lshrrev_b32 v107, 16, v[vgprValuC+107]           // convert C to bf16
buffer_store_b16 v107, v136, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v137                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+108], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+108], v[vgprValuC+108] // check Nan
v_bfe_u32 v97, v[vgprValuC+108], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+108], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+108], v97, v99, s68
v_lshrrev_b32 v108, 16, v[vgprValuC+108]           // convert C to bf16
buffer_store_b16 v108, v138, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v139                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+109], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+109], v[vgprValuC+109] // check Nan
v_bfe_u32 v97, v[vgprValuC+109], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+109], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+109], v97, v99, s68
v_lshrrev_b32 v109, 16, v[vgprValuC+109]           // convert C to bf16
buffer_store_b16 v109, v140, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v141                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+110], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+110], v[vgprValuC+110] // check Nan
v_bfe_u32 v97, v[vgprValuC+110], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+110], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+110], v97, v99, s68
v_lshrrev_b32 v110, 16, v[vgprValuC+110]           // convert C to bf16
buffer_store_b16 v110, v142, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v143                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+111], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+111], v[vgprValuC+111] // check Nan
v_bfe_u32 v97, v[vgprValuC+111], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+111], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+111], v97, v99, s68
v_lshrrev_b32 v111, 16, v[vgprValuC+111]           // convert C to bf16
buffer_store_b16 v111, v144, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v145                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+112], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+112], v[vgprValuC+112] // check Nan
v_bfe_u32 v97, v[vgprValuC+112], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+112], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+112], v97, v99, s68
v_lshrrev_b32 v112, 16, v[vgprValuC+112]           // convert C to bf16
buffer_store_b16 v112, v146, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v147                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+113], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+113], v[vgprValuC+113] // check Nan
v_bfe_u32 v97, v[vgprValuC+113], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+113], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+113], v97, v99, s68
v_lshrrev_b32 v113, 16, v[vgprValuC+113]           // convert C to bf16
buffer_store_b16 v113, v148, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v149                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+114], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+114], v[vgprValuC+114] // check Nan
v_bfe_u32 v97, v[vgprValuC+114], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+114], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+114], v97, v99, s68
v_lshrrev_b32 v114, 16, v[vgprValuC+114]           // convert C to bf16
buffer_store_b16 v114, v150, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v151                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+115], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+115], v[vgprValuC+115] // check Nan
v_bfe_u32 v97, v[vgprValuC+115], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+115], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+115], v97, v99, s68
v_lshrrev_b32 v115, 16, v[vgprValuC+115]           // convert C to bf16
buffer_store_b16 v115, v152, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v153                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+116], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+116], v[vgprValuC+116] // check Nan
v_bfe_u32 v97, v[vgprValuC+116], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+116], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+116], v97, v99, s68
v_lshrrev_b32 v116, 16, v[vgprValuC+116]           // convert C to bf16
buffer_store_b16 v116, v154, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v155                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+117], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+117], v[vgprValuC+117] // check Nan
v_bfe_u32 v97, v[vgprValuC+117], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+117], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+117], v97, v99, s68
v_lshrrev_b32 v117, 16, v[vgprValuC+117]           // convert C to bf16
buffer_store_b16 v117, v156, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v157                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+118], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+118], v[vgprValuC+118] // check Nan
v_bfe_u32 v97, v[vgprValuC+118], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+118], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+118], v97, v99, s68
v_lshrrev_b32 v118, 16, v[vgprValuC+118]           // convert C to bf16
buffer_store_b16 v118, v158, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v159                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+119], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+119], v[vgprValuC+119] // check Nan
v_bfe_u32 v97, v[vgprValuC+119], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+119], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+119], v97, v99, s68
v_lshrrev_b32 v119, 16, v[vgprValuC+119]           // convert C to bf16
buffer_store_b16 v119, v160, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v161                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+120], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+120], v[vgprValuC+120] // check Nan
v_bfe_u32 v97, v[vgprValuC+120], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+120], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+120], v97, v99, s68
v_lshrrev_b32 v120, 16, v[vgprValuC+120]           // convert C to bf16
buffer_store_b16 v120, v162, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v163                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+121], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+121], v[vgprValuC+121] // check Nan
v_bfe_u32 v97, v[vgprValuC+121], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+121], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+121], v97, v99, s68
v_lshrrev_b32 v121, 16, v[vgprValuC+121]           // convert C to bf16
buffer_store_b16 v121, v164, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v94, 16, v165                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+122], v94, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+122], v[vgprValuC+122] // check Nan
v_bfe_u32 v97, v[vgprValuC+122], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v97, v[vgprValuC+122], v97, v100        // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+122], v97, v99, s68
v_lshrrev_b32 v122, 16, v[vgprValuC+122]           // convert C to bf16
buffer_store_b16 v122, v166, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_SK_Partials_1:
label_GW_Partials_E0:
s_mov_b64 s[sgprSrdWS:sgprSrdWS+1], s[sgprAddressWS:sgprAddressWS+1]
s_mov_b32 s[sgprSrdWS+2], BufferOOB
s_mov_b32 s[sgprSrdWS+3], Srd127_96

s_mul_i32 s7, 0x12000, s[sgprPersistentWorkGroupIndex] // Offset to correct partials tile (low word)
s_mul_hi_u32 s8, 0x12000, s[sgprPersistentWorkGroupIndex] // partials tile offset (high word) for 64-bit SRD
s_add_u32 s[sgprSrdWS+0], s[sgprSrdWS+0], s7       // add lo to SRD
s_addc_u32 s[sgprSrdWS+1], s[sgprSrdWS+1], s8      // add hi (offset high word + lo carry) to SRD

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=150 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Partials Write Batch #0 (d1,d0,vc1,vc0) = */
/*      (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (8,2,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (9,2,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (10,2,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (11,2,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (12,2,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (13,2,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (14,2,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (15,2,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (16,2,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (17,2,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (18,2,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (19,2,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (20,2,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (21,2,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (22,2,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1); (23,2,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v[vgprValuC+103], v[vgprValuC+0]         // copy MI out reg to vreg[0]
v_mov_b32 v[vgprValuC+104], v[vgprValuC+8]         // copy MI out reg to vreg[1]
v_mov_b32 v[vgprValuC+105], v[vgprValuC+16]        // copy MI out reg to vreg[2]
v_mov_b32 v[vgprValuC+106], v[vgprValuC+1]         // copy MI out reg to vreg[3]
v_mov_b32 v[vgprValuC+107], v[vgprValuC+9]         // copy MI out reg to vreg[4]
v_mov_b32 v[vgprValuC+108], v[vgprValuC+17]        // copy MI out reg to vreg[5]
v_mov_b32 v[vgprValuC+109], v[vgprValuC+2]         // copy MI out reg to vreg[6]
v_mov_b32 v[vgprValuC+110], v[vgprValuC+10]        // copy MI out reg to vreg[7]
v_mov_b32 v[vgprValuC+111], v[vgprValuC+18]        // copy MI out reg to vreg[8]
v_mov_b32 v[vgprValuC+112], v[vgprValuC+3]         // copy MI out reg to vreg[9]
v_mov_b32 v[vgprValuC+113], v[vgprValuC+11]        // copy MI out reg to vreg[10]
v_mov_b32 v[vgprValuC+114], v[vgprValuC+19]        // copy MI out reg to vreg[11]
v_mov_b32 v[vgprValuC+115], v[vgprValuC+4]         // copy MI out reg to vreg[12]
v_mov_b32 v[vgprValuC+116], v[vgprValuC+12]        // copy MI out reg to vreg[13]
v_mov_b32 v[vgprValuC+117], v[vgprValuC+20]        // copy MI out reg to vreg[14]
v_mov_b32 v[vgprValuC+118], v[vgprValuC+5]         // copy MI out reg to vreg[15]
v_mov_b32 v[vgprValuC+119], v[vgprValuC+13]        // copy MI out reg to vreg[16]
v_mov_b32 v[vgprValuC+120], v[vgprValuC+21]        // copy MI out reg to vreg[17]
v_mov_b32 v[vgprValuC+121], v[vgprValuC+6]         // copy MI out reg to vreg[18]
v_mov_b32 v[vgprValuC+122], v[vgprValuC+14]        // copy MI out reg to vreg[19]
v_mov_b32 v[vgprValuC+123], v[vgprValuC+22]        // copy MI out reg to vreg[20]
v_mov_b32 v[vgprValuC+124], v[vgprValuC+7]         // copy MI out reg to vreg[21]
v_mov_b32 v[vgprValuC+125], v[vgprValuC+15]        // copy MI out reg to vreg[22]
v_mov_b32 v[vgprValuC+126], v[vgprValuC+23]        // copy MI out reg to vreg[23]
v_mov_b32 v[vgprValuC+127], v[vgprValuC+24]        // copy MI out reg to vreg[24]
v_mov_b32 v[vgprValuC+128], v[vgprValuC+32]        // copy MI out reg to vreg[25]
v_mov_b32 v[vgprValuC+129], v[vgprValuC+40]        // copy MI out reg to vreg[26]
v_mov_b32 v[vgprValuC+130], v[vgprValuC+25]        // copy MI out reg to vreg[27]
v_mov_b32 v[vgprValuC+131], v[vgprValuC+33]        // copy MI out reg to vreg[28]
v_mov_b32 v[vgprValuC+132], v[vgprValuC+41]        // copy MI out reg to vreg[29]
v_mov_b32 v[vgprValuC+133], v[vgprValuC+26]        // copy MI out reg to vreg[30]
v_mov_b32 v[vgprValuC+134], v[vgprValuC+34]        // copy MI out reg to vreg[31]
v_mov_b32 v[vgprValuC+135], v[vgprValuC+42]        // copy MI out reg to vreg[32]
v_mov_b32 v[vgprValuC+136], v[vgprValuC+27]        // copy MI out reg to vreg[33]
v_mov_b32 v[vgprValuC+137], v[vgprValuC+35]        // copy MI out reg to vreg[34]
v_mov_b32 v[vgprValuC+138], v[vgprValuC+43]        // copy MI out reg to vreg[35]
v_mov_b32 v[vgprValuC+139], v[vgprValuC+28]        // copy MI out reg to vreg[36]
v_mov_b32 v[vgprValuC+140], v[vgprValuC+36]        // copy MI out reg to vreg[37]
v_mov_b32 v[vgprValuC+141], v[vgprValuC+44]        // copy MI out reg to vreg[38]
v_mov_b32 v[vgprValuC+142], v[vgprValuC+29]        // copy MI out reg to vreg[39]
v_mov_b32 v[vgprValuC+143], v[vgprValuC+37]        // copy MI out reg to vreg[40]
v_mov_b32 v[vgprValuC+144], v[vgprValuC+45]        // copy MI out reg to vreg[41]
v_mov_b32 v[vgprValuC+145], v[vgprValuC+30]        // copy MI out reg to vreg[42]
v_mov_b32 v[vgprValuC+146], v[vgprValuC+38]        // copy MI out reg to vreg[43]
v_mov_b32 v[vgprValuC+147], v[vgprValuC+46]        // copy MI out reg to vreg[44]
v_mov_b32 v[vgprValuC+148], v[vgprValuC+31]        // copy MI out reg to vreg[45]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+39]        // copy MI out reg to vreg[46]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+47]        // copy MI out reg to vreg[47]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+48]        // copy MI out reg to vreg[48]
v_mov_b32 v[vgprValuC+152], v[vgprValuC+56]        // copy MI out reg to vreg[49]
v_mov_b32 v[vgprValuC+153], v[vgprValuC+64]        // copy MI out reg to vreg[50]
v_mov_b32 v[vgprValuC+154], v[vgprValuC+49]        // copy MI out reg to vreg[51]
v_mov_b32 v[vgprValuC+155], v[vgprValuC+57]        // copy MI out reg to vreg[52]
v_mov_b32 v[vgprValuC+156], v[vgprValuC+65]        // copy MI out reg to vreg[53]
v_mov_b32 v[vgprValuC+157], v[vgprValuC+50]        // copy MI out reg to vreg[54]
v_mov_b32 v[vgprValuC+158], v[vgprValuC+58]        // copy MI out reg to vreg[55]
v_mov_b32 v[vgprValuC+159], v[vgprValuC+66]        // copy MI out reg to vreg[56]
v_mov_b32 v[vgprValuC+160], v[vgprValuC+51]        // copy MI out reg to vreg[57]
v_mov_b32 v[vgprValuC+161], v[vgprValuC+59]        // copy MI out reg to vreg[58]
v_mov_b32 v[vgprValuC+162], v[vgprValuC+67]        // copy MI out reg to vreg[59]
v_mov_b32 v[vgprValuC+163], v[vgprValuC+52]        // copy MI out reg to vreg[60]
v_mov_b32 v[vgprValuC+164], v[vgprValuC+60]        // copy MI out reg to vreg[61]
v_mov_b32 v[vgprValuC+165], v[vgprValuC+68]        // copy MI out reg to vreg[62]
v_mov_b32 v[vgprValuC+166], v[vgprValuC+53]        // copy MI out reg to vreg[63]
v_mov_b32 v[vgprValuC+167], v[vgprValuC+61]        // copy MI out reg to vreg[64]
v_mov_b32 v[vgprValuC+168], v[vgprValuC+69]        // copy MI out reg to vreg[65]
v_mov_b32 v[vgprValuC+169], v[vgprValuC+54]        // copy MI out reg to vreg[66]
v_mov_b32 v[vgprValuC+170], v[vgprValuC+62]        // copy MI out reg to vreg[67]
v_mov_b32 v[vgprValuC+171], v[vgprValuC+70]        // copy MI out reg to vreg[68]
v_mov_b32 v[vgprValuC+172], v[vgprValuC+55]        // copy MI out reg to vreg[69]
v_mov_b32 v[vgprValuC+173], v[vgprValuC+63]        // copy MI out reg to vreg[70]
v_mov_b32 v[vgprValuC+174], v[vgprValuC+71]        // copy MI out reg to vreg[71]

/* apply mask, calc new C and issue writes */
v_mov_b32 v98, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v99, 0x7fff0000                          // fp32 Nan
v_mov_b32 v100, 0x7fff                             // rounding bias for bfloat16
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
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v151, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v152, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v153, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v154, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v155, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v156, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v157, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v158, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v159, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v160, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v161, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v162, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v163, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v164, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v165, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v166, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v167, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v168, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v169, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v170, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v171, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v172, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v173, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v174, v101, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
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
s_mov_b64 s[68:69], s[sgprAddressFlags:sgprAddressFlags+1]
s_mov_b32 s70, BufferOOB
s_mov_b32 s71, Srd127_96
buffer_store_b32 v103, v104, s[68:71], s8 offen offset:0 glc dlc // set flag
s_waitcnt_vscnt null 0                             // wait for data store
label_SK_SkipFlagSet:
s_waitcnt lgkmcnt(0)                               // wait for flag
s_branch label_GW_End_1                            // jump to end
label_GW_End_1:
label_PersistentLoopClose:
s_cmp_ge_u32 s[sgprPersistentIteration], s[sgprPersistentIterationEnd] // Check whether assigned work is exhausted
s_cbranch_scc1 label_StaticGrid_CloseDone
s_getpc_b64 s[68:69]                               // addr of next instr
s_add_i32 s70, label_PersistentLoopStart, 4        // target branch offset
s_abs_i32 s70, s70                                 // abs offset
s_sub_u32 s68, s68, s70                            // sub target branch offset
s_subb_u32 s69, s69, 0                             // sub high and carry
s_setpc_b64 s[68:69]                               // branch to label_PersistentLoopStart
label_StaticGrid_CloseDone:
label_KernelEnd:
s_endpgm                                           // Kernel End
label_ASM_End:  /// The end of the kernel
