
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
.text
.protected RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBGZPU8_UserArgs_MT64x160x64_MI16x16x1_SK_gfx1151
.globl RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBGZPU8_UserArgs_MT64x160x64_MI16x16x1_SK_gfx1151
.p2align 8
.type RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBGZPU8_UserArgs_MT64x160x64_MI16x16x1_SK_gfx1151,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBGZPU8_UserArgs_MT64x160x64_MI16x16x1_SK_gfx1151
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 256 // vgprs
  .amdhsa_next_free_sgpr 90 // sgprs
  .amdhsa_group_segment_fixed_size 32256 // lds bytes
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
/* Num SGPR   =90 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 16 x 5 */
/* SubGroup= 4 x 32 */
/* VectorWidthA=1 */
/* VectorWidthB=1 */
/* GlobalReadVectorWidthA=16, GlobalReadVectorWidthB=8 */
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
  - .name: RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBGZPU8_UserArgs_MT64x160x64_MI16x16x1_SK_gfx1151
    .symbol: 'RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBGZPU8_UserArgs_MT64x160x64_MI16x16x1_SK_gfx1151.kd'
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
    .group_segment_fixed_size:   32256
    .kernarg_segment_align:      8
    .kernarg_segment_size:       208
    .max_flat_workgroup_size:    128
    .private_segment_fixed_size: 0
    .sgpr_count:                 90
    .sgpr_spill_count:           0
    .vgpr_count:                 256
    .vgpr_spill_count:           0
    .wavefront_size:             32
...
.end_amdgpu_metadata
RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBGZPU8_UserArgs_MT64x160x64_MI16x16x1_SK_gfx1151:
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
.set vgprBase, 106
.set vgprLocalWriteAddrA, 102
.set vgprLocalWriteAddrB, 103
.set vgprGlobalReadOffsetA, 80
.set vgprGlobalReadOffsetB, 82
.set vgprGlobalReadOffsetScaleA, 92
.set vgprG2LScaleA, 94
.set vgprG2LScaleZeroA, 96
.set vgprGlobalReadOffsetScaleZeroA, 98
.set vgprGlobalReadByteOffsetScaleZeroA, 100
.set vgprLocalReadAddrA, 104
.set vgprLocalReadAddrB, 105
.set vgprSerial, 220

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
.set sgprAddressScaleZeroA, 72
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

.set MT0, 64
.set MT1, 160
.set DepthU, 64
/* Number of elements to shift-left SRD */
.set SrdShiftLeftA, 16
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
s_load_b32 s[sgprScaleAGMinus1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0xb8
s_load_b64 s[sgprAddressScaleA:sgprAddressScaleA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x80
s_load_b64 s[sgprAddressScaleZeroA:sgprAddressScaleZeroA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x90
s_branch label_LoadArgsEnd
label_HBMArgs:

/* Load address of kernel arguments */
s_load_b64 s[sgprKernArgAddress:sgprKernArgAddress+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 16
s_waitcnt lgkmcnt(0)                               // wait for args to load
label_LoadArgsEnd:
s_mov_b32 s[sgprArgType], s44
s_mov_b32 m0, 0x7e00                               // LDS clamp at 32256 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id
v_mov_b32 v[vgprG2LScaleA+0], 0                    // scaleA: low half of the f32 scale
v_mov_b32 v[vgprG2LScaleA+1], 0                    // scaleA: low half of the f32 scale
s_mov_b32 vcc_hi, 0                                // Ensure hi bits are zero

/* remap workgroup to XCCs */
s_lshr_b32 s84, s[sgprWGM], 0x10                   // Get WGMXCC
s_ff1_i32_b32 s84, s84                             // Get log(WGMXCC)
s_lshr_b32 s85, s[sgprWGM], 0x16                   // Get CU_Count
/* remap WGs if WGMXCC > 1 ( log(WGMXCC) > 0 ) */
s_cmp_gt_i32 s84, 0
s_cbranch_scc0 label_skip_WGMXCC
/* only remap WGs in the range */
s_lshr_b32 s81, s46, s84
s_lshl_b32 s81, s81, s84
s_cmp_ge_u32 s[sgprWorkGroup0], s81
s_cbranch_scc1 label_skip_WGMXCC
s_cmp_eq_u32 s85, 0                                // CU_Count == 0 ?
s_cbranch_scc0 label_XCCG_nonzero
s_lshr_b32 s81, s[sgprWorkGroup0], s84
s_bfm_b32 s82, s84, 0
s_and_b32 s82, s[sgprWorkGroup0], s82
s_lshr_b32 s83, s46, s84
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
v_cvt_f64_u32 v[0:1], s85                          // s82 = s46 / s85
v_rcp_f64 v[0:1], v[0:1]                           // s82 = s46 / s85
v_cvt_f64_u32 v[2:3], s46                          // s82 = s46 / s85
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s82 = s46 / s85
v_cvt_u32_f64 v0, v[0:1]                           // s82 = s46 / s85
v_mul_lo_u32 v1, v0, s85                           // s82 = s46 / s85
v_sub_nc_u32 v2, s46, v1                           // s82 = s46 / s85
v_cmp_ge_u32 vcc_lo, v2, s85                       // s82 = s46 / s85
s_mov_b32 exec_lo, vcc_lo                          // s82 = s46 / s85
v_add_nc_u32 v0, v0, 1                             // s82 = s46 / s85
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s82, v0                        // quotient
s_mul_i32 s82, s82, s85
s_sub_u32 s83, s46, s82
s_cmp_gt_u32 s[sgprWorkGroup0], s82
s_cselect_b32 s82, s83, s85
s_lshr_b32 s82, s82, s84
s_bfm_b32 s83, s84, 0
s_and_b32 s83, s[sgprWorkGroup0], s83
s_mul_i32 s82, s82, s83
/* WorkGroup0 = temp0 + temp1 */
s_add_u32 s[sgprWorkGroup0], s81, s82
label_skip_WGMXCC:  /// skip WGMXCC if no enough WGs to remap
s_cmp_eq_u32 s44, 3
s_cbranch_scc1 label_ArgType3_Routed_To_ArgType0
s_cmp_eq_u32 s44, 0
s_cbranch_scc0 label_MultiGemm
label_ArgType3_Routed_To_ArgType0:
/* init: add vgpr [106...269) to pool */
/* init: add vgpr [0...80) to pool */
/* init: add agpr [0...0) to pool */
s_waitcnt lgkmcnt(0)                               // wait for 128/0 bytes of kern args over preload
s_lshr_b32 s[sgprNumWorkGroups0], s[sgprSizesFree+0], 6 // s[sgprNumWorkGroups0] = s[sgprSizesFree+0] / 64
s_and_b32 s45, 63, s[sgprSizesFree+0]              // s45 = s[sgprSizesFree+0] % 64
s_cmp_gt_u32 s45, 0                                // Check if remainder > 0
s_addc_u32 s[sgprNumWorkGroups0], s[sgprNumWorkGroups0], 0 // Increment if remainder > 0
s_mul_hi_u32 s75, s[sgprSizesFree+1], 53687092     // STATIC_DIV: divisor=160
s_mul_i32 s74, s[sgprSizesFree+1], 53687092        // tmp = dividend * magic
s_lshr_b64 s[74:75], s[74:75], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s[sgprNumWorkGroups1], s74               // quotient
s_mul_i32 s74, s[sgprNumWorkGroups1], 160          // quotient*divisor
s_sub_u32 s45, s[sgprSizesFree+1], s74             // rReg = dividend - quotient*divisor
s_cmp_gt_u32 s45, 0                                // Check if remainder > 0
s_addc_u32 s[sgprNumWorkGroups1], s[sgprNumWorkGroups1], 0 // Increment if remainder > 0
s_branch label_MultiGemmEnd
label_MultiGemm:

/* Check if custom structure pointer is null */
s_and_b32 s44, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s44, 2                                // ArgType == 2 ?
s_cbranch_scc1 label_IsExternalValid               // branch if ArgType == 2
s_mov_b32 s10, 152                                 // KernArgAddressOffset
s_mul_i32 s86, s11, 4
s_mov_b64 s[80:81], s[sgprKernArgAddress:sgprKernArgAddress+1]
s_branch label_IsExternalValidEnd
label_IsExternalValid:
s_mov_b32 s10, 256
s_mov_b32 s86, 0
s_mov_b64 s[80:81], s[sgprKernArgAddress:sgprKernArgAddress+1]
label_IsExternalValidEnd:

/* Grouped Gemm:: prefetch 1 arg load */
s_mov_b32 s9, 1
s_mov_b32 s87, 0
s_load_b128 s[12:15], s[80:81], s86
s_cmpk_eq_u32 s11, 1                               // if gemm_count is 1?
s_cbranch_scc1 label_wgTable_noLoadLoop

/* Grouped Gemm:: accumulate numTiles for each gemm */
/* Grouped Gemm:: loop start */
label_Loop_GemmCount:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s84, s12, 6                             // s84 = s12 / 64
s_and_b32 s82, 63, s12                             // s82 = s12 % 64
s_addc_u32 s84, s84, 0
s_mul_hi_u32 s83, s13, 53687092                    // STATIC_DIV: divisor=160
s_mul_i32 s82, s13, 53687092                       // tmp = dividend * magic
s_lshr_b64 s[82:83], s[82:83], 33                  // tmp0 = quotient
s_mul_i32 s83, s82, 160                            // tmp1 = quotient * divisor
s_cmp_lg_u32 s83, s13                              // if (quotient * divisor != dividend), result+=1
s_addc_u32 s85, s82, 0                             // if (quotient * divisor != dividend), result+=1
s_mul_i32 s84, s84, s85
s_mul_i32 s84, s84, s14
s_add_u32 s87, s87, s84
s_cmp_lt_u32 s[sgprWorkGroup0], s87
s_cbranch_scc1 label_FOUND
s_add_u32 s86, s86, s10
s_load_b128 s[12:15], s[80:81], s86
s_add_u32 s9, s9, 1
s_cmp_lt_u32 s9, s11
s_cbranch_scc1 label_Loop_GemmCount

/* Grouped Gemm:: noLoadLoop */
label_wgTable_noLoadLoop:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s84, s12, 6                             // s84 = s12 / 64
s_and_b32 s82, 63, s12                             // s82 = s12 % 64
s_addc_u32 s84, s84, 0
s_mul_hi_u32 s83, s13, 53687092                    // STATIC_DIV: divisor=160
s_mul_i32 s82, s13, 53687092                       // tmp = dividend * magic
s_lshr_b64 s[82:83], s[82:83], 33                  // tmp0 = quotient
s_mul_i32 s83, s82, 160                            // tmp1 = quotient * divisor
s_cmp_lg_u32 s83, s13                              // if (quotient * divisor != dividend), result+=1
s_addc_u32 s85, s82, 0                             // if (quotient * divisor != dividend), result+=1
s_mul_i32 s84, s84, s85
s_mul_i32 s84, s84, s14
s_add_u32 s87, s87, s84

/* Grouped Gemm:: gemmIndex found */
label_FOUND:
s_sub_u32 s81, s9, 1
s_sub_u32 s80, s87, s84
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s80
/* Check if custom structure pointer is null */
s_and_b32 s44, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s44, 2                                // ArgType == 2 ?
s_cbranch_scc1 label_LoadExternalStruct            // branch if ArgType == 2

/* Grouped Gemm: offset argument address to gemm */
/* Grouped Gemm: offset address from wg_table_start to args_start */
s_lshl2_add_u32 s[sgprKernArgAddress], s11, s[sgprKernArgAddress]
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s81, s81, 152                            // KernArgAddressOffset
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s81
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_b512 s[16:31], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_b256 s[32:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
s_load_b128 s[40:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 112 // 112
s_load_b32 s[sgprScaleAGMinus1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0xb8
s_load_b64 s[sgprAddressScaleA:sgprAddressScaleA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x80
s_load_b64 s[sgprAddressScaleZeroA:sgprAddressScaleZeroA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x90
s_branch label_LoadExternalStructEnd
label_LoadExternalStruct:
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s81, s81, 256
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s81
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
s_load_b64 s[sgprAddressScaleZeroA:sgprAddressScaleZeroA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x9c
label_LoadExternalStructEnd:
/* init: add vgpr [106...269) to pool */
/* init: add vgpr [0...80) to pool */
/* init: add agpr [0...0) to pool */
s_waitcnt lgkmcnt(0)                               // wait for 128/0 bytes of kern args over preload
s_lshr_b32 s[sgprNumWorkGroups0], s[sgprSizesFree+0], 6 // s[sgprNumWorkGroups0] = s[sgprSizesFree+0] / 64
s_and_b32 s45, 63, s[sgprSizesFree+0]              // s45 = s[sgprSizesFree+0] % 64
s_cmp_gt_u32 s45, 0                                // Check if remainder > 0
s_addc_u32 s[sgprNumWorkGroups0], s[sgprNumWorkGroups0], 0 // Increment if remainder > 0
s_mul_hi_u32 s75, s[sgprSizesFree+1], 53687092     // STATIC_DIV: divisor=160
s_mul_i32 s74, s[sgprSizesFree+1], 53687092        // tmp = dividend * magic
s_lshr_b64 s[74:75], s[74:75], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s[sgprNumWorkGroups1], s74               // quotient
s_mul_i32 s74, s[sgprNumWorkGroups1], 160          // quotient*divisor
s_sub_u32 s45, s[sgprSizesFree+1], s74             // rReg = dividend - quotient*divisor
s_cmp_gt_u32 s45, 0                                // Check if remainder > 0
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
.set sgprShadowLimitA, 74
.set sgprShadowLimitB, 88
s_and_b32 s11, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s11, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_Skip_Address_Prepad_For_Pointer_Array
s_sub_u32 s[sgprAddressA+0], s[sgprAddressA+0], 8  // pre-pad to make room for possible pointer shift
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
s_mov_b32 s74, s47                                 // park dpSectionSize
s_mov_b32 s75, s44                                 // park current tile idx
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
s_mov_b32 s44, s75                                 // restore current tile idx
s_add_u32 s45, s[sgprPersistentIteration], s74     // Offset to start of SK section
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], s74 // Offset to start of SK section
s_mul_i32 s11, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1] // totalTiles = nwg0 * nwg1
s_mul_i32 s11, s11, s[sgprSizesFree+2]             // totalTiles *= batch dim 0
s_mul_i32 s11, s11, s[sgprItersPerTile]            // totalIters = totalTiles * itersPerTile
s_min_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], s11 // Cap ending iter at total SK iters
s_cmp_lt_u32 s[sgprPersistentIteration], s11       // Make sure there's work to do
s_cbranch_scc1 label_NoBranch_2                    // Only branch on scc0
s_getpc_b64 s[80:81]                               // addr of next instr
s_add_i32 s82, label_KernelEnd, 4                  // target branch offset
s_add_u32 s80, s80, s82                            // add target branch offset
s_addc_u32 s81, s81, 0                             // add high and carry
s_setpc_b64 s[80:81]                               // branch to label_KernelEnd
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
s_getpc_b64 s[80:81]                               // addr of next instr
s_add_i32 s82, label_PersistentLoopClose, 4        // target branch offset
s_add_u32 s80, s80, s82                            // add target branch offset
s_addc_u32 s81, s81, 0                             // add high and carry
s_setpc_b64 s[80:81]                               // branch to label_PersistentLoopClose
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
v_and_b32 v3, 1, v3                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v1, v3, 10, v1                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(1024); 7. final local read offset: flrOffset = lrOffset + WOffset

/* local read addresses: final offsets a */
v_lshrrev_b32 v2, 5, v[vgprSerial]                 // 2 = Serial / 32
v_lshrrev_b32 v2, 2, v2                            // LSU offset: Get LSU wave_id
s_mov_b32 s11, 64                                  // LSU offset: stride = lsuStride(64) when umlds==True
v_mul_lo_u32 v2, s11, v2                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_nc_u32 v[vgprLocalReadAddrA], v2, v0         // Final Offset: offset = (lro0+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrA], 1, v[vgprLocalReadAddrA] //  (multiple bpe)
v_lshrrev_b32 v3, 7, v[vgprLocalReadAddrA]         // Final Offset: padding 16 per block 128
v_lshl_add_u32 v[vgprLocalReadAddrA], v3, 4, v[vgprLocalReadAddrA] // Final Offset: padding 16 per block 128

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 5, v[vgprSerial]                 // 0 = Serial / 32
v_lshrrev_b32 v0, 2, v0                            // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(64) when umlds==True (dup assign opt.)
v_mul_lo_u32 v0, s11, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
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
/* LVCA = 4 */
/* v1 = A-unroll = serial%LVCA */
v_lshrrev_b32 v0, 2, v[vgprSerial]                 // 0 = Serial / 4
v_and_b32 v1, 3, v[vgprSerial]                     // 1 = Serial % 4
/* unroll *= glvw */
v_lshlrev_b32 v1, 4, v1                            // v1 = v1 * 16
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

/* global read addresses: tile offsets b */
v_mov_b32 v8, v2                                   // groB1J_0
v_add_co_u32 v9, vcc_lo, 16, v8                    // groB1J_1 += LSPB
v_add_co_u32 v10, vcc_lo, 16, v9                   // groB1J_2 += LSPB
v_add_co_u32 v11, vcc_lo, 16, v10                  // groB1J_3 += LSPB
v_add_co_u32 v12, vcc_lo, 16, v11                  // groB1J_4 += LSPB
v_add_co_u32 v13, vcc_lo, 16, v12                  // groB1J_5 += LSPB
v_add_co_u32 v14, vcc_lo, 16, v13                  // groB1J_6 += LSPB
v_add_co_u32 v15, vcc_lo, 16, v14                  // groB1J_7 += LSPB
v_add_co_u32 v16, vcc_lo, 16, v15                  // groB1J_8 += LSPB
v_add_co_u32 v17, vcc_lo, 16, v16                  // groB1J_9 += LSPB

/* global read addresses: unroll offsets a */
v_mov_b32 v18, v1                                  // groAL_0

/* global read addresses: unroll offsets b */
v_mov_b32 v19, v3                                  // groBL_0

/* global read addresses: addresses a */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s47, s[sgprWorkGroup0], 64            // WorkGroup[01] * MT
s_mul_i32 s46, s[sgprWorkGroup0], 64               // WorkGroup[01] * MT
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
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 8 // extend limit for pre-pad
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
s_load_b64 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 0xa8 // Load batchOffsetA from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s44        // Add batch offset to A address (low)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s45       // Add batch offset to A address (high)
s_sub_u32 s[sgprSrdA+0], s[sgprSrdA+0], 8          // pre-pad to make room for possible pointer shift
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
s_mul_i32 s44, s[sgprWorkGroup0], 64               // scaleA: workgroup row origin
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

/* global read addresses: block-scale A zero-point srd */
s_mul_i32 s44, s[sgprWorkGroup0], 64               // scaleZeroA: workgroup row origin
s_lshr_b32 s44, s44, 1                             // scaleZeroA: aligned row origin to bytes
s_mul_i32 s44, s44, s[sgprStrideScaleA]            // scaleZeroA: * kGroups
s_add_u32 s45, s[sgprSizeI], 7                     // scaleZeroA: SizeI + 7
s_lshr_b32 s45, s45, 3                             // scaleZeroA: ceil(SizeI/8) words
s_lshl_b32 s45, s45, 2                             // scaleZeroA: words to bytes
s_mul_i32 s45, s45, s[sgprStrideScaleA]            // scaleZeroA: total bytes
s_sub_u32 s[sgprSrdScaleZeroA+2], s45, s44         // scaleZeroA: buffer limit from the workgroup origin
s_add_u32 s[sgprSrdScaleZeroA+0], s[sgprAddressScaleZeroA+0], s44 // scaleZeroA: SRD base lo
s_addc_u32 s[sgprSrdScaleZeroA+1], s[sgprAddressScaleZeroA+1], 0 // scaleZeroA: SRD base hi
s_mov_b32 s[sgprSrdScaleZeroA+3], Srd127_96        // scaleZeroA: set bits 127_96 in SRD
s_and_b32 s[sgprScaleAKCnt], s[sgprStreamKLocalStart], s[sgprScaleAKMask] // scaleA: position of the first iteration in its group
s_mul_i32 s44, s[sgprStreamKLocalStart], 64        // scaleA: first K of this range
s_lshr_b32 s44, s44, s[sgprScaleAGShift]           // scaleA: groups before this range
s_mul_i32 s45, s44, 2                              // scaleA: skipped bytes
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], s45 // scaleA: skip groups before the split (lower)
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA: skip groups before the split (upper)
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], s45 // scaleA: limit shrinks by the skip
s_lshl_b32 s45, s44, 2                             // scaleZeroA: skipped bytes
s_add_u32 s[sgprSrdScaleZeroA+0], s[sgprSrdScaleZeroA+0], s45 // scaleZeroA: skip groups before the split (lower)
s_addc_u32 s[sgprSrdScaleZeroA+1], s[sgprSrdScaleZeroA+1], 0 // scaleZeroA: skip groups before the split (upper)
s_sub_u32 s[sgprSrdScaleZeroA+2], s[sgprSrdScaleZeroA+2], s45 // scaleZeroA: limit shrinks by the skip

/* global read addresses: addresses b */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s47, s[sgprWorkGroup1], 160           // WorkGroup[01] * MT
s_mul_i32 s46, s[sgprWorkGroup1], 160              // WorkGroup[01] * MT
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
s_load_b64 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 0xb0 // Load batchOffsetB from kernel args
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
v_mul_lo_u32 v20, s[sgprStrideA0I], v[6]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+0+0], vcc_lo, v[18], v[20+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+0+0], 0x10, v[vgprGlobalReadOffsetA+0+0] // add prepad for pointer shift
v_lshrrev_b32 v20, s[sgprScaleAGShift], v18        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+0], s[sgprStrideScaleA], v6 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+0], v20, v[vgprGlobalReadOffsetScaleA+0] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+0], 1, v[vgprGlobalReadOffsetScaleA+0] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetScaleZeroA+0], 3, v6 // scaleZeroA: row / 8
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleZeroA+0], s[sgprStrideScaleA], v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: (row/8) * kGroups
v_add_nc_u32 v[vgprGlobalReadOffsetScaleZeroA+0], v20, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: + kGroup -> word offset
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleZeroA+0], 3, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: word to nibble index
v_and_b32 v20, 7, v6                               // scaleZeroA: nibble = row & 7
v_or_b32 v[vgprGlobalReadOffsetScaleZeroA+0], v20, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: (word << 3) | nibble
v_lshrrev_b32 v[vgprGlobalReadByteOffsetScaleZeroA+0], 1, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: save loop-invariant byte offset
v_lshrrev_b32 v[vgprGlobalReadOffsetA+0], 1, v[vgprGlobalReadOffsetA+0] //  (multiple bpe)
v_mul_lo_u32 v20, s[sgprStrideA0I], v[7]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+1+0], vcc_lo, v[18], v[20+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+1+0], 0x10, v[vgprGlobalReadOffsetA+1+0] // add prepad for pointer shift
v_lshrrev_b32 v20, s[sgprScaleAGShift], v18        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+1], s[sgprStrideScaleA], v7 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+1], v20, v[vgprGlobalReadOffsetScaleA+1] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+1], 1, v[vgprGlobalReadOffsetScaleA+1] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetScaleZeroA+1], 3, v7 // scaleZeroA: row / 8
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleZeroA+1], s[sgprStrideScaleA], v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: (row/8) * kGroups
v_add_nc_u32 v[vgprGlobalReadOffsetScaleZeroA+1], v20, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: + kGroup -> word offset
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleZeroA+1], 3, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: word to nibble index
v_and_b32 v20, 7, v7                               // scaleZeroA: nibble = row & 7
v_or_b32 v[vgprGlobalReadOffsetScaleZeroA+1], v20, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: (word << 3) | nibble
v_lshrrev_b32 v[vgprGlobalReadByteOffsetScaleZeroA+1], 1, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: save loop-invariant byte offset
v_lshrrev_b32 v[vgprGlobalReadOffsetA+1], 1, v[vgprGlobalReadOffsetA+1] //  (multiple bpe)
/* ============================================================= */

/* global read addresses: final offsets b */
/* ============================================================= */
v_mul_lo_u32 v20, s[sgprStrideB1J], v[8]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+0+0], vcc_lo, v[19], v[20+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+0+0], 0x8, v[vgprGlobalReadOffsetB+0+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+0], 1, v[vgprGlobalReadOffsetB+0] //  (multiple bpe)
v_mul_lo_u32 v20, s[sgprStrideB1J], v[9]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+1+0], vcc_lo, v[19], v[20+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+1+0], 0x8, v[vgprGlobalReadOffsetB+1+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+1], 1, v[vgprGlobalReadOffsetB+1] //  (multiple bpe)
v_mul_lo_u32 v20, s[sgprStrideB1J], v[10]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+2+0], vcc_lo, v[19], v[20+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+2+0], 0x8, v[vgprGlobalReadOffsetB+2+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+2], 1, v[vgprGlobalReadOffsetB+2] //  (multiple bpe)
v_mul_lo_u32 v20, s[sgprStrideB1J], v[11]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+3+0], vcc_lo, v[19], v[20+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+3+0], 0x8, v[vgprGlobalReadOffsetB+3+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+3], 1, v[vgprGlobalReadOffsetB+3] //  (multiple bpe)
v_mul_lo_u32 v20, s[sgprStrideB1J], v[12]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+4+0], vcc_lo, v[19], v[20+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+4+0], 0x8, v[vgprGlobalReadOffsetB+4+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+4], 1, v[vgprGlobalReadOffsetB+4] //  (multiple bpe)
v_mul_lo_u32 v20, s[sgprStrideB1J], v[13]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+5+0], vcc_lo, v[19], v[20+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+5+0], 0x8, v[vgprGlobalReadOffsetB+5+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+5], 1, v[vgprGlobalReadOffsetB+5] //  (multiple bpe)
v_mul_lo_u32 v20, s[sgprStrideB1J], v[14]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+6+0], vcc_lo, v[19], v[20+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+6+0], 0x8, v[vgprGlobalReadOffsetB+6+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+6], 1, v[vgprGlobalReadOffsetB+6] //  (multiple bpe)
v_mul_lo_u32 v20, s[sgprStrideB1J], v[15]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+7+0], vcc_lo, v[19], v[20+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+7+0], 0x8, v[vgprGlobalReadOffsetB+7+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+7], 1, v[vgprGlobalReadOffsetB+7] //  (multiple bpe)
v_mul_lo_u32 v20, s[sgprStrideB1J], v[16]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+8+0], vcc_lo, v[19], v[20+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+8+0], 0x8, v[vgprGlobalReadOffsetB+8+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+8], 1, v[vgprGlobalReadOffsetB+8] //  (multiple bpe)
v_mul_lo_u32 v20, s[sgprStrideB1J], v[17]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+9+0], vcc_lo, v[19], v[20+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+9+0], 0x8, v[vgprGlobalReadOffsetB+9+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+9], 1, v[vgprGlobalReadOffsetB+9] //  (multiple bpe)
/* ============================================================= */

/* global read addresses: increments a */
.set GlobalReadIncsA, 32

/* global read addresses: increments b */
.set GlobalReadIncsB, 128
/* declare loop num iterations */

/* initC: remove ValuC vgpr buffer [0...80) from pool */

/* initC: remove acc vgpr buffer [0...0) from pool */

/* initC: remove ValuA/B vgpr buffer [106...163) from pool */
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
buffer_load_b64 v[vgprG2LA+6:vgprG2LA+6+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b64 v[vgprG2LA+14:vgprG2LA+14+1], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_1_0

/* global read block-scale A */
buffer_load_d16_hi_b16 v[vgprG2LScaleA+0], v[vgprGlobalReadOffsetScaleA+0], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 0 as f32
buffer_load_d16_hi_b16 v[vgprG2LScaleA+1], v[vgprGlobalReadOffsetScaleA+1], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 1 as f32
buffer_load_d16_u8 v[vgprG2LScaleZeroA+0], v[vgprGlobalReadByteOffsetScaleZeroA+0], s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 0
buffer_load_d16_u8 v[vgprG2LScaleZeroA+1], v[vgprGlobalReadByteOffsetScaleZeroA+1], s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 1
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
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], GlobalReadIncsA // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], GlobalReadIncsA // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

/* global read inc block-scale A (2 bytes every 1 iters) */
s_add_u32 s[sgprScaleAKCnt], s[sgprScaleAKCnt], 1  // scaleA: one more K iteration done
s_and_b32 s44, s[sgprScaleAKCnt], s[sgprScaleAKMask] // scaleA: SCC = counter has not wrapped
s_cselect_b32 s44, 0, s[sgprScaleAInc]             // scaleA: advance only on a wrap
s_lshl_b32 s45, s44, 1                             // scaleZeroA: advance only on a wrap
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], s44 // scaleA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], s44 // scaleA limit -= inc
s_add_u32 s[sgprSrdScaleZeroA+0], s[sgprSrdScaleZeroA+0], s45 // scaleZeroA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleZeroA+1], s[sgprSrdScaleZeroA+1], 0 // scaleZeroA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleZeroA+2], s[sgprSrdScaleZeroA+2], s45 // scaleZeroA limit -= inc

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
v_mov_b32 v223, v[vgprG2LScaleA+0]                 // scaleA: copy on the other VGPR bank
v_and_b32 v236, 1, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v236, 2, v236                        // scaleZeroA: nibble shift
v_bfe_u32 v236, v[vgprG2LScaleZeroA+0], v236, 0x4  // scaleZeroA: z
v_lshl_add_u32 v237, v236, 16, s[sgprScaleALiftMagic+4] // scaleZeroA: 2^7 + c, exact
v_lshl_add_u32 v232, v236, 0, s[sgprScaleALiftMagic+0] // scaleZeroA: 2^23 + c, exact
v_lshl_add_u32 v233, v236, 4, s[sgprScaleALiftMagic+1] // scaleZeroA: 2^19 + c, exact
v_lshl_add_u32 v234, v236, 8, s[sgprScaleALiftMagic+2] // scaleZeroA: 2^15 + c, exact
v_lshl_add_u32 v235, v236, 12, s[sgprScaleALiftMagic+3] // scaleZeroA: 2^11 + c, exact
v_mov_b32 v222, v[vgprG2LA+6+0]                    // w4a16: save packed int4 dword 0
v_and_or_b32 v224, v222, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v225, v222, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v227, v222, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v226, v222, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v224, v232, v224 :: v_dual_subrev_f32 v225, v233, v225 // w4a16: n - c, exact
v_dual_subrev_f32 v227, v234, v227 :: v_dual_subrev_f32 v226, v235, v226 // w4a16: n - c, exact
v_dual_mul_f32 v224, v94, v224 :: v_dual_mul_f32 v225, v223, v225 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v227, v94, v227 :: v_dual_add_nc_u32 v224, s[sgprScaleALiftMagic+5], v224 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v226, v223, v226 :: v_dual_add_nc_u32 v225, s[sgprScaleALiftMagic+5], v225 // w4a16: (n - c) * s :: round
v_and_or_b32 v228, v222, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v222, 20, v222                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v229, v222, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v231, v222, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v230, v222, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v228, v237, v228 :: v_dual_add_nc_u32 v227, s[sgprScaleALiftMagic+5], v227 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v229, v232, v229 :: v_dual_add_nc_u32 v226, s[sgprScaleALiftMagic+5], v226 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v231, v233, v231 :: v_dual_subrev_f32 v230, v234, v230 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+0+0], v225, v224, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+1], v226, v227, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v228, v94, v228 :: v_dual_mul_f32 v229, v223, v229 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v231, v94, v231 :: v_dual_add_nc_u32 v228, s[sgprScaleALiftMagic+5], v228 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v230, v223, v230 :: v_dual_add_nc_u32 v229, s[sgprScaleALiftMagic+5], v229 // w4a16: (n - c) * s :: round
v_add_nc_u32 v231, s[sgprScaleALiftMagic+5], v231  // w4a16: round
v_add_nc_u32 v230, s[sgprScaleALiftMagic+5], v230  // w4a16: round
v_perm_b32 v[vgprG2LA+0+2], v229, v228, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+3], v230, v231, 0x07060302 // w4a16: pack 2 bf16
v_mov_b32 v222, v[vgprG2LA+6+1]                    // w4a16: save packed int4 dword 1
v_and_or_b32 v224, v222, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v225, v222, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v227, v222, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v226, v222, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v224, v232, v224 :: v_dual_subrev_f32 v225, v233, v225 // w4a16: n - c, exact
v_dual_subrev_f32 v227, v234, v227 :: v_dual_subrev_f32 v226, v235, v226 // w4a16: n - c, exact
v_dual_mul_f32 v224, v94, v224 :: v_dual_mul_f32 v225, v223, v225 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v227, v94, v227 :: v_dual_add_nc_u32 v224, s[sgprScaleALiftMagic+5], v224 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v226, v223, v226 :: v_dual_add_nc_u32 v225, s[sgprScaleALiftMagic+5], v225 // w4a16: (n - c) * s :: round
v_and_or_b32 v228, v222, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v222, 20, v222                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v229, v222, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v231, v222, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v230, v222, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v228, v237, v228 :: v_dual_add_nc_u32 v227, s[sgprScaleALiftMagic+5], v227 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v229, v232, v229 :: v_dual_add_nc_u32 v226, s[sgprScaleALiftMagic+5], v226 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v231, v233, v231 :: v_dual_subrev_f32 v230, v234, v230 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+0+4], v225, v224, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+5], v226, v227, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v228, v94, v228 :: v_dual_mul_f32 v229, v223, v229 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v231, v94, v231 :: v_dual_add_nc_u32 v228, s[sgprScaleALiftMagic+5], v228 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v230, v223, v230 :: v_dual_add_nc_u32 v229, s[sgprScaleALiftMagic+5], v229 // w4a16: (n - c) * s :: round
v_add_nc_u32 v231, s[sgprScaleALiftMagic+5], v231  // w4a16: round
v_add_nc_u32 v230, s[sgprScaleALiftMagic+5], v230  // w4a16: round
v_perm_b32 v[vgprG2LA+0+6], v229, v228, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+7], v230, v231, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:16 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
v_mov_b32 v222, v[vgprG2LScaleA+1]                 // scaleA: copy on the other VGPR bank
v_and_b32 v236, 1, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v236, 2, v236                        // scaleZeroA: nibble shift
v_bfe_u32 v236, v[vgprG2LScaleZeroA+1], v236, 0x4  // scaleZeroA: z
v_lshl_add_u32 v237, v236, 16, s[sgprScaleALiftMagic+4] // scaleZeroA: 2^7 + c, exact
v_lshl_add_u32 v232, v236, 0, s[sgprScaleALiftMagic+0] // scaleZeroA: 2^23 + c, exact
v_lshl_add_u32 v233, v236, 4, s[sgprScaleALiftMagic+1] // scaleZeroA: 2^19 + c, exact
v_lshl_add_u32 v234, v236, 8, s[sgprScaleALiftMagic+2] // scaleZeroA: 2^15 + c, exact
v_lshl_add_u32 v235, v236, 12, s[sgprScaleALiftMagic+3] // scaleZeroA: 2^11 + c, exact
v_mov_b32 v223, v[vgprG2LA+14+0]                   // w4a16: save packed int4 dword 0
v_and_or_b32 v224, v223, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v225, v223, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v227, v223, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v226, v223, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v224, v232, v224 :: v_dual_subrev_f32 v225, v233, v225 // w4a16: n - c, exact
v_dual_subrev_f32 v227, v234, v227 :: v_dual_subrev_f32 v226, v235, v226 // w4a16: n - c, exact
v_dual_mul_f32 v224, v95, v224 :: v_dual_mul_f32 v225, v222, v225 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v227, v95, v227 :: v_dual_add_nc_u32 v224, s[sgprScaleALiftMagic+5], v224 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v226, v222, v226 :: v_dual_add_nc_u32 v225, s[sgprScaleALiftMagic+5], v225 // w4a16: (n - c) * s :: round
v_and_or_b32 v228, v223, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v223, 20, v223                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v229, v223, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v231, v223, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v230, v223, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v228, v237, v228 :: v_dual_add_nc_u32 v227, s[sgprScaleALiftMagic+5], v227 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v229, v232, v229 :: v_dual_add_nc_u32 v226, s[sgprScaleALiftMagic+5], v226 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v231, v233, v231 :: v_dual_subrev_f32 v230, v234, v230 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+8+0], v225, v224, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+1], v226, v227, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v228, v95, v228 :: v_dual_mul_f32 v229, v222, v229 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v231, v95, v231 :: v_dual_add_nc_u32 v228, s[sgprScaleALiftMagic+5], v228 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v230, v222, v230 :: v_dual_add_nc_u32 v229, s[sgprScaleALiftMagic+5], v229 // w4a16: (n - c) * s :: round
v_add_nc_u32 v231, s[sgprScaleALiftMagic+5], v231  // w4a16: round
v_add_nc_u32 v230, s[sgprScaleALiftMagic+5], v230  // w4a16: round
v_perm_b32 v[vgprG2LA+8+2], v229, v228, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+3], v230, v231, 0x07060302 // w4a16: pack 2 bf16
v_mov_b32 v223, v[vgprG2LA+14+1]                   // w4a16: save packed int4 dword 1
v_and_or_b32 v224, v223, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v225, v223, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v227, v223, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v226, v223, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v224, v232, v224 :: v_dual_subrev_f32 v225, v233, v225 // w4a16: n - c, exact
v_dual_subrev_f32 v227, v234, v227 :: v_dual_subrev_f32 v226, v235, v226 // w4a16: n - c, exact
v_dual_mul_f32 v224, v95, v224 :: v_dual_mul_f32 v225, v222, v225 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v227, v95, v227 :: v_dual_add_nc_u32 v224, s[sgprScaleALiftMagic+5], v224 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v226, v222, v226 :: v_dual_add_nc_u32 v225, s[sgprScaleALiftMagic+5], v225 // w4a16: (n - c) * s :: round
v_and_or_b32 v228, v223, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v223, 20, v223                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v229, v223, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v231, v223, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v230, v223, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v228, v237, v228 :: v_dual_add_nc_u32 v227, s[sgprScaleALiftMagic+5], v227 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v229, v232, v229 :: v_dual_add_nc_u32 v226, s[sgprScaleALiftMagic+5], v226 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v231, v233, v231 :: v_dual_subrev_f32 v230, v234, v230 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+8+4], v225, v224, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+5], v226, v227, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v228, v95, v228 :: v_dual_mul_f32 v229, v222, v229 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v231, v95, v231 :: v_dual_add_nc_u32 v228, s[sgprScaleALiftMagic+5], v228 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v230, v222, v230 :: v_dual_add_nc_u32 v229, s[sgprScaleALiftMagic+5], v229 // w4a16: (n - c) * s :: round
v_add_nc_u32 v231, s[sgprScaleALiftMagic+5], v231  // w4a16: round
v_add_nc_u32 v230, s[sgprScaleALiftMagic+5], v230  // w4a16: round
v_perm_b32 v[vgprG2LA+8+6], v229, v228, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+7], v230, v231, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:4608 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 4608 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+12:vgprG2LA+12+3] offset:4624 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 4608 sync LDS0

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
/*  grEndMfmaIndex:3, lwStartMfmaIndex:37, lwEndMfmaIndex:39  */
/*  numMfmaForLR:9, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:36 */
/*  mfmaIndex:0  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4608 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4624 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4608 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4624 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b64 v[vgprG2LA+6:vgprG2LA+6+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b64 v[vgprG2LA+14:vgprG2LA+14+1], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_1_0

/* global read block-scale A */
buffer_load_d16_hi_b16 v[vgprG2LScaleA+0], v[vgprGlobalReadOffsetScaleA+0], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 0 as f32
buffer_load_d16_hi_b16 v[vgprG2LScaleA+1], v[vgprGlobalReadOffsetScaleA+1], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 1 as f32
buffer_load_d16_u8 v[vgprG2LScaleZeroA+0], v[vgprGlobalReadByteOffsetScaleZeroA+0], s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 0
buffer_load_d16_u8 v[vgprG2LScaleZeroA+1], v[vgprGlobalReadByteOffsetScaleZeroA+1], s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 1
buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:1  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:9216 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b128 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b128 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b128 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0
buffer_load_b128 v[vgprG2LB+16:vgprG2LB+16+3], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_4_0
s_waitcnt lgkmcnt(3)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:2  */
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:9232 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b128 v[vgprG2LB+20:vgprG2LB+20+3], v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_5_0
buffer_load_b128 v[vgprG2LB+24:vgprG2LB+24+3], v[vgprGlobalReadOffsetB+6], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_6_0
buffer_load_b128 v[vgprG2LB+28:vgprG2LB+28+3], v[vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_7_0
buffer_load_b128 v[vgprG2LB+32:vgprG2LB+32+3], v[vgprGlobalReadOffsetB+8], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_8_0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:3  */
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:13824 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b128 v[vgprG2LB+36:vgprG2LB+36+3], v[vgprGlobalReadOffsetB+9], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_9_0

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], GlobalReadIncsA // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], GlobalReadIncsA // limit -= inc)
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:4  */
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:13840 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:5  */

/* global read inc block-scale A (2 bytes every 1 iters) */
s_add_u32 s[sgprScaleAKCnt], s[sgprScaleAKCnt], 1  // scaleA: one more K iteration done
s_and_b32 s44, s[sgprScaleAKCnt], s[sgprScaleAKMask] // scaleA: SCC = counter has not wrapped
s_cselect_b32 s44, 0, s[sgprScaleAInc]             // scaleA: advance only on a wrap
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:6  */
ds_load_b128 v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprLocalReadAddrB+0] offset:18432 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[vgprLocalReadAddrB+0] offset:18448 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_lshl_b32 s45, s44, 1                             // scaleZeroA: advance only on a wrap
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], s44 // scaleA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA SRD += inc(upper)
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:7  */
/* localReadsVacancy: latencyLeft 5 */
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], s44 // scaleA limit -= inc
s_add_u32 s[sgprSrdScaleZeroA+0], s[sgprSrdScaleZeroA+0], s45 // scaleZeroA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleZeroA+1], s[sgprSrdScaleZeroA+1], 0 // scaleZeroA SRD += inc(upper)
v_wmma_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/*  mfmaIndex:8  */
/* localReadsVacancy: latencyLeft 5 */
s_sub_u32 s[sgprSrdScaleZeroA+2], s[sgprSrdScaleZeroA+2], s45 // scaleZeroA limit -= inc

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], GlobalReadIncsB // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0x0       // gra SRD += inc(upper)
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
/*  mfmaIndex:9  */
/* localReadsVacancy: latencyLeft 5 */
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], GlobalReadIncsB // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
v_wmma_f32_16x16x16_bf16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=10 */

/* iter 1 */
/*  grEndMfmaIndex:3, lwStartMfmaIndex:37, lwEndMfmaIndex:39  */
/*  numMfmaForLR:9, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:36 */
/*  mfmaIndex:10  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4640 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4656 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4640 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4656 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:11  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:9248 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(3)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:12  */
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:9264 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:13  */
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:13856 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:14  */
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:13872 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:15  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:16  */
ds_load_b128 v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprLocalReadAddrB+0] offset:18464 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[vgprLocalReadAddrB+0] offset:18480 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:17  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/*  mfmaIndex:18  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
/*  mfmaIndex:19  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=0 readsPerIterB=10 */

/* iter 2 */
/*  grEndMfmaIndex:3, lwStartMfmaIndex:37, lwEndMfmaIndex:39  */
/*  numMfmaForLR:9, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:36 */
/*  mfmaIndex:20  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4672 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4688 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4672 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4688 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:21  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:9280 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(3)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:22  */
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:9296 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:23  */
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:13888 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:24  */
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:13904 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:25  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:26  */
ds_load_b128 v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprLocalReadAddrB+0] offset:18496 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[vgprLocalReadAddrB+0] offset:18512 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:27  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/*  mfmaIndex:28  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
/*  mfmaIndex:29  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=0 readsPerIterB=10 */

/* iter 3 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:3, lwStartMfmaIndex:37, lwEndMfmaIndex:39  */
/*  numMfmaForLR:9, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:36 */
/*  mfmaIndex:30  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4704 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4720 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4704 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4720 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:31  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:9312 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(3)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:32  */
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:9328 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:33  */
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:13920 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:34  */
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:13936 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:35  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:36  */
/* schedule remaining localreads for one buffer scheduling */
ds_load_b128 v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprLocalReadAddrB+0] offset:18528 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[vgprLocalReadAddrB+0] offset:18544 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_wmma_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:37  */
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(11)                                // wait for global read before writing to local
v_mov_b32 v223, v[vgprG2LScaleA+0]                 // scaleA: copy on the other VGPR bank
v_and_b32 v236, 1, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v236, 2, v236                        // scaleZeroA: nibble shift
v_bfe_u32 v236, v[vgprG2LScaleZeroA+0], v236, 0x4  // scaleZeroA: z
v_lshl_add_u32 v237, v236, 16, s[sgprScaleALiftMagic+4] // scaleZeroA: 2^7 + c, exact
v_lshl_add_u32 v232, v236, 0, s[sgprScaleALiftMagic+0] // scaleZeroA: 2^23 + c, exact
v_lshl_add_u32 v233, v236, 4, s[sgprScaleALiftMagic+1] // scaleZeroA: 2^19 + c, exact
v_lshl_add_u32 v234, v236, 8, s[sgprScaleALiftMagic+2] // scaleZeroA: 2^15 + c, exact
v_lshl_add_u32 v235, v236, 12, s[sgprScaleALiftMagic+3] // scaleZeroA: 2^11 + c, exact
v_mov_b32 v222, v[vgprG2LA+6+0]                    // w4a16: save packed int4 dword 0
v_and_or_b32 v224, v222, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v225, v222, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v227, v222, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v226, v222, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v224, v232, v224 :: v_dual_subrev_f32 v225, v233, v225 // w4a16: n - c, exact
v_dual_subrev_f32 v227, v234, v227 :: v_dual_subrev_f32 v226, v235, v226 // w4a16: n - c, exact
v_dual_mul_f32 v224, v94, v224 :: v_dual_mul_f32 v225, v223, v225 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v227, v94, v227 :: v_dual_add_nc_u32 v224, s[sgprScaleALiftMagic+5], v224 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v226, v223, v226 :: v_dual_add_nc_u32 v225, s[sgprScaleALiftMagic+5], v225 // w4a16: (n - c) * s :: round
v_and_or_b32 v228, v222, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v222, 20, v222                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v229, v222, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v231, v222, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v230, v222, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v228, v237, v228 :: v_dual_add_nc_u32 v227, s[sgprScaleALiftMagic+5], v227 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v229, v232, v229 :: v_dual_add_nc_u32 v226, s[sgprScaleALiftMagic+5], v226 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v231, v233, v231 :: v_dual_subrev_f32 v230, v234, v230 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+0+0], v225, v224, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+1], v226, v227, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v228, v94, v228 :: v_dual_mul_f32 v229, v223, v229 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v231, v94, v231 :: v_dual_add_nc_u32 v228, s[sgprScaleALiftMagic+5], v228 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v230, v223, v230 :: v_dual_add_nc_u32 v229, s[sgprScaleALiftMagic+5], v229 // w4a16: (n - c) * s :: round
v_add_nc_u32 v231, s[sgprScaleALiftMagic+5], v231  // w4a16: round
v_add_nc_u32 v230, s[sgprScaleALiftMagic+5], v230  // w4a16: round
v_perm_b32 v[vgprG2LA+0+2], v229, v228, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+3], v230, v231, 0x07060302 // w4a16: pack 2 bf16
v_mov_b32 v222, v[vgprG2LA+6+1]                    // w4a16: save packed int4 dword 1
v_and_or_b32 v224, v222, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v225, v222, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v227, v222, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v226, v222, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v224, v232, v224 :: v_dual_subrev_f32 v225, v233, v225 // w4a16: n - c, exact
v_dual_subrev_f32 v227, v234, v227 :: v_dual_subrev_f32 v226, v235, v226 // w4a16: n - c, exact
v_dual_mul_f32 v224, v94, v224 :: v_dual_mul_f32 v225, v223, v225 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v227, v94, v227 :: v_dual_add_nc_u32 v224, s[sgprScaleALiftMagic+5], v224 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v226, v223, v226 :: v_dual_add_nc_u32 v225, s[sgprScaleALiftMagic+5], v225 // w4a16: (n - c) * s :: round
v_and_or_b32 v228, v222, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v222, 20, v222                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v229, v222, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v231, v222, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v230, v222, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v228, v237, v228 :: v_dual_add_nc_u32 v227, s[sgprScaleALiftMagic+5], v227 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v229, v232, v229 :: v_dual_add_nc_u32 v226, s[sgprScaleALiftMagic+5], v226 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v231, v233, v231 :: v_dual_subrev_f32 v230, v234, v230 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+0+4], v225, v224, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+5], v226, v227, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v228, v94, v228 :: v_dual_mul_f32 v229, v223, v229 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v231, v94, v231 :: v_dual_add_nc_u32 v228, s[sgprScaleALiftMagic+5], v228 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v230, v223, v230 :: v_dual_add_nc_u32 v229, s[sgprScaleALiftMagic+5], v229 // w4a16: (n - c) * s :: round
v_add_nc_u32 v231, s[sgprScaleALiftMagic+5], v231  // w4a16: round
v_add_nc_u32 v230, s[sgprScaleALiftMagic+5], v230  // w4a16: round
v_perm_b32 v[vgprG2LA+0+6], v229, v228, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+7], v230, v231, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:16 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(10)                                // wait for global read before writing to local
v_mov_b32 v222, v[vgprG2LScaleA+1]                 // scaleA: copy on the other VGPR bank
v_and_b32 v236, 1, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v236, 2, v236                        // scaleZeroA: nibble shift
v_bfe_u32 v236, v[vgprG2LScaleZeroA+1], v236, 0x4  // scaleZeroA: z
v_lshl_add_u32 v237, v236, 16, s[sgprScaleALiftMagic+4] // scaleZeroA: 2^7 + c, exact
v_lshl_add_u32 v232, v236, 0, s[sgprScaleALiftMagic+0] // scaleZeroA: 2^23 + c, exact
v_lshl_add_u32 v233, v236, 4, s[sgprScaleALiftMagic+1] // scaleZeroA: 2^19 + c, exact
v_lshl_add_u32 v234, v236, 8, s[sgprScaleALiftMagic+2] // scaleZeroA: 2^15 + c, exact
v_lshl_add_u32 v235, v236, 12, s[sgprScaleALiftMagic+3] // scaleZeroA: 2^11 + c, exact
v_mov_b32 v223, v[vgprG2LA+14+0]                   // w4a16: save packed int4 dword 0
v_and_or_b32 v224, v223, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v225, v223, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v227, v223, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v226, v223, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v224, v232, v224 :: v_dual_subrev_f32 v225, v233, v225 // w4a16: n - c, exact
v_dual_subrev_f32 v227, v234, v227 :: v_dual_subrev_f32 v226, v235, v226 // w4a16: n - c, exact
v_dual_mul_f32 v224, v95, v224 :: v_dual_mul_f32 v225, v222, v225 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v227, v95, v227 :: v_dual_add_nc_u32 v224, s[sgprScaleALiftMagic+5], v224 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v226, v222, v226 :: v_dual_add_nc_u32 v225, s[sgprScaleALiftMagic+5], v225 // w4a16: (n - c) * s :: round
v_and_or_b32 v228, v223, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v223, 20, v223                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v229, v223, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v231, v223, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v230, v223, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v228, v237, v228 :: v_dual_add_nc_u32 v227, s[sgprScaleALiftMagic+5], v227 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v229, v232, v229 :: v_dual_add_nc_u32 v226, s[sgprScaleALiftMagic+5], v226 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v231, v233, v231 :: v_dual_subrev_f32 v230, v234, v230 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+8+0], v225, v224, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+1], v226, v227, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v228, v95, v228 :: v_dual_mul_f32 v229, v222, v229 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v231, v95, v231 :: v_dual_add_nc_u32 v228, s[sgprScaleALiftMagic+5], v228 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v230, v222, v230 :: v_dual_add_nc_u32 v229, s[sgprScaleALiftMagic+5], v229 // w4a16: (n - c) * s :: round
v_add_nc_u32 v231, s[sgprScaleALiftMagic+5], v231  // w4a16: round
v_add_nc_u32 v230, s[sgprScaleALiftMagic+5], v230  // w4a16: round
v_perm_b32 v[vgprG2LA+8+2], v229, v228, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+3], v230, v231, 0x07060302 // w4a16: pack 2 bf16
v_mov_b32 v223, v[vgprG2LA+14+1]                   // w4a16: save packed int4 dword 1
v_and_or_b32 v224, v223, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v225, v223, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v227, v223, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v226, v223, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v224, v232, v224 :: v_dual_subrev_f32 v225, v233, v225 // w4a16: n - c, exact
v_dual_subrev_f32 v227, v234, v227 :: v_dual_subrev_f32 v226, v235, v226 // w4a16: n - c, exact
v_dual_mul_f32 v224, v95, v224 :: v_dual_mul_f32 v225, v222, v225 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v227, v95, v227 :: v_dual_add_nc_u32 v224, s[sgprScaleALiftMagic+5], v224 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v226, v222, v226 :: v_dual_add_nc_u32 v225, s[sgprScaleALiftMagic+5], v225 // w4a16: (n - c) * s :: round
v_and_or_b32 v228, v223, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v223, 20, v223                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v229, v223, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v231, v223, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v230, v223, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v228, v237, v228 :: v_dual_add_nc_u32 v227, s[sgprScaleALiftMagic+5], v227 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v229, v232, v229 :: v_dual_add_nc_u32 v226, s[sgprScaleALiftMagic+5], v226 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v231, v233, v231 :: v_dual_subrev_f32 v230, v234, v230 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+8+4], v225, v224, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+5], v226, v227, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v228, v95, v228 :: v_dual_mul_f32 v229, v222, v229 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v231, v95, v231 :: v_dual_add_nc_u32 v228, s[sgprScaleALiftMagic+5], v228 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v230, v222, v230 :: v_dual_add_nc_u32 v229, s[sgprScaleALiftMagic+5], v229 // w4a16: (n - c) * s :: round
v_add_nc_u32 v231, s[sgprScaleALiftMagic+5], v231  // w4a16: round
v_add_nc_u32 v230, s[sgprScaleALiftMagic+5], v230  // w4a16: round
v_perm_b32 v[vgprG2LA+8+6], v229, v228, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+7], v230, v231, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:4608 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 4608 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+12:vgprG2LA+12+3] offset:4624 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 4608 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(9)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(8)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4:vgprG2LB+4+3] offset:2304 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 2304 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/*  mfmaIndex:38  */
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8:vgprG2LB+8+3] offset:4608 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 4608 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(6)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12:vgprG2LB+12+3] offset:6912 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 6912 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(5)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+16:vgprG2LB+16+3] offset:9216 // lwoB_0_0_4_0 = (0*LSCB)*(MT1J+PAD) + (4*LSPB) = 9216 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(4)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+20:vgprG2LB+20+3] offset:11520 // lwoB_0_0_5_0 = (0*LSCB)*(MT1J+PAD) + (5*LSPB) = 11520 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
/*  mfmaIndex:39  */
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(3)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+24:vgprG2LB+24+3] offset:13824 // lwoB_0_0_6_0 = (0*LSCB)*(MT1J+PAD) + (6*LSPB) = 13824 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(2)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+28:vgprG2LB+28+3] offset:16128 // lwoB_0_0_7_0 = (0*LSCB)*(MT1J+PAD) + (7*LSPB) = 16128 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(1)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+32:vgprG2LB+32+3] offset:18432 // lwoB_0_0_8_0 = (0*LSCB)*(MT1J+PAD) + (8*LSPB) = 18432 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(0)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+36:vgprG2LB+36+3] offset:20736 // lwoB_0_0_9_0 = (0*LSCB)*(MT1J+PAD) + (9*LSPB) = 20736 sync LDS0

/* local write swap offsets a */

/* local write swap offsets b */

/* local read swap offsets a */

/* local read swap offsets b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
/* numPrefetchIter=0 */
/* dataAtIterA=3 numReadsIterA=4 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=3 numReadsIterB=4 skipReadsIterB=0 readsPerIterB=10 */

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
/*  grEndMfmaIndex:0, lwStartMfmaIndex:39, lwEndMfmaIndex:39  */
/*  numMfmaForLR:9, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:38 */
/*  mfmaIndex:0  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4608 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4624 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4608 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4624 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:1  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:9216 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(3)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:2  */
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:9232 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:3  */
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:13824 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:4  */
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:13840 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:5  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:6  */
ds_load_b128 v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprLocalReadAddrB+0] offset:18432 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[vgprLocalReadAddrB+0] offset:18448 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:7  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/*  mfmaIndex:8  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
/*  mfmaIndex:9  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=10 */

/* iter 1 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:39, lwEndMfmaIndex:39  */
/*  numMfmaForLR:9, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:38 */
/*  mfmaIndex:10  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4640 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4656 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4640 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4656 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:11  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:9248 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(3)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:12  */
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:9264 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:13  */
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:13856 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:14  */
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:13872 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:15  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:16  */
ds_load_b128 v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprLocalReadAddrB+0] offset:18464 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[vgprLocalReadAddrB+0] offset:18480 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:17  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/*  mfmaIndex:18  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
/*  mfmaIndex:19  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=0 readsPerIterB=10 */

/* iter 2 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:39, lwEndMfmaIndex:39  */
/*  numMfmaForLR:9, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:38 */
/*  mfmaIndex:20  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4672 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4688 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4672 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4688 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:21  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:9280 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(3)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:22  */
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:9296 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:23  */
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:13888 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:24  */
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:13904 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:25  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:26  */
ds_load_b128 v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprLocalReadAddrB+0] offset:18496 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[vgprLocalReadAddrB+0] offset:18512 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:27  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/*  mfmaIndex:28  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
/*  mfmaIndex:29  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=0 readsPerIterB=10 */

/* iter 3 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:39, lwEndMfmaIndex:39  */
/*  numMfmaForLR:9, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:38 */
/*  mfmaIndex:30  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4704 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4720 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4704 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4720 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:31  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:9312 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(3)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:32  */
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:9328 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:33  */
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:13920 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:34  */
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:13936 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:35  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:36  */
ds_load_b128 v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprLocalReadAddrB+0] offset:18528 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[vgprLocalReadAddrB+0] offset:18544 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:37  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/*  mfmaIndex:38  */
/* schedule remaining localreads for one buffer scheduling */
/* localReadsVacancy: latencyLeft 5 */
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_wmma_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
/*  mfmaIndex:39  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuB_X0_I0+32+0+0:vgprValuB_X0_I0+32+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
/* numPrefetchIter=0 */
/* dataAtIterA=3 numReadsIterA=4 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=3 numReadsIterB=4 skipReadsIterB=0 readsPerIterB=10 */
label_toPGR1end_OrdNLL:
label_PrefetchGlobalLastIterEnd:

/* Tail: add ValuA/B vgpr buffer [106...163) to pool */

/* Tail: add address/G2L vgpr [163...220) to pool */
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
.set sgprShadowLimitA, UNDEF
.set sgprSrdScaleZeroA, UNDEF
.set sgprSrdA, UNDEF
.set sgprSrdB, UNDEF
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
s_load_b64 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 0xa0 // Load batchOffsetC from kernel args
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
s_load_b64 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x98 // Load batchOffsetD from kernel args
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
v_lshrrev_b32 v110, 5, v[vgprSerial]               // 110 = Serial / 32
v_lshrrev_b32 v111, 1, v110                        // 111 = 110 / 2
v_mul_lo_u32 v111, 0x10, v111                      // wave coordination offset 1
v_and_b32 v107, 31, v[vgprSerial]                  // v107 = v[vgprSerial] % 32
v_lshrrev_b32 v107, 4, v107                        // 107 = 107 / 16
                                                   // thread0 * continuous_output (multiplier is 1, do nothing)
v_add_lshl_u32 v107, v111, v107, 0                 // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v108, v107, s[sgprStrideC1J]          //  offset 1
v_mul_lo_u32 v109, v107, s[sgprStrideD1J]          //  offset 1
v_and_b32 v106, 1, v110                            // v106 = v110 % 2
v_mul_lo_u32 v106, 0x10, v106                      // wave coordination offset 0
v_and_b32 v111, 15, v[vgprSerial]                  // v111 = v[vgprSerial] % 16
v_add_lshl_u32 v106, v111, v106, 0                 // coordination 0 = vwA * (wave_id0 + tid0)
s_mul_i32 s7, 64, s[sgprWorkGroup0]                // wgp0 * MT0
v_add_nc_u32 v106, s7, v106                        // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s7, 160, s[sgprWorkGroup1]               // wgp1 * MT1
v_add_nc_u32 v107, s7, v107                        // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1

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
s_and_b32 s68, 63, s[sgprSizeI]                    // s68 = s[sgprSizeI] % 64
s_add_u32 s69, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s69                // wg0 >= nwg0-1 ?
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_MB_Else         // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 160 > 0 -> Edge store; else -> NonEdge store */
s_mul_hi_u32 s71, s[sgprSizeJ], 53687092           // STATIC_DIV: divisor=160
s_mul_i32 s70, s[sgprSizeJ], 53687092              // tmp = dividend * magic
s_lshr_b64 s[70:71], s[70:71], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s69, s70                                 // quotient
s_mul_i32 s70, s69, 160                            // quotient*divisor
s_sub_u32 s68, s[sgprSizeJ], s70                   // rReg = dividend - quotient*divisor
s_add_u32 s69, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s69                // wg1 >= nwg1-1
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_MB_Then         // jump if edges required
label_GW_B0_FD0_VW1_MB_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=136 */
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
v_add_lshl_u32 v117, v109, v106, 2                 // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=106, coord0Vgpr=106 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (19, 0, 0, 0), (19, 1, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0), (22, 0, 0, 0), (22, 1, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0), (24, 0, 0, 0), (24, 1, 0, 0), (25, 0, 0, 0), (25, 1, 0, 0), (26, 0, 0, 0), (26, 1, 0, 0), (27, 0, 0, 0), (27, 1, 0, 0), (28, 0, 0, 0), (28, 1, 0, 0), (29, 0, 0, 0), (29, 1, 0, 0), (30, 0, 0, 0), (30, 1, 0, 0), (31, 0, 0, 0), (31, 1, 0, 0), (32, 0, 0, 0), (32, 1, 0, 0), (33, 0, 0, 0), (33, 1, 0, 0), (34, 0, 0, 0), (34, 1, 0, 0), (35, 0, 0, 0), (35, 1, 0, 0), (36, 0, 0, 0), (36, 1, 0, 0), (37, 0, 0, 0), (37, 1, 0, 0), (38, 0, 0, 0), (38, 1, 0, 0), (39, 0, 0, 0), (39, 1, 0, 0)] */
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
v_mov_b32 v[vgprValuC+185], v[vgprValuC+65]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+186], v[vgprValuC+73]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+187], v[vgprValuC+66]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+188], v[vgprValuC+74]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+189], v[vgprValuC+67]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+190], v[vgprValuC+75]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+191], v[vgprValuC+68]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+192], v[vgprValuC+76]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+193], v[vgprValuC+69]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+194], v[vgprValuC+77]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+195], v[vgprValuC+70]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+196], v[vgprValuC+78]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+197], v[vgprValuC+71]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+198], v[vgprValuC+79]        // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b32 v119, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v120, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v121, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v122, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v123, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v124, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v125, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v126, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v127, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v128, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v129, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v130, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v131, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v132, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v133, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v134, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 72                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b32 v135, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v136, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v137, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v138, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v139, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v140, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v141, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v142, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v143, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v144, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v145, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v146, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v147, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v148, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v149, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v150, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 72                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b32 v151, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v152, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v153, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v154, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v155, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v156, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v157, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v158, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v159, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v160, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v161, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v162, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v163, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v164, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v165, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v166, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 72                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b32 v167, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v168, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v169, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v170, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v171, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v172, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v173, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v174, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v175, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v176, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v177, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v178, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v179, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v180, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v181, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v182, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 72                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b32 v183, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v184, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v185, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v186, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v187, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v188, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v189, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v190, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v191, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v192, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v193, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v194, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v195, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v196, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v197, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v198, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_B0_FD0_VW1_MB_NonEdgeEnd:
label_GW_B0_FD0_VW1_MB_Else:
label_GW_B0_FD0_VW1_MB_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=68 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1); (24,0,0,0:vw1); (24,1,0,0:vw1); (25,0,0,0:vw1); (25,1,0,0:vw1); (26,0,0,0:vw1); (26,1,0,0:vw1); (27,0,0,0:vw1); (27,1,0,0:vw1); (28,0,0,0:vw1); (28,1,0,0:vw1); (29,0,0,0:vw1); (29,1,0,0:vw1); (30,0,0,0:vw1); (30,1,0,0:vw1); (31,0,0,0:vw1); (31,1,0,0:vw1); (32,0,0,0:vw1); (32,1,0,0:vw1); (33,0,0,0:vw1); (33,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v112, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v185, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v185, v112, v185, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v186, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v186, v112, v186, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v187, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v187, v112, v187, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v188, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v188, v112, v188, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v189, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v189, v112, v189, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v190, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v190, v112, v190, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v191, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v191, v112, v191, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v192, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v192, v112, v192, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v193, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v193, v112, v193, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v194, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v194, v112, v194, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v195, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v195, v112, v195, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v196, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v196, v112, v196, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v197, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v197, v112, v197, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v198, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v198, v112, v198, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v199, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v199, v112, v199, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v200, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v200, v112, v200, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 18                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v201, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v201, v112, v201, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v202, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v202, v112, v202, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v203, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v203, v112, v203, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v204, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v204, v112, v204, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v205, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v205, v112, v205, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v206, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v206, v112, v206, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v207, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v207, v112, v207, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v208, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v208, v112, v208, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v209, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v209, v112, v209, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v210, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v210, v112, v210, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v211, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v211, v112, v211, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v212, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v212, v112, v212, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v213, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v213, v112, v213, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v214, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v214, v112, v214, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v215, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v215, v112, v215, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v216, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v216, v112, v216, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 18                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v217, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v217, v112, v217, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v218, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v218, v112, v218, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v219, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v219, v112, v219, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v221, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v221, v112, v221, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v222, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v222, v112, v222, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v223, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v223, v112, v223, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v224, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v224, v112, v224, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v225, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v225, v112, v225, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v226, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v226, v112, v226, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v227, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v227, v112, v227, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v228, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v228, v112, v228, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v229, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v229, v112, v229, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v230, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v230, v112, v230, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v231, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v231, v112, v231, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v232, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v232, v112, v232, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v233, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v233, v112, v233, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 18                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v234, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v234, v112, v234, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v235, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v235, v112, v235, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v236, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v236, v112, v236, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v237, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v237, v112, v237, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v238, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v238, v112, v238, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v239, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v239, v112, v239, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v240, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v240, v112, v240, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v241, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v241, v112, v241, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v242, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v242, v112, v242, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v243, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v243, v112, v243, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v244, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v244, v112, v244, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v245, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v245, v112, v245, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v246, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v246, v112, v246, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v247, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v247, v112, v247, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v248, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v248, v112, v248, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v249, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v249, v112, v249, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(32,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 18                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v250, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v250, v112, v250, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(32,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v251, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v251, v112, v251, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(33,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v252, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v252, v112, v252, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(33,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v253, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v253, v112, v253, s70                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (19, 0, 0, 0), (19, 1, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0), (22, 0, 0, 0), (22, 1, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0), (24, 0, 0, 0), (24, 1, 0, 0), (25, 0, 0, 0), (25, 1, 0, 0), (26, 0, 0, 0), (26, 1, 0, 0), (27, 0, 0, 0), (27, 1, 0, 0), (28, 0, 0, 0), (28, 1, 0, 0), (29, 0, 0, 0), (29, 1, 0, 0), (30, 0, 0, 0), (30, 1, 0, 0), (31, 0, 0, 0), (31, 1, 0, 0), (32, 0, 0, 0), (32, 1, 0, 0), (33, 0, 0, 0), (33, 1, 0, 0)] */
v_mov_b32 v[vgprValuC+117], v[vgprValuC+0]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+118], v[vgprValuC+8]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+119], v[vgprValuC+1]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+120], v[vgprValuC+9]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+121], v[vgprValuC+2]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+122], v[vgprValuC+10]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+123], v[vgprValuC+3]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+124], v[vgprValuC+11]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+125], v[vgprValuC+4]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+126], v[vgprValuC+12]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+127], v[vgprValuC+5]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+128], v[vgprValuC+13]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+129], v[vgprValuC+6]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+130], v[vgprValuC+14]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+131], v[vgprValuC+7]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+132], v[vgprValuC+15]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+133], v[vgprValuC+16]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+134], v[vgprValuC+24]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+135], v[vgprValuC+17]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+136], v[vgprValuC+25]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+137], v[vgprValuC+18]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+138], v[vgprValuC+26]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+139], v[vgprValuC+19]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+140], v[vgprValuC+27]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+141], v[vgprValuC+20]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+142], v[vgprValuC+28]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+143], v[vgprValuC+21]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+144], v[vgprValuC+29]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+145], v[vgprValuC+22]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+146], v[vgprValuC+30]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+147], v[vgprValuC+23]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+148], v[vgprValuC+31]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+149], v[vgprValuC+32]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+150], v[vgprValuC+40]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+151], v[vgprValuC+33]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+152], v[vgprValuC+41]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+153], v[vgprValuC+34]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+154], v[vgprValuC+42]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+155], v[vgprValuC+35]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+156], v[vgprValuC+43]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+157], v[vgprValuC+36]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+158], v[vgprValuC+44]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+159], v[vgprValuC+37]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+160], v[vgprValuC+45]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+161], v[vgprValuC+38]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+162], v[vgprValuC+46]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+163], v[vgprValuC+39]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+164], v[vgprValuC+47]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+165], v[vgprValuC+48]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+166], v[vgprValuC+56]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+167], v[vgprValuC+49]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+168], v[vgprValuC+57]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+169], v[vgprValuC+50]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+170], v[vgprValuC+58]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+171], v[vgprValuC+51]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+172], v[vgprValuC+59]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+173], v[vgprValuC+52]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+174], v[vgprValuC+60]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+175], v[vgprValuC+53]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+176], v[vgprValuC+61]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+177], v[vgprValuC+54]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+178], v[vgprValuC+62]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+179], v[vgprValuC+55]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+180], v[vgprValuC+63]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+181], v[vgprValuC+64]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+182], v[vgprValuC+72]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+183], v[vgprValuC+65]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+184], v[vgprValuC+73]        // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b32 v117, v185, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v118, v186, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v119, v187, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v120, v188, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v121, v189, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v122, v190, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v123, v191, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v124, v192, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v125, v193, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v126, v194, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v127, v195, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v128, v196, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v129, v197, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v130, v198, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v131, v199, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v132, v200, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v133, v201, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v134, v202, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v135, v203, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v136, v204, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v137, v205, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v138, v206, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v139, v207, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v140, v208, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v141, v209, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v142, v210, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v143, v211, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v144, v212, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v145, v213, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v146, v214, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v147, v215, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v148, v216, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v149, v217, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v150, v218, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v151, v219, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v152, v221, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v153, v222, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v154, v223, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v155, v224, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v156, v225, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v157, v226, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v158, v227, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v159, v228, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v160, v229, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v161, v230, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v162, v231, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v163, v232, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v164, v233, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v165, v234, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v166, v235, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v167, v236, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v168, v237, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v169, v238, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v170, v239, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v171, v240, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v172, v241, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v173, v242, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v174, v243, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v175, v244, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v176, v245, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v177, v246, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v178, v247, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v179, v248, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v180, v249, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v181, v250, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v182, v251, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v183, v252, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v184, v253, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (34,0,0,0:vw1); (34,1,0,0:vw1); (35,0,0,0:vw1); (35,1,0,0:vw1); (36,0,0,0:vw1); (36,1,0,0:vw1); (37,0,0,0:vw1); (37,1,0,0:vw1); (38,0,0,0:vw1); (38,1,0,0:vw1); (39,0,0,0:vw1); (39,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v112, BufferOOB
/* (d1,vc1,d0,vc0)=(34,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v129, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v129, v112, v129, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(34,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v130, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v112, v130, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(35,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v131, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v131, v112, v131, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(35,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v132, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v112, v132, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(36,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v133, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v133, v112, v133, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(36,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v134, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v112, v134, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(37,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v135, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v135, v112, v135, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(37,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v136, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v112, v136, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(38,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v137, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v137, v112, v137, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(38,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v138, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v112, v138, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(39,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v139, v109, v106, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v139, v112, v139, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(39,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v140, v109, v110, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v112, v140, s70                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(34, 0, 0, 0), (34, 1, 0, 0), (35, 0, 0, 0), (35, 1, 0, 0), (36, 0, 0, 0), (36, 1, 0, 0), (37, 0, 0, 0), (37, 1, 0, 0), (38, 0, 0, 0), (38, 1, 0, 0), (39, 0, 0, 0), (39, 1, 0, 0)] */
v_mov_b32 v[vgprValuC+117], v[vgprValuC+66]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+118], v[vgprValuC+74]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+119], v[vgprValuC+67]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+120], v[vgprValuC+75]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+121], v[vgprValuC+68]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+122], v[vgprValuC+76]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+123], v[vgprValuC+69]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+124], v[vgprValuC+77]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+125], v[vgprValuC+70]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+126], v[vgprValuC+78]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+127], v[vgprValuC+71]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+128], v[vgprValuC+79]        // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b32 v117, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v118, v130, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v119, v131, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v120, v132, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v121, v133, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v122, v134, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v123, v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v124, v136, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v125, v137, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v126, v138, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v127, v139, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v128, v140, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
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
s_getpc_b64 s[74:75]                               // addr of next instr
s_add_i32 s76, label_SK_Partials_1, 4              // target branch offset
s_add_u32 s74, s74, s76                            // add target branch offset
s_addc_u32 s75, s75, 0                             // add high and carry
s_setpc_b64 s[74:75]                               // branch to label_SK_Partials_1
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
v_mov_b32 v117, s70                                // move flag value to vgpr
v_mov_b32 v118, 0                                  // zero vaddr offset
s_mov_b64 s[76:77], s[sgprAddressFlags:sgprAddressFlags+1]
s_mov_b32 s78, BufferOOB
s_mov_b32 s79, Srd127_96
buffer_store_b32 v117, v118, s[76:79], s68 offen offset:0 glc dlc // reset flag
s_waitcnt_vscnt null 0                             // wait for data store
label_SK_SkipFlagReset:
label_Fixup_E0:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=66 */
s_mov_b64 s[sgprSrdWS:sgprSrdWS+1], s[sgprAddressWS:sgprAddressWS+1]
s_mov_b32 s[sgprSrdWS+2], BufferOOB
s_mov_b32 s[sgprSrdWS+3], Srd127_96

s_mul_i32 s62, 0xa000, s7                          // Offset to correct partials tile (low word)
s_mul_hi_u32 s11, 0xa000, s7                       // partials tile offset (high word) for 64-bit SRD
s_add_u32 s[sgprSrdWS+0], s[sgprSrdWS+0], s62      // add lo to SRD
s_addc_u32 s[sgprSrdWS+1], s[sgprSrdWS+1], s11     // add hi (offset high word + lo carry) to SRD
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Fixup Batch #0 (d1,d0,vc1,vc0) =       */
/*      (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1); (24,0,0,0:vw1); (24,1,0,0:vw1); (25,0,0,0:vw1); (25,1,0,0:vw1); (26,0,0,0:vw1); (26,1,0,0:vw1); (27,0,0,0:vw1); (27,1,0,0:vw1); (28,0,0,0:vw1); (28,1,0,0:vw1); (29,0,0,0:vw1); (29,1,0,0:vw1); (30,0,0,0:vw1); (30,1,0,0:vw1); (31,0,0,0:vw1); (31,1,0,0:vw1); (32,0,0,0:vw1); (32,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_lshlrev_b32 v118, 2, v[vgprSerial]               // v118 = v[vgprSerial] * 4
s_mov_b32 s62, 0                                   // Init sgpr offset
buffer_load_b32 v185, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v186, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v187, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v188, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v189, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v190, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v191, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v192, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v193, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v194, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v195, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v196, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v197, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v198, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v199, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v200, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v201, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v202, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v203, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v204, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v205, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v206, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v207, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v208, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v209, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v210, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v211, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v212, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v213, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v214, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v215, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v216, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v217, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v218, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v219, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v221, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v222, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v223, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v224, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v225, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v226, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v227, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v228, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v229, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v230, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v231, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v232, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v233, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v234, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v235, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v236, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v237, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v238, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v239, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v240, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v241, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v242, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v243, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v244, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v245, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v246, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v247, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v248, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v249, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v250, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v251, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
v_mov_b32 v[vgprValuC+119], v[vgprValuC+0]         // copy MI out reg to vreg[0]
v_mov_b32 v[vgprValuC+120], v[vgprValuC+8]         // copy MI out reg to vreg[1]
v_mov_b32 v[vgprValuC+121], v[vgprValuC+1]         // copy MI out reg to vreg[2]
v_mov_b32 v[vgprValuC+122], v[vgprValuC+9]         // copy MI out reg to vreg[3]
v_mov_b32 v[vgprValuC+123], v[vgprValuC+2]         // copy MI out reg to vreg[4]
v_mov_b32 v[vgprValuC+124], v[vgprValuC+10]        // copy MI out reg to vreg[5]
v_mov_b32 v[vgprValuC+125], v[vgprValuC+3]         // copy MI out reg to vreg[6]
v_mov_b32 v[vgprValuC+126], v[vgprValuC+11]        // copy MI out reg to vreg[7]
v_mov_b32 v[vgprValuC+127], v[vgprValuC+4]         // copy MI out reg to vreg[8]
v_mov_b32 v[vgprValuC+128], v[vgprValuC+12]        // copy MI out reg to vreg[9]
v_mov_b32 v[vgprValuC+129], v[vgprValuC+5]         // copy MI out reg to vreg[10]
v_mov_b32 v[vgprValuC+130], v[vgprValuC+13]        // copy MI out reg to vreg[11]
v_mov_b32 v[vgprValuC+131], v[vgprValuC+6]         // copy MI out reg to vreg[12]
v_mov_b32 v[vgprValuC+132], v[vgprValuC+14]        // copy MI out reg to vreg[13]
v_mov_b32 v[vgprValuC+133], v[vgprValuC+7]         // copy MI out reg to vreg[14]
v_mov_b32 v[vgprValuC+134], v[vgprValuC+15]        // copy MI out reg to vreg[15]
v_mov_b32 v[vgprValuC+135], v[vgprValuC+16]        // copy MI out reg to vreg[16]
v_mov_b32 v[vgprValuC+136], v[vgprValuC+24]        // copy MI out reg to vreg[17]
v_mov_b32 v[vgprValuC+137], v[vgprValuC+17]        // copy MI out reg to vreg[18]
v_mov_b32 v[vgprValuC+138], v[vgprValuC+25]        // copy MI out reg to vreg[19]
v_mov_b32 v[vgprValuC+139], v[vgprValuC+18]        // copy MI out reg to vreg[20]
v_mov_b32 v[vgprValuC+140], v[vgprValuC+26]        // copy MI out reg to vreg[21]
v_mov_b32 v[vgprValuC+141], v[vgprValuC+19]        // copy MI out reg to vreg[22]
v_mov_b32 v[vgprValuC+142], v[vgprValuC+27]        // copy MI out reg to vreg[23]
v_mov_b32 v[vgprValuC+143], v[vgprValuC+20]        // copy MI out reg to vreg[24]
v_mov_b32 v[vgprValuC+144], v[vgprValuC+28]        // copy MI out reg to vreg[25]
v_mov_b32 v[vgprValuC+145], v[vgprValuC+21]        // copy MI out reg to vreg[26]
v_mov_b32 v[vgprValuC+146], v[vgprValuC+29]        // copy MI out reg to vreg[27]
v_mov_b32 v[vgprValuC+147], v[vgprValuC+22]        // copy MI out reg to vreg[28]
v_mov_b32 v[vgprValuC+148], v[vgprValuC+30]        // copy MI out reg to vreg[29]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+23]        // copy MI out reg to vreg[30]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+31]        // copy MI out reg to vreg[31]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+32]        // copy MI out reg to vreg[32]
v_mov_b32 v[vgprValuC+152], v[vgprValuC+40]        // copy MI out reg to vreg[33]
v_mov_b32 v[vgprValuC+153], v[vgprValuC+33]        // copy MI out reg to vreg[34]
v_mov_b32 v[vgprValuC+154], v[vgprValuC+41]        // copy MI out reg to vreg[35]
v_mov_b32 v[vgprValuC+155], v[vgprValuC+34]        // copy MI out reg to vreg[36]
v_mov_b32 v[vgprValuC+156], v[vgprValuC+42]        // copy MI out reg to vreg[37]
v_mov_b32 v[vgprValuC+157], v[vgprValuC+35]        // copy MI out reg to vreg[38]
v_mov_b32 v[vgprValuC+158], v[vgprValuC+43]        // copy MI out reg to vreg[39]
v_mov_b32 v[vgprValuC+159], v[vgprValuC+36]        // copy MI out reg to vreg[40]
v_mov_b32 v[vgprValuC+160], v[vgprValuC+44]        // copy MI out reg to vreg[41]
v_mov_b32 v[vgprValuC+161], v[vgprValuC+37]        // copy MI out reg to vreg[42]
v_mov_b32 v[vgprValuC+162], v[vgprValuC+45]        // copy MI out reg to vreg[43]
v_mov_b32 v[vgprValuC+163], v[vgprValuC+38]        // copy MI out reg to vreg[44]
v_mov_b32 v[vgprValuC+164], v[vgprValuC+46]        // copy MI out reg to vreg[45]
v_mov_b32 v[vgprValuC+165], v[vgprValuC+39]        // copy MI out reg to vreg[46]
v_mov_b32 v[vgprValuC+166], v[vgprValuC+47]        // copy MI out reg to vreg[47]
v_mov_b32 v[vgprValuC+167], v[vgprValuC+48]        // copy MI out reg to vreg[48]
v_mov_b32 v[vgprValuC+168], v[vgprValuC+56]        // copy MI out reg to vreg[49]
v_mov_b32 v[vgprValuC+169], v[vgprValuC+49]        // copy MI out reg to vreg[50]
v_mov_b32 v[vgprValuC+170], v[vgprValuC+57]        // copy MI out reg to vreg[51]
v_mov_b32 v[vgprValuC+171], v[vgprValuC+50]        // copy MI out reg to vreg[52]
v_mov_b32 v[vgprValuC+172], v[vgprValuC+58]        // copy MI out reg to vreg[53]
v_mov_b32 v[vgprValuC+173], v[vgprValuC+51]        // copy MI out reg to vreg[54]
v_mov_b32 v[vgprValuC+174], v[vgprValuC+59]        // copy MI out reg to vreg[55]
v_mov_b32 v[vgprValuC+175], v[vgprValuC+52]        // copy MI out reg to vreg[56]
v_mov_b32 v[vgprValuC+176], v[vgprValuC+60]        // copy MI out reg to vreg[57]
v_mov_b32 v[vgprValuC+177], v[vgprValuC+53]        // copy MI out reg to vreg[58]
v_mov_b32 v[vgprValuC+178], v[vgprValuC+61]        // copy MI out reg to vreg[59]
v_mov_b32 v[vgprValuC+179], v[vgprValuC+54]        // copy MI out reg to vreg[60]
v_mov_b32 v[vgprValuC+180], v[vgprValuC+62]        // copy MI out reg to vreg[61]
v_mov_b32 v[vgprValuC+181], v[vgprValuC+55]        // copy MI out reg to vreg[62]
v_mov_b32 v[vgprValuC+182], v[vgprValuC+63]        // copy MI out reg to vreg[63]
v_mov_b32 v[vgprValuC+183], v[vgprValuC+64]        // copy MI out reg to vreg[64]
v_mov_b32 v[vgprValuC+184], v[vgprValuC+72]        // copy MI out reg to vreg[65]

/* apply mask, calc new C and issue writes */
v_mov_b32 v114, 0xffff0000                         // mask for pack two bfloat16 element to 32bit
v_mov_b32 v115, 0x7fff0000                         // fp32 Nan
v_mov_b32 v116, 0x7fff                             // rounding bias for bfloat16

s_waitcnt vmcnt(63)                                // wait C (interleaved) 65 = 66 - 0 - 1
v_add_f32 v[vgprValuC+119], v[vgprValuC+119], v185 // accum partials

s_waitcnt vmcnt(63)                                // wait C (interleaved) 64 = 66 - 1 - 1
v_add_f32 v[vgprValuC+120], v[vgprValuC+120], v186 // accum partials

s_waitcnt vmcnt(63)                                // wait C (interleaved) 63 = 66 - 2 - 1
v_add_f32 v[vgprValuC+121], v[vgprValuC+121], v187 // accum partials

s_waitcnt vmcnt(62)                                // wait C (interleaved) 62 = 66 - 3 - 1
v_add_f32 v[vgprValuC+122], v[vgprValuC+122], v188 // accum partials

s_waitcnt vmcnt(61)                                // wait C (interleaved) 61 = 66 - 4 - 1
v_add_f32 v[vgprValuC+123], v[vgprValuC+123], v189 // accum partials

s_waitcnt vmcnt(60)                                // wait C (interleaved) 60 = 66 - 5 - 1
v_add_f32 v[vgprValuC+124], v[vgprValuC+124], v190 // accum partials

s_waitcnt vmcnt(59)                                // wait C (interleaved) 59 = 66 - 6 - 1
v_add_f32 v[vgprValuC+125], v[vgprValuC+125], v191 // accum partials

s_waitcnt vmcnt(58)                                // wait C (interleaved) 58 = 66 - 7 - 1
v_add_f32 v[vgprValuC+126], v[vgprValuC+126], v192 // accum partials

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 66 - 8 - 1
v_add_f32 v[vgprValuC+127], v[vgprValuC+127], v193 // accum partials

s_waitcnt vmcnt(56)                                // wait C (interleaved) 56 = 66 - 9 - 1
v_add_f32 v[vgprValuC+128], v[vgprValuC+128], v194 // accum partials

s_waitcnt vmcnt(55)                                // wait C (interleaved) 55 = 66 - 10 - 1
v_add_f32 v[vgprValuC+129], v[vgprValuC+129], v195 // accum partials

s_waitcnt vmcnt(54)                                // wait C (interleaved) 54 = 66 - 11 - 1
v_add_f32 v[vgprValuC+130], v[vgprValuC+130], v196 // accum partials

s_waitcnt vmcnt(53)                                // wait C (interleaved) 53 = 66 - 12 - 1
v_add_f32 v[vgprValuC+131], v[vgprValuC+131], v197 // accum partials

s_waitcnt vmcnt(52)                                // wait C (interleaved) 52 = 66 - 13 - 1
v_add_f32 v[vgprValuC+132], v[vgprValuC+132], v198 // accum partials

s_waitcnt vmcnt(51)                                // wait C (interleaved) 51 = 66 - 14 - 1
v_add_f32 v[vgprValuC+133], v[vgprValuC+133], v199 // accum partials

s_waitcnt vmcnt(50)                                // wait C (interleaved) 50 = 66 - 15 - 1
v_add_f32 v[vgprValuC+134], v[vgprValuC+134], v200 // accum partials

s_waitcnt vmcnt(49)                                // wait C (interleaved) 49 = 66 - 16 - 1
v_add_f32 v[vgprValuC+135], v[vgprValuC+135], v201 // accum partials

s_waitcnt vmcnt(48)                                // wait C (interleaved) 48 = 66 - 17 - 1
v_add_f32 v[vgprValuC+136], v[vgprValuC+136], v202 // accum partials

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 66 - 18 - 1
v_add_f32 v[vgprValuC+137], v[vgprValuC+137], v203 // accum partials

s_waitcnt vmcnt(46)                                // wait C (interleaved) 46 = 66 - 19 - 1
v_add_f32 v[vgprValuC+138], v[vgprValuC+138], v204 // accum partials

s_waitcnt vmcnt(45)                                // wait C (interleaved) 45 = 66 - 20 - 1
v_add_f32 v[vgprValuC+139], v[vgprValuC+139], v205 // accum partials

s_waitcnt vmcnt(44)                                // wait C (interleaved) 44 = 66 - 21 - 1
v_add_f32 v[vgprValuC+140], v[vgprValuC+140], v206 // accum partials

s_waitcnt vmcnt(43)                                // wait C (interleaved) 43 = 66 - 22 - 1
v_add_f32 v[vgprValuC+141], v[vgprValuC+141], v207 // accum partials

s_waitcnt vmcnt(42)                                // wait C (interleaved) 42 = 66 - 23 - 1
v_add_f32 v[vgprValuC+142], v[vgprValuC+142], v208 // accum partials

s_waitcnt vmcnt(41)                                // wait C (interleaved) 41 = 66 - 24 - 1
v_add_f32 v[vgprValuC+143], v[vgprValuC+143], v209 // accum partials

s_waitcnt vmcnt(40)                                // wait C (interleaved) 40 = 66 - 25 - 1
v_add_f32 v[vgprValuC+144], v[vgprValuC+144], v210 // accum partials

s_waitcnt vmcnt(39)                                // wait C (interleaved) 39 = 66 - 26 - 1
v_add_f32 v[vgprValuC+145], v[vgprValuC+145], v211 // accum partials

s_waitcnt vmcnt(38)                                // wait C (interleaved) 38 = 66 - 27 - 1
v_add_f32 v[vgprValuC+146], v[vgprValuC+146], v212 // accum partials

s_waitcnt vmcnt(37)                                // wait C (interleaved) 37 = 66 - 28 - 1
v_add_f32 v[vgprValuC+147], v[vgprValuC+147], v213 // accum partials

s_waitcnt vmcnt(36)                                // wait C (interleaved) 36 = 66 - 29 - 1
v_add_f32 v[vgprValuC+148], v[vgprValuC+148], v214 // accum partials

s_waitcnt vmcnt(35)                                // wait C (interleaved) 35 = 66 - 30 - 1
v_add_f32 v[vgprValuC+149], v[vgprValuC+149], v215 // accum partials

s_waitcnt vmcnt(34)                                // wait C (interleaved) 34 = 66 - 31 - 1
v_add_f32 v[vgprValuC+150], v[vgprValuC+150], v216 // accum partials

s_waitcnt vmcnt(33)                                // wait C (interleaved) 33 = 66 - 32 - 1
v_add_f32 v[vgprValuC+151], v[vgprValuC+151], v217 // accum partials

s_waitcnt vmcnt(32)                                // wait C (interleaved) 32 = 66 - 33 - 1
v_add_f32 v[vgprValuC+152], v[vgprValuC+152], v218 // accum partials

s_waitcnt vmcnt(31)                                // wait C (interleaved) 31 = 66 - 34 - 1
v_add_f32 v[vgprValuC+153], v[vgprValuC+153], v219 // accum partials

s_waitcnt vmcnt(30)                                // wait C (interleaved) 30 = 66 - 35 - 1
v_add_f32 v[vgprValuC+154], v[vgprValuC+154], v221 // accum partials

s_waitcnt vmcnt(29)                                // wait C (interleaved) 29 = 66 - 36 - 1
v_add_f32 v[vgprValuC+155], v[vgprValuC+155], v222 // accum partials

s_waitcnt vmcnt(28)                                // wait C (interleaved) 28 = 66 - 37 - 1
v_add_f32 v[vgprValuC+156], v[vgprValuC+156], v223 // accum partials

s_waitcnt vmcnt(27)                                // wait C (interleaved) 27 = 66 - 38 - 1
v_add_f32 v[vgprValuC+157], v[vgprValuC+157], v224 // accum partials

s_waitcnt vmcnt(26)                                // wait C (interleaved) 26 = 66 - 39 - 1
v_add_f32 v[vgprValuC+158], v[vgprValuC+158], v225 // accum partials

s_waitcnt vmcnt(25)                                // wait C (interleaved) 25 = 66 - 40 - 1
v_add_f32 v[vgprValuC+159], v[vgprValuC+159], v226 // accum partials

s_waitcnt vmcnt(24)                                // wait C (interleaved) 24 = 66 - 41 - 1
v_add_f32 v[vgprValuC+160], v[vgprValuC+160], v227 // accum partials

s_waitcnt vmcnt(23)                                // wait C (interleaved) 23 = 66 - 42 - 1
v_add_f32 v[vgprValuC+161], v[vgprValuC+161], v228 // accum partials

s_waitcnt vmcnt(22)                                // wait C (interleaved) 22 = 66 - 43 - 1
v_add_f32 v[vgprValuC+162], v[vgprValuC+162], v229 // accum partials

s_waitcnt vmcnt(21)                                // wait C (interleaved) 21 = 66 - 44 - 1
v_add_f32 v[vgprValuC+163], v[vgprValuC+163], v230 // accum partials

s_waitcnt vmcnt(20)                                // wait C (interleaved) 20 = 66 - 45 - 1
v_add_f32 v[vgprValuC+164], v[vgprValuC+164], v231 // accum partials

s_waitcnt vmcnt(19)                                // wait C (interleaved) 19 = 66 - 46 - 1
v_add_f32 v[vgprValuC+165], v[vgprValuC+165], v232 // accum partials

s_waitcnt vmcnt(18)                                // wait C (interleaved) 18 = 66 - 47 - 1
v_add_f32 v[vgprValuC+166], v[vgprValuC+166], v233 // accum partials

s_waitcnt vmcnt(17)                                // wait C (interleaved) 17 = 66 - 48 - 1
v_add_f32 v[vgprValuC+167], v[vgprValuC+167], v234 // accum partials

s_waitcnt vmcnt(16)                                // wait C (interleaved) 16 = 66 - 49 - 1
v_add_f32 v[vgprValuC+168], v[vgprValuC+168], v235 // accum partials

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 66 - 50 - 1
v_add_f32 v[vgprValuC+169], v[vgprValuC+169], v236 // accum partials

s_waitcnt vmcnt(14)                                // wait C (interleaved) 14 = 66 - 51 - 1
v_add_f32 v[vgprValuC+170], v[vgprValuC+170], v237 // accum partials

s_waitcnt vmcnt(13)                                // wait C (interleaved) 13 = 66 - 52 - 1
v_add_f32 v[vgprValuC+171], v[vgprValuC+171], v238 // accum partials

s_waitcnt vmcnt(12)                                // wait C (interleaved) 12 = 66 - 53 - 1
v_add_f32 v[vgprValuC+172], v[vgprValuC+172], v239 // accum partials

s_waitcnt vmcnt(11)                                // wait C (interleaved) 11 = 66 - 54 - 1
v_add_f32 v[vgprValuC+173], v[vgprValuC+173], v240 // accum partials

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 66 - 55 - 1
v_add_f32 v[vgprValuC+174], v[vgprValuC+174], v241 // accum partials

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 66 - 56 - 1
v_add_f32 v[vgprValuC+175], v[vgprValuC+175], v242 // accum partials

s_waitcnt vmcnt(8)                                 // wait C (interleaved) 8 = 66 - 57 - 1
v_add_f32 v[vgprValuC+176], v[vgprValuC+176], v243 // accum partials

s_waitcnt vmcnt(7)                                 // wait C (interleaved) 7 = 66 - 58 - 1
v_add_f32 v[vgprValuC+177], v[vgprValuC+177], v244 // accum partials

s_waitcnt vmcnt(6)                                 // wait C (interleaved) 6 = 66 - 59 - 1
v_add_f32 v[vgprValuC+178], v[vgprValuC+178], v245 // accum partials

s_waitcnt vmcnt(5)                                 // wait C (interleaved) 5 = 66 - 60 - 1
v_add_f32 v[vgprValuC+179], v[vgprValuC+179], v246 // accum partials

s_waitcnt vmcnt(4)                                 // wait C (interleaved) 4 = 66 - 61 - 1
v_add_f32 v[vgprValuC+180], v[vgprValuC+180], v247 // accum partials

s_waitcnt vmcnt(3)                                 // wait C (interleaved) 3 = 66 - 62 - 1
v_add_f32 v[vgprValuC+181], v[vgprValuC+181], v248 // accum partials

s_waitcnt vmcnt(2)                                 // wait C (interleaved) 2 = 66 - 63 - 1
v_add_f32 v[vgprValuC+182], v[vgprValuC+182], v249 // accum partials

s_waitcnt vmcnt(1)                                 // wait C (interleaved) 1 = 66 - 64 - 1
v_add_f32 v[vgprValuC+183], v[vgprValuC+183], v250 // accum partials

s_waitcnt vmcnt(0)                                 // wait C (interleaved) 0 = 66 - 65 - 1
v_add_f32 v[vgprValuC+184], v[vgprValuC+184], v251 // accum partials
v_mov_b32 v[vgprValuC+0], v[vgprValuC+119]         // copy vreg[0] to MI out reg
v_mov_b32 v[vgprValuC+8], v[vgprValuC+120]         // copy vreg[1] to MI out reg
v_mov_b32 v[vgprValuC+1], v[vgprValuC+121]         // copy vreg[2] to MI out reg
v_mov_b32 v[vgprValuC+9], v[vgprValuC+122]         // copy vreg[3] to MI out reg
v_mov_b32 v[vgprValuC+2], v[vgprValuC+123]         // copy vreg[4] to MI out reg
v_mov_b32 v[vgprValuC+10], v[vgprValuC+124]        // copy vreg[5] to MI out reg
v_mov_b32 v[vgprValuC+3], v[vgprValuC+125]         // copy vreg[6] to MI out reg
v_mov_b32 v[vgprValuC+11], v[vgprValuC+126]        // copy vreg[7] to MI out reg
v_mov_b32 v[vgprValuC+4], v[vgprValuC+127]         // copy vreg[8] to MI out reg
v_mov_b32 v[vgprValuC+12], v[vgprValuC+128]        // copy vreg[9] to MI out reg
v_mov_b32 v[vgprValuC+5], v[vgprValuC+129]         // copy vreg[10] to MI out reg
v_mov_b32 v[vgprValuC+13], v[vgprValuC+130]        // copy vreg[11] to MI out reg
v_mov_b32 v[vgprValuC+6], v[vgprValuC+131]         // copy vreg[12] to MI out reg
v_mov_b32 v[vgprValuC+14], v[vgprValuC+132]        // copy vreg[13] to MI out reg
v_mov_b32 v[vgprValuC+7], v[vgprValuC+133]         // copy vreg[14] to MI out reg
v_mov_b32 v[vgprValuC+15], v[vgprValuC+134]        // copy vreg[15] to MI out reg
v_mov_b32 v[vgprValuC+16], v[vgprValuC+135]        // copy vreg[16] to MI out reg
v_mov_b32 v[vgprValuC+24], v[vgprValuC+136]        // copy vreg[17] to MI out reg
v_mov_b32 v[vgprValuC+17], v[vgprValuC+137]        // copy vreg[18] to MI out reg
v_mov_b32 v[vgprValuC+25], v[vgprValuC+138]        // copy vreg[19] to MI out reg
v_mov_b32 v[vgprValuC+18], v[vgprValuC+139]        // copy vreg[20] to MI out reg
v_mov_b32 v[vgprValuC+26], v[vgprValuC+140]        // copy vreg[21] to MI out reg
v_mov_b32 v[vgprValuC+19], v[vgprValuC+141]        // copy vreg[22] to MI out reg
v_mov_b32 v[vgprValuC+27], v[vgprValuC+142]        // copy vreg[23] to MI out reg
v_mov_b32 v[vgprValuC+20], v[vgprValuC+143]        // copy vreg[24] to MI out reg
v_mov_b32 v[vgprValuC+28], v[vgprValuC+144]        // copy vreg[25] to MI out reg
v_mov_b32 v[vgprValuC+21], v[vgprValuC+145]        // copy vreg[26] to MI out reg
v_mov_b32 v[vgprValuC+29], v[vgprValuC+146]        // copy vreg[27] to MI out reg
v_mov_b32 v[vgprValuC+22], v[vgprValuC+147]        // copy vreg[28] to MI out reg
v_mov_b32 v[vgprValuC+30], v[vgprValuC+148]        // copy vreg[29] to MI out reg
v_mov_b32 v[vgprValuC+23], v[vgprValuC+149]        // copy vreg[30] to MI out reg
v_mov_b32 v[vgprValuC+31], v[vgprValuC+150]        // copy vreg[31] to MI out reg
v_mov_b32 v[vgprValuC+32], v[vgprValuC+151]        // copy vreg[32] to MI out reg
v_mov_b32 v[vgprValuC+40], v[vgprValuC+152]        // copy vreg[33] to MI out reg
v_mov_b32 v[vgprValuC+33], v[vgprValuC+153]        // copy vreg[34] to MI out reg
v_mov_b32 v[vgprValuC+41], v[vgprValuC+154]        // copy vreg[35] to MI out reg
v_mov_b32 v[vgprValuC+34], v[vgprValuC+155]        // copy vreg[36] to MI out reg
v_mov_b32 v[vgprValuC+42], v[vgprValuC+156]        // copy vreg[37] to MI out reg
v_mov_b32 v[vgprValuC+35], v[vgprValuC+157]        // copy vreg[38] to MI out reg
v_mov_b32 v[vgprValuC+43], v[vgprValuC+158]        // copy vreg[39] to MI out reg
v_mov_b32 v[vgprValuC+36], v[vgprValuC+159]        // copy vreg[40] to MI out reg
v_mov_b32 v[vgprValuC+44], v[vgprValuC+160]        // copy vreg[41] to MI out reg
v_mov_b32 v[vgprValuC+37], v[vgprValuC+161]        // copy vreg[42] to MI out reg
v_mov_b32 v[vgprValuC+45], v[vgprValuC+162]        // copy vreg[43] to MI out reg
v_mov_b32 v[vgprValuC+38], v[vgprValuC+163]        // copy vreg[44] to MI out reg
v_mov_b32 v[vgprValuC+46], v[vgprValuC+164]        // copy vreg[45] to MI out reg
v_mov_b32 v[vgprValuC+39], v[vgprValuC+165]        // copy vreg[46] to MI out reg
v_mov_b32 v[vgprValuC+47], v[vgprValuC+166]        // copy vreg[47] to MI out reg
v_mov_b32 v[vgprValuC+48], v[vgprValuC+167]        // copy vreg[48] to MI out reg
v_mov_b32 v[vgprValuC+56], v[vgprValuC+168]        // copy vreg[49] to MI out reg
v_mov_b32 v[vgprValuC+49], v[vgprValuC+169]        // copy vreg[50] to MI out reg
v_mov_b32 v[vgprValuC+57], v[vgprValuC+170]        // copy vreg[51] to MI out reg
v_mov_b32 v[vgprValuC+50], v[vgprValuC+171]        // copy vreg[52] to MI out reg
v_mov_b32 v[vgprValuC+58], v[vgprValuC+172]        // copy vreg[53] to MI out reg
v_mov_b32 v[vgprValuC+51], v[vgprValuC+173]        // copy vreg[54] to MI out reg
v_mov_b32 v[vgprValuC+59], v[vgprValuC+174]        // copy vreg[55] to MI out reg
v_mov_b32 v[vgprValuC+52], v[vgprValuC+175]        // copy vreg[56] to MI out reg
v_mov_b32 v[vgprValuC+60], v[vgprValuC+176]        // copy vreg[57] to MI out reg
v_mov_b32 v[vgprValuC+53], v[vgprValuC+177]        // copy vreg[58] to MI out reg
v_mov_b32 v[vgprValuC+61], v[vgprValuC+178]        // copy vreg[59] to MI out reg
v_mov_b32 v[vgprValuC+54], v[vgprValuC+179]        // copy vreg[60] to MI out reg
v_mov_b32 v[vgprValuC+62], v[vgprValuC+180]        // copy vreg[61] to MI out reg
v_mov_b32 v[vgprValuC+55], v[vgprValuC+181]        // copy vreg[62] to MI out reg
v_mov_b32 v[vgprValuC+63], v[vgprValuC+182]        // copy vreg[63] to MI out reg
v_mov_b32 v[vgprValuC+64], v[vgprValuC+183]        // copy vreg[64] to MI out reg
v_mov_b32 v[vgprValuC+72], v[vgprValuC+184]        // copy vreg[65] to MI out reg
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Fixup Batch #1 (d1,d0,vc1,vc0) =       */
/*      (33,0,0,0:vw1); (33,1,0,0:vw1); (34,0,0,0:vw1); (34,1,0,0:vw1); (35,0,0,0:vw1); (35,1,0,0:vw1); (36,0,0,0:vw1); (36,1,0,0:vw1); (37,0,0,0:vw1); (37,1,0,0:vw1); (38,0,0,0:vw1); (38,1,0,0:vw1); (39,0,0,0:vw1); (39,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v133, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v134, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v135, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v136, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v137, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v138, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v139, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v140, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v141, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v142, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v143, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v144, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v145, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v146, v118, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
v_mov_b32 v[vgprValuC+119], v[vgprValuC+65]        // copy MI out reg to vreg[66]
v_mov_b32 v[vgprValuC+120], v[vgprValuC+73]        // copy MI out reg to vreg[67]
v_mov_b32 v[vgprValuC+121], v[vgprValuC+66]        // copy MI out reg to vreg[68]
v_mov_b32 v[vgprValuC+122], v[vgprValuC+74]        // copy MI out reg to vreg[69]
v_mov_b32 v[vgprValuC+123], v[vgprValuC+67]        // copy MI out reg to vreg[70]
v_mov_b32 v[vgprValuC+124], v[vgprValuC+75]        // copy MI out reg to vreg[71]
v_mov_b32 v[vgprValuC+125], v[vgprValuC+68]        // copy MI out reg to vreg[72]
v_mov_b32 v[vgprValuC+126], v[vgprValuC+76]        // copy MI out reg to vreg[73]
v_mov_b32 v[vgprValuC+127], v[vgprValuC+69]        // copy MI out reg to vreg[74]
v_mov_b32 v[vgprValuC+128], v[vgprValuC+77]        // copy MI out reg to vreg[75]
v_mov_b32 v[vgprValuC+129], v[vgprValuC+70]        // copy MI out reg to vreg[76]
v_mov_b32 v[vgprValuC+130], v[vgprValuC+78]        // copy MI out reg to vreg[77]
v_mov_b32 v[vgprValuC+131], v[vgprValuC+71]        // copy MI out reg to vreg[78]
v_mov_b32 v[vgprValuC+132], v[vgprValuC+79]        // copy MI out reg to vreg[79]

/* apply mask, calc new C and issue writes */
v_mov_b32 v114, 0xffff0000                         // mask for pack two bfloat16 element to 32bit
v_mov_b32 v115, 0x7fff0000                         // fp32 Nan
v_mov_b32 v116, 0x7fff                             // rounding bias for bfloat16

s_waitcnt vmcnt(13)                                // wait C (interleaved) 13 = 14 - 0 - 1
v_add_f32 v[vgprValuC+119], v[vgprValuC+119], v133 // accum partials

s_waitcnt vmcnt(12)                                // wait C (interleaved) 12 = 14 - 1 - 1
v_add_f32 v[vgprValuC+120], v[vgprValuC+120], v134 // accum partials

s_waitcnt vmcnt(11)                                // wait C (interleaved) 11 = 14 - 2 - 1
v_add_f32 v[vgprValuC+121], v[vgprValuC+121], v135 // accum partials

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 14 - 3 - 1
v_add_f32 v[vgprValuC+122], v[vgprValuC+122], v136 // accum partials

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 14 - 4 - 1
v_add_f32 v[vgprValuC+123], v[vgprValuC+123], v137 // accum partials

s_waitcnt vmcnt(8)                                 // wait C (interleaved) 8 = 14 - 5 - 1
v_add_f32 v[vgprValuC+124], v[vgprValuC+124], v138 // accum partials

s_waitcnt vmcnt(7)                                 // wait C (interleaved) 7 = 14 - 6 - 1
v_add_f32 v[vgprValuC+125], v[vgprValuC+125], v139 // accum partials

s_waitcnt vmcnt(6)                                 // wait C (interleaved) 6 = 14 - 7 - 1
v_add_f32 v[vgprValuC+126], v[vgprValuC+126], v140 // accum partials

s_waitcnt vmcnt(5)                                 // wait C (interleaved) 5 = 14 - 8 - 1
v_add_f32 v[vgprValuC+127], v[vgprValuC+127], v141 // accum partials

s_waitcnt vmcnt(4)                                 // wait C (interleaved) 4 = 14 - 9 - 1
v_add_f32 v[vgprValuC+128], v[vgprValuC+128], v142 // accum partials

s_waitcnt vmcnt(3)                                 // wait C (interleaved) 3 = 14 - 10 - 1
v_add_f32 v[vgprValuC+129], v[vgprValuC+129], v143 // accum partials

s_waitcnt vmcnt(2)                                 // wait C (interleaved) 2 = 14 - 11 - 1
v_add_f32 v[vgprValuC+130], v[vgprValuC+130], v144 // accum partials

s_waitcnt vmcnt(1)                                 // wait C (interleaved) 1 = 14 - 12 - 1
v_add_f32 v[vgprValuC+131], v[vgprValuC+131], v145 // accum partials

s_waitcnt vmcnt(0)                                 // wait C (interleaved) 0 = 14 - 13 - 1
v_add_f32 v[vgprValuC+132], v[vgprValuC+132], v146 // accum partials
v_mov_b32 v[vgprValuC+65], v[vgprValuC+119]        // copy vreg[66] to MI out reg
v_mov_b32 v[vgprValuC+73], v[vgprValuC+120]        // copy vreg[67] to MI out reg
v_mov_b32 v[vgprValuC+66], v[vgprValuC+121]        // copy vreg[68] to MI out reg
v_mov_b32 v[vgprValuC+74], v[vgprValuC+122]        // copy vreg[69] to MI out reg
v_mov_b32 v[vgprValuC+67], v[vgprValuC+123]        // copy vreg[70] to MI out reg
v_mov_b32 v[vgprValuC+75], v[vgprValuC+124]        // copy vreg[71] to MI out reg
v_mov_b32 v[vgprValuC+68], v[vgprValuC+125]        // copy vreg[72] to MI out reg
v_mov_b32 v[vgprValuC+76], v[vgprValuC+126]        // copy vreg[73] to MI out reg
v_mov_b32 v[vgprValuC+69], v[vgprValuC+127]        // copy vreg[74] to MI out reg
v_mov_b32 v[vgprValuC+77], v[vgprValuC+128]        // copy vreg[75] to MI out reg
v_mov_b32 v[vgprValuC+70], v[vgprValuC+129]        // copy vreg[76] to MI out reg
v_mov_b32 v[vgprValuC+78], v[vgprValuC+130]        // copy vreg[77] to MI out reg
v_mov_b32 v[vgprValuC+71], v[vgprValuC+131]        // copy vreg[78] to MI out reg
v_mov_b32 v[vgprValuC+79], v[vgprValuC+132]        // copy vreg[79] to MI out reg
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_mul_i32 s11, s[sgprskTiles], s[sgprItersPerTile]
s_mul_i32 s53, s[sgprSKItersPerWG], s[sgprskGrid]
s_sub_u32 s11, s11, s53                            // skTiles * ItersPerTile - SKItersPerWG * skGrid
s_bitcmp1_b32 s[sgprMagicShiftItersPerTile], 29    // USO on? (bit 29 of MagicShiftItersPerTile); off -> historical global peer size
s_cbranch_scc0 label_SK_PeerGlobal                 // USO off -> historical global mapping
s_cmp_eq_u32 s[sgprskTiles], 0                     // skTiles == 0?
s_cbranch_scc1 label_SK_PeerNoTiles                // global peer size
v_cvt_f32_u32 v117, s[sgprskTiles]                 // skGrid % skTiles
v_rcp_iflag_f32 v117, v117                         // skGrid % skTiles
v_mov_b32 v118, 0x4f7ffffe                         // 2^32 - 512
v_mul_f32 v117, v118, v117                         // 2^32 / divisor
v_cvt_u32_f32 v117, v117                           // skGrid % skTiles
v_mul_lo_u32 v118, v117, s[sgprskTiles]            // refine 2^32 / divisor
v_sub_nc_u32 v118, 0, v118                         // refine 2^32 / divisor
v_mul_hi_u32 v118, v117, v118                      // refine 2^32 / divisor
v_add_nc_u32 v117, v117, v118                      // refine 2^32 / divisor
v_mul_hi_u32 v117, s[sgprskGrid], v117             // skGrid % skTiles
v_mul_lo_u32 v118, v117, s[sgprskTiles]            // skGrid % skTiles
v_sub_nc_u32 v118, s[sgprskGrid], v118             // skGrid % skTiles
v_cmp_ge_u32 vcc_lo, v118, s[sgprskTiles]          // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v117, 1, v117                         // quotient + 1
v_sub_nc_u32 v118, v118, s[sgprskTiles]            // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v118, s[sgprskTiles]          // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v117, 1, v117                         // quotient + 1
v_sub_nc_u32 v118, v118, s[sgprskTiles]            // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s53, v117                      // quotient
v_readfirstlane_b32 s53, v118                      // remainder
s_cmp_eq_u32 s53, 0                                // skGrid % skTiles == 0?
s_cbranch_scc0 label_SK_PeerGlobal                 // ragged -> global peer
label_SK_PeerPerTile:
v_cvt_f32_u32 v117, s[sgprskTiles]                 // F = skGrid / skTiles
v_rcp_iflag_f32 v117, v117                         // F = skGrid / skTiles
v_mov_b32 v118, 0x4f7ffffe                         // 2^32 - 512
v_mul_f32 v117, v118, v117                         // 2^32 / divisor
v_cvt_u32_f32 v117, v117                           // F = skGrid / skTiles
v_mul_lo_u32 v118, v117, s[sgprskTiles]            // refine 2^32 / divisor
v_sub_nc_u32 v118, 0, v118                         // refine 2^32 / divisor
v_mul_hi_u32 v118, v117, v118                      // refine 2^32 / divisor
v_add_nc_u32 v117, v117, v118                      // refine 2^32 / divisor
v_mul_hi_u32 v117, s[sgprskGrid], v117             // F = skGrid / skTiles
v_mul_lo_u32 v118, v117, s[sgprskTiles]            // F = skGrid / skTiles
v_sub_nc_u32 v118, s[sgprskGrid], v118             // F = skGrid / skTiles
v_cmp_ge_u32 vcc_lo, v118, s[sgprskTiles]          // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v117, 1, v117                         // quotient + 1
v_sub_nc_u32 v118, v118, s[sgprskTiles]            // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v118, s[sgprskTiles]          // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v117, 1, v117                         // quotient + 1
v_sub_nc_u32 v118, v118, s[sgprskTiles]            // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s53, v117                      // quotient
s_mov_b32 s11, s53                                 // F = skGrid / skTiles
v_cvt_f32_u32 v117, s11                            // s = cta % F
v_rcp_iflag_f32 v117, v117                         // s = cta % F
v_mov_b32 v118, 0x4f7ffffe                         // 2^32 - 512
v_mul_f32 v117, v118, v117                         // 2^32 / divisor
v_cvt_u32_f32 v117, v117                           // s = cta % F
v_mul_lo_u32 v118, v117, s11                       // refine 2^32 / divisor
v_sub_nc_u32 v118, 0, v118                         // refine 2^32 / divisor
v_mul_hi_u32 v118, v117, v118                      // refine 2^32 / divisor
v_add_nc_u32 v117, v117, v118                      // refine 2^32 / divisor
v_mul_hi_u32 v117, s7, v117                        // s = cta % F
v_mul_lo_u32 v118, v117, s11                       // s = cta % F
v_sub_nc_u32 v118, s7, v118                        // s = cta % F
v_cmp_ge_u32 vcc_lo, v118, s11                     // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v117, 1, v117                         // quotient + 1
v_sub_nc_u32 v118, v118, s11                       // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v118, s11                     // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v117, 1, v117                         // quotient + 1
v_sub_nc_u32 v118, v118, s11                       // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s53, v117                      // quotient
v_readfirstlane_b32 s53, v118                      // remainder
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

/* Edge/NonEdge store path check (M): Size % 64 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s68, 63, s[sgprSizeI]                    // s68 = s[sgprSizeI] % 64
s_add_u32 s69, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s69                // wg0 >= nwg0-1 ?
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_GSU1_Else       // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 160 > 0 -> Edge store; else -> NonEdge store */
s_mul_hi_u32 s71, s[sgprSizeJ], 53687092           // STATIC_DIV: divisor=160
s_mul_i32 s70, s[sgprSizeJ], 53687092              // tmp = dividend * magic
s_lshr_b64 s[70:71], s[70:71], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s69, s70                                 // quotient
s_mul_i32 s70, s69, 160                            // quotient*divisor
s_sub_u32 s68, s[sgprSizeJ], s70                   // rReg = dividend - quotient*divisor
s_add_u32 s69, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s69                // wg1 >= nwg1-1
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_GSU1_Then       // jump if edges required
label_GW_B0_FD0_VW1_GSU1_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=136 */
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
v_add_lshl_u32 v117, v109, v106, 1                 // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=106, coord0Vgpr=106 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (19, 0, 0, 0), (19, 1, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0), (22, 0, 0, 0), (22, 1, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0), (24, 0, 0, 0), (24, 1, 0, 0), (25, 0, 0, 0), (25, 1, 0, 0), (26, 0, 0, 0), (26, 1, 0, 0), (27, 0, 0, 0), (27, 1, 0, 0), (28, 0, 0, 0), (28, 1, 0, 0), (29, 0, 0, 0), (29, 1, 0, 0), (30, 0, 0, 0), (30, 1, 0, 0), (31, 0, 0, 0), (31, 1, 0, 0), (32, 0, 0, 0), (32, 1, 0, 0), (33, 0, 0, 0), (33, 1, 0, 0), (34, 0, 0, 0), (34, 1, 0, 0), (35, 0, 0, 0), (35, 1, 0, 0), (36, 0, 0, 0), (36, 1, 0, 0), (37, 0, 0, 0), (37, 1, 0, 0), (38, 0, 0, 0), (38, 1, 0, 0), (39, 0, 0, 0), (39, 1, 0, 0)] */
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
v_mul_f32 v[vgprValuC+185], s[sgprAlpha], v[vgprValuC+65] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+186], s[sgprAlpha], v[vgprValuC+73] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+187], s[sgprAlpha], v[vgprValuC+66] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+188], s[sgprAlpha], v[vgprValuC+74] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+189], s[sgprAlpha], v[vgprValuC+67] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+190], s[sgprAlpha], v[vgprValuC+75] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+191], s[sgprAlpha], v[vgprValuC+68] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+192], s[sgprAlpha], v[vgprValuC+76] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+193], s[sgprAlpha], v[vgprValuC+69] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+194], s[sgprAlpha], v[vgprValuC+77] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+195], s[sgprAlpha], v[vgprValuC+70] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+196], s[sgprAlpha], v[vgprValuC+78] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+197], s[sgprAlpha], v[vgprValuC+71] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+198], s[sgprAlpha], v[vgprValuC+79] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v114, 0xffff0000                         // mask for pack two bfloat16 element to 32bit
v_mov_b32 v115, 0x7fff0000                         // fp32 Nan
v_mov_b32 v116, 0x7fff                             // rounding bias for bfloat16
v_cmp_u_f32 s8, v[vgprValuC+119], v[vgprValuC+119] // check Nan
v_bfe_u32 v113, v[vgprValuC+119], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+119], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+119], v113, v115, s8
v_lshrrev_b32 v119, 16, v[vgprValuC+119]           // convert C to bf16
buffer_store_b16 v119, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+120], v[vgprValuC+120] // check Nan
v_bfe_u32 v113, v[vgprValuC+120], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+120], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+120], v113, v115, s8
v_lshrrev_b32 v120, 16, v[vgprValuC+120]           // convert C to bf16
buffer_store_b16 v120, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+121], v[vgprValuC+121] // check Nan
v_bfe_u32 v113, v[vgprValuC+121], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+121], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+121], v113, v115, s8
v_lshrrev_b32 v121, 16, v[vgprValuC+121]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v121, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+122], v[vgprValuC+122] // check Nan
v_bfe_u32 v113, v[vgprValuC+122], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+122], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+122], v113, v115, s8
v_lshrrev_b32 v122, 16, v[vgprValuC+122]           // convert C to bf16
buffer_store_b16 v122, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+123], v[vgprValuC+123] // check Nan
v_bfe_u32 v113, v[vgprValuC+123], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+123], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+123], v113, v115, s8
v_lshrrev_b32 v123, 16, v[vgprValuC+123]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v123, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+124], v[vgprValuC+124] // check Nan
v_bfe_u32 v113, v[vgprValuC+124], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+124], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+124], v113, v115, s8
v_lshrrev_b32 v124, 16, v[vgprValuC+124]           // convert C to bf16
buffer_store_b16 v124, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+125], v[vgprValuC+125] // check Nan
v_bfe_u32 v113, v[vgprValuC+125], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+125], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+125], v113, v115, s8
v_lshrrev_b32 v125, 16, v[vgprValuC+125]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v125, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+126], v[vgprValuC+126] // check Nan
v_bfe_u32 v113, v[vgprValuC+126], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+126], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+126], v113, v115, s8
v_lshrrev_b32 v126, 16, v[vgprValuC+126]           // convert C to bf16
buffer_store_b16 v126, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+127], v[vgprValuC+127] // check Nan
v_bfe_u32 v113, v[vgprValuC+127], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+127], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+127], v113, v115, s8
v_lshrrev_b32 v127, 16, v[vgprValuC+127]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v127, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+128], v[vgprValuC+128] // check Nan
v_bfe_u32 v113, v[vgprValuC+128], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+128], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+128], v113, v115, s8
v_lshrrev_b32 v128, 16, v[vgprValuC+128]           // convert C to bf16
buffer_store_b16 v128, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+129], v[vgprValuC+129] // check Nan
v_bfe_u32 v113, v[vgprValuC+129], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+129], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+129], v113, v115, s8
v_lshrrev_b32 v129, 16, v[vgprValuC+129]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v129, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+130], v[vgprValuC+130] // check Nan
v_bfe_u32 v113, v[vgprValuC+130], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+130], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+130], v113, v115, s8
v_lshrrev_b32 v130, 16, v[vgprValuC+130]           // convert C to bf16
buffer_store_b16 v130, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+131], v[vgprValuC+131] // check Nan
v_bfe_u32 v113, v[vgprValuC+131], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+131], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+131], v113, v115, s8
v_lshrrev_b32 v131, 16, v[vgprValuC+131]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v131, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+132], v[vgprValuC+132] // check Nan
v_bfe_u32 v113, v[vgprValuC+132], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+132], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+132], v113, v115, s8
v_lshrrev_b32 v132, 16, v[vgprValuC+132]           // convert C to bf16
buffer_store_b16 v132, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+133], v[vgprValuC+133] // check Nan
v_bfe_u32 v113, v[vgprValuC+133], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+133], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+133], v113, v115, s8
v_lshrrev_b32 v133, 16, v[vgprValuC+133]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v133, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+134], v[vgprValuC+134] // check Nan
v_bfe_u32 v113, v[vgprValuC+134], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+134], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+134], v113, v115, s8
v_lshrrev_b32 v134, 16, v[vgprValuC+134]           // convert C to bf16
buffer_store_b16 v134, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+135], v[vgprValuC+135] // check Nan
v_bfe_u32 v113, v[vgprValuC+135], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+135], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+135], v113, v115, s8
v_lshrrev_b32 v135, 16, v[vgprValuC+135]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 36                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b16 v135, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+136], v[vgprValuC+136] // check Nan
v_bfe_u32 v113, v[vgprValuC+136], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+136], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+136], v113, v115, s8
v_lshrrev_b32 v136, 16, v[vgprValuC+136]           // convert C to bf16
buffer_store_b16 v136, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+137], v[vgprValuC+137] // check Nan
v_bfe_u32 v113, v[vgprValuC+137], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+137], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+137], v113, v115, s8
v_lshrrev_b32 v137, 16, v[vgprValuC+137]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v137, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+138], v[vgprValuC+138] // check Nan
v_bfe_u32 v113, v[vgprValuC+138], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+138], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+138], v113, v115, s8
v_lshrrev_b32 v138, 16, v[vgprValuC+138]           // convert C to bf16
buffer_store_b16 v138, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+139], v[vgprValuC+139] // check Nan
v_bfe_u32 v113, v[vgprValuC+139], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+139], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+139], v113, v115, s8
v_lshrrev_b32 v139, 16, v[vgprValuC+139]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v139, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+140], v[vgprValuC+140] // check Nan
v_bfe_u32 v113, v[vgprValuC+140], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+140], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+140], v113, v115, s8
v_lshrrev_b32 v140, 16, v[vgprValuC+140]           // convert C to bf16
buffer_store_b16 v140, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+141], v[vgprValuC+141] // check Nan
v_bfe_u32 v113, v[vgprValuC+141], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+141], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+141], v113, v115, s8
v_lshrrev_b32 v141, 16, v[vgprValuC+141]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v141, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+142], v[vgprValuC+142] // check Nan
v_bfe_u32 v113, v[vgprValuC+142], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+142], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+142], v113, v115, s8
v_lshrrev_b32 v142, 16, v[vgprValuC+142]           // convert C to bf16
buffer_store_b16 v142, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+143], v[vgprValuC+143] // check Nan
v_bfe_u32 v113, v[vgprValuC+143], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+143], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+143], v113, v115, s8
v_lshrrev_b32 v143, 16, v[vgprValuC+143]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v143, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+144], v[vgprValuC+144] // check Nan
v_bfe_u32 v113, v[vgprValuC+144], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+144], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+144], v113, v115, s8
v_lshrrev_b32 v144, 16, v[vgprValuC+144]           // convert C to bf16
buffer_store_b16 v144, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+145], v[vgprValuC+145] // check Nan
v_bfe_u32 v113, v[vgprValuC+145], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+145], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+145], v113, v115, s8
v_lshrrev_b32 v145, 16, v[vgprValuC+145]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v145, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+146], v[vgprValuC+146] // check Nan
v_bfe_u32 v113, v[vgprValuC+146], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+146], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+146], v113, v115, s8
v_lshrrev_b32 v146, 16, v[vgprValuC+146]           // convert C to bf16
buffer_store_b16 v146, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+147], v[vgprValuC+147] // check Nan
v_bfe_u32 v113, v[vgprValuC+147], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+147], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+147], v113, v115, s8
v_lshrrev_b32 v147, 16, v[vgprValuC+147]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v147, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+148], v[vgprValuC+148] // check Nan
v_bfe_u32 v113, v[vgprValuC+148], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+148], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+148], v113, v115, s8
v_lshrrev_b32 v148, 16, v[vgprValuC+148]           // convert C to bf16
buffer_store_b16 v148, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+149], v[vgprValuC+149] // check Nan
v_bfe_u32 v113, v[vgprValuC+149], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+149], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+149], v113, v115, s8
v_lshrrev_b32 v149, 16, v[vgprValuC+149]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v149, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+150], v[vgprValuC+150] // check Nan
v_bfe_u32 v113, v[vgprValuC+150], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+150], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+150], v113, v115, s8
v_lshrrev_b32 v150, 16, v[vgprValuC+150]           // convert C to bf16
buffer_store_b16 v150, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+151], v[vgprValuC+151] // check Nan
v_bfe_u32 v113, v[vgprValuC+151], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+151], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+151], v113, v115, s8
v_lshrrev_b32 v151, 16, v[vgprValuC+151]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 36                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b16 v151, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+152], v[vgprValuC+152] // check Nan
v_bfe_u32 v113, v[vgprValuC+152], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+152], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+152], v113, v115, s8
v_lshrrev_b32 v152, 16, v[vgprValuC+152]           // convert C to bf16
buffer_store_b16 v152, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+153], v[vgprValuC+153] // check Nan
v_bfe_u32 v113, v[vgprValuC+153], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+153], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+153], v113, v115, s8
v_lshrrev_b32 v153, 16, v[vgprValuC+153]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v153, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+154], v[vgprValuC+154] // check Nan
v_bfe_u32 v113, v[vgprValuC+154], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+154], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+154], v113, v115, s8
v_lshrrev_b32 v154, 16, v[vgprValuC+154]           // convert C to bf16
buffer_store_b16 v154, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+155], v[vgprValuC+155] // check Nan
v_bfe_u32 v113, v[vgprValuC+155], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+155], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+155], v113, v115, s8
v_lshrrev_b32 v155, 16, v[vgprValuC+155]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v155, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+156], v[vgprValuC+156] // check Nan
v_bfe_u32 v113, v[vgprValuC+156], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+156], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+156], v113, v115, s8
v_lshrrev_b32 v156, 16, v[vgprValuC+156]           // convert C to bf16
buffer_store_b16 v156, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+157], v[vgprValuC+157] // check Nan
v_bfe_u32 v113, v[vgprValuC+157], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+157], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+157], v113, v115, s8
v_lshrrev_b32 v157, 16, v[vgprValuC+157]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v157, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+158], v[vgprValuC+158] // check Nan
v_bfe_u32 v113, v[vgprValuC+158], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+158], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+158], v113, v115, s8
v_lshrrev_b32 v158, 16, v[vgprValuC+158]           // convert C to bf16
buffer_store_b16 v158, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+159], v[vgprValuC+159] // check Nan
v_bfe_u32 v113, v[vgprValuC+159], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+159], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+159], v113, v115, s8
v_lshrrev_b32 v159, 16, v[vgprValuC+159]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v159, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+160], v[vgprValuC+160] // check Nan
v_bfe_u32 v113, v[vgprValuC+160], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+160], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+160], v113, v115, s8
v_lshrrev_b32 v160, 16, v[vgprValuC+160]           // convert C to bf16
buffer_store_b16 v160, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+161], v[vgprValuC+161] // check Nan
v_bfe_u32 v113, v[vgprValuC+161], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+161], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+161], v113, v115, s8
v_lshrrev_b32 v161, 16, v[vgprValuC+161]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v161, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+162], v[vgprValuC+162] // check Nan
v_bfe_u32 v113, v[vgprValuC+162], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+162], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+162], v113, v115, s8
v_lshrrev_b32 v162, 16, v[vgprValuC+162]           // convert C to bf16
buffer_store_b16 v162, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+163], v[vgprValuC+163] // check Nan
v_bfe_u32 v113, v[vgprValuC+163], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+163], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+163], v113, v115, s8
v_lshrrev_b32 v163, 16, v[vgprValuC+163]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v163, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+164], v[vgprValuC+164] // check Nan
v_bfe_u32 v113, v[vgprValuC+164], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+164], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+164], v113, v115, s8
v_lshrrev_b32 v164, 16, v[vgprValuC+164]           // convert C to bf16
buffer_store_b16 v164, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+165], v[vgprValuC+165] // check Nan
v_bfe_u32 v113, v[vgprValuC+165], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+165], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+165], v113, v115, s8
v_lshrrev_b32 v165, 16, v[vgprValuC+165]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v165, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+166], v[vgprValuC+166] // check Nan
v_bfe_u32 v113, v[vgprValuC+166], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+166], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+166], v113, v115, s8
v_lshrrev_b32 v166, 16, v[vgprValuC+166]           // convert C to bf16
buffer_store_b16 v166, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+167], v[vgprValuC+167] // check Nan
v_bfe_u32 v113, v[vgprValuC+167], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+167], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+167], v113, v115, s8
v_lshrrev_b32 v167, 16, v[vgprValuC+167]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 36                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b16 v167, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+168], v[vgprValuC+168] // check Nan
v_bfe_u32 v113, v[vgprValuC+168], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+168], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+168], v113, v115, s8
v_lshrrev_b32 v168, 16, v[vgprValuC+168]           // convert C to bf16
buffer_store_b16 v168, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+169], v[vgprValuC+169] // check Nan
v_bfe_u32 v113, v[vgprValuC+169], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+169], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+169], v113, v115, s8
v_lshrrev_b32 v169, 16, v[vgprValuC+169]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v169, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+170], v[vgprValuC+170] // check Nan
v_bfe_u32 v113, v[vgprValuC+170], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+170], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+170], v113, v115, s8
v_lshrrev_b32 v170, 16, v[vgprValuC+170]           // convert C to bf16
buffer_store_b16 v170, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+171], v[vgprValuC+171] // check Nan
v_bfe_u32 v113, v[vgprValuC+171], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+171], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+171], v113, v115, s8
v_lshrrev_b32 v171, 16, v[vgprValuC+171]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v171, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+172], v[vgprValuC+172] // check Nan
v_bfe_u32 v113, v[vgprValuC+172], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+172], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+172], v113, v115, s8
v_lshrrev_b32 v172, 16, v[vgprValuC+172]           // convert C to bf16
buffer_store_b16 v172, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+173], v[vgprValuC+173] // check Nan
v_bfe_u32 v113, v[vgprValuC+173], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+173], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+173], v113, v115, s8
v_lshrrev_b32 v173, 16, v[vgprValuC+173]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v173, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+174], v[vgprValuC+174] // check Nan
v_bfe_u32 v113, v[vgprValuC+174], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+174], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+174], v113, v115, s8
v_lshrrev_b32 v174, 16, v[vgprValuC+174]           // convert C to bf16
buffer_store_b16 v174, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+175], v[vgprValuC+175] // check Nan
v_bfe_u32 v113, v[vgprValuC+175], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+175], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+175], v113, v115, s8
v_lshrrev_b32 v175, 16, v[vgprValuC+175]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v175, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+176], v[vgprValuC+176] // check Nan
v_bfe_u32 v113, v[vgprValuC+176], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+176], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+176], v113, v115, s8
v_lshrrev_b32 v176, 16, v[vgprValuC+176]           // convert C to bf16
buffer_store_b16 v176, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+177], v[vgprValuC+177] // check Nan
v_bfe_u32 v113, v[vgprValuC+177], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+177], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+177], v113, v115, s8
v_lshrrev_b32 v177, 16, v[vgprValuC+177]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v177, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+178], v[vgprValuC+178] // check Nan
v_bfe_u32 v113, v[vgprValuC+178], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+178], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+178], v113, v115, s8
v_lshrrev_b32 v178, 16, v[vgprValuC+178]           // convert C to bf16
buffer_store_b16 v178, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+179], v[vgprValuC+179] // check Nan
v_bfe_u32 v113, v[vgprValuC+179], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+179], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+179], v113, v115, s8
v_lshrrev_b32 v179, 16, v[vgprValuC+179]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v179, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+180], v[vgprValuC+180] // check Nan
v_bfe_u32 v113, v[vgprValuC+180], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+180], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+180], v113, v115, s8
v_lshrrev_b32 v180, 16, v[vgprValuC+180]           // convert C to bf16
buffer_store_b16 v180, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+181], v[vgprValuC+181] // check Nan
v_bfe_u32 v113, v[vgprValuC+181], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+181], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+181], v113, v115, s8
v_lshrrev_b32 v181, 16, v[vgprValuC+181]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v181, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+182], v[vgprValuC+182] // check Nan
v_bfe_u32 v113, v[vgprValuC+182], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+182], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+182], v113, v115, s8
v_lshrrev_b32 v182, 16, v[vgprValuC+182]           // convert C to bf16
buffer_store_b16 v182, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+183], v[vgprValuC+183] // check Nan
v_bfe_u32 v113, v[vgprValuC+183], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+183], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+183], v113, v115, s8
v_lshrrev_b32 v183, 16, v[vgprValuC+183]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 36                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b16 v183, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+184], v[vgprValuC+184] // check Nan
v_bfe_u32 v113, v[vgprValuC+184], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+184], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+184], v113, v115, s8
v_lshrrev_b32 v184, 16, v[vgprValuC+184]           // convert C to bf16
buffer_store_b16 v184, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+185], v[vgprValuC+185] // check Nan
v_bfe_u32 v113, v[vgprValuC+185], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+185], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+185], v113, v115, s8
v_lshrrev_b32 v185, 16, v[vgprValuC+185]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v185, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+186], v[vgprValuC+186] // check Nan
v_bfe_u32 v113, v[vgprValuC+186], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+186], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+186], v113, v115, s8
v_lshrrev_b32 v186, 16, v[vgprValuC+186]           // convert C to bf16
buffer_store_b16 v186, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+187], v[vgprValuC+187] // check Nan
v_bfe_u32 v113, v[vgprValuC+187], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+187], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+187], v113, v115, s8
v_lshrrev_b32 v187, 16, v[vgprValuC+187]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v187, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+188], v[vgprValuC+188] // check Nan
v_bfe_u32 v113, v[vgprValuC+188], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+188], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+188], v113, v115, s8
v_lshrrev_b32 v188, 16, v[vgprValuC+188]           // convert C to bf16
buffer_store_b16 v188, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+189], v[vgprValuC+189] // check Nan
v_bfe_u32 v113, v[vgprValuC+189], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+189], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+189], v113, v115, s8
v_lshrrev_b32 v189, 16, v[vgprValuC+189]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v189, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+190], v[vgprValuC+190] // check Nan
v_bfe_u32 v113, v[vgprValuC+190], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+190], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+190], v113, v115, s8
v_lshrrev_b32 v190, 16, v[vgprValuC+190]           // convert C to bf16
buffer_store_b16 v190, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+191], v[vgprValuC+191] // check Nan
v_bfe_u32 v113, v[vgprValuC+191], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+191], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+191], v113, v115, s8
v_lshrrev_b32 v191, 16, v[vgprValuC+191]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v191, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+192], v[vgprValuC+192] // check Nan
v_bfe_u32 v113, v[vgprValuC+192], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+192], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+192], v113, v115, s8
v_lshrrev_b32 v192, 16, v[vgprValuC+192]           // convert C to bf16
buffer_store_b16 v192, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+193], v[vgprValuC+193] // check Nan
v_bfe_u32 v113, v[vgprValuC+193], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+193], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+193], v113, v115, s8
v_lshrrev_b32 v193, 16, v[vgprValuC+193]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v193, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+194], v[vgprValuC+194] // check Nan
v_bfe_u32 v113, v[vgprValuC+194], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+194], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+194], v113, v115, s8
v_lshrrev_b32 v194, 16, v[vgprValuC+194]           // convert C to bf16
buffer_store_b16 v194, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+195], v[vgprValuC+195] // check Nan
v_bfe_u32 v113, v[vgprValuC+195], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+195], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+195], v113, v115, s8
v_lshrrev_b32 v195, 16, v[vgprValuC+195]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v195, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+196], v[vgprValuC+196] // check Nan
v_bfe_u32 v113, v[vgprValuC+196], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+196], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+196], v113, v115, s8
v_lshrrev_b32 v196, 16, v[vgprValuC+196]           // convert C to bf16
buffer_store_b16 v196, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+197], v[vgprValuC+197] // check Nan
v_bfe_u32 v113, v[vgprValuC+197], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+197], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+197], v113, v115, s8
v_lshrrev_b32 v197, 16, v[vgprValuC+197]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v197, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+198], v[vgprValuC+198] // check Nan
v_bfe_u32 v113, v[vgprValuC+198], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+198], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+198], v113, v115, s8
v_lshrrev_b32 v198, 16, v[vgprValuC+198]           // convert C to bf16
buffer_store_b16 v198, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW1_GSU1_NonEdgeEnd:
label_GW_B0_FD0_VW1_GSU1_Else:
label_GW_B0_FD0_VW1_GSU1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=68 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1); (24,0,0,0:vw1); (24,1,0,0:vw1); (25,0,0,0:vw1); (25,1,0,0:vw1); (26,0,0,0:vw1); (26,1,0,0:vw1); (27,0,0,0:vw1); (27,1,0,0:vw1); (28,0,0,0:vw1); (28,1,0,0:vw1); (29,0,0,0:vw1); (29,1,0,0:vw1); (30,0,0,0:vw1); (30,1,0,0:vw1); (31,0,0,0:vw1); (31,1,0,0:vw1); (32,0,0,0:vw1); (32,1,0,0:vw1); (33,0,0,0:vw1); (33,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v112, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v185, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v185, v112, v185, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v186, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v186, v112, v186, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v187, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v187, v112, v187, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v188, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v188, v112, v188, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v189, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v189, v112, v189, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v190, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v190, v112, v190, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v191, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v191, v112, v191, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v192, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v192, v112, v192, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v193, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v193, v112, v193, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v194, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v194, v112, v194, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v195, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v195, v112, v195, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v196, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v196, v112, v196, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v197, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v197, v112, v197, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v198, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v198, v112, v198, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v199, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v199, v112, v199, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v200, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v200, v112, v200, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 18                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v201, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v201, v112, v201, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v202, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v202, v112, v202, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v203, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v203, v112, v203, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v204, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v204, v112, v204, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v205, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v205, v112, v205, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v206, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v206, v112, v206, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v207, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v207, v112, v207, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v208, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v208, v112, v208, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v209, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v209, v112, v209, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v210, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v210, v112, v210, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v211, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v211, v112, v211, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v212, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v212, v112, v212, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v213, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v213, v112, v213, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v214, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v214, v112, v214, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v215, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v215, v112, v215, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v216, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v216, v112, v216, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 18                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v217, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v217, v112, v217, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v218, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v218, v112, v218, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v219, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v219, v112, v219, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v221, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v221, v112, v221, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v222, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v222, v112, v222, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v223, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v223, v112, v223, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v224, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v224, v112, v224, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v225, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v225, v112, v225, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v226, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v226, v112, v226, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v227, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v227, v112, v227, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v228, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v228, v112, v228, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v229, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v229, v112, v229, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v230, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v230, v112, v230, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v231, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v231, v112, v231, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v232, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v232, v112, v232, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v233, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v233, v112, v233, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 18                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v234, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v234, v112, v234, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v235, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v235, v112, v235, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v236, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v236, v112, v236, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v237, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v237, v112, v237, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v238, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v238, v112, v238, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v239, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v239, v112, v239, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v240, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v240, v112, v240, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v241, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v241, v112, v241, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v242, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v242, v112, v242, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v243, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v243, v112, v243, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v244, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v244, v112, v244, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v245, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v245, v112, v245, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v246, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v246, v112, v246, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v247, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v247, v112, v247, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v248, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v248, v112, v248, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v249, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v249, v112, v249, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(32,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 18                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v250, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v250, v112, v250, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(32,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v251, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v251, v112, v251, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(33,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v252, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v252, v112, v252, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(33,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v253, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v253, v112, v253, s70                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (19, 0, 0, 0), (19, 1, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0), (22, 0, 0, 0), (22, 1, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0), (24, 0, 0, 0), (24, 1, 0, 0), (25, 0, 0, 0), (25, 1, 0, 0), (26, 0, 0, 0), (26, 1, 0, 0), (27, 0, 0, 0), (27, 1, 0, 0), (28, 0, 0, 0), (28, 1, 0, 0), (29, 0, 0, 0), (29, 1, 0, 0), (30, 0, 0, 0), (30, 1, 0, 0), (31, 0, 0, 0), (31, 1, 0, 0), (32, 0, 0, 0), (32, 1, 0, 0), (33, 0, 0, 0), (33, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+128], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+129], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+130], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+131], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+132], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+133], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+134], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+135], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+152], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+154], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+158], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+160], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+162], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+163], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+164], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+48] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+166], s[sgprAlpha], v[vgprValuC+56] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+167], s[sgprAlpha], v[vgprValuC+49] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+168], s[sgprAlpha], v[vgprValuC+57] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+169], s[sgprAlpha], v[vgprValuC+50] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+170], s[sgprAlpha], v[vgprValuC+58] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+171], s[sgprAlpha], v[vgprValuC+51] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+172], s[sgprAlpha], v[vgprValuC+59] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+173], s[sgprAlpha], v[vgprValuC+52] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+174], s[sgprAlpha], v[vgprValuC+60] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+175], s[sgprAlpha], v[vgprValuC+53] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+176], s[sgprAlpha], v[vgprValuC+61] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+177], s[sgprAlpha], v[vgprValuC+54] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+178], s[sgprAlpha], v[vgprValuC+62] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+179], s[sgprAlpha], v[vgprValuC+55] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+180], s[sgprAlpha], v[vgprValuC+63] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+181], s[sgprAlpha], v[vgprValuC+64] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+182], s[sgprAlpha], v[vgprValuC+72] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+183], s[sgprAlpha], v[vgprValuC+65] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+184], s[sgprAlpha], v[vgprValuC+73] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v114, 0xffff0000                         // mask for pack two bfloat16 element to 32bit
v_mov_b32 v115, 0x7fff0000                         // fp32 Nan
v_mov_b32 v116, 0x7fff                             // rounding bias for bfloat16
v_cmp_u_f32 s68, v[vgprValuC+117], v[vgprValuC+117] // check Nan
v_bfe_u32 v113, v[vgprValuC+117], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+117], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+117], v113, v115, s68
v_lshrrev_b32 v117, 16, v[vgprValuC+117]           // convert C to bf16
buffer_store_b16 v117, v185, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+118], v[vgprValuC+118] // check Nan
v_bfe_u32 v113, v[vgprValuC+118], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+118], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+118], v113, v115, s68
v_lshrrev_b32 v118, 16, v[vgprValuC+118]           // convert C to bf16
buffer_store_b16 v118, v186, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+119], v[vgprValuC+119] // check Nan
v_bfe_u32 v113, v[vgprValuC+119], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+119], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+119], v113, v115, s68
v_lshrrev_b32 v119, 16, v[vgprValuC+119]           // convert C to bf16
buffer_store_b16 v119, v187, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+120], v[vgprValuC+120] // check Nan
v_bfe_u32 v113, v[vgprValuC+120], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+120], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+120], v113, v115, s68
v_lshrrev_b32 v120, 16, v[vgprValuC+120]           // convert C to bf16
buffer_store_b16 v120, v188, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+121], v[vgprValuC+121] // check Nan
v_bfe_u32 v113, v[vgprValuC+121], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+121], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+121], v113, v115, s68
v_lshrrev_b32 v121, 16, v[vgprValuC+121]           // convert C to bf16
buffer_store_b16 v121, v189, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+122], v[vgprValuC+122] // check Nan
v_bfe_u32 v113, v[vgprValuC+122], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+122], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+122], v113, v115, s68
v_lshrrev_b32 v122, 16, v[vgprValuC+122]           // convert C to bf16
buffer_store_b16 v122, v190, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+123], v[vgprValuC+123] // check Nan
v_bfe_u32 v113, v[vgprValuC+123], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+123], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+123], v113, v115, s68
v_lshrrev_b32 v123, 16, v[vgprValuC+123]           // convert C to bf16
buffer_store_b16 v123, v191, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+124], v[vgprValuC+124] // check Nan
v_bfe_u32 v113, v[vgprValuC+124], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+124], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+124], v113, v115, s68
v_lshrrev_b32 v124, 16, v[vgprValuC+124]           // convert C to bf16
buffer_store_b16 v124, v192, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+125], v[vgprValuC+125] // check Nan
v_bfe_u32 v113, v[vgprValuC+125], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+125], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+125], v113, v115, s68
v_lshrrev_b32 v125, 16, v[vgprValuC+125]           // convert C to bf16
buffer_store_b16 v125, v193, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+126], v[vgprValuC+126] // check Nan
v_bfe_u32 v113, v[vgprValuC+126], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+126], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+126], v113, v115, s68
v_lshrrev_b32 v126, 16, v[vgprValuC+126]           // convert C to bf16
buffer_store_b16 v126, v194, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+127], v[vgprValuC+127] // check Nan
v_bfe_u32 v113, v[vgprValuC+127], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+127], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+127], v113, v115, s68
v_lshrrev_b32 v127, 16, v[vgprValuC+127]           // convert C to bf16
buffer_store_b16 v127, v195, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+128], v[vgprValuC+128] // check Nan
v_bfe_u32 v113, v[vgprValuC+128], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+128], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+128], v113, v115, s68
v_lshrrev_b32 v128, 16, v[vgprValuC+128]           // convert C to bf16
buffer_store_b16 v128, v196, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+129], v[vgprValuC+129] // check Nan
v_bfe_u32 v113, v[vgprValuC+129], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+129], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+129], v113, v115, s68
v_lshrrev_b32 v129, 16, v[vgprValuC+129]           // convert C to bf16
buffer_store_b16 v129, v197, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+130], v[vgprValuC+130] // check Nan
v_bfe_u32 v113, v[vgprValuC+130], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+130], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+130], v113, v115, s68
v_lshrrev_b32 v130, 16, v[vgprValuC+130]           // convert C to bf16
buffer_store_b16 v130, v198, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+131], v[vgprValuC+131] // check Nan
v_bfe_u32 v113, v[vgprValuC+131], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+131], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+131], v113, v115, s68
v_lshrrev_b32 v131, 16, v[vgprValuC+131]           // convert C to bf16
buffer_store_b16 v131, v199, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+132], v[vgprValuC+132] // check Nan
v_bfe_u32 v113, v[vgprValuC+132], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+132], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+132], v113, v115, s68
v_lshrrev_b32 v132, 16, v[vgprValuC+132]           // convert C to bf16
buffer_store_b16 v132, v200, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+133], v[vgprValuC+133] // check Nan
v_bfe_u32 v113, v[vgprValuC+133], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+133], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+133], v113, v115, s68
v_lshrrev_b32 v133, 16, v[vgprValuC+133]           // convert C to bf16
buffer_store_b16 v133, v201, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+134], v[vgprValuC+134] // check Nan
v_bfe_u32 v113, v[vgprValuC+134], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+134], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+134], v113, v115, s68
v_lshrrev_b32 v134, 16, v[vgprValuC+134]           // convert C to bf16
buffer_store_b16 v134, v202, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+135], v[vgprValuC+135] // check Nan
v_bfe_u32 v113, v[vgprValuC+135], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+135], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+135], v113, v115, s68
v_lshrrev_b32 v135, 16, v[vgprValuC+135]           // convert C to bf16
buffer_store_b16 v135, v203, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+136], v[vgprValuC+136] // check Nan
v_bfe_u32 v113, v[vgprValuC+136], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+136], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+136], v113, v115, s68
v_lshrrev_b32 v136, 16, v[vgprValuC+136]           // convert C to bf16
buffer_store_b16 v136, v204, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+137], v[vgprValuC+137] // check Nan
v_bfe_u32 v113, v[vgprValuC+137], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+137], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+137], v113, v115, s68
v_lshrrev_b32 v137, 16, v[vgprValuC+137]           // convert C to bf16
buffer_store_b16 v137, v205, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+138], v[vgprValuC+138] // check Nan
v_bfe_u32 v113, v[vgprValuC+138], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+138], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+138], v113, v115, s68
v_lshrrev_b32 v138, 16, v[vgprValuC+138]           // convert C to bf16
buffer_store_b16 v138, v206, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+139], v[vgprValuC+139] // check Nan
v_bfe_u32 v113, v[vgprValuC+139], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+139], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+139], v113, v115, s68
v_lshrrev_b32 v139, 16, v[vgprValuC+139]           // convert C to bf16
buffer_store_b16 v139, v207, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+140], v[vgprValuC+140] // check Nan
v_bfe_u32 v113, v[vgprValuC+140], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+140], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+140], v113, v115, s68
v_lshrrev_b32 v140, 16, v[vgprValuC+140]           // convert C to bf16
buffer_store_b16 v140, v208, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+141], v[vgprValuC+141] // check Nan
v_bfe_u32 v113, v[vgprValuC+141], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+141], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+141], v113, v115, s68
v_lshrrev_b32 v141, 16, v[vgprValuC+141]           // convert C to bf16
buffer_store_b16 v141, v209, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+142], v[vgprValuC+142] // check Nan
v_bfe_u32 v113, v[vgprValuC+142], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+142], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+142], v113, v115, s68
v_lshrrev_b32 v142, 16, v[vgprValuC+142]           // convert C to bf16
buffer_store_b16 v142, v210, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+143], v[vgprValuC+143] // check Nan
v_bfe_u32 v113, v[vgprValuC+143], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+143], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+143], v113, v115, s68
v_lshrrev_b32 v143, 16, v[vgprValuC+143]           // convert C to bf16
buffer_store_b16 v143, v211, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+144], v[vgprValuC+144] // check Nan
v_bfe_u32 v113, v[vgprValuC+144], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+144], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+144], v113, v115, s68
v_lshrrev_b32 v144, 16, v[vgprValuC+144]           // convert C to bf16
buffer_store_b16 v144, v212, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+145], v[vgprValuC+145] // check Nan
v_bfe_u32 v113, v[vgprValuC+145], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+145], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+145], v113, v115, s68
v_lshrrev_b32 v145, 16, v[vgprValuC+145]           // convert C to bf16
buffer_store_b16 v145, v213, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+146], v[vgprValuC+146] // check Nan
v_bfe_u32 v113, v[vgprValuC+146], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+146], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+146], v113, v115, s68
v_lshrrev_b32 v146, 16, v[vgprValuC+146]           // convert C to bf16
buffer_store_b16 v146, v214, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+147], v[vgprValuC+147] // check Nan
v_bfe_u32 v113, v[vgprValuC+147], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+147], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+147], v113, v115, s68
v_lshrrev_b32 v147, 16, v[vgprValuC+147]           // convert C to bf16
buffer_store_b16 v147, v215, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+148], v[vgprValuC+148] // check Nan
v_bfe_u32 v113, v[vgprValuC+148], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+148], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+148], v113, v115, s68
v_lshrrev_b32 v148, 16, v[vgprValuC+148]           // convert C to bf16
buffer_store_b16 v148, v216, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+149], v[vgprValuC+149] // check Nan
v_bfe_u32 v113, v[vgprValuC+149], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+149], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+149], v113, v115, s68
v_lshrrev_b32 v149, 16, v[vgprValuC+149]           // convert C to bf16
buffer_store_b16 v149, v217, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+150], v[vgprValuC+150] // check Nan
v_bfe_u32 v113, v[vgprValuC+150], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+150], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+150], v113, v115, s68
v_lshrrev_b32 v150, 16, v[vgprValuC+150]           // convert C to bf16
buffer_store_b16 v150, v218, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+151], v[vgprValuC+151] // check Nan
v_bfe_u32 v113, v[vgprValuC+151], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+151], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+151], v113, v115, s68
v_lshrrev_b32 v151, 16, v[vgprValuC+151]           // convert C to bf16
buffer_store_b16 v151, v219, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+152], v[vgprValuC+152] // check Nan
v_bfe_u32 v113, v[vgprValuC+152], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+152], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+152], v113, v115, s68
v_lshrrev_b32 v152, 16, v[vgprValuC+152]           // convert C to bf16
buffer_store_b16 v152, v221, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+153], v[vgprValuC+153] // check Nan
v_bfe_u32 v113, v[vgprValuC+153], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+153], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+153], v113, v115, s68
v_lshrrev_b32 v153, 16, v[vgprValuC+153]           // convert C to bf16
buffer_store_b16 v153, v222, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+154], v[vgprValuC+154] // check Nan
v_bfe_u32 v113, v[vgprValuC+154], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+154], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+154], v113, v115, s68
v_lshrrev_b32 v154, 16, v[vgprValuC+154]           // convert C to bf16
buffer_store_b16 v154, v223, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+155], v[vgprValuC+155] // check Nan
v_bfe_u32 v113, v[vgprValuC+155], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+155], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+155], v113, v115, s68
v_lshrrev_b32 v155, 16, v[vgprValuC+155]           // convert C to bf16
buffer_store_b16 v155, v224, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+156], v[vgprValuC+156] // check Nan
v_bfe_u32 v113, v[vgprValuC+156], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+156], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+156], v113, v115, s68
v_lshrrev_b32 v156, 16, v[vgprValuC+156]           // convert C to bf16
buffer_store_b16 v156, v225, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+157], v[vgprValuC+157] // check Nan
v_bfe_u32 v113, v[vgprValuC+157], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+157], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+157], v113, v115, s68
v_lshrrev_b32 v157, 16, v[vgprValuC+157]           // convert C to bf16
buffer_store_b16 v157, v226, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+158], v[vgprValuC+158] // check Nan
v_bfe_u32 v113, v[vgprValuC+158], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+158], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+158], v113, v115, s68
v_lshrrev_b32 v158, 16, v[vgprValuC+158]           // convert C to bf16
buffer_store_b16 v158, v227, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+159], v[vgprValuC+159] // check Nan
v_bfe_u32 v113, v[vgprValuC+159], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+159], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+159], v113, v115, s68
v_lshrrev_b32 v159, 16, v[vgprValuC+159]           // convert C to bf16
buffer_store_b16 v159, v228, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+160], v[vgprValuC+160] // check Nan
v_bfe_u32 v113, v[vgprValuC+160], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+160], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+160], v113, v115, s68
v_lshrrev_b32 v160, 16, v[vgprValuC+160]           // convert C to bf16
buffer_store_b16 v160, v229, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+161], v[vgprValuC+161] // check Nan
v_bfe_u32 v113, v[vgprValuC+161], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+161], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+161], v113, v115, s68
v_lshrrev_b32 v161, 16, v[vgprValuC+161]           // convert C to bf16
buffer_store_b16 v161, v230, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+162], v[vgprValuC+162] // check Nan
v_bfe_u32 v113, v[vgprValuC+162], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+162], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+162], v113, v115, s68
v_lshrrev_b32 v162, 16, v[vgprValuC+162]           // convert C to bf16
buffer_store_b16 v162, v231, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+163], v[vgprValuC+163] // check Nan
v_bfe_u32 v113, v[vgprValuC+163], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+163], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+163], v113, v115, s68
v_lshrrev_b32 v163, 16, v[vgprValuC+163]           // convert C to bf16
buffer_store_b16 v163, v232, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+164], v[vgprValuC+164] // check Nan
v_bfe_u32 v113, v[vgprValuC+164], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+164], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+164], v113, v115, s68
v_lshrrev_b32 v164, 16, v[vgprValuC+164]           // convert C to bf16
buffer_store_b16 v164, v233, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+165], v[vgprValuC+165] // check Nan
v_bfe_u32 v113, v[vgprValuC+165], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+165], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+165], v113, v115, s68
v_lshrrev_b32 v165, 16, v[vgprValuC+165]           // convert C to bf16
buffer_store_b16 v165, v234, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+166], v[vgprValuC+166] // check Nan
v_bfe_u32 v113, v[vgprValuC+166], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+166], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+166], v113, v115, s68
v_lshrrev_b32 v166, 16, v[vgprValuC+166]           // convert C to bf16
buffer_store_b16 v166, v235, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+167], v[vgprValuC+167] // check Nan
v_bfe_u32 v113, v[vgprValuC+167], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+167], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+167], v113, v115, s68
v_lshrrev_b32 v167, 16, v[vgprValuC+167]           // convert C to bf16
buffer_store_b16 v167, v236, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+168], v[vgprValuC+168] // check Nan
v_bfe_u32 v113, v[vgprValuC+168], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+168], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+168], v113, v115, s68
v_lshrrev_b32 v168, 16, v[vgprValuC+168]           // convert C to bf16
buffer_store_b16 v168, v237, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+169], v[vgprValuC+169] // check Nan
v_bfe_u32 v113, v[vgprValuC+169], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+169], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+169], v113, v115, s68
v_lshrrev_b32 v169, 16, v[vgprValuC+169]           // convert C to bf16
buffer_store_b16 v169, v238, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+170], v[vgprValuC+170] // check Nan
v_bfe_u32 v113, v[vgprValuC+170], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+170], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+170], v113, v115, s68
v_lshrrev_b32 v170, 16, v[vgprValuC+170]           // convert C to bf16
buffer_store_b16 v170, v239, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+171], v[vgprValuC+171] // check Nan
v_bfe_u32 v113, v[vgprValuC+171], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+171], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+171], v113, v115, s68
v_lshrrev_b32 v171, 16, v[vgprValuC+171]           // convert C to bf16
buffer_store_b16 v171, v240, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+172], v[vgprValuC+172] // check Nan
v_bfe_u32 v113, v[vgprValuC+172], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+172], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+172], v113, v115, s68
v_lshrrev_b32 v172, 16, v[vgprValuC+172]           // convert C to bf16
buffer_store_b16 v172, v241, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+173], v[vgprValuC+173] // check Nan
v_bfe_u32 v113, v[vgprValuC+173], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+173], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+173], v113, v115, s68
v_lshrrev_b32 v173, 16, v[vgprValuC+173]           // convert C to bf16
buffer_store_b16 v173, v242, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+174], v[vgprValuC+174] // check Nan
v_bfe_u32 v113, v[vgprValuC+174], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+174], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+174], v113, v115, s68
v_lshrrev_b32 v174, 16, v[vgprValuC+174]           // convert C to bf16
buffer_store_b16 v174, v243, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+175], v[vgprValuC+175] // check Nan
v_bfe_u32 v113, v[vgprValuC+175], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+175], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+175], v113, v115, s68
v_lshrrev_b32 v175, 16, v[vgprValuC+175]           // convert C to bf16
buffer_store_b16 v175, v244, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+176], v[vgprValuC+176] // check Nan
v_bfe_u32 v113, v[vgprValuC+176], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+176], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+176], v113, v115, s68
v_lshrrev_b32 v176, 16, v[vgprValuC+176]           // convert C to bf16
buffer_store_b16 v176, v245, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+177], v[vgprValuC+177] // check Nan
v_bfe_u32 v113, v[vgprValuC+177], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+177], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+177], v113, v115, s68
v_lshrrev_b32 v177, 16, v[vgprValuC+177]           // convert C to bf16
buffer_store_b16 v177, v246, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+178], v[vgprValuC+178] // check Nan
v_bfe_u32 v113, v[vgprValuC+178], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+178], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+178], v113, v115, s68
v_lshrrev_b32 v178, 16, v[vgprValuC+178]           // convert C to bf16
buffer_store_b16 v178, v247, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+179], v[vgprValuC+179] // check Nan
v_bfe_u32 v113, v[vgprValuC+179], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+179], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+179], v113, v115, s68
v_lshrrev_b32 v179, 16, v[vgprValuC+179]           // convert C to bf16
buffer_store_b16 v179, v248, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+180], v[vgprValuC+180] // check Nan
v_bfe_u32 v113, v[vgprValuC+180], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+180], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+180], v113, v115, s68
v_lshrrev_b32 v180, 16, v[vgprValuC+180]           // convert C to bf16
buffer_store_b16 v180, v249, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+181], v[vgprValuC+181] // check Nan
v_bfe_u32 v113, v[vgprValuC+181], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+181], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+181], v113, v115, s68
v_lshrrev_b32 v181, 16, v[vgprValuC+181]           // convert C to bf16
buffer_store_b16 v181, v250, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+182], v[vgprValuC+182] // check Nan
v_bfe_u32 v113, v[vgprValuC+182], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+182], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+182], v113, v115, s68
v_lshrrev_b32 v182, 16, v[vgprValuC+182]           // convert C to bf16
buffer_store_b16 v182, v251, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+183], v[vgprValuC+183] // check Nan
v_bfe_u32 v113, v[vgprValuC+183], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+183], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+183], v113, v115, s68
v_lshrrev_b32 v183, 16, v[vgprValuC+183]           // convert C to bf16
buffer_store_b16 v183, v252, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+184], v[vgprValuC+184] // check Nan
v_bfe_u32 v113, v[vgprValuC+184], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+184], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+184], v113, v115, s68
v_lshrrev_b32 v184, 16, v[vgprValuC+184]           // convert C to bf16
buffer_store_b16 v184, v253, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (34,0,0,0:vw1); (34,1,0,0:vw1); (35,0,0,0:vw1); (35,1,0,0:vw1); (36,0,0,0:vw1); (36,1,0,0:vw1); (37,0,0,0:vw1); (37,1,0,0:vw1); (38,0,0,0:vw1); (38,1,0,0:vw1); (39,0,0,0:vw1); (39,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v112, BufferOOB
/* (d1,vc1,d0,vc0)=(34,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v129, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v129, v112, v129, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(34,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v130, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v112, v130, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(35,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v131, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v131, v112, v131, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(35,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v132, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v112, v132, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(36,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v133, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v133, v112, v133, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(36,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v134, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v112, v134, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(37,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v135, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v135, v112, v135, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(37,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v136, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v112, v136, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(38,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v137, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v137, v112, v137, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(38,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v138, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v112, v138, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(39,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v139, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v139, v112, v139, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(39,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v140, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v112, v140, s70                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(34, 0, 0, 0), (34, 1, 0, 0), (35, 0, 0, 0), (35, 1, 0, 0), (36, 0, 0, 0), (36, 1, 0, 0), (37, 0, 0, 0), (37, 1, 0, 0), (38, 0, 0, 0), (38, 1, 0, 0), (39, 0, 0, 0), (39, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+66] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+74] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+67] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+75] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+68] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+76] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+69] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+77] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+70] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+78] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+71] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+128], s[sgprAlpha], v[vgprValuC+79] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v114, 0xffff0000                         // mask for pack two bfloat16 element to 32bit
v_mov_b32 v115, 0x7fff0000                         // fp32 Nan
v_mov_b32 v116, 0x7fff                             // rounding bias for bfloat16
v_cmp_u_f32 s68, v[vgprValuC+117], v[vgprValuC+117] // check Nan
v_bfe_u32 v113, v[vgprValuC+117], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+117], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+117], v113, v115, s68
v_lshrrev_b32 v117, 16, v[vgprValuC+117]           // convert C to bf16
buffer_store_b16 v117, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+118], v[vgprValuC+118] // check Nan
v_bfe_u32 v113, v[vgprValuC+118], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+118], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+118], v113, v115, s68
v_lshrrev_b32 v118, 16, v[vgprValuC+118]           // convert C to bf16
buffer_store_b16 v118, v130, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+119], v[vgprValuC+119] // check Nan
v_bfe_u32 v113, v[vgprValuC+119], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+119], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+119], v113, v115, s68
v_lshrrev_b32 v119, 16, v[vgprValuC+119]           // convert C to bf16
buffer_store_b16 v119, v131, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+120], v[vgprValuC+120] // check Nan
v_bfe_u32 v113, v[vgprValuC+120], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+120], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+120], v113, v115, s68
v_lshrrev_b32 v120, 16, v[vgprValuC+120]           // convert C to bf16
buffer_store_b16 v120, v132, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+121], v[vgprValuC+121] // check Nan
v_bfe_u32 v113, v[vgprValuC+121], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+121], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+121], v113, v115, s68
v_lshrrev_b32 v121, 16, v[vgprValuC+121]           // convert C to bf16
buffer_store_b16 v121, v133, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+122], v[vgprValuC+122] // check Nan
v_bfe_u32 v113, v[vgprValuC+122], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+122], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+122], v113, v115, s68
v_lshrrev_b32 v122, 16, v[vgprValuC+122]           // convert C to bf16
buffer_store_b16 v122, v134, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+123], v[vgprValuC+123] // check Nan
v_bfe_u32 v113, v[vgprValuC+123], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+123], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+123], v113, v115, s68
v_lshrrev_b32 v123, 16, v[vgprValuC+123]           // convert C to bf16
buffer_store_b16 v123, v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+124], v[vgprValuC+124] // check Nan
v_bfe_u32 v113, v[vgprValuC+124], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+124], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+124], v113, v115, s68
v_lshrrev_b32 v124, 16, v[vgprValuC+124]           // convert C to bf16
buffer_store_b16 v124, v136, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+125], v[vgprValuC+125] // check Nan
v_bfe_u32 v113, v[vgprValuC+125], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+125], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+125], v113, v115, s68
v_lshrrev_b32 v125, 16, v[vgprValuC+125]           // convert C to bf16
buffer_store_b16 v125, v137, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+126], v[vgprValuC+126] // check Nan
v_bfe_u32 v113, v[vgprValuC+126], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+126], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+126], v113, v115, s68
v_lshrrev_b32 v126, 16, v[vgprValuC+126]           // convert C to bf16
buffer_store_b16 v126, v138, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+127], v[vgprValuC+127] // check Nan
v_bfe_u32 v113, v[vgprValuC+127], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+127], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+127], v113, v115, s68
v_lshrrev_b32 v127, 16, v[vgprValuC+127]           // convert C to bf16
buffer_store_b16 v127, v139, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+128], v[vgprValuC+128] // check Nan
v_bfe_u32 v113, v[vgprValuC+128], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+128], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+128], v113, v115, s68
v_lshrrev_b32 v128, 16, v[vgprValuC+128]           // convert C to bf16
buffer_store_b16 v128, v140, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_GSU1:
label_GW_B1_FD0_GSU1:

/* Edge/NonEdge store path check (M): Size % 64 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s68, 63, s[sgprSizeI]                    // s68 = s[sgprSizeI] % 64
s_add_u32 s69, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s69                // wg0 >= nwg0-1 ?
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW1_GSU1_Else       // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 160 > 0 -> Edge store; else -> NonEdge store */
s_mul_hi_u32 s71, s[sgprSizeJ], 53687092           // STATIC_DIV: divisor=160
s_mul_i32 s70, s[sgprSizeJ], 53687092              // tmp = dividend * magic
s_lshr_b64 s[70:71], s[70:71], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s69, s70                                 // quotient
s_mul_i32 s70, s69, 160                            // quotient*divisor
s_sub_u32 s68, s[sgprSizeJ], s70                   // rReg = dividend - quotient*divisor
s_add_u32 s69, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s69                // wg1 >= nwg1-1
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
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
v_add_lshl_u32 v118, v108, v106, 1                 // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=106, coord0Vgpr=106 (multiple bpe)
buffer_load_d16_b16 v185, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
buffer_load_d16_b16 v186, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v187, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
buffer_load_d16_b16 v188, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v189, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
buffer_load_d16_b16 v190, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v191, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
buffer_load_d16_b16 v192, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v193, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
buffer_load_d16_b16 v194, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v195, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
buffer_load_d16_b16 v196, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v197, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
buffer_load_d16_b16 v198, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v199, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
buffer_load_d16_b16 v200, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 36                 // scale StrideC *= numRows(18) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_load_d16_b16 v201, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
buffer_load_d16_b16 v202, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v203, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(9,0,1,0) */
buffer_load_d16_b16 v204, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v205, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(10,0,1,0) */
buffer_load_d16_b16 v206, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v207, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(11,0,1,0) */
buffer_load_d16_b16 v208, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v209, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
buffer_load_d16_b16 v210, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v211, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(13,0,1,0) */
buffer_load_d16_b16 v212, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v213, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(14,0,1,0) */
buffer_load_d16_b16 v214, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v215, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(15,0,1,0) */
buffer_load_d16_b16 v216, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 36                 // scale StrideC *= numRows(18) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_load_d16_b16 v217, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
buffer_load_d16_b16 v218, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v219, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(17,0,1,0) */
buffer_load_d16_b16 v221, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v222, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(18,0,1,0) */
buffer_load_d16_b16 v223, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v224, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(19,0,1,0) */
buffer_load_d16_b16 v225, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v226, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
buffer_load_d16_b16 v227, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v228, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(21,0,1,0) */
buffer_load_d16_b16 v229, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v230, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(22,0,1,0) */
buffer_load_d16_b16 v231, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v232, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(23,0,1,0) */
buffer_load_d16_b16 v233, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 36                 // scale StrideC *= numRows(18) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_load_d16_b16 v234, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(24,0,1,0) */
buffer_load_d16_b16 v235, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v236, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(25,0,1,0) */
buffer_load_d16_b16 v237, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v238, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(26,0,1,0) */
buffer_load_d16_b16 v239, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v240, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(27,0,1,0) */
buffer_load_d16_b16 v241, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v242, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(28,0,1,0) */
buffer_load_d16_b16 v243, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v244, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(29,0,1,0) */
buffer_load_d16_b16 v245, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v246, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(30,0,1,0) */
buffer_load_d16_b16 v247, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v248, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(31,0,1,0) */
buffer_load_d16_b16 v249, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(32,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 36                 // scale StrideC *= numRows(18) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_load_d16_b16 v250, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(32,0,1,0) */
buffer_load_d16_b16 v251, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
v_add_lshl_u32 v117, v109, v106, 1                 // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=106, coord0Vgpr=106 (multiple bpe)

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
v_mov_b32 v114, 0xffff0000                         // mask for pack two bfloat16 element to 32bit
v_mov_b32 v115, 0x7fff0000                         // fp32 Nan
v_mov_b32 v116, 0x7fff                             // rounding bias for bfloat16

s_waitcnt vmcnt(63)                                // vlcnt(65) = 66 - 1 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v185                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+119], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+119], v[vgprValuC+119] // check Nan
v_bfe_u32 v113, v[vgprValuC+119], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+119], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+119], v113, v115, s8
v_lshrrev_b32 v119, 16, v[vgprValuC+119]           // convert C to bf16
buffer_store_b16 v119, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(63)                                // vlcnt(64) = 66 - 2 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v186                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+120], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+120], v[vgprValuC+120] // check Nan
v_bfe_u32 v113, v[vgprValuC+120], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+120], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+120], v113, v115, s8
v_lshrrev_b32 v120, 16, v[vgprValuC+120]           // convert C to bf16
buffer_store_b16 v120, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(63)                                // vlcnt(63) = 66 - 3 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v187                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+121], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+121], v[vgprValuC+121] // check Nan
v_bfe_u32 v113, v[vgprValuC+121], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+121], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+121], v113, v115, s8
v_lshrrev_b32 v121, 16, v[vgprValuC+121]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v121, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(62)                                // vlcnt(62) = 66 - 4 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v188                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+122], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+122], v[vgprValuC+122] // check Nan
v_bfe_u32 v113, v[vgprValuC+122], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+122], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+122], v113, v115, s8
v_lshrrev_b32 v122, 16, v[vgprValuC+122]           // convert C to bf16
buffer_store_b16 v122, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(61)                                // vlcnt(61) = 66 - 5 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v189                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+123], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+123], v[vgprValuC+123] // check Nan
v_bfe_u32 v113, v[vgprValuC+123], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+123], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+123], v113, v115, s8
v_lshrrev_b32 v123, 16, v[vgprValuC+123]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v123, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(60)                                // vlcnt(60) = 66 - 6 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v190                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+124], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+124], v[vgprValuC+124] // check Nan
v_bfe_u32 v113, v[vgprValuC+124], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+124], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+124], v113, v115, s8
v_lshrrev_b32 v124, 16, v[vgprValuC+124]           // convert C to bf16
buffer_store_b16 v124, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(59)                                // vlcnt(59) = 66 - 7 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v191                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+125], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+125], v[vgprValuC+125] // check Nan
v_bfe_u32 v113, v[vgprValuC+125], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+125], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+125], v113, v115, s8
v_lshrrev_b32 v125, 16, v[vgprValuC+125]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v125, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(58)                                // vlcnt(58) = 66 - 8 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v192                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+126], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+126], v[vgprValuC+126] // check Nan
v_bfe_u32 v113, v[vgprValuC+126], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+126], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+126], v113, v115, s8
v_lshrrev_b32 v126, 16, v[vgprValuC+126]           // convert C to bf16
buffer_store_b16 v126, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(57)                                // vlcnt(57) = 66 - 9 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v193                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+127], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+127], v[vgprValuC+127] // check Nan
v_bfe_u32 v113, v[vgprValuC+127], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+127], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+127], v113, v115, s8
v_lshrrev_b32 v127, 16, v[vgprValuC+127]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v127, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(56)                                // vlcnt(56) = 66 - 10 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v194                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+128], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+128], v[vgprValuC+128] // check Nan
v_bfe_u32 v113, v[vgprValuC+128], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+128], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+128], v113, v115, s8
v_lshrrev_b32 v128, 16, v[vgprValuC+128]           // convert C to bf16
buffer_store_b16 v128, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(55)                                // vlcnt(55) = 66 - 11 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v195                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+129], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+129], v[vgprValuC+129] // check Nan
v_bfe_u32 v113, v[vgprValuC+129], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+129], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+129], v113, v115, s8
v_lshrrev_b32 v129, 16, v[vgprValuC+129]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v129, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(54)                                // vlcnt(54) = 66 - 12 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v196                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+130], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+130], v[vgprValuC+130] // check Nan
v_bfe_u32 v113, v[vgprValuC+130], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+130], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+130], v113, v115, s8
v_lshrrev_b32 v130, 16, v[vgprValuC+130]           // convert C to bf16
buffer_store_b16 v130, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(53)                                // vlcnt(53) = 66 - 13 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v197                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+131], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+131], v[vgprValuC+131] // check Nan
v_bfe_u32 v113, v[vgprValuC+131], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+131], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+131], v113, v115, s8
v_lshrrev_b32 v131, 16, v[vgprValuC+131]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v131, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(52)                                // vlcnt(52) = 66 - 14 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v198                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+132], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+132], v[vgprValuC+132] // check Nan
v_bfe_u32 v113, v[vgprValuC+132], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+132], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+132], v113, v115, s8
v_lshrrev_b32 v132, 16, v[vgprValuC+132]           // convert C to bf16
buffer_store_b16 v132, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(51)                                // vlcnt(51) = 66 - 15 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v199                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+133], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+133], v[vgprValuC+133] // check Nan
v_bfe_u32 v113, v[vgprValuC+133], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+133], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+133], v113, v115, s8
v_lshrrev_b32 v133, 16, v[vgprValuC+133]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v133, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(50)                                // vlcnt(50) = 66 - 16 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v200                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+134], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+134], v[vgprValuC+134] // check Nan
v_bfe_u32 v113, v[vgprValuC+134], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+134], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+134], v113, v115, s8
v_lshrrev_b32 v134, 16, v[vgprValuC+134]           // convert C to bf16
buffer_store_b16 v134, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(49)                                // vlcnt(49) = 66 - 17 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v201                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+135], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+135], v[vgprValuC+135] // check Nan
v_bfe_u32 v113, v[vgprValuC+135], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+135], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+135], v113, v115, s8
v_lshrrev_b32 v135, 16, v[vgprValuC+135]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 36                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b16 v135, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(48)                                // vlcnt(48) = 66 - 18 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v202                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+136], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+136], v[vgprValuC+136] // check Nan
v_bfe_u32 v113, v[vgprValuC+136], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+136], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+136], v113, v115, s8
v_lshrrev_b32 v136, 16, v[vgprValuC+136]           // convert C to bf16
buffer_store_b16 v136, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(47)                                // vlcnt(47) = 66 - 19 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v203                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+137], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+137], v[vgprValuC+137] // check Nan
v_bfe_u32 v113, v[vgprValuC+137], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+137], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+137], v113, v115, s8
v_lshrrev_b32 v137, 16, v[vgprValuC+137]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v137, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(46)                                // vlcnt(46) = 66 - 20 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v204                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+138], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+138], v[vgprValuC+138] // check Nan
v_bfe_u32 v113, v[vgprValuC+138], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+138], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+138], v113, v115, s8
v_lshrrev_b32 v138, 16, v[vgprValuC+138]           // convert C to bf16
buffer_store_b16 v138, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(45)                                // vlcnt(45) = 66 - 21 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v205                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+139], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+139], v[vgprValuC+139] // check Nan
v_bfe_u32 v113, v[vgprValuC+139], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+139], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+139], v113, v115, s8
v_lshrrev_b32 v139, 16, v[vgprValuC+139]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v139, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(44)                                // vlcnt(44) = 66 - 22 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v206                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+140], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+140], v[vgprValuC+140] // check Nan
v_bfe_u32 v113, v[vgprValuC+140], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+140], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+140], v113, v115, s8
v_lshrrev_b32 v140, 16, v[vgprValuC+140]           // convert C to bf16
buffer_store_b16 v140, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(43)                                // vlcnt(43) = 66 - 23 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v207                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+141], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+141], v[vgprValuC+141] // check Nan
v_bfe_u32 v113, v[vgprValuC+141], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+141], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+141], v113, v115, s8
v_lshrrev_b32 v141, 16, v[vgprValuC+141]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v141, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(42)                                // vlcnt(42) = 66 - 24 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v208                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+142], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+142], v[vgprValuC+142] // check Nan
v_bfe_u32 v113, v[vgprValuC+142], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+142], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+142], v113, v115, s8
v_lshrrev_b32 v142, 16, v[vgprValuC+142]           // convert C to bf16
buffer_store_b16 v142, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(41)                                // vlcnt(41) = 66 - 25 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v209                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+143], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+143], v[vgprValuC+143] // check Nan
v_bfe_u32 v113, v[vgprValuC+143], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+143], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+143], v113, v115, s8
v_lshrrev_b32 v143, 16, v[vgprValuC+143]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v143, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(40)                                // vlcnt(40) = 66 - 26 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v210                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+144], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+144], v[vgprValuC+144] // check Nan
v_bfe_u32 v113, v[vgprValuC+144], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+144], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+144], v113, v115, s8
v_lshrrev_b32 v144, 16, v[vgprValuC+144]           // convert C to bf16
buffer_store_b16 v144, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(39)                                // vlcnt(39) = 66 - 27 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v211                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+145], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+145], v[vgprValuC+145] // check Nan
v_bfe_u32 v113, v[vgprValuC+145], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+145], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+145], v113, v115, s8
v_lshrrev_b32 v145, 16, v[vgprValuC+145]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v145, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(38)                                // vlcnt(38) = 66 - 28 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v212                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+146], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+146], v[vgprValuC+146] // check Nan
v_bfe_u32 v113, v[vgprValuC+146], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+146], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+146], v113, v115, s8
v_lshrrev_b32 v146, 16, v[vgprValuC+146]           // convert C to bf16
buffer_store_b16 v146, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(37)                                // vlcnt(37) = 66 - 29 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v213                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+147], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+147], v[vgprValuC+147] // check Nan
v_bfe_u32 v113, v[vgprValuC+147], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+147], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+147], v113, v115, s8
v_lshrrev_b32 v147, 16, v[vgprValuC+147]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v147, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(36)                                // vlcnt(36) = 66 - 30 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v214                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+148], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+148], v[vgprValuC+148] // check Nan
v_bfe_u32 v113, v[vgprValuC+148], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+148], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+148], v113, v115, s8
v_lshrrev_b32 v148, 16, v[vgprValuC+148]           // convert C to bf16
buffer_store_b16 v148, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(35)                                // vlcnt(35) = 66 - 31 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v215                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+149], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+149], v[vgprValuC+149] // check Nan
v_bfe_u32 v113, v[vgprValuC+149], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+149], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+149], v113, v115, s8
v_lshrrev_b32 v149, 16, v[vgprValuC+149]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v149, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(34)                                // vlcnt(34) = 66 - 32 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v216                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+150], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+150], v[vgprValuC+150] // check Nan
v_bfe_u32 v113, v[vgprValuC+150], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+150], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+150], v113, v115, s8
v_lshrrev_b32 v150, 16, v[vgprValuC+150]           // convert C to bf16
buffer_store_b16 v150, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(33)                                // vlcnt(33) = 66 - 33 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v217                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+151], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+151], v[vgprValuC+151] // check Nan
v_bfe_u32 v113, v[vgprValuC+151], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+151], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+151], v113, v115, s8
v_lshrrev_b32 v151, 16, v[vgprValuC+151]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 36                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b16 v151, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(32)                                // vlcnt(32) = 66 - 34 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v218                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+152], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+152], v[vgprValuC+152] // check Nan
v_bfe_u32 v113, v[vgprValuC+152], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+152], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+152], v113, v115, s8
v_lshrrev_b32 v152, 16, v[vgprValuC+152]           // convert C to bf16
buffer_store_b16 v152, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(31)                                // vlcnt(31) = 66 - 35 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v219                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+153], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+153], v[vgprValuC+153] // check Nan
v_bfe_u32 v113, v[vgprValuC+153], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+153], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+153], v113, v115, s8
v_lshrrev_b32 v153, 16, v[vgprValuC+153]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v153, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(30)                                // vlcnt(30) = 66 - 36 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v221                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+154], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+154], v[vgprValuC+154] // check Nan
v_bfe_u32 v113, v[vgprValuC+154], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+154], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+154], v113, v115, s8
v_lshrrev_b32 v154, 16, v[vgprValuC+154]           // convert C to bf16
buffer_store_b16 v154, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(29)                                // vlcnt(29) = 66 - 37 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v222                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+155], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+155], v[vgprValuC+155] // check Nan
v_bfe_u32 v113, v[vgprValuC+155], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+155], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+155], v113, v115, s8
v_lshrrev_b32 v155, 16, v[vgprValuC+155]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v155, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(28)                                // vlcnt(28) = 66 - 38 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v223                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+156], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+156], v[vgprValuC+156] // check Nan
v_bfe_u32 v113, v[vgprValuC+156], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+156], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+156], v113, v115, s8
v_lshrrev_b32 v156, 16, v[vgprValuC+156]           // convert C to bf16
buffer_store_b16 v156, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(27)                                // vlcnt(27) = 66 - 39 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v224                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+157], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+157], v[vgprValuC+157] // check Nan
v_bfe_u32 v113, v[vgprValuC+157], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+157], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+157], v113, v115, s8
v_lshrrev_b32 v157, 16, v[vgprValuC+157]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v157, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(26)                                // vlcnt(26) = 66 - 40 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v225                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+158], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+158], v[vgprValuC+158] // check Nan
v_bfe_u32 v113, v[vgprValuC+158], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+158], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+158], v113, v115, s8
v_lshrrev_b32 v158, 16, v[vgprValuC+158]           // convert C to bf16
buffer_store_b16 v158, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(25)                                // vlcnt(25) = 66 - 41 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v226                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+159], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+159], v[vgprValuC+159] // check Nan
v_bfe_u32 v113, v[vgprValuC+159], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+159], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+159], v113, v115, s8
v_lshrrev_b32 v159, 16, v[vgprValuC+159]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v159, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(24)                                // vlcnt(24) = 66 - 42 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v227                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+160], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+160], v[vgprValuC+160] // check Nan
v_bfe_u32 v113, v[vgprValuC+160], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+160], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+160], v113, v115, s8
v_lshrrev_b32 v160, 16, v[vgprValuC+160]           // convert C to bf16
buffer_store_b16 v160, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(23)                                // vlcnt(23) = 66 - 43 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v228                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+161], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+161], v[vgprValuC+161] // check Nan
v_bfe_u32 v113, v[vgprValuC+161], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+161], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+161], v113, v115, s8
v_lshrrev_b32 v161, 16, v[vgprValuC+161]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v161, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(22)                                // vlcnt(22) = 66 - 44 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v229                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+162], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+162], v[vgprValuC+162] // check Nan
v_bfe_u32 v113, v[vgprValuC+162], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+162], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+162], v113, v115, s8
v_lshrrev_b32 v162, 16, v[vgprValuC+162]           // convert C to bf16
buffer_store_b16 v162, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(21)                                // vlcnt(21) = 66 - 45 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v230                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+163], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+163], v[vgprValuC+163] // check Nan
v_bfe_u32 v113, v[vgprValuC+163], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+163], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+163], v113, v115, s8
v_lshrrev_b32 v163, 16, v[vgprValuC+163]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v163, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(20)                                // vlcnt(20) = 66 - 46 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v231                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+164], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+164], v[vgprValuC+164] // check Nan
v_bfe_u32 v113, v[vgprValuC+164], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+164], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+164], v113, v115, s8
v_lshrrev_b32 v164, 16, v[vgprValuC+164]           // convert C to bf16
buffer_store_b16 v164, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(19)                                // vlcnt(19) = 66 - 47 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v232                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+165], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+165], v[vgprValuC+165] // check Nan
v_bfe_u32 v113, v[vgprValuC+165], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+165], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+165], v113, v115, s8
v_lshrrev_b32 v165, 16, v[vgprValuC+165]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v165, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(18)                                // vlcnt(18) = 66 - 48 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v233                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+166], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+166], v[vgprValuC+166] // check Nan
v_bfe_u32 v113, v[vgprValuC+166], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+166], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+166], v113, v115, s8
v_lshrrev_b32 v166, 16, v[vgprValuC+166]           // convert C to bf16
buffer_store_b16 v166, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(17)                                // vlcnt(17) = 66 - 49 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v234                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+167], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+167], v[vgprValuC+167] // check Nan
v_bfe_u32 v113, v[vgprValuC+167], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+167], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+167], v113, v115, s8
v_lshrrev_b32 v167, 16, v[vgprValuC+167]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 36                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b16 v167, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(16)                                // vlcnt(16) = 66 - 50 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v235                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+168], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+168], v[vgprValuC+168] // check Nan
v_bfe_u32 v113, v[vgprValuC+168], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+168], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+168], v113, v115, s8
v_lshrrev_b32 v168, 16, v[vgprValuC+168]           // convert C to bf16
buffer_store_b16 v168, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(15)                                // vlcnt(15) = 66 - 51 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v236                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+169], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+169], v[vgprValuC+169] // check Nan
v_bfe_u32 v113, v[vgprValuC+169], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+169], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+169], v113, v115, s8
v_lshrrev_b32 v169, 16, v[vgprValuC+169]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v169, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(14)                                // vlcnt(14) = 66 - 52 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v237                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+170], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+170], v[vgprValuC+170] // check Nan
v_bfe_u32 v113, v[vgprValuC+170], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+170], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+170], v113, v115, s8
v_lshrrev_b32 v170, 16, v[vgprValuC+170]           // convert C to bf16
buffer_store_b16 v170, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(13)                                // vlcnt(13) = 66 - 53 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v238                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+171], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+171], v[vgprValuC+171] // check Nan
v_bfe_u32 v113, v[vgprValuC+171], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+171], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+171], v113, v115, s8
v_lshrrev_b32 v171, 16, v[vgprValuC+171]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v171, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(12)                                // vlcnt(12) = 66 - 54 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v239                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+172], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+172], v[vgprValuC+172] // check Nan
v_bfe_u32 v113, v[vgprValuC+172], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+172], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+172], v113, v115, s8
v_lshrrev_b32 v172, 16, v[vgprValuC+172]           // convert C to bf16
buffer_store_b16 v172, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(11)                                // vlcnt(11) = 66 - 55 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v240                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+173], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+173], v[vgprValuC+173] // check Nan
v_bfe_u32 v113, v[vgprValuC+173], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+173], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+173], v113, v115, s8
v_lshrrev_b32 v173, 16, v[vgprValuC+173]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v173, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(10)                                // vlcnt(10) = 66 - 56 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v241                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+174], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+174], v[vgprValuC+174] // check Nan
v_bfe_u32 v113, v[vgprValuC+174], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+174], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+174], v113, v115, s8
v_lshrrev_b32 v174, 16, v[vgprValuC+174]           // convert C to bf16
buffer_store_b16 v174, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(9)                                 // vlcnt(9) = 66 - 57 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v242                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+175], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+175], v[vgprValuC+175] // check Nan
v_bfe_u32 v113, v[vgprValuC+175], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+175], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+175], v113, v115, s8
v_lshrrev_b32 v175, 16, v[vgprValuC+175]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v175, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(8)                                 // vlcnt(8) = 66 - 58 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v243                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+176], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+176], v[vgprValuC+176] // check Nan
v_bfe_u32 v113, v[vgprValuC+176], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+176], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+176], v113, v115, s8
v_lshrrev_b32 v176, 16, v[vgprValuC+176]           // convert C to bf16
buffer_store_b16 v176, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(7)                                 // vlcnt(7) = 66 - 59 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v244                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+177], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+177], v[vgprValuC+177] // check Nan
v_bfe_u32 v113, v[vgprValuC+177], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+177], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+177], v113, v115, s8
v_lshrrev_b32 v177, 16, v[vgprValuC+177]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v177, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(6)                                 // vlcnt(6) = 66 - 60 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v245                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+178], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+178], v[vgprValuC+178] // check Nan
v_bfe_u32 v113, v[vgprValuC+178], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+178], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+178], v113, v115, s8
v_lshrrev_b32 v178, 16, v[vgprValuC+178]           // convert C to bf16
buffer_store_b16 v178, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(5)                                 // vlcnt(5) = 66 - 61 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v246                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+179], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+179], v[vgprValuC+179] // check Nan
v_bfe_u32 v113, v[vgprValuC+179], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+179], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+179], v113, v115, s8
v_lshrrev_b32 v179, 16, v[vgprValuC+179]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v179, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(4)                                 // vlcnt(4) = 66 - 62 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v247                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+180], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+180], v[vgprValuC+180] // check Nan
v_bfe_u32 v113, v[vgprValuC+180], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+180], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+180], v113, v115, s8
v_lshrrev_b32 v180, 16, v[vgprValuC+180]           // convert C to bf16
buffer_store_b16 v180, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(3)                                 // vlcnt(3) = 66 - 63 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v248                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+181], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+181], v[vgprValuC+181] // check Nan
v_bfe_u32 v113, v[vgprValuC+181], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+181], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+181], v113, v115, s8
v_lshrrev_b32 v181, 16, v[vgprValuC+181]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v181, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(2)                                 // vlcnt(2) = 66 - 64 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v249                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+182], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+182], v[vgprValuC+182] // check Nan
v_bfe_u32 v113, v[vgprValuC+182], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+182], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+182], v113, v115, s8
v_lshrrev_b32 v182, 16, v[vgprValuC+182]           // convert C to bf16
buffer_store_b16 v182, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(1)                                 // vlcnt(1) = 66 - 65 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v250                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+183], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+183], v[vgprValuC+183] // check Nan
v_bfe_u32 v113, v[vgprValuC+183], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+183], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+183], v113, v115, s8
v_lshrrev_b32 v183, 16, v[vgprValuC+183]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 36                 // scale StrideD *= numRows(18) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(18): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(18): gra SRD += inc(upper)
buffer_store_b16 v183, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(0)                                 // vlcnt(0) = 66 - 66 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v251                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+184], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+184], v[vgprValuC+184] // check Nan
v_bfe_u32 v113, v[vgprValuC+184], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+184], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+184], v113, v115, s8
v_lshrrev_b32 v184, 16, v[vgprValuC+184]           // convert C to bf16
buffer_store_b16 v184, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
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
buffer_load_d16_b16 v133, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(33,0,1,0) */
buffer_load_d16_b16 v134, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(34,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v135, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(34,0,1,0) */
buffer_load_d16_b16 v136, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(35,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v137, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(35,0,1,0) */
buffer_load_d16_b16 v138, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(36,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v139, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(36,0,1,0) */
buffer_load_d16_b16 v140, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(37,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v141, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(37,0,1,0) */
buffer_load_d16_b16 v142, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(38,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v143, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(38,0,1,0) */
buffer_load_d16_b16 v144, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(39,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v145, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(39,0,1,0) */
buffer_load_d16_b16 v146, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C

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
v_mov_b32 v114, 0xffff0000                         // mask for pack two bfloat16 element to 32bit
v_mov_b32 v115, 0x7fff0000                         // fp32 Nan
v_mov_b32 v116, 0x7fff                             // rounding bias for bfloat16

s_waitcnt vmcnt(13)                                // vlcnt(13) = 14 - 1 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v133                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+119], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+119], v[vgprValuC+119] // check Nan
v_bfe_u32 v113, v[vgprValuC+119], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+119], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+119], v113, v115, s8
v_lshrrev_b32 v119, 16, v[vgprValuC+119]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v119, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(12)                                // vlcnt(12) = 14 - 2 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v134                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+120], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+120], v[vgprValuC+120] // check Nan
v_bfe_u32 v113, v[vgprValuC+120], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+120], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+120], v113, v115, s8
v_lshrrev_b32 v120, 16, v[vgprValuC+120]           // convert C to bf16
buffer_store_b16 v120, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(11)                                // vlcnt(11) = 14 - 3 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v135                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+121], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+121], v[vgprValuC+121] // check Nan
v_bfe_u32 v113, v[vgprValuC+121], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+121], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+121], v113, v115, s8
v_lshrrev_b32 v121, 16, v[vgprValuC+121]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v121, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(10)                                // vlcnt(10) = 14 - 4 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v136                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+122], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+122], v[vgprValuC+122] // check Nan
v_bfe_u32 v113, v[vgprValuC+122], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+122], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+122], v113, v115, s8
v_lshrrev_b32 v122, 16, v[vgprValuC+122]           // convert C to bf16
buffer_store_b16 v122, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(9)                                 // vlcnt(9) = 14 - 5 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v137                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+123], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+123], v[vgprValuC+123] // check Nan
v_bfe_u32 v113, v[vgprValuC+123], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+123], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+123], v113, v115, s8
v_lshrrev_b32 v123, 16, v[vgprValuC+123]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v123, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(8)                                 // vlcnt(8) = 14 - 6 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v138                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+124], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+124], v[vgprValuC+124] // check Nan
v_bfe_u32 v113, v[vgprValuC+124], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+124], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+124], v113, v115, s8
v_lshrrev_b32 v124, 16, v[vgprValuC+124]           // convert C to bf16
buffer_store_b16 v124, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(7)                                 // vlcnt(7) = 14 - 7 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v139                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+125], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+125], v[vgprValuC+125] // check Nan
v_bfe_u32 v113, v[vgprValuC+125], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+125], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+125], v113, v115, s8
v_lshrrev_b32 v125, 16, v[vgprValuC+125]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v125, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(6)                                 // vlcnt(6) = 14 - 8 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v140                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+126], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+126], v[vgprValuC+126] // check Nan
v_bfe_u32 v113, v[vgprValuC+126], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+126], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+126], v113, v115, s8
v_lshrrev_b32 v126, 16, v[vgprValuC+126]           // convert C to bf16
buffer_store_b16 v126, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(5)                                 // vlcnt(5) = 14 - 9 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v141                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+127], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+127], v[vgprValuC+127] // check Nan
v_bfe_u32 v113, v[vgprValuC+127], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+127], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+127], v113, v115, s8
v_lshrrev_b32 v127, 16, v[vgprValuC+127]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v127, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(4)                                 // vlcnt(4) = 14 - 10 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v142                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+128], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+128], v[vgprValuC+128] // check Nan
v_bfe_u32 v113, v[vgprValuC+128], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+128], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+128], v113, v115, s8
v_lshrrev_b32 v128, 16, v[vgprValuC+128]           // convert C to bf16
buffer_store_b16 v128, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(3)                                 // vlcnt(3) = 14 - 11 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v143                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+129], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+129], v[vgprValuC+129] // check Nan
v_bfe_u32 v113, v[vgprValuC+129], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+129], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+129], v113, v115, s8
v_lshrrev_b32 v129, 16, v[vgprValuC+129]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v129, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(2)                                 // vlcnt(2) = 14 - 12 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v144                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+130], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+130], v[vgprValuC+130] // check Nan
v_bfe_u32 v113, v[vgprValuC+130], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+130], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+130], v113, v115, s8
v_lshrrev_b32 v130, 16, v[vgprValuC+130]           // convert C to bf16
buffer_store_b16 v130, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(1)                                 // vlcnt(1) = 14 - 13 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v145                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+131], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+131], v[vgprValuC+131] // check Nan
v_bfe_u32 v113, v[vgprValuC+131], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+131], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+131], v113, v115, s8
v_lshrrev_b32 v131, 16, v[vgprValuC+131]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v131, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(0)                                 // vlcnt(0) = 14 - 14 (beta) (interleaved)
v_lshlrev_b32 v110, 16, v146                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+132], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+132], v[vgprValuC+132] // check Nan
v_bfe_u32 v113, v[vgprValuC+132], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+132], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+132], v113, v115, s8
v_lshrrev_b32 v132, 16, v[vgprValuC+132]           // convert C to bf16
buffer_store_b16 v132, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
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
v_mov_b32 v112, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v162, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v162, v112, v162, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v161, v162, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v162, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v162, v112, v162, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v164, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v164, v112, v164, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v163, v164, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v164, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v164, v112, v164, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v166, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v166, v112, v166, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v165, v166, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v166, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v166, v112, v166, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v168, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v168, v112, v168, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v167, v168, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v168, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v168, v112, v168, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v170, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v170, v112, v170, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v169, v170, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v170, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v170, v112, v170, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v172, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v172, v112, v172, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v171, v172, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v172, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v172, v112, v172, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v174, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v174, v112, v174, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v173, v174, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v174, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v174, v112, v174, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v176, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v176, v112, v176, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v175, v176, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v176, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v176, v112, v176, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v178, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v178, v112, v178, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v177, v178, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v178, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v178, v112, v178, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v180, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v180, v112, v180, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v179, v180, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v180, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v180, v112, v180, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v182, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v182, v112, v182, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v181, v182, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v182, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v182, v112, v182, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v184, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v184, v112, v184, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v183, v184, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v184, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v184, v112, v184, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v186, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v186, v112, v186, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v185, v186, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v186, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v186, v112, v186, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v188, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v188, v112, v188, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v187, v188, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v188, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v188, v112, v188, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v190, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v190, v112, v190, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v189, v190, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v190, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v190, v112, v190, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v192, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v192, v112, v192, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v191, v192, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v192, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v192, v112, v192, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 18                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v194, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v194, v112, v194, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v193, v194, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v194, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v194, v112, v194, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v196, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v196, v112, v196, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v195, v196, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v196, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v196, v112, v196, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v198, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v198, v112, v198, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v197, v198, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v198, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v198, v112, v198, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v200, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v200, v112, v200, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v199, v200, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v200, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v200, v112, v200, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v202, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v202, v112, v202, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v201, v202, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v202, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v202, v112, v202, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v204, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v204, v112, v204, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v203, v204, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v204, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v204, v112, v204, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v206, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v206, v112, v206, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v205, v206, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v206, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v206, v112, v206, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v208, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v208, v112, v208, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v207, v208, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v208, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v208, v112, v208, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v210, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v210, v112, v210, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v209, v210, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v210, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v210, v112, v210, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v212, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v212, v112, v212, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v211, v212, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v212, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v212, v112, v212, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v214, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v214, v112, v214, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v213, v214, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v214, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v214, v112, v214, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v216, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v216, v112, v216, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v215, v216, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v216, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v216, v112, v216, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v218, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v218, v112, v218, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v217, v218, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v218, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v218, v112, v218, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v221, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v221, v112, v221, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v219, v221, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v221, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v221, v112, v221, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v223, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v223, v112, v223, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v222, v223, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v223, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v223, v112, v223, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v225, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v225, v112, v225, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v224, v225, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v225, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v225, v112, v225, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 18                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v227, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v227, v112, v227, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v226, v227, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v227, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v227, v112, v227, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v229, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v229, v112, v229, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v228, v229, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v229, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v229, v112, v229, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v231, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v231, v112, v231, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v230, v231, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v231, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v231, v112, v231, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v233, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v233, v112, v233, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v232, v233, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v233, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v233, v112, v233, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v235, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v235, v112, v235, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v234, v235, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v235, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v235, v112, v235, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v237, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v237, v112, v237, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v236, v237, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v237, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v237, v112, v237, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v239, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v239, v112, v239, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v238, v239, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v239, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v239, v112, v239, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v241, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v241, v112, v241, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v240, v241, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v241, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v241, v112, v241, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v243, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v243, v112, v243, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v242, v243, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v243, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v243, v112, v243, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v245, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v245, v112, v245, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v244, v245, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v245, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v245, v112, v245, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v247, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v247, v112, v247, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v246, v247, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v247, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v247, v112, v247, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v249, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v249, v112, v249, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v248, v249, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v249, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v249, v112, v249, s70                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (19, 0, 0, 0), (19, 1, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+128], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+129], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+130], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+131], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+132], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+133], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+134], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+135], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+152], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+154], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+158], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+160], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_mov_b32 v114, 0xffff0000                         // mask for pack two bfloat16 element to 32bit
v_mov_b32 v115, 0x7fff0000                         // fp32 Nan
v_mov_b32 v116, 0x7fff                             // rounding bias for bfloat16
v_lshlrev_b32 v110, 16, v161                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+117], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+117], v[vgprValuC+117] // check Nan
v_bfe_u32 v113, v[vgprValuC+117], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+117], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+117], v113, v115, s68
v_lshrrev_b32 v117, 16, v[vgprValuC+117]           // convert C to bf16
buffer_store_b16 v117, v162, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v163                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+118], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+118], v[vgprValuC+118] // check Nan
v_bfe_u32 v113, v[vgprValuC+118], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+118], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+118], v113, v115, s68
v_lshrrev_b32 v118, 16, v[vgprValuC+118]           // convert C to bf16
buffer_store_b16 v118, v164, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v165                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+119], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+119], v[vgprValuC+119] // check Nan
v_bfe_u32 v113, v[vgprValuC+119], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+119], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+119], v113, v115, s68
v_lshrrev_b32 v119, 16, v[vgprValuC+119]           // convert C to bf16
buffer_store_b16 v119, v166, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v167                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+120], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+120], v[vgprValuC+120] // check Nan
v_bfe_u32 v113, v[vgprValuC+120], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+120], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+120], v113, v115, s68
v_lshrrev_b32 v120, 16, v[vgprValuC+120]           // convert C to bf16
buffer_store_b16 v120, v168, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v169                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+121], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+121], v[vgprValuC+121] // check Nan
v_bfe_u32 v113, v[vgprValuC+121], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+121], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+121], v113, v115, s68
v_lshrrev_b32 v121, 16, v[vgprValuC+121]           // convert C to bf16
buffer_store_b16 v121, v170, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v171                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+122], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+122], v[vgprValuC+122] // check Nan
v_bfe_u32 v113, v[vgprValuC+122], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+122], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+122], v113, v115, s68
v_lshrrev_b32 v122, 16, v[vgprValuC+122]           // convert C to bf16
buffer_store_b16 v122, v172, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v173                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+123], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+123], v[vgprValuC+123] // check Nan
v_bfe_u32 v113, v[vgprValuC+123], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+123], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+123], v113, v115, s68
v_lshrrev_b32 v123, 16, v[vgprValuC+123]           // convert C to bf16
buffer_store_b16 v123, v174, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v175                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+124], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+124], v[vgprValuC+124] // check Nan
v_bfe_u32 v113, v[vgprValuC+124], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+124], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+124], v113, v115, s68
v_lshrrev_b32 v124, 16, v[vgprValuC+124]           // convert C to bf16
buffer_store_b16 v124, v176, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v177                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+125], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+125], v[vgprValuC+125] // check Nan
v_bfe_u32 v113, v[vgprValuC+125], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+125], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+125], v113, v115, s68
v_lshrrev_b32 v125, 16, v[vgprValuC+125]           // convert C to bf16
buffer_store_b16 v125, v178, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v179                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+126], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+126], v[vgprValuC+126] // check Nan
v_bfe_u32 v113, v[vgprValuC+126], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+126], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+126], v113, v115, s68
v_lshrrev_b32 v126, 16, v[vgprValuC+126]           // convert C to bf16
buffer_store_b16 v126, v180, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v181                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+127], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+127], v[vgprValuC+127] // check Nan
v_bfe_u32 v113, v[vgprValuC+127], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+127], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+127], v113, v115, s68
v_lshrrev_b32 v127, 16, v[vgprValuC+127]           // convert C to bf16
buffer_store_b16 v127, v182, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v183                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+128], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+128], v[vgprValuC+128] // check Nan
v_bfe_u32 v113, v[vgprValuC+128], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+128], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+128], v113, v115, s68
v_lshrrev_b32 v128, 16, v[vgprValuC+128]           // convert C to bf16
buffer_store_b16 v128, v184, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v185                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+129], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+129], v[vgprValuC+129] // check Nan
v_bfe_u32 v113, v[vgprValuC+129], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+129], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+129], v113, v115, s68
v_lshrrev_b32 v129, 16, v[vgprValuC+129]           // convert C to bf16
buffer_store_b16 v129, v186, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v187                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+130], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+130], v[vgprValuC+130] // check Nan
v_bfe_u32 v113, v[vgprValuC+130], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+130], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+130], v113, v115, s68
v_lshrrev_b32 v130, 16, v[vgprValuC+130]           // convert C to bf16
buffer_store_b16 v130, v188, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v189                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+131], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+131], v[vgprValuC+131] // check Nan
v_bfe_u32 v113, v[vgprValuC+131], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+131], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+131], v113, v115, s68
v_lshrrev_b32 v131, 16, v[vgprValuC+131]           // convert C to bf16
buffer_store_b16 v131, v190, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v191                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+132], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+132], v[vgprValuC+132] // check Nan
v_bfe_u32 v113, v[vgprValuC+132], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+132], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+132], v113, v115, s68
v_lshrrev_b32 v132, 16, v[vgprValuC+132]           // convert C to bf16
buffer_store_b16 v132, v192, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v193                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+133], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+133], v[vgprValuC+133] // check Nan
v_bfe_u32 v113, v[vgprValuC+133], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+133], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+133], v113, v115, s68
v_lshrrev_b32 v133, 16, v[vgprValuC+133]           // convert C to bf16
buffer_store_b16 v133, v194, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v195                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+134], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+134], v[vgprValuC+134] // check Nan
v_bfe_u32 v113, v[vgprValuC+134], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+134], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+134], v113, v115, s68
v_lshrrev_b32 v134, 16, v[vgprValuC+134]           // convert C to bf16
buffer_store_b16 v134, v196, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v197                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+135], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+135], v[vgprValuC+135] // check Nan
v_bfe_u32 v113, v[vgprValuC+135], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+135], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+135], v113, v115, s68
v_lshrrev_b32 v135, 16, v[vgprValuC+135]           // convert C to bf16
buffer_store_b16 v135, v198, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v199                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+136], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+136], v[vgprValuC+136] // check Nan
v_bfe_u32 v113, v[vgprValuC+136], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+136], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+136], v113, v115, s68
v_lshrrev_b32 v136, 16, v[vgprValuC+136]           // convert C to bf16
buffer_store_b16 v136, v200, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v201                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+137], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+137], v[vgprValuC+137] // check Nan
v_bfe_u32 v113, v[vgprValuC+137], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+137], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+137], v113, v115, s68
v_lshrrev_b32 v137, 16, v[vgprValuC+137]           // convert C to bf16
buffer_store_b16 v137, v202, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v203                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+138], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+138], v[vgprValuC+138] // check Nan
v_bfe_u32 v113, v[vgprValuC+138], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+138], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+138], v113, v115, s68
v_lshrrev_b32 v138, 16, v[vgprValuC+138]           // convert C to bf16
buffer_store_b16 v138, v204, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v205                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+139], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+139], v[vgprValuC+139] // check Nan
v_bfe_u32 v113, v[vgprValuC+139], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+139], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+139], v113, v115, s68
v_lshrrev_b32 v139, 16, v[vgprValuC+139]           // convert C to bf16
buffer_store_b16 v139, v206, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v207                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+140], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+140], v[vgprValuC+140] // check Nan
v_bfe_u32 v113, v[vgprValuC+140], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+140], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+140], v113, v115, s68
v_lshrrev_b32 v140, 16, v[vgprValuC+140]           // convert C to bf16
buffer_store_b16 v140, v208, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v209                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+141], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+141], v[vgprValuC+141] // check Nan
v_bfe_u32 v113, v[vgprValuC+141], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+141], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+141], v113, v115, s68
v_lshrrev_b32 v141, 16, v[vgprValuC+141]           // convert C to bf16
buffer_store_b16 v141, v210, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v211                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+142], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+142], v[vgprValuC+142] // check Nan
v_bfe_u32 v113, v[vgprValuC+142], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+142], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+142], v113, v115, s68
v_lshrrev_b32 v142, 16, v[vgprValuC+142]           // convert C to bf16
buffer_store_b16 v142, v212, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v213                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+143], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+143], v[vgprValuC+143] // check Nan
v_bfe_u32 v113, v[vgprValuC+143], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+143], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+143], v113, v115, s68
v_lshrrev_b32 v143, 16, v[vgprValuC+143]           // convert C to bf16
buffer_store_b16 v143, v214, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v215                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+144], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+144], v[vgprValuC+144] // check Nan
v_bfe_u32 v113, v[vgprValuC+144], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+144], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+144], v113, v115, s68
v_lshrrev_b32 v144, 16, v[vgprValuC+144]           // convert C to bf16
buffer_store_b16 v144, v216, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v217                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+145], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+145], v[vgprValuC+145] // check Nan
v_bfe_u32 v113, v[vgprValuC+145], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+145], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+145], v113, v115, s68
v_lshrrev_b32 v145, 16, v[vgprValuC+145]           // convert C to bf16
buffer_store_b16 v145, v218, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v219                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+146], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+146], v[vgprValuC+146] // check Nan
v_bfe_u32 v113, v[vgprValuC+146], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+146], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+146], v113, v115, s68
v_lshrrev_b32 v146, 16, v[vgprValuC+146]           // convert C to bf16
buffer_store_b16 v146, v221, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v222                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+147], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+147], v[vgprValuC+147] // check Nan
v_bfe_u32 v113, v[vgprValuC+147], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+147], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+147], v113, v115, s68
v_lshrrev_b32 v147, 16, v[vgprValuC+147]           // convert C to bf16
buffer_store_b16 v147, v223, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v224                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+148], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+148], v[vgprValuC+148] // check Nan
v_bfe_u32 v113, v[vgprValuC+148], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+148], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+148], v113, v115, s68
v_lshrrev_b32 v148, 16, v[vgprValuC+148]           // convert C to bf16
buffer_store_b16 v148, v225, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v226                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+149], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+149], v[vgprValuC+149] // check Nan
v_bfe_u32 v113, v[vgprValuC+149], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+149], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+149], v113, v115, s68
v_lshrrev_b32 v149, 16, v[vgprValuC+149]           // convert C to bf16
buffer_store_b16 v149, v227, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v228                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+150], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+150], v[vgprValuC+150] // check Nan
v_bfe_u32 v113, v[vgprValuC+150], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+150], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+150], v113, v115, s68
v_lshrrev_b32 v150, 16, v[vgprValuC+150]           // convert C to bf16
buffer_store_b16 v150, v229, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v230                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+151], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+151], v[vgprValuC+151] // check Nan
v_bfe_u32 v113, v[vgprValuC+151], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+151], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+151], v113, v115, s68
v_lshrrev_b32 v151, 16, v[vgprValuC+151]           // convert C to bf16
buffer_store_b16 v151, v231, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v232                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+152], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+152], v[vgprValuC+152] // check Nan
v_bfe_u32 v113, v[vgprValuC+152], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+152], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+152], v113, v115, s68
v_lshrrev_b32 v152, 16, v[vgprValuC+152]           // convert C to bf16
buffer_store_b16 v152, v233, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v234                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+153], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+153], v[vgprValuC+153] // check Nan
v_bfe_u32 v113, v[vgprValuC+153], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+153], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+153], v113, v115, s68
v_lshrrev_b32 v153, 16, v[vgprValuC+153]           // convert C to bf16
buffer_store_b16 v153, v235, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v236                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+154], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+154], v[vgprValuC+154] // check Nan
v_bfe_u32 v113, v[vgprValuC+154], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+154], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+154], v113, v115, s68
v_lshrrev_b32 v154, 16, v[vgprValuC+154]           // convert C to bf16
buffer_store_b16 v154, v237, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v238                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+155], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+155], v[vgprValuC+155] // check Nan
v_bfe_u32 v113, v[vgprValuC+155], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+155], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+155], v113, v115, s68
v_lshrrev_b32 v155, 16, v[vgprValuC+155]           // convert C to bf16
buffer_store_b16 v155, v239, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v240                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+156], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+156], v[vgprValuC+156] // check Nan
v_bfe_u32 v113, v[vgprValuC+156], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+156], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+156], v113, v115, s68
v_lshrrev_b32 v156, 16, v[vgprValuC+156]           // convert C to bf16
buffer_store_b16 v156, v241, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v242                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+157], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+157], v[vgprValuC+157] // check Nan
v_bfe_u32 v113, v[vgprValuC+157], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+157], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+157], v113, v115, s68
v_lshrrev_b32 v157, 16, v[vgprValuC+157]           // convert C to bf16
buffer_store_b16 v157, v243, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v244                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+158], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+158], v[vgprValuC+158] // check Nan
v_bfe_u32 v113, v[vgprValuC+158], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+158], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+158], v113, v115, s68
v_lshrrev_b32 v158, 16, v[vgprValuC+158]           // convert C to bf16
buffer_store_b16 v158, v245, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v246                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+159], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+159], v[vgprValuC+159] // check Nan
v_bfe_u32 v113, v[vgprValuC+159], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+159], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+159], v113, v115, s68
v_lshrrev_b32 v159, 16, v[vgprValuC+159]           // convert C to bf16
buffer_store_b16 v159, v247, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v248                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+160], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+160], v[vgprValuC+160] // check Nan
v_bfe_u32 v113, v[vgprValuC+160], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+160], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+160], v113, v115, s68
v_lshrrev_b32 v160, 16, v[vgprValuC+160]           // convert C to bf16
buffer_store_b16 v160, v249, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (22,0,0,0:vw1); (22,1,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1); (24,0,0,0:vw1); (24,1,0,0:vw1); (25,0,0,0:vw1); (25,1,0,0:vw1); (26,0,0,0:vw1); (26,1,0,0:vw1); (27,0,0,0:vw1); (27,1,0,0:vw1); (28,0,0,0:vw1); (28,1,0,0:vw1); (29,0,0,0:vw1); (29,1,0,0:vw1); (30,0,0,0:vw1); (30,1,0,0:vw1); (31,0,0,0:vw1); (31,1,0,0:vw1); (32,0,0,0:vw1); (32,1,0,0:vw1); (33,0,0,0:vw1); (33,1,0,0:vw1); (34,0,0,0:vw1); (34,1,0,0:vw1); (35,0,0,0:vw1); (35,1,0,0:vw1); (36,0,0,0:vw1); (36,1,0,0:vw1); (37,0,0,0:vw1); (37,1,0,0:vw1); (38,0,0,0:vw1); (38,1,0,0:vw1); (39,0,0,0:vw1); (39,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v112, BufferOOB
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v154, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v112, v154, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v153, v154, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v154, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v112, v154, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v156, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v112, v156, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v155, v156, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v156, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v112, v156, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v158, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v112, v158, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v157, v158, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v158, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v112, v158, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v160, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v160, v112, v160, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v159, v160, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v160, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v160, v112, v160, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 18                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v162, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v162, v112, v162, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v161, v162, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v162, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v162, v112, v162, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v164, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v164, v112, v164, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v163, v164, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v164, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v164, v112, v164, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v166, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v166, v112, v166, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v165, v166, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v166, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v166, v112, v166, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v168, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v168, v112, v168, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v167, v168, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v168, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v168, v112, v168, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v170, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v170, v112, v170, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v169, v170, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v170, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v170, v112, v170, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v172, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v172, v112, v172, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v171, v172, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v172, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v172, v112, v172, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v174, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v174, v112, v174, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v173, v174, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v174, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v174, v112, v174, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v176, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v176, v112, v176, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v175, v176, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v176, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v176, v112, v176, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v178, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v178, v112, v178, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v177, v178, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v178, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v178, v112, v178, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v180, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v180, v112, v180, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v179, v180, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v180, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v180, v112, v180, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v182, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v182, v112, v182, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v181, v182, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v182, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v182, v112, v182, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v184, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v184, v112, v184, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v183, v184, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v184, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v184, v112, v184, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v186, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v186, v112, v186, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v185, v186, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v186, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v186, v112, v186, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v188, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v188, v112, v188, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v187, v188, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v188, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v188, v112, v188, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v190, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v190, v112, v190, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v189, v190, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v190, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v190, v112, v190, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v192, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v192, v112, v192, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v191, v192, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v192, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v192, v112, v192, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(32,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 18                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 18                // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 18                // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v194, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v194, v112, v194, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v193, v194, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v194, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v194, v112, v194, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(32,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v196, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v196, v112, v196, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v195, v196, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v196, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v196, v112, v196, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(33,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v198, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v198, v112, v198, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v197, v198, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v198, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v198, v112, v198, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(33,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v200, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v200, v112, v200, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v199, v200, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v200, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v200, v112, v200, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(34,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v202, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v202, v112, v202, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v201, v202, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v202, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v202, v112, v202, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(34,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v204, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v204, v112, v204, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v203, v204, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v204, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v204, v112, v204, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(35,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v206, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v206, v112, v206, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v205, v206, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v206, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v206, v112, v206, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(35,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v208, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v208, v112, v208, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v207, v208, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v208, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v208, v112, v208, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(36,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v210, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v210, v112, v210, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v209, v210, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v210, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v210, v112, v210, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(36,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v212, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v212, v112, v212, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v211, v212, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v212, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v212, v112, v212, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(37,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v214, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v214, v112, v214, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v213, v214, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v214, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v214, v112, v214, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(37,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v216, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v216, v112, v216, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v215, v216, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v216, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v216, v112, v216, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(38,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v218, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v218, v112, v218, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v217, v218, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v218, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v218, v112, v218, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(38,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v221, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v221, v112, v221, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v219, v221, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v221, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v221, v112, v221, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(39,0,0,0) */
v_add_co_u32 v107, vcc_lo, v107, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v108, v108, s68                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v109, v109, s68                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v106, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v223, v108, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v223, v112, v223, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v222, v223, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v223, v109, v106, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v223, v112, v223, s70                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(39,0,1,0) */
v_add_co_u32 v110, vcc_lo, v106, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v110, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s70, v107, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v225, v108, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v225, v112, v225, s70                // LDC clip if OOB. offset
buffer_load_d16_b16 v224, v225, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v225, v109, v110, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v225, v112, v225, s70                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(22, 0, 0, 0), (22, 1, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0), (24, 0, 0, 0), (24, 1, 0, 0), (25, 0, 0, 0), (25, 1, 0, 0), (26, 0, 0, 0), (26, 1, 0, 0), (27, 0, 0, 0), (27, 1, 0, 0), (28, 0, 0, 0), (28, 1, 0, 0), (29, 0, 0, 0), (29, 1, 0, 0), (30, 0, 0, 0), (30, 1, 0, 0), (31, 0, 0, 0), (31, 1, 0, 0), (32, 0, 0, 0), (32, 1, 0, 0), (33, 0, 0, 0), (33, 1, 0, 0), (34, 0, 0, 0), (34, 1, 0, 0), (35, 0, 0, 0), (35, 1, 0, 0), (36, 0, 0, 0), (36, 1, 0, 0), (37, 0, 0, 0), (37, 1, 0, 0), (38, 0, 0, 0), (38, 1, 0, 0), (39, 0, 0, 0), (39, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+48] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+56] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+49] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+57] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+50] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+58] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+51] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+128], s[sgprAlpha], v[vgprValuC+59] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+129], s[sgprAlpha], v[vgprValuC+52] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+130], s[sgprAlpha], v[vgprValuC+60] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+131], s[sgprAlpha], v[vgprValuC+53] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+132], s[sgprAlpha], v[vgprValuC+61] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+133], s[sgprAlpha], v[vgprValuC+54] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+134], s[sgprAlpha], v[vgprValuC+62] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+135], s[sgprAlpha], v[vgprValuC+55] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+63] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+64] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+72] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+65] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+73] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+66] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+74] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+67] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+75] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+68] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+76] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+69] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+77] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+70] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+78] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+71] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+152], s[sgprAlpha], v[vgprValuC+79] // Multiply MI out reg with alpha
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_mov_b32 v114, 0xffff0000                         // mask for pack two bfloat16 element to 32bit
v_mov_b32 v115, 0x7fff0000                         // fp32 Nan
v_mov_b32 v116, 0x7fff                             // rounding bias for bfloat16
v_lshlrev_b32 v110, 16, v153                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+117], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+117], v[vgprValuC+117] // check Nan
v_bfe_u32 v113, v[vgprValuC+117], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+117], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+117], v113, v115, s68
v_lshrrev_b32 v117, 16, v[vgprValuC+117]           // convert C to bf16
buffer_store_b16 v117, v154, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v155                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+118], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+118], v[vgprValuC+118] // check Nan
v_bfe_u32 v113, v[vgprValuC+118], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+118], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+118], v113, v115, s68
v_lshrrev_b32 v118, 16, v[vgprValuC+118]           // convert C to bf16
buffer_store_b16 v118, v156, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v157                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+119], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+119], v[vgprValuC+119] // check Nan
v_bfe_u32 v113, v[vgprValuC+119], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+119], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+119], v113, v115, s68
v_lshrrev_b32 v119, 16, v[vgprValuC+119]           // convert C to bf16
buffer_store_b16 v119, v158, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v159                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+120], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+120], v[vgprValuC+120] // check Nan
v_bfe_u32 v113, v[vgprValuC+120], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+120], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+120], v113, v115, s68
v_lshrrev_b32 v120, 16, v[vgprValuC+120]           // convert C to bf16
buffer_store_b16 v120, v160, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v161                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+121], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+121], v[vgprValuC+121] // check Nan
v_bfe_u32 v113, v[vgprValuC+121], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+121], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+121], v113, v115, s68
v_lshrrev_b32 v121, 16, v[vgprValuC+121]           // convert C to bf16
buffer_store_b16 v121, v162, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v163                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+122], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+122], v[vgprValuC+122] // check Nan
v_bfe_u32 v113, v[vgprValuC+122], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+122], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+122], v113, v115, s68
v_lshrrev_b32 v122, 16, v[vgprValuC+122]           // convert C to bf16
buffer_store_b16 v122, v164, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v165                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+123], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+123], v[vgprValuC+123] // check Nan
v_bfe_u32 v113, v[vgprValuC+123], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+123], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+123], v113, v115, s68
v_lshrrev_b32 v123, 16, v[vgprValuC+123]           // convert C to bf16
buffer_store_b16 v123, v166, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v167                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+124], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+124], v[vgprValuC+124] // check Nan
v_bfe_u32 v113, v[vgprValuC+124], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+124], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+124], v113, v115, s68
v_lshrrev_b32 v124, 16, v[vgprValuC+124]           // convert C to bf16
buffer_store_b16 v124, v168, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v169                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+125], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+125], v[vgprValuC+125] // check Nan
v_bfe_u32 v113, v[vgprValuC+125], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+125], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+125], v113, v115, s68
v_lshrrev_b32 v125, 16, v[vgprValuC+125]           // convert C to bf16
buffer_store_b16 v125, v170, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v171                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+126], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+126], v[vgprValuC+126] // check Nan
v_bfe_u32 v113, v[vgprValuC+126], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+126], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+126], v113, v115, s68
v_lshrrev_b32 v126, 16, v[vgprValuC+126]           // convert C to bf16
buffer_store_b16 v126, v172, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v173                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+127], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+127], v[vgprValuC+127] // check Nan
v_bfe_u32 v113, v[vgprValuC+127], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+127], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+127], v113, v115, s68
v_lshrrev_b32 v127, 16, v[vgprValuC+127]           // convert C to bf16
buffer_store_b16 v127, v174, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v175                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+128], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+128], v[vgprValuC+128] // check Nan
v_bfe_u32 v113, v[vgprValuC+128], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+128], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+128], v113, v115, s68
v_lshrrev_b32 v128, 16, v[vgprValuC+128]           // convert C to bf16
buffer_store_b16 v128, v176, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v177                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+129], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+129], v[vgprValuC+129] // check Nan
v_bfe_u32 v113, v[vgprValuC+129], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+129], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+129], v113, v115, s68
v_lshrrev_b32 v129, 16, v[vgprValuC+129]           // convert C to bf16
buffer_store_b16 v129, v178, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v179                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+130], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+130], v[vgprValuC+130] // check Nan
v_bfe_u32 v113, v[vgprValuC+130], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+130], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+130], v113, v115, s68
v_lshrrev_b32 v130, 16, v[vgprValuC+130]           // convert C to bf16
buffer_store_b16 v130, v180, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v181                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+131], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+131], v[vgprValuC+131] // check Nan
v_bfe_u32 v113, v[vgprValuC+131], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+131], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+131], v113, v115, s68
v_lshrrev_b32 v131, 16, v[vgprValuC+131]           // convert C to bf16
buffer_store_b16 v131, v182, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v183                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+132], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+132], v[vgprValuC+132] // check Nan
v_bfe_u32 v113, v[vgprValuC+132], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+132], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+132], v113, v115, s68
v_lshrrev_b32 v132, 16, v[vgprValuC+132]           // convert C to bf16
buffer_store_b16 v132, v184, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v185                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+133], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+133], v[vgprValuC+133] // check Nan
v_bfe_u32 v113, v[vgprValuC+133], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+133], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+133], v113, v115, s68
v_lshrrev_b32 v133, 16, v[vgprValuC+133]           // convert C to bf16
buffer_store_b16 v133, v186, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v187                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+134], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+134], v[vgprValuC+134] // check Nan
v_bfe_u32 v113, v[vgprValuC+134], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+134], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+134], v113, v115, s68
v_lshrrev_b32 v134, 16, v[vgprValuC+134]           // convert C to bf16
buffer_store_b16 v134, v188, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v189                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+135], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+135], v[vgprValuC+135] // check Nan
v_bfe_u32 v113, v[vgprValuC+135], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+135], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+135], v113, v115, s68
v_lshrrev_b32 v135, 16, v[vgprValuC+135]           // convert C to bf16
buffer_store_b16 v135, v190, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v191                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+136], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+136], v[vgprValuC+136] // check Nan
v_bfe_u32 v113, v[vgprValuC+136], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+136], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+136], v113, v115, s68
v_lshrrev_b32 v136, 16, v[vgprValuC+136]           // convert C to bf16
buffer_store_b16 v136, v192, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v193                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+137], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+137], v[vgprValuC+137] // check Nan
v_bfe_u32 v113, v[vgprValuC+137], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+137], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+137], v113, v115, s68
v_lshrrev_b32 v137, 16, v[vgprValuC+137]           // convert C to bf16
buffer_store_b16 v137, v194, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v195                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+138], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+138], v[vgprValuC+138] // check Nan
v_bfe_u32 v113, v[vgprValuC+138], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+138], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+138], v113, v115, s68
v_lshrrev_b32 v138, 16, v[vgprValuC+138]           // convert C to bf16
buffer_store_b16 v138, v196, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v197                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+139], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+139], v[vgprValuC+139] // check Nan
v_bfe_u32 v113, v[vgprValuC+139], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+139], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+139], v113, v115, s68
v_lshrrev_b32 v139, 16, v[vgprValuC+139]           // convert C to bf16
buffer_store_b16 v139, v198, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v199                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+140], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+140], v[vgprValuC+140] // check Nan
v_bfe_u32 v113, v[vgprValuC+140], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+140], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+140], v113, v115, s68
v_lshrrev_b32 v140, 16, v[vgprValuC+140]           // convert C to bf16
buffer_store_b16 v140, v200, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v201                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+141], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+141], v[vgprValuC+141] // check Nan
v_bfe_u32 v113, v[vgprValuC+141], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+141], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+141], v113, v115, s68
v_lshrrev_b32 v141, 16, v[vgprValuC+141]           // convert C to bf16
buffer_store_b16 v141, v202, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v203                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+142], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+142], v[vgprValuC+142] // check Nan
v_bfe_u32 v113, v[vgprValuC+142], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+142], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+142], v113, v115, s68
v_lshrrev_b32 v142, 16, v[vgprValuC+142]           // convert C to bf16
buffer_store_b16 v142, v204, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v205                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+143], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+143], v[vgprValuC+143] // check Nan
v_bfe_u32 v113, v[vgprValuC+143], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+143], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+143], v113, v115, s68
v_lshrrev_b32 v143, 16, v[vgprValuC+143]           // convert C to bf16
buffer_store_b16 v143, v206, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v207                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+144], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+144], v[vgprValuC+144] // check Nan
v_bfe_u32 v113, v[vgprValuC+144], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+144], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+144], v113, v115, s68
v_lshrrev_b32 v144, 16, v[vgprValuC+144]           // convert C to bf16
buffer_store_b16 v144, v208, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v209                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+145], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+145], v[vgprValuC+145] // check Nan
v_bfe_u32 v113, v[vgprValuC+145], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+145], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+145], v113, v115, s68
v_lshrrev_b32 v145, 16, v[vgprValuC+145]           // convert C to bf16
buffer_store_b16 v145, v210, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v211                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+146], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+146], v[vgprValuC+146] // check Nan
v_bfe_u32 v113, v[vgprValuC+146], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+146], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+146], v113, v115, s68
v_lshrrev_b32 v146, 16, v[vgprValuC+146]           // convert C to bf16
buffer_store_b16 v146, v212, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v213                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+147], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+147], v[vgprValuC+147] // check Nan
v_bfe_u32 v113, v[vgprValuC+147], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+147], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+147], v113, v115, s68
v_lshrrev_b32 v147, 16, v[vgprValuC+147]           // convert C to bf16
buffer_store_b16 v147, v214, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v215                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+148], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+148], v[vgprValuC+148] // check Nan
v_bfe_u32 v113, v[vgprValuC+148], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+148], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+148], v113, v115, s68
v_lshrrev_b32 v148, 16, v[vgprValuC+148]           // convert C to bf16
buffer_store_b16 v148, v216, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v217                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+149], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+149], v[vgprValuC+149] // check Nan
v_bfe_u32 v113, v[vgprValuC+149], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+149], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+149], v113, v115, s68
v_lshrrev_b32 v149, 16, v[vgprValuC+149]           // convert C to bf16
buffer_store_b16 v149, v218, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v219                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+150], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+150], v[vgprValuC+150] // check Nan
v_bfe_u32 v113, v[vgprValuC+150], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+150], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+150], v113, v115, s68
v_lshrrev_b32 v150, 16, v[vgprValuC+150]           // convert C to bf16
buffer_store_b16 v150, v221, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v222                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+151], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+151], v[vgprValuC+151] // check Nan
v_bfe_u32 v113, v[vgprValuC+151], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+151], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+151], v113, v115, s68
v_lshrrev_b32 v151, 16, v[vgprValuC+151]           // convert C to bf16
buffer_store_b16 v151, v223, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v110, 16, v224                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+152], v110, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+152], v[vgprValuC+152] // check Nan
v_bfe_u32 v113, v[vgprValuC+152], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v113, v[vgprValuC+152], v113, v116      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+152], v113, v115, s68
v_lshrrev_b32 v152, 16, v[vgprValuC+152]           // convert C to bf16
buffer_store_b16 v152, v225, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_SK_Partials_1:
label_GW_Partials_E0:
s_mov_b64 s[sgprSrdWS:sgprSrdWS+1], s[sgprAddressWS:sgprAddressWS+1]
s_mov_b32 s[sgprSrdWS+2], BufferOOB
s_mov_b32 s[sgprSrdWS+3], Srd127_96

s_mul_i32 s7, 0xa000, s[sgprPersistentWorkGroupIndex] // Offset to correct partials tile (low word)
s_mul_hi_u32 s8, 0xa000, s[sgprPersistentWorkGroupIndex] // partials tile offset (high word) for 64-bit SRD
s_add_u32 s[sgprSrdWS+0], s[sgprSrdWS+0], s7       // add lo to SRD
s_addc_u32 s[sgprSrdWS+1], s[sgprSrdWS+1], s8      // add hi (offset high word + lo carry) to SRD

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=136 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Partials Write Batch #0 (d1,d0,vc1,vc0) = */
/*      (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1); (24,0,0,0:vw1); (24,1,0,0:vw1); (25,0,0,0:vw1); (25,1,0,0:vw1); (26,0,0,0:vw1); (26,1,0,0:vw1); (27,0,0,0:vw1); (27,1,0,0:vw1); (28,0,0,0:vw1); (28,1,0,0:vw1); (29,0,0,0:vw1); (29,1,0,0:vw1); (30,0,0,0:vw1); (30,1,0,0:vw1); (31,0,0,0:vw1); (31,1,0,0:vw1); (32,0,0,0:vw1); (32,1,0,0:vw1); (33,0,0,0:vw1); (33,1,0,0:vw1); (34,0,0,0:vw1); (34,1,0,0:vw1); (35,0,0,0:vw1); (35,1,0,0:vw1); (36,0,0,0:vw1); (36,1,0,0:vw1); (37,0,0,0:vw1); (37,1,0,0:vw1); (38,0,0,0:vw1); (38,1,0,0:vw1); (39,0,0,0:vw1); (39,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v[vgprValuC+119], v[vgprValuC+0]         // copy MI out reg to vreg[0]
v_mov_b32 v[vgprValuC+120], v[vgprValuC+8]         // copy MI out reg to vreg[1]
v_mov_b32 v[vgprValuC+121], v[vgprValuC+1]         // copy MI out reg to vreg[2]
v_mov_b32 v[vgprValuC+122], v[vgprValuC+9]         // copy MI out reg to vreg[3]
v_mov_b32 v[vgprValuC+123], v[vgprValuC+2]         // copy MI out reg to vreg[4]
v_mov_b32 v[vgprValuC+124], v[vgprValuC+10]        // copy MI out reg to vreg[5]
v_mov_b32 v[vgprValuC+125], v[vgprValuC+3]         // copy MI out reg to vreg[6]
v_mov_b32 v[vgprValuC+126], v[vgprValuC+11]        // copy MI out reg to vreg[7]
v_mov_b32 v[vgprValuC+127], v[vgprValuC+4]         // copy MI out reg to vreg[8]
v_mov_b32 v[vgprValuC+128], v[vgprValuC+12]        // copy MI out reg to vreg[9]
v_mov_b32 v[vgprValuC+129], v[vgprValuC+5]         // copy MI out reg to vreg[10]
v_mov_b32 v[vgprValuC+130], v[vgprValuC+13]        // copy MI out reg to vreg[11]
v_mov_b32 v[vgprValuC+131], v[vgprValuC+6]         // copy MI out reg to vreg[12]
v_mov_b32 v[vgprValuC+132], v[vgprValuC+14]        // copy MI out reg to vreg[13]
v_mov_b32 v[vgprValuC+133], v[vgprValuC+7]         // copy MI out reg to vreg[14]
v_mov_b32 v[vgprValuC+134], v[vgprValuC+15]        // copy MI out reg to vreg[15]
v_mov_b32 v[vgprValuC+135], v[vgprValuC+16]        // copy MI out reg to vreg[16]
v_mov_b32 v[vgprValuC+136], v[vgprValuC+24]        // copy MI out reg to vreg[17]
v_mov_b32 v[vgprValuC+137], v[vgprValuC+17]        // copy MI out reg to vreg[18]
v_mov_b32 v[vgprValuC+138], v[vgprValuC+25]        // copy MI out reg to vreg[19]
v_mov_b32 v[vgprValuC+139], v[vgprValuC+18]        // copy MI out reg to vreg[20]
v_mov_b32 v[vgprValuC+140], v[vgprValuC+26]        // copy MI out reg to vreg[21]
v_mov_b32 v[vgprValuC+141], v[vgprValuC+19]        // copy MI out reg to vreg[22]
v_mov_b32 v[vgprValuC+142], v[vgprValuC+27]        // copy MI out reg to vreg[23]
v_mov_b32 v[vgprValuC+143], v[vgprValuC+20]        // copy MI out reg to vreg[24]
v_mov_b32 v[vgprValuC+144], v[vgprValuC+28]        // copy MI out reg to vreg[25]
v_mov_b32 v[vgprValuC+145], v[vgprValuC+21]        // copy MI out reg to vreg[26]
v_mov_b32 v[vgprValuC+146], v[vgprValuC+29]        // copy MI out reg to vreg[27]
v_mov_b32 v[vgprValuC+147], v[vgprValuC+22]        // copy MI out reg to vreg[28]
v_mov_b32 v[vgprValuC+148], v[vgprValuC+30]        // copy MI out reg to vreg[29]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+23]        // copy MI out reg to vreg[30]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+31]        // copy MI out reg to vreg[31]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+32]        // copy MI out reg to vreg[32]
v_mov_b32 v[vgprValuC+152], v[vgprValuC+40]        // copy MI out reg to vreg[33]
v_mov_b32 v[vgprValuC+153], v[vgprValuC+33]        // copy MI out reg to vreg[34]
v_mov_b32 v[vgprValuC+154], v[vgprValuC+41]        // copy MI out reg to vreg[35]
v_mov_b32 v[vgprValuC+155], v[vgprValuC+34]        // copy MI out reg to vreg[36]
v_mov_b32 v[vgprValuC+156], v[vgprValuC+42]        // copy MI out reg to vreg[37]
v_mov_b32 v[vgprValuC+157], v[vgprValuC+35]        // copy MI out reg to vreg[38]
v_mov_b32 v[vgprValuC+158], v[vgprValuC+43]        // copy MI out reg to vreg[39]
v_mov_b32 v[vgprValuC+159], v[vgprValuC+36]        // copy MI out reg to vreg[40]
v_mov_b32 v[vgprValuC+160], v[vgprValuC+44]        // copy MI out reg to vreg[41]
v_mov_b32 v[vgprValuC+161], v[vgprValuC+37]        // copy MI out reg to vreg[42]
v_mov_b32 v[vgprValuC+162], v[vgprValuC+45]        // copy MI out reg to vreg[43]
v_mov_b32 v[vgprValuC+163], v[vgprValuC+38]        // copy MI out reg to vreg[44]
v_mov_b32 v[vgprValuC+164], v[vgprValuC+46]        // copy MI out reg to vreg[45]
v_mov_b32 v[vgprValuC+165], v[vgprValuC+39]        // copy MI out reg to vreg[46]
v_mov_b32 v[vgprValuC+166], v[vgprValuC+47]        // copy MI out reg to vreg[47]
v_mov_b32 v[vgprValuC+167], v[vgprValuC+48]        // copy MI out reg to vreg[48]
v_mov_b32 v[vgprValuC+168], v[vgprValuC+56]        // copy MI out reg to vreg[49]
v_mov_b32 v[vgprValuC+169], v[vgprValuC+49]        // copy MI out reg to vreg[50]
v_mov_b32 v[vgprValuC+170], v[vgprValuC+57]        // copy MI out reg to vreg[51]
v_mov_b32 v[vgprValuC+171], v[vgprValuC+50]        // copy MI out reg to vreg[52]
v_mov_b32 v[vgprValuC+172], v[vgprValuC+58]        // copy MI out reg to vreg[53]
v_mov_b32 v[vgprValuC+173], v[vgprValuC+51]        // copy MI out reg to vreg[54]
v_mov_b32 v[vgprValuC+174], v[vgprValuC+59]        // copy MI out reg to vreg[55]
v_mov_b32 v[vgprValuC+175], v[vgprValuC+52]        // copy MI out reg to vreg[56]
v_mov_b32 v[vgprValuC+176], v[vgprValuC+60]        // copy MI out reg to vreg[57]
v_mov_b32 v[vgprValuC+177], v[vgprValuC+53]        // copy MI out reg to vreg[58]
v_mov_b32 v[vgprValuC+178], v[vgprValuC+61]        // copy MI out reg to vreg[59]
v_mov_b32 v[vgprValuC+179], v[vgprValuC+54]        // copy MI out reg to vreg[60]
v_mov_b32 v[vgprValuC+180], v[vgprValuC+62]        // copy MI out reg to vreg[61]
v_mov_b32 v[vgprValuC+181], v[vgprValuC+55]        // copy MI out reg to vreg[62]
v_mov_b32 v[vgprValuC+182], v[vgprValuC+63]        // copy MI out reg to vreg[63]
v_mov_b32 v[vgprValuC+183], v[vgprValuC+64]        // copy MI out reg to vreg[64]
v_mov_b32 v[vgprValuC+184], v[vgprValuC+72]        // copy MI out reg to vreg[65]
v_mov_b32 v[vgprValuC+185], v[vgprValuC+65]        // copy MI out reg to vreg[66]
v_mov_b32 v[vgprValuC+186], v[vgprValuC+73]        // copy MI out reg to vreg[67]
v_mov_b32 v[vgprValuC+187], v[vgprValuC+66]        // copy MI out reg to vreg[68]
v_mov_b32 v[vgprValuC+188], v[vgprValuC+74]        // copy MI out reg to vreg[69]
v_mov_b32 v[vgprValuC+189], v[vgprValuC+67]        // copy MI out reg to vreg[70]
v_mov_b32 v[vgprValuC+190], v[vgprValuC+75]        // copy MI out reg to vreg[71]
v_mov_b32 v[vgprValuC+191], v[vgprValuC+68]        // copy MI out reg to vreg[72]
v_mov_b32 v[vgprValuC+192], v[vgprValuC+76]        // copy MI out reg to vreg[73]
v_mov_b32 v[vgprValuC+193], v[vgprValuC+69]        // copy MI out reg to vreg[74]
v_mov_b32 v[vgprValuC+194], v[vgprValuC+77]        // copy MI out reg to vreg[75]
v_mov_b32 v[vgprValuC+195], v[vgprValuC+70]        // copy MI out reg to vreg[76]
v_mov_b32 v[vgprValuC+196], v[vgprValuC+78]        // copy MI out reg to vreg[77]
v_mov_b32 v[vgprValuC+197], v[vgprValuC+71]        // copy MI out reg to vreg[78]
v_mov_b32 v[vgprValuC+198], v[vgprValuC+79]        // copy MI out reg to vreg[79]

/* apply mask, calc new C and issue writes */
v_mov_b32 v114, 0xffff0000                         // mask for pack two bfloat16 element to 32bit
v_mov_b32 v115, 0x7fff0000                         // fp32 Nan
v_mov_b32 v116, 0x7fff                             // rounding bias for bfloat16
v_lshlrev_b32 v117, 2, v[vgprSerial]               // v117 = v[vgprSerial] * 4
s_mov_b32 s8, 0                                    // Init sgpr offset
buffer_store_b32 v119, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v120, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v121, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v122, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v123, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v124, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v125, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v126, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v127, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v128, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v129, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v130, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v131, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v132, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v133, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v134, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v135, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v136, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v137, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v138, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v139, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v140, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v141, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v142, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v143, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v144, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v145, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v146, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v147, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v148, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v149, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v150, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v151, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v152, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v153, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v154, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v155, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v156, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v157, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v158, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v159, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v160, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v161, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v162, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v163, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v164, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v165, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v166, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v167, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v168, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v169, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v170, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v171, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v172, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v173, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v174, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v175, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v176, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v177, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v178, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v179, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v180, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v181, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v182, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v183, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v184, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v185, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v186, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v187, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v188, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v189, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v190, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v191, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v192, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v193, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v194, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v195, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v196, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v197, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v198, v117, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_waitcnt_vscnt null 0                             // wait for data store
s_barrier                                          // store all data before setting flag
s_lshl_b32 s8, s[sgprPersistentWorkGroupIndex], 2  // flag offset based on physical partial slot
v_readfirstlane_b32 s7, v[vgprSerial]              // Wave 0 updates flags
s_cmp_eq_u32 s7, 0                                 // Check for wave 0
s_cbranch_scc0 label_SK_SkipFlagSet                // Skip flag set
s_mov_b32 s7, 1                                    // flag data
v_mov_b32 v119, s7                                 // move flag value to vgpr
v_mov_b32 v120, 0                                  // zero vaddr offset
s_mov_b64 s[68:69], s[sgprAddressFlags:sgprAddressFlags+1]
s_mov_b32 s70, BufferOOB
s_mov_b32 s71, Srd127_96
buffer_store_b32 v119, v120, s[68:71], s8 offen offset:0 glc dlc // set flag
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
