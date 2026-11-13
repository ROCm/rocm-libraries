
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
.text
.protected RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBG_UserArgs_MT192x256x64_MI16x16x1_SK_gfx1151
.globl RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBG_UserArgs_MT192x256x64_MI16x16x1_SK_gfx1151
.p2align 8
.type RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBG_UserArgs_MT192x256x64_MI16x16x1_SK_gfx1151,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBG_UserArgs_MT192x256x64_MI16x16x1_SK_gfx1151
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 256 // vgprs
  .amdhsa_next_free_sgpr 100 // sgprs
  .amdhsa_group_segment_fixed_size 64512 // lds bytes
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
/* Num SGPR   =100 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 24 x 4 */
/* SubGroup= 8 x 64 */
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
  - .name: RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBG_UserArgs_MT192x256x64_MI16x16x1_SK_gfx1151
    .symbol: 'RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBG_UserArgs_MT192x256x64_MI16x16x1_SK_gfx1151.kd'
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
    .group_segment_fixed_size:   64512
    .kernarg_segment_align:      8
    .kernarg_segment_size:       200
    .max_flat_workgroup_size:    512
    .private_segment_fixed_size: 0
    .sgpr_count:                 100
    .sgpr_spill_count:           0
    .vgpr_count:                 256
    .vgpr_spill_count:           0
    .wavefront_size:             32
...
.end_amdgpu_metadata
RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBG_UserArgs_MT192x256x64_MI16x16x1_SK_gfx1151:
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
/* ValuC range: [0-96), serializedStore enabled */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprBase, 118
.set vgprLocalWriteAddrA, 113
.set vgprLocalWriteAddrB, 114
.set vgprGlobalReadOffsetA, 96
.set vgprGlobalReadOffsetB, 99
.set vgprGlobalReadOffsetScaleA, 107
.set vgprG2LScaleA, 110
.set vgprLocalReadAddrA, 204
.set vgprLocalReadAddrB, 115
.set vgprSerial, 205

/******************************************/
/* VGPR Macro Assignments                 */
/******************************************/
.set vgprValuA_X0_I0_BASE, vgprBase+0
.set vgprValuB_X0_I0_BASE, vgprBase+25
.set vgprG2LA_BASE, vgprBase+58
.set vgprG2LB_BASE, vgprBase+70
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
.set sgprScaleALiftMagic, 73

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

.set MT0, 192
.set MT1, 256
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
s_mov_b32 m0, 0xfc00                               // LDS clamp at 64512 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id
v_mov_b32 v[vgprG2LScaleA+0], 0                    // scaleA: low half of the f32 scale
v_mov_b32 v[vgprG2LScaleA+1], 0                    // scaleA: low half of the f32 scale
v_mov_b32 v[vgprG2LScaleA+2], 0                    // scaleA: low half of the f32 scale
s_mov_b32 vcc_hi, 0                                // Ensure hi bits are zero

/* remap workgroup to XCCs */
s_lshr_b32 s88, s[sgprWGM], 0x10                   // Get WGMXCC
s_ff1_i32_b32 s88, s88                             // Get log(WGMXCC)
s_lshr_b32 s89, s[sgprWGM], 0x16                   // Get CU_Count
/* remap WGs if WGMXCC > 1 ( log(WGMXCC) > 0 ) */
s_cmp_gt_i32 s88, 0
s_cbranch_scc0 label_skip_WGMXCC
/* only remap WGs in the range */
s_lshr_b32 s85, s18, s88
s_lshl_b32 s85, s85, s88
s_cmp_ge_u32 s[sgprWorkGroup0], s85
s_cbranch_scc1 label_skip_WGMXCC
s_cmp_eq_u32 s89, 0                                // CU_Count == 0 ?
s_cbranch_scc0 label_XCCG_nonzero
s_lshr_b32 s85, s[sgprWorkGroup0], s88
s_bfm_b32 s86, s88, 0
s_and_b32 s86, s[sgprWorkGroup0], s86
s_lshr_b32 s87, s18, s88
s_mul_i32 s86, s86, s87
s_add_u32 s[sgprWorkGroup0], s85, s86
s_branch label_skip_WGMXCC
label_XCCG_nonzero:
/* temp0 = (wg//CU_Count)*CU_Count */
v_cvt_f64_u32 v[0:1], s89                          // s85 = s[sgprWorkGroup0] / s89
v_rcp_f64 v[0:1], v[0:1]                           // s85 = s[sgprWorkGroup0] / s89
v_cvt_f64_u32 v[2:3], s[sgprWorkGroup0]            // s85 = s[sgprWorkGroup0] / s89
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s85 = s[sgprWorkGroup0] / s89
v_cvt_u32_f64 v0, v[0:1]                           // s85 = s[sgprWorkGroup0] / s89
v_mul_lo_u32 v1, v0, s89                           // s85 = s[sgprWorkGroup0] / s89
v_sub_nc_u32 v2, s[sgprWorkGroup0], v1             // s85 = s[sgprWorkGroup0] / s89
v_cmp_ge_u32 vcc_lo, v2, s89                       // s85 = s[sgprWorkGroup0] / s89
s_mov_b32 exec_lo, vcc_lo                          // s85 = s[sgprWorkGroup0] / s89
v_add_nc_u32 v0, v0, 1                             // s85 = s[sgprWorkGroup0] / s89
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v1, v0, s89                           // s85 = s[sgprWorkGroup0] / s89
v_sub_nc_u32 v2, s[sgprWorkGroup0], v1             // s85 = s[sgprWorkGroup0] / s89
v_readfirstlane_b32 s85, v0                        // quotient
v_readfirstlane_b32 s86, v2                        // remainder
s_mul_i32 s85, s85, s89
/* temp1 = (wg%CU_Count)//WGMXCC */
s_lshr_b32 s86, s86, s88
/* temp0 = temp0 + temp1 */
s_add_u32 s85, s85, s86
/* temp1 = (wg%WGMXCC) * ((WGs - (WGs//CU_Count) * CU_Count) if (wg > (WGs//CU_Count) * CU_Count) else CU_Count)//WGMXCC */
v_cvt_f64_u32 v[0:1], s89                          // s86 = s18 / s89
v_rcp_f64 v[0:1], v[0:1]                           // s86 = s18 / s89
v_cvt_f64_u32 v[2:3], s18                          // s86 = s18 / s89
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s86 = s18 / s89
v_cvt_u32_f64 v0, v[0:1]                           // s86 = s18 / s89
v_mul_lo_u32 v1, v0, s89                           // s86 = s18 / s89
v_sub_nc_u32 v2, s18, v1                           // s86 = s18 / s89
v_cmp_ge_u32 vcc_lo, v2, s89                       // s86 = s18 / s89
s_mov_b32 exec_lo, vcc_lo                          // s86 = s18 / s89
v_add_nc_u32 v0, v0, 1                             // s86 = s18 / s89
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s86, v0                        // quotient
s_mul_i32 s86, s86, s89
s_sub_u32 s87, s18, s86
s_cmp_gt_u32 s[sgprWorkGroup0], s86
s_cselect_b32 s86, s87, s89
s_lshr_b32 s86, s86, s88
s_bfm_b32 s87, s88, 0
s_and_b32 s87, s[sgprWorkGroup0], s87
s_mul_i32 s86, s86, s87
/* WorkGroup0 = temp0 + temp1 */
s_add_u32 s[sgprWorkGroup0], s85, s86
label_skip_WGMXCC:  /// skip WGMXCC if no enough WGs to remap
s_cmp_eq_u32 s16, 3
s_cbranch_scc1 label_ArgType3_Routed_To_ArgType0
s_cmp_eq_u32 s16, 0
s_cbranch_scc0 label_MultiGemm
label_ArgType3_Routed_To_ArgType0:
/* init: add vgpr [118...293) to pool */
/* init: add vgpr [0...96) to pool */
/* init: add agpr [0...0) to pool */
s_waitcnt lgkmcnt(0)                               // wait for 128/0 bytes of kern args over preload
s_mul_hi_u32 s15, s[sgprSizesFree+0], 44739243     // STATIC_DIV: divisor=192
s_mul_i32 s14, s[sgprSizesFree+0], 44739243        // tmp = dividend * magic
s_lshr_b64 s[14:15], s[14:15], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s[sgprNumWorkGroups0], s14               // quotient
s_mul_i32 s14, s[sgprNumWorkGroups0], 192          // quotient*divisor
s_sub_u32 s12, s[sgprSizesFree+0], s14             // rReg = dividend - quotient*divisor
s_cmp_gt_u32 s12, 0                                // Check if remainder > 0
s_addc_u32 s[sgprNumWorkGroups0], s[sgprNumWorkGroups0], 0 // Increment if remainder > 0
s_lshr_b32 s[sgprNumWorkGroups1], s[sgprSizesFree+1], 8 // s[sgprNumWorkGroups1] = s[sgprSizesFree+1] / 256
s_and_b32 s12, 255, s[sgprSizesFree+1]             // s12 = s[sgprSizesFree+1] % 256
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
s_mul_hi_u32 s15, s20, 44739243                    // STATIC_DIV: divisor=192
s_mul_i32 s14, s20, 44739243                       // tmp = dividend * magic
s_lshr_b64 s[14:15], s[14:15], 33                  // tmp0 = quotient
s_mul_i32 s15, s14, 192                            // tmp1 = quotient * divisor
s_cmp_lg_u32 s15, s20                              // if (quotient * divisor != dividend), result+=1
s_addc_u32 s16, s14, 0                             // if (quotient * divisor != dividend), result+=1
s_lshr_b32 s17, s21, 8                             // s17 = s21 / 256
s_and_b32 s14, 255, s21                            // s14 = s21 % 256
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
s_mul_hi_u32 s15, s20, 44739243                    // STATIC_DIV: divisor=192
s_mul_i32 s14, s20, 44739243                       // tmp = dividend * magic
s_lshr_b64 s[14:15], s[14:15], 33                  // tmp0 = quotient
s_mul_i32 s15, s14, 192                            // tmp1 = quotient * divisor
s_cmp_lg_u32 s15, s20                              // if (quotient * divisor != dividend), result+=1
s_addc_u32 s16, s14, 0                             // if (quotient * divisor != dividend), result+=1
s_lshr_b32 s17, s21, 8                             // s17 = s21 / 256
s_and_b32 s14, 255, s21                            // s14 = s21 % 256
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
/* init: add vgpr [118...293) to pool */
/* init: add vgpr [0...96) to pool */
/* init: add agpr [0...0) to pool */
s_waitcnt lgkmcnt(0)                               // wait for 128/0 bytes of kern args over preload
s_mul_hi_u32 s15, s[sgprSizesFree+0], 44739243     // STATIC_DIV: divisor=192
s_mul_i32 s14, s[sgprSizesFree+0], 44739243        // tmp = dividend * magic
s_lshr_b64 s[14:15], s[14:15], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s[sgprNumWorkGroups0], s14               // quotient
s_mul_i32 s14, s[sgprNumWorkGroups0], 192          // quotient*divisor
s_sub_u32 s12, s[sgprSizesFree+0], s14             // rReg = dividend - quotient*divisor
s_cmp_gt_u32 s12, 0                                // Check if remainder > 0
s_addc_u32 s[sgprNumWorkGroups0], s[sgprNumWorkGroups0], 0 // Increment if remainder > 0
s_lshr_b32 s[sgprNumWorkGroups1], s[sgprSizesFree+1], 8 // s[sgprNumWorkGroups1] = s[sgprSizesFree+1] / 256
s_and_b32 s12, 255, s[sgprSizesFree+1]             // s12 = s[sgprSizesFree+1] % 256
s_cmp_gt_u32 s12, 0                                // Check if remainder > 0
s_addc_u32 s[sgprNumWorkGroups1], s[sgprNumWorkGroups1], 0 // Increment if remainder > 0

/* Early stop if N(SizeFreeJ) == 0 */
s_cmp_eq_u32 s[sgprSizeJ], 0
s_cbranch_scc0 label_NoEarlyStop_N0
label_EarlyStop_if_N_is_0:
s_endpgm
label_NoEarlyStop_N0:

label_MultiGemmEnd:
.set sgprSrdA, 84
.set sgprSrdB, 88
.set sgprShadowLimitA, 92
.set sgprShadowLimitB, 94
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
v_lshrrev_b32 v3, 7, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(128)
v_and_b32 v3, 3, v3                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(4)
v_lshl_add_u32 v1, v3, 10, v1                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(1024); 7. final local read offset: flrOffset = lrOffset + WOffset

/* local read addresses: final offsets a */
v_lshrrev_b32 v2, 5, v[vgprSerial]                 // 2 = Serial / 32
v_lshrrev_b32 v2, 4, v2                            // LSU offset: Get LSU wave_id
s_mov_b32 s11, 64                                  // LSU offset: stride = lsuStride(64) when umlds==True
v_mul_lo_u32 v2, s11, v2                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_nc_u32 v[vgprLocalReadAddrA], v2, v0         // Final Offset: offset = (lro0+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrA], 1, v[vgprLocalReadAddrA] //  (multiple bpe)
v_lshrrev_b32 v3, 7, v[vgprLocalReadAddrA]         // Final Offset: padding 16 per block 128
v_lshl_add_u32 v[vgprLocalReadAddrA], v3, 4, v[vgprLocalReadAddrA] // Final Offset: padding 16 per block 128

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 5, v[vgprSerial]                 // 0 = Serial / 32
v_lshrrev_b32 v0, 4, v0                            // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(64) when umlds==True (dup assign opt.)
v_mul_lo_u32 v0, s11, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_nc_u32 v[vgprLocalReadAddrB], v0, v1         // Final Offset: offset = (lro1+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrB], 1, v[vgprLocalReadAddrB] //  (multiple bpe)
v_lshrrev_b32 v2, 7, v[vgprLocalReadAddrB]         // Final Offset: padding 16 per block 128
v_lshl_add_u32 v[vgprLocalReadAddrB], v2, 4, v[vgprLocalReadAddrB] // Final Offset: padding 16 per block 128

/* local read addresses: declare addresses a */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, 0x6c00, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

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
v_add_co_u32 v[vgprLocalWriteAddrB], vcc_lo, 0x6c00, v[vgprLocalWriteAddrB] // lwFOB = lw1J + lwL*MT1J + LDS_OFFSET_B=27648

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
v_add_co_u32 v7, vcc_lo, 64, v6                    // groA0I_1 += LSPA
v_add_co_u32 v8, vcc_lo, 64, v7                    // groA0I_2 += LSPA

/* global read addresses: tile offsets b */
v_mov_b32 v9, v2                                   // groB1J_0
v_add_co_u32 v10, vcc_lo, 32, v9                   // groB1J_1 += LSPB
v_add_co_u32 v11, vcc_lo, 32, v10                  // groB1J_2 += LSPB
v_add_co_u32 v12, vcc_lo, 32, v11                  // groB1J_3 += LSPB
v_add_co_u32 v13, vcc_lo, 32, v12                  // groB1J_4 += LSPB
v_add_co_u32 v14, vcc_lo, 32, v13                  // groB1J_5 += LSPB
v_add_co_u32 v15, vcc_lo, 32, v14                  // groB1J_6 += LSPB
v_add_co_u32 v16, vcc_lo, 32, v15                  // groB1J_7 += LSPB

/* global read addresses: unroll offsets a */
v_mov_b32 v17, v1                                  // groAL_0

/* global read addresses: unroll offsets b */
v_mov_b32 v18, v3                                  // groBL_0

/* global read addresses: addresses a */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s15, s[sgprWorkGroup0], 192           // WorkGroup[01] * MT
s_mul_i32 s14, s[sgprWorkGroup0], 192              // WorkGroup[01] * MT
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
s_mul_i32 s12, s[sgprWorkGroup0], 192              // scaleA: workgroup row origin
s_mul_i32 s12, s12, s[sgprStrideScaleA]            // scaleA: * row stride
s_mul_i32 s12, s12, 2                              // scaleA: elements -> bytes
s_mul_i32 s13, s[sgprSizeI], s[sgprStrideScaleA]   // scaleA: SizeI * row stride
s_mul_i32 s13, s13, 2                              // scaleA: tensor bytes
s_sub_u32 s[sgprSrdScaleA+2], s13, s12             // scaleA: buffer limit from the workgroup origin
s_add_u32 s[sgprSrdScaleA+0], s[sgprAddressScaleA+0], s12 // scaleA: SRD base lo
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
s_mul_i32 s12, s[sgprStreamKLocalStart], 64        // scaleA: first K of this range
s_lshr_b32 s12, s12, s[sgprScaleAGShift]           // scaleA: groups before this range
s_mul_i32 s13, s12, 2                              // scaleA: skipped bytes
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], s13 // scaleA: skip groups before the split (lower)
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA: skip groups before the split (upper)
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], s13 // scaleA: limit shrinks by the skip

/* global read addresses: addresses b */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s15, s[sgprWorkGroup1], 256           // WorkGroup[01] * MT
s_mul_i32 s14, s[sgprWorkGroup1], 256              // WorkGroup[01] * MT
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
v_mul_lo_u32 v19, s[sgprStrideA0I], v[6]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+0+0], vcc_lo, v[17], v[19+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+0+0], 0x8, v[vgprGlobalReadOffsetA+0+0] // add prepad for pointer shift
v_lshrrev_b32 v19, s[sgprScaleAGShift], v17        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+0], s[sgprStrideScaleA], v6 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+0], v19, v[vgprGlobalReadOffsetScaleA+0] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+0], 1, v[vgprGlobalReadOffsetScaleA+0] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetA+0], 1, v[vgprGlobalReadOffsetA+0] //  (multiple bpe)
v_mul_lo_u32 v19, s[sgprStrideA0I], v[7]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+1+0], vcc_lo, v[17], v[19+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+1+0], 0x8, v[vgprGlobalReadOffsetA+1+0] // add prepad for pointer shift
v_lshrrev_b32 v19, s[sgprScaleAGShift], v17        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+1], s[sgprStrideScaleA], v7 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+1], v19, v[vgprGlobalReadOffsetScaleA+1] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+1], 1, v[vgprGlobalReadOffsetScaleA+1] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetA+1], 1, v[vgprGlobalReadOffsetA+1] //  (multiple bpe)
v_mul_lo_u32 v19, s[sgprStrideA0I], v[8]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+2+0], vcc_lo, v[17], v[19+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+2+0], 0x8, v[vgprGlobalReadOffsetA+2+0] // add prepad for pointer shift
v_lshrrev_b32 v19, s[sgprScaleAGShift], v17        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+2], s[sgprStrideScaleA], v8 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+2], v19, v[vgprGlobalReadOffsetScaleA+2] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+2], 1, v[vgprGlobalReadOffsetScaleA+2] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetA+2], 1, v[vgprGlobalReadOffsetA+2] //  (multiple bpe)
/* ============================================================= */

/* global read addresses: final offsets b */
/* ============================================================= */
v_mul_lo_u32 v6, s[sgprStrideB1J], v[9]            // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+0+0], vcc_lo, v[18], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+0+0], 0x4, v[vgprGlobalReadOffsetB+0+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+0], 1, v[vgprGlobalReadOffsetB+0] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[10]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+1+0], vcc_lo, v[18], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+1+0], 0x4, v[vgprGlobalReadOffsetB+1+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+1], 1, v[vgprGlobalReadOffsetB+1] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[11]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+2+0], vcc_lo, v[18], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+2+0], 0x4, v[vgprGlobalReadOffsetB+2+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+2], 1, v[vgprGlobalReadOffsetB+2] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[12]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+3+0], vcc_lo, v[18], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+3+0], 0x4, v[vgprGlobalReadOffsetB+3+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+3], 1, v[vgprGlobalReadOffsetB+3] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[13]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+4+0], vcc_lo, v[18], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+4+0], 0x4, v[vgprGlobalReadOffsetB+4+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+4], 1, v[vgprGlobalReadOffsetB+4] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[14]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+5+0], vcc_lo, v[18], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+5+0], 0x4, v[vgprGlobalReadOffsetB+5+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+5], 1, v[vgprGlobalReadOffsetB+5] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[15]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+6+0], vcc_lo, v[18], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+6+0], 0x4, v[vgprGlobalReadOffsetB+6+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+6], 1, v[vgprGlobalReadOffsetB+6] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[16]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+7+0], vcc_lo, v[18], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+7+0], 0x4, v[vgprGlobalReadOffsetB+7+0] // add prepad for pointer shift
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
s_cbranch_scc1 label_ShadowInitStart               // skip to ShadowInitStart iter b/c numIter==0
buffer_load_b32 v[vgprG2LA+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b32 v[vgprG2LA+6], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b32 v[vgprG2LA+10], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_2_0

/* global read block-scale A */
buffer_load_d16_hi_b16 v[vgprG2LScaleA+0], v[vgprGlobalReadOffsetScaleA+0], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 0 as f32
buffer_load_d16_hi_b16 v[vgprG2LScaleA+1], v[vgprGlobalReadOffsetScaleA+1], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 1 as f32
buffer_load_d16_hi_b16 v[vgprG2LScaleA+2], v[vgprGlobalReadOffsetScaleA+2], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 2 as f32
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

s_mul_i32 s98, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_mul_hi_u32 s97, s98, s[sgprStrideC1J]            // ScaleC s98 by Stride
s_mul_i32 s96, s98, s[sgprStrideC1J]               // ScaleC s98 by Stride
s_lshl_b64 s[96:97], s[96:97], s11                 // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s96        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s97       // add hi to SRD
s_mul_hi_u32 s97, s98, s[sgprStrideD1J]            // ScaleD s98 by Stride
s_mul_i32 s96, s98, s[sgprStrideD1J]               // ScaleD s98 by Stride
s_lshl_b64 s[96:97], s[96:97], s52                 // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s96        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s97       // add hi to SRD

s_and_b32 s53, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s53, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_StridedBatchedGemmLoadC
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_GeneralBatchedGemmLoadC
label_StridedBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s97, s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride
s_mul_i32 s96, s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride
s_lshl_b64 s[96:97], s[96:97], s11                 // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s96        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s97       // add hi to SRD
s_branch label_GeneralBatchedGemmLoadC_End
label_GeneralBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for General Batched GEMM
s_mul_i32 s96, 8, s[sgprWorkGroup2]                // Compute stride in bytes into Pointer Array
s_add_u32 s96, s96, s[sgprAddressC+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s97, s[sgprAddressC+1], 0               // Offsetting to the location [Higher half of address]
s_load_b64 s[96:97], s[96:97], 0                   // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for the Matrix Address Load from the Pointer Array
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s96        // Offsetting within the Batch Matrix [Lower half of address]
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s97       // Offsetting within the Batch Matrix [Higher half of address]
s_load_b64 s[96:97], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x98 // Load batchOffsetC from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s96        // Add matrix address to SRD (low)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s97       // Add matrix address to SRD (high)
label_GeneralBatchedGemmLoadC_End:  /// End of label GeneralBatchedGemmLoadC
s_and_b32 s53, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s53, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_StridedBatchedGemmLoadD
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_GeneralBatchedGemmLoadD
label_StridedBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s97, s[sgprWorkGroup2], s[sgprStrideDK] // ScaleD s[sgprWorkGroup2] by Stride
s_mul_i32 s96, s[sgprWorkGroup2], s[sgprStrideDK]  // ScaleD s[sgprWorkGroup2] by Stride
s_lshl_b64 s[96:97], s[96:97], s52                 // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s96        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s97       // add hi to SRD
s_branch label_GeneralBatchedGemmLoadD_End
label_GeneralBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for General Batched GEMM
s_mul_i32 s96, 8, s[sgprWorkGroup2]                // Compute stride in bytes into Pointer Array
s_add_u32 s96, s96, s[sgprAddressD+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s97, s[sgprAddressD+1], 0               // Offsetting to the location [Higher half of address]
s_load_b64 s[96:97], s[96:97], 0                   // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for the Matrix Address Load from the Pointer Array
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s96        // Offsetting within the Batch Matrix [Lower half of address]
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s97       // Offsetting within the Batch Matrix [Higher half of address]
s_load_b64 s[96:97], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x90 // Load batchOffsetD from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s96        // Add matrix address to SRD (low)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s97       // Add matrix address to SRD (high)
label_GeneralBatchedGemmLoadD_End:  /// End of label GeneralBatchedGemmLoadD

s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_SK_SplitSrd                   // Skip this block if using single-kernel stream-k fixup
s_cmp_eq_u32 s[sgprskTiles], 1                     // split == 1 ?
s_cbranch_scc1 label_SK_SplitSrd                   // branch if split == 1
// Split Output Buffer offset: Free0 + (Free1-1)*StrideC1J + (Free2-1)*StrideCK * SplitIdx * bpe%s
s_mul_hi_u32 s97, s[sgprSizesFree+0], s[sgprSkPartialIdx] // Free0
s_mul_i32 s96, s[sgprSizesFree+0], s[sgprSkPartialIdx] // Free0
s_sub_u32 s98, s[sgprSizesFree+1], 1               // Free1
s_mul_i32 s98, s98, s[sgprSkPartialIdx]            // Free1
s_mul_hi_u32 s99, s98, s[sgprStrideC1J]            // Free1
s_mul_i32 s98, s98, s[sgprStrideC1J]               // Free1
s_add_u32 s96, s96, s98                            // Free1
s_addc_u32 s97, s97, s99                           // Free1
s_sub_u32 s98, s[sgprSizesFree+2], 1               // Free2
s_mul_i32 s98, s98, s[sgprSkPartialIdx]            // Free2
s_mul_hi_u32 s99, s98, s[sgprStrideCK]             // Free2
s_mul_i32 s98, s98, s[sgprStrideCK]                // Free2
s_add_u32 s96, s96, s98                            // Free2
s_addc_u32 s97, s97, s99                           // Free2
s_lshl_b64 s[96:97], s[96:97], 2                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s96        // add lo GSU offset to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s97       // add hi GSU offset to SRD
label_SK_SplitSrd:

/* initC: remove ValuC vgpr buffer [0...96) from pool */

/* initC: remove acc vgpr buffer [0...0) from pool */

/* initC: remove ValuA/B vgpr buffer [118...175) from pool */
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
v_mov_b32 v207, v[vgprG2LScaleA+0]                 // scaleA: copy on the other VGPR bank
v_xor_b32 v206, 0x88888888, v[vgprG2LA+2+0]        // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v208, v206, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v209, v206, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v211, v206, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v210, v206, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v208, s[sgprScaleALiftMagic+6], v208 :: v_dual_subrev_f32 v209, s[sgprScaleALiftMagic+7], v209 // w4a16: n - c, exact
v_dual_subrev_f32 v211, s[sgprScaleALiftMagic+8], v211 :: v_dual_subrev_f32 v210, s[sgprScaleALiftMagic+9], v210 // w4a16: n - c, exact
v_dual_mul_f32 v208, v110, v208 :: v_dual_mul_f32 v209, v207, v209 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v211, v110, v211 :: v_dual_add_nc_u32 v208, s[sgprScaleALiftMagic+5], v208 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v210, v207, v210 :: v_dual_add_nc_u32 v209, s[sgprScaleALiftMagic+5], v209 // w4a16: (n - c) * s :: round
v_and_or_b32 v212, v206, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v206, 20, v206                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v213, v206, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v215, v206, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v214, v206, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v212, s[sgprScaleALiftMagic+10], v212 :: v_dual_add_nc_u32 v211, s[sgprScaleALiftMagic+5], v211 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v213, s[sgprScaleALiftMagic+6], v213 :: v_dual_add_nc_u32 v210, s[sgprScaleALiftMagic+5], v210 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v215, s[sgprScaleALiftMagic+7], v215 :: v_dual_subrev_f32 v214, s[sgprScaleALiftMagic+8], v214 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+0+0], v209, v208, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+1], v210, v211, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v212, v110, v212 :: v_dual_mul_f32 v213, v207, v213 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v215, v110, v215 :: v_dual_add_nc_u32 v212, s[sgprScaleALiftMagic+5], v212 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v214, v207, v214 :: v_dual_add_nc_u32 v213, s[sgprScaleALiftMagic+5], v213 // w4a16: (n - c) * s :: round
v_add_nc_u32 v215, s[sgprScaleALiftMagic+5], v215  // w4a16: round
v_add_nc_u32 v214, s[sgprScaleALiftMagic+5], v214  // w4a16: round
v_perm_b32 v[vgprG2LA+0+2], v213, v212, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+3], v214, v215, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
v_mov_b32 v206, v[vgprG2LScaleA+1]                 // scaleA: copy on the other VGPR bank
v_xor_b32 v207, 0x88888888, v[vgprG2LA+6+0]        // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v208, v207, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v209, v207, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v211, v207, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v210, v207, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v208, s[sgprScaleALiftMagic+6], v208 :: v_dual_subrev_f32 v209, s[sgprScaleALiftMagic+7], v209 // w4a16: n - c, exact
v_dual_subrev_f32 v211, s[sgprScaleALiftMagic+8], v211 :: v_dual_subrev_f32 v210, s[sgprScaleALiftMagic+9], v210 // w4a16: n - c, exact
v_dual_mul_f32 v208, v111, v208 :: v_dual_mul_f32 v209, v206, v209 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v211, v111, v211 :: v_dual_add_nc_u32 v208, s[sgprScaleALiftMagic+5], v208 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v210, v206, v210 :: v_dual_add_nc_u32 v209, s[sgprScaleALiftMagic+5], v209 // w4a16: (n - c) * s :: round
v_and_or_b32 v212, v207, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v207, 20, v207                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v213, v207, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v215, v207, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v214, v207, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v212, s[sgprScaleALiftMagic+10], v212 :: v_dual_add_nc_u32 v211, s[sgprScaleALiftMagic+5], v211 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v213, s[sgprScaleALiftMagic+6], v213 :: v_dual_add_nc_u32 v210, s[sgprScaleALiftMagic+5], v210 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v215, s[sgprScaleALiftMagic+7], v215 :: v_dual_subrev_f32 v214, s[sgprScaleALiftMagic+8], v214 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+4+0], v209, v208, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+4+1], v210, v211, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v212, v111, v212 :: v_dual_mul_f32 v213, v206, v213 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v215, v111, v215 :: v_dual_add_nc_u32 v212, s[sgprScaleALiftMagic+5], v212 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v214, v206, v214 :: v_dual_add_nc_u32 v213, s[sgprScaleALiftMagic+5], v213 // w4a16: (n - c) * s :: round
v_add_nc_u32 v215, s[sgprScaleALiftMagic+5], v215  // w4a16: round
v_add_nc_u32 v214, s[sgprScaleALiftMagic+5], v214  // w4a16: round
v_perm_b32 v[vgprG2LA+4+2], v213, v212, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+4+3], v214, v215, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:9216 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 9216 sync LDS0
v_mov_b32 v206, v[vgprG2LScaleA+2]                 // scaleA: copy on the other VGPR bank
v_xor_b32 v207, 0x88888888, v[vgprG2LA+10+0]       // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v208, v207, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v209, v207, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v211, v207, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v210, v207, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v208, s[sgprScaleALiftMagic+6], v208 :: v_dual_subrev_f32 v209, s[sgprScaleALiftMagic+7], v209 // w4a16: n - c, exact
v_dual_subrev_f32 v211, s[sgprScaleALiftMagic+8], v211 :: v_dual_subrev_f32 v210, s[sgprScaleALiftMagic+9], v210 // w4a16: n - c, exact
v_dual_mul_f32 v208, v112, v208 :: v_dual_mul_f32 v209, v206, v209 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v211, v112, v211 :: v_dual_add_nc_u32 v208, s[sgprScaleALiftMagic+5], v208 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v210, v206, v210 :: v_dual_add_nc_u32 v209, s[sgprScaleALiftMagic+5], v209 // w4a16: (n - c) * s :: round
v_and_or_b32 v212, v207, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v207, 20, v207                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v213, v207, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v215, v207, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v214, v207, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v212, s[sgprScaleALiftMagic+10], v212 :: v_dual_add_nc_u32 v211, s[sgprScaleALiftMagic+5], v211 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v213, s[sgprScaleALiftMagic+6], v213 :: v_dual_add_nc_u32 v210, s[sgprScaleALiftMagic+5], v210 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v215, s[sgprScaleALiftMagic+7], v215 :: v_dual_subrev_f32 v214, s[sgprScaleALiftMagic+8], v214 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+8+0], v209, v208, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+1], v210, v211, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v212, v112, v212 :: v_dual_mul_f32 v213, v206, v213 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v215, v112, v215 :: v_dual_add_nc_u32 v212, s[sgprScaleALiftMagic+5], v212 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v214, v206, v214 :: v_dual_add_nc_u32 v213, s[sgprScaleALiftMagic+5], v213 // w4a16: (n - c) * s :: round
v_add_nc_u32 v215, s[sgprScaleALiftMagic+5], v215  // w4a16: round
v_add_nc_u32 v214, s[sgprScaleALiftMagic+5], v214  // w4a16: round
v_perm_b32 v[vgprG2LA+8+2], v213, v212, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+3], v214, v215, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:18432 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 18432 sync LDS0

/* local write b */
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+1] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+2:vgprG2LB+2+1] offset:4608 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 4608 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4:vgprG2LB+4+1] offset:9216 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 9216 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+6:vgprG2LB+6+1] offset:13824 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 13824 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8:vgprG2LB+8+1] offset:18432 // lwoB_0_0_4_0 = (0*LSCB)*(MT1J+PAD) + (4*LSPB) = 18432 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+10:vgprG2LB+10+1] offset:23040 // lwoB_0_0_5_0 = (0*LSCB)*(MT1J+PAD) + (5*LSPB) = 23040 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12:vgprG2LB+12+1] offset:27648 // lwoB_0_0_6_0 = (0*LSCB)*(MT1J+PAD) + (6*LSPB) = 27648 sync LDS0
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+14:vgprG2LB+14+1] offset:32256 // lwoB_0_0_7_0 = (0*LSCB)*(MT1J+PAD) + (7*LSPB) = 32256 sync LDS0

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
buffer_load_b64 v[vgprG2LB+0:vgprG2LB+0+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b64 v[vgprG2LB+2:vgprG2LB+2+1], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b64 v[vgprG2LB+4:vgprG2LB+4+1], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b64 v[vgprG2LB+6:vgprG2LB+6+1], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0
buffer_load_b64 v[vgprG2LB+8:vgprG2LB+8+1], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_4_0
buffer_load_b64 v[vgprG2LB+10:vgprG2LB+10+1], v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_5_0
buffer_load_b64 v[vgprG2LB+12:vgprG2LB+12+1], v[vgprGlobalReadOffsetB+6], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_6_0
buffer_load_b64 v[vgprG2LB+14:vgprG2LB+14+1], v[vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_7_0
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
/*  grEndMfmaIndex:6, lwStartMfmaIndex:47, lwEndMfmaIndex:47  */
/*  numMfmaForLR:11, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:46 */
/*  mfmaIndex:0  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:9216 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:9232 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA+0] offset:18432 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA+0] offset:18448 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:9216 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:9232 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(6)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:1  */

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], GlobalReadIncsA // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], GlobalReadIncsA // limit -= inc)
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:2  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:18432 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32
s_waitcnt lgkmcnt(3)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:3  */
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:18448 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* global read inc block-scale A (2 bytes every 1 iters) */
s_add_u32 s[sgprScaleAKCnt], s[sgprScaleAKCnt], 1  // scaleA: one more K iteration done
s_and_b32 s11, s[sgprScaleAKCnt], s[sgprScaleAKMask] // scaleA: SCC = counter has not wrapped
s_cselect_b32 s11, 0, s[sgprScaleAInc]             // scaleA: advance only on a wrap
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:4  */
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], s11 // scaleA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], s11 // scaleA limit -= inc
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:5  */

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], GlobalReadIncsB // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], GlobalReadIncsB // limit -= inc)
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:6  */
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:27648 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:27664 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:7  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/*  mfmaIndex:8  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
/*  mfmaIndex:9  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
/*  mfmaIndex:10  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+80:vgprValuC+80+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+80:vgprValuC+80+7] // left value = v[80+0:87+0]
/*  mfmaIndex:11  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+88:vgprValuC+88+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+88:vgprValuC+88+7] // left value = v[88+0:95+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=6 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=8 */

/* iter 1 */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:47, lwEndMfmaIndex:47  */
/*  numMfmaForLR:11, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:46 */
/*  mfmaIndex:12  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:9248 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:9264 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA+0] offset:18464 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA+0] offset:18480 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:9248 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:9264 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(6)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:13  */
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:14  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:18464 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(3)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:15  */
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:18480 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:16  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:17  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:18  */
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:27680 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:27696 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:19  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/*  mfmaIndex:20  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
/*  mfmaIndex:21  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
/*  mfmaIndex:22  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+80:vgprValuC+80+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+80:vgprValuC+80+7] // left value = v[80+0:87+0]
/*  mfmaIndex:23  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+88:vgprValuC+88+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+88:vgprValuC+88+7] // left value = v[88+0:95+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=0 readsPerIterA=6 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=0 readsPerIterB=8 */

/* iter 2 */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:47, lwEndMfmaIndex:47  */
/*  numMfmaForLR:11, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:46 */
/*  mfmaIndex:24  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:9280 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:9296 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA+0] offset:18496 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA+0] offset:18512 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:9280 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:9296 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(6)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:25  */
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:26  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:18496 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(3)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:27  */
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:18512 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:28  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:29  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:30  */
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:27712 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:27728 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:31  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/*  mfmaIndex:32  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
/*  mfmaIndex:33  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
/*  mfmaIndex:34  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+80:vgprValuC+80+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+80:vgprValuC+80+7] // left value = v[80+0:87+0]
/*  mfmaIndex:35  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+88:vgprValuC+88+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+88:vgprValuC+88+7] // left value = v[88+0:95+0]
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=0 readsPerIterA=6 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=0 readsPerIterB=8 */

/* iter 3 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:47, lwEndMfmaIndex:47  */
/*  numMfmaForLR:11, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:46 */
/*  mfmaIndex:36  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:9312 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:9328 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA+0] offset:18528 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA+0] offset:18544 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:9312 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:9328 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(6)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:37  */
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:38  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:18528 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(3)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:39  */
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:18544 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:40  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:41  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:42  */
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:27744 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:27760 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:43  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/*  mfmaIndex:44  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
/*  mfmaIndex:45  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
/*  mfmaIndex:46  */
/* schedule remaining localreads for one buffer scheduling */
/* localReadsVacancy: latencyLeft 5 */
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_wmma_f32_16x16x16_bf16 v[vgprValuC+80:vgprValuC+80+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+80:vgprValuC+80+7] // left value = v[80+0:87+0]
/*  mfmaIndex:47  */
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(10)                                // wait for global read before writing to local
v_mov_b32 v207, v[vgprG2LScaleA+0]                 // scaleA: copy on the other VGPR bank
v_xor_b32 v206, 0x88888888, v[vgprG2LA+2+0]        // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v208, v206, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v209, v206, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v211, v206, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v210, v206, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v208, s[sgprScaleALiftMagic+6], v208 :: v_dual_subrev_f32 v209, s[sgprScaleALiftMagic+7], v209 // w4a16: n - c, exact
v_dual_subrev_f32 v211, s[sgprScaleALiftMagic+8], v211 :: v_dual_subrev_f32 v210, s[sgprScaleALiftMagic+9], v210 // w4a16: n - c, exact
v_dual_mul_f32 v208, v110, v208 :: v_dual_mul_f32 v209, v207, v209 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v211, v110, v211 :: v_dual_add_nc_u32 v208, s[sgprScaleALiftMagic+5], v208 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v210, v207, v210 :: v_dual_add_nc_u32 v209, s[sgprScaleALiftMagic+5], v209 // w4a16: (n - c) * s :: round
v_and_or_b32 v212, v206, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v206, 20, v206                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v213, v206, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v215, v206, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v214, v206, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v212, s[sgprScaleALiftMagic+10], v212 :: v_dual_add_nc_u32 v211, s[sgprScaleALiftMagic+5], v211 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v213, s[sgprScaleALiftMagic+6], v213 :: v_dual_add_nc_u32 v210, s[sgprScaleALiftMagic+5], v210 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v215, s[sgprScaleALiftMagic+7], v215 :: v_dual_subrev_f32 v214, s[sgprScaleALiftMagic+8], v214 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+0+0], v209, v208, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+1], v210, v211, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v212, v110, v212 :: v_dual_mul_f32 v213, v207, v213 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v215, v110, v215 :: v_dual_add_nc_u32 v212, s[sgprScaleALiftMagic+5], v212 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v214, v207, v214 :: v_dual_add_nc_u32 v213, s[sgprScaleALiftMagic+5], v213 // w4a16: (n - c) * s :: round
v_add_nc_u32 v215, s[sgprScaleALiftMagic+5], v215  // w4a16: round
v_add_nc_u32 v214, s[sgprScaleALiftMagic+5], v214  // w4a16: round
v_perm_b32 v[vgprG2LA+0+2], v213, v212, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+3], v214, v215, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
buffer_load_b32 v[vgprG2LA+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(10)                                // wait for global read before writing to local
v_mov_b32 v206, v[vgprG2LScaleA+1]                 // scaleA: copy on the other VGPR bank
v_xor_b32 v207, 0x88888888, v[vgprG2LA+6+0]        // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v208, v207, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v209, v207, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v211, v207, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v210, v207, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v208, s[sgprScaleALiftMagic+6], v208 :: v_dual_subrev_f32 v209, s[sgprScaleALiftMagic+7], v209 // w4a16: n - c, exact
v_dual_subrev_f32 v211, s[sgprScaleALiftMagic+8], v211 :: v_dual_subrev_f32 v210, s[sgprScaleALiftMagic+9], v210 // w4a16: n - c, exact
v_dual_mul_f32 v208, v111, v208 :: v_dual_mul_f32 v209, v206, v209 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v211, v111, v211 :: v_dual_add_nc_u32 v208, s[sgprScaleALiftMagic+5], v208 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v210, v206, v210 :: v_dual_add_nc_u32 v209, s[sgprScaleALiftMagic+5], v209 // w4a16: (n - c) * s :: round
v_and_or_b32 v212, v207, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v207, 20, v207                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v213, v207, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v215, v207, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v214, v207, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v212, s[sgprScaleALiftMagic+10], v212 :: v_dual_add_nc_u32 v211, s[sgprScaleALiftMagic+5], v211 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v213, s[sgprScaleALiftMagic+6], v213 :: v_dual_add_nc_u32 v210, s[sgprScaleALiftMagic+5], v210 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v215, s[sgprScaleALiftMagic+7], v215 :: v_dual_subrev_f32 v214, s[sgprScaleALiftMagic+8], v214 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+4+0], v209, v208, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+4+1], v210, v211, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v212, v111, v212 :: v_dual_mul_f32 v213, v206, v213 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v215, v111, v215 :: v_dual_add_nc_u32 v212, s[sgprScaleALiftMagic+5], v212 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v214, v206, v214 :: v_dual_add_nc_u32 v213, s[sgprScaleALiftMagic+5], v213 // w4a16: (n - c) * s :: round
v_add_nc_u32 v215, s[sgprScaleALiftMagic+5], v215  // w4a16: round
v_add_nc_u32 v214, s[sgprScaleALiftMagic+5], v214  // w4a16: round
v_perm_b32 v[vgprG2LA+4+2], v213, v212, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+4+3], v214, v215, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:9216 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 9216 sync LDS0
buffer_load_b32 v[vgprG2LA+6], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_1_0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(10)                                // wait for global read before writing to local
v_mov_b32 v206, v[vgprG2LScaleA+2]                 // scaleA: copy on the other VGPR bank
v_xor_b32 v207, 0x88888888, v[vgprG2LA+10+0]       // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v208, v207, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v209, v207, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v211, v207, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v210, v207, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v208, s[sgprScaleALiftMagic+6], v208 :: v_dual_subrev_f32 v209, s[sgprScaleALiftMagic+7], v209 // w4a16: n - c, exact
v_dual_subrev_f32 v211, s[sgprScaleALiftMagic+8], v211 :: v_dual_subrev_f32 v210, s[sgprScaleALiftMagic+9], v210 // w4a16: n - c, exact
v_dual_mul_f32 v208, v112, v208 :: v_dual_mul_f32 v209, v206, v209 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v211, v112, v211 :: v_dual_add_nc_u32 v208, s[sgprScaleALiftMagic+5], v208 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v210, v206, v210 :: v_dual_add_nc_u32 v209, s[sgprScaleALiftMagic+5], v209 // w4a16: (n - c) * s :: round
v_and_or_b32 v212, v207, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v207, 20, v207                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v213, v207, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v215, v207, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v214, v207, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v212, s[sgprScaleALiftMagic+10], v212 :: v_dual_add_nc_u32 v211, s[sgprScaleALiftMagic+5], v211 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v213, s[sgprScaleALiftMagic+6], v213 :: v_dual_add_nc_u32 v210, s[sgprScaleALiftMagic+5], v210 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v215, s[sgprScaleALiftMagic+7], v215 :: v_dual_subrev_f32 v214, s[sgprScaleALiftMagic+8], v214 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+8+0], v209, v208, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+1], v210, v211, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v212, v112, v212 :: v_dual_mul_f32 v213, v206, v213 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v215, v112, v215 :: v_dual_add_nc_u32 v212, s[sgprScaleALiftMagic+5], v212 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v214, v206, v214 :: v_dual_add_nc_u32 v213, s[sgprScaleALiftMagic+5], v213 // w4a16: (n - c) * s :: round
v_add_nc_u32 v215, s[sgprScaleALiftMagic+5], v215  // w4a16: round
v_add_nc_u32 v214, s[sgprScaleALiftMagic+5], v214  // w4a16: round
v_perm_b32 v[vgprG2LA+8+2], v213, v212, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+3], v214, v215, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:18432 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 18432 sync LDS0
buffer_load_b32 v[vgprG2LA+10], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_2_0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(10)                                // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+1] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0

/* global read block-scale A */
buffer_load_d16_hi_b16 v[vgprG2LScaleA+0], v[vgprGlobalReadOffsetScaleA+0], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 0 as f32
buffer_load_d16_hi_b16 v[vgprG2LScaleA+1], v[vgprGlobalReadOffsetScaleA+1], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 1 as f32
buffer_load_d16_hi_b16 v[vgprG2LScaleA+2], v[vgprGlobalReadOffsetScaleA+2], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 2 as f32
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(12)                                // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+2:vgprG2LB+2+1] offset:4608 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 4608 sync LDS0
buffer_load_b64 v[vgprG2LB+0:vgprG2LB+0+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(12)                                // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4:vgprG2LB+4+1] offset:9216 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 9216 sync LDS0
buffer_load_b64 v[vgprG2LB+2:vgprG2LB+2+1], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(12)                                // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+6:vgprG2LB+6+1] offset:13824 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 13824 sync LDS0
buffer_load_b64 v[vgprG2LB+4:vgprG2LB+4+1], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(12)                                // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8:vgprG2LB+8+1] offset:18432 // lwoB_0_0_4_0 = (0*LSCB)*(MT1J+PAD) + (4*LSPB) = 18432 sync LDS0
buffer_load_b64 v[vgprG2LB+6:vgprG2LB+6+1], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(12)                                // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+10:vgprG2LB+10+1] offset:23040 // lwoB_0_0_5_0 = (0*LSCB)*(MT1J+PAD) + (5*LSPB) = 23040 sync LDS0
buffer_load_b64 v[vgprG2LB+8:vgprG2LB+8+1], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_4_0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(12)                                // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12:vgprG2LB+12+1] offset:27648 // lwoB_0_0_6_0 = (0*LSCB)*(MT1J+PAD) + (6*LSPB) = 27648 sync LDS0
buffer_load_b64 v[vgprG2LB+10:vgprG2LB+10+1], v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_5_0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(12)                                // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+14:vgprG2LB+14+1] offset:32256 // lwoB_0_0_7_0 = (0*LSCB)*(MT1J+PAD) + (7*LSPB) = 32256 sync LDS0
buffer_load_b64 v[vgprG2LB+12:vgprG2LB+12+1], v[vgprGlobalReadOffsetB+6], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_6_0
buffer_load_b64 v[vgprG2LB+14:vgprG2LB+14+1], v[vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_7_0

/* local write swap offsets a */

/* local write swap offsets b */

/* local read swap offsets a */

/* local read swap offsets b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+88:vgprValuC+88+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+88:vgprValuC+88+7] // left value = v[88+0:95+0]
/* numPrefetchIter=0 */
/* dataAtIterA=3 numReadsIterA=4 skipReadsIterA=0 readsPerIterA=6 */
/* dataAtIterB=3 numReadsIterB=4 skipReadsIterB=0 readsPerIterB=8 */

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
/*  grEndMfmaIndex:6, lwStartMfmaIndex:47, lwEndMfmaIndex:47  */
/*  numMfmaForLR:11, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:46 */
/*  mfmaIndex:0  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:9216 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:9232 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA+0] offset:18432 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA+0] offset:18448 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:9216 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:9232 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

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
s_waitcnt lgkmcnt(6)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:1  */
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:2  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:18432 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(3)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:3  */
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:18448 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:4  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:5  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:6  */
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:27648 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:27664 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:7  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/*  mfmaIndex:8  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
/*  mfmaIndex:9  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
/*  mfmaIndex:10  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+80:vgprValuC+80+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+80:vgprValuC+80+7] // left value = v[80+0:87+0]
/*  mfmaIndex:11  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+88:vgprValuC+88+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+88:vgprValuC+88+7] // left value = v[88+0:95+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=6 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=8 */

/* iter 1 */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:47, lwEndMfmaIndex:47  */
/*  numMfmaForLR:11, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:46 */
/*  mfmaIndex:12  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:9248 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:9264 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA+0] offset:18464 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA+0] offset:18480 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:9248 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:9264 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(6)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:13  */
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:14  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:18464 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(3)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:15  */
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:18480 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:16  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:17  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:18  */
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:27680 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:27696 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:19  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/*  mfmaIndex:20  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
/*  mfmaIndex:21  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
/*  mfmaIndex:22  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+80:vgprValuC+80+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+80:vgprValuC+80+7] // left value = v[80+0:87+0]
/*  mfmaIndex:23  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+88:vgprValuC+88+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+88:vgprValuC+88+7] // left value = v[88+0:95+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=0 readsPerIterA=6 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=0 readsPerIterB=8 */

/* iter 2 */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:47, lwEndMfmaIndex:47  */
/*  numMfmaForLR:11, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:46 */
/*  mfmaIndex:24  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:9280 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:9296 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA+0] offset:18496 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA+0] offset:18512 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:9280 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:9296 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(6)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:25  */
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:26  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:18496 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(3)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:27  */
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:18512 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:28  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:29  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:30  */
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:27712 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:27728 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:31  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/*  mfmaIndex:32  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
/*  mfmaIndex:33  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
/*  mfmaIndex:34  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+80:vgprValuC+80+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+80:vgprValuC+80+7] // left value = v[80+0:87+0]
/*  mfmaIndex:35  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+88:vgprValuC+88+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+88:vgprValuC+88+7] // left value = v[88+0:95+0]
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=0 readsPerIterA=6 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=0 readsPerIterB=8 */

/* iter 3 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:47, lwEndMfmaIndex:47  */
/*  numMfmaForLR:11, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:46 */
/*  mfmaIndex:36  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:9312 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:9328 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA+0] offset:18528 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA+0] offset:18544 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:9312 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:9328 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(6)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:37  */
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:38  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:18528 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(3)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:39  */
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:18544 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:40  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:41  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:42  */
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:27744 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:27760 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:43  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/*  mfmaIndex:44  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
/*  mfmaIndex:45  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
/*  mfmaIndex:46  */
/* schedule remaining localreads for one buffer scheduling */
/* localReadsVacancy: latencyLeft 5 */
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_wmma_f32_16x16x16_bf16 v[vgprValuC+80:vgprValuC+80+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+80:vgprValuC+80+7] // left value = v[80+0:87+0]
/*  mfmaIndex:47  */
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(10)                                // wait for global read before writing to local
v_mov_b32 v207, v[vgprG2LScaleA+0]                 // scaleA: copy on the other VGPR bank
v_xor_b32 v206, 0x88888888, v[vgprG2LA+2+0]        // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v208, v206, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v209, v206, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v211, v206, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v210, v206, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v208, s[sgprScaleALiftMagic+6], v208 :: v_dual_subrev_f32 v209, s[sgprScaleALiftMagic+7], v209 // w4a16: n - c, exact
v_dual_subrev_f32 v211, s[sgprScaleALiftMagic+8], v211 :: v_dual_subrev_f32 v210, s[sgprScaleALiftMagic+9], v210 // w4a16: n - c, exact
v_dual_mul_f32 v208, v110, v208 :: v_dual_mul_f32 v209, v207, v209 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v211, v110, v211 :: v_dual_add_nc_u32 v208, s[sgprScaleALiftMagic+5], v208 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v210, v207, v210 :: v_dual_add_nc_u32 v209, s[sgprScaleALiftMagic+5], v209 // w4a16: (n - c) * s :: round
v_and_or_b32 v212, v206, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v206, 20, v206                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v213, v206, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v215, v206, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v214, v206, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v212, s[sgprScaleALiftMagic+10], v212 :: v_dual_add_nc_u32 v211, s[sgprScaleALiftMagic+5], v211 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v213, s[sgprScaleALiftMagic+6], v213 :: v_dual_add_nc_u32 v210, s[sgprScaleALiftMagic+5], v210 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v215, s[sgprScaleALiftMagic+7], v215 :: v_dual_subrev_f32 v214, s[sgprScaleALiftMagic+8], v214 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+0+0], v209, v208, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+1], v210, v211, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v212, v110, v212 :: v_dual_mul_f32 v213, v207, v213 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v215, v110, v215 :: v_dual_add_nc_u32 v212, s[sgprScaleALiftMagic+5], v212 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v214, v207, v214 :: v_dual_add_nc_u32 v213, s[sgprScaleALiftMagic+5], v213 // w4a16: (n - c) * s :: round
v_add_nc_u32 v215, s[sgprScaleALiftMagic+5], v215  // w4a16: round
v_add_nc_u32 v214, s[sgprScaleALiftMagic+5], v214  // w4a16: round
v_perm_b32 v[vgprG2LA+0+2], v213, v212, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+3], v214, v215, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(9)                                 // wait for global read before writing to local
v_mov_b32 v206, v[vgprG2LScaleA+1]                 // scaleA: copy on the other VGPR bank
v_xor_b32 v207, 0x88888888, v[vgprG2LA+6+0]        // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v208, v207, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v209, v207, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v211, v207, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v210, v207, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v208, s[sgprScaleALiftMagic+6], v208 :: v_dual_subrev_f32 v209, s[sgprScaleALiftMagic+7], v209 // w4a16: n - c, exact
v_dual_subrev_f32 v211, s[sgprScaleALiftMagic+8], v211 :: v_dual_subrev_f32 v210, s[sgprScaleALiftMagic+9], v210 // w4a16: n - c, exact
v_dual_mul_f32 v208, v111, v208 :: v_dual_mul_f32 v209, v206, v209 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v211, v111, v211 :: v_dual_add_nc_u32 v208, s[sgprScaleALiftMagic+5], v208 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v210, v206, v210 :: v_dual_add_nc_u32 v209, s[sgprScaleALiftMagic+5], v209 // w4a16: (n - c) * s :: round
v_and_or_b32 v212, v207, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v207, 20, v207                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v213, v207, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v215, v207, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v214, v207, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v212, s[sgprScaleALiftMagic+10], v212 :: v_dual_add_nc_u32 v211, s[sgprScaleALiftMagic+5], v211 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v213, s[sgprScaleALiftMagic+6], v213 :: v_dual_add_nc_u32 v210, s[sgprScaleALiftMagic+5], v210 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v215, s[sgprScaleALiftMagic+7], v215 :: v_dual_subrev_f32 v214, s[sgprScaleALiftMagic+8], v214 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+4+0], v209, v208, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+4+1], v210, v211, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v212, v111, v212 :: v_dual_mul_f32 v213, v206, v213 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v215, v111, v215 :: v_dual_add_nc_u32 v212, s[sgprScaleALiftMagic+5], v212 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v214, v206, v214 :: v_dual_add_nc_u32 v213, s[sgprScaleALiftMagic+5], v213 // w4a16: (n - c) * s :: round
v_add_nc_u32 v215, s[sgprScaleALiftMagic+5], v215  // w4a16: round
v_add_nc_u32 v214, s[sgprScaleALiftMagic+5], v214  // w4a16: round
v_perm_b32 v[vgprG2LA+4+2], v213, v212, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+4+3], v214, v215, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:9216 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 9216 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(8)                                 // wait for global read before writing to local
v_mov_b32 v206, v[vgprG2LScaleA+2]                 // scaleA: copy on the other VGPR bank
v_xor_b32 v207, 0x88888888, v[vgprG2LA+10+0]       // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v208, v207, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v209, v207, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v211, v207, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v210, v207, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v208, s[sgprScaleALiftMagic+6], v208 :: v_dual_subrev_f32 v209, s[sgprScaleALiftMagic+7], v209 // w4a16: n - c, exact
v_dual_subrev_f32 v211, s[sgprScaleALiftMagic+8], v211 :: v_dual_subrev_f32 v210, s[sgprScaleALiftMagic+9], v210 // w4a16: n - c, exact
v_dual_mul_f32 v208, v112, v208 :: v_dual_mul_f32 v209, v206, v209 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v211, v112, v211 :: v_dual_add_nc_u32 v208, s[sgprScaleALiftMagic+5], v208 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v210, v206, v210 :: v_dual_add_nc_u32 v209, s[sgprScaleALiftMagic+5], v209 // w4a16: (n - c) * s :: round
v_and_or_b32 v212, v207, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v207, 20, v207                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v213, v207, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v215, v207, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v214, v207, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v212, s[sgprScaleALiftMagic+10], v212 :: v_dual_add_nc_u32 v211, s[sgprScaleALiftMagic+5], v211 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v213, s[sgprScaleALiftMagic+6], v213 :: v_dual_add_nc_u32 v210, s[sgprScaleALiftMagic+5], v210 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v215, s[sgprScaleALiftMagic+7], v215 :: v_dual_subrev_f32 v214, s[sgprScaleALiftMagic+8], v214 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+8+0], v209, v208, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+1], v210, v211, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v212, v112, v212 :: v_dual_mul_f32 v213, v206, v213 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v215, v112, v215 :: v_dual_add_nc_u32 v212, s[sgprScaleALiftMagic+5], v212 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v214, v206, v214 :: v_dual_add_nc_u32 v213, s[sgprScaleALiftMagic+5], v213 // w4a16: (n - c) * s :: round
v_add_nc_u32 v215, s[sgprScaleALiftMagic+5], v215  // w4a16: round
v_add_nc_u32 v214, s[sgprScaleALiftMagic+5], v214  // w4a16: round
v_perm_b32 v[vgprG2LA+8+2], v213, v212, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+3], v214, v215, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:18432 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 18432 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+1] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(6)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+2:vgprG2LB+2+1] offset:4608 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 4608 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(5)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4:vgprG2LB+4+1] offset:9216 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 9216 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(4)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+6:vgprG2LB+6+1] offset:13824 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 13824 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(3)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8:vgprG2LB+8+1] offset:18432 // lwoB_0_0_4_0 = (0*LSCB)*(MT1J+PAD) + (4*LSPB) = 18432 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(2)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+10:vgprG2LB+10+1] offset:23040 // lwoB_0_0_5_0 = (0*LSCB)*(MT1J+PAD) + (5*LSPB) = 23040 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(1)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12:vgprG2LB+12+1] offset:27648 // lwoB_0_0_6_0 = (0*LSCB)*(MT1J+PAD) + (6*LSPB) = 27648 sync LDS0
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(0)                                 // wait for global read before writing to local
ds_store_b64 v[vgprLocalWriteAddrB+0], v[vgprG2LB+14:vgprG2LB+14+1] offset:32256 // lwoB_0_0_7_0 = (0*LSCB)*(MT1J+PAD) + (7*LSPB) = 32256 sync LDS0

/* local write swap offsets a */

/* local write swap offsets b */

/* local read swap offsets a */

/* local read swap offsets b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+88:vgprValuC+88+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+88:vgprValuC+88+7] // left value = v[88+0:95+0]
/* numPrefetchIter=0 */
/* dataAtIterA=3 numReadsIterA=4 skipReadsIterA=0 readsPerIterA=6 */
/* dataAtIterB=3 numReadsIterB=4 skipReadsIterB=0 readsPerIterB=8 */
label_toPGR1:

/******************************************/
/* Ord. NoLoadLoop - Begin                */
/******************************************/
s_waitcnt lgkmcnt(0)                               // 4wait for local write
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier                                          // wait for local write done, sync

/* iter 0 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:47, lwEndMfmaIndex:47  */
/*  numMfmaForLR:11, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:46 */
/*  mfmaIndex:0  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:9216 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:9232 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA+0] offset:18432 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA+0] offset:18448 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:9216 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:9232 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(6)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:1  */
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:2  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:18432 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(3)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:3  */
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:18448 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:4  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:5  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:6  */
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:27648 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:27664 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:7  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/*  mfmaIndex:8  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
/*  mfmaIndex:9  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
/*  mfmaIndex:10  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+80:vgprValuC+80+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+80:vgprValuC+80+7] // left value = v[80+0:87+0]
/*  mfmaIndex:11  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+88:vgprValuC+88+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+88:vgprValuC+88+7] // left value = v[88+0:95+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=6 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=8 */

/* iter 1 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:47, lwEndMfmaIndex:47  */
/*  numMfmaForLR:11, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:46 */
/*  mfmaIndex:12  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:9248 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:9264 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA+0] offset:18464 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA+0] offset:18480 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:9248 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:9264 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(6)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:13  */
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:14  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:18464 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(3)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:15  */
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:18480 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:16  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:17  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:18  */
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:27680 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:27696 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:19  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/*  mfmaIndex:20  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
/*  mfmaIndex:21  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
/*  mfmaIndex:22  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+80:vgprValuC+80+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+80:vgprValuC+80+7] // left value = v[80+0:87+0]
/*  mfmaIndex:23  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+88:vgprValuC+88+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+88:vgprValuC+88+7] // left value = v[88+0:95+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=0 readsPerIterA=6 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=0 readsPerIterB=8 */

/* iter 2 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:47, lwEndMfmaIndex:47  */
/*  numMfmaForLR:11, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:46 */
/*  mfmaIndex:24  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:9280 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:9296 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA+0] offset:18496 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA+0] offset:18512 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:9280 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:9296 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(6)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:25  */
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:26  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:18496 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(3)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:27  */
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:18512 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:28  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:29  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:30  */
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:27712 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:27728 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:31  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/*  mfmaIndex:32  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
/*  mfmaIndex:33  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
/*  mfmaIndex:34  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+80:vgprValuC+80+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+80:vgprValuC+80+7] // left value = v[80+0:87+0]
/*  mfmaIndex:35  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+88:vgprValuC+88+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+88:vgprValuC+88+7] // left value = v[88+0:95+0]
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=0 readsPerIterA=6 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=0 readsPerIterB=8 */

/* iter 3 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:47, lwEndMfmaIndex:47  */
/*  numMfmaForLR:11, syncPlrMfmaIndex:0 , sync1LdsMfmaIndex:46 */
/*  mfmaIndex:36  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:9312 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:9328 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA+0] offset:18528 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA+0] offset:18544 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:9312 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:9328 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(6)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:37  */
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:38  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:18528 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(3)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:39  */
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:18544 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:40  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:41  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/*  mfmaIndex:42  */
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:27744 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:27760 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
/*  mfmaIndex:43  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
/*  mfmaIndex:44  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
/*  mfmaIndex:45  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
/*  mfmaIndex:46  */
/* schedule remaining localreads for one buffer scheduling */
/* localReadsVacancy: latencyLeft 5 */
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_wmma_f32_16x16x16_bf16 v[vgprValuC+80:vgprValuC+80+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+80:vgprValuC+80+7] // left value = v[80+0:87+0]
/*  mfmaIndex:47  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+88:vgprValuC+88+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+88:vgprValuC+88+7] // left value = v[88+0:95+0]
/* numPrefetchIter=0 */
/* dataAtIterA=3 numReadsIterA=4 skipReadsIterA=0 readsPerIterA=6 */
/* dataAtIterB=3 numReadsIterB=4 skipReadsIterB=0 readsPerIterB=8 */
label_toPGR1end_OrdNLL:
label_PrefetchGlobalLastIterEnd:

/* Tail: add ValuA/B vgpr buffer [118...175) to pool */

/* Tail: add address/G2L vgpr [175...204) to pool */
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

/* Mapping of Acc register -> C Vgpr register */

/* Multiply MI out register with Alpha -> C Vgpr register */

/* not-LocalSplitU: global write indices */
/* computeStoreVgprs */
v_lshrrev_b32 v122, 5, v[vgprSerial]               // 122 = Serial / 32
v_lshrrev_b32 v123, 2, v122                        // 123 = 122 / 4
v_mul_lo_u32 v123, 0x10, v123                      // wave coordination offset 1
v_and_b32 v119, 31, v[vgprSerial]                  // v119 = v[vgprSerial] % 32
v_lshrrev_b32 v119, 4, v119                        // 119 = 119 / 16
                                                   // thread0 * continuous_output (multiplier is 1, do nothing)
v_add_lshl_u32 v119, v123, v119, 0                 // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v120, v119, s[sgprStrideC1J]          //  offset 1
v_mul_lo_u32 v121, v119, s[sgprStrideD1J]          //  offset 1
v_and_b32 v118, 3, v122                            // v118 = v122 % 4
v_mul_lo_u32 v118, 0x10, v118                      // wave coordination offset 0
v_and_b32 v123, 15, v[vgprSerial]                  // v123 = v[vgprSerial] % 16
v_add_lshl_u32 v118, v123, v118, 0                 // coordination 0 = vwA * (wave_id0 + tid0)
s_mul_i32 s7, 192, s[sgprWorkGroup0]               // wgp0 * MT0
v_add_nc_u32 v118, s7, v118                        // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s7, 256, s[sgprWorkGroup1]               // wgp1 * MT1
v_add_nc_u32 v119, s7, v119                        // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1

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

/* Edge/NonEdge store path check (M): Size % 192 > 0 -> Edge store; else -> NonEdge store */
s_mul_hi_u32 s67, s[sgprSizeI], 44739243           // STATIC_DIV: divisor=192
s_mul_i32 s66, s[sgprSizeI], 44739243              // tmp = dividend * magic
s_lshr_b64 s[66:67], s[66:67], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s65, s66                                 // quotient
s_mul_i32 s66, s65, 192                            // quotient*divisor
s_sub_u32 s64, s[sgprSizeI], s66                   // rReg = dividend - quotient*divisor
s_add_u32 s65, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s65                // wg0 >= nwg0-1 ?
s_cselect_b32 s64, s64, 0                          // set rem
s_cmpk_gt_u32 s64, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_MB_Else         // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 256 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s64, 255, s[sgprSizeJ]                   // s64 = s[sgprSizeJ] % 256
s_add_u32 s65, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s65                // wg1 >= nwg1-1
s_cselect_b32 s64, s64, 0                          // set rem
s_cmpk_gt_u32 s64, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_MB_Then         // jump if edges required
label_GW_B0_FD0_VW1_MB_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=122 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */
s_sleep 7                                          // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (8,2,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (9,2,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (10,2,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (11,2,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (12,2,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (13,2,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (14,2,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (15,2,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (16,2,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (17,2,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (18,2,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (19,2,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (20,2,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (21,2,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (22,2,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1); (23,2,0,0:vw1); (24,0,0,0:vw1); (24,1,0,0:vw1); (24,2,0,0:vw1); (25,0,0,0:vw1); (25,1,0,0:vw1); (25,2,0,0:vw1); (26,0,0,0:vw1); (26,1,0,0:vw1); (26,2,0,0:vw1); (27,0,0,0:vw1); (27,1,0,0:vw1); (27,2,0,0:vw1); (28,0,0,0:vw1); (28,1,0,0:vw1); (28,2,0,0:vw1); (29,0,0,0:vw1); (29,1,0,0:vw1); (29,2,0,0:vw1); (30,0,0,0:vw1); (30,1,0,0:vw1); (30,2,0,0:vw1); (31,0,0,0:vw1); (31,1,0,0:vw1); (31,2,0,0:vw1) */
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
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
/* (d1,vc1,d0,vc0)=(24,0,1,0) */
/* (d1,vc1,d0,vc0)=(24,0,2,0) */
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
/* (d1,vc1,d0,vc0)=(25,0,1,0) */
/* (d1,vc1,d0,vc0)=(25,0,2,0) */
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
/* (d1,vc1,d0,vc0)=(26,0,1,0) */
/* (d1,vc1,d0,vc0)=(26,0,2,0) */
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
/* (d1,vc1,d0,vc0)=(27,0,1,0) */
/* (d1,vc1,d0,vc0)=(27,0,2,0) */
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
/* (d1,vc1,d0,vc0)=(28,0,1,0) */
/* (d1,vc1,d0,vc0)=(28,0,2,0) */
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
/* (d1,vc1,d0,vc0)=(29,0,1,0) */
/* (d1,vc1,d0,vc0)=(29,0,2,0) */
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
/* (d1,vc1,d0,vc0)=(30,0,1,0) */
/* (d1,vc1,d0,vc0)=(30,0,2,0) */
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
/* (d1,vc1,d0,vc0)=(31,0,1,0) */
/* (d1,vc1,d0,vc0)=(31,0,2,0) */
v_add_lshl_u32 v129, v121, v118, 2                 // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=118, coord0Vgpr=118 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (5, 2, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (7, 2, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (8, 2, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (9, 2, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (10, 2, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (11, 2, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (12, 2, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (13, 2, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (14, 2, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (15, 2, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (16, 2, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (17, 2, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (18, 2, 0, 0), (19, 0, 0, 0), (19, 1, 0, 0), (19, 2, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0), (20, 2, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0), (21, 2, 0, 0), (22, 0, 0, 0), (22, 1, 0, 0), (22, 2, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0), (23, 2, 0, 0), (24, 0, 0, 0), (24, 1, 0, 0), (24, 2, 0, 0), (25, 0, 0, 0), (25, 1, 0, 0), (25, 2, 0, 0), (26, 0, 0, 0), (26, 1, 0, 0), (26, 2, 0, 0), (27, 0, 0, 0), (27, 1, 0, 0), (27, 2, 0, 0), (28, 0, 0, 0), (28, 1, 0, 0), (28, 2, 0, 0), (29, 0, 0, 0), (29, 1, 0, 0), (29, 2, 0, 0), (30, 0, 0, 0), (30, 1, 0, 0), (30, 2, 0, 0), (31, 0, 0, 0), (31, 1, 0, 0), (31, 2, 0, 0)] */
v_mov_b32 v[vgprValuC+131], v[vgprValuC+0]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+132], v[vgprValuC+8]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+133], v[vgprValuC+16]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+134], v[vgprValuC+1]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+135], v[vgprValuC+9]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+136], v[vgprValuC+17]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+137], v[vgprValuC+2]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+138], v[vgprValuC+10]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+139], v[vgprValuC+18]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+140], v[vgprValuC+3]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+141], v[vgprValuC+11]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+142], v[vgprValuC+19]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+143], v[vgprValuC+4]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+144], v[vgprValuC+12]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+145], v[vgprValuC+20]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+146], v[vgprValuC+5]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+147], v[vgprValuC+13]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+148], v[vgprValuC+21]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+149], v[vgprValuC+6]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+150], v[vgprValuC+14]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+151], v[vgprValuC+22]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+152], v[vgprValuC+7]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+153], v[vgprValuC+15]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+154], v[vgprValuC+23]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+155], v[vgprValuC+24]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+156], v[vgprValuC+32]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+157], v[vgprValuC+40]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+158], v[vgprValuC+25]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+159], v[vgprValuC+33]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+160], v[vgprValuC+41]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+161], v[vgprValuC+26]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+162], v[vgprValuC+34]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+163], v[vgprValuC+42]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+164], v[vgprValuC+27]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+165], v[vgprValuC+35]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+166], v[vgprValuC+43]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+167], v[vgprValuC+28]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+168], v[vgprValuC+36]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+169], v[vgprValuC+44]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+170], v[vgprValuC+29]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+171], v[vgprValuC+37]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+172], v[vgprValuC+45]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+173], v[vgprValuC+30]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+174], v[vgprValuC+38]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+175], v[vgprValuC+46]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+176], v[vgprValuC+31]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+177], v[vgprValuC+39]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+178], v[vgprValuC+47]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+179], v[vgprValuC+48]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+180], v[vgprValuC+56]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+181], v[vgprValuC+64]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+182], v[vgprValuC+49]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+183], v[vgprValuC+57]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+184], v[vgprValuC+65]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+185], v[vgprValuC+50]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+186], v[vgprValuC+58]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+187], v[vgprValuC+66]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+188], v[vgprValuC+51]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+189], v[vgprValuC+59]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+190], v[vgprValuC+67]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+191], v[vgprValuC+52]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+192], v[vgprValuC+60]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+193], v[vgprValuC+68]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+194], v[vgprValuC+53]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+195], v[vgprValuC+61]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+196], v[vgprValuC+69]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+197], v[vgprValuC+54]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+198], v[vgprValuC+62]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+199], v[vgprValuC+70]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+200], v[vgprValuC+55]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+201], v[vgprValuC+63]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+202], v[vgprValuC+71]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+203], v[vgprValuC+72]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+206], v[vgprValuC+80]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+207], v[vgprValuC+88]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+208], v[vgprValuC+73]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+209], v[vgprValuC+81]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+210], v[vgprValuC+89]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+211], v[vgprValuC+74]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+212], v[vgprValuC+82]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+213], v[vgprValuC+90]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+214], v[vgprValuC+75]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+215], v[vgprValuC+83]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+216], v[vgprValuC+91]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+217], v[vgprValuC+76]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+218], v[vgprValuC+84]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+219], v[vgprValuC+92]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+220], v[vgprValuC+77]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+221], v[vgprValuC+85]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+222], v[vgprValuC+93]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+223], v[vgprValuC+78]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+224], v[vgprValuC+86]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+225], v[vgprValuC+94]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+226], v[vgprValuC+79]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+227], v[vgprValuC+87]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+228], v[vgprValuC+95]        // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b32 v131, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v132, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v133, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v134, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v135, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v136, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v137, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v138, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v139, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v140, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v141, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v142, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v143, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v144, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v145, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v146, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v147, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v148, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v149, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v150, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v151, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v152, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v153, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v154, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 200                // scale StrideD *= numRows(50) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(50): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(50): gra SRD += inc(upper)
buffer_store_b32 v155, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v156, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v157, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v158, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v159, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v160, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v161, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v162, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v163, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v164, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v165, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v166, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v167, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v168, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v169, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v170, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v171, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v172, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v173, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v174, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v175, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v176, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v177, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v178, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 200                // scale StrideD *= numRows(50) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(50): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(50): gra SRD += inc(upper)
buffer_store_b32 v179, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v180, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v181, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v182, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v183, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v184, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v185, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v186, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v187, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v188, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v189, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v190, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v191, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v192, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v193, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v194, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v195, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v196, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v197, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v198, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v199, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v200, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v201, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v202, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 200                // scale StrideD *= numRows(50) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(50): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(50): gra SRD += inc(upper)
buffer_store_b32 v203, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v206, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v207, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v208, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v209, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v210, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v211, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v212, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v213, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v214, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v215, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v216, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v217, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v218, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v219, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v220, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v221, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v222, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v223, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v224, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v225, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v226, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v227, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v228, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_B0_FD0_VW1_MB_NonEdgeEnd:
label_GW_B0_FD0_VW1_MB_Else:
label_GW_B0_FD0_VW1_MB_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=62 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */
s_sleep 7                                          // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (8,2,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (9,2,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (10,2,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (11,2,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (12,2,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (13,2,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (14,2,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (15,2,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (16,2,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (17,2,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (18,2,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (19,2,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v124, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v191, v121, v118, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v191, v124, v191, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v192, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v192, v124, v192, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v193, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v193, v124, v193, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v194, v121, v118, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v194, v124, v194, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v195, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v195, v124, v195, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v196, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v196, v124, v196, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v197, v121, v118, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v197, v124, v197, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v198, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v198, v124, v198, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v199, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v199, v124, v199, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v200, v121, v118, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v200, v124, v200, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v201, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v201, v124, v201, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v202, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v202, v124, v202, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v203, v121, v118, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v203, v124, v203, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v206, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v206, v124, v206, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v207, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v207, v124, v207, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v208, v121, v118, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v208, v124, v208, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v209, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v209, v124, v209, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v210, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v210, v124, v210, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v211, v121, v118, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v211, v124, v211, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v212, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v212, v124, v212, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v213, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v213, v124, v213, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v214, v121, v118, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v214, v124, v214, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v215, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v215, v124, v215, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v216, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v216, v124, v216, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 50                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 50                // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 50                // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v217, v121, v118, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v217, v124, v217, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v218, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v218, v124, v218, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v219, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v219, v124, v219, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v220, v121, v118, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v220, v124, v220, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v221, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v221, v124, v221, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v222, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v222, v124, v222, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v223, v121, v118, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v223, v124, v223, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v224, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v224, v124, v224, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v225, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v225, v124, v225, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v226, v121, v118, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v226, v124, v226, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v227, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v227, v124, v227, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v228, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v228, v124, v228, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v229, v121, v118, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v229, v124, v229, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v230, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v230, v124, v230, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v231, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v231, v124, v231, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v232, v121, v118, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v232, v124, v232, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v233, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v233, v124, v233, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v234, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v234, v124, v234, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v235, v121, v118, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v235, v124, v235, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v236, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v236, v124, v236, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v237, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v237, v124, v237, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v238, v121, v118, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v238, v124, v238, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v239, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v239, v124, v239, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v240, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v240, v124, v240, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 50                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 50                // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 50                // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v241, v121, v118, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v241, v124, v241, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v242, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v242, v124, v242, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v243, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v243, v124, v243, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v244, v121, v118, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v244, v124, v244, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v245, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v245, v124, v245, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v246, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v246, v124, v246, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v247, v121, v118, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v247, v124, v247, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v248, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v248, v124, v248, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v249, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v249, v124, v249, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v250, v121, v118, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v250, v124, v250, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v251, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v251, v124, v251, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v252, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v252, v124, v252, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v253, v121, v118, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v253, v124, v253, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v254, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v254, v124, v254, s66                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (5, 2, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (7, 2, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (8, 2, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (9, 2, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (10, 2, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (11, 2, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (12, 2, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (13, 2, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (14, 2, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (15, 2, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (16, 2, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (17, 2, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (18, 2, 0, 0), (19, 0, 0, 0), (19, 1, 0, 0), (19, 2, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0)] */
v_mov_b32 v[vgprValuC+129], v[vgprValuC+0]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+130], v[vgprValuC+8]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+131], v[vgprValuC+16]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+132], v[vgprValuC+1]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+133], v[vgprValuC+9]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+134], v[vgprValuC+17]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+135], v[vgprValuC+2]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+136], v[vgprValuC+10]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+137], v[vgprValuC+18]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+138], v[vgprValuC+3]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+139], v[vgprValuC+11]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+140], v[vgprValuC+19]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+141], v[vgprValuC+4]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+142], v[vgprValuC+12]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+143], v[vgprValuC+20]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+144], v[vgprValuC+5]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+145], v[vgprValuC+13]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+146], v[vgprValuC+21]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+147], v[vgprValuC+6]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+148], v[vgprValuC+14]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+149], v[vgprValuC+22]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+150], v[vgprValuC+7]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+151], v[vgprValuC+15]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+152], v[vgprValuC+23]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+153], v[vgprValuC+24]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+154], v[vgprValuC+32]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+155], v[vgprValuC+40]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+156], v[vgprValuC+25]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+157], v[vgprValuC+33]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+158], v[vgprValuC+41]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+159], v[vgprValuC+26]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+160], v[vgprValuC+34]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+161], v[vgprValuC+42]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+162], v[vgprValuC+27]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+163], v[vgprValuC+35]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+164], v[vgprValuC+43]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+165], v[vgprValuC+28]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+166], v[vgprValuC+36]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+167], v[vgprValuC+44]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+168], v[vgprValuC+29]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+169], v[vgprValuC+37]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+170], v[vgprValuC+45]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+171], v[vgprValuC+30]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+172], v[vgprValuC+38]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+173], v[vgprValuC+46]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+174], v[vgprValuC+31]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+175], v[vgprValuC+39]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+176], v[vgprValuC+47]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+177], v[vgprValuC+48]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+178], v[vgprValuC+56]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+179], v[vgprValuC+64]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+180], v[vgprValuC+49]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+181], v[vgprValuC+57]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+182], v[vgprValuC+65]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+183], v[vgprValuC+50]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+184], v[vgprValuC+58]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+185], v[vgprValuC+66]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+186], v[vgprValuC+51]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+187], v[vgprValuC+59]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+188], v[vgprValuC+67]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+189], v[vgprValuC+52]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+190], v[vgprValuC+60]        // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b32 v129, v191, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v130, v192, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v131, v193, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v132, v194, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v133, v195, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v134, v196, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v135, v197, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v136, v198, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v137, v199, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v138, v200, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v139, v201, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v140, v202, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v141, v203, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v142, v206, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v143, v207, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v144, v208, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v145, v209, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v146, v210, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v147, v211, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v148, v212, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v149, v213, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v150, v214, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v151, v215, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v152, v216, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v153, v217, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v154, v218, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v155, v219, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v156, v220, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v157, v221, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v158, v222, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v159, v223, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v160, v224, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v161, v225, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v162, v226, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v163, v227, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v164, v228, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v165, v229, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v166, v230, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v167, v231, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v168, v232, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v169, v233, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v170, v234, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v171, v235, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v172, v236, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v173, v237, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v174, v238, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v175, v239, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v176, v240, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v177, v241, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v178, v242, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v179, v243, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v180, v244, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v181, v245, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v182, v246, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v183, v247, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v184, v248, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v185, v249, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v186, v250, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v187, v251, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v188, v252, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v189, v253, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v190, v254, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */
s_sleep 7                                          // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (20,2,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (21,2,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (22,2,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1); (23,2,0,0:vw1); (24,0,0,0:vw1); (24,1,0,0:vw1); (24,2,0,0:vw1); (25,0,0,0:vw1); (25,1,0,0:vw1); (25,2,0,0:vw1); (26,0,0,0:vw1); (26,1,0,0:vw1); (26,2,0,0:vw1); (27,0,0,0:vw1); (27,1,0,0:vw1); (27,2,0,0:vw1); (28,0,0,0:vw1); (28,1,0,0:vw1); (28,2,0,0:vw1); (29,0,0,0:vw1); (29,1,0,0:vw1); (29,2,0,0:vw1); (30,0,0,0:vw1); (30,1,0,0:vw1); (30,2,0,0:vw1); (31,0,0,0:vw1); (31,1,0,0:vw1); (31,2,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v124, BufferOOB
/* (d1,vc1,d0,vc0)=(20,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v163, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v163, v124, v163, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v164, v121, v118, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v164, v124, v164, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v165, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v165, v124, v165, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v166, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v166, v124, v166, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v167, v121, v118, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v167, v124, v167, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v168, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v168, v124, v168, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v169, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v169, v124, v169, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v170, v121, v118, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v170, v124, v170, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v171, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v171, v124, v171, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v172, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v172, v124, v172, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 50                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 50                // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 50                // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v173, v121, v118, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v173, v124, v173, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v174, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v174, v124, v174, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v175, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v175, v124, v175, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v176, v121, v118, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v176, v124, v176, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v177, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v177, v124, v177, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v178, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v178, v124, v178, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v179, v121, v118, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v179, v124, v179, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v180, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v180, v124, v180, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v181, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v181, v124, v181, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v182, v121, v118, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v182, v124, v182, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v183, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v183, v124, v183, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v184, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v184, v124, v184, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v185, v121, v118, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v185, v124, v185, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v186, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v186, v124, v186, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v187, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v187, v124, v187, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v188, v121, v118, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v188, v124, v188, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v189, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v189, v124, v189, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v190, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v190, v124, v190, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v191, v121, v118, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v191, v124, v191, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v192, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v192, v124, v192, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v193, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v193, v124, v193, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v194, v121, v118, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v194, v124, v194, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v195, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v195, v124, v195, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v196, v121, v122, 2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v196, v124, v196, s66                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(20, 2, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0), (21, 2, 0, 0), (22, 0, 0, 0), (22, 1, 0, 0), (22, 2, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0), (23, 2, 0, 0), (24, 0, 0, 0), (24, 1, 0, 0), (24, 2, 0, 0), (25, 0, 0, 0), (25, 1, 0, 0), (25, 2, 0, 0), (26, 0, 0, 0), (26, 1, 0, 0), (26, 2, 0, 0), (27, 0, 0, 0), (27, 1, 0, 0), (27, 2, 0, 0), (28, 0, 0, 0), (28, 1, 0, 0), (28, 2, 0, 0), (29, 0, 0, 0), (29, 1, 0, 0), (29, 2, 0, 0), (30, 0, 0, 0), (30, 1, 0, 0), (30, 2, 0, 0), (31, 0, 0, 0), (31, 1, 0, 0), (31, 2, 0, 0)] */
v_mov_b32 v[vgprValuC+129], v[vgprValuC+68]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+130], v[vgprValuC+53]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+131], v[vgprValuC+61]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+132], v[vgprValuC+69]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+133], v[vgprValuC+54]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+134], v[vgprValuC+62]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+135], v[vgprValuC+70]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+136], v[vgprValuC+55]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+137], v[vgprValuC+63]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+138], v[vgprValuC+71]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+139], v[vgprValuC+72]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+140], v[vgprValuC+80]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+141], v[vgprValuC+88]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+142], v[vgprValuC+73]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+143], v[vgprValuC+81]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+144], v[vgprValuC+89]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+145], v[vgprValuC+74]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+146], v[vgprValuC+82]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+147], v[vgprValuC+90]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+148], v[vgprValuC+75]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+149], v[vgprValuC+83]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+150], v[vgprValuC+91]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+151], v[vgprValuC+76]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+152], v[vgprValuC+84]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+153], v[vgprValuC+92]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+154], v[vgprValuC+77]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+155], v[vgprValuC+85]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+156], v[vgprValuC+93]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+157], v[vgprValuC+78]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+158], v[vgprValuC+86]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+159], v[vgprValuC+94]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+160], v[vgprValuC+79]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+161], v[vgprValuC+87]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+162], v[vgprValuC+95]        // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b32 v129, v163, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v130, v164, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v131, v165, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v132, v166, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v133, v167, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v134, v168, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v135, v169, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v136, v170, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v137, v171, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v138, v172, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v139, v173, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v140, v174, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v141, v175, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v142, v176, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v143, v177, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v144, v178, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v145, v179, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v146, v180, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v147, v181, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v148, v182, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v149, v183, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v150, v184, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v151, v185, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v152, v186, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v153, v187, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v154, v188, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v155, v189, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v156, v190, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v157, v191, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v158, v192, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v159, v193, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v160, v194, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v161, v195, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v162, v196, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
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
v_mov_b32 v129, s66                                // move flag value to vgpr
v_mov_b32 v130, 0                                  // zero vaddr offset
s_mov_b64 s[72:73], s[sgprAddressFlags:sgprAddressFlags+1]
s_mov_b32 s74, BufferOOB
s_mov_b32 s75, Srd127_96
buffer_store_b32 v129, v130, s[72:75], s64 offen offset:0 glc dlc // reset flag
s_waitcnt_vscnt null 0                             // wait for data store
label_SK_SkipFlagReset:
label_Fixup_E0:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=60 */
s_mov_b64 s[sgprSrdWS:sgprSrdWS+1], s[sgprAddressWS:sgprAddressWS+1]
s_mov_b32 s[sgprSrdWS+2], BufferOOB
s_mov_b32 s[sgprSrdWS+3], Srd127_96

s_mul_i32 s64, 0x30000, s7                         // Offset to correct partials tile (low word)
s_mul_hi_u32 s11, 0x30000, s7                      // partials tile offset (high word) for 64-bit SRD
s_add_u32 s[sgprSrdWS+0], s[sgprSrdWS+0], s64      // add lo to SRD
s_addc_u32 s[sgprSrdWS+1], s[sgprSrdWS+1], s11     // add hi (offset high word + lo carry) to SRD
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 7                                          // optimization: sync and wait
s_barrier

/******************************************/
/* Fixup Batch #0 (d1,d0,vc1,vc0) =       */
/*      (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (8,2,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (9,2,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (10,2,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (11,2,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (12,2,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (13,2,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (14,2,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (15,2,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (16,2,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (17,2,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (18,2,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (19,2,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_lshlrev_b32 v130, 2, v[vgprSerial]               // v130 = v[vgprSerial] * 4
s_mov_b32 s64, 0                                   // Init sgpr offset
buffer_load_b32 v191, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v192, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v193, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v194, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v195, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v196, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v197, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v198, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v199, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v200, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v201, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v202, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v203, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v206, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v207, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v208, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v209, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v210, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v211, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v212, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v213, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v214, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v215, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v216, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v217, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v218, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v219, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v220, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v221, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v222, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v223, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v224, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v225, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v226, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v227, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v228, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v229, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v230, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v231, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v232, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v233, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v234, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v235, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v236, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v237, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v238, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v239, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v240, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v241, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v242, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v243, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v244, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v245, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v246, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v247, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v248, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v249, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v250, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v251, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v252, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
v_mov_b32 v[vgprValuC+131], v[vgprValuC+0]         // copy MI out reg to vreg[0]
v_mov_b32 v[vgprValuC+132], v[vgprValuC+8]         // copy MI out reg to vreg[1]
v_mov_b32 v[vgprValuC+133], v[vgprValuC+16]        // copy MI out reg to vreg[2]
v_mov_b32 v[vgprValuC+134], v[vgprValuC+1]         // copy MI out reg to vreg[3]
v_mov_b32 v[vgprValuC+135], v[vgprValuC+9]         // copy MI out reg to vreg[4]
v_mov_b32 v[vgprValuC+136], v[vgprValuC+17]        // copy MI out reg to vreg[5]
v_mov_b32 v[vgprValuC+137], v[vgprValuC+2]         // copy MI out reg to vreg[6]
v_mov_b32 v[vgprValuC+138], v[vgprValuC+10]        // copy MI out reg to vreg[7]
v_mov_b32 v[vgprValuC+139], v[vgprValuC+18]        // copy MI out reg to vreg[8]
v_mov_b32 v[vgprValuC+140], v[vgprValuC+3]         // copy MI out reg to vreg[9]
v_mov_b32 v[vgprValuC+141], v[vgprValuC+11]        // copy MI out reg to vreg[10]
v_mov_b32 v[vgprValuC+142], v[vgprValuC+19]        // copy MI out reg to vreg[11]
v_mov_b32 v[vgprValuC+143], v[vgprValuC+4]         // copy MI out reg to vreg[12]
v_mov_b32 v[vgprValuC+144], v[vgprValuC+12]        // copy MI out reg to vreg[13]
v_mov_b32 v[vgprValuC+145], v[vgprValuC+20]        // copy MI out reg to vreg[14]
v_mov_b32 v[vgprValuC+146], v[vgprValuC+5]         // copy MI out reg to vreg[15]
v_mov_b32 v[vgprValuC+147], v[vgprValuC+13]        // copy MI out reg to vreg[16]
v_mov_b32 v[vgprValuC+148], v[vgprValuC+21]        // copy MI out reg to vreg[17]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+6]         // copy MI out reg to vreg[18]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+14]        // copy MI out reg to vreg[19]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+22]        // copy MI out reg to vreg[20]
v_mov_b32 v[vgprValuC+152], v[vgprValuC+7]         // copy MI out reg to vreg[21]
v_mov_b32 v[vgprValuC+153], v[vgprValuC+15]        // copy MI out reg to vreg[22]
v_mov_b32 v[vgprValuC+154], v[vgprValuC+23]        // copy MI out reg to vreg[23]
v_mov_b32 v[vgprValuC+155], v[vgprValuC+24]        // copy MI out reg to vreg[24]
v_mov_b32 v[vgprValuC+156], v[vgprValuC+32]        // copy MI out reg to vreg[25]
v_mov_b32 v[vgprValuC+157], v[vgprValuC+40]        // copy MI out reg to vreg[26]
v_mov_b32 v[vgprValuC+158], v[vgprValuC+25]        // copy MI out reg to vreg[27]
v_mov_b32 v[vgprValuC+159], v[vgprValuC+33]        // copy MI out reg to vreg[28]
v_mov_b32 v[vgprValuC+160], v[vgprValuC+41]        // copy MI out reg to vreg[29]
v_mov_b32 v[vgprValuC+161], v[vgprValuC+26]        // copy MI out reg to vreg[30]
v_mov_b32 v[vgprValuC+162], v[vgprValuC+34]        // copy MI out reg to vreg[31]
v_mov_b32 v[vgprValuC+163], v[vgprValuC+42]        // copy MI out reg to vreg[32]
v_mov_b32 v[vgprValuC+164], v[vgprValuC+27]        // copy MI out reg to vreg[33]
v_mov_b32 v[vgprValuC+165], v[vgprValuC+35]        // copy MI out reg to vreg[34]
v_mov_b32 v[vgprValuC+166], v[vgprValuC+43]        // copy MI out reg to vreg[35]
v_mov_b32 v[vgprValuC+167], v[vgprValuC+28]        // copy MI out reg to vreg[36]
v_mov_b32 v[vgprValuC+168], v[vgprValuC+36]        // copy MI out reg to vreg[37]
v_mov_b32 v[vgprValuC+169], v[vgprValuC+44]        // copy MI out reg to vreg[38]
v_mov_b32 v[vgprValuC+170], v[vgprValuC+29]        // copy MI out reg to vreg[39]
v_mov_b32 v[vgprValuC+171], v[vgprValuC+37]        // copy MI out reg to vreg[40]
v_mov_b32 v[vgprValuC+172], v[vgprValuC+45]        // copy MI out reg to vreg[41]
v_mov_b32 v[vgprValuC+173], v[vgprValuC+30]        // copy MI out reg to vreg[42]
v_mov_b32 v[vgprValuC+174], v[vgprValuC+38]        // copy MI out reg to vreg[43]
v_mov_b32 v[vgprValuC+175], v[vgprValuC+46]        // copy MI out reg to vreg[44]
v_mov_b32 v[vgprValuC+176], v[vgprValuC+31]        // copy MI out reg to vreg[45]
v_mov_b32 v[vgprValuC+177], v[vgprValuC+39]        // copy MI out reg to vreg[46]
v_mov_b32 v[vgprValuC+178], v[vgprValuC+47]        // copy MI out reg to vreg[47]
v_mov_b32 v[vgprValuC+179], v[vgprValuC+48]        // copy MI out reg to vreg[48]
v_mov_b32 v[vgprValuC+180], v[vgprValuC+56]        // copy MI out reg to vreg[49]
v_mov_b32 v[vgprValuC+181], v[vgprValuC+64]        // copy MI out reg to vreg[50]
v_mov_b32 v[vgprValuC+182], v[vgprValuC+49]        // copy MI out reg to vreg[51]
v_mov_b32 v[vgprValuC+183], v[vgprValuC+57]        // copy MI out reg to vreg[52]
v_mov_b32 v[vgprValuC+184], v[vgprValuC+65]        // copy MI out reg to vreg[53]
v_mov_b32 v[vgprValuC+185], v[vgprValuC+50]        // copy MI out reg to vreg[54]
v_mov_b32 v[vgprValuC+186], v[vgprValuC+58]        // copy MI out reg to vreg[55]
v_mov_b32 v[vgprValuC+187], v[vgprValuC+66]        // copy MI out reg to vreg[56]
v_mov_b32 v[vgprValuC+188], v[vgprValuC+51]        // copy MI out reg to vreg[57]
v_mov_b32 v[vgprValuC+189], v[vgprValuC+59]        // copy MI out reg to vreg[58]
v_mov_b32 v[vgprValuC+190], v[vgprValuC+67]        // copy MI out reg to vreg[59]

/* apply mask, calc new C and issue writes */
v_mov_b32 v126, 0xffff0000                         // mask for pack two bfloat16 element to 32bit
v_mov_b32 v127, 0x7fff0000                         // fp32 Nan
v_mov_b32 v128, 0x7fff                             // rounding bias for bfloat16

s_waitcnt vmcnt(59)                                // wait C (interleaved) 59 = 60 - 0 - 1
v_add_f32 v[vgprValuC+131], v[vgprValuC+131], v191 // accum partials

s_waitcnt vmcnt(58)                                // wait C (interleaved) 58 = 60 - 1 - 1
v_add_f32 v[vgprValuC+132], v[vgprValuC+132], v192 // accum partials

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 60 - 2 - 1
v_add_f32 v[vgprValuC+133], v[vgprValuC+133], v193 // accum partials

s_waitcnt vmcnt(56)                                // wait C (interleaved) 56 = 60 - 3 - 1
v_add_f32 v[vgprValuC+134], v[vgprValuC+134], v194 // accum partials

s_waitcnt vmcnt(55)                                // wait C (interleaved) 55 = 60 - 4 - 1
v_add_f32 v[vgprValuC+135], v[vgprValuC+135], v195 // accum partials

s_waitcnt vmcnt(54)                                // wait C (interleaved) 54 = 60 - 5 - 1
v_add_f32 v[vgprValuC+136], v[vgprValuC+136], v196 // accum partials

s_waitcnt vmcnt(53)                                // wait C (interleaved) 53 = 60 - 6 - 1
v_add_f32 v[vgprValuC+137], v[vgprValuC+137], v197 // accum partials

s_waitcnt vmcnt(52)                                // wait C (interleaved) 52 = 60 - 7 - 1
v_add_f32 v[vgprValuC+138], v[vgprValuC+138], v198 // accum partials

s_waitcnt vmcnt(51)                                // wait C (interleaved) 51 = 60 - 8 - 1
v_add_f32 v[vgprValuC+139], v[vgprValuC+139], v199 // accum partials

s_waitcnt vmcnt(50)                                // wait C (interleaved) 50 = 60 - 9 - 1
v_add_f32 v[vgprValuC+140], v[vgprValuC+140], v200 // accum partials

s_waitcnt vmcnt(49)                                // wait C (interleaved) 49 = 60 - 10 - 1
v_add_f32 v[vgprValuC+141], v[vgprValuC+141], v201 // accum partials

s_waitcnt vmcnt(48)                                // wait C (interleaved) 48 = 60 - 11 - 1
v_add_f32 v[vgprValuC+142], v[vgprValuC+142], v202 // accum partials

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 60 - 12 - 1
v_add_f32 v[vgprValuC+143], v[vgprValuC+143], v203 // accum partials

s_waitcnt vmcnt(46)                                // wait C (interleaved) 46 = 60 - 13 - 1
v_add_f32 v[vgprValuC+144], v[vgprValuC+144], v206 // accum partials

s_waitcnt vmcnt(45)                                // wait C (interleaved) 45 = 60 - 14 - 1
v_add_f32 v[vgprValuC+145], v[vgprValuC+145], v207 // accum partials

s_waitcnt vmcnt(44)                                // wait C (interleaved) 44 = 60 - 15 - 1
v_add_f32 v[vgprValuC+146], v[vgprValuC+146], v208 // accum partials

s_waitcnt vmcnt(43)                                // wait C (interleaved) 43 = 60 - 16 - 1
v_add_f32 v[vgprValuC+147], v[vgprValuC+147], v209 // accum partials

s_waitcnt vmcnt(42)                                // wait C (interleaved) 42 = 60 - 17 - 1
v_add_f32 v[vgprValuC+148], v[vgprValuC+148], v210 // accum partials

s_waitcnt vmcnt(41)                                // wait C (interleaved) 41 = 60 - 18 - 1
v_add_f32 v[vgprValuC+149], v[vgprValuC+149], v211 // accum partials

s_waitcnt vmcnt(40)                                // wait C (interleaved) 40 = 60 - 19 - 1
v_add_f32 v[vgprValuC+150], v[vgprValuC+150], v212 // accum partials

s_waitcnt vmcnt(39)                                // wait C (interleaved) 39 = 60 - 20 - 1
v_add_f32 v[vgprValuC+151], v[vgprValuC+151], v213 // accum partials

s_waitcnt vmcnt(38)                                // wait C (interleaved) 38 = 60 - 21 - 1
v_add_f32 v[vgprValuC+152], v[vgprValuC+152], v214 // accum partials

s_waitcnt vmcnt(37)                                // wait C (interleaved) 37 = 60 - 22 - 1
v_add_f32 v[vgprValuC+153], v[vgprValuC+153], v215 // accum partials

s_waitcnt vmcnt(36)                                // wait C (interleaved) 36 = 60 - 23 - 1
v_add_f32 v[vgprValuC+154], v[vgprValuC+154], v216 // accum partials

s_waitcnt vmcnt(35)                                // wait C (interleaved) 35 = 60 - 24 - 1
v_add_f32 v[vgprValuC+155], v[vgprValuC+155], v217 // accum partials

s_waitcnt vmcnt(34)                                // wait C (interleaved) 34 = 60 - 25 - 1
v_add_f32 v[vgprValuC+156], v[vgprValuC+156], v218 // accum partials

s_waitcnt vmcnt(33)                                // wait C (interleaved) 33 = 60 - 26 - 1
v_add_f32 v[vgprValuC+157], v[vgprValuC+157], v219 // accum partials

s_waitcnt vmcnt(32)                                // wait C (interleaved) 32 = 60 - 27 - 1
v_add_f32 v[vgprValuC+158], v[vgprValuC+158], v220 // accum partials

s_waitcnt vmcnt(31)                                // wait C (interleaved) 31 = 60 - 28 - 1
v_add_f32 v[vgprValuC+159], v[vgprValuC+159], v221 // accum partials

s_waitcnt vmcnt(30)                                // wait C (interleaved) 30 = 60 - 29 - 1
v_add_f32 v[vgprValuC+160], v[vgprValuC+160], v222 // accum partials

s_waitcnt vmcnt(29)                                // wait C (interleaved) 29 = 60 - 30 - 1
v_add_f32 v[vgprValuC+161], v[vgprValuC+161], v223 // accum partials

s_waitcnt vmcnt(28)                                // wait C (interleaved) 28 = 60 - 31 - 1
v_add_f32 v[vgprValuC+162], v[vgprValuC+162], v224 // accum partials

s_waitcnt vmcnt(27)                                // wait C (interleaved) 27 = 60 - 32 - 1
v_add_f32 v[vgprValuC+163], v[vgprValuC+163], v225 // accum partials

s_waitcnt vmcnt(26)                                // wait C (interleaved) 26 = 60 - 33 - 1
v_add_f32 v[vgprValuC+164], v[vgprValuC+164], v226 // accum partials

s_waitcnt vmcnt(25)                                // wait C (interleaved) 25 = 60 - 34 - 1
v_add_f32 v[vgprValuC+165], v[vgprValuC+165], v227 // accum partials

s_waitcnt vmcnt(24)                                // wait C (interleaved) 24 = 60 - 35 - 1
v_add_f32 v[vgprValuC+166], v[vgprValuC+166], v228 // accum partials

s_waitcnt vmcnt(23)                                // wait C (interleaved) 23 = 60 - 36 - 1
v_add_f32 v[vgprValuC+167], v[vgprValuC+167], v229 // accum partials

s_waitcnt vmcnt(22)                                // wait C (interleaved) 22 = 60 - 37 - 1
v_add_f32 v[vgprValuC+168], v[vgprValuC+168], v230 // accum partials

s_waitcnt vmcnt(21)                                // wait C (interleaved) 21 = 60 - 38 - 1
v_add_f32 v[vgprValuC+169], v[vgprValuC+169], v231 // accum partials

s_waitcnt vmcnt(20)                                // wait C (interleaved) 20 = 60 - 39 - 1
v_add_f32 v[vgprValuC+170], v[vgprValuC+170], v232 // accum partials

s_waitcnt vmcnt(19)                                // wait C (interleaved) 19 = 60 - 40 - 1
v_add_f32 v[vgprValuC+171], v[vgprValuC+171], v233 // accum partials

s_waitcnt vmcnt(18)                                // wait C (interleaved) 18 = 60 - 41 - 1
v_add_f32 v[vgprValuC+172], v[vgprValuC+172], v234 // accum partials

s_waitcnt vmcnt(17)                                // wait C (interleaved) 17 = 60 - 42 - 1
v_add_f32 v[vgprValuC+173], v[vgprValuC+173], v235 // accum partials

s_waitcnt vmcnt(16)                                // wait C (interleaved) 16 = 60 - 43 - 1
v_add_f32 v[vgprValuC+174], v[vgprValuC+174], v236 // accum partials

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 60 - 44 - 1
v_add_f32 v[vgprValuC+175], v[vgprValuC+175], v237 // accum partials

s_waitcnt vmcnt(14)                                // wait C (interleaved) 14 = 60 - 45 - 1
v_add_f32 v[vgprValuC+176], v[vgprValuC+176], v238 // accum partials

s_waitcnt vmcnt(13)                                // wait C (interleaved) 13 = 60 - 46 - 1
v_add_f32 v[vgprValuC+177], v[vgprValuC+177], v239 // accum partials

s_waitcnt vmcnt(12)                                // wait C (interleaved) 12 = 60 - 47 - 1
v_add_f32 v[vgprValuC+178], v[vgprValuC+178], v240 // accum partials

s_waitcnt vmcnt(11)                                // wait C (interleaved) 11 = 60 - 48 - 1
v_add_f32 v[vgprValuC+179], v[vgprValuC+179], v241 // accum partials

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 60 - 49 - 1
v_add_f32 v[vgprValuC+180], v[vgprValuC+180], v242 // accum partials

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 60 - 50 - 1
v_add_f32 v[vgprValuC+181], v[vgprValuC+181], v243 // accum partials

s_waitcnt vmcnt(8)                                 // wait C (interleaved) 8 = 60 - 51 - 1
v_add_f32 v[vgprValuC+182], v[vgprValuC+182], v244 // accum partials

s_waitcnt vmcnt(7)                                 // wait C (interleaved) 7 = 60 - 52 - 1
v_add_f32 v[vgprValuC+183], v[vgprValuC+183], v245 // accum partials

s_waitcnt vmcnt(6)                                 // wait C (interleaved) 6 = 60 - 53 - 1
v_add_f32 v[vgprValuC+184], v[vgprValuC+184], v246 // accum partials

s_waitcnt vmcnt(5)                                 // wait C (interleaved) 5 = 60 - 54 - 1
v_add_f32 v[vgprValuC+185], v[vgprValuC+185], v247 // accum partials

s_waitcnt vmcnt(4)                                 // wait C (interleaved) 4 = 60 - 55 - 1
v_add_f32 v[vgprValuC+186], v[vgprValuC+186], v248 // accum partials

s_waitcnt vmcnt(3)                                 // wait C (interleaved) 3 = 60 - 56 - 1
v_add_f32 v[vgprValuC+187], v[vgprValuC+187], v249 // accum partials

s_waitcnt vmcnt(2)                                 // wait C (interleaved) 2 = 60 - 57 - 1
v_add_f32 v[vgprValuC+188], v[vgprValuC+188], v250 // accum partials

s_waitcnt vmcnt(1)                                 // wait C (interleaved) 1 = 60 - 58 - 1
v_add_f32 v[vgprValuC+189], v[vgprValuC+189], v251 // accum partials

s_waitcnt vmcnt(0)                                 // wait C (interleaved) 0 = 60 - 59 - 1
v_add_f32 v[vgprValuC+190], v[vgprValuC+190], v252 // accum partials
v_mov_b32 v[vgprValuC+0], v[vgprValuC+131]         // copy vreg[0] to MI out reg
v_mov_b32 v[vgprValuC+8], v[vgprValuC+132]         // copy vreg[1] to MI out reg
v_mov_b32 v[vgprValuC+16], v[vgprValuC+133]        // copy vreg[2] to MI out reg
v_mov_b32 v[vgprValuC+1], v[vgprValuC+134]         // copy vreg[3] to MI out reg
v_mov_b32 v[vgprValuC+9], v[vgprValuC+135]         // copy vreg[4] to MI out reg
v_mov_b32 v[vgprValuC+17], v[vgprValuC+136]        // copy vreg[5] to MI out reg
v_mov_b32 v[vgprValuC+2], v[vgprValuC+137]         // copy vreg[6] to MI out reg
v_mov_b32 v[vgprValuC+10], v[vgprValuC+138]        // copy vreg[7] to MI out reg
v_mov_b32 v[vgprValuC+18], v[vgprValuC+139]        // copy vreg[8] to MI out reg
v_mov_b32 v[vgprValuC+3], v[vgprValuC+140]         // copy vreg[9] to MI out reg
v_mov_b32 v[vgprValuC+11], v[vgprValuC+141]        // copy vreg[10] to MI out reg
v_mov_b32 v[vgprValuC+19], v[vgprValuC+142]        // copy vreg[11] to MI out reg
v_mov_b32 v[vgprValuC+4], v[vgprValuC+143]         // copy vreg[12] to MI out reg
v_mov_b32 v[vgprValuC+12], v[vgprValuC+144]        // copy vreg[13] to MI out reg
v_mov_b32 v[vgprValuC+20], v[vgprValuC+145]        // copy vreg[14] to MI out reg
v_mov_b32 v[vgprValuC+5], v[vgprValuC+146]         // copy vreg[15] to MI out reg
v_mov_b32 v[vgprValuC+13], v[vgprValuC+147]        // copy vreg[16] to MI out reg
v_mov_b32 v[vgprValuC+21], v[vgprValuC+148]        // copy vreg[17] to MI out reg
v_mov_b32 v[vgprValuC+6], v[vgprValuC+149]         // copy vreg[18] to MI out reg
v_mov_b32 v[vgprValuC+14], v[vgprValuC+150]        // copy vreg[19] to MI out reg
v_mov_b32 v[vgprValuC+22], v[vgprValuC+151]        // copy vreg[20] to MI out reg
v_mov_b32 v[vgprValuC+7], v[vgprValuC+152]         // copy vreg[21] to MI out reg
v_mov_b32 v[vgprValuC+15], v[vgprValuC+153]        // copy vreg[22] to MI out reg
v_mov_b32 v[vgprValuC+23], v[vgprValuC+154]        // copy vreg[23] to MI out reg
v_mov_b32 v[vgprValuC+24], v[vgprValuC+155]        // copy vreg[24] to MI out reg
v_mov_b32 v[vgprValuC+32], v[vgprValuC+156]        // copy vreg[25] to MI out reg
v_mov_b32 v[vgprValuC+40], v[vgprValuC+157]        // copy vreg[26] to MI out reg
v_mov_b32 v[vgprValuC+25], v[vgprValuC+158]        // copy vreg[27] to MI out reg
v_mov_b32 v[vgprValuC+33], v[vgprValuC+159]        // copy vreg[28] to MI out reg
v_mov_b32 v[vgprValuC+41], v[vgprValuC+160]        // copy vreg[29] to MI out reg
v_mov_b32 v[vgprValuC+26], v[vgprValuC+161]        // copy vreg[30] to MI out reg
v_mov_b32 v[vgprValuC+34], v[vgprValuC+162]        // copy vreg[31] to MI out reg
v_mov_b32 v[vgprValuC+42], v[vgprValuC+163]        // copy vreg[32] to MI out reg
v_mov_b32 v[vgprValuC+27], v[vgprValuC+164]        // copy vreg[33] to MI out reg
v_mov_b32 v[vgprValuC+35], v[vgprValuC+165]        // copy vreg[34] to MI out reg
v_mov_b32 v[vgprValuC+43], v[vgprValuC+166]        // copy vreg[35] to MI out reg
v_mov_b32 v[vgprValuC+28], v[vgprValuC+167]        // copy vreg[36] to MI out reg
v_mov_b32 v[vgprValuC+36], v[vgprValuC+168]        // copy vreg[37] to MI out reg
v_mov_b32 v[vgprValuC+44], v[vgprValuC+169]        // copy vreg[38] to MI out reg
v_mov_b32 v[vgprValuC+29], v[vgprValuC+170]        // copy vreg[39] to MI out reg
v_mov_b32 v[vgprValuC+37], v[vgprValuC+171]        // copy vreg[40] to MI out reg
v_mov_b32 v[vgprValuC+45], v[vgprValuC+172]        // copy vreg[41] to MI out reg
v_mov_b32 v[vgprValuC+30], v[vgprValuC+173]        // copy vreg[42] to MI out reg
v_mov_b32 v[vgprValuC+38], v[vgprValuC+174]        // copy vreg[43] to MI out reg
v_mov_b32 v[vgprValuC+46], v[vgprValuC+175]        // copy vreg[44] to MI out reg
v_mov_b32 v[vgprValuC+31], v[vgprValuC+176]        // copy vreg[45] to MI out reg
v_mov_b32 v[vgprValuC+39], v[vgprValuC+177]        // copy vreg[46] to MI out reg
v_mov_b32 v[vgprValuC+47], v[vgprValuC+178]        // copy vreg[47] to MI out reg
v_mov_b32 v[vgprValuC+48], v[vgprValuC+179]        // copy vreg[48] to MI out reg
v_mov_b32 v[vgprValuC+56], v[vgprValuC+180]        // copy vreg[49] to MI out reg
v_mov_b32 v[vgprValuC+64], v[vgprValuC+181]        // copy vreg[50] to MI out reg
v_mov_b32 v[vgprValuC+49], v[vgprValuC+182]        // copy vreg[51] to MI out reg
v_mov_b32 v[vgprValuC+57], v[vgprValuC+183]        // copy vreg[52] to MI out reg
v_mov_b32 v[vgprValuC+65], v[vgprValuC+184]        // copy vreg[53] to MI out reg
v_mov_b32 v[vgprValuC+50], v[vgprValuC+185]        // copy vreg[54] to MI out reg
v_mov_b32 v[vgprValuC+58], v[vgprValuC+186]        // copy vreg[55] to MI out reg
v_mov_b32 v[vgprValuC+66], v[vgprValuC+187]        // copy vreg[56] to MI out reg
v_mov_b32 v[vgprValuC+51], v[vgprValuC+188]        // copy vreg[57] to MI out reg
v_mov_b32 v[vgprValuC+59], v[vgprValuC+189]        // copy vreg[58] to MI out reg
v_mov_b32 v[vgprValuC+67], v[vgprValuC+190]        // copy vreg[59] to MI out reg
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 7                                          // optimization: sync and wait
s_barrier

/******************************************/
/* Fixup Batch #1 (d1,d0,vc1,vc0) =       */
/*      (20,0,0,0:vw1); (20,1,0,0:vw1); (20,2,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (21,2,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (22,2,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1); (23,2,0,0:vw1); (24,0,0,0:vw1); (24,1,0,0:vw1); (24,2,0,0:vw1); (25,0,0,0:vw1); (25,1,0,0:vw1); (25,2,0,0:vw1); (26,0,0,0:vw1); (26,1,0,0:vw1); (26,2,0,0:vw1); (27,0,0,0:vw1); (27,1,0,0:vw1); (27,2,0,0:vw1); (28,0,0,0:vw1); (28,1,0,0:vw1); (28,2,0,0:vw1); (29,0,0,0:vw1); (29,1,0,0:vw1); (29,2,0,0:vw1); (30,0,0,0:vw1); (30,1,0,0:vw1); (30,2,0,0:vw1); (31,0,0,0:vw1); (31,1,0,0:vw1); (31,2,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v167, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v168, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v169, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v170, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v171, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v172, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v173, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v174, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v175, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v176, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v177, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v178, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v179, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v180, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v181, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v182, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v183, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v184, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v185, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v186, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v187, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v188, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v189, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v190, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v191, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v192, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v193, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v194, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v195, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v196, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v197, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v198, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v199, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v200, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v201, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
s_add_u32 s64, s64, 2048                           // Inc sgpr offset
buffer_load_b32 v202, v130, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 glc slc // load WS
v_mov_b32 v[vgprValuC+131], v[vgprValuC+52]        // copy MI out reg to vreg[60]
v_mov_b32 v[vgprValuC+132], v[vgprValuC+60]        // copy MI out reg to vreg[61]
v_mov_b32 v[vgprValuC+133], v[vgprValuC+68]        // copy MI out reg to vreg[62]
v_mov_b32 v[vgprValuC+134], v[vgprValuC+53]        // copy MI out reg to vreg[63]
v_mov_b32 v[vgprValuC+135], v[vgprValuC+61]        // copy MI out reg to vreg[64]
v_mov_b32 v[vgprValuC+136], v[vgprValuC+69]        // copy MI out reg to vreg[65]
v_mov_b32 v[vgprValuC+137], v[vgprValuC+54]        // copy MI out reg to vreg[66]
v_mov_b32 v[vgprValuC+138], v[vgprValuC+62]        // copy MI out reg to vreg[67]
v_mov_b32 v[vgprValuC+139], v[vgprValuC+70]        // copy MI out reg to vreg[68]
v_mov_b32 v[vgprValuC+140], v[vgprValuC+55]        // copy MI out reg to vreg[69]
v_mov_b32 v[vgprValuC+141], v[vgprValuC+63]        // copy MI out reg to vreg[70]
v_mov_b32 v[vgprValuC+142], v[vgprValuC+71]        // copy MI out reg to vreg[71]
v_mov_b32 v[vgprValuC+143], v[vgprValuC+72]        // copy MI out reg to vreg[72]
v_mov_b32 v[vgprValuC+144], v[vgprValuC+80]        // copy MI out reg to vreg[73]
v_mov_b32 v[vgprValuC+145], v[vgprValuC+88]        // copy MI out reg to vreg[74]
v_mov_b32 v[vgprValuC+146], v[vgprValuC+73]        // copy MI out reg to vreg[75]
v_mov_b32 v[vgprValuC+147], v[vgprValuC+81]        // copy MI out reg to vreg[76]
v_mov_b32 v[vgprValuC+148], v[vgprValuC+89]        // copy MI out reg to vreg[77]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+74]        // copy MI out reg to vreg[78]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+82]        // copy MI out reg to vreg[79]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+90]        // copy MI out reg to vreg[80]
v_mov_b32 v[vgprValuC+152], v[vgprValuC+75]        // copy MI out reg to vreg[81]
v_mov_b32 v[vgprValuC+153], v[vgprValuC+83]        // copy MI out reg to vreg[82]
v_mov_b32 v[vgprValuC+154], v[vgprValuC+91]        // copy MI out reg to vreg[83]
v_mov_b32 v[vgprValuC+155], v[vgprValuC+76]        // copy MI out reg to vreg[84]
v_mov_b32 v[vgprValuC+156], v[vgprValuC+84]        // copy MI out reg to vreg[85]
v_mov_b32 v[vgprValuC+157], v[vgprValuC+92]        // copy MI out reg to vreg[86]
v_mov_b32 v[vgprValuC+158], v[vgprValuC+77]        // copy MI out reg to vreg[87]
v_mov_b32 v[vgprValuC+159], v[vgprValuC+85]        // copy MI out reg to vreg[88]
v_mov_b32 v[vgprValuC+160], v[vgprValuC+93]        // copy MI out reg to vreg[89]
v_mov_b32 v[vgprValuC+161], v[vgprValuC+78]        // copy MI out reg to vreg[90]
v_mov_b32 v[vgprValuC+162], v[vgprValuC+86]        // copy MI out reg to vreg[91]
v_mov_b32 v[vgprValuC+163], v[vgprValuC+94]        // copy MI out reg to vreg[92]
v_mov_b32 v[vgprValuC+164], v[vgprValuC+79]        // copy MI out reg to vreg[93]
v_mov_b32 v[vgprValuC+165], v[vgprValuC+87]        // copy MI out reg to vreg[94]
v_mov_b32 v[vgprValuC+166], v[vgprValuC+95]        // copy MI out reg to vreg[95]

/* apply mask, calc new C and issue writes */
v_mov_b32 v126, 0xffff0000                         // mask for pack two bfloat16 element to 32bit
v_mov_b32 v127, 0x7fff0000                         // fp32 Nan
v_mov_b32 v128, 0x7fff                             // rounding bias for bfloat16

s_waitcnt vmcnt(35)                                // wait C (interleaved) 35 = 36 - 0 - 1
v_add_f32 v[vgprValuC+131], v[vgprValuC+131], v167 // accum partials

s_waitcnt vmcnt(34)                                // wait C (interleaved) 34 = 36 - 1 - 1
v_add_f32 v[vgprValuC+132], v[vgprValuC+132], v168 // accum partials

s_waitcnt vmcnt(33)                                // wait C (interleaved) 33 = 36 - 2 - 1
v_add_f32 v[vgprValuC+133], v[vgprValuC+133], v169 // accum partials

s_waitcnt vmcnt(32)                                // wait C (interleaved) 32 = 36 - 3 - 1
v_add_f32 v[vgprValuC+134], v[vgprValuC+134], v170 // accum partials

s_waitcnt vmcnt(31)                                // wait C (interleaved) 31 = 36 - 4 - 1
v_add_f32 v[vgprValuC+135], v[vgprValuC+135], v171 // accum partials

s_waitcnt vmcnt(30)                                // wait C (interleaved) 30 = 36 - 5 - 1
v_add_f32 v[vgprValuC+136], v[vgprValuC+136], v172 // accum partials

s_waitcnt vmcnt(29)                                // wait C (interleaved) 29 = 36 - 6 - 1
v_add_f32 v[vgprValuC+137], v[vgprValuC+137], v173 // accum partials

s_waitcnt vmcnt(28)                                // wait C (interleaved) 28 = 36 - 7 - 1
v_add_f32 v[vgprValuC+138], v[vgprValuC+138], v174 // accum partials

s_waitcnt vmcnt(27)                                // wait C (interleaved) 27 = 36 - 8 - 1
v_add_f32 v[vgprValuC+139], v[vgprValuC+139], v175 // accum partials

s_waitcnt vmcnt(26)                                // wait C (interleaved) 26 = 36 - 9 - 1
v_add_f32 v[vgprValuC+140], v[vgprValuC+140], v176 // accum partials

s_waitcnt vmcnt(25)                                // wait C (interleaved) 25 = 36 - 10 - 1
v_add_f32 v[vgprValuC+141], v[vgprValuC+141], v177 // accum partials

s_waitcnt vmcnt(24)                                // wait C (interleaved) 24 = 36 - 11 - 1
v_add_f32 v[vgprValuC+142], v[vgprValuC+142], v178 // accum partials

s_waitcnt vmcnt(23)                                // wait C (interleaved) 23 = 36 - 12 - 1
v_add_f32 v[vgprValuC+143], v[vgprValuC+143], v179 // accum partials

s_waitcnt vmcnt(22)                                // wait C (interleaved) 22 = 36 - 13 - 1
v_add_f32 v[vgprValuC+144], v[vgprValuC+144], v180 // accum partials

s_waitcnt vmcnt(21)                                // wait C (interleaved) 21 = 36 - 14 - 1
v_add_f32 v[vgprValuC+145], v[vgprValuC+145], v181 // accum partials

s_waitcnt vmcnt(20)                                // wait C (interleaved) 20 = 36 - 15 - 1
v_add_f32 v[vgprValuC+146], v[vgprValuC+146], v182 // accum partials

s_waitcnt vmcnt(19)                                // wait C (interleaved) 19 = 36 - 16 - 1
v_add_f32 v[vgprValuC+147], v[vgprValuC+147], v183 // accum partials

s_waitcnt vmcnt(18)                                // wait C (interleaved) 18 = 36 - 17 - 1
v_add_f32 v[vgprValuC+148], v[vgprValuC+148], v184 // accum partials

s_waitcnt vmcnt(17)                                // wait C (interleaved) 17 = 36 - 18 - 1
v_add_f32 v[vgprValuC+149], v[vgprValuC+149], v185 // accum partials

s_waitcnt vmcnt(16)                                // wait C (interleaved) 16 = 36 - 19 - 1
v_add_f32 v[vgprValuC+150], v[vgprValuC+150], v186 // accum partials

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 36 - 20 - 1
v_add_f32 v[vgprValuC+151], v[vgprValuC+151], v187 // accum partials

s_waitcnt vmcnt(14)                                // wait C (interleaved) 14 = 36 - 21 - 1
v_add_f32 v[vgprValuC+152], v[vgprValuC+152], v188 // accum partials

s_waitcnt vmcnt(13)                                // wait C (interleaved) 13 = 36 - 22 - 1
v_add_f32 v[vgprValuC+153], v[vgprValuC+153], v189 // accum partials

s_waitcnt vmcnt(12)                                // wait C (interleaved) 12 = 36 - 23 - 1
v_add_f32 v[vgprValuC+154], v[vgprValuC+154], v190 // accum partials

s_waitcnt vmcnt(11)                                // wait C (interleaved) 11 = 36 - 24 - 1
v_add_f32 v[vgprValuC+155], v[vgprValuC+155], v191 // accum partials

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 36 - 25 - 1
v_add_f32 v[vgprValuC+156], v[vgprValuC+156], v192 // accum partials

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 36 - 26 - 1
v_add_f32 v[vgprValuC+157], v[vgprValuC+157], v193 // accum partials

s_waitcnt vmcnt(8)                                 // wait C (interleaved) 8 = 36 - 27 - 1
v_add_f32 v[vgprValuC+158], v[vgprValuC+158], v194 // accum partials

s_waitcnt vmcnt(7)                                 // wait C (interleaved) 7 = 36 - 28 - 1
v_add_f32 v[vgprValuC+159], v[vgprValuC+159], v195 // accum partials

s_waitcnt vmcnt(6)                                 // wait C (interleaved) 6 = 36 - 29 - 1
v_add_f32 v[vgprValuC+160], v[vgprValuC+160], v196 // accum partials

s_waitcnt vmcnt(5)                                 // wait C (interleaved) 5 = 36 - 30 - 1
v_add_f32 v[vgprValuC+161], v[vgprValuC+161], v197 // accum partials

s_waitcnt vmcnt(4)                                 // wait C (interleaved) 4 = 36 - 31 - 1
v_add_f32 v[vgprValuC+162], v[vgprValuC+162], v198 // accum partials

s_waitcnt vmcnt(3)                                 // wait C (interleaved) 3 = 36 - 32 - 1
v_add_f32 v[vgprValuC+163], v[vgprValuC+163], v199 // accum partials

s_waitcnt vmcnt(2)                                 // wait C (interleaved) 2 = 36 - 33 - 1
v_add_f32 v[vgprValuC+164], v[vgprValuC+164], v200 // accum partials

s_waitcnt vmcnt(1)                                 // wait C (interleaved) 1 = 36 - 34 - 1
v_add_f32 v[vgprValuC+165], v[vgprValuC+165], v201 // accum partials

s_waitcnt vmcnt(0)                                 // wait C (interleaved) 0 = 36 - 35 - 1
v_add_f32 v[vgprValuC+166], v[vgprValuC+166], v202 // accum partials
v_mov_b32 v[vgprValuC+52], v[vgprValuC+131]        // copy vreg[60] to MI out reg
v_mov_b32 v[vgprValuC+60], v[vgprValuC+132]        // copy vreg[61] to MI out reg
v_mov_b32 v[vgprValuC+68], v[vgprValuC+133]        // copy vreg[62] to MI out reg
v_mov_b32 v[vgprValuC+53], v[vgprValuC+134]        // copy vreg[63] to MI out reg
v_mov_b32 v[vgprValuC+61], v[vgprValuC+135]        // copy vreg[64] to MI out reg
v_mov_b32 v[vgprValuC+69], v[vgprValuC+136]        // copy vreg[65] to MI out reg
v_mov_b32 v[vgprValuC+54], v[vgprValuC+137]        // copy vreg[66] to MI out reg
v_mov_b32 v[vgprValuC+62], v[vgprValuC+138]        // copy vreg[67] to MI out reg
v_mov_b32 v[vgprValuC+70], v[vgprValuC+139]        // copy vreg[68] to MI out reg
v_mov_b32 v[vgprValuC+55], v[vgprValuC+140]        // copy vreg[69] to MI out reg
v_mov_b32 v[vgprValuC+63], v[vgprValuC+141]        // copy vreg[70] to MI out reg
v_mov_b32 v[vgprValuC+71], v[vgprValuC+142]        // copy vreg[71] to MI out reg
v_mov_b32 v[vgprValuC+72], v[vgprValuC+143]        // copy vreg[72] to MI out reg
v_mov_b32 v[vgprValuC+80], v[vgprValuC+144]        // copy vreg[73] to MI out reg
v_mov_b32 v[vgprValuC+88], v[vgprValuC+145]        // copy vreg[74] to MI out reg
v_mov_b32 v[vgprValuC+73], v[vgprValuC+146]        // copy vreg[75] to MI out reg
v_mov_b32 v[vgprValuC+81], v[vgprValuC+147]        // copy vreg[76] to MI out reg
v_mov_b32 v[vgprValuC+89], v[vgprValuC+148]        // copy vreg[77] to MI out reg
v_mov_b32 v[vgprValuC+74], v[vgprValuC+149]        // copy vreg[78] to MI out reg
v_mov_b32 v[vgprValuC+82], v[vgprValuC+150]        // copy vreg[79] to MI out reg
v_mov_b32 v[vgprValuC+90], v[vgprValuC+151]        // copy vreg[80] to MI out reg
v_mov_b32 v[vgprValuC+75], v[vgprValuC+152]        // copy vreg[81] to MI out reg
v_mov_b32 v[vgprValuC+83], v[vgprValuC+153]        // copy vreg[82] to MI out reg
v_mov_b32 v[vgprValuC+91], v[vgprValuC+154]        // copy vreg[83] to MI out reg
v_mov_b32 v[vgprValuC+76], v[vgprValuC+155]        // copy vreg[84] to MI out reg
v_mov_b32 v[vgprValuC+84], v[vgprValuC+156]        // copy vreg[85] to MI out reg
v_mov_b32 v[vgprValuC+92], v[vgprValuC+157]        // copy vreg[86] to MI out reg
v_mov_b32 v[vgprValuC+77], v[vgprValuC+158]        // copy vreg[87] to MI out reg
v_mov_b32 v[vgprValuC+85], v[vgprValuC+159]        // copy vreg[88] to MI out reg
v_mov_b32 v[vgprValuC+93], v[vgprValuC+160]        // copy vreg[89] to MI out reg
v_mov_b32 v[vgprValuC+78], v[vgprValuC+161]        // copy vreg[90] to MI out reg
v_mov_b32 v[vgprValuC+86], v[vgprValuC+162]        // copy vreg[91] to MI out reg
v_mov_b32 v[vgprValuC+94], v[vgprValuC+163]        // copy vreg[92] to MI out reg
v_mov_b32 v[vgprValuC+79], v[vgprValuC+164]        // copy vreg[93] to MI out reg
v_mov_b32 v[vgprValuC+87], v[vgprValuC+165]        // copy vreg[94] to MI out reg
v_mov_b32 v[vgprValuC+95], v[vgprValuC+166]        // copy vreg[95] to MI out reg
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_mul_i32 s11, s[sgprskTiles], s[sgprItersPerTile]
s_mul_i32 s61, s[sgprSKItersPerWG], s[sgprskGrid]
s_sub_u32 s11, s11, s61                            // skTiles * ItersPerTile - SKItersPerWG * skGrid
s_bitcmp1_b32 s[sgprMagicShiftItersPerTile], 29    // USO on? (bit 29 of MagicShiftItersPerTile); off -> historical global peer size
s_cbranch_scc0 label_SK_PeerGlobal                 // USO off -> historical global mapping
s_cmp_eq_u32 s[sgprskTiles], 0                     // skTiles == 0?
s_cbranch_scc1 label_SK_PeerNoTiles                // global peer size
v_cvt_f32_u32 v129, s[sgprskTiles]                 // skGrid % skTiles
v_rcp_iflag_f32 v129, v129                         // skGrid % skTiles
v_mov_b32 v130, 0x4f7ffffe                         // 2^32 - 512
v_mul_f32 v129, v130, v129                         // 2^32 / divisor
v_cvt_u32_f32 v129, v129                           // skGrid % skTiles
v_mul_lo_u32 v130, v129, s[sgprskTiles]            // refine 2^32 / divisor
v_sub_nc_u32 v130, 0, v130                         // refine 2^32 / divisor
v_mul_hi_u32 v130, v129, v130                      // refine 2^32 / divisor
v_add_nc_u32 v129, v129, v130                      // refine 2^32 / divisor
v_mul_hi_u32 v129, s[sgprskGrid], v129             // skGrid % skTiles
v_mul_lo_u32 v130, v129, s[sgprskTiles]            // skGrid % skTiles
v_sub_nc_u32 v130, s[sgprskGrid], v130             // skGrid % skTiles
v_cmp_ge_u32 vcc_lo, v130, s[sgprskTiles]          // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v129, 1, v129                         // quotient + 1
v_sub_nc_u32 v130, v130, s[sgprskTiles]            // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v130, s[sgprskTiles]          // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v129, 1, v129                         // quotient + 1
v_sub_nc_u32 v130, v130, s[sgprskTiles]            // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s61, v129                      // quotient
v_readfirstlane_b32 s61, v130                      // remainder
s_cmp_eq_u32 s61, 0                                // skGrid % skTiles == 0?
s_cbranch_scc0 label_SK_PeerGlobal                 // ragged -> global peer
label_SK_PeerPerTile:
v_cvt_f32_u32 v129, s[sgprskTiles]                 // F = skGrid / skTiles
v_rcp_iflag_f32 v129, v129                         // F = skGrid / skTiles
v_mov_b32 v130, 0x4f7ffffe                         // 2^32 - 512
v_mul_f32 v129, v130, v129                         // 2^32 / divisor
v_cvt_u32_f32 v129, v129                           // F = skGrid / skTiles
v_mul_lo_u32 v130, v129, s[sgprskTiles]            // refine 2^32 / divisor
v_sub_nc_u32 v130, 0, v130                         // refine 2^32 / divisor
v_mul_hi_u32 v130, v129, v130                      // refine 2^32 / divisor
v_add_nc_u32 v129, v129, v130                      // refine 2^32 / divisor
v_mul_hi_u32 v129, s[sgprskGrid], v129             // F = skGrid / skTiles
v_mul_lo_u32 v130, v129, s[sgprskTiles]            // F = skGrid / skTiles
v_sub_nc_u32 v130, s[sgprskGrid], v130             // F = skGrid / skTiles
v_cmp_ge_u32 vcc_lo, v130, s[sgprskTiles]          // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v129, 1, v129                         // quotient + 1
v_sub_nc_u32 v130, v130, s[sgprskTiles]            // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v130, s[sgprskTiles]          // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v129, 1, v129                         // quotient + 1
v_sub_nc_u32 v130, v130, s[sgprskTiles]            // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s61, v129                      // quotient
s_mov_b32 s11, s61                                 // F = skGrid / skTiles
v_cvt_f32_u32 v129, s11                            // s = cta % F
v_rcp_iflag_f32 v129, v129                         // s = cta % F
v_mov_b32 v130, 0x4f7ffffe                         // 2^32 - 512
v_mul_f32 v129, v130, v129                         // 2^32 / divisor
v_cvt_u32_f32 v129, v129                           // s = cta % F
v_mul_lo_u32 v130, v129, s11                       // refine 2^32 / divisor
v_sub_nc_u32 v130, 0, v130                         // refine 2^32 / divisor
v_mul_hi_u32 v130, v129, v130                      // refine 2^32 / divisor
v_add_nc_u32 v129, v129, v130                      // refine 2^32 / divisor
v_mul_hi_u32 v129, s7, v129                        // s = cta % F
v_mul_lo_u32 v130, v129, s11                       // s = cta % F
v_sub_nc_u32 v130, s7, v130                        // s = cta % F
v_cmp_ge_u32 vcc_lo, v130, s11                     // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v129, 1, v129                         // quotient + 1
v_sub_nc_u32 v130, v130, s11                       // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v130, s11                     // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v129, 1, v129                         // quotient + 1
v_sub_nc_u32 v130, v130, s11                       // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s61, v129                      // quotient
v_readfirstlane_b32 s61, v130                      // remainder
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

/* Edge/NonEdge store path check (M): Size % 192 > 0 -> Edge store; else -> NonEdge store */
s_mul_hi_u32 s67, s[sgprSizeI], 44739243           // STATIC_DIV: divisor=192
s_mul_i32 s66, s[sgprSizeI], 44739243              // tmp = dividend * magic
s_lshr_b64 s[66:67], s[66:67], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s65, s66                                 // quotient
s_mul_i32 s66, s65, 192                            // quotient*divisor
s_sub_u32 s64, s[sgprSizeI], s66                   // rReg = dividend - quotient*divisor
s_add_u32 s65, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s65                // wg0 >= nwg0-1 ?
s_cselect_b32 s64, s64, 0                          // set rem
s_cmpk_gt_u32 s64, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_GSU1_Else       // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 256 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s64, 255, s[sgprSizeJ]                   // s64 = s[sgprSizeJ] % 256
s_add_u32 s65, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s65                // wg1 >= nwg1-1
s_cselect_b32 s64, s64, 0                          // set rem
s_cmpk_gt_u32 s64, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_GSU1_Then       // jump if edges required
label_GW_B0_FD0_VW1_GSU1_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=122 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */
s_sleep 7                                          // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (8,2,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (9,2,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (10,2,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (11,2,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (12,2,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (13,2,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (14,2,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (15,2,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (16,2,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (17,2,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (18,2,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (19,2,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (20,2,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (21,2,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (22,2,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1); (23,2,0,0:vw1); (24,0,0,0:vw1); (24,1,0,0:vw1); (24,2,0,0:vw1); (25,0,0,0:vw1); (25,1,0,0:vw1); (25,2,0,0:vw1); (26,0,0,0:vw1); (26,1,0,0:vw1); (26,2,0,0:vw1); (27,0,0,0:vw1); (27,1,0,0:vw1); (27,2,0,0:vw1); (28,0,0,0:vw1); (28,1,0,0:vw1); (28,2,0,0:vw1); (29,0,0,0:vw1); (29,1,0,0:vw1); (29,2,0,0:vw1); (30,0,0,0:vw1); (30,1,0,0:vw1); (30,2,0,0:vw1); (31,0,0,0:vw1); (31,1,0,0:vw1); (31,2,0,0:vw1) */
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
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
/* (d1,vc1,d0,vc0)=(24,0,1,0) */
/* (d1,vc1,d0,vc0)=(24,0,2,0) */
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
/* (d1,vc1,d0,vc0)=(25,0,1,0) */
/* (d1,vc1,d0,vc0)=(25,0,2,0) */
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
/* (d1,vc1,d0,vc0)=(26,0,1,0) */
/* (d1,vc1,d0,vc0)=(26,0,2,0) */
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
/* (d1,vc1,d0,vc0)=(27,0,1,0) */
/* (d1,vc1,d0,vc0)=(27,0,2,0) */
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
/* (d1,vc1,d0,vc0)=(28,0,1,0) */
/* (d1,vc1,d0,vc0)=(28,0,2,0) */
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
/* (d1,vc1,d0,vc0)=(29,0,1,0) */
/* (d1,vc1,d0,vc0)=(29,0,2,0) */
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
/* (d1,vc1,d0,vc0)=(30,0,1,0) */
/* (d1,vc1,d0,vc0)=(30,0,2,0) */
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
/* (d1,vc1,d0,vc0)=(31,0,1,0) */
/* (d1,vc1,d0,vc0)=(31,0,2,0) */
v_add_lshl_u32 v129, v121, v118, 1                 // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=118, coord0Vgpr=118 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (5, 2, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (7, 2, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (8, 2, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (9, 2, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (10, 2, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (11, 2, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (12, 2, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (13, 2, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (14, 2, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (15, 2, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (16, 2, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (17, 2, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (18, 2, 0, 0), (19, 0, 0, 0), (19, 1, 0, 0), (19, 2, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0), (20, 2, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0), (21, 2, 0, 0), (22, 0, 0, 0), (22, 1, 0, 0), (22, 2, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0), (23, 2, 0, 0), (24, 0, 0, 0), (24, 1, 0, 0), (24, 2, 0, 0), (25, 0, 0, 0), (25, 1, 0, 0), (25, 2, 0, 0), (26, 0, 0, 0), (26, 1, 0, 0), (26, 2, 0, 0), (27, 0, 0, 0), (27, 1, 0, 0), (27, 2, 0, 0), (28, 0, 0, 0), (28, 1, 0, 0), (28, 2, 0, 0), (29, 0, 0, 0), (29, 1, 0, 0), (29, 2, 0, 0), (30, 0, 0, 0), (30, 1, 0, 0), (30, 2, 0, 0), (31, 0, 0, 0), (31, 1, 0, 0), (31, 2, 0, 0)] */
v_mul_f32 v[vgprValuC+131], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+132], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+133], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+134], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+135], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+152], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+154], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+158], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+160], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+162], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+163], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+164], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+166], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+167], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+168], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+169], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+170], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+171], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+172], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+173], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+174], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+175], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+176], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+177], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+178], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+179], s[sgprAlpha], v[vgprValuC+48] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+180], s[sgprAlpha], v[vgprValuC+56] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+181], s[sgprAlpha], v[vgprValuC+64] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+182], s[sgprAlpha], v[vgprValuC+49] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+183], s[sgprAlpha], v[vgprValuC+57] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+184], s[sgprAlpha], v[vgprValuC+65] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+185], s[sgprAlpha], v[vgprValuC+50] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+186], s[sgprAlpha], v[vgprValuC+58] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+187], s[sgprAlpha], v[vgprValuC+66] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+188], s[sgprAlpha], v[vgprValuC+51] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+189], s[sgprAlpha], v[vgprValuC+59] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+190], s[sgprAlpha], v[vgprValuC+67] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+191], s[sgprAlpha], v[vgprValuC+52] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+192], s[sgprAlpha], v[vgprValuC+60] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+193], s[sgprAlpha], v[vgprValuC+68] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+194], s[sgprAlpha], v[vgprValuC+53] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+195], s[sgprAlpha], v[vgprValuC+61] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+196], s[sgprAlpha], v[vgprValuC+69] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+197], s[sgprAlpha], v[vgprValuC+54] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+198], s[sgprAlpha], v[vgprValuC+62] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+199], s[sgprAlpha], v[vgprValuC+70] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+200], s[sgprAlpha], v[vgprValuC+55] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+201], s[sgprAlpha], v[vgprValuC+63] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+202], s[sgprAlpha], v[vgprValuC+71] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+203], s[sgprAlpha], v[vgprValuC+72] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+206], s[sgprAlpha], v[vgprValuC+80] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+207], s[sgprAlpha], v[vgprValuC+88] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+208], s[sgprAlpha], v[vgprValuC+73] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+209], s[sgprAlpha], v[vgprValuC+81] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+210], s[sgprAlpha], v[vgprValuC+89] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+211], s[sgprAlpha], v[vgprValuC+74] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+212], s[sgprAlpha], v[vgprValuC+82] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+213], s[sgprAlpha], v[vgprValuC+90] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+214], s[sgprAlpha], v[vgprValuC+75] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+215], s[sgprAlpha], v[vgprValuC+83] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+216], s[sgprAlpha], v[vgprValuC+91] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+217], s[sgprAlpha], v[vgprValuC+76] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+218], s[sgprAlpha], v[vgprValuC+84] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+219], s[sgprAlpha], v[vgprValuC+92] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+220], s[sgprAlpha], v[vgprValuC+77] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+221], s[sgprAlpha], v[vgprValuC+85] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+222], s[sgprAlpha], v[vgprValuC+93] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+223], s[sgprAlpha], v[vgprValuC+78] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+224], s[sgprAlpha], v[vgprValuC+86] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+225], s[sgprAlpha], v[vgprValuC+94] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+226], s[sgprAlpha], v[vgprValuC+79] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+227], s[sgprAlpha], v[vgprValuC+87] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+228], s[sgprAlpha], v[vgprValuC+95] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v126, 0xffff0000                         // mask for pack two bfloat16 element to 32bit
v_mov_b32 v127, 0x7fff0000                         // fp32 Nan
v_mov_b32 v128, 0x7fff                             // rounding bias for bfloat16
v_cmp_u_f32 s8, v[vgprValuC+131], v[vgprValuC+131] // check Nan
v_bfe_u32 v125, v[vgprValuC+131], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+131], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+131], v125, v127, s8
v_lshrrev_b32 v131, 16, v[vgprValuC+131]           // convert C to bf16
buffer_store_b16 v131, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+132], v[vgprValuC+132] // check Nan
v_bfe_u32 v125, v[vgprValuC+132], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+132], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+132], v125, v127, s8
v_lshrrev_b32 v132, 16, v[vgprValuC+132]           // convert C to bf16
buffer_store_b16 v132, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+133], v[vgprValuC+133] // check Nan
v_bfe_u32 v125, v[vgprValuC+133], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+133], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+133], v125, v127, s8
v_lshrrev_b32 v133, 16, v[vgprValuC+133]           // convert C to bf16
buffer_store_b16 v133, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+134], v[vgprValuC+134] // check Nan
v_bfe_u32 v125, v[vgprValuC+134], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+134], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+134], v125, v127, s8
v_lshrrev_b32 v134, 16, v[vgprValuC+134]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v134, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+135], v[vgprValuC+135] // check Nan
v_bfe_u32 v125, v[vgprValuC+135], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+135], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+135], v125, v127, s8
v_lshrrev_b32 v135, 16, v[vgprValuC+135]           // convert C to bf16
buffer_store_b16 v135, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+136], v[vgprValuC+136] // check Nan
v_bfe_u32 v125, v[vgprValuC+136], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+136], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+136], v125, v127, s8
v_lshrrev_b32 v136, 16, v[vgprValuC+136]           // convert C to bf16
buffer_store_b16 v136, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+137], v[vgprValuC+137] // check Nan
v_bfe_u32 v125, v[vgprValuC+137], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+137], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+137], v125, v127, s8
v_lshrrev_b32 v137, 16, v[vgprValuC+137]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v137, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+138], v[vgprValuC+138] // check Nan
v_bfe_u32 v125, v[vgprValuC+138], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+138], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+138], v125, v127, s8
v_lshrrev_b32 v138, 16, v[vgprValuC+138]           // convert C to bf16
buffer_store_b16 v138, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+139], v[vgprValuC+139] // check Nan
v_bfe_u32 v125, v[vgprValuC+139], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+139], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+139], v125, v127, s8
v_lshrrev_b32 v139, 16, v[vgprValuC+139]           // convert C to bf16
buffer_store_b16 v139, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+140], v[vgprValuC+140] // check Nan
v_bfe_u32 v125, v[vgprValuC+140], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+140], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+140], v125, v127, s8
v_lshrrev_b32 v140, 16, v[vgprValuC+140]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v140, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+141], v[vgprValuC+141] // check Nan
v_bfe_u32 v125, v[vgprValuC+141], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+141], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+141], v125, v127, s8
v_lshrrev_b32 v141, 16, v[vgprValuC+141]           // convert C to bf16
buffer_store_b16 v141, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+142], v[vgprValuC+142] // check Nan
v_bfe_u32 v125, v[vgprValuC+142], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+142], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+142], v125, v127, s8
v_lshrrev_b32 v142, 16, v[vgprValuC+142]           // convert C to bf16
buffer_store_b16 v142, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+143], v[vgprValuC+143] // check Nan
v_bfe_u32 v125, v[vgprValuC+143], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+143], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+143], v125, v127, s8
v_lshrrev_b32 v143, 16, v[vgprValuC+143]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v143, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+144], v[vgprValuC+144] // check Nan
v_bfe_u32 v125, v[vgprValuC+144], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+144], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+144], v125, v127, s8
v_lshrrev_b32 v144, 16, v[vgprValuC+144]           // convert C to bf16
buffer_store_b16 v144, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+145], v[vgprValuC+145] // check Nan
v_bfe_u32 v125, v[vgprValuC+145], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+145], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+145], v125, v127, s8
v_lshrrev_b32 v145, 16, v[vgprValuC+145]           // convert C to bf16
buffer_store_b16 v145, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+146], v[vgprValuC+146] // check Nan
v_bfe_u32 v125, v[vgprValuC+146], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+146], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+146], v125, v127, s8
v_lshrrev_b32 v146, 16, v[vgprValuC+146]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v146, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+147], v[vgprValuC+147] // check Nan
v_bfe_u32 v125, v[vgprValuC+147], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+147], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+147], v125, v127, s8
v_lshrrev_b32 v147, 16, v[vgprValuC+147]           // convert C to bf16
buffer_store_b16 v147, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+148], v[vgprValuC+148] // check Nan
v_bfe_u32 v125, v[vgprValuC+148], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+148], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+148], v125, v127, s8
v_lshrrev_b32 v148, 16, v[vgprValuC+148]           // convert C to bf16
buffer_store_b16 v148, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+149], v[vgprValuC+149] // check Nan
v_bfe_u32 v125, v[vgprValuC+149], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+149], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+149], v125, v127, s8
v_lshrrev_b32 v149, 16, v[vgprValuC+149]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v149, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+150], v[vgprValuC+150] // check Nan
v_bfe_u32 v125, v[vgprValuC+150], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+150], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+150], v125, v127, s8
v_lshrrev_b32 v150, 16, v[vgprValuC+150]           // convert C to bf16
buffer_store_b16 v150, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+151], v[vgprValuC+151] // check Nan
v_bfe_u32 v125, v[vgprValuC+151], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+151], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+151], v125, v127, s8
v_lshrrev_b32 v151, 16, v[vgprValuC+151]           // convert C to bf16
buffer_store_b16 v151, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+152], v[vgprValuC+152] // check Nan
v_bfe_u32 v125, v[vgprValuC+152], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+152], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+152], v125, v127, s8
v_lshrrev_b32 v152, 16, v[vgprValuC+152]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v152, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+153], v[vgprValuC+153] // check Nan
v_bfe_u32 v125, v[vgprValuC+153], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+153], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+153], v125, v127, s8
v_lshrrev_b32 v153, 16, v[vgprValuC+153]           // convert C to bf16
buffer_store_b16 v153, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+154], v[vgprValuC+154] // check Nan
v_bfe_u32 v125, v[vgprValuC+154], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+154], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+154], v125, v127, s8
v_lshrrev_b32 v154, 16, v[vgprValuC+154]           // convert C to bf16
buffer_store_b16 v154, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+155], v[vgprValuC+155] // check Nan
v_bfe_u32 v125, v[vgprValuC+155], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+155], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+155], v125, v127, s8
v_lshrrev_b32 v155, 16, v[vgprValuC+155]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 100                // scale StrideD *= numRows(50) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(50): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(50): gra SRD += inc(upper)
buffer_store_b16 v155, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+156], v[vgprValuC+156] // check Nan
v_bfe_u32 v125, v[vgprValuC+156], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+156], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+156], v125, v127, s8
v_lshrrev_b32 v156, 16, v[vgprValuC+156]           // convert C to bf16
buffer_store_b16 v156, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+157], v[vgprValuC+157] // check Nan
v_bfe_u32 v125, v[vgprValuC+157], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+157], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+157], v125, v127, s8
v_lshrrev_b32 v157, 16, v[vgprValuC+157]           // convert C to bf16
buffer_store_b16 v157, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+158], v[vgprValuC+158] // check Nan
v_bfe_u32 v125, v[vgprValuC+158], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+158], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+158], v125, v127, s8
v_lshrrev_b32 v158, 16, v[vgprValuC+158]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v158, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+159], v[vgprValuC+159] // check Nan
v_bfe_u32 v125, v[vgprValuC+159], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+159], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+159], v125, v127, s8
v_lshrrev_b32 v159, 16, v[vgprValuC+159]           // convert C to bf16
buffer_store_b16 v159, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+160], v[vgprValuC+160] // check Nan
v_bfe_u32 v125, v[vgprValuC+160], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+160], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+160], v125, v127, s8
v_lshrrev_b32 v160, 16, v[vgprValuC+160]           // convert C to bf16
buffer_store_b16 v160, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+161], v[vgprValuC+161] // check Nan
v_bfe_u32 v125, v[vgprValuC+161], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+161], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+161], v125, v127, s8
v_lshrrev_b32 v161, 16, v[vgprValuC+161]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v161, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+162], v[vgprValuC+162] // check Nan
v_bfe_u32 v125, v[vgprValuC+162], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+162], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+162], v125, v127, s8
v_lshrrev_b32 v162, 16, v[vgprValuC+162]           // convert C to bf16
buffer_store_b16 v162, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+163], v[vgprValuC+163] // check Nan
v_bfe_u32 v125, v[vgprValuC+163], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+163], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+163], v125, v127, s8
v_lshrrev_b32 v163, 16, v[vgprValuC+163]           // convert C to bf16
buffer_store_b16 v163, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+164], v[vgprValuC+164] // check Nan
v_bfe_u32 v125, v[vgprValuC+164], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+164], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+164], v125, v127, s8
v_lshrrev_b32 v164, 16, v[vgprValuC+164]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v164, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+165], v[vgprValuC+165] // check Nan
v_bfe_u32 v125, v[vgprValuC+165], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+165], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+165], v125, v127, s8
v_lshrrev_b32 v165, 16, v[vgprValuC+165]           // convert C to bf16
buffer_store_b16 v165, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+166], v[vgprValuC+166] // check Nan
v_bfe_u32 v125, v[vgprValuC+166], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+166], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+166], v125, v127, s8
v_lshrrev_b32 v166, 16, v[vgprValuC+166]           // convert C to bf16
buffer_store_b16 v166, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+167], v[vgprValuC+167] // check Nan
v_bfe_u32 v125, v[vgprValuC+167], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+167], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+167], v125, v127, s8
v_lshrrev_b32 v167, 16, v[vgprValuC+167]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v167, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+168], v[vgprValuC+168] // check Nan
v_bfe_u32 v125, v[vgprValuC+168], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+168], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+168], v125, v127, s8
v_lshrrev_b32 v168, 16, v[vgprValuC+168]           // convert C to bf16
buffer_store_b16 v168, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+169], v[vgprValuC+169] // check Nan
v_bfe_u32 v125, v[vgprValuC+169], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+169], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+169], v125, v127, s8
v_lshrrev_b32 v169, 16, v[vgprValuC+169]           // convert C to bf16
buffer_store_b16 v169, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+170], v[vgprValuC+170] // check Nan
v_bfe_u32 v125, v[vgprValuC+170], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+170], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+170], v125, v127, s8
v_lshrrev_b32 v170, 16, v[vgprValuC+170]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v170, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+171], v[vgprValuC+171] // check Nan
v_bfe_u32 v125, v[vgprValuC+171], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+171], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+171], v125, v127, s8
v_lshrrev_b32 v171, 16, v[vgprValuC+171]           // convert C to bf16
buffer_store_b16 v171, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+172], v[vgprValuC+172] // check Nan
v_bfe_u32 v125, v[vgprValuC+172], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+172], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+172], v125, v127, s8
v_lshrrev_b32 v172, 16, v[vgprValuC+172]           // convert C to bf16
buffer_store_b16 v172, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+173], v[vgprValuC+173] // check Nan
v_bfe_u32 v125, v[vgprValuC+173], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+173], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+173], v125, v127, s8
v_lshrrev_b32 v173, 16, v[vgprValuC+173]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v173, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+174], v[vgprValuC+174] // check Nan
v_bfe_u32 v125, v[vgprValuC+174], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+174], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+174], v125, v127, s8
v_lshrrev_b32 v174, 16, v[vgprValuC+174]           // convert C to bf16
buffer_store_b16 v174, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+175], v[vgprValuC+175] // check Nan
v_bfe_u32 v125, v[vgprValuC+175], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+175], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+175], v125, v127, s8
v_lshrrev_b32 v175, 16, v[vgprValuC+175]           // convert C to bf16
buffer_store_b16 v175, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+176], v[vgprValuC+176] // check Nan
v_bfe_u32 v125, v[vgprValuC+176], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+176], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+176], v125, v127, s8
v_lshrrev_b32 v176, 16, v[vgprValuC+176]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v176, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+177], v[vgprValuC+177] // check Nan
v_bfe_u32 v125, v[vgprValuC+177], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+177], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+177], v125, v127, s8
v_lshrrev_b32 v177, 16, v[vgprValuC+177]           // convert C to bf16
buffer_store_b16 v177, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+178], v[vgprValuC+178] // check Nan
v_bfe_u32 v125, v[vgprValuC+178], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+178], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+178], v125, v127, s8
v_lshrrev_b32 v178, 16, v[vgprValuC+178]           // convert C to bf16
buffer_store_b16 v178, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+179], v[vgprValuC+179] // check Nan
v_bfe_u32 v125, v[vgprValuC+179], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+179], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+179], v125, v127, s8
v_lshrrev_b32 v179, 16, v[vgprValuC+179]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 100                // scale StrideD *= numRows(50) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(50): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(50): gra SRD += inc(upper)
buffer_store_b16 v179, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+180], v[vgprValuC+180] // check Nan
v_bfe_u32 v125, v[vgprValuC+180], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+180], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+180], v125, v127, s8
v_lshrrev_b32 v180, 16, v[vgprValuC+180]           // convert C to bf16
buffer_store_b16 v180, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+181], v[vgprValuC+181] // check Nan
v_bfe_u32 v125, v[vgprValuC+181], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+181], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+181], v125, v127, s8
v_lshrrev_b32 v181, 16, v[vgprValuC+181]           // convert C to bf16
buffer_store_b16 v181, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+182], v[vgprValuC+182] // check Nan
v_bfe_u32 v125, v[vgprValuC+182], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+182], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+182], v125, v127, s8
v_lshrrev_b32 v182, 16, v[vgprValuC+182]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v182, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+183], v[vgprValuC+183] // check Nan
v_bfe_u32 v125, v[vgprValuC+183], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+183], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+183], v125, v127, s8
v_lshrrev_b32 v183, 16, v[vgprValuC+183]           // convert C to bf16
buffer_store_b16 v183, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+184], v[vgprValuC+184] // check Nan
v_bfe_u32 v125, v[vgprValuC+184], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+184], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+184], v125, v127, s8
v_lshrrev_b32 v184, 16, v[vgprValuC+184]           // convert C to bf16
buffer_store_b16 v184, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+185], v[vgprValuC+185] // check Nan
v_bfe_u32 v125, v[vgprValuC+185], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+185], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+185], v125, v127, s8
v_lshrrev_b32 v185, 16, v[vgprValuC+185]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v185, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+186], v[vgprValuC+186] // check Nan
v_bfe_u32 v125, v[vgprValuC+186], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+186], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+186], v125, v127, s8
v_lshrrev_b32 v186, 16, v[vgprValuC+186]           // convert C to bf16
buffer_store_b16 v186, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+187], v[vgprValuC+187] // check Nan
v_bfe_u32 v125, v[vgprValuC+187], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+187], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+187], v125, v127, s8
v_lshrrev_b32 v187, 16, v[vgprValuC+187]           // convert C to bf16
buffer_store_b16 v187, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+188], v[vgprValuC+188] // check Nan
v_bfe_u32 v125, v[vgprValuC+188], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+188], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+188], v125, v127, s8
v_lshrrev_b32 v188, 16, v[vgprValuC+188]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v188, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+189], v[vgprValuC+189] // check Nan
v_bfe_u32 v125, v[vgprValuC+189], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+189], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+189], v125, v127, s8
v_lshrrev_b32 v189, 16, v[vgprValuC+189]           // convert C to bf16
buffer_store_b16 v189, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+190], v[vgprValuC+190] // check Nan
v_bfe_u32 v125, v[vgprValuC+190], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+190], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+190], v125, v127, s8
v_lshrrev_b32 v190, 16, v[vgprValuC+190]           // convert C to bf16
buffer_store_b16 v190, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+191], v[vgprValuC+191] // check Nan
v_bfe_u32 v125, v[vgprValuC+191], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+191], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+191], v125, v127, s8
v_lshrrev_b32 v191, 16, v[vgprValuC+191]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v191, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+192], v[vgprValuC+192] // check Nan
v_bfe_u32 v125, v[vgprValuC+192], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+192], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+192], v125, v127, s8
v_lshrrev_b32 v192, 16, v[vgprValuC+192]           // convert C to bf16
buffer_store_b16 v192, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+193], v[vgprValuC+193] // check Nan
v_bfe_u32 v125, v[vgprValuC+193], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+193], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+193], v125, v127, s8
v_lshrrev_b32 v193, 16, v[vgprValuC+193]           // convert C to bf16
buffer_store_b16 v193, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+194], v[vgprValuC+194] // check Nan
v_bfe_u32 v125, v[vgprValuC+194], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+194], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+194], v125, v127, s8
v_lshrrev_b32 v194, 16, v[vgprValuC+194]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v194, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+195], v[vgprValuC+195] // check Nan
v_bfe_u32 v125, v[vgprValuC+195], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+195], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+195], v125, v127, s8
v_lshrrev_b32 v195, 16, v[vgprValuC+195]           // convert C to bf16
buffer_store_b16 v195, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+196], v[vgprValuC+196] // check Nan
v_bfe_u32 v125, v[vgprValuC+196], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+196], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+196], v125, v127, s8
v_lshrrev_b32 v196, 16, v[vgprValuC+196]           // convert C to bf16
buffer_store_b16 v196, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+197], v[vgprValuC+197] // check Nan
v_bfe_u32 v125, v[vgprValuC+197], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+197], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+197], v125, v127, s8
v_lshrrev_b32 v197, 16, v[vgprValuC+197]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v197, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+198], v[vgprValuC+198] // check Nan
v_bfe_u32 v125, v[vgprValuC+198], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+198], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+198], v125, v127, s8
v_lshrrev_b32 v198, 16, v[vgprValuC+198]           // convert C to bf16
buffer_store_b16 v198, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+199], v[vgprValuC+199] // check Nan
v_bfe_u32 v125, v[vgprValuC+199], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+199], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+199], v125, v127, s8
v_lshrrev_b32 v199, 16, v[vgprValuC+199]           // convert C to bf16
buffer_store_b16 v199, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+200], v[vgprValuC+200] // check Nan
v_bfe_u32 v125, v[vgprValuC+200], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+200], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+200], v125, v127, s8
v_lshrrev_b32 v200, 16, v[vgprValuC+200]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v200, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+201], v[vgprValuC+201] // check Nan
v_bfe_u32 v125, v[vgprValuC+201], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+201], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+201], v125, v127, s8
v_lshrrev_b32 v201, 16, v[vgprValuC+201]           // convert C to bf16
buffer_store_b16 v201, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+202], v[vgprValuC+202] // check Nan
v_bfe_u32 v125, v[vgprValuC+202], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+202], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+202], v125, v127, s8
v_lshrrev_b32 v202, 16, v[vgprValuC+202]           // convert C to bf16
buffer_store_b16 v202, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+203], v[vgprValuC+203] // check Nan
v_bfe_u32 v125, v[vgprValuC+203], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+203], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+203], v125, v127, s8
v_lshrrev_b32 v203, 16, v[vgprValuC+203]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 100                // scale StrideD *= numRows(50) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(50): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(50): gra SRD += inc(upper)
buffer_store_b16 v203, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+206], v[vgprValuC+206] // check Nan
v_bfe_u32 v125, v[vgprValuC+206], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+206], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+206], v125, v127, s8
v_lshrrev_b32 v206, 16, v[vgprValuC+206]           // convert C to bf16
buffer_store_b16 v206, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+207], v[vgprValuC+207] // check Nan
v_bfe_u32 v125, v[vgprValuC+207], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+207], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+207], v125, v127, s8
v_lshrrev_b32 v207, 16, v[vgprValuC+207]           // convert C to bf16
buffer_store_b16 v207, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+208], v[vgprValuC+208] // check Nan
v_bfe_u32 v125, v[vgprValuC+208], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+208], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+208], v125, v127, s8
v_lshrrev_b32 v208, 16, v[vgprValuC+208]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v208, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+209], v[vgprValuC+209] // check Nan
v_bfe_u32 v125, v[vgprValuC+209], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+209], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+209], v125, v127, s8
v_lshrrev_b32 v209, 16, v[vgprValuC+209]           // convert C to bf16
buffer_store_b16 v209, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+210], v[vgprValuC+210] // check Nan
v_bfe_u32 v125, v[vgprValuC+210], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+210], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+210], v125, v127, s8
v_lshrrev_b32 v210, 16, v[vgprValuC+210]           // convert C to bf16
buffer_store_b16 v210, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+211], v[vgprValuC+211] // check Nan
v_bfe_u32 v125, v[vgprValuC+211], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+211], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+211], v125, v127, s8
v_lshrrev_b32 v211, 16, v[vgprValuC+211]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v211, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+212], v[vgprValuC+212] // check Nan
v_bfe_u32 v125, v[vgprValuC+212], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+212], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+212], v125, v127, s8
v_lshrrev_b32 v212, 16, v[vgprValuC+212]           // convert C to bf16
buffer_store_b16 v212, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+213], v[vgprValuC+213] // check Nan
v_bfe_u32 v125, v[vgprValuC+213], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+213], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+213], v125, v127, s8
v_lshrrev_b32 v213, 16, v[vgprValuC+213]           // convert C to bf16
buffer_store_b16 v213, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+214], v[vgprValuC+214] // check Nan
v_bfe_u32 v125, v[vgprValuC+214], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+214], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+214], v125, v127, s8
v_lshrrev_b32 v214, 16, v[vgprValuC+214]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v214, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+215], v[vgprValuC+215] // check Nan
v_bfe_u32 v125, v[vgprValuC+215], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+215], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+215], v125, v127, s8
v_lshrrev_b32 v215, 16, v[vgprValuC+215]           // convert C to bf16
buffer_store_b16 v215, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+216], v[vgprValuC+216] // check Nan
v_bfe_u32 v125, v[vgprValuC+216], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+216], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+216], v125, v127, s8
v_lshrrev_b32 v216, 16, v[vgprValuC+216]           // convert C to bf16
buffer_store_b16 v216, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+217], v[vgprValuC+217] // check Nan
v_bfe_u32 v125, v[vgprValuC+217], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+217], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+217], v125, v127, s8
v_lshrrev_b32 v217, 16, v[vgprValuC+217]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v217, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+218], v[vgprValuC+218] // check Nan
v_bfe_u32 v125, v[vgprValuC+218], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+218], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+218], v125, v127, s8
v_lshrrev_b32 v218, 16, v[vgprValuC+218]           // convert C to bf16
buffer_store_b16 v218, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+219], v[vgprValuC+219] // check Nan
v_bfe_u32 v125, v[vgprValuC+219], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+219], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+219], v125, v127, s8
v_lshrrev_b32 v219, 16, v[vgprValuC+219]           // convert C to bf16
buffer_store_b16 v219, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+220], v[vgprValuC+220] // check Nan
v_bfe_u32 v125, v[vgprValuC+220], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+220], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+220], v125, v127, s8
v_lshrrev_b32 v220, 16, v[vgprValuC+220]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v220, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+221], v[vgprValuC+221] // check Nan
v_bfe_u32 v125, v[vgprValuC+221], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+221], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+221], v125, v127, s8
v_lshrrev_b32 v221, 16, v[vgprValuC+221]           // convert C to bf16
buffer_store_b16 v221, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+222], v[vgprValuC+222] // check Nan
v_bfe_u32 v125, v[vgprValuC+222], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+222], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+222], v125, v127, s8
v_lshrrev_b32 v222, 16, v[vgprValuC+222]           // convert C to bf16
buffer_store_b16 v222, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+223], v[vgprValuC+223] // check Nan
v_bfe_u32 v125, v[vgprValuC+223], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+223], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+223], v125, v127, s8
v_lshrrev_b32 v223, 16, v[vgprValuC+223]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v223, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+224], v[vgprValuC+224] // check Nan
v_bfe_u32 v125, v[vgprValuC+224], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+224], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+224], v125, v127, s8
v_lshrrev_b32 v224, 16, v[vgprValuC+224]           // convert C to bf16
buffer_store_b16 v224, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+225], v[vgprValuC+225] // check Nan
v_bfe_u32 v125, v[vgprValuC+225], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+225], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+225], v125, v127, s8
v_lshrrev_b32 v225, 16, v[vgprValuC+225]           // convert C to bf16
buffer_store_b16 v225, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+226], v[vgprValuC+226] // check Nan
v_bfe_u32 v125, v[vgprValuC+226], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+226], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+226], v125, v127, s8
v_lshrrev_b32 v226, 16, v[vgprValuC+226]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v226, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+227], v[vgprValuC+227] // check Nan
v_bfe_u32 v125, v[vgprValuC+227], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+227], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+227], v125, v127, s8
v_lshrrev_b32 v227, 16, v[vgprValuC+227]           // convert C to bf16
buffer_store_b16 v227, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+228], v[vgprValuC+228] // check Nan
v_bfe_u32 v125, v[vgprValuC+228], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+228], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+228], v125, v127, s8
v_lshrrev_b32 v228, 16, v[vgprValuC+228]           // convert C to bf16
buffer_store_b16 v228, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW1_GSU1_NonEdgeEnd:
label_GW_B0_FD0_VW1_GSU1_Else:
label_GW_B0_FD0_VW1_GSU1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=62 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */
s_sleep 7                                          // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (8,2,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (9,2,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (10,2,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (11,2,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (12,2,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (13,2,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (14,2,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (15,2,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (16,2,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (17,2,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (18,2,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (19,2,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v124, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v191, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v191, v124, v191, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v192, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v192, v124, v192, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v193, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v193, v124, v193, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v194, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v194, v124, v194, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v195, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v195, v124, v195, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v196, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v196, v124, v196, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v197, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v197, v124, v197, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v198, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v198, v124, v198, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v199, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v199, v124, v199, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v200, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v200, v124, v200, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v201, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v201, v124, v201, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v202, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v202, v124, v202, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v203, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v203, v124, v203, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v206, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v206, v124, v206, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v207, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v207, v124, v207, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v208, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v208, v124, v208, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v209, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v209, v124, v209, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v210, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v210, v124, v210, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v211, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v211, v124, v211, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v212, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v212, v124, v212, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v213, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v213, v124, v213, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v214, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v214, v124, v214, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v215, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v215, v124, v215, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v216, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v216, v124, v216, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 50                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 50                // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 50                // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v217, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v217, v124, v217, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v218, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v218, v124, v218, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v219, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v219, v124, v219, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v220, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v220, v124, v220, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v221, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v221, v124, v221, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v222, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v222, v124, v222, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v223, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v223, v124, v223, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v224, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v224, v124, v224, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v225, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v225, v124, v225, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v226, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v226, v124, v226, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v227, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v227, v124, v227, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v228, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v228, v124, v228, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v229, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v229, v124, v229, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v230, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v230, v124, v230, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v231, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v231, v124, v231, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v232, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v232, v124, v232, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v233, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v233, v124, v233, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v234, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v234, v124, v234, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v235, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v235, v124, v235, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v236, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v236, v124, v236, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v237, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v237, v124, v237, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v238, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v238, v124, v238, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v239, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v239, v124, v239, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v240, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v240, v124, v240, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 50                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 50                // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 50                // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v241, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v241, v124, v241, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v242, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v242, v124, v242, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v243, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v243, v124, v243, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v244, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v244, v124, v244, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v245, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v245, v124, v245, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v246, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v246, v124, v246, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v247, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v247, v124, v247, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v248, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v248, v124, v248, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v249, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v249, v124, v249, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v250, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v250, v124, v250, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v251, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v251, v124, v251, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v252, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v252, v124, v252, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v253, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v253, v124, v253, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v254, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v254, v124, v254, s66                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (5, 2, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (7, 2, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (8, 2, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (9, 2, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (10, 2, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (11, 2, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (12, 2, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (13, 2, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (14, 2, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (15, 2, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (16, 2, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (17, 2, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (18, 2, 0, 0), (19, 0, 0, 0), (19, 1, 0, 0), (19, 2, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+129], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+130], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+131], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+132], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+133], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+134], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+135], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+152], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+154], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+158], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+160], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+162], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+163], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+164], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+166], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+167], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+168], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+169], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+170], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+171], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+172], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+173], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+174], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+175], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+176], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+177], s[sgprAlpha], v[vgprValuC+48] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+178], s[sgprAlpha], v[vgprValuC+56] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+179], s[sgprAlpha], v[vgprValuC+64] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+180], s[sgprAlpha], v[vgprValuC+49] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+181], s[sgprAlpha], v[vgprValuC+57] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+182], s[sgprAlpha], v[vgprValuC+65] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+183], s[sgprAlpha], v[vgprValuC+50] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+184], s[sgprAlpha], v[vgprValuC+58] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+185], s[sgprAlpha], v[vgprValuC+66] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+186], s[sgprAlpha], v[vgprValuC+51] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+187], s[sgprAlpha], v[vgprValuC+59] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+188], s[sgprAlpha], v[vgprValuC+67] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+189], s[sgprAlpha], v[vgprValuC+52] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+190], s[sgprAlpha], v[vgprValuC+60] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v126, 0xffff0000                         // mask for pack two bfloat16 element to 32bit
v_mov_b32 v127, 0x7fff0000                         // fp32 Nan
v_mov_b32 v128, 0x7fff                             // rounding bias for bfloat16
v_cmp_u_f32 s64, v[vgprValuC+129], v[vgprValuC+129] // check Nan
v_bfe_u32 v125, v[vgprValuC+129], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+129], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+129], v125, v127, s64
v_lshrrev_b32 v129, 16, v[vgprValuC+129]           // convert C to bf16
buffer_store_b16 v129, v191, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+130], v[vgprValuC+130] // check Nan
v_bfe_u32 v125, v[vgprValuC+130], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+130], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+130], v125, v127, s64
v_lshrrev_b32 v130, 16, v[vgprValuC+130]           // convert C to bf16
buffer_store_b16 v130, v192, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+131], v[vgprValuC+131] // check Nan
v_bfe_u32 v125, v[vgprValuC+131], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+131], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+131], v125, v127, s64
v_lshrrev_b32 v131, 16, v[vgprValuC+131]           // convert C to bf16
buffer_store_b16 v131, v193, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+132], v[vgprValuC+132] // check Nan
v_bfe_u32 v125, v[vgprValuC+132], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+132], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+132], v125, v127, s64
v_lshrrev_b32 v132, 16, v[vgprValuC+132]           // convert C to bf16
buffer_store_b16 v132, v194, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+133], v[vgprValuC+133] // check Nan
v_bfe_u32 v125, v[vgprValuC+133], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+133], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+133], v125, v127, s64
v_lshrrev_b32 v133, 16, v[vgprValuC+133]           // convert C to bf16
buffer_store_b16 v133, v195, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+134], v[vgprValuC+134] // check Nan
v_bfe_u32 v125, v[vgprValuC+134], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+134], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+134], v125, v127, s64
v_lshrrev_b32 v134, 16, v[vgprValuC+134]           // convert C to bf16
buffer_store_b16 v134, v196, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+135], v[vgprValuC+135] // check Nan
v_bfe_u32 v125, v[vgprValuC+135], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+135], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+135], v125, v127, s64
v_lshrrev_b32 v135, 16, v[vgprValuC+135]           // convert C to bf16
buffer_store_b16 v135, v197, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+136], v[vgprValuC+136] // check Nan
v_bfe_u32 v125, v[vgprValuC+136], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+136], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+136], v125, v127, s64
v_lshrrev_b32 v136, 16, v[vgprValuC+136]           // convert C to bf16
buffer_store_b16 v136, v198, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+137], v[vgprValuC+137] // check Nan
v_bfe_u32 v125, v[vgprValuC+137], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+137], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+137], v125, v127, s64
v_lshrrev_b32 v137, 16, v[vgprValuC+137]           // convert C to bf16
buffer_store_b16 v137, v199, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+138], v[vgprValuC+138] // check Nan
v_bfe_u32 v125, v[vgprValuC+138], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+138], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+138], v125, v127, s64
v_lshrrev_b32 v138, 16, v[vgprValuC+138]           // convert C to bf16
buffer_store_b16 v138, v200, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+139], v[vgprValuC+139] // check Nan
v_bfe_u32 v125, v[vgprValuC+139], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+139], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+139], v125, v127, s64
v_lshrrev_b32 v139, 16, v[vgprValuC+139]           // convert C to bf16
buffer_store_b16 v139, v201, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+140], v[vgprValuC+140] // check Nan
v_bfe_u32 v125, v[vgprValuC+140], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+140], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+140], v125, v127, s64
v_lshrrev_b32 v140, 16, v[vgprValuC+140]           // convert C to bf16
buffer_store_b16 v140, v202, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+141], v[vgprValuC+141] // check Nan
v_bfe_u32 v125, v[vgprValuC+141], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+141], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+141], v125, v127, s64
v_lshrrev_b32 v141, 16, v[vgprValuC+141]           // convert C to bf16
buffer_store_b16 v141, v203, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+142], v[vgprValuC+142] // check Nan
v_bfe_u32 v125, v[vgprValuC+142], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+142], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+142], v125, v127, s64
v_lshrrev_b32 v142, 16, v[vgprValuC+142]           // convert C to bf16
buffer_store_b16 v142, v206, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+143], v[vgprValuC+143] // check Nan
v_bfe_u32 v125, v[vgprValuC+143], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+143], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+143], v125, v127, s64
v_lshrrev_b32 v143, 16, v[vgprValuC+143]           // convert C to bf16
buffer_store_b16 v143, v207, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+144], v[vgprValuC+144] // check Nan
v_bfe_u32 v125, v[vgprValuC+144], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+144], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+144], v125, v127, s64
v_lshrrev_b32 v144, 16, v[vgprValuC+144]           // convert C to bf16
buffer_store_b16 v144, v208, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+145], v[vgprValuC+145] // check Nan
v_bfe_u32 v125, v[vgprValuC+145], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+145], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+145], v125, v127, s64
v_lshrrev_b32 v145, 16, v[vgprValuC+145]           // convert C to bf16
buffer_store_b16 v145, v209, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+146], v[vgprValuC+146] // check Nan
v_bfe_u32 v125, v[vgprValuC+146], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+146], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+146], v125, v127, s64
v_lshrrev_b32 v146, 16, v[vgprValuC+146]           // convert C to bf16
buffer_store_b16 v146, v210, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+147], v[vgprValuC+147] // check Nan
v_bfe_u32 v125, v[vgprValuC+147], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+147], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+147], v125, v127, s64
v_lshrrev_b32 v147, 16, v[vgprValuC+147]           // convert C to bf16
buffer_store_b16 v147, v211, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+148], v[vgprValuC+148] // check Nan
v_bfe_u32 v125, v[vgprValuC+148], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+148], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+148], v125, v127, s64
v_lshrrev_b32 v148, 16, v[vgprValuC+148]           // convert C to bf16
buffer_store_b16 v148, v212, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+149], v[vgprValuC+149] // check Nan
v_bfe_u32 v125, v[vgprValuC+149], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+149], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+149], v125, v127, s64
v_lshrrev_b32 v149, 16, v[vgprValuC+149]           // convert C to bf16
buffer_store_b16 v149, v213, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+150], v[vgprValuC+150] // check Nan
v_bfe_u32 v125, v[vgprValuC+150], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+150], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+150], v125, v127, s64
v_lshrrev_b32 v150, 16, v[vgprValuC+150]           // convert C to bf16
buffer_store_b16 v150, v214, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+151], v[vgprValuC+151] // check Nan
v_bfe_u32 v125, v[vgprValuC+151], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+151], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+151], v125, v127, s64
v_lshrrev_b32 v151, 16, v[vgprValuC+151]           // convert C to bf16
buffer_store_b16 v151, v215, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+152], v[vgprValuC+152] // check Nan
v_bfe_u32 v125, v[vgprValuC+152], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+152], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+152], v125, v127, s64
v_lshrrev_b32 v152, 16, v[vgprValuC+152]           // convert C to bf16
buffer_store_b16 v152, v216, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+153], v[vgprValuC+153] // check Nan
v_bfe_u32 v125, v[vgprValuC+153], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+153], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+153], v125, v127, s64
v_lshrrev_b32 v153, 16, v[vgprValuC+153]           // convert C to bf16
buffer_store_b16 v153, v217, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+154], v[vgprValuC+154] // check Nan
v_bfe_u32 v125, v[vgprValuC+154], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+154], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+154], v125, v127, s64
v_lshrrev_b32 v154, 16, v[vgprValuC+154]           // convert C to bf16
buffer_store_b16 v154, v218, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+155], v[vgprValuC+155] // check Nan
v_bfe_u32 v125, v[vgprValuC+155], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+155], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+155], v125, v127, s64
v_lshrrev_b32 v155, 16, v[vgprValuC+155]           // convert C to bf16
buffer_store_b16 v155, v219, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+156], v[vgprValuC+156] // check Nan
v_bfe_u32 v125, v[vgprValuC+156], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+156], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+156], v125, v127, s64
v_lshrrev_b32 v156, 16, v[vgprValuC+156]           // convert C to bf16
buffer_store_b16 v156, v220, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+157], v[vgprValuC+157] // check Nan
v_bfe_u32 v125, v[vgprValuC+157], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+157], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+157], v125, v127, s64
v_lshrrev_b32 v157, 16, v[vgprValuC+157]           // convert C to bf16
buffer_store_b16 v157, v221, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+158], v[vgprValuC+158] // check Nan
v_bfe_u32 v125, v[vgprValuC+158], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+158], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+158], v125, v127, s64
v_lshrrev_b32 v158, 16, v[vgprValuC+158]           // convert C to bf16
buffer_store_b16 v158, v222, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+159], v[vgprValuC+159] // check Nan
v_bfe_u32 v125, v[vgprValuC+159], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+159], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+159], v125, v127, s64
v_lshrrev_b32 v159, 16, v[vgprValuC+159]           // convert C to bf16
buffer_store_b16 v159, v223, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+160], v[vgprValuC+160] // check Nan
v_bfe_u32 v125, v[vgprValuC+160], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+160], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+160], v125, v127, s64
v_lshrrev_b32 v160, 16, v[vgprValuC+160]           // convert C to bf16
buffer_store_b16 v160, v224, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+161], v[vgprValuC+161] // check Nan
v_bfe_u32 v125, v[vgprValuC+161], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+161], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+161], v125, v127, s64
v_lshrrev_b32 v161, 16, v[vgprValuC+161]           // convert C to bf16
buffer_store_b16 v161, v225, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+162], v[vgprValuC+162] // check Nan
v_bfe_u32 v125, v[vgprValuC+162], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+162], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+162], v125, v127, s64
v_lshrrev_b32 v162, 16, v[vgprValuC+162]           // convert C to bf16
buffer_store_b16 v162, v226, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+163], v[vgprValuC+163] // check Nan
v_bfe_u32 v125, v[vgprValuC+163], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+163], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+163], v125, v127, s64
v_lshrrev_b32 v163, 16, v[vgprValuC+163]           // convert C to bf16
buffer_store_b16 v163, v227, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+164], v[vgprValuC+164] // check Nan
v_bfe_u32 v125, v[vgprValuC+164], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+164], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+164], v125, v127, s64
v_lshrrev_b32 v164, 16, v[vgprValuC+164]           // convert C to bf16
buffer_store_b16 v164, v228, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+165], v[vgprValuC+165] // check Nan
v_bfe_u32 v125, v[vgprValuC+165], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+165], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+165], v125, v127, s64
v_lshrrev_b32 v165, 16, v[vgprValuC+165]           // convert C to bf16
buffer_store_b16 v165, v229, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+166], v[vgprValuC+166] // check Nan
v_bfe_u32 v125, v[vgprValuC+166], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+166], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+166], v125, v127, s64
v_lshrrev_b32 v166, 16, v[vgprValuC+166]           // convert C to bf16
buffer_store_b16 v166, v230, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+167], v[vgprValuC+167] // check Nan
v_bfe_u32 v125, v[vgprValuC+167], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+167], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+167], v125, v127, s64
v_lshrrev_b32 v167, 16, v[vgprValuC+167]           // convert C to bf16
buffer_store_b16 v167, v231, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+168], v[vgprValuC+168] // check Nan
v_bfe_u32 v125, v[vgprValuC+168], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+168], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+168], v125, v127, s64
v_lshrrev_b32 v168, 16, v[vgprValuC+168]           // convert C to bf16
buffer_store_b16 v168, v232, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+169], v[vgprValuC+169] // check Nan
v_bfe_u32 v125, v[vgprValuC+169], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+169], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+169], v125, v127, s64
v_lshrrev_b32 v169, 16, v[vgprValuC+169]           // convert C to bf16
buffer_store_b16 v169, v233, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+170], v[vgprValuC+170] // check Nan
v_bfe_u32 v125, v[vgprValuC+170], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+170], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+170], v125, v127, s64
v_lshrrev_b32 v170, 16, v[vgprValuC+170]           // convert C to bf16
buffer_store_b16 v170, v234, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+171], v[vgprValuC+171] // check Nan
v_bfe_u32 v125, v[vgprValuC+171], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+171], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+171], v125, v127, s64
v_lshrrev_b32 v171, 16, v[vgprValuC+171]           // convert C to bf16
buffer_store_b16 v171, v235, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+172], v[vgprValuC+172] // check Nan
v_bfe_u32 v125, v[vgprValuC+172], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+172], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+172], v125, v127, s64
v_lshrrev_b32 v172, 16, v[vgprValuC+172]           // convert C to bf16
buffer_store_b16 v172, v236, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+173], v[vgprValuC+173] // check Nan
v_bfe_u32 v125, v[vgprValuC+173], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+173], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+173], v125, v127, s64
v_lshrrev_b32 v173, 16, v[vgprValuC+173]           // convert C to bf16
buffer_store_b16 v173, v237, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+174], v[vgprValuC+174] // check Nan
v_bfe_u32 v125, v[vgprValuC+174], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+174], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+174], v125, v127, s64
v_lshrrev_b32 v174, 16, v[vgprValuC+174]           // convert C to bf16
buffer_store_b16 v174, v238, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+175], v[vgprValuC+175] // check Nan
v_bfe_u32 v125, v[vgprValuC+175], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+175], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+175], v125, v127, s64
v_lshrrev_b32 v175, 16, v[vgprValuC+175]           // convert C to bf16
buffer_store_b16 v175, v239, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+176], v[vgprValuC+176] // check Nan
v_bfe_u32 v125, v[vgprValuC+176], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+176], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+176], v125, v127, s64
v_lshrrev_b32 v176, 16, v[vgprValuC+176]           // convert C to bf16
buffer_store_b16 v176, v240, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+177], v[vgprValuC+177] // check Nan
v_bfe_u32 v125, v[vgprValuC+177], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+177], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+177], v125, v127, s64
v_lshrrev_b32 v177, 16, v[vgprValuC+177]           // convert C to bf16
buffer_store_b16 v177, v241, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+178], v[vgprValuC+178] // check Nan
v_bfe_u32 v125, v[vgprValuC+178], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+178], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+178], v125, v127, s64
v_lshrrev_b32 v178, 16, v[vgprValuC+178]           // convert C to bf16
buffer_store_b16 v178, v242, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+179], v[vgprValuC+179] // check Nan
v_bfe_u32 v125, v[vgprValuC+179], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+179], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+179], v125, v127, s64
v_lshrrev_b32 v179, 16, v[vgprValuC+179]           // convert C to bf16
buffer_store_b16 v179, v243, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+180], v[vgprValuC+180] // check Nan
v_bfe_u32 v125, v[vgprValuC+180], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+180], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+180], v125, v127, s64
v_lshrrev_b32 v180, 16, v[vgprValuC+180]           // convert C to bf16
buffer_store_b16 v180, v244, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+181], v[vgprValuC+181] // check Nan
v_bfe_u32 v125, v[vgprValuC+181], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+181], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+181], v125, v127, s64
v_lshrrev_b32 v181, 16, v[vgprValuC+181]           // convert C to bf16
buffer_store_b16 v181, v245, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+182], v[vgprValuC+182] // check Nan
v_bfe_u32 v125, v[vgprValuC+182], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+182], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+182], v125, v127, s64
v_lshrrev_b32 v182, 16, v[vgprValuC+182]           // convert C to bf16
buffer_store_b16 v182, v246, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+183], v[vgprValuC+183] // check Nan
v_bfe_u32 v125, v[vgprValuC+183], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+183], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+183], v125, v127, s64
v_lshrrev_b32 v183, 16, v[vgprValuC+183]           // convert C to bf16
buffer_store_b16 v183, v247, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+184], v[vgprValuC+184] // check Nan
v_bfe_u32 v125, v[vgprValuC+184], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+184], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+184], v125, v127, s64
v_lshrrev_b32 v184, 16, v[vgprValuC+184]           // convert C to bf16
buffer_store_b16 v184, v248, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+185], v[vgprValuC+185] // check Nan
v_bfe_u32 v125, v[vgprValuC+185], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+185], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+185], v125, v127, s64
v_lshrrev_b32 v185, 16, v[vgprValuC+185]           // convert C to bf16
buffer_store_b16 v185, v249, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+186], v[vgprValuC+186] // check Nan
v_bfe_u32 v125, v[vgprValuC+186], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+186], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+186], v125, v127, s64
v_lshrrev_b32 v186, 16, v[vgprValuC+186]           // convert C to bf16
buffer_store_b16 v186, v250, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+187], v[vgprValuC+187] // check Nan
v_bfe_u32 v125, v[vgprValuC+187], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+187], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+187], v125, v127, s64
v_lshrrev_b32 v187, 16, v[vgprValuC+187]           // convert C to bf16
buffer_store_b16 v187, v251, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+188], v[vgprValuC+188] // check Nan
v_bfe_u32 v125, v[vgprValuC+188], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+188], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+188], v125, v127, s64
v_lshrrev_b32 v188, 16, v[vgprValuC+188]           // convert C to bf16
buffer_store_b16 v188, v252, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+189], v[vgprValuC+189] // check Nan
v_bfe_u32 v125, v[vgprValuC+189], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+189], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+189], v125, v127, s64
v_lshrrev_b32 v189, 16, v[vgprValuC+189]           // convert C to bf16
buffer_store_b16 v189, v253, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+190], v[vgprValuC+190] // check Nan
v_bfe_u32 v125, v[vgprValuC+190], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+190], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+190], v125, v127, s64
v_lshrrev_b32 v190, 16, v[vgprValuC+190]           // convert C to bf16
buffer_store_b16 v190, v254, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */
s_sleep 7                                          // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (20,2,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (21,2,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (22,2,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1); (23,2,0,0:vw1); (24,0,0,0:vw1); (24,1,0,0:vw1); (24,2,0,0:vw1); (25,0,0,0:vw1); (25,1,0,0:vw1); (25,2,0,0:vw1); (26,0,0,0:vw1); (26,1,0,0:vw1); (26,2,0,0:vw1); (27,0,0,0:vw1); (27,1,0,0:vw1); (27,2,0,0:vw1); (28,0,0,0:vw1); (28,1,0,0:vw1); (28,2,0,0:vw1); (29,0,0,0:vw1); (29,1,0,0:vw1); (29,2,0,0:vw1); (30,0,0,0:vw1); (30,1,0,0:vw1); (30,2,0,0:vw1); (31,0,0,0:vw1); (31,1,0,0:vw1); (31,2,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v124, BufferOOB
/* (d1,vc1,d0,vc0)=(20,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v163, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v163, v124, v163, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v164, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v164, v124, v164, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v165, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v165, v124, v165, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v166, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v166, v124, v166, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v167, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v167, v124, v167, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v168, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v168, v124, v168, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v169, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v169, v124, v169, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v170, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v170, v124, v170, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v171, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v171, v124, v171, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v172, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v172, v124, v172, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 50                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 50                // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 50                // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v173, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v173, v124, v173, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v174, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v174, v124, v174, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v175, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v175, v124, v175, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v176, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v176, v124, v176, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v177, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v177, v124, v177, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v178, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v178, v124, v178, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v179, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v179, v124, v179, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v180, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v180, v124, v180, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v181, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v181, v124, v181, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v182, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v182, v124, v182, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v183, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v183, v124, v183, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v184, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v184, v124, v184, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v185, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v185, v124, v185, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v186, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v186, v124, v186, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v187, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v187, v124, v187, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v188, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v188, v124, v188, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v189, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v189, v124, v189, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v190, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v190, v124, v190, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v191, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v191, v124, v191, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v192, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v192, v124, v192, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v193, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v193, v124, v193, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v194, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v194, v124, v194, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v195, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v195, v124, v195, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v196, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v196, v124, v196, s66                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(20, 2, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0), (21, 2, 0, 0), (22, 0, 0, 0), (22, 1, 0, 0), (22, 2, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0), (23, 2, 0, 0), (24, 0, 0, 0), (24, 1, 0, 0), (24, 2, 0, 0), (25, 0, 0, 0), (25, 1, 0, 0), (25, 2, 0, 0), (26, 0, 0, 0), (26, 1, 0, 0), (26, 2, 0, 0), (27, 0, 0, 0), (27, 1, 0, 0), (27, 2, 0, 0), (28, 0, 0, 0), (28, 1, 0, 0), (28, 2, 0, 0), (29, 0, 0, 0), (29, 1, 0, 0), (29, 2, 0, 0), (30, 0, 0, 0), (30, 1, 0, 0), (30, 2, 0, 0), (31, 0, 0, 0), (31, 1, 0, 0), (31, 2, 0, 0)] */
v_mul_f32 v[vgprValuC+129], s[sgprAlpha], v[vgprValuC+68] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+130], s[sgprAlpha], v[vgprValuC+53] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+131], s[sgprAlpha], v[vgprValuC+61] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+132], s[sgprAlpha], v[vgprValuC+69] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+133], s[sgprAlpha], v[vgprValuC+54] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+134], s[sgprAlpha], v[vgprValuC+62] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+135], s[sgprAlpha], v[vgprValuC+70] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+55] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+63] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+71] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+72] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+80] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+88] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+73] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+81] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+89] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+74] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+82] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+90] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+75] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+83] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+91] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+76] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+152], s[sgprAlpha], v[vgprValuC+84] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+92] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+154], s[sgprAlpha], v[vgprValuC+77] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+85] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+93] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+78] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+158], s[sgprAlpha], v[vgprValuC+86] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+94] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+160], s[sgprAlpha], v[vgprValuC+79] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+87] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+162], s[sgprAlpha], v[vgprValuC+95] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v126, 0xffff0000                         // mask for pack two bfloat16 element to 32bit
v_mov_b32 v127, 0x7fff0000                         // fp32 Nan
v_mov_b32 v128, 0x7fff                             // rounding bias for bfloat16
v_cmp_u_f32 s64, v[vgprValuC+129], v[vgprValuC+129] // check Nan
v_bfe_u32 v125, v[vgprValuC+129], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+129], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+129], v125, v127, s64
v_lshrrev_b32 v129, 16, v[vgprValuC+129]           // convert C to bf16
buffer_store_b16 v129, v163, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+130], v[vgprValuC+130] // check Nan
v_bfe_u32 v125, v[vgprValuC+130], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+130], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+130], v125, v127, s64
v_lshrrev_b32 v130, 16, v[vgprValuC+130]           // convert C to bf16
buffer_store_b16 v130, v164, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+131], v[vgprValuC+131] // check Nan
v_bfe_u32 v125, v[vgprValuC+131], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+131], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+131], v125, v127, s64
v_lshrrev_b32 v131, 16, v[vgprValuC+131]           // convert C to bf16
buffer_store_b16 v131, v165, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+132], v[vgprValuC+132] // check Nan
v_bfe_u32 v125, v[vgprValuC+132], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+132], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+132], v125, v127, s64
v_lshrrev_b32 v132, 16, v[vgprValuC+132]           // convert C to bf16
buffer_store_b16 v132, v166, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+133], v[vgprValuC+133] // check Nan
v_bfe_u32 v125, v[vgprValuC+133], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+133], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+133], v125, v127, s64
v_lshrrev_b32 v133, 16, v[vgprValuC+133]           // convert C to bf16
buffer_store_b16 v133, v167, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+134], v[vgprValuC+134] // check Nan
v_bfe_u32 v125, v[vgprValuC+134], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+134], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+134], v125, v127, s64
v_lshrrev_b32 v134, 16, v[vgprValuC+134]           // convert C to bf16
buffer_store_b16 v134, v168, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+135], v[vgprValuC+135] // check Nan
v_bfe_u32 v125, v[vgprValuC+135], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+135], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+135], v125, v127, s64
v_lshrrev_b32 v135, 16, v[vgprValuC+135]           // convert C to bf16
buffer_store_b16 v135, v169, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+136], v[vgprValuC+136] // check Nan
v_bfe_u32 v125, v[vgprValuC+136], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+136], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+136], v125, v127, s64
v_lshrrev_b32 v136, 16, v[vgprValuC+136]           // convert C to bf16
buffer_store_b16 v136, v170, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+137], v[vgprValuC+137] // check Nan
v_bfe_u32 v125, v[vgprValuC+137], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+137], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+137], v125, v127, s64
v_lshrrev_b32 v137, 16, v[vgprValuC+137]           // convert C to bf16
buffer_store_b16 v137, v171, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+138], v[vgprValuC+138] // check Nan
v_bfe_u32 v125, v[vgprValuC+138], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+138], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+138], v125, v127, s64
v_lshrrev_b32 v138, 16, v[vgprValuC+138]           // convert C to bf16
buffer_store_b16 v138, v172, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+139], v[vgprValuC+139] // check Nan
v_bfe_u32 v125, v[vgprValuC+139], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+139], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+139], v125, v127, s64
v_lshrrev_b32 v139, 16, v[vgprValuC+139]           // convert C to bf16
buffer_store_b16 v139, v173, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+140], v[vgprValuC+140] // check Nan
v_bfe_u32 v125, v[vgprValuC+140], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+140], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+140], v125, v127, s64
v_lshrrev_b32 v140, 16, v[vgprValuC+140]           // convert C to bf16
buffer_store_b16 v140, v174, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+141], v[vgprValuC+141] // check Nan
v_bfe_u32 v125, v[vgprValuC+141], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+141], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+141], v125, v127, s64
v_lshrrev_b32 v141, 16, v[vgprValuC+141]           // convert C to bf16
buffer_store_b16 v141, v175, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+142], v[vgprValuC+142] // check Nan
v_bfe_u32 v125, v[vgprValuC+142], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+142], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+142], v125, v127, s64
v_lshrrev_b32 v142, 16, v[vgprValuC+142]           // convert C to bf16
buffer_store_b16 v142, v176, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+143], v[vgprValuC+143] // check Nan
v_bfe_u32 v125, v[vgprValuC+143], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+143], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+143], v125, v127, s64
v_lshrrev_b32 v143, 16, v[vgprValuC+143]           // convert C to bf16
buffer_store_b16 v143, v177, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+144], v[vgprValuC+144] // check Nan
v_bfe_u32 v125, v[vgprValuC+144], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+144], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+144], v125, v127, s64
v_lshrrev_b32 v144, 16, v[vgprValuC+144]           // convert C to bf16
buffer_store_b16 v144, v178, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+145], v[vgprValuC+145] // check Nan
v_bfe_u32 v125, v[vgprValuC+145], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+145], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+145], v125, v127, s64
v_lshrrev_b32 v145, 16, v[vgprValuC+145]           // convert C to bf16
buffer_store_b16 v145, v179, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+146], v[vgprValuC+146] // check Nan
v_bfe_u32 v125, v[vgprValuC+146], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+146], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+146], v125, v127, s64
v_lshrrev_b32 v146, 16, v[vgprValuC+146]           // convert C to bf16
buffer_store_b16 v146, v180, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+147], v[vgprValuC+147] // check Nan
v_bfe_u32 v125, v[vgprValuC+147], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+147], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+147], v125, v127, s64
v_lshrrev_b32 v147, 16, v[vgprValuC+147]           // convert C to bf16
buffer_store_b16 v147, v181, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+148], v[vgprValuC+148] // check Nan
v_bfe_u32 v125, v[vgprValuC+148], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+148], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+148], v125, v127, s64
v_lshrrev_b32 v148, 16, v[vgprValuC+148]           // convert C to bf16
buffer_store_b16 v148, v182, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+149], v[vgprValuC+149] // check Nan
v_bfe_u32 v125, v[vgprValuC+149], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+149], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+149], v125, v127, s64
v_lshrrev_b32 v149, 16, v[vgprValuC+149]           // convert C to bf16
buffer_store_b16 v149, v183, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+150], v[vgprValuC+150] // check Nan
v_bfe_u32 v125, v[vgprValuC+150], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+150], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+150], v125, v127, s64
v_lshrrev_b32 v150, 16, v[vgprValuC+150]           // convert C to bf16
buffer_store_b16 v150, v184, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+151], v[vgprValuC+151] // check Nan
v_bfe_u32 v125, v[vgprValuC+151], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+151], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+151], v125, v127, s64
v_lshrrev_b32 v151, 16, v[vgprValuC+151]           // convert C to bf16
buffer_store_b16 v151, v185, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+152], v[vgprValuC+152] // check Nan
v_bfe_u32 v125, v[vgprValuC+152], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+152], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+152], v125, v127, s64
v_lshrrev_b32 v152, 16, v[vgprValuC+152]           // convert C to bf16
buffer_store_b16 v152, v186, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+153], v[vgprValuC+153] // check Nan
v_bfe_u32 v125, v[vgprValuC+153], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+153], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+153], v125, v127, s64
v_lshrrev_b32 v153, 16, v[vgprValuC+153]           // convert C to bf16
buffer_store_b16 v153, v187, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+154], v[vgprValuC+154] // check Nan
v_bfe_u32 v125, v[vgprValuC+154], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+154], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+154], v125, v127, s64
v_lshrrev_b32 v154, 16, v[vgprValuC+154]           // convert C to bf16
buffer_store_b16 v154, v188, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+155], v[vgprValuC+155] // check Nan
v_bfe_u32 v125, v[vgprValuC+155], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+155], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+155], v125, v127, s64
v_lshrrev_b32 v155, 16, v[vgprValuC+155]           // convert C to bf16
buffer_store_b16 v155, v189, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+156], v[vgprValuC+156] // check Nan
v_bfe_u32 v125, v[vgprValuC+156], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+156], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+156], v125, v127, s64
v_lshrrev_b32 v156, 16, v[vgprValuC+156]           // convert C to bf16
buffer_store_b16 v156, v190, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+157], v[vgprValuC+157] // check Nan
v_bfe_u32 v125, v[vgprValuC+157], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+157], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+157], v125, v127, s64
v_lshrrev_b32 v157, 16, v[vgprValuC+157]           // convert C to bf16
buffer_store_b16 v157, v191, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+158], v[vgprValuC+158] // check Nan
v_bfe_u32 v125, v[vgprValuC+158], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+158], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+158], v125, v127, s64
v_lshrrev_b32 v158, 16, v[vgprValuC+158]           // convert C to bf16
buffer_store_b16 v158, v192, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+159], v[vgprValuC+159] // check Nan
v_bfe_u32 v125, v[vgprValuC+159], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+159], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+159], v125, v127, s64
v_lshrrev_b32 v159, 16, v[vgprValuC+159]           // convert C to bf16
buffer_store_b16 v159, v193, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+160], v[vgprValuC+160] // check Nan
v_bfe_u32 v125, v[vgprValuC+160], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+160], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+160], v125, v127, s64
v_lshrrev_b32 v160, 16, v[vgprValuC+160]           // convert C to bf16
buffer_store_b16 v160, v194, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+161], v[vgprValuC+161] // check Nan
v_bfe_u32 v125, v[vgprValuC+161], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+161], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+161], v125, v127, s64
v_lshrrev_b32 v161, 16, v[vgprValuC+161]           // convert C to bf16
buffer_store_b16 v161, v195, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s64, v[vgprValuC+162], v[vgprValuC+162] // check Nan
v_bfe_u32 v125, v[vgprValuC+162], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+162], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+162], v125, v127, s64
v_lshrrev_b32 v162, 16, v[vgprValuC+162]           // convert C to bf16
buffer_store_b16 v162, v196, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_GSU1:
label_GW_B1_FD0_GSU1:

/* Edge/NonEdge store path check (M): Size % 192 > 0 -> Edge store; else -> NonEdge store */
s_mul_hi_u32 s67, s[sgprSizeI], 44739243           // STATIC_DIV: divisor=192
s_mul_i32 s66, s[sgprSizeI], 44739243              // tmp = dividend * magic
s_lshr_b64 s[66:67], s[66:67], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s65, s66                                 // quotient
s_mul_i32 s66, s65, 192                            // quotient*divisor
s_sub_u32 s64, s[sgprSizeI], s66                   // rReg = dividend - quotient*divisor
s_add_u32 s65, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s65                // wg0 >= nwg0-1 ?
s_cselect_b32 s64, s64, 0                          // set rem
s_cmpk_gt_u32 s64, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW1_GSU1_Else       // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 256 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s64, 255, s[sgprSizeJ]                   // s64 = s[sgprSizeJ] % 256
s_add_u32 s65, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s65                // wg1 >= nwg1-1
s_cselect_b32 s64, s64, 0                          // set rem
s_cmpk_gt_u32 s64, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW1_GSU1_Then       // jump if edges required
label_GW_B1_FD0_VW1_GSU1_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=60 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */
s_sleep 7                                          // optimization: sync and wait
s_barrier

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (5, 2, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (7, 2, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (8, 2, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (9, 2, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (10, 2, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (11, 2, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (12, 2, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (13, 2, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (14, 2, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (15, 2, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (16, 2, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (17, 2, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (18, 2, 0, 0), (19, 0, 0, 0), (19, 1, 0, 0), (19, 2, 0, 0)] */
v_mul_f32 v[vgprValuC+131], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+132], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+133], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+134], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+135], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+152], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+154], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+158], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+160], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+162], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+163], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+164], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+166], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+167], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+168], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+169], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+170], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+171], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+172], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+173], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+174], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+175], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+176], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+177], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+178], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+179], s[sgprAlpha], v[vgprValuC+48] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+180], s[sgprAlpha], v[vgprValuC+56] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+181], s[sgprAlpha], v[vgprValuC+64] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+182], s[sgprAlpha], v[vgprValuC+49] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+183], s[sgprAlpha], v[vgprValuC+57] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+184], s[sgprAlpha], v[vgprValuC+65] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+185], s[sgprAlpha], v[vgprValuC+50] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+186], s[sgprAlpha], v[vgprValuC+58] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+187], s[sgprAlpha], v[vgprValuC+66] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+188], s[sgprAlpha], v[vgprValuC+51] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+189], s[sgprAlpha], v[vgprValuC+59] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+190], s[sgprAlpha], v[vgprValuC+67] // Multiply MI out reg with alpha

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (8,2,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (9,2,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (10,2,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (11,2,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (12,2,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (13,2,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (14,2,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (15,2,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (16,2,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (17,2,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (18,2,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (19,2,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v130, v120, v118, 1                 // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=118, coord0Vgpr=118 (multiple bpe)
buffer_load_d16_b16 v191, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
buffer_load_d16_b16 v192, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
buffer_load_d16_b16 v193, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v194, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
buffer_load_d16_b16 v195, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
buffer_load_d16_b16 v196, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v197, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
buffer_load_d16_b16 v198, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
buffer_load_d16_b16 v199, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v200, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
buffer_load_d16_b16 v201, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
buffer_load_d16_b16 v202, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v203, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
buffer_load_d16_b16 v206, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(4,0,2,0) */
buffer_load_d16_b16 v207, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v208, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
buffer_load_d16_b16 v209, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(5,0,2,0) */
buffer_load_d16_b16 v210, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v211, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
buffer_load_d16_b16 v212, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(6,0,2,0) */
buffer_load_d16_b16 v213, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v214, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
buffer_load_d16_b16 v215, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(7,0,2,0) */
buffer_load_d16_b16 v216, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 100                // scale StrideC *= numRows(50) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(50): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(50): gra SRD += inc(upper)
buffer_load_d16_b16 v217, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
buffer_load_d16_b16 v218, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(8,0,2,0) */
buffer_load_d16_b16 v219, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v220, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(9,0,1,0) */
buffer_load_d16_b16 v221, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(9,0,2,0) */
buffer_load_d16_b16 v222, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v223, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(10,0,1,0) */
buffer_load_d16_b16 v224, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(10,0,2,0) */
buffer_load_d16_b16 v225, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v226, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(11,0,1,0) */
buffer_load_d16_b16 v227, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(11,0,2,0) */
buffer_load_d16_b16 v228, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v229, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
buffer_load_d16_b16 v230, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(12,0,2,0) */
buffer_load_d16_b16 v231, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v232, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(13,0,1,0) */
buffer_load_d16_b16 v233, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(13,0,2,0) */
buffer_load_d16_b16 v234, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v235, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(14,0,1,0) */
buffer_load_d16_b16 v236, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(14,0,2,0) */
buffer_load_d16_b16 v237, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v238, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(15,0,1,0) */
buffer_load_d16_b16 v239, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(15,0,2,0) */
buffer_load_d16_b16 v240, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 100                // scale StrideC *= numRows(50) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(50): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(50): gra SRD += inc(upper)
buffer_load_d16_b16 v241, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
buffer_load_d16_b16 v242, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(16,0,2,0) */
buffer_load_d16_b16 v243, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v244, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(17,0,1,0) */
buffer_load_d16_b16 v245, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(17,0,2,0) */
buffer_load_d16_b16 v246, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v247, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(18,0,1,0) */
buffer_load_d16_b16 v248, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(18,0,2,0) */
buffer_load_d16_b16 v249, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v250, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(19,0,1,0) */
buffer_load_d16_b16 v251, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(19,0,2,0) */
buffer_load_d16_b16 v252, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
v_add_lshl_u32 v129, v121, v118, 1                 // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=118, coord0Vgpr=118 (multiple bpe)
s_sleep 7                                          // optimization: sync and wait
s_barrier

/* apply mask, calc new C and issue writes */
v_mov_b32 v126, 0xffff0000                         // mask for pack two bfloat16 element to 32bit
v_mov_b32 v127, 0x7fff0000                         // fp32 Nan
v_mov_b32 v128, 0x7fff                             // rounding bias for bfloat16

s_waitcnt vmcnt(59)                                // vlcnt(59) = 60 - 1 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v191                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+131], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+131], v[vgprValuC+131] // check Nan
v_bfe_u32 v125, v[vgprValuC+131], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+131], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+131], v125, v127, s8
v_lshrrev_b32 v131, 16, v[vgprValuC+131]           // convert C to bf16
buffer_store_b16 v131, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(58)                                // vlcnt(58) = 60 - 2 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v192                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+132], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+132], v[vgprValuC+132] // check Nan
v_bfe_u32 v125, v[vgprValuC+132], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+132], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+132], v125, v127, s8
v_lshrrev_b32 v132, 16, v[vgprValuC+132]           // convert C to bf16
buffer_store_b16 v132, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(57)                                // vlcnt(57) = 60 - 3 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v193                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+133], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+133], v[vgprValuC+133] // check Nan
v_bfe_u32 v125, v[vgprValuC+133], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+133], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+133], v125, v127, s8
v_lshrrev_b32 v133, 16, v[vgprValuC+133]           // convert C to bf16
buffer_store_b16 v133, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(56)                                // vlcnt(56) = 60 - 4 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v194                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+134], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+134], v[vgprValuC+134] // check Nan
v_bfe_u32 v125, v[vgprValuC+134], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+134], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+134], v125, v127, s8
v_lshrrev_b32 v134, 16, v[vgprValuC+134]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v134, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(55)                                // vlcnt(55) = 60 - 5 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v195                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+135], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+135], v[vgprValuC+135] // check Nan
v_bfe_u32 v125, v[vgprValuC+135], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+135], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+135], v125, v127, s8
v_lshrrev_b32 v135, 16, v[vgprValuC+135]           // convert C to bf16
buffer_store_b16 v135, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(54)                                // vlcnt(54) = 60 - 6 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v196                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+136], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+136], v[vgprValuC+136] // check Nan
v_bfe_u32 v125, v[vgprValuC+136], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+136], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+136], v125, v127, s8
v_lshrrev_b32 v136, 16, v[vgprValuC+136]           // convert C to bf16
buffer_store_b16 v136, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(53)                                // vlcnt(53) = 60 - 7 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v197                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+137], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+137], v[vgprValuC+137] // check Nan
v_bfe_u32 v125, v[vgprValuC+137], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+137], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+137], v125, v127, s8
v_lshrrev_b32 v137, 16, v[vgprValuC+137]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v137, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(52)                                // vlcnt(52) = 60 - 8 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v198                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+138], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+138], v[vgprValuC+138] // check Nan
v_bfe_u32 v125, v[vgprValuC+138], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+138], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+138], v125, v127, s8
v_lshrrev_b32 v138, 16, v[vgprValuC+138]           // convert C to bf16
buffer_store_b16 v138, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(51)                                // vlcnt(51) = 60 - 9 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v199                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+139], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+139], v[vgprValuC+139] // check Nan
v_bfe_u32 v125, v[vgprValuC+139], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+139], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+139], v125, v127, s8
v_lshrrev_b32 v139, 16, v[vgprValuC+139]           // convert C to bf16
buffer_store_b16 v139, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(50)                                // vlcnt(50) = 60 - 10 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v200                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+140], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+140], v[vgprValuC+140] // check Nan
v_bfe_u32 v125, v[vgprValuC+140], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+140], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+140], v125, v127, s8
v_lshrrev_b32 v140, 16, v[vgprValuC+140]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v140, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(49)                                // vlcnt(49) = 60 - 11 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v201                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+141], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+141], v[vgprValuC+141] // check Nan
v_bfe_u32 v125, v[vgprValuC+141], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+141], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+141], v125, v127, s8
v_lshrrev_b32 v141, 16, v[vgprValuC+141]           // convert C to bf16
buffer_store_b16 v141, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(48)                                // vlcnt(48) = 60 - 12 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v202                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+142], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+142], v[vgprValuC+142] // check Nan
v_bfe_u32 v125, v[vgprValuC+142], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+142], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+142], v125, v127, s8
v_lshrrev_b32 v142, 16, v[vgprValuC+142]           // convert C to bf16
buffer_store_b16 v142, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(47)                                // vlcnt(47) = 60 - 13 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v203                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+143], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+143], v[vgprValuC+143] // check Nan
v_bfe_u32 v125, v[vgprValuC+143], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+143], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+143], v125, v127, s8
v_lshrrev_b32 v143, 16, v[vgprValuC+143]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v143, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(46)                                // vlcnt(46) = 60 - 14 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v206                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+144], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+144], v[vgprValuC+144] // check Nan
v_bfe_u32 v125, v[vgprValuC+144], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+144], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+144], v125, v127, s8
v_lshrrev_b32 v144, 16, v[vgprValuC+144]           // convert C to bf16
buffer_store_b16 v144, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(45)                                // vlcnt(45) = 60 - 15 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v207                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+145], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+145], v[vgprValuC+145] // check Nan
v_bfe_u32 v125, v[vgprValuC+145], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+145], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+145], v125, v127, s8
v_lshrrev_b32 v145, 16, v[vgprValuC+145]           // convert C to bf16
buffer_store_b16 v145, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(44)                                // vlcnt(44) = 60 - 16 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v208                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+146], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+146], v[vgprValuC+146] // check Nan
v_bfe_u32 v125, v[vgprValuC+146], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+146], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+146], v125, v127, s8
v_lshrrev_b32 v146, 16, v[vgprValuC+146]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v146, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(43)                                // vlcnt(43) = 60 - 17 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v209                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+147], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+147], v[vgprValuC+147] // check Nan
v_bfe_u32 v125, v[vgprValuC+147], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+147], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+147], v125, v127, s8
v_lshrrev_b32 v147, 16, v[vgprValuC+147]           // convert C to bf16
buffer_store_b16 v147, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(42)                                // vlcnt(42) = 60 - 18 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v210                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+148], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+148], v[vgprValuC+148] // check Nan
v_bfe_u32 v125, v[vgprValuC+148], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+148], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+148], v125, v127, s8
v_lshrrev_b32 v148, 16, v[vgprValuC+148]           // convert C to bf16
buffer_store_b16 v148, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(41)                                // vlcnt(41) = 60 - 19 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v211                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+149], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+149], v[vgprValuC+149] // check Nan
v_bfe_u32 v125, v[vgprValuC+149], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+149], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+149], v125, v127, s8
v_lshrrev_b32 v149, 16, v[vgprValuC+149]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v149, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(40)                                // vlcnt(40) = 60 - 20 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v212                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+150], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+150], v[vgprValuC+150] // check Nan
v_bfe_u32 v125, v[vgprValuC+150], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+150], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+150], v125, v127, s8
v_lshrrev_b32 v150, 16, v[vgprValuC+150]           // convert C to bf16
buffer_store_b16 v150, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(39)                                // vlcnt(39) = 60 - 21 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v213                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+151], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+151], v[vgprValuC+151] // check Nan
v_bfe_u32 v125, v[vgprValuC+151], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+151], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+151], v125, v127, s8
v_lshrrev_b32 v151, 16, v[vgprValuC+151]           // convert C to bf16
buffer_store_b16 v151, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(38)                                // vlcnt(38) = 60 - 22 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v214                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+152], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+152], v[vgprValuC+152] // check Nan
v_bfe_u32 v125, v[vgprValuC+152], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+152], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+152], v125, v127, s8
v_lshrrev_b32 v152, 16, v[vgprValuC+152]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v152, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(37)                                // vlcnt(37) = 60 - 23 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v215                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+153], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+153], v[vgprValuC+153] // check Nan
v_bfe_u32 v125, v[vgprValuC+153], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+153], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+153], v125, v127, s8
v_lshrrev_b32 v153, 16, v[vgprValuC+153]           // convert C to bf16
buffer_store_b16 v153, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(36)                                // vlcnt(36) = 60 - 24 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v216                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+154], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+154], v[vgprValuC+154] // check Nan
v_bfe_u32 v125, v[vgprValuC+154], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+154], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+154], v125, v127, s8
v_lshrrev_b32 v154, 16, v[vgprValuC+154]           // convert C to bf16
buffer_store_b16 v154, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(35)                                // vlcnt(35) = 60 - 25 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v217                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+155], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+155], v[vgprValuC+155] // check Nan
v_bfe_u32 v125, v[vgprValuC+155], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+155], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+155], v125, v127, s8
v_lshrrev_b32 v155, 16, v[vgprValuC+155]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 100                // scale StrideD *= numRows(50) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(50): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(50): gra SRD += inc(upper)
buffer_store_b16 v155, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(34)                                // vlcnt(34) = 60 - 26 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v218                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+156], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+156], v[vgprValuC+156] // check Nan
v_bfe_u32 v125, v[vgprValuC+156], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+156], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+156], v125, v127, s8
v_lshrrev_b32 v156, 16, v[vgprValuC+156]           // convert C to bf16
buffer_store_b16 v156, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(33)                                // vlcnt(33) = 60 - 27 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v219                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+157], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+157], v[vgprValuC+157] // check Nan
v_bfe_u32 v125, v[vgprValuC+157], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+157], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+157], v125, v127, s8
v_lshrrev_b32 v157, 16, v[vgprValuC+157]           // convert C to bf16
buffer_store_b16 v157, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(32)                                // vlcnt(32) = 60 - 28 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v220                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+158], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+158], v[vgprValuC+158] // check Nan
v_bfe_u32 v125, v[vgprValuC+158], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+158], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+158], v125, v127, s8
v_lshrrev_b32 v158, 16, v[vgprValuC+158]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v158, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(31)                                // vlcnt(31) = 60 - 29 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v221                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+159], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+159], v[vgprValuC+159] // check Nan
v_bfe_u32 v125, v[vgprValuC+159], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+159], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+159], v125, v127, s8
v_lshrrev_b32 v159, 16, v[vgprValuC+159]           // convert C to bf16
buffer_store_b16 v159, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(30)                                // vlcnt(30) = 60 - 30 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v222                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+160], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+160], v[vgprValuC+160] // check Nan
v_bfe_u32 v125, v[vgprValuC+160], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+160], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+160], v125, v127, s8
v_lshrrev_b32 v160, 16, v[vgprValuC+160]           // convert C to bf16
buffer_store_b16 v160, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(29)                                // vlcnt(29) = 60 - 31 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v223                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+161], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+161], v[vgprValuC+161] // check Nan
v_bfe_u32 v125, v[vgprValuC+161], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+161], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+161], v125, v127, s8
v_lshrrev_b32 v161, 16, v[vgprValuC+161]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v161, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(28)                                // vlcnt(28) = 60 - 32 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v224                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+162], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+162], v[vgprValuC+162] // check Nan
v_bfe_u32 v125, v[vgprValuC+162], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+162], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+162], v125, v127, s8
v_lshrrev_b32 v162, 16, v[vgprValuC+162]           // convert C to bf16
buffer_store_b16 v162, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(27)                                // vlcnt(27) = 60 - 33 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v225                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+163], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+163], v[vgprValuC+163] // check Nan
v_bfe_u32 v125, v[vgprValuC+163], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+163], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+163], v125, v127, s8
v_lshrrev_b32 v163, 16, v[vgprValuC+163]           // convert C to bf16
buffer_store_b16 v163, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(26)                                // vlcnt(26) = 60 - 34 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v226                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+164], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+164], v[vgprValuC+164] // check Nan
v_bfe_u32 v125, v[vgprValuC+164], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+164], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+164], v125, v127, s8
v_lshrrev_b32 v164, 16, v[vgprValuC+164]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v164, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(25)                                // vlcnt(25) = 60 - 35 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v227                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+165], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+165], v[vgprValuC+165] // check Nan
v_bfe_u32 v125, v[vgprValuC+165], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+165], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+165], v125, v127, s8
v_lshrrev_b32 v165, 16, v[vgprValuC+165]           // convert C to bf16
buffer_store_b16 v165, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(24)                                // vlcnt(24) = 60 - 36 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v228                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+166], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+166], v[vgprValuC+166] // check Nan
v_bfe_u32 v125, v[vgprValuC+166], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+166], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+166], v125, v127, s8
v_lshrrev_b32 v166, 16, v[vgprValuC+166]           // convert C to bf16
buffer_store_b16 v166, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(23)                                // vlcnt(23) = 60 - 37 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v229                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+167], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+167], v[vgprValuC+167] // check Nan
v_bfe_u32 v125, v[vgprValuC+167], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+167], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+167], v125, v127, s8
v_lshrrev_b32 v167, 16, v[vgprValuC+167]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v167, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(22)                                // vlcnt(22) = 60 - 38 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v230                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+168], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+168], v[vgprValuC+168] // check Nan
v_bfe_u32 v125, v[vgprValuC+168], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+168], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+168], v125, v127, s8
v_lshrrev_b32 v168, 16, v[vgprValuC+168]           // convert C to bf16
buffer_store_b16 v168, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(21)                                // vlcnt(21) = 60 - 39 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v231                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+169], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+169], v[vgprValuC+169] // check Nan
v_bfe_u32 v125, v[vgprValuC+169], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+169], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+169], v125, v127, s8
v_lshrrev_b32 v169, 16, v[vgprValuC+169]           // convert C to bf16
buffer_store_b16 v169, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(20)                                // vlcnt(20) = 60 - 40 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v232                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+170], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+170], v[vgprValuC+170] // check Nan
v_bfe_u32 v125, v[vgprValuC+170], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+170], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+170], v125, v127, s8
v_lshrrev_b32 v170, 16, v[vgprValuC+170]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v170, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(19)                                // vlcnt(19) = 60 - 41 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v233                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+171], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+171], v[vgprValuC+171] // check Nan
v_bfe_u32 v125, v[vgprValuC+171], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+171], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+171], v125, v127, s8
v_lshrrev_b32 v171, 16, v[vgprValuC+171]           // convert C to bf16
buffer_store_b16 v171, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(18)                                // vlcnt(18) = 60 - 42 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v234                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+172], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+172], v[vgprValuC+172] // check Nan
v_bfe_u32 v125, v[vgprValuC+172], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+172], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+172], v125, v127, s8
v_lshrrev_b32 v172, 16, v[vgprValuC+172]           // convert C to bf16
buffer_store_b16 v172, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(17)                                // vlcnt(17) = 60 - 43 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v235                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+173], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+173], v[vgprValuC+173] // check Nan
v_bfe_u32 v125, v[vgprValuC+173], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+173], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+173], v125, v127, s8
v_lshrrev_b32 v173, 16, v[vgprValuC+173]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v173, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(16)                                // vlcnt(16) = 60 - 44 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v236                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+174], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+174], v[vgprValuC+174] // check Nan
v_bfe_u32 v125, v[vgprValuC+174], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+174], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+174], v125, v127, s8
v_lshrrev_b32 v174, 16, v[vgprValuC+174]           // convert C to bf16
buffer_store_b16 v174, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(15)                                // vlcnt(15) = 60 - 45 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v237                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+175], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+175], v[vgprValuC+175] // check Nan
v_bfe_u32 v125, v[vgprValuC+175], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+175], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+175], v125, v127, s8
v_lshrrev_b32 v175, 16, v[vgprValuC+175]           // convert C to bf16
buffer_store_b16 v175, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(14)                                // vlcnt(14) = 60 - 46 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v238                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+176], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+176], v[vgprValuC+176] // check Nan
v_bfe_u32 v125, v[vgprValuC+176], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+176], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+176], v125, v127, s8
v_lshrrev_b32 v176, 16, v[vgprValuC+176]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v176, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(13)                                // vlcnt(13) = 60 - 47 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v239                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+177], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+177], v[vgprValuC+177] // check Nan
v_bfe_u32 v125, v[vgprValuC+177], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+177], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+177], v125, v127, s8
v_lshrrev_b32 v177, 16, v[vgprValuC+177]           // convert C to bf16
buffer_store_b16 v177, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(12)                                // vlcnt(12) = 60 - 48 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v240                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+178], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+178], v[vgprValuC+178] // check Nan
v_bfe_u32 v125, v[vgprValuC+178], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+178], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+178], v125, v127, s8
v_lshrrev_b32 v178, 16, v[vgprValuC+178]           // convert C to bf16
buffer_store_b16 v178, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(11)                                // vlcnt(11) = 60 - 49 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v241                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+179], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+179], v[vgprValuC+179] // check Nan
v_bfe_u32 v125, v[vgprValuC+179], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+179], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+179], v125, v127, s8
v_lshrrev_b32 v179, 16, v[vgprValuC+179]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 100                // scale StrideD *= numRows(50) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(50): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(50): gra SRD += inc(upper)
buffer_store_b16 v179, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(10)                                // vlcnt(10) = 60 - 50 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v242                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+180], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+180], v[vgprValuC+180] // check Nan
v_bfe_u32 v125, v[vgprValuC+180], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+180], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+180], v125, v127, s8
v_lshrrev_b32 v180, 16, v[vgprValuC+180]           // convert C to bf16
buffer_store_b16 v180, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(9)                                 // vlcnt(9) = 60 - 51 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v243                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+181], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+181], v[vgprValuC+181] // check Nan
v_bfe_u32 v125, v[vgprValuC+181], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+181], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+181], v125, v127, s8
v_lshrrev_b32 v181, 16, v[vgprValuC+181]           // convert C to bf16
buffer_store_b16 v181, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(8)                                 // vlcnt(8) = 60 - 52 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v244                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+182], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+182], v[vgprValuC+182] // check Nan
v_bfe_u32 v125, v[vgprValuC+182], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+182], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+182], v125, v127, s8
v_lshrrev_b32 v182, 16, v[vgprValuC+182]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v182, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(7)                                 // vlcnt(7) = 60 - 53 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v245                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+183], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+183], v[vgprValuC+183] // check Nan
v_bfe_u32 v125, v[vgprValuC+183], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+183], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+183], v125, v127, s8
v_lshrrev_b32 v183, 16, v[vgprValuC+183]           // convert C to bf16
buffer_store_b16 v183, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(6)                                 // vlcnt(6) = 60 - 54 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v246                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+184], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+184], v[vgprValuC+184] // check Nan
v_bfe_u32 v125, v[vgprValuC+184], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+184], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+184], v125, v127, s8
v_lshrrev_b32 v184, 16, v[vgprValuC+184]           // convert C to bf16
buffer_store_b16 v184, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(5)                                 // vlcnt(5) = 60 - 55 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v247                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+185], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+185], v[vgprValuC+185] // check Nan
v_bfe_u32 v125, v[vgprValuC+185], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+185], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+185], v125, v127, s8
v_lshrrev_b32 v185, 16, v[vgprValuC+185]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v185, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(4)                                 // vlcnt(4) = 60 - 56 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v248                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+186], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+186], v[vgprValuC+186] // check Nan
v_bfe_u32 v125, v[vgprValuC+186], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+186], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+186], v125, v127, s8
v_lshrrev_b32 v186, 16, v[vgprValuC+186]           // convert C to bf16
buffer_store_b16 v186, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(3)                                 // vlcnt(3) = 60 - 57 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v249                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+187], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+187], v[vgprValuC+187] // check Nan
v_bfe_u32 v125, v[vgprValuC+187], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+187], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+187], v125, v127, s8
v_lshrrev_b32 v187, 16, v[vgprValuC+187]           // convert C to bf16
buffer_store_b16 v187, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(2)                                 // vlcnt(2) = 60 - 58 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v250                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+188], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+188], v[vgprValuC+188] // check Nan
v_bfe_u32 v125, v[vgprValuC+188], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+188], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+188], v125, v127, s8
v_lshrrev_b32 v188, 16, v[vgprValuC+188]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v188, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(1)                                 // vlcnt(1) = 60 - 59 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v251                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+189], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+189], v[vgprValuC+189] // check Nan
v_bfe_u32 v125, v[vgprValuC+189], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+189], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+189], v125, v127, s8
v_lshrrev_b32 v189, 16, v[vgprValuC+189]           // convert C to bf16
buffer_store_b16 v189, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(0)                                 // vlcnt(0) = 60 - 60 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v252                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+190], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+190], v[vgprValuC+190] // check Nan
v_bfe_u32 v125, v[vgprValuC+190], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+190], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+190], v125, v127, s8
v_lshrrev_b32 v190, 16, v[vgprValuC+190]           // convert C to bf16
buffer_store_b16 v190, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */
s_sleep 7                                          // optimization: sync and wait
s_barrier

/* rC *= alpha batchElements=[(20, 0, 0, 0), (20, 1, 0, 0), (20, 2, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0), (21, 2, 0, 0), (22, 0, 0, 0), (22, 1, 0, 0), (22, 2, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0), (23, 2, 0, 0), (24, 0, 0, 0), (24, 1, 0, 0), (24, 2, 0, 0), (25, 0, 0, 0), (25, 1, 0, 0), (25, 2, 0, 0), (26, 0, 0, 0), (26, 1, 0, 0), (26, 2, 0, 0), (27, 0, 0, 0), (27, 1, 0, 0), (27, 2, 0, 0), (28, 0, 0, 0), (28, 1, 0, 0), (28, 2, 0, 0), (29, 0, 0, 0), (29, 1, 0, 0), (29, 2, 0, 0), (30, 0, 0, 0), (30, 1, 0, 0), (30, 2, 0, 0), (31, 0, 0, 0), (31, 1, 0, 0), (31, 2, 0, 0)] */
v_mul_f32 v[vgprValuC+131], s[sgprAlpha], v[vgprValuC+52] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+132], s[sgprAlpha], v[vgprValuC+60] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+133], s[sgprAlpha], v[vgprValuC+68] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+134], s[sgprAlpha], v[vgprValuC+53] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+135], s[sgprAlpha], v[vgprValuC+61] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+69] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+54] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+62] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+70] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+55] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+63] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+71] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+72] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+80] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+88] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+73] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+81] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+89] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+74] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+82] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+90] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+152], s[sgprAlpha], v[vgprValuC+75] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+83] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+154], s[sgprAlpha], v[vgprValuC+91] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+76] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+84] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+92] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+158], s[sgprAlpha], v[vgprValuC+77] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+85] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+160], s[sgprAlpha], v[vgprValuC+93] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+78] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+162], s[sgprAlpha], v[vgprValuC+86] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+163], s[sgprAlpha], v[vgprValuC+94] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+164], s[sgprAlpha], v[vgprValuC+79] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+87] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+166], s[sgprAlpha], v[vgprValuC+95] // Multiply MI out reg with alpha

/******************************************/
/* Global Write Beta Batch #1 (d1,d0,vc1,vc0) = */
/*    (20,0,0,0:vw1); (20,1,0,0:vw1); (20,2,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (21,2,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (22,2,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1); (23,2,0,0:vw1); (24,0,0,0:vw1); (24,1,0,0:vw1); (24,2,0,0:vw1); (25,0,0,0:vw1); (25,1,0,0:vw1); (25,2,0,0:vw1); (26,0,0,0:vw1); (26,1,0,0:vw1); (26,2,0,0:vw1); (27,0,0,0:vw1); (27,1,0,0:vw1); (27,2,0,0:vw1); (28,0,0,0:vw1); (28,1,0,0:vw1); (28,2,0,0:vw1); (29,0,0,0:vw1); (29,1,0,0:vw1); (29,2,0,0:vw1); (30,0,0,0:vw1); (30,1,0,0:vw1); (30,2,0,0:vw1); (31,0,0,0:vw1); (31,1,0,0:vw1); (31,2,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v167, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
buffer_load_d16_b16 v168, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(20,0,2,0) */
buffer_load_d16_b16 v169, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v170, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(21,0,1,0) */
buffer_load_d16_b16 v171, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(21,0,2,0) */
buffer_load_d16_b16 v172, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v173, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(22,0,1,0) */
buffer_load_d16_b16 v174, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(22,0,2,0) */
buffer_load_d16_b16 v175, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v176, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(23,0,1,0) */
buffer_load_d16_b16 v177, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(23,0,2,0) */
buffer_load_d16_b16 v178, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 100                // scale StrideC *= numRows(50) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(50): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(50): gra SRD += inc(upper)
buffer_load_d16_b16 v179, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(24,0,1,0) */
buffer_load_d16_b16 v180, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(24,0,2,0) */
buffer_load_d16_b16 v181, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v182, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(25,0,1,0) */
buffer_load_d16_b16 v183, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(25,0,2,0) */
buffer_load_d16_b16 v184, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v185, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(26,0,1,0) */
buffer_load_d16_b16 v186, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(26,0,2,0) */
buffer_load_d16_b16 v187, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v188, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(27,0,1,0) */
buffer_load_d16_b16 v189, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(27,0,2,0) */
buffer_load_d16_b16 v190, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v191, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(28,0,1,0) */
buffer_load_d16_b16 v192, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(28,0,2,0) */
buffer_load_d16_b16 v193, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v194, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(29,0,1,0) */
buffer_load_d16_b16 v195, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(29,0,2,0) */
buffer_load_d16_b16 v196, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v197, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(30,0,1,0) */
buffer_load_d16_b16 v198, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(30,0,2,0) */
buffer_load_d16_b16 v199, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v200, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(31,0,1,0) */
buffer_load_d16_b16 v201, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(31,0,2,0) */
buffer_load_d16_b16 v202, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
s_sleep 7                                          // optimization: sync and wait
s_barrier

/* apply mask, calc new C and issue writes */
v_mov_b32 v126, 0xffff0000                         // mask for pack two bfloat16 element to 32bit
v_mov_b32 v127, 0x7fff0000                         // fp32 Nan
v_mov_b32 v128, 0x7fff                             // rounding bias for bfloat16

s_waitcnt vmcnt(35)                                // vlcnt(35) = 36 - 1 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v167                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+131], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+131], v[vgprValuC+131] // check Nan
v_bfe_u32 v125, v[vgprValuC+131], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+131], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+131], v125, v127, s8
v_lshrrev_b32 v131, 16, v[vgprValuC+131]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v131, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(34)                                // vlcnt(34) = 36 - 2 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v168                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+132], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+132], v[vgprValuC+132] // check Nan
v_bfe_u32 v125, v[vgprValuC+132], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+132], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+132], v125, v127, s8
v_lshrrev_b32 v132, 16, v[vgprValuC+132]           // convert C to bf16
buffer_store_b16 v132, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(33)                                // vlcnt(33) = 36 - 3 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v169                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+133], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+133], v[vgprValuC+133] // check Nan
v_bfe_u32 v125, v[vgprValuC+133], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+133], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+133], v125, v127, s8
v_lshrrev_b32 v133, 16, v[vgprValuC+133]           // convert C to bf16
buffer_store_b16 v133, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(32)                                // vlcnt(32) = 36 - 4 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v170                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+134], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+134], v[vgprValuC+134] // check Nan
v_bfe_u32 v125, v[vgprValuC+134], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+134], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+134], v125, v127, s8
v_lshrrev_b32 v134, 16, v[vgprValuC+134]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v134, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(31)                                // vlcnt(31) = 36 - 5 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v171                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+135], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+135], v[vgprValuC+135] // check Nan
v_bfe_u32 v125, v[vgprValuC+135], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+135], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+135], v125, v127, s8
v_lshrrev_b32 v135, 16, v[vgprValuC+135]           // convert C to bf16
buffer_store_b16 v135, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(30)                                // vlcnt(30) = 36 - 6 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v172                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+136], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+136], v[vgprValuC+136] // check Nan
v_bfe_u32 v125, v[vgprValuC+136], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+136], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+136], v125, v127, s8
v_lshrrev_b32 v136, 16, v[vgprValuC+136]           // convert C to bf16
buffer_store_b16 v136, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(29)                                // vlcnt(29) = 36 - 7 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v173                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+137], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+137], v[vgprValuC+137] // check Nan
v_bfe_u32 v125, v[vgprValuC+137], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+137], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+137], v125, v127, s8
v_lshrrev_b32 v137, 16, v[vgprValuC+137]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v137, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(28)                                // vlcnt(28) = 36 - 8 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v174                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+138], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+138], v[vgprValuC+138] // check Nan
v_bfe_u32 v125, v[vgprValuC+138], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+138], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+138], v125, v127, s8
v_lshrrev_b32 v138, 16, v[vgprValuC+138]           // convert C to bf16
buffer_store_b16 v138, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(27)                                // vlcnt(27) = 36 - 9 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v175                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+139], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+139], v[vgprValuC+139] // check Nan
v_bfe_u32 v125, v[vgprValuC+139], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+139], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+139], v125, v127, s8
v_lshrrev_b32 v139, 16, v[vgprValuC+139]           // convert C to bf16
buffer_store_b16 v139, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(26)                                // vlcnt(26) = 36 - 10 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v176                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+140], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+140], v[vgprValuC+140] // check Nan
v_bfe_u32 v125, v[vgprValuC+140], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+140], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+140], v125, v127, s8
v_lshrrev_b32 v140, 16, v[vgprValuC+140]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v140, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(25)                                // vlcnt(25) = 36 - 11 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v177                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+141], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+141], v[vgprValuC+141] // check Nan
v_bfe_u32 v125, v[vgprValuC+141], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+141], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+141], v125, v127, s8
v_lshrrev_b32 v141, 16, v[vgprValuC+141]           // convert C to bf16
buffer_store_b16 v141, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(24)                                // vlcnt(24) = 36 - 12 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v178                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+142], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+142], v[vgprValuC+142] // check Nan
v_bfe_u32 v125, v[vgprValuC+142], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+142], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+142], v125, v127, s8
v_lshrrev_b32 v142, 16, v[vgprValuC+142]           // convert C to bf16
buffer_store_b16 v142, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(23)                                // vlcnt(23) = 36 - 13 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v179                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+143], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+143], v[vgprValuC+143] // check Nan
v_bfe_u32 v125, v[vgprValuC+143], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+143], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+143], v125, v127, s8
v_lshrrev_b32 v143, 16, v[vgprValuC+143]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 100                // scale StrideD *= numRows(50) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(50): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(50): gra SRD += inc(upper)
buffer_store_b16 v143, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(22)                                // vlcnt(22) = 36 - 14 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v180                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+144], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+144], v[vgprValuC+144] // check Nan
v_bfe_u32 v125, v[vgprValuC+144], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+144], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+144], v125, v127, s8
v_lshrrev_b32 v144, 16, v[vgprValuC+144]           // convert C to bf16
buffer_store_b16 v144, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(21)                                // vlcnt(21) = 36 - 15 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v181                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+145], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+145], v[vgprValuC+145] // check Nan
v_bfe_u32 v125, v[vgprValuC+145], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+145], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+145], v125, v127, s8
v_lshrrev_b32 v145, 16, v[vgprValuC+145]           // convert C to bf16
buffer_store_b16 v145, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(20)                                // vlcnt(20) = 36 - 16 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v182                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+146], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+146], v[vgprValuC+146] // check Nan
v_bfe_u32 v125, v[vgprValuC+146], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+146], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+146], v125, v127, s8
v_lshrrev_b32 v146, 16, v[vgprValuC+146]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v146, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(19)                                // vlcnt(19) = 36 - 17 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v183                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+147], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+147], v[vgprValuC+147] // check Nan
v_bfe_u32 v125, v[vgprValuC+147], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+147], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+147], v125, v127, s8
v_lshrrev_b32 v147, 16, v[vgprValuC+147]           // convert C to bf16
buffer_store_b16 v147, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(18)                                // vlcnt(18) = 36 - 18 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v184                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+148], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+148], v[vgprValuC+148] // check Nan
v_bfe_u32 v125, v[vgprValuC+148], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+148], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+148], v125, v127, s8
v_lshrrev_b32 v148, 16, v[vgprValuC+148]           // convert C to bf16
buffer_store_b16 v148, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(17)                                // vlcnt(17) = 36 - 19 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v185                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+149], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+149], v[vgprValuC+149] // check Nan
v_bfe_u32 v125, v[vgprValuC+149], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+149], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+149], v125, v127, s8
v_lshrrev_b32 v149, 16, v[vgprValuC+149]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v149, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(16)                                // vlcnt(16) = 36 - 20 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v186                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+150], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+150], v[vgprValuC+150] // check Nan
v_bfe_u32 v125, v[vgprValuC+150], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+150], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+150], v125, v127, s8
v_lshrrev_b32 v150, 16, v[vgprValuC+150]           // convert C to bf16
buffer_store_b16 v150, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(15)                                // vlcnt(15) = 36 - 21 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v187                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+151], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+151], v[vgprValuC+151] // check Nan
v_bfe_u32 v125, v[vgprValuC+151], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+151], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+151], v125, v127, s8
v_lshrrev_b32 v151, 16, v[vgprValuC+151]           // convert C to bf16
buffer_store_b16 v151, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(14)                                // vlcnt(14) = 36 - 22 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v188                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+152], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+152], v[vgprValuC+152] // check Nan
v_bfe_u32 v125, v[vgprValuC+152], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+152], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+152], v125, v127, s8
v_lshrrev_b32 v152, 16, v[vgprValuC+152]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v152, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(13)                                // vlcnt(13) = 36 - 23 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v189                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+153], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+153], v[vgprValuC+153] // check Nan
v_bfe_u32 v125, v[vgprValuC+153], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+153], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+153], v125, v127, s8
v_lshrrev_b32 v153, 16, v[vgprValuC+153]           // convert C to bf16
buffer_store_b16 v153, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(12)                                // vlcnt(12) = 36 - 24 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v190                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+154], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+154], v[vgprValuC+154] // check Nan
v_bfe_u32 v125, v[vgprValuC+154], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+154], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+154], v125, v127, s8
v_lshrrev_b32 v154, 16, v[vgprValuC+154]           // convert C to bf16
buffer_store_b16 v154, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(11)                                // vlcnt(11) = 36 - 25 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v191                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+155], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+155], v[vgprValuC+155] // check Nan
v_bfe_u32 v125, v[vgprValuC+155], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+155], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+155], v125, v127, s8
v_lshrrev_b32 v155, 16, v[vgprValuC+155]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v155, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(10)                                // vlcnt(10) = 36 - 26 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v192                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+156], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+156], v[vgprValuC+156] // check Nan
v_bfe_u32 v125, v[vgprValuC+156], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+156], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+156], v125, v127, s8
v_lshrrev_b32 v156, 16, v[vgprValuC+156]           // convert C to bf16
buffer_store_b16 v156, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(9)                                 // vlcnt(9) = 36 - 27 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v193                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+157], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+157], v[vgprValuC+157] // check Nan
v_bfe_u32 v125, v[vgprValuC+157], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+157], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+157], v125, v127, s8
v_lshrrev_b32 v157, 16, v[vgprValuC+157]           // convert C to bf16
buffer_store_b16 v157, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(8)                                 // vlcnt(8) = 36 - 28 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v194                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+158], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+158], v[vgprValuC+158] // check Nan
v_bfe_u32 v125, v[vgprValuC+158], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+158], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+158], v125, v127, s8
v_lshrrev_b32 v158, 16, v[vgprValuC+158]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v158, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(7)                                 // vlcnt(7) = 36 - 29 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v195                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+159], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+159], v[vgprValuC+159] // check Nan
v_bfe_u32 v125, v[vgprValuC+159], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+159], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+159], v125, v127, s8
v_lshrrev_b32 v159, 16, v[vgprValuC+159]           // convert C to bf16
buffer_store_b16 v159, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(6)                                 // vlcnt(6) = 36 - 30 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v196                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+160], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+160], v[vgprValuC+160] // check Nan
v_bfe_u32 v125, v[vgprValuC+160], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+160], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+160], v125, v127, s8
v_lshrrev_b32 v160, 16, v[vgprValuC+160]           // convert C to bf16
buffer_store_b16 v160, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(5)                                 // vlcnt(5) = 36 - 31 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v197                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+161], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+161], v[vgprValuC+161] // check Nan
v_bfe_u32 v125, v[vgprValuC+161], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+161], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+161], v125, v127, s8
v_lshrrev_b32 v161, 16, v[vgprValuC+161]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v161, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(4)                                 // vlcnt(4) = 36 - 32 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v198                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+162], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+162], v[vgprValuC+162] // check Nan
v_bfe_u32 v125, v[vgprValuC+162], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+162], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+162], v125, v127, s8
v_lshrrev_b32 v162, 16, v[vgprValuC+162]           // convert C to bf16
buffer_store_b16 v162, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(3)                                 // vlcnt(3) = 36 - 33 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v199                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+163], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+163], v[vgprValuC+163] // check Nan
v_bfe_u32 v125, v[vgprValuC+163], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+163], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+163], v125, v127, s8
v_lshrrev_b32 v163, 16, v[vgprValuC+163]           // convert C to bf16
buffer_store_b16 v163, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(2)                                 // vlcnt(2) = 36 - 34 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v200                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+164], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+164], v[vgprValuC+164] // check Nan
v_bfe_u32 v125, v[vgprValuC+164], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+164], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+164], v125, v127, s8
v_lshrrev_b32 v164, 16, v[vgprValuC+164]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v164, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(1)                                 // vlcnt(1) = 36 - 35 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v201                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+165], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+165], v[vgprValuC+165] // check Nan
v_bfe_u32 v125, v[vgprValuC+165], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+165], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+165], v125, v127, s8
v_lshrrev_b32 v165, 16, v[vgprValuC+165]           // convert C to bf16
buffer_store_b16 v165, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(0)                                 // vlcnt(0) = 36 - 36 (beta) (interleaved)
v_lshlrev_b32 v122, 16, v202                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+166], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+166], v[vgprValuC+166] // check Nan
v_bfe_u32 v125, v[vgprValuC+166], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+166], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+166], v125, v127, s8
v_lshrrev_b32 v166, 16, v[vgprValuC+166]           // convert C to bf16
buffer_store_b16 v166, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_FD0_VW1_GSU1_NonEdgeEnd:
label_GW_B1_FD0_VW1_GSU1_Else:
label_GW_B1_FD0_VW1_GSU1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=40 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */
s_sleep 7                                          // optimization: sync and wait
s_barrier

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (5, 2, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (7, 2, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (8, 2, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (9, 2, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (10, 2, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (11, 2, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (12, 2, 0, 0), (13, 0, 0, 0)] */
v_mul_f32 v[vgprValuC+129], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+130], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+131], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+132], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+133], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+134], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+135], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+152], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+154], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+158], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+160], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+162], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+163], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+164], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+166], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+167], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+168], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (8,2,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (9,2,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (10,2,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (11,2,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (12,2,0,0:vw1); (13,0,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v124, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v170, v120, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v170, v124, v170, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v169, v170, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v170, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v170, v124, v170, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v172, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v172, v124, v172, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v171, v172, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v172, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v172, v124, v172, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v174, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v174, v124, v174, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v173, v174, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v174, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v174, v124, v174, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v176, v120, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v176, v124, v176, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v175, v176, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v176, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v176, v124, v176, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v178, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v178, v124, v178, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v177, v178, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v178, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v178, v124, v178, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v180, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v180, v124, v180, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v179, v180, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v180, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v180, v124, v180, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v182, v120, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v182, v124, v182, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v181, v182, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v182, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v182, v124, v182, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v184, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v184, v124, v184, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v183, v184, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v184, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v184, v124, v184, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v186, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v186, v124, v186, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v185, v186, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v186, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v186, v124, v186, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v188, v120, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v188, v124, v188, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v187, v188, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v188, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v188, v124, v188, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v190, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v190, v124, v190, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v189, v190, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v190, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v190, v124, v190, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v192, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v192, v124, v192, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v191, v192, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v192, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v192, v124, v192, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v194, v120, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v194, v124, v194, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v193, v194, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v194, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v194, v124, v194, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v196, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v196, v124, v196, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v195, v196, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v196, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v196, v124, v196, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v198, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v198, v124, v198, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v197, v198, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v198, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v198, v124, v198, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v200, v120, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v200, v124, v200, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v199, v200, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v200, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v200, v124, v200, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v202, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v202, v124, v202, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v201, v202, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v202, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v202, v124, v202, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v206, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v206, v124, v206, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v203, v206, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v206, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v206, v124, v206, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v208, v120, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v208, v124, v208, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v207, v208, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v208, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v208, v124, v208, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v210, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v210, v124, v210, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v209, v210, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v210, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v210, v124, v210, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v212, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v212, v124, v212, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v211, v212, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v212, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v212, v124, v212, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v214, v120, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v214, v124, v214, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v213, v214, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v214, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v214, v124, v214, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v216, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v216, v124, v216, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v215, v216, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v216, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v216, v124, v216, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v218, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v218, v124, v218, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v217, v218, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v218, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v218, v124, v218, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 50                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 50                // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 50                // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v220, v120, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v220, v124, v220, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v219, v220, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v220, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v220, v124, v220, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v222, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v222, v124, v222, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v221, v222, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v222, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v222, v124, v222, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v224, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v224, v124, v224, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v223, v224, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v224, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v224, v124, v224, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v226, v120, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v226, v124, v226, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v225, v226, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v226, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v226, v124, v226, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v228, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v228, v124, v228, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v227, v228, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v228, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v228, v124, v228, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v230, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v230, v124, v230, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v229, v230, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v230, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v230, v124, v230, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v232, v120, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v232, v124, v232, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v231, v232, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v232, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v232, v124, v232, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v234, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v234, v124, v234, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v233, v234, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v234, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v234, v124, v234, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v236, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v236, v124, v236, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v235, v236, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v236, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v236, v124, v236, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v238, v120, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v238, v124, v238, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v237, v238, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v238, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v238, v124, v238, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v240, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v240, v124, v240, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v239, v240, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v240, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v240, v124, v240, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v242, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v242, v124, v242, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v241, v242, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v242, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v242, v124, v242, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v244, v120, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v244, v124, v244, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v243, v244, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v244, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v244, v124, v244, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v246, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v246, v124, v246, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v245, v246, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v246, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v246, v124, v246, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v248, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v248, v124, v248, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v247, v248, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v248, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v248, v124, v248, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v250, v120, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v250, v124, v250, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v249, v250, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v250, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v250, v124, v250, s66                // LDD clip if OOB. offset
s_sleep 7                                          // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_mov_b32 v126, 0xffff0000                         // mask for pack two bfloat16 element to 32bit
v_mov_b32 v127, 0x7fff0000                         // fp32 Nan
v_mov_b32 v128, 0x7fff                             // rounding bias for bfloat16
v_lshlrev_b32 v122, 16, v169                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+129], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+129], v[vgprValuC+129] // check Nan
v_bfe_u32 v125, v[vgprValuC+129], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+129], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+129], v125, v127, s64
v_lshrrev_b32 v129, 16, v[vgprValuC+129]           // convert C to bf16
buffer_store_b16 v129, v170, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v171                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+130], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+130], v[vgprValuC+130] // check Nan
v_bfe_u32 v125, v[vgprValuC+130], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+130], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+130], v125, v127, s64
v_lshrrev_b32 v130, 16, v[vgprValuC+130]           // convert C to bf16
buffer_store_b16 v130, v172, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v173                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+131], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+131], v[vgprValuC+131] // check Nan
v_bfe_u32 v125, v[vgprValuC+131], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+131], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+131], v125, v127, s64
v_lshrrev_b32 v131, 16, v[vgprValuC+131]           // convert C to bf16
buffer_store_b16 v131, v174, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v175                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+132], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+132], v[vgprValuC+132] // check Nan
v_bfe_u32 v125, v[vgprValuC+132], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+132], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+132], v125, v127, s64
v_lshrrev_b32 v132, 16, v[vgprValuC+132]           // convert C to bf16
buffer_store_b16 v132, v176, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v177                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+133], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+133], v[vgprValuC+133] // check Nan
v_bfe_u32 v125, v[vgprValuC+133], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+133], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+133], v125, v127, s64
v_lshrrev_b32 v133, 16, v[vgprValuC+133]           // convert C to bf16
buffer_store_b16 v133, v178, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v179                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+134], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+134], v[vgprValuC+134] // check Nan
v_bfe_u32 v125, v[vgprValuC+134], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+134], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+134], v125, v127, s64
v_lshrrev_b32 v134, 16, v[vgprValuC+134]           // convert C to bf16
buffer_store_b16 v134, v180, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v181                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+135], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+135], v[vgprValuC+135] // check Nan
v_bfe_u32 v125, v[vgprValuC+135], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+135], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+135], v125, v127, s64
v_lshrrev_b32 v135, 16, v[vgprValuC+135]           // convert C to bf16
buffer_store_b16 v135, v182, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v183                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+136], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+136], v[vgprValuC+136] // check Nan
v_bfe_u32 v125, v[vgprValuC+136], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+136], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+136], v125, v127, s64
v_lshrrev_b32 v136, 16, v[vgprValuC+136]           // convert C to bf16
buffer_store_b16 v136, v184, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v185                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+137], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+137], v[vgprValuC+137] // check Nan
v_bfe_u32 v125, v[vgprValuC+137], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+137], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+137], v125, v127, s64
v_lshrrev_b32 v137, 16, v[vgprValuC+137]           // convert C to bf16
buffer_store_b16 v137, v186, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v187                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+138], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+138], v[vgprValuC+138] // check Nan
v_bfe_u32 v125, v[vgprValuC+138], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+138], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+138], v125, v127, s64
v_lshrrev_b32 v138, 16, v[vgprValuC+138]           // convert C to bf16
buffer_store_b16 v138, v188, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v189                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+139], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+139], v[vgprValuC+139] // check Nan
v_bfe_u32 v125, v[vgprValuC+139], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+139], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+139], v125, v127, s64
v_lshrrev_b32 v139, 16, v[vgprValuC+139]           // convert C to bf16
buffer_store_b16 v139, v190, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v191                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+140], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+140], v[vgprValuC+140] // check Nan
v_bfe_u32 v125, v[vgprValuC+140], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+140], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+140], v125, v127, s64
v_lshrrev_b32 v140, 16, v[vgprValuC+140]           // convert C to bf16
buffer_store_b16 v140, v192, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v193                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+141], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+141], v[vgprValuC+141] // check Nan
v_bfe_u32 v125, v[vgprValuC+141], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+141], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+141], v125, v127, s64
v_lshrrev_b32 v141, 16, v[vgprValuC+141]           // convert C to bf16
buffer_store_b16 v141, v194, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v195                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+142], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+142], v[vgprValuC+142] // check Nan
v_bfe_u32 v125, v[vgprValuC+142], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+142], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+142], v125, v127, s64
v_lshrrev_b32 v142, 16, v[vgprValuC+142]           // convert C to bf16
buffer_store_b16 v142, v196, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v197                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+143], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+143], v[vgprValuC+143] // check Nan
v_bfe_u32 v125, v[vgprValuC+143], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+143], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+143], v125, v127, s64
v_lshrrev_b32 v143, 16, v[vgprValuC+143]           // convert C to bf16
buffer_store_b16 v143, v198, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v199                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+144], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+144], v[vgprValuC+144] // check Nan
v_bfe_u32 v125, v[vgprValuC+144], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+144], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+144], v125, v127, s64
v_lshrrev_b32 v144, 16, v[vgprValuC+144]           // convert C to bf16
buffer_store_b16 v144, v200, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v201                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+145], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+145], v[vgprValuC+145] // check Nan
v_bfe_u32 v125, v[vgprValuC+145], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+145], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+145], v125, v127, s64
v_lshrrev_b32 v145, 16, v[vgprValuC+145]           // convert C to bf16
buffer_store_b16 v145, v202, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v203                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+146], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+146], v[vgprValuC+146] // check Nan
v_bfe_u32 v125, v[vgprValuC+146], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+146], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+146], v125, v127, s64
v_lshrrev_b32 v146, 16, v[vgprValuC+146]           // convert C to bf16
buffer_store_b16 v146, v206, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v207                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+147], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+147], v[vgprValuC+147] // check Nan
v_bfe_u32 v125, v[vgprValuC+147], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+147], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+147], v125, v127, s64
v_lshrrev_b32 v147, 16, v[vgprValuC+147]           // convert C to bf16
buffer_store_b16 v147, v208, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v209                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+148], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+148], v[vgprValuC+148] // check Nan
v_bfe_u32 v125, v[vgprValuC+148], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+148], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+148], v125, v127, s64
v_lshrrev_b32 v148, 16, v[vgprValuC+148]           // convert C to bf16
buffer_store_b16 v148, v210, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v211                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+149], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+149], v[vgprValuC+149] // check Nan
v_bfe_u32 v125, v[vgprValuC+149], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+149], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+149], v125, v127, s64
v_lshrrev_b32 v149, 16, v[vgprValuC+149]           // convert C to bf16
buffer_store_b16 v149, v212, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v213                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+150], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+150], v[vgprValuC+150] // check Nan
v_bfe_u32 v125, v[vgprValuC+150], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+150], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+150], v125, v127, s64
v_lshrrev_b32 v150, 16, v[vgprValuC+150]           // convert C to bf16
buffer_store_b16 v150, v214, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v215                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+151], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+151], v[vgprValuC+151] // check Nan
v_bfe_u32 v125, v[vgprValuC+151], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+151], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+151], v125, v127, s64
v_lshrrev_b32 v151, 16, v[vgprValuC+151]           // convert C to bf16
buffer_store_b16 v151, v216, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v217                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+152], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+152], v[vgprValuC+152] // check Nan
v_bfe_u32 v125, v[vgprValuC+152], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+152], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+152], v125, v127, s64
v_lshrrev_b32 v152, 16, v[vgprValuC+152]           // convert C to bf16
buffer_store_b16 v152, v218, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v219                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+153], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+153], v[vgprValuC+153] // check Nan
v_bfe_u32 v125, v[vgprValuC+153], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+153], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+153], v125, v127, s64
v_lshrrev_b32 v153, 16, v[vgprValuC+153]           // convert C to bf16
buffer_store_b16 v153, v220, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v221                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+154], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+154], v[vgprValuC+154] // check Nan
v_bfe_u32 v125, v[vgprValuC+154], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+154], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+154], v125, v127, s64
v_lshrrev_b32 v154, 16, v[vgprValuC+154]           // convert C to bf16
buffer_store_b16 v154, v222, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v223                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+155], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+155], v[vgprValuC+155] // check Nan
v_bfe_u32 v125, v[vgprValuC+155], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+155], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+155], v125, v127, s64
v_lshrrev_b32 v155, 16, v[vgprValuC+155]           // convert C to bf16
buffer_store_b16 v155, v224, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v225                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+156], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+156], v[vgprValuC+156] // check Nan
v_bfe_u32 v125, v[vgprValuC+156], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+156], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+156], v125, v127, s64
v_lshrrev_b32 v156, 16, v[vgprValuC+156]           // convert C to bf16
buffer_store_b16 v156, v226, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v227                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+157], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+157], v[vgprValuC+157] // check Nan
v_bfe_u32 v125, v[vgprValuC+157], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+157], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+157], v125, v127, s64
v_lshrrev_b32 v157, 16, v[vgprValuC+157]           // convert C to bf16
buffer_store_b16 v157, v228, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v229                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+158], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+158], v[vgprValuC+158] // check Nan
v_bfe_u32 v125, v[vgprValuC+158], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+158], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+158], v125, v127, s64
v_lshrrev_b32 v158, 16, v[vgprValuC+158]           // convert C to bf16
buffer_store_b16 v158, v230, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v231                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+159], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+159], v[vgprValuC+159] // check Nan
v_bfe_u32 v125, v[vgprValuC+159], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+159], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+159], v125, v127, s64
v_lshrrev_b32 v159, 16, v[vgprValuC+159]           // convert C to bf16
buffer_store_b16 v159, v232, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v233                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+160], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+160], v[vgprValuC+160] // check Nan
v_bfe_u32 v125, v[vgprValuC+160], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+160], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+160], v125, v127, s64
v_lshrrev_b32 v160, 16, v[vgprValuC+160]           // convert C to bf16
buffer_store_b16 v160, v234, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v235                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+161], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+161], v[vgprValuC+161] // check Nan
v_bfe_u32 v125, v[vgprValuC+161], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+161], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+161], v125, v127, s64
v_lshrrev_b32 v161, 16, v[vgprValuC+161]           // convert C to bf16
buffer_store_b16 v161, v236, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v237                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+162], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+162], v[vgprValuC+162] // check Nan
v_bfe_u32 v125, v[vgprValuC+162], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+162], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+162], v125, v127, s64
v_lshrrev_b32 v162, 16, v[vgprValuC+162]           // convert C to bf16
buffer_store_b16 v162, v238, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v239                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+163], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+163], v[vgprValuC+163] // check Nan
v_bfe_u32 v125, v[vgprValuC+163], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+163], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+163], v125, v127, s64
v_lshrrev_b32 v163, 16, v[vgprValuC+163]           // convert C to bf16
buffer_store_b16 v163, v240, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v241                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+164], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+164], v[vgprValuC+164] // check Nan
v_bfe_u32 v125, v[vgprValuC+164], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+164], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+164], v125, v127, s64
v_lshrrev_b32 v164, 16, v[vgprValuC+164]           // convert C to bf16
buffer_store_b16 v164, v242, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v243                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+165], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+165], v[vgprValuC+165] // check Nan
v_bfe_u32 v125, v[vgprValuC+165], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+165], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+165], v125, v127, s64
v_lshrrev_b32 v165, 16, v[vgprValuC+165]           // convert C to bf16
buffer_store_b16 v165, v244, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v245                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+166], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+166], v[vgprValuC+166] // check Nan
v_bfe_u32 v125, v[vgprValuC+166], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+166], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+166], v125, v127, s64
v_lshrrev_b32 v166, 16, v[vgprValuC+166]           // convert C to bf16
buffer_store_b16 v166, v246, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v247                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+167], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+167], v[vgprValuC+167] // check Nan
v_bfe_u32 v125, v[vgprValuC+167], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+167], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+167], v125, v127, s64
v_lshrrev_b32 v167, 16, v[vgprValuC+167]           // convert C to bf16
buffer_store_b16 v167, v248, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v249                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+168], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+168], v[vgprValuC+168] // check Nan
v_bfe_u32 v125, v[vgprValuC+168], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+168], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+168], v125, v127, s64
v_lshrrev_b32 v168, 16, v[vgprValuC+168]           // convert C to bf16
buffer_store_b16 v168, v250, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */
s_sleep 7                                          // optimization: sync and wait
s_barrier

/* rC *= alpha batchElements=[(13, 1, 0, 0), (13, 2, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (14, 2, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (15, 2, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (16, 2, 0, 0), (17, 0, 0, 0), (17, 1, 0, 0), (17, 2, 0, 0), (18, 0, 0, 0), (18, 1, 0, 0), (18, 2, 0, 0), (19, 0, 0, 0), (19, 1, 0, 0), (19, 2, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0), (20, 2, 0, 0), (21, 0, 0, 0), (21, 1, 0, 0), (21, 2, 0, 0), (22, 0, 0, 0), (22, 1, 0, 0), (22, 2, 0, 0), (23, 0, 0, 0), (23, 1, 0, 0), (23, 2, 0, 0), (24, 0, 0, 0), (24, 1, 0, 0), (24, 2, 0, 0), (25, 0, 0, 0), (25, 1, 0, 0), (25, 2, 0, 0), (26, 0, 0, 0), (26, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+129], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+130], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+131], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+132], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+133], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+134], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+135], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+48] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+56] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+64] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+49] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+57] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+65] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+50] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+58] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+66] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+51] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+59] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+67] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+52] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+60] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+68] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+152], s[sgprAlpha], v[vgprValuC+53] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+61] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+154], s[sgprAlpha], v[vgprValuC+69] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+54] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+62] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+70] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+158], s[sgprAlpha], v[vgprValuC+55] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+63] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+160], s[sgprAlpha], v[vgprValuC+71] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+72] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+162], s[sgprAlpha], v[vgprValuC+80] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+163], s[sgprAlpha], v[vgprValuC+88] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+164], s[sgprAlpha], v[vgprValuC+73] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+81] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+166], s[sgprAlpha], v[vgprValuC+89] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+167], s[sgprAlpha], v[vgprValuC+74] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+168], s[sgprAlpha], v[vgprValuC+82] // Multiply MI out reg with alpha

/******************************************/
/* Global Write Beta Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (13,1,0,0:vw1); (13,2,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (14,2,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (15,2,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (16,2,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (17,2,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (18,2,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (19,2,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (20,2,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (21,2,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (22,2,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1); (23,2,0,0:vw1); (24,0,0,0:vw1); (24,1,0,0:vw1); (24,2,0,0:vw1); (25,0,0,0:vw1); (25,1,0,0:vw1); (25,2,0,0:vw1); (26,0,0,0:vw1); (26,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v124, BufferOOB
/* (d1,vc1,d0,vc0)=(13,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v170, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v170, v124, v170, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v169, v170, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v170, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v170, v124, v170, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v172, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v172, v124, v172, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v171, v172, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v172, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v172, v124, v172, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v174, v120, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v174, v124, v174, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v173, v174, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v174, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v174, v124, v174, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v176, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v176, v124, v176, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v175, v176, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v176, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v176, v124, v176, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v178, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v178, v124, v178, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v177, v178, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v178, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v178, v124, v178, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v180, v120, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v180, v124, v180, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v179, v180, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v180, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v180, v124, v180, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v182, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v182, v124, v182, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v181, v182, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v182, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v182, v124, v182, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v184, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v184, v124, v184, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v183, v184, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v184, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v184, v124, v184, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 50                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 50                // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 50                // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v186, v120, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v186, v124, v186, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v185, v186, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v186, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v186, v124, v186, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v188, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v188, v124, v188, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v187, v188, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v188, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v188, v124, v188, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v190, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v190, v124, v190, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v189, v190, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v190, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v190, v124, v190, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v192, v120, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v192, v124, v192, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v191, v192, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v192, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v192, v124, v192, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v194, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v194, v124, v194, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v193, v194, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v194, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v194, v124, v194, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(17,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v196, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v196, v124, v196, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v195, v196, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v196, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v196, v124, v196, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v198, v120, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v198, v124, v198, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v197, v198, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v198, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v198, v124, v198, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v200, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v200, v124, v200, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v199, v200, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v200, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v200, v124, v200, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(18,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v202, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v202, v124, v202, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v201, v202, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v202, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v202, v124, v202, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v206, v120, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v206, v124, v206, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v203, v206, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v206, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v206, v124, v206, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v208, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v208, v124, v208, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v207, v208, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v208, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v208, v124, v208, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(19,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v210, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v210, v124, v210, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v209, v210, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v210, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v210, v124, v210, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v212, v120, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v212, v124, v212, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v211, v212, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v212, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v212, v124, v212, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v214, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v214, v124, v214, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v213, v214, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v214, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v214, v124, v214, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v216, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v216, v124, v216, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v215, v216, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v216, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v216, v124, v216, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v218, v120, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v218, v124, v218, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v217, v218, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v218, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v218, v124, v218, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v220, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v220, v124, v220, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v219, v220, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v220, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v220, v124, v220, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(21,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v222, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v222, v124, v222, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v221, v222, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v222, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v222, v124, v222, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v224, v120, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v224, v124, v224, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v223, v224, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v224, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v224, v124, v224, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v226, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v226, v124, v226, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v225, v226, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v226, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v226, v124, v226, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(22,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v228, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v228, v124, v228, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v227, v228, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v228, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v228, v124, v228, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v230, v120, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v230, v124, v230, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v229, v230, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v230, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v230, v124, v230, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v232, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v232, v124, v232, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v231, v232, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v232, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v232, v124, v232, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(23,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v234, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v234, v124, v234, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v233, v234, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v234, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v234, v124, v234, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 50                // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 50                // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 50                // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v236, v120, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v236, v124, v236, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v235, v236, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v236, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v236, v124, v236, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v238, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v238, v124, v238, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v237, v238, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v238, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v238, v124, v238, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(24,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v240, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v240, v124, v240, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v239, v240, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v240, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v240, v124, v240, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v242, v120, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v242, v124, v242, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v241, v242, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v242, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v242, v124, v242, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v244, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v244, v124, v244, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v243, v244, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v244, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v244, v124, v244, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(25,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v246, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v246, v124, v246, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v245, v246, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v246, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v246, v124, v246, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v248, v120, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v248, v124, v248, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v247, v248, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v248, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v248, v124, v248, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(26,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v250, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v250, v124, v250, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v249, v250, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v250, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v250, v124, v250, s66                // LDD clip if OOB. offset
s_sleep 7                                          // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_mov_b32 v126, 0xffff0000                         // mask for pack two bfloat16 element to 32bit
v_mov_b32 v127, 0x7fff0000                         // fp32 Nan
v_mov_b32 v128, 0x7fff                             // rounding bias for bfloat16
v_lshlrev_b32 v122, 16, v169                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+129], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+129], v[vgprValuC+129] // check Nan
v_bfe_u32 v125, v[vgprValuC+129], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+129], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+129], v125, v127, s64
v_lshrrev_b32 v129, 16, v[vgprValuC+129]           // convert C to bf16
buffer_store_b16 v129, v170, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v171                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+130], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+130], v[vgprValuC+130] // check Nan
v_bfe_u32 v125, v[vgprValuC+130], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+130], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+130], v125, v127, s64
v_lshrrev_b32 v130, 16, v[vgprValuC+130]           // convert C to bf16
buffer_store_b16 v130, v172, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v173                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+131], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+131], v[vgprValuC+131] // check Nan
v_bfe_u32 v125, v[vgprValuC+131], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+131], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+131], v125, v127, s64
v_lshrrev_b32 v131, 16, v[vgprValuC+131]           // convert C to bf16
buffer_store_b16 v131, v174, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v175                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+132], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+132], v[vgprValuC+132] // check Nan
v_bfe_u32 v125, v[vgprValuC+132], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+132], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+132], v125, v127, s64
v_lshrrev_b32 v132, 16, v[vgprValuC+132]           // convert C to bf16
buffer_store_b16 v132, v176, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v177                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+133], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+133], v[vgprValuC+133] // check Nan
v_bfe_u32 v125, v[vgprValuC+133], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+133], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+133], v125, v127, s64
v_lshrrev_b32 v133, 16, v[vgprValuC+133]           // convert C to bf16
buffer_store_b16 v133, v178, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v179                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+134], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+134], v[vgprValuC+134] // check Nan
v_bfe_u32 v125, v[vgprValuC+134], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+134], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+134], v125, v127, s64
v_lshrrev_b32 v134, 16, v[vgprValuC+134]           // convert C to bf16
buffer_store_b16 v134, v180, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v181                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+135], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+135], v[vgprValuC+135] // check Nan
v_bfe_u32 v125, v[vgprValuC+135], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+135], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+135], v125, v127, s64
v_lshrrev_b32 v135, 16, v[vgprValuC+135]           // convert C to bf16
buffer_store_b16 v135, v182, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v183                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+136], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+136], v[vgprValuC+136] // check Nan
v_bfe_u32 v125, v[vgprValuC+136], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+136], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+136], v125, v127, s64
v_lshrrev_b32 v136, 16, v[vgprValuC+136]           // convert C to bf16
buffer_store_b16 v136, v184, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v185                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+137], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+137], v[vgprValuC+137] // check Nan
v_bfe_u32 v125, v[vgprValuC+137], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+137], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+137], v125, v127, s64
v_lshrrev_b32 v137, 16, v[vgprValuC+137]           // convert C to bf16
buffer_store_b16 v137, v186, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v187                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+138], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+138], v[vgprValuC+138] // check Nan
v_bfe_u32 v125, v[vgprValuC+138], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+138], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+138], v125, v127, s64
v_lshrrev_b32 v138, 16, v[vgprValuC+138]           // convert C to bf16
buffer_store_b16 v138, v188, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v189                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+139], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+139], v[vgprValuC+139] // check Nan
v_bfe_u32 v125, v[vgprValuC+139], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+139], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+139], v125, v127, s64
v_lshrrev_b32 v139, 16, v[vgprValuC+139]           // convert C to bf16
buffer_store_b16 v139, v190, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v191                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+140], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+140], v[vgprValuC+140] // check Nan
v_bfe_u32 v125, v[vgprValuC+140], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+140], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+140], v125, v127, s64
v_lshrrev_b32 v140, 16, v[vgprValuC+140]           // convert C to bf16
buffer_store_b16 v140, v192, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v193                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+141], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+141], v[vgprValuC+141] // check Nan
v_bfe_u32 v125, v[vgprValuC+141], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+141], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+141], v125, v127, s64
v_lshrrev_b32 v141, 16, v[vgprValuC+141]           // convert C to bf16
buffer_store_b16 v141, v194, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v195                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+142], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+142], v[vgprValuC+142] // check Nan
v_bfe_u32 v125, v[vgprValuC+142], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+142], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+142], v125, v127, s64
v_lshrrev_b32 v142, 16, v[vgprValuC+142]           // convert C to bf16
buffer_store_b16 v142, v196, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v197                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+143], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+143], v[vgprValuC+143] // check Nan
v_bfe_u32 v125, v[vgprValuC+143], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+143], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+143], v125, v127, s64
v_lshrrev_b32 v143, 16, v[vgprValuC+143]           // convert C to bf16
buffer_store_b16 v143, v198, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v199                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+144], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+144], v[vgprValuC+144] // check Nan
v_bfe_u32 v125, v[vgprValuC+144], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+144], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+144], v125, v127, s64
v_lshrrev_b32 v144, 16, v[vgprValuC+144]           // convert C to bf16
buffer_store_b16 v144, v200, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v201                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+145], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+145], v[vgprValuC+145] // check Nan
v_bfe_u32 v125, v[vgprValuC+145], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+145], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+145], v125, v127, s64
v_lshrrev_b32 v145, 16, v[vgprValuC+145]           // convert C to bf16
buffer_store_b16 v145, v202, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v203                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+146], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+146], v[vgprValuC+146] // check Nan
v_bfe_u32 v125, v[vgprValuC+146], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+146], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+146], v125, v127, s64
v_lshrrev_b32 v146, 16, v[vgprValuC+146]           // convert C to bf16
buffer_store_b16 v146, v206, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v207                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+147], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+147], v[vgprValuC+147] // check Nan
v_bfe_u32 v125, v[vgprValuC+147], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+147], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+147], v125, v127, s64
v_lshrrev_b32 v147, 16, v[vgprValuC+147]           // convert C to bf16
buffer_store_b16 v147, v208, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v209                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+148], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+148], v[vgprValuC+148] // check Nan
v_bfe_u32 v125, v[vgprValuC+148], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+148], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+148], v125, v127, s64
v_lshrrev_b32 v148, 16, v[vgprValuC+148]           // convert C to bf16
buffer_store_b16 v148, v210, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v211                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+149], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+149], v[vgprValuC+149] // check Nan
v_bfe_u32 v125, v[vgprValuC+149], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+149], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+149], v125, v127, s64
v_lshrrev_b32 v149, 16, v[vgprValuC+149]           // convert C to bf16
buffer_store_b16 v149, v212, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v213                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+150], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+150], v[vgprValuC+150] // check Nan
v_bfe_u32 v125, v[vgprValuC+150], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+150], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+150], v125, v127, s64
v_lshrrev_b32 v150, 16, v[vgprValuC+150]           // convert C to bf16
buffer_store_b16 v150, v214, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v215                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+151], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+151], v[vgprValuC+151] // check Nan
v_bfe_u32 v125, v[vgprValuC+151], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+151], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+151], v125, v127, s64
v_lshrrev_b32 v151, 16, v[vgprValuC+151]           // convert C to bf16
buffer_store_b16 v151, v216, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v217                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+152], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+152], v[vgprValuC+152] // check Nan
v_bfe_u32 v125, v[vgprValuC+152], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+152], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+152], v125, v127, s64
v_lshrrev_b32 v152, 16, v[vgprValuC+152]           // convert C to bf16
buffer_store_b16 v152, v218, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v219                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+153], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+153], v[vgprValuC+153] // check Nan
v_bfe_u32 v125, v[vgprValuC+153], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+153], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+153], v125, v127, s64
v_lshrrev_b32 v153, 16, v[vgprValuC+153]           // convert C to bf16
buffer_store_b16 v153, v220, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v221                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+154], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+154], v[vgprValuC+154] // check Nan
v_bfe_u32 v125, v[vgprValuC+154], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+154], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+154], v125, v127, s64
v_lshrrev_b32 v154, 16, v[vgprValuC+154]           // convert C to bf16
buffer_store_b16 v154, v222, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v223                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+155], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+155], v[vgprValuC+155] // check Nan
v_bfe_u32 v125, v[vgprValuC+155], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+155], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+155], v125, v127, s64
v_lshrrev_b32 v155, 16, v[vgprValuC+155]           // convert C to bf16
buffer_store_b16 v155, v224, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v225                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+156], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+156], v[vgprValuC+156] // check Nan
v_bfe_u32 v125, v[vgprValuC+156], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+156], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+156], v125, v127, s64
v_lshrrev_b32 v156, 16, v[vgprValuC+156]           // convert C to bf16
buffer_store_b16 v156, v226, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v227                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+157], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+157], v[vgprValuC+157] // check Nan
v_bfe_u32 v125, v[vgprValuC+157], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+157], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+157], v125, v127, s64
v_lshrrev_b32 v157, 16, v[vgprValuC+157]           // convert C to bf16
buffer_store_b16 v157, v228, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v229                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+158], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+158], v[vgprValuC+158] // check Nan
v_bfe_u32 v125, v[vgprValuC+158], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+158], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+158], v125, v127, s64
v_lshrrev_b32 v158, 16, v[vgprValuC+158]           // convert C to bf16
buffer_store_b16 v158, v230, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v231                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+159], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+159], v[vgprValuC+159] // check Nan
v_bfe_u32 v125, v[vgprValuC+159], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+159], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+159], v125, v127, s64
v_lshrrev_b32 v159, 16, v[vgprValuC+159]           // convert C to bf16
buffer_store_b16 v159, v232, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v233                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+160], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+160], v[vgprValuC+160] // check Nan
v_bfe_u32 v125, v[vgprValuC+160], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+160], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+160], v125, v127, s64
v_lshrrev_b32 v160, 16, v[vgprValuC+160]           // convert C to bf16
buffer_store_b16 v160, v234, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v235                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+161], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+161], v[vgprValuC+161] // check Nan
v_bfe_u32 v125, v[vgprValuC+161], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+161], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+161], v125, v127, s64
v_lshrrev_b32 v161, 16, v[vgprValuC+161]           // convert C to bf16
buffer_store_b16 v161, v236, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v237                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+162], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+162], v[vgprValuC+162] // check Nan
v_bfe_u32 v125, v[vgprValuC+162], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+162], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+162], v125, v127, s64
v_lshrrev_b32 v162, 16, v[vgprValuC+162]           // convert C to bf16
buffer_store_b16 v162, v238, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v239                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+163], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+163], v[vgprValuC+163] // check Nan
v_bfe_u32 v125, v[vgprValuC+163], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+163], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+163], v125, v127, s64
v_lshrrev_b32 v163, 16, v[vgprValuC+163]           // convert C to bf16
buffer_store_b16 v163, v240, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v241                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+164], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+164], v[vgprValuC+164] // check Nan
v_bfe_u32 v125, v[vgprValuC+164], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+164], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+164], v125, v127, s64
v_lshrrev_b32 v164, 16, v[vgprValuC+164]           // convert C to bf16
buffer_store_b16 v164, v242, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v243                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+165], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+165], v[vgprValuC+165] // check Nan
v_bfe_u32 v125, v[vgprValuC+165], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+165], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+165], v125, v127, s64
v_lshrrev_b32 v165, 16, v[vgprValuC+165]           // convert C to bf16
buffer_store_b16 v165, v244, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v245                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+166], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+166], v[vgprValuC+166] // check Nan
v_bfe_u32 v125, v[vgprValuC+166], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+166], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+166], v125, v127, s64
v_lshrrev_b32 v166, 16, v[vgprValuC+166]           // convert C to bf16
buffer_store_b16 v166, v246, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v247                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+167], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+167], v[vgprValuC+167] // check Nan
v_bfe_u32 v125, v[vgprValuC+167], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+167], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+167], v125, v127, s64
v_lshrrev_b32 v167, 16, v[vgprValuC+167]           // convert C to bf16
buffer_store_b16 v167, v248, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v249                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+168], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+168], v[vgprValuC+168] // check Nan
v_bfe_u32 v125, v[vgprValuC+168], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+168], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+168], v125, v127, s64
v_lshrrev_b32 v168, 16, v[vgprValuC+168]           // convert C to bf16
buffer_store_b16 v168, v250, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */
s_sleep 7                                          // optimization: sync and wait
s_barrier

/* rC *= alpha batchElements=[(26, 2, 0, 0), (27, 0, 0, 0), (27, 1, 0, 0), (27, 2, 0, 0), (28, 0, 0, 0), (28, 1, 0, 0), (28, 2, 0, 0), (29, 0, 0, 0), (29, 1, 0, 0), (29, 2, 0, 0), (30, 0, 0, 0), (30, 1, 0, 0), (30, 2, 0, 0), (31, 0, 0, 0), (31, 1, 0, 0), (31, 2, 0, 0)] */
v_mul_f32 v[vgprValuC+129], s[sgprAlpha], v[vgprValuC+90] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+130], s[sgprAlpha], v[vgprValuC+75] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+131], s[sgprAlpha], v[vgprValuC+83] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+132], s[sgprAlpha], v[vgprValuC+91] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+133], s[sgprAlpha], v[vgprValuC+76] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+134], s[sgprAlpha], v[vgprValuC+84] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+135], s[sgprAlpha], v[vgprValuC+92] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+77] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+85] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+93] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+78] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+86] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+94] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+79] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+87] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+95] // Multiply MI out reg with alpha

/******************************************/
/* Global Write Beta Edge Batch #2 (d1,d0,vc1,vc0) = */
/*    (26,2,0,0:vw1); (27,0,0,0:vw1); (27,1,0,0:vw1); (27,2,0,0:vw1); (28,0,0,0:vw1); (28,1,0,0:vw1); (28,2,0,0:vw1); (29,0,0,0:vw1); (29,1,0,0:vw1); (29,2,0,0:vw1); (30,0,0,0:vw1); (30,1,0,0:vw1); (30,2,0,0:vw1); (31,0,0,0:vw1); (31,1,0,0:vw1); (31,2,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v124, BufferOOB
/* (d1,vc1,d0,vc0)=(26,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v146, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v146, v124, v146, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v145, v146, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v146, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v146, v124, v146, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v148, v120, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v148, v124, v148, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v147, v148, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v148, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v148, v124, v148, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v150, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v150, v124, v150, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v149, v150, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v150, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v150, v124, v150, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(27,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v152, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v152, v124, v152, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v151, v152, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v152, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v152, v124, v152, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v154, v120, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v124, v154, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v153, v154, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v154, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v124, v154, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v156, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v124, v156, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v155, v156, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v156, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v124, v156, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(28,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v158, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v124, v158, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v157, v158, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v158, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v124, v158, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v160, v120, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v160, v124, v160, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v159, v160, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v160, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v160, v124, v160, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v162, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v162, v124, v162, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v161, v162, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v162, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v162, v124, v162, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(29,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v164, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v164, v124, v164, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v163, v164, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v164, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v164, v124, v164, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v166, v120, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v166, v124, v166, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v165, v166, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v166, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v166, v124, v166, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v168, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v168, v124, v168, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v167, v168, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v168, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v168, v124, v168, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(30,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v170, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v170, v124, v170, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v169, v170, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v170, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v170, v124, v170, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
v_add_co_u32 v119, vcc_lo, v119, 2                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v120, v120, s64                       // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v121, v121, s64                       // Move coutRowPtrD to next row
v_cmp_lt_u32 s64, v118, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v172, v120, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v172, v124, v172, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v171, v172, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v172, v121, v118, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v172, v124, v172, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,1,0) */
v_add_co_u32 v122, vcc_lo, v118, 64                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v174, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v174, v124, v174, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v173, v174, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v174, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v174, v124, v174, s66                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(31,0,2,0) */
s_mov_b32 s64, 128                                 // coordOffset0 d0=2 vc0=0
v_add_co_u32 v122, vcc_lo, v118, s64               // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s64, v122, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s66, v119, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s66, s64, s66                            // in0 && in1
v_add_lshl_u32 v176, v120, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v176, v124, v176, s66                // LDC clip if OOB. offset
buffer_load_d16_b16 v175, v176, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v176, v121, v122, 1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v176, v124, v176, s66                // LDD clip if OOB. offset
s_sleep 7                                          // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_mov_b32 v126, 0xffff0000                         // mask for pack two bfloat16 element to 32bit
v_mov_b32 v127, 0x7fff0000                         // fp32 Nan
v_mov_b32 v128, 0x7fff                             // rounding bias for bfloat16
v_lshlrev_b32 v122, 16, v145                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+129], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+129], v[vgprValuC+129] // check Nan
v_bfe_u32 v125, v[vgprValuC+129], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+129], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+129], v125, v127, s64
v_lshrrev_b32 v129, 16, v[vgprValuC+129]           // convert C to bf16
buffer_store_b16 v129, v146, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v147                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+130], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+130], v[vgprValuC+130] // check Nan
v_bfe_u32 v125, v[vgprValuC+130], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+130], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+130], v125, v127, s64
v_lshrrev_b32 v130, 16, v[vgprValuC+130]           // convert C to bf16
buffer_store_b16 v130, v148, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v149                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+131], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+131], v[vgprValuC+131] // check Nan
v_bfe_u32 v125, v[vgprValuC+131], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+131], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+131], v125, v127, s64
v_lshrrev_b32 v131, 16, v[vgprValuC+131]           // convert C to bf16
buffer_store_b16 v131, v150, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v151                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+132], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+132], v[vgprValuC+132] // check Nan
v_bfe_u32 v125, v[vgprValuC+132], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+132], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+132], v125, v127, s64
v_lshrrev_b32 v132, 16, v[vgprValuC+132]           // convert C to bf16
buffer_store_b16 v132, v152, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v153                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+133], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+133], v[vgprValuC+133] // check Nan
v_bfe_u32 v125, v[vgprValuC+133], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+133], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+133], v125, v127, s64
v_lshrrev_b32 v133, 16, v[vgprValuC+133]           // convert C to bf16
buffer_store_b16 v133, v154, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v155                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+134], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+134], v[vgprValuC+134] // check Nan
v_bfe_u32 v125, v[vgprValuC+134], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+134], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+134], v125, v127, s64
v_lshrrev_b32 v134, 16, v[vgprValuC+134]           // convert C to bf16
buffer_store_b16 v134, v156, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v157                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+135], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+135], v[vgprValuC+135] // check Nan
v_bfe_u32 v125, v[vgprValuC+135], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+135], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+135], v125, v127, s64
v_lshrrev_b32 v135, 16, v[vgprValuC+135]           // convert C to bf16
buffer_store_b16 v135, v158, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v159                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+136], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+136], v[vgprValuC+136] // check Nan
v_bfe_u32 v125, v[vgprValuC+136], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+136], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+136], v125, v127, s64
v_lshrrev_b32 v136, 16, v[vgprValuC+136]           // convert C to bf16
buffer_store_b16 v136, v160, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v161                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+137], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+137], v[vgprValuC+137] // check Nan
v_bfe_u32 v125, v[vgprValuC+137], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+137], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+137], v125, v127, s64
v_lshrrev_b32 v137, 16, v[vgprValuC+137]           // convert C to bf16
buffer_store_b16 v137, v162, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v163                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+138], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+138], v[vgprValuC+138] // check Nan
v_bfe_u32 v125, v[vgprValuC+138], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+138], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+138], v125, v127, s64
v_lshrrev_b32 v138, 16, v[vgprValuC+138]           // convert C to bf16
buffer_store_b16 v138, v164, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v165                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+139], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+139], v[vgprValuC+139] // check Nan
v_bfe_u32 v125, v[vgprValuC+139], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+139], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+139], v125, v127, s64
v_lshrrev_b32 v139, 16, v[vgprValuC+139]           // convert C to bf16
buffer_store_b16 v139, v166, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v167                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+140], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+140], v[vgprValuC+140] // check Nan
v_bfe_u32 v125, v[vgprValuC+140], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+140], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+140], v125, v127, s64
v_lshrrev_b32 v140, 16, v[vgprValuC+140]           // convert C to bf16
buffer_store_b16 v140, v168, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v169                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+141], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+141], v[vgprValuC+141] // check Nan
v_bfe_u32 v125, v[vgprValuC+141], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+141], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+141], v125, v127, s64
v_lshrrev_b32 v141, 16, v[vgprValuC+141]           // convert C to bf16
buffer_store_b16 v141, v170, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v171                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+142], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+142], v[vgprValuC+142] // check Nan
v_bfe_u32 v125, v[vgprValuC+142], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+142], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+142], v125, v127, s64
v_lshrrev_b32 v142, 16, v[vgprValuC+142]           // convert C to bf16
buffer_store_b16 v142, v172, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v173                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+143], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+143], v[vgprValuC+143] // check Nan
v_bfe_u32 v125, v[vgprValuC+143], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+143], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+143], v125, v127, s64
v_lshrrev_b32 v143, 16, v[vgprValuC+143]           // convert C to bf16
buffer_store_b16 v143, v174, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v122, 16, v175                       // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+144], v122, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s64, v[vgprValuC+144], v[vgprValuC+144] // check Nan
v_bfe_u32 v125, v[vgprValuC+144], 16, 1            // Non-Nan case: store lsb of bf16
v_add3_u32 v125, v[vgprValuC+144], v125, v128      // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+144], v125, v127, s64
v_lshrrev_b32 v144, 16, v[vgprValuC+144]           // convert C to bf16
buffer_store_b16 v144, v176, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_SK_Partials_1:
label_GW_Partials_E0:
s_mov_b64 s[sgprSrdWS:sgprSrdWS+1], s[sgprAddressWS:sgprAddressWS+1]
s_mov_b32 s[sgprSrdWS+2], BufferOOB
s_mov_b32 s[sgprSrdWS+3], Srd127_96

s_mul_i32 s7, 0x30000, s[sgprPersistentWorkGroupIndex] // Offset to correct partials tile (low word)
s_mul_hi_u32 s8, 0x30000, s[sgprPersistentWorkGroupIndex] // partials tile offset (high word) for 64-bit SRD
s_add_u32 s[sgprSrdWS+0], s[sgprSrdWS+0], s7       // add lo to SRD
s_addc_u32 s[sgprSrdWS+1], s[sgprSrdWS+1], s8      // add hi (offset high word + lo carry) to SRD

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=122 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 7                                          // optimization: sync and wait
s_barrier

/******************************************/
/* Partials Write Batch #0 (d1,d0,vc1,vc0) = */
/*      (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (8,2,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (9,2,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (10,2,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (11,2,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (12,2,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (13,2,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (14,2,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (15,2,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (16,2,0,0:vw1); (17,0,0,0:vw1); (17,1,0,0:vw1); (17,2,0,0:vw1); (18,0,0,0:vw1); (18,1,0,0:vw1); (18,2,0,0:vw1); (19,0,0,0:vw1); (19,1,0,0:vw1); (19,2,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1); (20,2,0,0:vw1); (21,0,0,0:vw1); (21,1,0,0:vw1); (21,2,0,0:vw1); (22,0,0,0:vw1); (22,1,0,0:vw1); (22,2,0,0:vw1); (23,0,0,0:vw1); (23,1,0,0:vw1); (23,2,0,0:vw1); (24,0,0,0:vw1); (24,1,0,0:vw1); (24,2,0,0:vw1); (25,0,0,0:vw1); (25,1,0,0:vw1); (25,2,0,0:vw1); (26,0,0,0:vw1); (26,1,0,0:vw1); (26,2,0,0:vw1); (27,0,0,0:vw1); (27,1,0,0:vw1); (27,2,0,0:vw1); (28,0,0,0:vw1); (28,1,0,0:vw1); (28,2,0,0:vw1); (29,0,0,0:vw1); (29,1,0,0:vw1); (29,2,0,0:vw1); (30,0,0,0:vw1); (30,1,0,0:vw1); (30,2,0,0:vw1); (31,0,0,0:vw1); (31,1,0,0:vw1); (31,2,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v[vgprValuC+131], v[vgprValuC+0]         // copy MI out reg to vreg[0]
v_mov_b32 v[vgprValuC+132], v[vgprValuC+8]         // copy MI out reg to vreg[1]
v_mov_b32 v[vgprValuC+133], v[vgprValuC+16]        // copy MI out reg to vreg[2]
v_mov_b32 v[vgprValuC+134], v[vgprValuC+1]         // copy MI out reg to vreg[3]
v_mov_b32 v[vgprValuC+135], v[vgprValuC+9]         // copy MI out reg to vreg[4]
v_mov_b32 v[vgprValuC+136], v[vgprValuC+17]        // copy MI out reg to vreg[5]
v_mov_b32 v[vgprValuC+137], v[vgprValuC+2]         // copy MI out reg to vreg[6]
v_mov_b32 v[vgprValuC+138], v[vgprValuC+10]        // copy MI out reg to vreg[7]
v_mov_b32 v[vgprValuC+139], v[vgprValuC+18]        // copy MI out reg to vreg[8]
v_mov_b32 v[vgprValuC+140], v[vgprValuC+3]         // copy MI out reg to vreg[9]
v_mov_b32 v[vgprValuC+141], v[vgprValuC+11]        // copy MI out reg to vreg[10]
v_mov_b32 v[vgprValuC+142], v[vgprValuC+19]        // copy MI out reg to vreg[11]
v_mov_b32 v[vgprValuC+143], v[vgprValuC+4]         // copy MI out reg to vreg[12]
v_mov_b32 v[vgprValuC+144], v[vgprValuC+12]        // copy MI out reg to vreg[13]
v_mov_b32 v[vgprValuC+145], v[vgprValuC+20]        // copy MI out reg to vreg[14]
v_mov_b32 v[vgprValuC+146], v[vgprValuC+5]         // copy MI out reg to vreg[15]
v_mov_b32 v[vgprValuC+147], v[vgprValuC+13]        // copy MI out reg to vreg[16]
v_mov_b32 v[vgprValuC+148], v[vgprValuC+21]        // copy MI out reg to vreg[17]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+6]         // copy MI out reg to vreg[18]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+14]        // copy MI out reg to vreg[19]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+22]        // copy MI out reg to vreg[20]
v_mov_b32 v[vgprValuC+152], v[vgprValuC+7]         // copy MI out reg to vreg[21]
v_mov_b32 v[vgprValuC+153], v[vgprValuC+15]        // copy MI out reg to vreg[22]
v_mov_b32 v[vgprValuC+154], v[vgprValuC+23]        // copy MI out reg to vreg[23]
v_mov_b32 v[vgprValuC+155], v[vgprValuC+24]        // copy MI out reg to vreg[24]
v_mov_b32 v[vgprValuC+156], v[vgprValuC+32]        // copy MI out reg to vreg[25]
v_mov_b32 v[vgprValuC+157], v[vgprValuC+40]        // copy MI out reg to vreg[26]
v_mov_b32 v[vgprValuC+158], v[vgprValuC+25]        // copy MI out reg to vreg[27]
v_mov_b32 v[vgprValuC+159], v[vgprValuC+33]        // copy MI out reg to vreg[28]
v_mov_b32 v[vgprValuC+160], v[vgprValuC+41]        // copy MI out reg to vreg[29]
v_mov_b32 v[vgprValuC+161], v[vgprValuC+26]        // copy MI out reg to vreg[30]
v_mov_b32 v[vgprValuC+162], v[vgprValuC+34]        // copy MI out reg to vreg[31]
v_mov_b32 v[vgprValuC+163], v[vgprValuC+42]        // copy MI out reg to vreg[32]
v_mov_b32 v[vgprValuC+164], v[vgprValuC+27]        // copy MI out reg to vreg[33]
v_mov_b32 v[vgprValuC+165], v[vgprValuC+35]        // copy MI out reg to vreg[34]
v_mov_b32 v[vgprValuC+166], v[vgprValuC+43]        // copy MI out reg to vreg[35]
v_mov_b32 v[vgprValuC+167], v[vgprValuC+28]        // copy MI out reg to vreg[36]
v_mov_b32 v[vgprValuC+168], v[vgprValuC+36]        // copy MI out reg to vreg[37]
v_mov_b32 v[vgprValuC+169], v[vgprValuC+44]        // copy MI out reg to vreg[38]
v_mov_b32 v[vgprValuC+170], v[vgprValuC+29]        // copy MI out reg to vreg[39]
v_mov_b32 v[vgprValuC+171], v[vgprValuC+37]        // copy MI out reg to vreg[40]
v_mov_b32 v[vgprValuC+172], v[vgprValuC+45]        // copy MI out reg to vreg[41]
v_mov_b32 v[vgprValuC+173], v[vgprValuC+30]        // copy MI out reg to vreg[42]
v_mov_b32 v[vgprValuC+174], v[vgprValuC+38]        // copy MI out reg to vreg[43]
v_mov_b32 v[vgprValuC+175], v[vgprValuC+46]        // copy MI out reg to vreg[44]
v_mov_b32 v[vgprValuC+176], v[vgprValuC+31]        // copy MI out reg to vreg[45]
v_mov_b32 v[vgprValuC+177], v[vgprValuC+39]        // copy MI out reg to vreg[46]
v_mov_b32 v[vgprValuC+178], v[vgprValuC+47]        // copy MI out reg to vreg[47]
v_mov_b32 v[vgprValuC+179], v[vgprValuC+48]        // copy MI out reg to vreg[48]
v_mov_b32 v[vgprValuC+180], v[vgprValuC+56]        // copy MI out reg to vreg[49]
v_mov_b32 v[vgprValuC+181], v[vgprValuC+64]        // copy MI out reg to vreg[50]
v_mov_b32 v[vgprValuC+182], v[vgprValuC+49]        // copy MI out reg to vreg[51]
v_mov_b32 v[vgprValuC+183], v[vgprValuC+57]        // copy MI out reg to vreg[52]
v_mov_b32 v[vgprValuC+184], v[vgprValuC+65]        // copy MI out reg to vreg[53]
v_mov_b32 v[vgprValuC+185], v[vgprValuC+50]        // copy MI out reg to vreg[54]
v_mov_b32 v[vgprValuC+186], v[vgprValuC+58]        // copy MI out reg to vreg[55]
v_mov_b32 v[vgprValuC+187], v[vgprValuC+66]        // copy MI out reg to vreg[56]
v_mov_b32 v[vgprValuC+188], v[vgprValuC+51]        // copy MI out reg to vreg[57]
v_mov_b32 v[vgprValuC+189], v[vgprValuC+59]        // copy MI out reg to vreg[58]
v_mov_b32 v[vgprValuC+190], v[vgprValuC+67]        // copy MI out reg to vreg[59]
v_mov_b32 v[vgprValuC+191], v[vgprValuC+52]        // copy MI out reg to vreg[60]
v_mov_b32 v[vgprValuC+192], v[vgprValuC+60]        // copy MI out reg to vreg[61]
v_mov_b32 v[vgprValuC+193], v[vgprValuC+68]        // copy MI out reg to vreg[62]
v_mov_b32 v[vgprValuC+194], v[vgprValuC+53]        // copy MI out reg to vreg[63]
v_mov_b32 v[vgprValuC+195], v[vgprValuC+61]        // copy MI out reg to vreg[64]
v_mov_b32 v[vgprValuC+196], v[vgprValuC+69]        // copy MI out reg to vreg[65]
v_mov_b32 v[vgprValuC+197], v[vgprValuC+54]        // copy MI out reg to vreg[66]
v_mov_b32 v[vgprValuC+198], v[vgprValuC+62]        // copy MI out reg to vreg[67]
v_mov_b32 v[vgprValuC+199], v[vgprValuC+70]        // copy MI out reg to vreg[68]
v_mov_b32 v[vgprValuC+200], v[vgprValuC+55]        // copy MI out reg to vreg[69]
v_mov_b32 v[vgprValuC+201], v[vgprValuC+63]        // copy MI out reg to vreg[70]
v_mov_b32 v[vgprValuC+202], v[vgprValuC+71]        // copy MI out reg to vreg[71]
v_mov_b32 v[vgprValuC+203], v[vgprValuC+72]        // copy MI out reg to vreg[72]
v_mov_b32 v[vgprValuC+206], v[vgprValuC+80]        // copy MI out reg to vreg[73]
v_mov_b32 v[vgprValuC+207], v[vgprValuC+88]        // copy MI out reg to vreg[74]
v_mov_b32 v[vgprValuC+208], v[vgprValuC+73]        // copy MI out reg to vreg[75]
v_mov_b32 v[vgprValuC+209], v[vgprValuC+81]        // copy MI out reg to vreg[76]
v_mov_b32 v[vgprValuC+210], v[vgprValuC+89]        // copy MI out reg to vreg[77]
v_mov_b32 v[vgprValuC+211], v[vgprValuC+74]        // copy MI out reg to vreg[78]
v_mov_b32 v[vgprValuC+212], v[vgprValuC+82]        // copy MI out reg to vreg[79]
v_mov_b32 v[vgprValuC+213], v[vgprValuC+90]        // copy MI out reg to vreg[80]
v_mov_b32 v[vgprValuC+214], v[vgprValuC+75]        // copy MI out reg to vreg[81]
v_mov_b32 v[vgprValuC+215], v[vgprValuC+83]        // copy MI out reg to vreg[82]
v_mov_b32 v[vgprValuC+216], v[vgprValuC+91]        // copy MI out reg to vreg[83]
v_mov_b32 v[vgprValuC+217], v[vgprValuC+76]        // copy MI out reg to vreg[84]
v_mov_b32 v[vgprValuC+218], v[vgprValuC+84]        // copy MI out reg to vreg[85]
v_mov_b32 v[vgprValuC+219], v[vgprValuC+92]        // copy MI out reg to vreg[86]
v_mov_b32 v[vgprValuC+220], v[vgprValuC+77]        // copy MI out reg to vreg[87]
v_mov_b32 v[vgprValuC+221], v[vgprValuC+85]        // copy MI out reg to vreg[88]
v_mov_b32 v[vgprValuC+222], v[vgprValuC+93]        // copy MI out reg to vreg[89]
v_mov_b32 v[vgprValuC+223], v[vgprValuC+78]        // copy MI out reg to vreg[90]
v_mov_b32 v[vgprValuC+224], v[vgprValuC+86]        // copy MI out reg to vreg[91]
v_mov_b32 v[vgprValuC+225], v[vgprValuC+94]        // copy MI out reg to vreg[92]
v_mov_b32 v[vgprValuC+226], v[vgprValuC+79]        // copy MI out reg to vreg[93]
v_mov_b32 v[vgprValuC+227], v[vgprValuC+87]        // copy MI out reg to vreg[94]
v_mov_b32 v[vgprValuC+228], v[vgprValuC+95]        // copy MI out reg to vreg[95]

/* apply mask, calc new C and issue writes */
v_mov_b32 v126, 0xffff0000                         // mask for pack two bfloat16 element to 32bit
v_mov_b32 v127, 0x7fff0000                         // fp32 Nan
v_mov_b32 v128, 0x7fff                             // rounding bias for bfloat16
v_lshlrev_b32 v129, 2, v[vgprSerial]               // v129 = v[vgprSerial] * 4
s_mov_b32 s8, 0                                    // Init sgpr offset
buffer_store_b32 v131, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v132, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v133, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v134, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v135, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v136, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v137, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v138, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v139, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v140, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v141, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v142, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v143, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v144, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v145, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v146, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v147, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v148, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v149, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v150, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v151, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v152, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v153, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v154, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v155, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v156, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v157, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v158, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v159, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v160, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v161, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v162, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v163, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v164, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v165, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v166, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v167, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v168, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v169, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v170, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v171, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v172, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v173, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v174, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v175, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v176, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v177, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v178, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v179, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v180, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v181, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v182, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v183, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v184, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v185, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v186, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v187, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v188, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v189, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v190, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v191, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v192, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v193, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v194, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v195, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v196, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v197, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v198, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v199, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v200, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v201, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v202, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v203, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v206, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v207, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v208, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v209, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v210, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v211, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v212, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v213, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v214, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v215, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v216, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v217, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v218, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v219, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v220, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v221, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v222, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v223, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v224, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v225, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v226, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v227, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 2048                             // Inc sgpr offset
buffer_store_b32 v228, v129, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_waitcnt_vscnt null 0                             // wait for data store
s_barrier                                          // store all data before setting flag
s_lshl_b32 s8, s[sgprPersistentWorkGroupIndex], 2  // flag offset based on physical partial slot
v_readfirstlane_b32 s7, v[vgprSerial]              // Wave 0 updates flags
s_cmp_eq_u32 s7, 0                                 // Check for wave 0
s_cbranch_scc0 label_SK_SkipFlagSet                // Skip flag set
s_mov_b32 s7, 1                                    // flag data
v_mov_b32 v131, s7                                 // move flag value to vgpr
v_mov_b32 v132, 0                                  // zero vaddr offset
s_mov_b64 s[64:65], s[sgprAddressFlags:sgprAddressFlags+1]
s_mov_b32 s66, BufferOOB
s_mov_b32 s67, Srd127_96
buffer_store_b32 v131, v132, s[64:67], s8 offen offset:0 glc dlc // set flag
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
