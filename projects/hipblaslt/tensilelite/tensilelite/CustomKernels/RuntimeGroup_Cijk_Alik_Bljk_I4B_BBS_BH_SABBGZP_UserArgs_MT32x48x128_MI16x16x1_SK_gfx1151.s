
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
.text
.protected RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBGZP_UserArgs_MT32x48x128_MI16x16x1_SK_gfx1151
.globl RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBGZP_UserArgs_MT32x48x128_MI16x16x1_SK_gfx1151
.p2align 8
.type RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBGZP_UserArgs_MT32x48x128_MI16x16x1_SK_gfx1151,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBGZP_UserArgs_MT32x48x128_MI16x16x1_SK_gfx1151
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 256 // vgprs
  .amdhsa_next_free_sgpr 95 // sgprs
  .amdhsa_group_segment_fixed_size 24576 // lds bytes
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
/* Num SGPR   =95 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 16 x 3 */
/* SubGroup= 2 x 16 */
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
  - .name: RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBGZP_UserArgs_MT32x48x128_MI16x16x1_SK_gfx1151
    .symbol: 'RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBGZP_UserArgs_MT32x48x128_MI16x16x1_SK_gfx1151.kd'
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
    .group_segment_fixed_size:   24576
    .kernarg_segment_align:      8
    .kernarg_segment_size:       208
    .max_flat_workgroup_size:    128
    .private_segment_fixed_size: 0
    .sgpr_count:                 95
    .sgpr_spill_count:           0
    .vgpr_count:                 256
    .vgpr_spill_count:           0
    .wavefront_size:             32
...
.end_amdgpu_metadata
RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBGZP_UserArgs_MT32x48x128_MI16x16x1_SK_gfx1151:
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
.set vgprBase, 70
.set vgprLocalWriteAddrA, 66
.set vgprLocalWriteAddrB, 67
.set vgprGlobalReadOffsetA, 48
.set vgprGlobalReadOffsetB, 50
.set vgprGlobalReadOffsetScaleA, 56
.set vgprG2LScaleA, 58
.set vgprG2LScaleZeroA, 60
.set vgprGlobalReadOffsetScaleZeroA, 62
.set vgprGlobalReadByteOffsetScaleZeroA, 64
.set vgprLocalReadAddrA, 68
.set vgprLocalReadAddrB, 69
.set vgprSerial, 192

/******************************************/
/* VGPR Macro Assignments                 */
/******************************************/
.set vgprValuA_X0_I0_BASE, vgprBase+0
.set vgprValuB_X0_I0_BASE, vgprBase+33
.set vgprG2LA_BASE, vgprBase+82
.set vgprG2LB_BASE, vgprBase+98
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+16
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
.set sgprLSUTailLoopOffset, 7
.set sgprLoopCounterL, 8
.set sgprOrigLoopCounter, 9
.set sgprNumWorkGroups0, 10
.set sgprNumWorkGroups1, 11
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
.set sgprAddressScaleZeroA, 76
.set sgprSrdScaleZeroA, 80

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
.set MT1, 48
.set DepthU, 128
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
s_load_b32 s44, s[sgprKernArgAddress:sgprKernArgAddress+1], 0

/* Load packed kernel args (StaggerU/GSU) */
s_load_b32 s46, s[sgprKernArgAddress:sgprKernArgAddress+1], 4

/* Load WGM data */
s_load_b32 s[sgprWGM], s[sgprKernArgAddress:sgprKernArgAddress+1], 8

/* Load num of WGs */
s_load_b32 s47, s[sgprKernArgAddress:sgprKernArgAddress+1], 12
s_waitcnt lgkmcnt(0)                               // load args
s_lshr_b32 s45, s44, 0x1e                          // Get arg type
s_and_b32 s44, 0x3fffffff, s44                     // Get nums of gemm
s_cmp_eq_u32 s45, 3                                // Is kernel argType == 3
s_cbranch_scc1 label_Bypass_ArgType3_to_ArgType0_Instance1
s_cmp_eq_u32 s45, 0                                // Is kernel args
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
s_mov_b32 s[sgprArgType], s45
s_mov_b32 m0, 0x6000                               // LDS clamp at 24576 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id
v_mov_b32 v[vgprG2LScaleA+0], 0                    // scaleA: low half of the f32 scale
v_mov_b32 v[vgprG2LScaleA+1], 0                    // scaleA: low half of the f32 scale
s_mov_b32 vcc_hi, 0                                // Ensure hi bits are zero

/* remap workgroup to XCCs */
s_lshr_b32 s88, s[sgprWGM], 0x10                   // Get WGMXCC
s_ff1_i32_b32 s88, s88                             // Get log(WGMXCC)
s_lshr_b32 s89, s[sgprWGM], 0x16                   // Get CU_Count
/* remap WGs if WGMXCC > 1 ( log(WGMXCC) > 0 ) */
s_cmp_gt_i32 s88, 0
s_cbranch_scc0 label_skip_WGMXCC
/* only remap WGs in the range */
s_lshr_b32 s85, s47, s88
s_lshl_b32 s85, s85, s88
s_cmp_ge_u32 s[sgprWorkGroup0], s85
s_cbranch_scc1 label_skip_WGMXCC
s_cmp_eq_u32 s89, 0                                // CU_Count == 0 ?
s_cbranch_scc0 label_XCCG_nonzero
s_lshr_b32 s85, s[sgprWorkGroup0], s88
s_bfm_b32 s86, s88, 0
s_and_b32 s86, s[sgprWorkGroup0], s86
s_lshr_b32 s87, s47, s88
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
v_cvt_f64_u32 v[0:1], s89                          // s86 = s47 / s89
v_rcp_f64 v[0:1], v[0:1]                           // s86 = s47 / s89
v_cvt_f64_u32 v[2:3], s47                          // s86 = s47 / s89
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s86 = s47 / s89
v_cvt_u32_f64 v0, v[0:1]                           // s86 = s47 / s89
v_mul_lo_u32 v1, v0, s89                           // s86 = s47 / s89
v_sub_nc_u32 v2, s47, v1                           // s86 = s47 / s89
v_cmp_ge_u32 vcc_lo, v2, s89                       // s86 = s47 / s89
s_mov_b32 exec_lo, vcc_lo                          // s86 = s47 / s89
v_add_nc_u32 v0, v0, 1                             // s86 = s47 / s89
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s86, v0                        // quotient
s_mul_i32 s86, s86, s89
s_sub_u32 s87, s47, s86
s_cmp_gt_u32 s[sgprWorkGroup0], s86
s_cselect_b32 s86, s87, s89
s_lshr_b32 s86, s86, s88
s_bfm_b32 s87, s88, 0
s_and_b32 s87, s[sgprWorkGroup0], s87
s_mul_i32 s86, s86, s87
/* WorkGroup0 = temp0 + temp1 */
s_add_u32 s[sgprWorkGroup0], s85, s86
label_skip_WGMXCC:  /// skip WGMXCC if no enough WGs to remap
s_cmp_eq_u32 s45, 3
s_cbranch_scc1 label_ArgType3_Routed_To_ArgType0
s_cmp_eq_u32 s45, 0
s_cbranch_scc0 label_MultiGemm
label_ArgType3_Routed_To_ArgType0:
/* init: add vgpr [70...221) to pool */
/* init: add vgpr [0...48) to pool */
/* init: add agpr [0...0) to pool */
s_waitcnt lgkmcnt(0)                               // wait for 128/0 bytes of kern args over preload
s_lshr_b32 s[sgprNumWorkGroups0], s[sgprSizesFree+0], 5 // s[sgprNumWorkGroups0] = s[sgprSizesFree+0] / 32
s_and_b32 s46, 31, s[sgprSizesFree+0]              // s46 = s[sgprSizesFree+0] % 32
s_cmp_gt_u32 s46, 0                                // Check if remainder > 0
s_addc_u32 s[sgprNumWorkGroups0], s[sgprNumWorkGroups0], 0 // Increment if remainder > 0
s_mul_hi_u32 s79, s[sgprSizesFree+1], 178956971    // STATIC_DIV: divisor=48
s_mul_i32 s78, s[sgprSizesFree+1], 178956971       // tmp = dividend * magic
s_lshr_b64 s[78:79], s[78:79], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s[sgprNumWorkGroups1], s78               // quotient
s_mul_i32 s78, s[sgprNumWorkGroups1], 48           // quotient*divisor
s_sub_u32 s46, s[sgprSizesFree+1], s78             // rReg = dividend - quotient*divisor
s_cmp_gt_u32 s46, 0                                // Check if remainder > 0
s_addc_u32 s[sgprNumWorkGroups1], s[sgprNumWorkGroups1], 0 // Increment if remainder > 0
s_branch label_MultiGemmEnd
label_MultiGemm:

/* Check if custom structure pointer is null */
s_and_b32 s45, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s45, 2                                // ArgType == 2 ?
s_cbranch_scc1 label_IsExternalValid               // branch if ArgType == 2
s_mov_b32 s11, 152                                 // KernArgAddressOffset
s_mul_i32 s90, s44, 4
s_mov_b64 s[84:85], s[sgprKernArgAddress:sgprKernArgAddress+1]
s_branch label_IsExternalValidEnd
label_IsExternalValid:
s_mov_b32 s11, 256
s_mov_b32 s90, 0
s_mov_b64 s[84:85], s[sgprKernArgAddress:sgprKernArgAddress+1]
label_IsExternalValidEnd:

/* Grouped Gemm:: prefetch 1 arg load */
s_mov_b32 s10, 1
s_mov_b32 s91, 0
s_load_b128 s[12:15], s[84:85], s90
s_cmpk_eq_u32 s44, 1                               // if gemm_count is 1?
s_cbranch_scc1 label_wgTable_noLoadLoop

/* Grouped Gemm:: accumulate numTiles for each gemm */
/* Grouped Gemm:: loop start */
label_Loop_GemmCount:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s88, s12, 5                             // s88 = s12 / 32
s_and_b32 s86, 31, s12                             // s86 = s12 % 32
s_addc_u32 s88, s88, 0
s_mul_hi_u32 s87, s13, 178956971                   // STATIC_DIV: divisor=48
s_mul_i32 s86, s13, 178956971                      // tmp = dividend * magic
s_lshr_b64 s[86:87], s[86:87], 33                  // tmp0 = quotient
s_mul_i32 s87, s86, 48                             // tmp1 = quotient * divisor
s_cmp_lg_u32 s87, s13                              // if (quotient * divisor != dividend), result+=1
s_addc_u32 s89, s86, 0                             // if (quotient * divisor != dividend), result+=1
s_mul_i32 s88, s88, s89
s_mul_i32 s88, s88, s14
s_add_u32 s91, s91, s88
s_cmp_lt_u32 s[sgprWorkGroup0], s91
s_cbranch_scc1 label_FOUND
s_add_u32 s90, s90, s11
s_load_b128 s[12:15], s[84:85], s90
s_add_u32 s10, s10, 1
s_cmp_lt_u32 s10, s44
s_cbranch_scc1 label_Loop_GemmCount

/* Grouped Gemm:: noLoadLoop */
label_wgTable_noLoadLoop:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s88, s12, 5                             // s88 = s12 / 32
s_and_b32 s86, 31, s12                             // s86 = s12 % 32
s_addc_u32 s88, s88, 0
s_mul_hi_u32 s87, s13, 178956971                   // STATIC_DIV: divisor=48
s_mul_i32 s86, s13, 178956971                      // tmp = dividend * magic
s_lshr_b64 s[86:87], s[86:87], 33                  // tmp0 = quotient
s_mul_i32 s87, s86, 48                             // tmp1 = quotient * divisor
s_cmp_lg_u32 s87, s13                              // if (quotient * divisor != dividend), result+=1
s_addc_u32 s89, s86, 0                             // if (quotient * divisor != dividend), result+=1
s_mul_i32 s88, s88, s89
s_mul_i32 s88, s88, s14
s_add_u32 s91, s91, s88

/* Grouped Gemm:: gemmIndex found */
label_FOUND:
s_sub_u32 s85, s10, 1
s_sub_u32 s84, s91, s88
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s84
/* Check if custom structure pointer is null */
s_and_b32 s45, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s45, 2                                // ArgType == 2 ?
s_cbranch_scc1 label_LoadExternalStruct            // branch if ArgType == 2

/* Grouped Gemm: offset argument address to gemm */
/* Grouped Gemm: offset address from wg_table_start to args_start */
s_lshl2_add_u32 s[sgprKernArgAddress], s44, s[sgprKernArgAddress]
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s85, s85, 152                            // KernArgAddressOffset
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s85
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
s_mul_i32 s85, s85, 256
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s85
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
/* init: add vgpr [70...221) to pool */
/* init: add vgpr [0...48) to pool */
/* init: add agpr [0...0) to pool */
s_waitcnt lgkmcnt(0)                               // wait for 128/0 bytes of kern args over preload
s_lshr_b32 s[sgprNumWorkGroups0], s[sgprSizesFree+0], 5 // s[sgprNumWorkGroups0] = s[sgprSizesFree+0] / 32
s_and_b32 s46, 31, s[sgprSizesFree+0]              // s46 = s[sgprSizesFree+0] % 32
s_cmp_gt_u32 s46, 0                                // Check if remainder > 0
s_addc_u32 s[sgprNumWorkGroups0], s[sgprNumWorkGroups0], 0 // Increment if remainder > 0
s_mul_hi_u32 s79, s[sgprSizesFree+1], 178956971    // STATIC_DIV: divisor=48
s_mul_i32 s78, s[sgprSizesFree+1], 178956971       // tmp = dividend * magic
s_lshr_b64 s[78:79], s[78:79], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s[sgprNumWorkGroups1], s78               // quotient
s_mul_i32 s78, s[sgprNumWorkGroups1], 48           // quotient*divisor
s_sub_u32 s46, s[sgprSizesFree+1], s78             // rReg = dividend - quotient*divisor
s_cmp_gt_u32 s46, 0                                // Check if remainder > 0
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
.set sgprShadowLimitA, 78
.set sgprShadowLimitB, 92
s_and_b32 s44, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s44, 3                                // ArgType == 3 for General Batched GEMM
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
v_readfirstlane_b32 s44, v0                        // quotient
v_readfirstlane_b32 s45, v1                        // remainder
s_mul_i32 s46, s[sgprskTiles], s[sgprSKItersPerWG]
s_sub_u32 s46, s[sgprItersPerTile], s46            // extraIters = itersPerTile - SkSplit * skItersPerWG
s_mul_i32 s[sgprPersistentIteration], s45, s[sgprSKItersPerWG] // StreamK starting iteration (case: after extra iters)
s_cmp_lt_u32 s45, s46                              // Check if WG gets an extra iteration
s_cbranch_scc1 label_SK_HasExtra                   // Has extra iter
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s46 // This WG does not have an extra iteration
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIteration], s[sgprSKItersPerWG] // StreamK ending iteration (case: after extra iters)
s_branch label_SK_DoneExtra                        // Done init for parallel reduction
label_SK_HasExtra:
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s45 // This WG has an extra iteration
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIteration], s[sgprSKItersPerWG] // StreamK ending iteration (case: after extra iters)
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], 1 // StreamK ending iteration (case: after extra iters)
label_SK_DoneExtra:
s_mul_i32 s44, s44, s[sgprItersPerTile]            // Tile offset = tilesIdx * itersPerTile
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s44 // Offset to correct tile
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], s44 // Offset to correct tile
s_mov_b32 s[sgprSkPartialIdx], s45                 // Save partial idx for SrdD calculation
s_branch label_SK_InitDone                         // Done init for parallel reduction
label_SK_SplitInit:
s_mul_i32 s[sgprPersistentIteration], s[sgprPersistentWorkGroupIndex], s[sgprItersPerTile] // DP starting iteration (case: DP work to do)
s_mul_i32 s44, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1] // totalTiles = nwg0 * nwg1
s_mul_i32 s44, s44, s[sgprSizesFree+2]             // totalTiles *= batch dim 0
s_mul_i32 s44, s44, s[sgprItersPerTile]            // totalIters = totalTiles * itersPerTile
s_mov_b32 s[sgprPersistentIterationEnd], s44       // DP ending iteration (case: only DP work to do)
s_mul_i32 s44, s[sgprskTiles], s[sgprItersPerTile] // Total SK iters
s_cmp_lt_u32 s44, s[sgprPersistentIterationEnd]    // Check if there are DP tiles to do
s_cbranch_scc1 label_SK_InitDone                   // Done init
s_mul_i32 s44, s[sgprskTiles], s[sgprItersPerTile]
s_mul_i32 s45, s[sgprSKItersPerWG], s[sgprskGrid]
s_sub_u32 s44, s44, s45                            // skTiles * ItersPerTile - SKItersPerWG * skGrid
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
v_readfirstlane_b32 s45, v0                        // quotient
v_readfirstlane_b32 s46, v1                        // remainder
s_cmp_eq_u32 s46, 0                                // skGrid % skTiles == 0?
s_cbranch_scc1 label_SK_PerTileExtraIters          // all-partial -> per-tile extras
s_branch label_SK_GlobalExtraIters                 // ragged -> global mapping
label_SK_AssignNoTiles:
label_SK_GlobalExtraIters:
s_mul_i32 s[sgprPersistentIteration], s[sgprPersistentWorkGroupIndex], s[sgprSKItersPerWG] // StreamK starting iteration (case: after extra iters)
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s44 // Add extra iters
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIteration], s[sgprSKItersPerWG] // StreamK ending iteration (case: after extra iters)
s_add_u32 s46, s[sgprSKItersPerWG], 1              // Spread out extra iterations
s_mul_i32 s45, s[sgprPersistentWorkGroupIndex], s46 // StreamK starting iteration (case: before extra iters)
s_add_u32 s46, s45, s46                            // StreamK ending iteration (case: before extra iters)
s_cmp_lt_u32 s[sgprPersistentWorkGroupIndex], s44  // Check if lane gets an extra iteration
s_cselect_b32 s[sgprPersistentIteration], s45, s[sgprPersistentIteration] // Set start iter
s_cselect_b32 s[sgprPersistentIterationEnd], s46, s[sgprPersistentIterationEnd] // Set end iter
s_branch label_SK_AssignItersDone                  // skip per-tile path
label_SK_PerTileExtraIters:
s_mov_b32 s44, s45                                 // F = skGrid / skTiles
v_cvt_f32_u32 v0, s44                              // q = w/F, s = w%F
v_rcp_iflag_f32 v0, v0                             // q = w/F, s = w%F
v_mov_b32 v1, 0x4f7ffffe                           // 2^32 - 512
v_mul_f32 v0, v1, v0                               // 2^32 / divisor
v_cvt_u32_f32 v0, v0                               // q = w/F, s = w%F
v_mul_lo_u32 v1, v0, s44                           // refine 2^32 / divisor
v_sub_nc_u32 v1, 0, v1                             // refine 2^32 / divisor
v_mul_hi_u32 v1, v0, v1                            // refine 2^32 / divisor
v_add_nc_u32 v0, v0, v1                            // refine 2^32 / divisor
v_mul_hi_u32 v0, s[sgprPersistentWorkGroupIndex], v0 // q = w/F, s = w%F
v_mul_lo_u32 v1, v0, s44                           // q = w/F, s = w%F
v_sub_nc_u32 v1, s[sgprPersistentWorkGroupIndex], v1 // q = w/F, s = w%F
v_cmp_ge_u32 vcc_lo, v1, s44                       // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v0, 1, v0                             // quotient + 1
v_sub_nc_u32 v1, v1, s44                           // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v1, s44                       // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v0, 1, v0                             // quotient + 1
v_sub_nc_u32 v1, v1, s44                           // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s45, v0                        // quotient
v_readfirstlane_b32 s46, v1                        // remainder
s_mul_i32 s[sgprPersistentIteration], s45, s[sgprItersPerTile] // q * ItersPerTile
s_mul_i32 s45, s46, s[sgprSKItersPerWG]            // s * SKItersPerWG
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s45 // q*I + s*W
s_mul_i32 s45, s44, s[sgprSKItersPerWG]            // F * SKItersPerWG
s_sub_u32 s45, s[sgprItersPerTile], s45            // remI = ItersPerTile - F*SKItersPerWG
s_min_u32 s44, s46, s45                            // min(s, remI)
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s44 // start = q*I + s*W + min(s, remI)
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIteration], s[sgprSKItersPerWG] // start + SKItersPerWG
s_cmp_lt_u32 s46, s45                              // s < remI?
s_cselect_b32 s44, 1, 0                            // extra iter within tile
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], s44 // end = start + W + (s < remI)
label_SK_AssignItersDone:
s_mul_i32 s44, s[sgprskTiles], s[sgprItersPerTile] // Total SK iters
s_min_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], s44 // Cap ending iter at total SK iters
label_SK_InitDone:
s_mul_i32 s44, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1] // totalTiles = nwg0 * nwg1
s_mul_i32 s44, s44, s[sgprSizesFree+2]             // totalTiles *= batch dim 0
s_mul_i32 s44, s44, s[sgprItersPerTile]            // totalIters = totalTiles * itersPerTile
s_cmp_lt_u32 s[sgprPersistentIteration], s44       // Make sure there's work to do
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
s_mov_b32 s78, s47                                 // park dpSectionSize
s_mov_b32 s79, s44                                 // park current tile idx
s_mul_i32 s44, s[sgprskTiles], s[sgprItersPerTile]
s_mul_i32 s45, s[sgprSKItersPerWG], s[sgprskGrid]
s_sub_u32 s44, s44, s45                            // skTiles * ItersPerTile - SKItersPerWG * skGrid
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
v_readfirstlane_b32 s45, v0                        // quotient
v_readfirstlane_b32 s46, v1                        // remainder
s_cmp_eq_u32 s46, 0                                // skGrid % skTiles == 0?
s_cbranch_scc1 label_SK_PerTileExtraIters_1        // all-partial -> per-tile extras
s_branch label_SK_GlobalExtraIters_1               // ragged -> global mapping
label_SK_AssignNoTiles_1:
label_SK_GlobalExtraIters_1:
s_mul_i32 s[sgprPersistentIteration], s[sgprPersistentWorkGroupIndex], s[sgprSKItersPerWG] // StreamK starting iteration (case: after extra iters)
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s44 // Add extra iters
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIteration], s[sgprSKItersPerWG] // StreamK ending iteration (case: after extra iters)
s_add_u32 s46, s[sgprSKItersPerWG], 1              // Spread out extra iterations
s_mul_i32 s45, s[sgprPersistentWorkGroupIndex], s46 // StreamK starting iteration (case: before extra iters)
s_add_u32 s46, s45, s46                            // StreamK ending iteration (case: before extra iters)
s_cmp_lt_u32 s[sgprPersistentWorkGroupIndex], s44  // Check if lane gets an extra iteration
s_cselect_b32 s[sgprPersistentIteration], s45, s[sgprPersistentIteration] // Set start iter
s_cselect_b32 s[sgprPersistentIterationEnd], s46, s[sgprPersistentIterationEnd] // Set end iter
s_branch label_SK_AssignItersDone_1                // skip per-tile path
label_SK_PerTileExtraIters_1:
s_mov_b32 s44, s45                                 // F = skGrid / skTiles
v_cvt_f32_u32 v0, s44                              // q = w/F, s = w%F
v_rcp_iflag_f32 v0, v0                             // q = w/F, s = w%F
v_mov_b32 v1, 0x4f7ffffe                           // 2^32 - 512
v_mul_f32 v0, v1, v0                               // 2^32 / divisor
v_cvt_u32_f32 v0, v0                               // q = w/F, s = w%F
v_mul_lo_u32 v1, v0, s44                           // refine 2^32 / divisor
v_sub_nc_u32 v1, 0, v1                             // refine 2^32 / divisor
v_mul_hi_u32 v1, v0, v1                            // refine 2^32 / divisor
v_add_nc_u32 v0, v0, v1                            // refine 2^32 / divisor
v_mul_hi_u32 v0, s[sgprPersistentWorkGroupIndex], v0 // q = w/F, s = w%F
v_mul_lo_u32 v1, v0, s44                           // q = w/F, s = w%F
v_sub_nc_u32 v1, s[sgprPersistentWorkGroupIndex], v1 // q = w/F, s = w%F
v_cmp_ge_u32 vcc_lo, v1, s44                       // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v0, 1, v0                             // quotient + 1
v_sub_nc_u32 v1, v1, s44                           // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v1, s44                       // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v0, 1, v0                             // quotient + 1
v_sub_nc_u32 v1, v1, s44                           // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s45, v0                        // quotient
v_readfirstlane_b32 s46, v1                        // remainder
s_mul_i32 s[sgprPersistentIteration], s45, s[sgprItersPerTile] // q * ItersPerTile
s_mul_i32 s45, s46, s[sgprSKItersPerWG]            // s * SKItersPerWG
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s45 // q*I + s*W
s_mul_i32 s45, s44, s[sgprSKItersPerWG]            // F * SKItersPerWG
s_sub_u32 s45, s[sgprItersPerTile], s45            // remI = ItersPerTile - F*SKItersPerWG
s_min_u32 s44, s46, s45                            // min(s, remI)
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s44 // start = q*I + s*W + min(s, remI)
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIteration], s[sgprSKItersPerWG] // start + SKItersPerWG
s_cmp_lt_u32 s46, s45                              // s < remI?
s_cselect_b32 s44, 1, 0                            // extra iter within tile
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], s44 // end = start + W + (s < remI)
label_SK_AssignItersDone_1:
s_mov_b32 s44, s79                                 // restore current tile idx
s_add_u32 s45, s[sgprPersistentIteration], s78     // Offset to start of SK section
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], s78 // Offset to start of SK section
s_mul_i32 s78, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1] // totalTiles = nwg0 * nwg1
s_mul_i32 s78, s78, s[sgprSizesFree+2]             // totalTiles *= batch dim 0
s_mul_i32 s78, s78, s[sgprItersPerTile]            // totalIters = totalTiles * itersPerTile
s_min_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], s78 // Cap ending iter at total SK iters
s_cmp_lt_u32 s[sgprPersistentIteration], s78       // Make sure there's work to do
s_cbranch_scc1 label_NoBranch_2                    // Only branch on scc0
s_getpc_b64 s[84:85]                               // addr of next instr
s_add_i32 s86, label_KernelEnd, 4                  // target branch offset
s_add_u32 s84, s84, s86                            // add target branch offset
s_addc_u32 s85, s85, 0                             // add high and carry
s_setpc_b64 s[84:85]                               // branch to label_KernelEnd
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
s_getpc_b64 s[84:85]                               // addr of next instr
s_add_i32 s86, label_PersistentLoopClose, 4        // target branch offset
s_add_u32 s84, s84, s86                            // add target branch offset
s_addc_u32 s85, s85, 0                             // add high and carry
s_setpc_b64 s[84:85]                               // branch to label_PersistentLoopClose
label_NoBranch_4:
s_mov_b32 s[sgprStreamKLocalEnd], s[sgprItersPerTile] // Skip iterations
label_SKAlphaCheck:
/* WGM Calculation */
s_mov_b32 s44, s[sgprWGM]                          // Restore WGM
s_sext_i32_i16 s44, s44                            // Restore WGM
s_cmp_gt_i32 s44, 1                                // WGM > 1 ?
s_cbranch_scc1 label_WGMPositive                   // branch if WGM > 1
s_cmp_ge_i32 s44, 0                                // WGM >= 0 ?
s_cbranch_scc1 label_WGM                           // branch if WGM >= 0
s_abs_i32 s44, s44                                 // abs(WGM)
v_cvt_f64_u32 v[0:1], s44                          // s45 = s[sgprWorkGroup0] / s44
v_rcp_f64 v[0:1], v[0:1]                           // s45 = s[sgprWorkGroup0] / s44
v_cvt_f64_u32 v[2:3], s[sgprWorkGroup0]            // s45 = s[sgprWorkGroup0] / s44
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s45 = s[sgprWorkGroup0] / s44
v_cvt_u32_f64 v0, v[0:1]                           // s45 = s[sgprWorkGroup0] / s44
v_mul_lo_u32 v1, v0, s44                           // s45 = s[sgprWorkGroup0] / s44
v_sub_nc_u32 v2, s[sgprWorkGroup0], v1             // s45 = s[sgprWorkGroup0] / s44
v_cmp_ge_u32 vcc_lo, v2, s44                       // s45 = s[sgprWorkGroup0] / s44
s_mov_b32 exec_lo, vcc_lo                          // s45 = s[sgprWorkGroup0] / s44
v_add_nc_u32 v0, v0, 1                             // s45 = s[sgprWorkGroup0] / s44
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s45, v0                        // quotient
s_mul_i32 s78, s45, s44                            // quotient * non-magic divisor
s_sub_u32 s78, s[sgprWorkGroup0], s78              // WorkGroup0=remainder
s_mul_i32 s78, s78, s[sgprNumWorkGroups1]          // (wg1 % WGM)*NumWorkGroups1
s_add_u32 s78, s78, s[sgprWorkGroup1]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups1
v_cvt_f64_u32 v[0:1], s44                          // s46 = s[sgprNumWorkGroups0] / s44
v_rcp_f64 v[0:1], v[0:1]                           // s46 = s[sgprNumWorkGroups0] / s44
v_cvt_f64_u32 v[2:3], s[sgprNumWorkGroups0]        // s46 = s[sgprNumWorkGroups0] / s44
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s46 = s[sgprNumWorkGroups0] / s44
v_cvt_u32_f64 v0, v[0:1]                           // s46 = s[sgprNumWorkGroups0] / s44
v_mul_lo_u32 v1, v0, s44                           // s46 = s[sgprNumWorkGroups0] / s44
v_sub_nc_u32 v2, s[sgprNumWorkGroups0], v1         // s46 = s[sgprNumWorkGroups0] / s44
v_cmp_ge_u32 vcc_lo, v2, s44                       // s46 = s[sgprNumWorkGroups0] / s44
s_mov_b32 exec_lo, vcc_lo                          // s46 = s[sgprNumWorkGroups0] / s44
v_add_nc_u32 v0, v0, 1                             // s46 = s[sgprNumWorkGroups0] / s44
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s46, v0                        // quotient
s_mul_i32 s47, s44, s46                            // quotient * non-magic divisor
s_sub_u32 s47, s[sgprNumWorkGroups0], s47          // NumWorkGroups0=remainder
s_cmp_eq_u32 s47, 0                                // remainder == 0 ?
s_cmov_b32 s47, s44                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s45, s46                              // blockId >= numFullBlocks ?
s_cselect_b32 s46, s47, s44
v_cvt_f64_u32 v[0:1], s46                          // s[sgprWorkGroup1] = s78 / s46
v_rcp_f64 v[0:1], v[0:1]                           // s[sgprWorkGroup1] = s78 / s46
v_cvt_f64_u32 v[2:3], s78                          // s[sgprWorkGroup1] = s78 / s46
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s[sgprWorkGroup1] = s78 / s46
v_cvt_u32_f64 v0, v[0:1]                           // s[sgprWorkGroup1] = s78 / s46
v_mul_lo_u32 v1, v0, s46                           // s[sgprWorkGroup1] = s78 / s46
v_sub_nc_u32 v2, s78, v1                           // s[sgprWorkGroup1] = s78 / s46
v_cmp_ge_u32 vcc_lo, v2, s46                       // s[sgprWorkGroup1] = s78 / s46
s_mov_b32 exec_lo, vcc_lo                          // s[sgprWorkGroup1] = s78 / s46
v_add_nc_u32 v0, v0, 1                             // s[sgprWorkGroup1] = s78 / s46
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v1, v0, s46                           // s[sgprWorkGroup1] = s78 / s46
v_sub_nc_u32 v2, s78, v1                           // s[sgprWorkGroup1] = s78 / s46
v_readfirstlane_b32 s[sgprWorkGroup1], v0          // quotient
v_readfirstlane_b32 s[sgprWorkGroup0], v2          // remainder
s_mul_i32 s[sgprWorkGroup0], s[sgprWorkGroup1], s46 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup0], s78, s[sgprWorkGroup0] // WorkGroup0=remainder
s_mul_i32 s45, s45, s44                            // blockId * WGM
s_add_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s45 // wg1 += blockId * WGM
s_branch label_WGM
label_WGMPositive:
s_mov_b32 s44, s44                                 // WGM
v_cvt_f64_u32 v[0:1], s44                          // s45 = s[sgprWorkGroup1] / s44
v_rcp_f64 v[0:1], v[0:1]                           // s45 = s[sgprWorkGroup1] / s44
v_cvt_f64_u32 v[2:3], s[sgprWorkGroup1]            // s45 = s[sgprWorkGroup1] / s44
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s45 = s[sgprWorkGroup1] / s44
v_cvt_u32_f64 v0, v[0:1]                           // s45 = s[sgprWorkGroup1] / s44
v_mul_lo_u32 v1, v0, s44                           // s45 = s[sgprWorkGroup1] / s44
v_sub_nc_u32 v2, s[sgprWorkGroup1], v1             // s45 = s[sgprWorkGroup1] / s44
v_cmp_ge_u32 vcc_lo, v2, s44                       // s45 = s[sgprWorkGroup1] / s44
s_mov_b32 exec_lo, vcc_lo                          // s45 = s[sgprWorkGroup1] / s44
v_add_nc_u32 v0, v0, 1                             // s45 = s[sgprWorkGroup1] / s44
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s45, v0                        // quotient
s_mul_i32 s78, s45, s44                            // quotient * non-magic divisor
s_sub_u32 s78, s[sgprWorkGroup1], s78              // WorkGroup1=remainder
s_mul_i32 s78, s78, s[sgprNumWorkGroups0]          // (wg1 % WGM)*NumWorkGroups0
s_add_u32 s78, s78, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups0
v_cvt_f64_u32 v[0:1], s44                          // s46 = s[sgprNumWorkGroups1] / s44
v_rcp_f64 v[0:1], v[0:1]                           // s46 = s[sgprNumWorkGroups1] / s44
v_cvt_f64_u32 v[2:3], s[sgprNumWorkGroups1]        // s46 = s[sgprNumWorkGroups1] / s44
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s46 = s[sgprNumWorkGroups1] / s44
v_cvt_u32_f64 v0, v[0:1]                           // s46 = s[sgprNumWorkGroups1] / s44
v_mul_lo_u32 v1, v0, s44                           // s46 = s[sgprNumWorkGroups1] / s44
v_sub_nc_u32 v2, s[sgprNumWorkGroups1], v1         // s46 = s[sgprNumWorkGroups1] / s44
v_cmp_ge_u32 vcc_lo, v2, s44                       // s46 = s[sgprNumWorkGroups1] / s44
s_mov_b32 exec_lo, vcc_lo                          // s46 = s[sgprNumWorkGroups1] / s44
v_add_nc_u32 v0, v0, 1                             // s46 = s[sgprNumWorkGroups1] / s44
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s46, v0                        // quotient
s_mul_i32 s47, s44, s46                            // quotient * non-magic divisor
s_sub_u32 s47, s[sgprNumWorkGroups1], s47          // NumWorkGroups1=remainder
s_cmp_eq_u32 s47, 0                                // remainder == 0 ?
s_cmov_b32 s47, s44                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s45, s46                              // blockId >= numFullBlocks ?
s_cselect_b32 s46, s47, s44
v_cvt_f64_u32 v[0:1], s46                          // s[sgprWorkGroup0] = s78 / s46
v_rcp_f64 v[0:1], v[0:1]                           // s[sgprWorkGroup0] = s78 / s46
v_cvt_f64_u32 v[2:3], s78                          // s[sgprWorkGroup0] = s78 / s46
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s[sgprWorkGroup0] = s78 / s46
v_cvt_u32_f64 v0, v[0:1]                           // s[sgprWorkGroup0] = s78 / s46
v_mul_lo_u32 v1, v0, s46                           // s[sgprWorkGroup0] = s78 / s46
v_sub_nc_u32 v2, s78, v1                           // s[sgprWorkGroup0] = s78 / s46
v_cmp_ge_u32 vcc_lo, v2, s46                       // s[sgprWorkGroup0] = s78 / s46
s_mov_b32 exec_lo, vcc_lo                          // s[sgprWorkGroup0] = s78 / s46
v_add_nc_u32 v0, v0, 1                             // s[sgprWorkGroup0] = s78 / s46
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v1, v0, s46                           // s[sgprWorkGroup0] = s78 / s46
v_sub_nc_u32 v2, s78, v1                           // s[sgprWorkGroup0] = s78 / s46
v_readfirstlane_b32 s[sgprWorkGroup0], v0          // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v2          // remainder
s_mul_i32 s[sgprWorkGroup1], s[sgprWorkGroup0], s46 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup1], s78, s[sgprWorkGroup1] // WorkGroup1=remainder
s_mul_i32 s45, s45, s44                            // blockId * WGM
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s45 // wg1 += blockId * WGM
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
/* lr1J */
v_and_b32 v2, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v1, 7, v1                            // 1. N offset: nOffset = nIdx * nStride(128)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)

/* local read addresses: final offsets a */
v_lshrrev_b32 v2, 5, v[vgprSerial]                 // 2 = Serial / 32
v_lshrrev_b32 v2, 0, v2                            // LSU offset: Get LSU wave_id
s_mov_b32 s44, 32                                  // LSU offset: stride = lsuStride(32) when umlds==True
v_mul_lo_u32 v2, s44, v2                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_nc_u32 v[vgprLocalReadAddrA], v2, v0         // Final Offset: offset = (lro0+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrA], 1, v[vgprLocalReadAddrA] //  (multiple bpe)
v_lshrrev_b32 v3, 8, v[vgprLocalReadAddrA]         // Final Offset: padding 16 per block 256
v_lshl_add_u32 v[vgprLocalReadAddrA], v3, 4, v[vgprLocalReadAddrA] // Final Offset: padding 16 per block 256

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 5, v[vgprSerial]                 // 0 = Serial / 32
v_lshrrev_b32 v0, 0, v0                            // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(32) when umlds==True (dup assign opt.)
v_mul_lo_u32 v0, s44, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_nc_u32 v[vgprLocalReadAddrB], v0, v1         // Final Offset: offset = (lro1+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrB], 1, v[vgprLocalReadAddrB] //  (multiple bpe)
v_lshrrev_b32 v2, 8, v[vgprLocalReadAddrB]         // Final Offset: padding 16 per block 256
v_lshl_add_u32 v[vgprLocalReadAddrB], v2, 4, v[vgprLocalReadAddrB] // Final Offset: padding 16 per block 256

/* local read addresses: declare addresses a */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, 0x2200, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

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
v_lshrrev_b32 v6, 8, v[vgprLocalWriteAddrA]        // padding 16 per block 256
v_lshl_add_u32 v[vgprLocalWriteAddrA], v6, 4, v[vgprLocalWriteAddrA] // padding 16 per block 256

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x80, v2     // lwBL**(DepthU_Compute + PAD)
v_add_nc_u32 v[vgprLocalWriteAddrB], v5, v[vgprLocalWriteAddrB] // lwFOB = (lwBB + lwBL*(DepthU+PAD))
v_lshlrev_b32 v[vgprLocalWriteAddrB], 1, v[vgprLocalWriteAddrB] //  (multiple bpe)
v_lshrrev_b32 v6, 8, v[vgprLocalWriteAddrB]        // padding 16 per block 256
v_lshl_add_u32 v[vgprLocalWriteAddrB], v6, 4, v[vgprLocalWriteAddrB] // padding 16 per block 256
v_add_co_u32 v[vgprLocalWriteAddrB], vcc_lo, 0x2200, v[vgprLocalWriteAddrB] // lwFOB = lw1J + lwL*MT1J + LDS_OFFSET_B=8704

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
v_add_co_u32 v9, vcc_lo, 8, v8                     // groB1J_1 += LSPB
v_add_co_u32 v10, vcc_lo, 8, v9                    // groB1J_2 += LSPB
v_add_co_u32 v11, vcc_lo, 8, v10                   // groB1J_3 += LSPB
v_add_co_u32 v12, vcc_lo, 8, v11                   // groB1J_4 += LSPB
v_add_co_u32 v13, vcc_lo, 8, v12                   // groB1J_5 += LSPB

/* global read addresses: unroll offsets a */
v_mov_b32 v14, v1                                  // groAL_0

/* global read addresses: unroll offsets b */
v_mov_b32 v15, v3                                  // groBL_0

/* global read addresses: addresses a */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s47, s[sgprWorkGroup0], 32            // WorkGroup[01] * MT
s_mul_i32 s46, s[sgprWorkGroup0], 32               // WorkGroup[01] * MT
s_mul_hi_u32 s47, s46, s[sgprStrideA0I]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s46, s46, s[sgprStrideA0I]               // tlu=0, scaled tile-offset by stride
s_mul_i32 s44, s[sgprStreamKLocalStart], 128       // StreamK tile start offset
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
s_and_b32 s94, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s94, 3                                // ArgType == 3 for General Batched GEMM
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
s_mul_i32 s44, s[sgprWorkGroup0], 32               // scaleA: workgroup row origin
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

/* global read addresses: block-scale A zero-point srd */
s_mul_i32 s44, s[sgprWorkGroup0], 32               // scaleZeroA: workgroup row origin
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
s_mul_i32 s44, s[sgprStreamKLocalStart], 128       // scaleA: first K of this range
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
s_mul_hi_u32 s47, s[sgprWorkGroup1], 48            // WorkGroup[01] * MT
s_mul_i32 s46, s[sgprWorkGroup1], 48               // WorkGroup[01] * MT
s_mul_hi_u32 s47, s46, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s46, s46, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_mul_i32 s44, s[sgprStreamKLocalStart], 128       // StreamK tile start offset
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
s_and_b32 s94, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s94, 3                                // ArgType == 3 for General Batched GEMM
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
v_mul_lo_u32 v16, s[sgprStrideA0I], v[6]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+0+0], vcc_lo, v[14], v[16+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+0+0], 0x10, v[vgprGlobalReadOffsetA+0+0] // add prepad for pointer shift
v_lshrrev_b32 v16, s[sgprScaleAGShift], v14        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+0], s[sgprStrideScaleA], v6 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+0], v16, v[vgprGlobalReadOffsetScaleA+0] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+0], 1, v[vgprGlobalReadOffsetScaleA+0] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetScaleZeroA+0], 3, v6 // scaleZeroA: row / 8
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleZeroA+0], s[sgprStrideScaleA], v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: (row/8) * kGroups
v_add_nc_u32 v[vgprGlobalReadOffsetScaleZeroA+0], v16, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: + kGroup -> word offset
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleZeroA+0], 3, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: word to nibble index
v_and_b32 v16, 7, v6                               // scaleZeroA: nibble = row & 7
v_or_b32 v[vgprGlobalReadOffsetScaleZeroA+0], v16, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: (word << 3) | nibble
v_lshrrev_b32 v[vgprGlobalReadByteOffsetScaleZeroA+0], 1, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: save loop-invariant byte offset
v_lshrrev_b32 v[vgprGlobalReadOffsetA+0], 1, v[vgprGlobalReadOffsetA+0] //  (multiple bpe)
v_mul_lo_u32 v16, s[sgprStrideA0I], v[7]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+1+0], vcc_lo, v[14], v[16+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+1+0], 0x10, v[vgprGlobalReadOffsetA+1+0] // add prepad for pointer shift
v_lshrrev_b32 v16, s[sgprScaleAGShift], v14        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+1], s[sgprStrideScaleA], v7 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+1], v16, v[vgprGlobalReadOffsetScaleA+1] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+1], 1, v[vgprGlobalReadOffsetScaleA+1] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetScaleZeroA+1], 3, v7 // scaleZeroA: row / 8
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleZeroA+1], s[sgprStrideScaleA], v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: (row/8) * kGroups
v_add_nc_u32 v[vgprGlobalReadOffsetScaleZeroA+1], v16, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: + kGroup -> word offset
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleZeroA+1], 3, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: word to nibble index
v_and_b32 v16, 7, v7                               // scaleZeroA: nibble = row & 7
v_or_b32 v[vgprGlobalReadOffsetScaleZeroA+1], v16, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: (word << 3) | nibble
v_lshrrev_b32 v[vgprGlobalReadByteOffsetScaleZeroA+1], 1, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: save loop-invariant byte offset
v_lshrrev_b32 v[vgprGlobalReadOffsetA+1], 1, v[vgprGlobalReadOffsetA+1] //  (multiple bpe)
/* ============================================================= */

/* global read addresses: final offsets b */
/* ============================================================= */
v_mul_lo_u32 v16, s[sgprStrideB1J], v[8]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+0+0], vcc_lo, v[15], v[16+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+0+0], 0x8, v[vgprGlobalReadOffsetB+0+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+0], 1, v[vgprGlobalReadOffsetB+0] //  (multiple bpe)
v_mul_lo_u32 v16, s[sgprStrideB1J], v[9]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+1+0], vcc_lo, v[15], v[16+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+1+0], 0x8, v[vgprGlobalReadOffsetB+1+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+1], 1, v[vgprGlobalReadOffsetB+1] //  (multiple bpe)
v_mul_lo_u32 v16, s[sgprStrideB1J], v[10]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+2+0], vcc_lo, v[15], v[16+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+2+0], 0x8, v[vgprGlobalReadOffsetB+2+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+2], 1, v[vgprGlobalReadOffsetB+2] //  (multiple bpe)
v_mul_lo_u32 v16, s[sgprStrideB1J], v[11]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+3+0], vcc_lo, v[15], v[16+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+3+0], 0x8, v[vgprGlobalReadOffsetB+3+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+3], 1, v[vgprGlobalReadOffsetB+3] //  (multiple bpe)
v_mul_lo_u32 v16, s[sgprStrideB1J], v[12]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+4+0], vcc_lo, v[15], v[16+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+4+0], 0x8, v[vgprGlobalReadOffsetB+4+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+4], 1, v[vgprGlobalReadOffsetB+4] //  (multiple bpe)
v_mul_lo_u32 v16, s[sgprStrideB1J], v[13]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+5+0], vcc_lo, v[15], v[16+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+5+0], 0x8, v[vgprGlobalReadOffsetB+5+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+5], 1, v[vgprGlobalReadOffsetB+5] //  (multiple bpe)
/* ============================================================= */

/* global read addresses: increments a */
.set GlobalReadIncsA, 64

/* global read addresses: increments b */
.set GlobalReadIncsB, 256
/* declare loop num iterations */

/* initC: remove ValuC vgpr buffer [0...48) from pool */

/* initC: remove acc vgpr buffer [0...0) from pool */

/* initC: remove ValuA/B vgpr buffer [70...151) from pool */
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
v_mov_b32 v195, v[vgprG2LScaleA+0]                 // scaleA: copy on the other VGPR bank
v_and_b32 v208, 1, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v208, 2, v208                        // scaleZeroA: nibble shift
v_bfe_i32 v208, v[vgprG2LScaleZeroA+0], v208, 0x4  // scaleZeroA: z
v_lshl_add_u32 v209, v208, 16, s[sgprScaleALiftMagic+10] // scaleZeroA: 2^7 + c, exact
v_lshl_add_u32 v204, v208, 0, s[sgprScaleALiftMagic+6] // scaleZeroA: 2^23 + c, exact
v_lshl_add_u32 v205, v208, 4, s[sgprScaleALiftMagic+7] // scaleZeroA: 2^19 + c, exact
v_lshl_add_u32 v206, v208, 8, s[sgprScaleALiftMagic+8] // scaleZeroA: 2^15 + c, exact
v_lshl_add_u32 v207, v208, 12, s[sgprScaleALiftMagic+9] // scaleZeroA: 2^11 + c, exact
v_xor_b32 v194, 0x88888888, v[vgprG2LA+6+0]        // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v196, v194, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v197, v194, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v199, v194, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v198, v194, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v196, v204, v196 :: v_dual_subrev_f32 v197, v205, v197 // w4a16: n - c, exact
v_dual_subrev_f32 v199, v206, v199 :: v_dual_subrev_f32 v198, v207, v198 // w4a16: n - c, exact
v_dual_mul_f32 v196, v58, v196 :: v_dual_mul_f32 v197, v195, v197 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v199, v58, v199 :: v_dual_add_nc_u32 v196, s[sgprScaleALiftMagic+5], v196 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v198, v195, v198 :: v_dual_add_nc_u32 v197, s[sgprScaleALiftMagic+5], v197 // w4a16: (n - c) * s :: round
v_and_or_b32 v200, v194, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v194, 20, v194                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v201, v194, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v203, v194, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v202, v194, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v200, v209, v200 :: v_dual_add_nc_u32 v199, s[sgprScaleALiftMagic+5], v199 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v201, v204, v201 :: v_dual_add_nc_u32 v198, s[sgprScaleALiftMagic+5], v198 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v203, v205, v203 :: v_dual_subrev_f32 v202, v206, v202 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+0+0], v197, v196, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+1], v198, v199, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v200, v58, v200 :: v_dual_mul_f32 v201, v195, v201 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v203, v58, v203 :: v_dual_add_nc_u32 v200, s[sgprScaleALiftMagic+5], v200 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v202, v195, v202 :: v_dual_add_nc_u32 v201, s[sgprScaleALiftMagic+5], v201 // w4a16: (n - c) * s :: round
v_add_nc_u32 v203, s[sgprScaleALiftMagic+5], v203  // w4a16: round
v_add_nc_u32 v202, s[sgprScaleALiftMagic+5], v202  // w4a16: round
v_perm_b32 v[vgprG2LA+0+2], v201, v200, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+3], v202, v203, 0x07060302 // w4a16: pack 2 bf16
v_xor_b32 v194, 0x88888888, v[vgprG2LA+6+1]        // w4a16: save packed int4 dword 1, biased to unsigned
v_and_or_b32 v196, v194, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v197, v194, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v199, v194, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v198, v194, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v196, v204, v196 :: v_dual_subrev_f32 v197, v205, v197 // w4a16: n - c, exact
v_dual_subrev_f32 v199, v206, v199 :: v_dual_subrev_f32 v198, v207, v198 // w4a16: n - c, exact
v_dual_mul_f32 v196, v58, v196 :: v_dual_mul_f32 v197, v195, v197 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v199, v58, v199 :: v_dual_add_nc_u32 v196, s[sgprScaleALiftMagic+5], v196 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v198, v195, v198 :: v_dual_add_nc_u32 v197, s[sgprScaleALiftMagic+5], v197 // w4a16: (n - c) * s :: round
v_and_or_b32 v200, v194, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v194, 20, v194                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v201, v194, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v203, v194, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v202, v194, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v200, v209, v200 :: v_dual_add_nc_u32 v199, s[sgprScaleALiftMagic+5], v199 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v201, v204, v201 :: v_dual_add_nc_u32 v198, s[sgprScaleALiftMagic+5], v198 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v203, v205, v203 :: v_dual_subrev_f32 v202, v206, v202 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+0+4], v197, v196, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+5], v198, v199, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v200, v58, v200 :: v_dual_mul_f32 v201, v195, v201 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v203, v58, v203 :: v_dual_add_nc_u32 v200, s[sgprScaleALiftMagic+5], v200 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v202, v195, v202 :: v_dual_add_nc_u32 v201, s[sgprScaleALiftMagic+5], v201 // w4a16: (n - c) * s :: round
v_add_nc_u32 v203, s[sgprScaleALiftMagic+5], v203  // w4a16: round
v_add_nc_u32 v202, s[sgprScaleALiftMagic+5], v202  // w4a16: round
v_perm_b32 v[vgprG2LA+0+6], v201, v200, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+7], v202, v203, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:16 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
v_mov_b32 v194, v[vgprG2LScaleA+1]                 // scaleA: copy on the other VGPR bank
v_and_b32 v208, 1, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v208, 2, v208                        // scaleZeroA: nibble shift
v_bfe_i32 v208, v[vgprG2LScaleZeroA+1], v208, 0x4  // scaleZeroA: z
v_lshl_add_u32 v209, v208, 16, s[sgprScaleALiftMagic+10] // scaleZeroA: 2^7 + c, exact
v_lshl_add_u32 v204, v208, 0, s[sgprScaleALiftMagic+6] // scaleZeroA: 2^23 + c, exact
v_lshl_add_u32 v205, v208, 4, s[sgprScaleALiftMagic+7] // scaleZeroA: 2^19 + c, exact
v_lshl_add_u32 v206, v208, 8, s[sgprScaleALiftMagic+8] // scaleZeroA: 2^15 + c, exact
v_lshl_add_u32 v207, v208, 12, s[sgprScaleALiftMagic+9] // scaleZeroA: 2^11 + c, exact
v_xor_b32 v195, 0x88888888, v[vgprG2LA+14+0]       // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v196, v195, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v197, v195, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v199, v195, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v198, v195, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v196, v204, v196 :: v_dual_subrev_f32 v197, v205, v197 // w4a16: n - c, exact
v_dual_subrev_f32 v199, v206, v199 :: v_dual_subrev_f32 v198, v207, v198 // w4a16: n - c, exact
v_dual_mul_f32 v196, v59, v196 :: v_dual_mul_f32 v197, v194, v197 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v199, v59, v199 :: v_dual_add_nc_u32 v196, s[sgprScaleALiftMagic+5], v196 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v198, v194, v198 :: v_dual_add_nc_u32 v197, s[sgprScaleALiftMagic+5], v197 // w4a16: (n - c) * s :: round
v_and_or_b32 v200, v195, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v195, 20, v195                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v201, v195, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v203, v195, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v202, v195, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v200, v209, v200 :: v_dual_add_nc_u32 v199, s[sgprScaleALiftMagic+5], v199 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v201, v204, v201 :: v_dual_add_nc_u32 v198, s[sgprScaleALiftMagic+5], v198 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v203, v205, v203 :: v_dual_subrev_f32 v202, v206, v202 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+8+0], v197, v196, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+1], v198, v199, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v200, v59, v200 :: v_dual_mul_f32 v201, v194, v201 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v203, v59, v203 :: v_dual_add_nc_u32 v200, s[sgprScaleALiftMagic+5], v200 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v202, v194, v202 :: v_dual_add_nc_u32 v201, s[sgprScaleALiftMagic+5], v201 // w4a16: (n - c) * s :: round
v_add_nc_u32 v203, s[sgprScaleALiftMagic+5], v203  // w4a16: round
v_add_nc_u32 v202, s[sgprScaleALiftMagic+5], v202  // w4a16: round
v_perm_b32 v[vgprG2LA+8+2], v201, v200, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+3], v202, v203, 0x07060302 // w4a16: pack 2 bf16
v_xor_b32 v195, 0x88888888, v[vgprG2LA+14+1]       // w4a16: save packed int4 dword 1, biased to unsigned
v_and_or_b32 v196, v195, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v197, v195, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v199, v195, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v198, v195, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v196, v204, v196 :: v_dual_subrev_f32 v197, v205, v197 // w4a16: n - c, exact
v_dual_subrev_f32 v199, v206, v199 :: v_dual_subrev_f32 v198, v207, v198 // w4a16: n - c, exact
v_dual_mul_f32 v196, v59, v196 :: v_dual_mul_f32 v197, v194, v197 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v199, v59, v199 :: v_dual_add_nc_u32 v196, s[sgprScaleALiftMagic+5], v196 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v198, v194, v198 :: v_dual_add_nc_u32 v197, s[sgprScaleALiftMagic+5], v197 // w4a16: (n - c) * s :: round
v_and_or_b32 v200, v195, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v195, 20, v195                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v201, v195, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v203, v195, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v202, v195, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v200, v209, v200 :: v_dual_add_nc_u32 v199, s[sgprScaleALiftMagic+5], v199 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v201, v204, v201 :: v_dual_add_nc_u32 v198, s[sgprScaleALiftMagic+5], v198 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v203, v205, v203 :: v_dual_subrev_f32 v202, v206, v202 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+8+4], v197, v196, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+5], v198, v199, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v200, v59, v200 :: v_dual_mul_f32 v201, v194, v201 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v203, v59, v203 :: v_dual_add_nc_u32 v200, s[sgprScaleALiftMagic+5], v200 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v202, v194, v202 :: v_dual_add_nc_u32 v201, s[sgprScaleALiftMagic+5], v201 // w4a16: (n - c) * s :: round
v_add_nc_u32 v203, s[sgprScaleALiftMagic+5], v203  // w4a16: round
v_add_nc_u32 v202, s[sgprScaleALiftMagic+5], v202  // w4a16: round
v_perm_b32 v[vgprG2LA+8+6], v201, v200, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+7], v202, v203, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:4352 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 4352 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+12:vgprG2LA+12+3] offset:4368 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 4352 sync LDS0

/* local write b */
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4:vgprG2LB+4+3] offset:2176 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 2176 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8:vgprG2LB+8+3] offset:4352 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 4352 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12:vgprG2LB+12+3] offset:6528 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 6528 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+16:vgprG2LB+16+3] offset:8704 // lwoB_0_0_4_0 = (0*LSCB)*(MT1J+PAD) + (4*LSPB) = 8704 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+20:vgprG2LB+20+3] offset:10880 // lwoB_0_0_5_0 = (0*LSCB)*(MT1J+PAD) + (5*LSPB) = 10880 sync LDS0

/* local write swap a */

/* local write swap b */
s_cmp_eq_u32 s[sgprLoopCounterL], 0x1              // PGR=2 but only 1 loop
s_cbranch_scc1 label_skipPGR2_1                    // PGR=2 but only 1 loop
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
s_branch label_skipPGR2_2                          // jump to PGR=2 label
label_skipPGR2_1:
label_skipPGR2_2:
s_waitcnt lgkmcnt(0)                               // 0prefetch wait for local write
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier                                          // LW to PLR, sync

/* local read prefetch a */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4352 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4368 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read prefetch b */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4352 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4368 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:8704 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:8720 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

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
/*  grEndMfmaIndex:3, lwStartMfmaIndex:3, lwEndMfmaIndex:3  */
/*  numMfmaForLR:5, syncPlrMfmaIndex:6 , sync1LdsMfmaIndex:2 */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=4 for iteration == 0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:1  */
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], GlobalReadIncsA // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], GlobalReadIncsA // limit -= inc)
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:2  */
/* schedule remaining localreads for one buffer scheduling */
ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA+0] offset:4384 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA+0] offset:4400 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB+0] offset:4384 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[vgprLocalReadAddrB+0] offset:4400 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprLocalReadAddrB+0] offset:8736 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+20:vgprValuB_X1_I0+20+3], v[vgprLocalReadAddrB+0] offset:8752 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
s_waitcnt lgkmcnt(10)                              // wait for prior local read local write
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:3  */

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
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
v_mov_b32 v195, v[vgprG2LScaleA+0]                 // scaleA: copy on the other VGPR bank
v_and_b32 v208, 1, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v208, 2, v208                        // scaleZeroA: nibble shift
v_bfe_i32 v208, v[vgprG2LScaleZeroA+0], v208, 0x4  // scaleZeroA: z
v_lshl_add_u32 v209, v208, 16, s[sgprScaleALiftMagic+10] // scaleZeroA: 2^7 + c, exact
v_lshl_add_u32 v204, v208, 0, s[sgprScaleALiftMagic+6] // scaleZeroA: 2^23 + c, exact
v_lshl_add_u32 v205, v208, 4, s[sgprScaleALiftMagic+7] // scaleZeroA: 2^19 + c, exact
v_lshl_add_u32 v206, v208, 8, s[sgprScaleALiftMagic+8] // scaleZeroA: 2^15 + c, exact
v_lshl_add_u32 v207, v208, 12, s[sgprScaleALiftMagic+9] // scaleZeroA: 2^11 + c, exact
v_xor_b32 v194, 0x88888888, v[vgprG2LA+6+0]        // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v196, v194, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v197, v194, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v199, v194, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v198, v194, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v196, v204, v196 :: v_dual_subrev_f32 v197, v205, v197 // w4a16: n - c, exact
v_dual_subrev_f32 v199, v206, v199 :: v_dual_subrev_f32 v198, v207, v198 // w4a16: n - c, exact
v_dual_mul_f32 v196, v58, v196 :: v_dual_mul_f32 v197, v195, v197 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v199, v58, v199 :: v_dual_add_nc_u32 v196, s[sgprScaleALiftMagic+5], v196 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v198, v195, v198 :: v_dual_add_nc_u32 v197, s[sgprScaleALiftMagic+5], v197 // w4a16: (n - c) * s :: round
v_and_or_b32 v200, v194, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v194, 20, v194                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v201, v194, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v203, v194, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v202, v194, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v200, v209, v200 :: v_dual_add_nc_u32 v199, s[sgprScaleALiftMagic+5], v199 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v201, v204, v201 :: v_dual_add_nc_u32 v198, s[sgprScaleALiftMagic+5], v198 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v203, v205, v203 :: v_dual_subrev_f32 v202, v206, v202 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+0+0], v197, v196, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+1], v198, v199, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v200, v58, v200 :: v_dual_mul_f32 v201, v195, v201 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v203, v58, v203 :: v_dual_add_nc_u32 v200, s[sgprScaleALiftMagic+5], v200 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v202, v195, v202 :: v_dual_add_nc_u32 v201, s[sgprScaleALiftMagic+5], v201 // w4a16: (n - c) * s :: round
v_add_nc_u32 v203, s[sgprScaleALiftMagic+5], v203  // w4a16: round
v_add_nc_u32 v202, s[sgprScaleALiftMagic+5], v202  // w4a16: round
v_perm_b32 v[vgprG2LA+0+2], v201, v200, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+3], v202, v203, 0x07060302 // w4a16: pack 2 bf16
v_xor_b32 v194, 0x88888888, v[vgprG2LA+6+1]        // w4a16: save packed int4 dword 1, biased to unsigned
v_and_or_b32 v196, v194, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v197, v194, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v199, v194, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v198, v194, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v196, v204, v196 :: v_dual_subrev_f32 v197, v205, v197 // w4a16: n - c, exact
v_dual_subrev_f32 v199, v206, v199 :: v_dual_subrev_f32 v198, v207, v198 // w4a16: n - c, exact
v_dual_mul_f32 v196, v58, v196 :: v_dual_mul_f32 v197, v195, v197 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v199, v58, v199 :: v_dual_add_nc_u32 v196, s[sgprScaleALiftMagic+5], v196 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v198, v195, v198 :: v_dual_add_nc_u32 v197, s[sgprScaleALiftMagic+5], v197 // w4a16: (n - c) * s :: round
v_and_or_b32 v200, v194, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v194, 20, v194                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v201, v194, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v203, v194, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v202, v194, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v200, v209, v200 :: v_dual_add_nc_u32 v199, s[sgprScaleALiftMagic+5], v199 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v201, v204, v201 :: v_dual_add_nc_u32 v198, s[sgprScaleALiftMagic+5], v198 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v203, v205, v203 :: v_dual_subrev_f32 v202, v206, v202 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+0+4], v197, v196, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+5], v198, v199, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v200, v58, v200 :: v_dual_mul_f32 v201, v195, v201 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v203, v58, v203 :: v_dual_add_nc_u32 v200, s[sgprScaleALiftMagic+5], v200 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v202, v195, v202 :: v_dual_add_nc_u32 v201, s[sgprScaleALiftMagic+5], v201 // w4a16: (n - c) * s :: round
v_add_nc_u32 v203, s[sgprScaleALiftMagic+5], v203  // w4a16: round
v_add_nc_u32 v202, s[sgprScaleALiftMagic+5], v202  // w4a16: round
v_perm_b32 v[vgprG2LA+0+6], v201, v200, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+7], v202, v203, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:16 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
buffer_load_b64 v[vgprG2LA+6:vgprG2LA+6+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
v_mov_b32 v194, v[vgprG2LScaleA+1]                 // scaleA: copy on the other VGPR bank
v_and_b32 v208, 1, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v208, 2, v208                        // scaleZeroA: nibble shift
v_bfe_i32 v208, v[vgprG2LScaleZeroA+1], v208, 0x4  // scaleZeroA: z
v_lshl_add_u32 v209, v208, 16, s[sgprScaleALiftMagic+10] // scaleZeroA: 2^7 + c, exact
v_lshl_add_u32 v204, v208, 0, s[sgprScaleALiftMagic+6] // scaleZeroA: 2^23 + c, exact
v_lshl_add_u32 v205, v208, 4, s[sgprScaleALiftMagic+7] // scaleZeroA: 2^19 + c, exact
v_lshl_add_u32 v206, v208, 8, s[sgprScaleALiftMagic+8] // scaleZeroA: 2^15 + c, exact
v_lshl_add_u32 v207, v208, 12, s[sgprScaleALiftMagic+9] // scaleZeroA: 2^11 + c, exact
v_xor_b32 v195, 0x88888888, v[vgprG2LA+14+0]       // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v196, v195, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v197, v195, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v199, v195, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v198, v195, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v196, v204, v196 :: v_dual_subrev_f32 v197, v205, v197 // w4a16: n - c, exact
v_dual_subrev_f32 v199, v206, v199 :: v_dual_subrev_f32 v198, v207, v198 // w4a16: n - c, exact
v_dual_mul_f32 v196, v59, v196 :: v_dual_mul_f32 v197, v194, v197 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v199, v59, v199 :: v_dual_add_nc_u32 v196, s[sgprScaleALiftMagic+5], v196 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v198, v194, v198 :: v_dual_add_nc_u32 v197, s[sgprScaleALiftMagic+5], v197 // w4a16: (n - c) * s :: round
v_and_or_b32 v200, v195, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v195, 20, v195                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v201, v195, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v203, v195, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v202, v195, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v200, v209, v200 :: v_dual_add_nc_u32 v199, s[sgprScaleALiftMagic+5], v199 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v201, v204, v201 :: v_dual_add_nc_u32 v198, s[sgprScaleALiftMagic+5], v198 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v203, v205, v203 :: v_dual_subrev_f32 v202, v206, v202 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+8+0], v197, v196, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+1], v198, v199, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v200, v59, v200 :: v_dual_mul_f32 v201, v194, v201 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v203, v59, v203 :: v_dual_add_nc_u32 v200, s[sgprScaleALiftMagic+5], v200 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v202, v194, v202 :: v_dual_add_nc_u32 v201, s[sgprScaleALiftMagic+5], v201 // w4a16: (n - c) * s :: round
v_add_nc_u32 v203, s[sgprScaleALiftMagic+5], v203  // w4a16: round
v_add_nc_u32 v202, s[sgprScaleALiftMagic+5], v202  // w4a16: round
v_perm_b32 v[vgprG2LA+8+2], v201, v200, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+3], v202, v203, 0x07060302 // w4a16: pack 2 bf16
v_xor_b32 v195, 0x88888888, v[vgprG2LA+14+1]       // w4a16: save packed int4 dword 1, biased to unsigned
v_and_or_b32 v196, v195, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v197, v195, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v199, v195, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v198, v195, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v196, v204, v196 :: v_dual_subrev_f32 v197, v205, v197 // w4a16: n - c, exact
v_dual_subrev_f32 v199, v206, v199 :: v_dual_subrev_f32 v198, v207, v198 // w4a16: n - c, exact
v_dual_mul_f32 v196, v59, v196 :: v_dual_mul_f32 v197, v194, v197 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v199, v59, v199 :: v_dual_add_nc_u32 v196, s[sgprScaleALiftMagic+5], v196 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v198, v194, v198 :: v_dual_add_nc_u32 v197, s[sgprScaleALiftMagic+5], v197 // w4a16: (n - c) * s :: round
v_and_or_b32 v200, v195, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v195, 20, v195                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v201, v195, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v203, v195, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v202, v195, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v200, v209, v200 :: v_dual_add_nc_u32 v199, s[sgprScaleALiftMagic+5], v199 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v201, v204, v201 :: v_dual_add_nc_u32 v198, s[sgprScaleALiftMagic+5], v198 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v203, v205, v203 :: v_dual_subrev_f32 v202, v206, v202 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+8+4], v197, v196, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+5], v198, v199, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v200, v59, v200 :: v_dual_mul_f32 v201, v194, v201 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v203, v59, v203 :: v_dual_add_nc_u32 v200, s[sgprScaleALiftMagic+5], v200 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v202, v194, v202 :: v_dual_add_nc_u32 v201, s[sgprScaleALiftMagic+5], v201 // w4a16: (n - c) * s :: round
v_add_nc_u32 v203, s[sgprScaleALiftMagic+5], v203  // w4a16: round
v_add_nc_u32 v202, s[sgprScaleALiftMagic+5], v202  // w4a16: round
v_perm_b32 v[vgprG2LA+8+6], v201, v200, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+7], v202, v203, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:4352 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 4352 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+12:vgprG2LA+12+3] offset:4368 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 4352 sync LDS0
buffer_load_b64 v[vgprG2LA+14:vgprG2LA+14+1], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_1_0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0

/* global read block-scale A */
buffer_load_d16_hi_b16 v[vgprG2LScaleA+0], v[vgprGlobalReadOffsetScaleA+0], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 0 as f32
buffer_load_d16_hi_b16 v[vgprG2LScaleA+1], v[vgprGlobalReadOffsetScaleA+1], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 1 as f32
buffer_load_d16_u8 v[vgprG2LScaleZeroA+0], v[vgprGlobalReadByteOffsetScaleZeroA+0], s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 0
buffer_load_d16_u8 v[vgprG2LScaleZeroA+1], v[vgprGlobalReadByteOffsetScaleZeroA+1], s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 1
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(10)                                // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4:vgprG2LB+4+3] offset:2176 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 2176 sync LDS0
buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(10)                                // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8:vgprG2LB+8+3] offset:4352 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 4352 sync LDS0
buffer_load_b128 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(10)                                // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12:vgprG2LB+12+3] offset:6528 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 6528 sync LDS0
buffer_load_b128 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(10)                                // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+16:vgprG2LB+16+3] offset:8704 // lwoB_0_0_4_0 = (0*LSCB)*(MT1J+PAD) + (4*LSPB) = 8704 sync LDS0
buffer_load_b128 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(10)                                // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+20:vgprG2LB+20+3] offset:10880 // lwoB_0_0_5_0 = (0*LSCB)*(MT1J+PAD) + (5*LSPB) = 10880 sync LDS0
buffer_load_b128 v[vgprG2LB+16:vgprG2LB+16+3], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_4_0
buffer_load_b128 v[vgprG2LB+20:vgprG2LB+20+3], v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_5_0

/* local write swap offsets a */

/* local write swap offsets b */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:4  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:5  */

/* local read swap offsets a */

/* local read swap offsets b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=4 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=6 */

/* iter 1 */
/*  grEndMfmaIndex:3, lwStartMfmaIndex:3, lwEndMfmaIndex:3  */
/*  numMfmaForLR:5, syncPlrMfmaIndex:6 , sync1LdsMfmaIndex:2 */
/*  mfmaIndex:6  */
s_waitcnt lgkmcnt(0)                               // 3wait for local write
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier                                          // PGR, and wait until LW done to sync
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:7  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:8  */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:9  */
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4352 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4368 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:10  */
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4352 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4368 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X1_I0+16+0+0:vgprValuB_X1_I0+16+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:11  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:8704 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:8720 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X1_I0+16+0+0:vgprValuB_X1_I0+16+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]

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
/*  grEndMfmaIndex:3, lwStartMfmaIndex:3, lwEndMfmaIndex:3  */
/*  numMfmaForLR:5, syncPlrMfmaIndex:6 , sync1LdsMfmaIndex:2 */
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
s_and_b32 s44, s[sgprScaleAKCnt], s[sgprScaleAKMask] // scaleA: SCC = counter has not wrapped
s_cselect_b32 s44, 0, s[sgprScaleAInc]             // scaleA: advance only on a wrap
s_lshl_b32 s45, s44, 1                             // scaleZeroA: advance only on a wrap
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], s44 // scaleA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], s44 // scaleA limit -= inc
s_add_u32 s[sgprSrdScaleZeroA+0], s[sgprSrdScaleZeroA+0], s45 // scaleZeroA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleZeroA+1], s[sgprSrdScaleZeroA+1], 0 // scaleZeroA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleZeroA+2], s[sgprSrdScaleZeroA+2], s45 // scaleZeroA limit -= inc

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
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:2  */
/* schedule remaining localreads for one buffer scheduling */
ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA+0] offset:4384 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA+0] offset:4400 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB+0] offset:4384 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[vgprLocalReadAddrB+0] offset:4400 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprLocalReadAddrB+0] offset:8736 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+20:vgprValuB_X1_I0+20+3], v[vgprLocalReadAddrB+0] offset:8752 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
s_waitcnt lgkmcnt(10)                              // wait for prior local read local write
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:3  */
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
v_mov_b32 v195, v[vgprG2LScaleA+0]                 // scaleA: copy on the other VGPR bank
v_and_b32 v208, 1, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v208, 2, v208                        // scaleZeroA: nibble shift
v_bfe_i32 v208, v[vgprG2LScaleZeroA+0], v208, 0x4  // scaleZeroA: z
v_lshl_add_u32 v209, v208, 16, s[sgprScaleALiftMagic+10] // scaleZeroA: 2^7 + c, exact
v_lshl_add_u32 v204, v208, 0, s[sgprScaleALiftMagic+6] // scaleZeroA: 2^23 + c, exact
v_lshl_add_u32 v205, v208, 4, s[sgprScaleALiftMagic+7] // scaleZeroA: 2^19 + c, exact
v_lshl_add_u32 v206, v208, 8, s[sgprScaleALiftMagic+8] // scaleZeroA: 2^15 + c, exact
v_lshl_add_u32 v207, v208, 12, s[sgprScaleALiftMagic+9] // scaleZeroA: 2^11 + c, exact
v_xor_b32 v194, 0x88888888, v[vgprG2LA+6+0]        // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v196, v194, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v197, v194, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v199, v194, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v198, v194, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v196, v204, v196 :: v_dual_subrev_f32 v197, v205, v197 // w4a16: n - c, exact
v_dual_subrev_f32 v199, v206, v199 :: v_dual_subrev_f32 v198, v207, v198 // w4a16: n - c, exact
v_dual_mul_f32 v196, v58, v196 :: v_dual_mul_f32 v197, v195, v197 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v199, v58, v199 :: v_dual_add_nc_u32 v196, s[sgprScaleALiftMagic+5], v196 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v198, v195, v198 :: v_dual_add_nc_u32 v197, s[sgprScaleALiftMagic+5], v197 // w4a16: (n - c) * s :: round
v_and_or_b32 v200, v194, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v194, 20, v194                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v201, v194, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v203, v194, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v202, v194, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v200, v209, v200 :: v_dual_add_nc_u32 v199, s[sgprScaleALiftMagic+5], v199 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v201, v204, v201 :: v_dual_add_nc_u32 v198, s[sgprScaleALiftMagic+5], v198 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v203, v205, v203 :: v_dual_subrev_f32 v202, v206, v202 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+0+0], v197, v196, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+1], v198, v199, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v200, v58, v200 :: v_dual_mul_f32 v201, v195, v201 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v203, v58, v203 :: v_dual_add_nc_u32 v200, s[sgprScaleALiftMagic+5], v200 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v202, v195, v202 :: v_dual_add_nc_u32 v201, s[sgprScaleALiftMagic+5], v201 // w4a16: (n - c) * s :: round
v_add_nc_u32 v203, s[sgprScaleALiftMagic+5], v203  // w4a16: round
v_add_nc_u32 v202, s[sgprScaleALiftMagic+5], v202  // w4a16: round
v_perm_b32 v[vgprG2LA+0+2], v201, v200, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+3], v202, v203, 0x07060302 // w4a16: pack 2 bf16
v_xor_b32 v194, 0x88888888, v[vgprG2LA+6+1]        // w4a16: save packed int4 dword 1, biased to unsigned
v_and_or_b32 v196, v194, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v197, v194, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v199, v194, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v198, v194, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v196, v204, v196 :: v_dual_subrev_f32 v197, v205, v197 // w4a16: n - c, exact
v_dual_subrev_f32 v199, v206, v199 :: v_dual_subrev_f32 v198, v207, v198 // w4a16: n - c, exact
v_dual_mul_f32 v196, v58, v196 :: v_dual_mul_f32 v197, v195, v197 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v199, v58, v199 :: v_dual_add_nc_u32 v196, s[sgprScaleALiftMagic+5], v196 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v198, v195, v198 :: v_dual_add_nc_u32 v197, s[sgprScaleALiftMagic+5], v197 // w4a16: (n - c) * s :: round
v_and_or_b32 v200, v194, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v194, 20, v194                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v201, v194, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v203, v194, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v202, v194, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v200, v209, v200 :: v_dual_add_nc_u32 v199, s[sgprScaleALiftMagic+5], v199 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v201, v204, v201 :: v_dual_add_nc_u32 v198, s[sgprScaleALiftMagic+5], v198 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v203, v205, v203 :: v_dual_subrev_f32 v202, v206, v202 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+0+4], v197, v196, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+5], v198, v199, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v200, v58, v200 :: v_dual_mul_f32 v201, v195, v201 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v203, v58, v203 :: v_dual_add_nc_u32 v200, s[sgprScaleALiftMagic+5], v200 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v202, v195, v202 :: v_dual_add_nc_u32 v201, s[sgprScaleALiftMagic+5], v201 // w4a16: (n - c) * s :: round
v_add_nc_u32 v203, s[sgprScaleALiftMagic+5], v203  // w4a16: round
v_add_nc_u32 v202, s[sgprScaleALiftMagic+5], v202  // w4a16: round
v_perm_b32 v[vgprG2LA+0+6], v201, v200, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+7], v202, v203, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:16 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(6)                                 // wait for global read before writing to local
v_mov_b32 v194, v[vgprG2LScaleA+1]                 // scaleA: copy on the other VGPR bank
v_and_b32 v208, 1, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v208, 2, v208                        // scaleZeroA: nibble shift
v_bfe_i32 v208, v[vgprG2LScaleZeroA+1], v208, 0x4  // scaleZeroA: z
v_lshl_add_u32 v209, v208, 16, s[sgprScaleALiftMagic+10] // scaleZeroA: 2^7 + c, exact
v_lshl_add_u32 v204, v208, 0, s[sgprScaleALiftMagic+6] // scaleZeroA: 2^23 + c, exact
v_lshl_add_u32 v205, v208, 4, s[sgprScaleALiftMagic+7] // scaleZeroA: 2^19 + c, exact
v_lshl_add_u32 v206, v208, 8, s[sgprScaleALiftMagic+8] // scaleZeroA: 2^15 + c, exact
v_lshl_add_u32 v207, v208, 12, s[sgprScaleALiftMagic+9] // scaleZeroA: 2^11 + c, exact
v_xor_b32 v195, 0x88888888, v[vgprG2LA+14+0]       // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v196, v195, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v197, v195, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v199, v195, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v198, v195, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v196, v204, v196 :: v_dual_subrev_f32 v197, v205, v197 // w4a16: n - c, exact
v_dual_subrev_f32 v199, v206, v199 :: v_dual_subrev_f32 v198, v207, v198 // w4a16: n - c, exact
v_dual_mul_f32 v196, v59, v196 :: v_dual_mul_f32 v197, v194, v197 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v199, v59, v199 :: v_dual_add_nc_u32 v196, s[sgprScaleALiftMagic+5], v196 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v198, v194, v198 :: v_dual_add_nc_u32 v197, s[sgprScaleALiftMagic+5], v197 // w4a16: (n - c) * s :: round
v_and_or_b32 v200, v195, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v195, 20, v195                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v201, v195, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v203, v195, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v202, v195, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v200, v209, v200 :: v_dual_add_nc_u32 v199, s[sgprScaleALiftMagic+5], v199 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v201, v204, v201 :: v_dual_add_nc_u32 v198, s[sgprScaleALiftMagic+5], v198 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v203, v205, v203 :: v_dual_subrev_f32 v202, v206, v202 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+8+0], v197, v196, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+1], v198, v199, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v200, v59, v200 :: v_dual_mul_f32 v201, v194, v201 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v203, v59, v203 :: v_dual_add_nc_u32 v200, s[sgprScaleALiftMagic+5], v200 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v202, v194, v202 :: v_dual_add_nc_u32 v201, s[sgprScaleALiftMagic+5], v201 // w4a16: (n - c) * s :: round
v_add_nc_u32 v203, s[sgprScaleALiftMagic+5], v203  // w4a16: round
v_add_nc_u32 v202, s[sgprScaleALiftMagic+5], v202  // w4a16: round
v_perm_b32 v[vgprG2LA+8+2], v201, v200, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+3], v202, v203, 0x07060302 // w4a16: pack 2 bf16
v_xor_b32 v195, 0x88888888, v[vgprG2LA+14+1]       // w4a16: save packed int4 dword 1, biased to unsigned
v_and_or_b32 v196, v195, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v197, v195, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v199, v195, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v198, v195, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v196, v204, v196 :: v_dual_subrev_f32 v197, v205, v197 // w4a16: n - c, exact
v_dual_subrev_f32 v199, v206, v199 :: v_dual_subrev_f32 v198, v207, v198 // w4a16: n - c, exact
v_dual_mul_f32 v196, v59, v196 :: v_dual_mul_f32 v197, v194, v197 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v199, v59, v199 :: v_dual_add_nc_u32 v196, s[sgprScaleALiftMagic+5], v196 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v198, v194, v198 :: v_dual_add_nc_u32 v197, s[sgprScaleALiftMagic+5], v197 // w4a16: (n - c) * s :: round
v_and_or_b32 v200, v195, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v195, 20, v195                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v201, v195, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v203, v195, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v202, v195, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v200, v209, v200 :: v_dual_add_nc_u32 v199, s[sgprScaleALiftMagic+5], v199 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v201, v204, v201 :: v_dual_add_nc_u32 v198, s[sgprScaleALiftMagic+5], v198 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v203, v205, v203 :: v_dual_subrev_f32 v202, v206, v202 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+8+4], v197, v196, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+5], v198, v199, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v200, v59, v200 :: v_dual_mul_f32 v201, v194, v201 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v203, v59, v203 :: v_dual_add_nc_u32 v200, s[sgprScaleALiftMagic+5], v200 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v202, v194, v202 :: v_dual_add_nc_u32 v201, s[sgprScaleALiftMagic+5], v201 // w4a16: (n - c) * s :: round
v_add_nc_u32 v203, s[sgprScaleALiftMagic+5], v203  // w4a16: round
v_add_nc_u32 v202, s[sgprScaleALiftMagic+5], v202  // w4a16: round
v_perm_b32 v[vgprG2LA+8+6], v201, v200, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+7], v202, v203, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:4352 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 4352 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+12:vgprG2LA+12+3] offset:4368 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 4352 sync LDS0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(5)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(4)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4:vgprG2LB+4+3] offset:2176 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 2176 sync LDS0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(3)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8:vgprG2LB+8+3] offset:4352 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 4352 sync LDS0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(2)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12:vgprG2LB+12+3] offset:6528 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 6528 sync LDS0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(1)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+16:vgprG2LB+16+3] offset:8704 // lwoB_0_0_4_0 = (0*LSCB)*(MT1J+PAD) + (4*LSPB) = 8704 sync LDS0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(0)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+20:vgprG2LB+20+3] offset:10880 // lwoB_0_0_5_0 = (0*LSCB)*(MT1J+PAD) + (5*LSPB) = 10880 sync LDS0

/* local write swap offsets a */

/* local write swap offsets b */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:4  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:5  */

/* local read swap offsets a */

/* local read swap offsets b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=4 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=6 */

/* iter 1 */
/*  grEndMfmaIndex:3, lwStartMfmaIndex:3, lwEndMfmaIndex:3  */
/*  numMfmaForLR:5, syncPlrMfmaIndex:6 , sync1LdsMfmaIndex:2 */
/*  mfmaIndex:6  */
s_waitcnt lgkmcnt(0)                               // 3wait for local write
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier                                          // noLoadLoop sync
s_waitcnt lgkmcnt(10)                              // wait for prior local read local write old=0, new=10 newLW=10 newLR=0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:7  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:8  */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:9  */
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4352 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4368 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:10  */
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:4352 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:4368 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X1_I0+16+0+0:vgprValuB_X1_I0+16+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:11  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:8704 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:8720 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X1_I0+16+0+0:vgprValuB_X1_I0+16+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/* numPrefetchIter=1 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=4 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=6 */
label_toPGR1:

/******************************************/
/* Ord. NoLoadLoop - Begin                */
/******************************************/

/* iter 0 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:3, lwEndMfmaIndex:3  */
/*  numMfmaForLR:5, syncPlrMfmaIndex:6 , sync1LdsMfmaIndex:2 */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=4 for iteration == 0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:1  */
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:2  */
/* schedule remaining localreads for one buffer scheduling */
ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA+0] offset:4384 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA+0] offset:4400 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB+0] offset:4384 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[vgprLocalReadAddrB+0] offset:4400 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprLocalReadAddrB+0] offset:8736 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+20:vgprValuB_X1_I0+20+3], v[vgprLocalReadAddrB+0] offset:8752 // L -> Reg lro=16 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
s_waitcnt lgkmcnt(10)                              // wait for prior local read local write
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:3  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:4  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:5  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=4 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=6 */

/* iter 1 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:3, lwEndMfmaIndex:3  */
/*  numMfmaForLR:5, syncPlrMfmaIndex:6 , sync1LdsMfmaIndex:2 */
/*  mfmaIndex:6  */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:7  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:8  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:9  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:10  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X1_I0+16+0+0:vgprValuB_X1_I0+16+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/*  mfmaIndex:11  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X1_I0+16+0+0:vgprValuB_X1_I0+16+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=6 */
label_toPGR1end_OrdNLL:
label_PrefetchGlobalLastIterEnd:

/* Tail: add ValuA/B vgpr buffer [70...151) to pool */

/* Tail: add address/G2L vgpr [151...192) to pool */
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
s_and_b32 s8, s[sgprArgType], 0xff                 // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s8, 3                                 // ArgType == 3 for General Batched GEMM
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

s_and_b32 s8, s[sgprArgType], 0xff                 // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s8, 3                                 // ArgType == 3 for General Batched GEMM
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

s_mov_b32 s8, 1
s_mov_b32 s9, 1
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_BPEDone                       // If synchronizer, use regular output BPE
s_cmp_eq_u32 s[sgprskTiles], 1                     // split == 1 ?
s_cbranch_scc1 label_BPEDone                       // If split == 1, use reguler output BPE
s_mov_b32 s8, 1
s_mov_b32 s9, 2
label_BPEDone:

s_mul_i32 s46, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_mul_hi_u32 s45, s46, s[sgprStrideC1J]            // ScaleC s46 by Stride
s_mul_i32 s44, s46, s[sgprStrideC1J]               // ScaleC s46 by Stride
s_lshl_b64 s[44:45], s[44:45], s8                  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s44        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s45       // add hi to SRD
s_mul_hi_u32 s45, s46, s[sgprStrideD1J]            // ScaleD s46 by Stride
s_mul_i32 s44, s46, s[sgprStrideD1J]               // ScaleD s46 by Stride
s_lshl_b64 s[44:45], s[44:45], s9                  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s44        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s45       // add hi to SRD

s_and_b32 s47, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s47, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_StridedBatchedGemmLoadC
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_GeneralBatchedGemmLoadC
label_StridedBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s45, s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride
s_mul_i32 s44, s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride
s_lshl_b64 s[44:45], s[44:45], s8                  // scale by bpe
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
s_and_b32 s47, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ are internal flags)
s_cmp_eq_u32 s47, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_StridedBatchedGemmLoadD
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_GeneralBatchedGemmLoadD
label_StridedBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s45, s[sgprWorkGroup2], s[sgprStrideDK] // ScaleD s[sgprWorkGroup2] by Stride
s_mul_i32 s44, s[sgprWorkGroup2], s[sgprStrideDK]  // ScaleD s[sgprWorkGroup2] by Stride
s_lshl_b64 s[44:45], s[44:45], s9                  // scale by bpe
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

/* LocalSplitU: local write and read */

/******************************************/
/* LocalSplitU Reduction                  */
/******************************************/
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier
label_localSplitULocalWriteAndRead:
/* computeStoreVgprs */
v_lshrrev_b32 v76, 5, v[vgprSerial]                // 76 = Serial / 32
v_lshrrev_b32 v77, 0, v76                          // 77 = 76 / 1
v_and_b32 v77, 0, v77                              // v77 = v77 % 1
v_mul_lo_u32 v77, 0x10, v77                        // wave coordination offset 1
v_and_b32 v71, 31, v[vgprSerial]                   // v71 = v[vgprSerial] % 32
v_lshrrev_b32 v71, 4, v71                          // 71 = 71 / 16
                                                   // thread0 * continuous_output (multiplier is 1, do nothing)
v_add_lshl_u32 v75, v77, v71, 0                    // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v72, v75, s[sgprStrideC1J]            //  offset 1
v_mul_lo_u32 v73, v75, s[sgprStrideD1J]            //  offset 1
v_and_b32 v70, 0, v76                              // v70 = v76 % 1
v_mul_lo_u32 v70, 0x10, v70                        // wave coordination offset 0
v_and_b32 v77, 15, v[vgprSerial]                   // v77 = v[vgprSerial] % 16
v_add_lshl_u32 v74, v77, v70, 0                    // coordination 0 = vwA * (wave_id0 + tid0)
s_mul_i32 s8, 32, s[sgprWorkGroup0]                // wgp0 * MT0
v_add_nc_u32 v70, s8, v74                          // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s8, 48, s[sgprWorkGroup1]                // wgp1 * MT1
v_add_nc_u32 v71, s8, v75                          // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1
.set vgprLsuReduction, 76
/* Size of vgprLsuReduction is 12 */
v_lshrrev_b32 v89, 5, v[vgprSerial]                // 89 = Serial / 32
v_lshrrev_b32 v88, 0, v89                          // Get LSU wave ID
v_and_b32 v89, 0x0, v89                            // Get wave ID

/* LocalSplitU: local write 1/1 */
label_localSplitULocalWriteAndRead_1:
s_mov_b32 s8, 0x1800                               // dataPerWave (6144)
v_and_b32 v93, 0x1f, v[vgprSerial]                 // initial addr
v_mul_lo_u32 v90, s8, v89                          // tmp = waveId * dataPerWave
v_lshl_add_u32 v93, v93, 2, v90                    // addr = initial addr + tmp
                                                   // ldsStride = waveNum * dataPerWave (6144) (dup assign opt.)
v_mul_lo_u32 v90, s8, v88                          // tmp = (waveNum * dataPerWave) * lsu_id
v_add_nc_u32 v93, v90, v93                         // addr += tmp
v_mov_b32 v92, v[vgprValuC+0]                      // copy acc[0] to vreg[92], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:0                     // arch[0]
v_mov_b32 v92, v[vgprValuC+8]                      // copy acc[8] to vreg[92], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:128                   // arch[1]
v_mov_b32 v92, v[vgprValuC+4]                      // copy acc[4] to vreg[92], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:256                   // arch[2]
v_mov_b32 v92, v[vgprValuC+12]                     // copy acc[12] to vreg[92], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:384                   // arch[3]
v_mov_b32 v92, v[vgprValuC+16]                     // copy acc[16] to vreg[92], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:512                   // arch[4]
v_mov_b32 v92, v[vgprValuC+24]                     // copy acc[24] to vreg[92], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:640                   // arch[5]
v_mov_b32 v92, v[vgprValuC+20]                     // copy acc[20] to vreg[92], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:768                   // arch[6]
v_mov_b32 v92, v[vgprValuC+28]                     // copy acc[28] to vreg[92], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:896                   // arch[7]
v_mov_b32 v92, v[vgprValuC+32]                     // copy acc[32] to vreg[92], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:1024                  // arch[8]
v_mov_b32 v92, v[vgprValuC+40]                     // copy acc[40] to vreg[92], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:1152                  // arch[9]
v_mov_b32 v92, v[vgprValuC+36]                     // copy acc[36] to vreg[92], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:1280                  // arch[10]
v_mov_b32 v92, v[vgprValuC+44]                     // copy acc[44] to vreg[92], LSU0 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:1408                  // arch[11]
v_mov_b32 v92, v[vgprValuC+1]                      // copy acc[1] to vreg[92], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:1536                  // arch[12]
v_mov_b32 v92, v[vgprValuC+9]                      // copy acc[9] to vreg[92], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:1664                  // arch[13]
v_mov_b32 v92, v[vgprValuC+5]                      // copy acc[5] to vreg[92], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:1792                  // arch[14]
v_mov_b32 v92, v[vgprValuC+13]                     // copy acc[13] to vreg[92], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:1920                  // arch[15]
v_mov_b32 v92, v[vgprValuC+17]                     // copy acc[17] to vreg[92], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:2048                  // arch[16]
v_mov_b32 v92, v[vgprValuC+25]                     // copy acc[25] to vreg[92], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:2176                  // arch[17]
v_mov_b32 v92, v[vgprValuC+21]                     // copy acc[21] to vreg[92], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:2304                  // arch[18]
v_mov_b32 v92, v[vgprValuC+29]                     // copy acc[29] to vreg[92], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:2432                  // arch[19]
v_mov_b32 v92, v[vgprValuC+33]                     // copy acc[33] to vreg[92], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:2560                  // arch[20]
v_mov_b32 v92, v[vgprValuC+41]                     // copy acc[41] to vreg[92], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:2688                  // arch[21]
v_mov_b32 v92, v[vgprValuC+37]                     // copy acc[37] to vreg[92], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:2816                  // arch[22]
v_mov_b32 v92, v[vgprValuC+45]                     // copy acc[45] to vreg[92], LSU1 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:2944                  // arch[23]
v_mov_b32 v92, v[vgprValuC+2]                      // copy acc[2] to vreg[92], LSU2 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:3072                  // arch[24]
v_mov_b32 v92, v[vgprValuC+10]                     // copy acc[10] to vreg[92], LSU2 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:3200                  // arch[25]
v_mov_b32 v92, v[vgprValuC+6]                      // copy acc[6] to vreg[92], LSU2 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:3328                  // arch[26]
v_mov_b32 v92, v[vgprValuC+14]                     // copy acc[14] to vreg[92], LSU2 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:3456                  // arch[27]
v_mov_b32 v92, v[vgprValuC+18]                     // copy acc[18] to vreg[92], LSU2 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:3584                  // arch[28]
v_mov_b32 v92, v[vgprValuC+26]                     // copy acc[26] to vreg[92], LSU2 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:3712                  // arch[29]
v_mov_b32 v92, v[vgprValuC+22]                     // copy acc[22] to vreg[92], LSU2 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:3840                  // arch[30]
v_mov_b32 v92, v[vgprValuC+30]                     // copy acc[30] to vreg[92], LSU2 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:3968                  // arch[31]
v_mov_b32 v92, v[vgprValuC+34]                     // copy acc[34] to vreg[92], LSU2 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:4096                  // arch[32]
v_mov_b32 v92, v[vgprValuC+42]                     // copy acc[42] to vreg[92], LSU2 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:4224                  // arch[33]
v_mov_b32 v92, v[vgprValuC+38]                     // copy acc[38] to vreg[92], LSU2 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:4352                  // arch[34]
v_mov_b32 v92, v[vgprValuC+46]                     // copy acc[46] to vreg[92], LSU2 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:4480                  // arch[35]
v_mov_b32 v92, v[vgprValuC+3]                      // copy acc[3] to vreg[92], LSU3 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:4608                  // arch[36]
v_mov_b32 v92, v[vgprValuC+11]                     // copy acc[11] to vreg[92], LSU3 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:4736                  // arch[37]
v_mov_b32 v92, v[vgprValuC+7]                      // copy acc[7] to vreg[92], LSU3 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:4864                  // arch[38]
v_mov_b32 v92, v[vgprValuC+15]                     // copy acc[15] to vreg[92], LSU3 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:4992                  // arch[39]
v_mov_b32 v92, v[vgprValuC+19]                     // copy acc[19] to vreg[92], LSU3 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:5120                  // arch[40]
v_mov_b32 v92, v[vgprValuC+27]                     // copy acc[27] to vreg[92], LSU3 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:5248                  // arch[41]
v_mov_b32 v92, v[vgprValuC+23]                     // copy acc[23] to vreg[92], LSU3 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:5376                  // arch[42]
v_mov_b32 v92, v[vgprValuC+31]                     // copy acc[31] to vreg[92], LSU3 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:5504                  // arch[43]
v_mov_b32 v92, v[vgprValuC+35]                     // copy acc[35] to vreg[92], LSU3 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:5632                  // arch[44]
v_mov_b32 v92, v[vgprValuC+43]                     // copy acc[43] to vreg[92], LSU3 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:5760                  // arch[45]
v_mov_b32 v92, v[vgprValuC+39]                     // copy acc[39] to vreg[92], LSU3 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:5888                  // arch[46]
v_mov_b32 v92, v[vgprValuC+47]                     // copy acc[47] to vreg[92], LSU3 will process
s_nop 1                                            // 2 wait states required before reading vgpr
ds_store_b32 v93, v92 offset:6016                  // arch[47]

/* LocalSplitU: local read 1/1 */
v_and_b32 v93, 0x1f, v[vgprSerial]                 // initial addr
                                                   // wave offset (6144) (dup assign opt.)
v_mul_lo_u32 v90, s8, v89                          // wave offset = wave_id * wave offset
v_lshl_add_u32 v93, v93, 2, v90                    // addr = initial addr + wave offset
s_mov_b32 s8, 0x600                                // LSU Process Offset 1536
v_mul_lo_u32 v90, s8, v88                          // lsu offset = lsu_id * LSU Process Offset
v_add_nc_u32 v93, v93, v90                         // addr += lsu offset
s_waitcnt lgkmcnt(0)                               // wait for all writes
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier                                          // post-lsu local write
label_localSplitULocalRead_1:
ds_load_b32 v[vgprLsuReduction+0], v93 offset:0    // r=0 i=0, from acc[0]
ds_load_b32 v96, v93 offset:6144                   // r=1 i=0, from acc[0]
ds_load_b32 v108, v93 offset:12288                 // r=2 i=0, from acc[0]
ds_load_b32 v120, v93 offset:18432                 // r=3 i=0, from acc[0]
ds_load_b32 v[vgprLsuReduction+1], v93 offset:128  // r=0 i=1, from acc[8]
ds_load_b32 v97, v93 offset:6272                   // r=1 i=1, from acc[8]
ds_load_b32 v109, v93 offset:12416                 // r=2 i=1, from acc[8]
ds_load_b32 v121, v93 offset:18560                 // r=3 i=1, from acc[8]
ds_load_b32 v[vgprLsuReduction+2], v93 offset:256  // r=0 i=2, from acc[4]
ds_load_b32 v98, v93 offset:6400                   // r=1 i=2, from acc[4]
ds_load_b32 v110, v93 offset:12544                 // r=2 i=2, from acc[4]
ds_load_b32 v122, v93 offset:18688                 // r=3 i=2, from acc[4]
ds_load_b32 v[vgprLsuReduction+3], v93 offset:384  // r=0 i=3, from acc[12]
ds_load_b32 v99, v93 offset:6528                   // r=1 i=3, from acc[12]
ds_load_b32 v111, v93 offset:12672                 // r=2 i=3, from acc[12]
ds_load_b32 v123, v93 offset:18816                 // r=3 i=3, from acc[12]
ds_load_b32 v[vgprLsuReduction+4], v93 offset:512  // r=0 i=4, from acc[16]
ds_load_b32 v100, v93 offset:6656                  // r=1 i=4, from acc[16]
ds_load_b32 v112, v93 offset:12800                 // r=2 i=4, from acc[16]
ds_load_b32 v124, v93 offset:18944                 // r=3 i=4, from acc[16]
ds_load_b32 v[vgprLsuReduction+5], v93 offset:640  // r=0 i=5, from acc[24]
ds_load_b32 v101, v93 offset:6784                  // r=1 i=5, from acc[24]
ds_load_b32 v113, v93 offset:12928                 // r=2 i=5, from acc[24]
ds_load_b32 v125, v93 offset:19072                 // r=3 i=5, from acc[24]
ds_load_b32 v[vgprLsuReduction+6], v93 offset:768  // r=0 i=6, from acc[20]
ds_load_b32 v102, v93 offset:6912                  // r=1 i=6, from acc[20]
ds_load_b32 v114, v93 offset:13056                 // r=2 i=6, from acc[20]
ds_load_b32 v126, v93 offset:19200                 // r=3 i=6, from acc[20]
ds_load_b32 v[vgprLsuReduction+7], v93 offset:896  // r=0 i=7, from acc[28]
ds_load_b32 v103, v93 offset:7040                  // r=1 i=7, from acc[28]
ds_load_b32 v115, v93 offset:13184                 // r=2 i=7, from acc[28]
ds_load_b32 v127, v93 offset:19328                 // r=3 i=7, from acc[28]
ds_load_b32 v[vgprLsuReduction+8], v93 offset:1024 // r=0 i=8, from acc[32]
ds_load_b32 v104, v93 offset:7168                  // r=1 i=8, from acc[32]
ds_load_b32 v116, v93 offset:13312                 // r=2 i=8, from acc[32]
ds_load_b32 v128, v93 offset:19456                 // r=3 i=8, from acc[32]
ds_load_b32 v[vgprLsuReduction+9], v93 offset:1152 // r=0 i=9, from acc[40]
ds_load_b32 v105, v93 offset:7296                  // r=1 i=9, from acc[40]
ds_load_b32 v117, v93 offset:13440                 // r=2 i=9, from acc[40]
ds_load_b32 v129, v93 offset:19584                 // r=3 i=9, from acc[40]
ds_load_b32 v[vgprLsuReduction+10], v93 offset:1280 // r=0 i=10, from acc[36]
ds_load_b32 v106, v93 offset:7424                  // r=1 i=10, from acc[36]
ds_load_b32 v118, v93 offset:13568                 // r=2 i=10, from acc[36]
ds_load_b32 v130, v93 offset:19712                 // r=3 i=10, from acc[36]
ds_load_b32 v[vgprLsuReduction+11], v93 offset:1408 // r=0 i=11, from acc[44]
ds_load_b32 v107, v93 offset:7552                  // r=1 i=11, from acc[44]
ds_load_b32 v119, v93 offset:13696                 // r=2 i=11, from acc[44]
ds_load_b32 v131, v93 offset:19840                 // r=3 i=11, from acc[44]
s_waitcnt lgkmcnt(44)                              // wait count is (48-4)
v_add_f32 v[vgprLsuReduction+0], v96, v[vgprLsuReduction+0]
v_add_f32 v[vgprLsuReduction+0], v108, v[vgprLsuReduction+0]
v_add_f32 v[vgprLsuReduction+0], v120, v[vgprLsuReduction+0]
s_waitcnt lgkmcnt(40)                              // wait count is (48-8)
v_add_f32 v[vgprLsuReduction+1], v97, v[vgprLsuReduction+1]
v_add_f32 v[vgprLsuReduction+1], v109, v[vgprLsuReduction+1]
v_add_f32 v[vgprLsuReduction+1], v121, v[vgprLsuReduction+1]
s_waitcnt lgkmcnt(36)                              // wait count is (48-12)
v_add_f32 v[vgprLsuReduction+2], v98, v[vgprLsuReduction+2]
v_add_f32 v[vgprLsuReduction+2], v110, v[vgprLsuReduction+2]
v_add_f32 v[vgprLsuReduction+2], v122, v[vgprLsuReduction+2]
s_waitcnt lgkmcnt(32)                              // wait count is (48-16)
v_add_f32 v[vgprLsuReduction+3], v99, v[vgprLsuReduction+3]
v_add_f32 v[vgprLsuReduction+3], v111, v[vgprLsuReduction+3]
v_add_f32 v[vgprLsuReduction+3], v123, v[vgprLsuReduction+3]
s_waitcnt lgkmcnt(28)                              // wait count is (48-20)
v_add_f32 v[vgprLsuReduction+4], v100, v[vgprLsuReduction+4]
v_add_f32 v[vgprLsuReduction+4], v112, v[vgprLsuReduction+4]
v_add_f32 v[vgprLsuReduction+4], v124, v[vgprLsuReduction+4]
s_waitcnt lgkmcnt(24)                              // wait count is (48-24)
v_add_f32 v[vgprLsuReduction+5], v101, v[vgprLsuReduction+5]
v_add_f32 v[vgprLsuReduction+5], v113, v[vgprLsuReduction+5]
v_add_f32 v[vgprLsuReduction+5], v125, v[vgprLsuReduction+5]
s_waitcnt lgkmcnt(20)                              // wait count is (48-28)
v_add_f32 v[vgprLsuReduction+6], v102, v[vgprLsuReduction+6]
v_add_f32 v[vgprLsuReduction+6], v114, v[vgprLsuReduction+6]
v_add_f32 v[vgprLsuReduction+6], v126, v[vgprLsuReduction+6]
s_waitcnt lgkmcnt(16)                              // wait count is (48-32)
v_add_f32 v[vgprLsuReduction+7], v103, v[vgprLsuReduction+7]
v_add_f32 v[vgprLsuReduction+7], v115, v[vgprLsuReduction+7]
v_add_f32 v[vgprLsuReduction+7], v127, v[vgprLsuReduction+7]
s_waitcnt lgkmcnt(12)                              // wait count is (48-36)
v_add_f32 v[vgprLsuReduction+8], v104, v[vgprLsuReduction+8]
v_add_f32 v[vgprLsuReduction+8], v116, v[vgprLsuReduction+8]
v_add_f32 v[vgprLsuReduction+8], v128, v[vgprLsuReduction+8]
s_waitcnt lgkmcnt(8)                               // wait count is (48-40)
v_add_f32 v[vgprLsuReduction+9], v105, v[vgprLsuReduction+9]
v_add_f32 v[vgprLsuReduction+9], v117, v[vgprLsuReduction+9]
v_add_f32 v[vgprLsuReduction+9], v129, v[vgprLsuReduction+9]
s_waitcnt lgkmcnt(4)                               // wait count is (48-44)
v_add_f32 v[vgprLsuReduction+10], v106, v[vgprLsuReduction+10]
v_add_f32 v[vgprLsuReduction+10], v118, v[vgprLsuReduction+10]
v_add_f32 v[vgprLsuReduction+10], v130, v[vgprLsuReduction+10]
s_waitcnt lgkmcnt(0)                               // wait count is (48-48)
v_add_f32 v[vgprLsuReduction+11], v107, v[vgprLsuReduction+11]
v_add_f32 v[vgprLsuReduction+11], v119, v[vgprLsuReduction+11]
v_add_f32 v[vgprLsuReduction+11], v131, v[vgprLsuReduction+11]
.set vgprValuC, 76

/* LocalSplitU: global write indices */
v_lshrrev_b32 v88, 5, v[vgprSerial]                // 88 = Serial / 32
v_lshrrev_b32 v88, 0, v88                          // 88 = 88 / 1
/* valid offset coord0 is zero. */
s_mov_b32 s8, 0x2                                  // a valid offset
v_mul_lo_u32 v90, v88, s8                          // wave LSU offset
v_add_nc_u32 v71, v90, v71                         // coord1 += LSU offset1
v_add_nc_u32 v75, v90, v75                         // coord1InMT += LSU offset1
v_mul_lo_u32 v72, v75, s[sgprStrideC1J]            //  offset 1
v_mul_lo_u32 v73, v75, s[sgprStrideD1J]            //  offset 1

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
s_and_b32 s68, 31, s[sgprSizeI]                    // s68 = s[sgprSizeI] % 32
s_add_u32 s69, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s69                // wg0 >= nwg0-1 ?
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_MB_Else_1       // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 48 > 0 -> Edge store; else -> NonEdge store */
s_mul_hi_u32 s71, s[sgprSizeJ], 178956971          // STATIC_DIV: divisor=48
s_mul_i32 s70, s[sgprSizeJ], 178956971             // tmp = dividend * magic
s_lshr_b64 s[70:71], s[70:71], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s69, s70                                 // quotient
s_mul_i32 s70, s69, 48                             // quotient*divisor
s_sub_u32 s68, s[sgprSizeJ], s70                   // rReg = dividend - quotient*divisor
s_add_u32 s69, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s69                // wg1 >= nwg1-1
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_MB_Then_1       // jump if edges required
label_GW_B0_FD0_VW1_MB_NonEdge_1:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=160 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
v_add_lshl_u32 v74, v73, v70, 2                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=70, coord0Vgpr=70 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0)] */

/* apply mask, calc new C and issue writes */
buffer_store_b32 v76, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v77, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
s_mul_i32 s8, s[sgprStrideD1J], 32                 // scale StrideD *= numRows(8) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(8): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(8): gra SRD += inc(upper)
buffer_store_b32 v78, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v79, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
s_mul_i32 s8, s[sgprStrideD1J], 32                 // scale StrideD *= numRows(8) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(8): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(8): gra SRD += inc(upper)
buffer_store_b32 v80, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v81, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
s_mul_i32 s8, s[sgprStrideD1J], 32                 // scale StrideD *= numRows(8) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(8): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(8): gra SRD += inc(upper)
buffer_store_b32 v82, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v83, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
s_mul_i32 s8, s[sgprStrideD1J], 32                 // scale StrideD *= numRows(8) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(8): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(8): gra SRD += inc(upper)
buffer_store_b32 v84, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v85, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
s_mul_i32 s8, s[sgprStrideD1J], 32                 // scale StrideD *= numRows(8) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(8): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(8): gra SRD += inc(upper)
buffer_store_b32 v86, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v87, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_B0_FD0_VW1_MB_NonEdgeEnd_1:
label_GW_B0_FD0_VW1_MB_Else_1:
label_GW_B0_FD0_VW1_MB_Then_1:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=80 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v90, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s68, v70, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v74, v73, v70, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v74, v90, v74, s70                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v88, vcc_lo, v70, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v88, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v75, v73, v88, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v75, v90, v75, s70                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v71, vcc_lo, v71, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 8                 // scale stride
v_add_nc_i32 v72, v72, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 8                 // scale stride
v_add_nc_i32 v73, v73, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v70, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v95, v73, v70, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v95, v90, v95, s70                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
v_add_co_u32 v88, vcc_lo, v70, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v88, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v96, v73, v88, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v96, v90, v96, s70                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
v_add_co_u32 v71, vcc_lo, v71, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 8                 // scale stride
v_add_nc_i32 v72, v72, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 8                 // scale stride
v_add_nc_i32 v73, v73, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v70, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v97, v73, v70, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v97, v90, v97, s70                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
v_add_co_u32 v88, vcc_lo, v70, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v88, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v98, v73, v88, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v98, v90, v98, s70                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
v_add_co_u32 v71, vcc_lo, v71, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 8                 // scale stride
v_add_nc_i32 v72, v72, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 8                 // scale stride
v_add_nc_i32 v73, v73, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v70, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v99, v73, v70, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v99, v90, v99, s70                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
v_add_co_u32 v88, vcc_lo, v70, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v88, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v100, v73, v88, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v100, v90, v100, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
v_add_co_u32 v71, vcc_lo, v71, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 8                 // scale stride
v_add_nc_i32 v72, v72, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 8                 // scale stride
v_add_nc_i32 v73, v73, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v70, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v101, v73, v70, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v101, v90, v101, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
v_add_co_u32 v88, vcc_lo, v70, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v88, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v102, v73, v88, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v102, v90, v102, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
v_add_co_u32 v71, vcc_lo, v71, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 8                 // scale stride
v_add_nc_i32 v72, v72, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 8                 // scale stride
v_add_nc_i32 v73, v73, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v70, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v103, v73, v70, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v103, v90, v103, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
v_add_co_u32 v88, vcc_lo, v70, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v88, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v104, v73, v88, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v104, v90, v104, s70                 // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0)] */

/* apply mask, calc new C and issue writes */
buffer_store_b32 v76, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v77, v75, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v78, v95, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v79, v96, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v80, v97, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v81, v98, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v82, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v83, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v84, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v85, v102, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v86, v103, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v87, v104, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
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
s_add_u32 s8, s[sgprPersistentWorkGroupIndex], 1   // input partial tile index
s_mul_hi_u32 s69, s[sgprPersistentIterationEnd], s[sgprMagicNumberItersPerTile] // s_magic mul, div alg 2
s_lshr_b32 s70, s[sgprMagicShiftItersPerTile], 31  // tmpS = extract abit
s_mul_i32 s68, s[sgprPersistentIterationEnd], s70  // s_magic mul, div alg 2
s_add_u32 s68, s68, s69
s_and_b32 s70, s[sgprMagicShiftItersPerTile], 2147483647 // tmpS = remove abit to final shift
s_lshr_b32 s68, s68, s70                           // sMagicDiv Alg 2
s_mul_i32 s68, s68, s[sgprItersPerTile]            // start iteration of partial tile
s_sub_u32 s9, s[sgprPersistentIterationEnd], s68   // calc iterations completed by this WG
label_SK_Fixup:
s_lshl_b32 s68, s8, 2                              // flag offset based on physical partial slot
s_load_b32 s70, s[sgprAddressFlags:sgprAddressFlags+1], s68 glc dlc // get flag
s_waitcnt lgkmcnt(0)                               // wait for flag load
s_cmp_eq_u32 s70, 1                                // check if ready
s_cbranch_scc0 label_SK_Fixup                      // if flag not set, wait and check again
s_barrier                                          // wait for all workgroups before resetting flag
v_readfirstlane_b32 s70, v[vgprSerial]             // Wave 0 updates flags
s_cmp_eq_u32 s70, 0                                // Check for wave 0
s_cbranch_scc0 label_SK_SkipFlagReset              // Skip flag reset
v_mov_b32 v74, s70                                 // move flag value to vgpr
v_mov_b32 v75, 0                                   // zero vaddr offset
s_mov_b64 s[72:73], s[sgprAddressFlags:sgprAddressFlags+1]
s_mov_b32 s74, BufferOOB
s_mov_b32 s75, Srd127_96
buffer_store_b32 v74, v75, s[72:75], s68 offen offset:0 glc dlc // reset flag
s_waitcnt_vscnt null 0                             // wait for data store
label_SK_SkipFlagReset:
label_Fixup_E0:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=78 */
s_mov_b64 s[sgprSrdWS:sgprSrdWS+1], s[sgprAddressWS:sgprAddressWS+1]
s_mov_b32 s[sgprSrdWS+2], BufferOOB
s_mov_b32 s[sgprSrdWS+3], Srd127_96

s_mul_i32 s62, 0x1800, s8                          // Offset to correct partials tile (low word)
s_mul_hi_u32 s53, 0x1800, s8                       // partials tile offset (high word) for 64-bit SRD
s_add_u32 s[sgprSrdWS+0], s[sgprSrdWS+0], s62      // add lo to SRD
s_addc_u32 s[sgprSrdWS+1], s[sgprSrdWS+1], s53     // add hi (offset high word + lo carry) to SRD
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Fixup Batch #0 (d1,d0,vc1,vc0) =       */
/*      (0,0,0,0:vw1); (0,1,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_lshlrev_b32 v75, 2, v[vgprSerial]                // v75 = v[vgprSerial] * 4
s_mov_b32 s62, 0                                   // Init sgpr offset
buffer_load_b32 v95, v75, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v96, v75, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v97, v75, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v98, v75, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v99, v75, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v100, v75, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v101, v75, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v102, v75, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v103, v75, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v104, v75, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v105, v75, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 512                            // Inc sgpr offset
buffer_load_b32 v106, v75, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS

/* apply mask, calc new C and issue writes */
v_mov_b32 v92, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v93, 0x7fff0000                          // fp32 Nan
v_mov_b32 v94, 0x7fff                              // rounding bias for bfloat16

s_waitcnt vmcnt(11)                                // wait C (interleaved) 11 = 12 - 0 - 1
v_add_f32 v[vgprValuC+0], v[vgprValuC+0], v95      // accum partials

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 12 - 1 - 1
v_add_f32 v[vgprValuC+1], v[vgprValuC+1], v96      // accum partials

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 12 - 2 - 1
v_add_f32 v[vgprValuC+2], v[vgprValuC+2], v97      // accum partials

s_waitcnt vmcnt(8)                                 // wait C (interleaved) 8 = 12 - 3 - 1
v_add_f32 v[vgprValuC+3], v[vgprValuC+3], v98      // accum partials

s_waitcnt vmcnt(7)                                 // wait C (interleaved) 7 = 12 - 4 - 1
v_add_f32 v[vgprValuC+4], v[vgprValuC+4], v99      // accum partials

s_waitcnt vmcnt(6)                                 // wait C (interleaved) 6 = 12 - 5 - 1
v_add_f32 v[vgprValuC+5], v[vgprValuC+5], v100     // accum partials

s_waitcnt vmcnt(5)                                 // wait C (interleaved) 5 = 12 - 6 - 1
v_add_f32 v[vgprValuC+6], v[vgprValuC+6], v101     // accum partials

s_waitcnt vmcnt(4)                                 // wait C (interleaved) 4 = 12 - 7 - 1
v_add_f32 v[vgprValuC+7], v[vgprValuC+7], v102     // accum partials

s_waitcnt vmcnt(3)                                 // wait C (interleaved) 3 = 12 - 8 - 1
v_add_f32 v[vgprValuC+8], v[vgprValuC+8], v103     // accum partials

s_waitcnt vmcnt(2)                                 // wait C (interleaved) 2 = 12 - 9 - 1
v_add_f32 v[vgprValuC+9], v[vgprValuC+9], v104     // accum partials

s_waitcnt vmcnt(1)                                 // wait C (interleaved) 1 = 12 - 10 - 1
v_add_f32 v[vgprValuC+10], v[vgprValuC+10], v105   // accum partials

s_waitcnt vmcnt(0)                                 // wait C (interleaved) 0 = 12 - 11 - 1
v_add_f32 v[vgprValuC+11], v[vgprValuC+11], v106   // accum partials
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_mul_i32 s53, s[sgprskTiles], s[sgprItersPerTile]
s_mul_i32 s61, s[sgprSKItersPerWG], s[sgprskGrid]
s_sub_u32 s53, s53, s61                            // skTiles * ItersPerTile - SKItersPerWG * skGrid
s_bitcmp1_b32 s[sgprMagicShiftItersPerTile], 29    // USO on? (bit 29 of MagicShiftItersPerTile); off -> historical global peer size
s_cbranch_scc0 label_SK_PeerGlobal                 // USO off -> historical global mapping
s_cmp_eq_u32 s[sgprskTiles], 0                     // skTiles == 0?
s_cbranch_scc1 label_SK_PeerNoTiles                // global peer size
v_cvt_f32_u32 v74, s[sgprskTiles]                  // skGrid % skTiles
v_rcp_iflag_f32 v74, v74                           // skGrid % skTiles
v_mov_b32 v75, 0x4f7ffffe                          // 2^32 - 512
v_mul_f32 v74, v75, v74                            // 2^32 / divisor
v_cvt_u32_f32 v74, v74                             // skGrid % skTiles
v_mul_lo_u32 v75, v74, s[sgprskTiles]              // refine 2^32 / divisor
v_sub_nc_u32 v75, 0, v75                           // refine 2^32 / divisor
v_mul_hi_u32 v75, v74, v75                         // refine 2^32 / divisor
v_add_nc_u32 v74, v74, v75                         // refine 2^32 / divisor
v_mul_hi_u32 v74, s[sgprskGrid], v74               // skGrid % skTiles
v_mul_lo_u32 v75, v74, s[sgprskTiles]              // skGrid % skTiles
v_sub_nc_u32 v75, s[sgprskGrid], v75               // skGrid % skTiles
v_cmp_ge_u32 vcc_lo, v75, s[sgprskTiles]           // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v74, 1, v74                           // quotient + 1
v_sub_nc_u32 v75, v75, s[sgprskTiles]              // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v75, s[sgprskTiles]           // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v74, 1, v74                           // quotient + 1
v_sub_nc_u32 v75, v75, s[sgprskTiles]              // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s61, v74                       // quotient
v_readfirstlane_b32 s61, v75                       // remainder
s_cmp_eq_u32 s61, 0                                // skGrid % skTiles == 0?
s_cbranch_scc0 label_SK_PeerGlobal                 // ragged -> global peer
label_SK_PeerPerTile:
v_cvt_f32_u32 v74, s[sgprskTiles]                  // F = skGrid / skTiles
v_rcp_iflag_f32 v74, v74                           // F = skGrid / skTiles
v_mov_b32 v75, 0x4f7ffffe                          // 2^32 - 512
v_mul_f32 v74, v75, v74                            // 2^32 / divisor
v_cvt_u32_f32 v74, v74                             // F = skGrid / skTiles
v_mul_lo_u32 v75, v74, s[sgprskTiles]              // refine 2^32 / divisor
v_sub_nc_u32 v75, 0, v75                           // refine 2^32 / divisor
v_mul_hi_u32 v75, v74, v75                         // refine 2^32 / divisor
v_add_nc_u32 v74, v74, v75                         // refine 2^32 / divisor
v_mul_hi_u32 v74, s[sgprskGrid], v74               // F = skGrid / skTiles
v_mul_lo_u32 v75, v74, s[sgprskTiles]              // F = skGrid / skTiles
v_sub_nc_u32 v75, s[sgprskGrid], v75               // F = skGrid / skTiles
v_cmp_ge_u32 vcc_lo, v75, s[sgprskTiles]           // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v74, 1, v74                           // quotient + 1
v_sub_nc_u32 v75, v75, s[sgprskTiles]              // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v75, s[sgprskTiles]           // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v74, 1, v74                           // quotient + 1
v_sub_nc_u32 v75, v75, s[sgprskTiles]              // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s61, v74                       // quotient
s_mov_b32 s53, s61                                 // F = skGrid / skTiles
v_cvt_f32_u32 v74, s53                             // s = cta % F
v_rcp_iflag_f32 v74, v74                           // s = cta % F
v_mov_b32 v75, 0x4f7ffffe                          // 2^32 - 512
v_mul_f32 v74, v75, v74                            // 2^32 / divisor
v_cvt_u32_f32 v74, v74                             // s = cta % F
v_mul_lo_u32 v75, v74, s53                         // refine 2^32 / divisor
v_sub_nc_u32 v75, 0, v75                           // refine 2^32 / divisor
v_mul_hi_u32 v75, v74, v75                         // refine 2^32 / divisor
v_add_nc_u32 v74, v74, v75                         // refine 2^32 / divisor
v_mul_hi_u32 v74, s8, v74                          // s = cta % F
v_mul_lo_u32 v75, v74, s53                         // s = cta % F
v_sub_nc_u32 v75, s8, v75                          // s = cta % F
v_cmp_ge_u32 vcc_lo, v75, s53                      // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v74, 1, v74                           // quotient + 1
v_sub_nc_u32 v75, v75, s53                         // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v75, s53                      // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v74, 1, v74                           // quotient + 1
v_sub_nc_u32 v75, v75, s53                         // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s61, v74                       // quotient
v_readfirstlane_b32 s61, v75                       // remainder
s_mul_i32 s53, s53, s[sgprSKItersPerWG]            // F * SKItersPerWG
s_sub_u32 s53, s[sgprItersPerTile], s53            // remI = ItersPerTile - F*W
s_cmp_lt_u32 s61, s53                              // s < remI?
s_cselect_b32 s61, 1, 0                            // extra iter within tile
s_add_u32 s61, s[sgprSKItersPerWG], s61            // chunk = W + (s < remI)
s_branch label_SK_PeerDone                         // skip global peer
label_SK_PeerNoTiles:
label_SK_PeerGlobal:
s_add_u32 s61, s[sgprSKItersPerWG], 1              // Add extra iter
s_cmp_lt_u32 s8, s53                               // Check if next WG had an extra iteration
s_cselect_b32 s61, s61, s[sgprSKItersPerWG]        // Select correct number of iterations for next WG
label_SK_PeerDone:
s_add_u32 s9, s9, s61                              // next partial tile iteration
s_add_u32 s8, s8, 1                                // next partial tile index
s_cmp_lt_u32 s9, s[sgprItersPerTile]               // done loading partial tiles?
s_cbranch_scc1 label_SK_Fixup                      // Branch to continue fixup loop
label_SK_Store:
s_cmpk_eq_u32 s[sgprBeta], 0                       // Beta == 0
s_cbranch_scc0 label_GW_B1_GSU1                    // Branch if Beta is not zero

label_GW_B0_GSU1:
label_GW_B0_FD0_GSU1:

/* Edge/NonEdge store path check (M): Size % 32 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s68, 31, s[sgprSizeI]                    // s68 = s[sgprSizeI] % 32
s_add_u32 s69, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s69                // wg0 >= nwg0-1 ?
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_GSU1_Else_1     // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 48 > 0 -> Edge store; else -> NonEdge store */
s_mul_hi_u32 s71, s[sgprSizeJ], 178956971          // STATIC_DIV: divisor=48
s_mul_i32 s70, s[sgprSizeJ], 178956971             // tmp = dividend * magic
s_lshr_b64 s[70:71], s[70:71], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s69, s70                                 // quotient
s_mul_i32 s70, s69, 48                             // quotient*divisor
s_sub_u32 s68, s[sgprSizeJ], s70                   // rReg = dividend - quotient*divisor
s_add_u32 s69, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s69                // wg1 >= nwg1-1
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_GSU1_Then_1     // jump if edges required
label_GW_B0_FD0_VW1_GSU1_NonEdge_1:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=160 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
v_add_lshl_u32 v74, v73, v70, 1                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=70, coord0Vgpr=70 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0)] */
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

/* apply mask, calc new C and issue writes */
v_mov_b32 v92, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v93, 0x7fff0000                          // fp32 Nan
v_mov_b32 v94, 0x7fff                              // rounding bias for bfloat16
v_cmp_u_f32 s8, v[vgprValuC+0], v[vgprValuC+0]     // check Nan
v_bfe_u32 v91, v[vgprValuC+0], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+0], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+0], v91, v93, s8
v_lshrrev_b32 v76, 16, v[vgprValuC+0]              // convert C to bf16
buffer_store_b16 v76, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+1], v[vgprValuC+1]     // check Nan
v_bfe_u32 v91, v[vgprValuC+1], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+1], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+1], v91, v93, s8
v_lshrrev_b32 v77, 16, v[vgprValuC+1]              // convert C to bf16
buffer_store_b16 v77, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
v_cmp_u_f32 s8, v[vgprValuC+2], v[vgprValuC+2]     // check Nan
v_bfe_u32 v91, v[vgprValuC+2], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+2], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+2], v91, v93, s8
v_lshrrev_b32 v78, 16, v[vgprValuC+2]              // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 16                 // scale StrideD *= numRows(8) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(8): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(8): gra SRD += inc(upper)
buffer_store_b16 v78, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+3], v[vgprValuC+3]     // check Nan
v_bfe_u32 v91, v[vgprValuC+3], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+3], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+3], v91, v93, s8
v_lshrrev_b32 v79, 16, v[vgprValuC+3]              // convert C to bf16
buffer_store_b16 v79, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
v_cmp_u_f32 s8, v[vgprValuC+4], v[vgprValuC+4]     // check Nan
v_bfe_u32 v91, v[vgprValuC+4], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+4], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+4], v91, v93, s8
v_lshrrev_b32 v80, 16, v[vgprValuC+4]              // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 16                 // scale StrideD *= numRows(8) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(8): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(8): gra SRD += inc(upper)
buffer_store_b16 v80, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+5], v[vgprValuC+5]     // check Nan
v_bfe_u32 v91, v[vgprValuC+5], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+5], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+5], v91, v93, s8
v_lshrrev_b32 v81, 16, v[vgprValuC+5]              // convert C to bf16
buffer_store_b16 v81, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
v_cmp_u_f32 s8, v[vgprValuC+6], v[vgprValuC+6]     // check Nan
v_bfe_u32 v91, v[vgprValuC+6], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+6], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+6], v91, v93, s8
v_lshrrev_b32 v82, 16, v[vgprValuC+6]              // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 16                 // scale StrideD *= numRows(8) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(8): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(8): gra SRD += inc(upper)
buffer_store_b16 v82, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+7], v[vgprValuC+7]     // check Nan
v_bfe_u32 v91, v[vgprValuC+7], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+7], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+7], v91, v93, s8
v_lshrrev_b32 v83, 16, v[vgprValuC+7]              // convert C to bf16
buffer_store_b16 v83, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
v_cmp_u_f32 s8, v[vgprValuC+8], v[vgprValuC+8]     // check Nan
v_bfe_u32 v91, v[vgprValuC+8], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+8], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+8], v91, v93, s8
v_lshrrev_b32 v84, 16, v[vgprValuC+8]              // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 16                 // scale StrideD *= numRows(8) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(8): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(8): gra SRD += inc(upper)
buffer_store_b16 v84, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+9], v[vgprValuC+9]     // check Nan
v_bfe_u32 v91, v[vgprValuC+9], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+9], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+9], v91, v93, s8
v_lshrrev_b32 v85, 16, v[vgprValuC+9]              // convert C to bf16
buffer_store_b16 v85, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
v_cmp_u_f32 s8, v[vgprValuC+10], v[vgprValuC+10]   // check Nan
v_bfe_u32 v91, v[vgprValuC+10], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+10], v91, v94          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+10], v91, v93, s8
v_lshrrev_b32 v86, 16, v[vgprValuC+10]             // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 16                 // scale StrideD *= numRows(8) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(8): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(8): gra SRD += inc(upper)
buffer_store_b16 v86, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+11], v[vgprValuC+11]   // check Nan
v_bfe_u32 v91, v[vgprValuC+11], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+11], v91, v94          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+11], v91, v93, s8
v_lshrrev_b32 v87, 16, v[vgprValuC+11]             // convert C to bf16
buffer_store_b16 v87, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW1_GSU1_NonEdgeEnd_1:
label_GW_B0_FD0_VW1_GSU1_Else_1:
label_GW_B0_FD0_VW1_GSU1_Then_1:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=80 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v90, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s68, v70, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v74, v73, v70, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v74, v90, v74, s70                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v88, vcc_lo, v70, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v88, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v75, v73, v88, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v75, v90, v75, s70                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v71, vcc_lo, v71, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 8                 // scale stride
v_add_nc_i32 v72, v72, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 8                 // scale stride
v_add_nc_i32 v73, v73, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v70, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v95, v73, v70, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v95, v90, v95, s70                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
v_add_co_u32 v88, vcc_lo, v70, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v88, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v96, v73, v88, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v96, v90, v96, s70                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
v_add_co_u32 v71, vcc_lo, v71, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 8                 // scale stride
v_add_nc_i32 v72, v72, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 8                 // scale stride
v_add_nc_i32 v73, v73, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v70, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v97, v73, v70, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v97, v90, v97, s70                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
v_add_co_u32 v88, vcc_lo, v70, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v88, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v98, v73, v88, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v98, v90, v98, s70                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
v_add_co_u32 v71, vcc_lo, v71, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 8                 // scale stride
v_add_nc_i32 v72, v72, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 8                 // scale stride
v_add_nc_i32 v73, v73, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v70, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v99, v73, v70, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v99, v90, v99, s70                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
v_add_co_u32 v88, vcc_lo, v70, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v88, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v100, v73, v88, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v100, v90, v100, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
v_add_co_u32 v71, vcc_lo, v71, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 8                 // scale stride
v_add_nc_i32 v72, v72, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 8                 // scale stride
v_add_nc_i32 v73, v73, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v70, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v101, v73, v70, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v101, v90, v101, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
v_add_co_u32 v88, vcc_lo, v70, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v88, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v102, v73, v88, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v102, v90, v102, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
v_add_co_u32 v71, vcc_lo, v71, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 8                 // scale stride
v_add_nc_i32 v72, v72, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 8                 // scale stride
v_add_nc_i32 v73, v73, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v70, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v103, v73, v70, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v103, v90, v103, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
v_add_co_u32 v88, vcc_lo, v70, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v88, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v104, v73, v88, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v104, v90, v104, s70                 // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0)] */
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

/* apply mask, calc new C and issue writes */
v_mov_b32 v92, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v93, 0x7fff0000                          // fp32 Nan
v_mov_b32 v94, 0x7fff                              // rounding bias for bfloat16
v_cmp_u_f32 s68, v[vgprValuC+0], v[vgprValuC+0]    // check Nan
v_bfe_u32 v91, v[vgprValuC+0], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+0], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+0], v91, v93, s68
v_lshrrev_b32 v76, 16, v[vgprValuC+0]              // convert C to bf16
buffer_store_b16 v76, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+1], v[vgprValuC+1]    // check Nan
v_bfe_u32 v91, v[vgprValuC+1], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+1], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+1], v91, v93, s68
v_lshrrev_b32 v77, 16, v[vgprValuC+1]              // convert C to bf16
buffer_store_b16 v77, v75, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+2], v[vgprValuC+2]    // check Nan
v_bfe_u32 v91, v[vgprValuC+2], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+2], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+2], v91, v93, s68
v_lshrrev_b32 v78, 16, v[vgprValuC+2]              // convert C to bf16
buffer_store_b16 v78, v95, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+3], v[vgprValuC+3]    // check Nan
v_bfe_u32 v91, v[vgprValuC+3], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+3], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+3], v91, v93, s68
v_lshrrev_b32 v79, 16, v[vgprValuC+3]              // convert C to bf16
buffer_store_b16 v79, v96, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+4], v[vgprValuC+4]    // check Nan
v_bfe_u32 v91, v[vgprValuC+4], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+4], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+4], v91, v93, s68
v_lshrrev_b32 v80, 16, v[vgprValuC+4]              // convert C to bf16
buffer_store_b16 v80, v97, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+5], v[vgprValuC+5]    // check Nan
v_bfe_u32 v91, v[vgprValuC+5], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+5], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+5], v91, v93, s68
v_lshrrev_b32 v81, 16, v[vgprValuC+5]              // convert C to bf16
buffer_store_b16 v81, v98, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+6], v[vgprValuC+6]    // check Nan
v_bfe_u32 v91, v[vgprValuC+6], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+6], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+6], v91, v93, s68
v_lshrrev_b32 v82, 16, v[vgprValuC+6]              // convert C to bf16
buffer_store_b16 v82, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+7], v[vgprValuC+7]    // check Nan
v_bfe_u32 v91, v[vgprValuC+7], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+7], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+7], v91, v93, s68
v_lshrrev_b32 v83, 16, v[vgprValuC+7]              // convert C to bf16
buffer_store_b16 v83, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+8], v[vgprValuC+8]    // check Nan
v_bfe_u32 v91, v[vgprValuC+8], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+8], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+8], v91, v93, s68
v_lshrrev_b32 v84, 16, v[vgprValuC+8]              // convert C to bf16
buffer_store_b16 v84, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+9], v[vgprValuC+9]    // check Nan
v_bfe_u32 v91, v[vgprValuC+9], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+9], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+9], v91, v93, s68
v_lshrrev_b32 v85, 16, v[vgprValuC+9]              // convert C to bf16
buffer_store_b16 v85, v102, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+10], v[vgprValuC+10]  // check Nan
v_bfe_u32 v91, v[vgprValuC+10], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+10], v91, v94          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+10], v91, v93, s68
v_lshrrev_b32 v86, 16, v[vgprValuC+10]             // convert C to bf16
buffer_store_b16 v86, v103, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+11], v[vgprValuC+11]  // check Nan
v_bfe_u32 v91, v[vgprValuC+11], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+11], v91, v94          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+11], v91, v93, s68
v_lshrrev_b32 v87, 16, v[vgprValuC+11]             // convert C to bf16
buffer_store_b16 v87, v104, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_GSU1:
label_GW_B1_FD0_GSU1:

/* Edge/NonEdge store path check (M): Size % 32 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s68, 31, s[sgprSizeI]                    // s68 = s[sgprSizeI] % 32
s_add_u32 s69, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s69                // wg0 >= nwg0-1 ?
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW1_GSU1_Else_1     // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 48 > 0 -> Edge store; else -> NonEdge store */
s_mul_hi_u32 s71, s[sgprSizeJ], 178956971          // STATIC_DIV: divisor=48
s_mul_i32 s70, s[sgprSizeJ], 178956971             // tmp = dividend * magic
s_lshr_b64 s[70:71], s[70:71], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s69, s70                                 // quotient
s_mul_i32 s70, s69, 48                             // quotient*divisor
s_sub_u32 s68, s[sgprSizeJ], s70                   // rReg = dividend - quotient*divisor
s_add_u32 s69, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s69                // wg1 >= nwg1-1
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW1_GSU1_Then_1     // jump if edges required
label_GW_B1_FD0_VW1_GSU1_NonEdge_1:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=78 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v75, v72, v70, 1                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=70, coord0Vgpr=70 (multiple bpe)
buffer_load_d16_b16 v95, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
buffer_load_d16_b16 v96, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 16                 // scale StrideC *= numRows(8) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(8): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(8): gra SRD += inc(upper)
buffer_load_d16_b16 v97, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
buffer_load_d16_b16 v98, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 16                 // scale StrideC *= numRows(8) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(8): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(8): gra SRD += inc(upper)
buffer_load_d16_b16 v99, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
buffer_load_d16_b16 v100, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 16                 // scale StrideC *= numRows(8) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(8): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(8): gra SRD += inc(upper)
buffer_load_d16_b16 v101, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
buffer_load_d16_b16 v102, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 16                 // scale StrideC *= numRows(8) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(8): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(8): gra SRD += inc(upper)
buffer_load_d16_b16 v103, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
buffer_load_d16_b16 v104, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 16                 // scale StrideC *= numRows(8) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(8): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(8): gra SRD += inc(upper)
buffer_load_d16_b16 v105, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
buffer_load_d16_b16 v106, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
v_add_lshl_u32 v74, v73, v70, 1                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=70, coord0Vgpr=70 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0)] */
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

/* apply mask, calc new C and issue writes */
v_mov_b32 v92, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v93, 0x7fff0000                          // fp32 Nan
v_mov_b32 v94, 0x7fff                              // rounding bias for bfloat16

s_waitcnt vmcnt(11)                                // vlcnt(11) = 12 - 1 (beta) (interleaved)
v_lshlrev_b32 v88, 16, v95                         // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+0], v88, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+0], v[vgprValuC+0]     // check Nan
v_bfe_u32 v91, v[vgprValuC+0], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+0], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+0], v91, v93, s8
v_lshrrev_b32 v76, 16, v[vgprValuC+0]              // convert C to bf16
buffer_store_b16 v76, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(10)                                // vlcnt(10) = 12 - 2 (beta) (interleaved)
v_lshlrev_b32 v88, 16, v96                         // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+1], v88, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+1], v[vgprValuC+1]     // check Nan
v_bfe_u32 v91, v[vgprValuC+1], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+1], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+1], v91, v93, s8
v_lshrrev_b32 v77, 16, v[vgprValuC+1]              // convert C to bf16
buffer_store_b16 v77, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

s_waitcnt vmcnt(9)                                 // vlcnt(9) = 12 - 3 (beta) (interleaved)
v_lshlrev_b32 v88, 16, v97                         // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+2], v88, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+2], v[vgprValuC+2]     // check Nan
v_bfe_u32 v91, v[vgprValuC+2], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+2], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+2], v91, v93, s8
v_lshrrev_b32 v78, 16, v[vgprValuC+2]              // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 16                 // scale StrideD *= numRows(8) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(8): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(8): gra SRD += inc(upper)
buffer_store_b16 v78, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(8)                                 // vlcnt(8) = 12 - 4 (beta) (interleaved)
v_lshlrev_b32 v88, 16, v98                         // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+3], v88, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+3], v[vgprValuC+3]     // check Nan
v_bfe_u32 v91, v[vgprValuC+3], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+3], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+3], v91, v93, s8
v_lshrrev_b32 v79, 16, v[vgprValuC+3]              // convert C to bf16
buffer_store_b16 v79, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

s_waitcnt vmcnt(7)                                 // vlcnt(7) = 12 - 5 (beta) (interleaved)
v_lshlrev_b32 v88, 16, v99                         // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+4], v88, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+4], v[vgprValuC+4]     // check Nan
v_bfe_u32 v91, v[vgprValuC+4], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+4], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+4], v91, v93, s8
v_lshrrev_b32 v80, 16, v[vgprValuC+4]              // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 16                 // scale StrideD *= numRows(8) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(8): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(8): gra SRD += inc(upper)
buffer_store_b16 v80, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(6)                                 // vlcnt(6) = 12 - 6 (beta) (interleaved)
v_lshlrev_b32 v88, 16, v100                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+5], v88, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+5], v[vgprValuC+5]     // check Nan
v_bfe_u32 v91, v[vgprValuC+5], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+5], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+5], v91, v93, s8
v_lshrrev_b32 v81, 16, v[vgprValuC+5]              // convert C to bf16
buffer_store_b16 v81, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

s_waitcnt vmcnt(5)                                 // vlcnt(5) = 12 - 7 (beta) (interleaved)
v_lshlrev_b32 v88, 16, v101                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+6], v88, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+6], v[vgprValuC+6]     // check Nan
v_bfe_u32 v91, v[vgprValuC+6], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+6], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+6], v91, v93, s8
v_lshrrev_b32 v82, 16, v[vgprValuC+6]              // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 16                 // scale StrideD *= numRows(8) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(8): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(8): gra SRD += inc(upper)
buffer_store_b16 v82, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(4)                                 // vlcnt(4) = 12 - 8 (beta) (interleaved)
v_lshlrev_b32 v88, 16, v102                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+7], v88, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+7], v[vgprValuC+7]     // check Nan
v_bfe_u32 v91, v[vgprValuC+7], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+7], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+7], v91, v93, s8
v_lshrrev_b32 v83, 16, v[vgprValuC+7]              // convert C to bf16
buffer_store_b16 v83, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

s_waitcnt vmcnt(3)                                 // vlcnt(3) = 12 - 9 (beta) (interleaved)
v_lshlrev_b32 v88, 16, v103                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+8], v88, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+8], v[vgprValuC+8]     // check Nan
v_bfe_u32 v91, v[vgprValuC+8], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+8], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+8], v91, v93, s8
v_lshrrev_b32 v84, 16, v[vgprValuC+8]              // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 16                 // scale StrideD *= numRows(8) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(8): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(8): gra SRD += inc(upper)
buffer_store_b16 v84, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(2)                                 // vlcnt(2) = 12 - 10 (beta) (interleaved)
v_lshlrev_b32 v88, 16, v104                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+9], v88, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+9], v[vgprValuC+9]     // check Nan
v_bfe_u32 v91, v[vgprValuC+9], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+9], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+9], v91, v93, s8
v_lshrrev_b32 v85, 16, v[vgprValuC+9]              // convert C to bf16
buffer_store_b16 v85, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

s_waitcnt vmcnt(1)                                 // vlcnt(1) = 12 - 11 (beta) (interleaved)
v_lshlrev_b32 v88, 16, v105                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+10], v88, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+10], v[vgprValuC+10]   // check Nan
v_bfe_u32 v91, v[vgprValuC+10], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+10], v91, v94          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+10], v91, v93, s8
v_lshrrev_b32 v86, 16, v[vgprValuC+10]             // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 16                 // scale StrideD *= numRows(8) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(8): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(8): gra SRD += inc(upper)
buffer_store_b16 v86, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(0)                                 // vlcnt(0) = 12 - 12 (beta) (interleaved)
v_lshlrev_b32 v88, 16, v106                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+11], v88, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+11], v[vgprValuC+11]   // check Nan
v_bfe_u32 v91, v[vgprValuC+11], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+11], v91, v94          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+11], v91, v93, s8
v_lshrrev_b32 v87, 16, v[vgprValuC+11]             // convert C to bf16
buffer_store_b16 v87, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_FD0_VW1_GSU1_NonEdgeEnd_1:
label_GW_B1_FD0_VW1_GSU1_Else_1:
label_GW_B1_FD0_VW1_GSU1_Then_1:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=52 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v90, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s68, v70, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v75, v72, v70, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v75, v90, v75, s70                   // LDC clip if OOB. offset
buffer_load_d16_b16 v74, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v75, v73, v70, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v75, v90, v75, s70                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v88, vcc_lo, v70, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v88, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v96, v72, v88, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v96, v90, v96, s70                   // LDC clip if OOB. offset
buffer_load_d16_b16 v95, v96, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v96, v73, v88, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v96, v90, v96, s70                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v71, vcc_lo, v71, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 8                 // scale stride
v_add_nc_i32 v72, v72, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 8                 // scale stride
v_add_nc_i32 v73, v73, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v70, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v98, v72, v70, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v98, v90, v98, s70                   // LDC clip if OOB. offset
buffer_load_d16_b16 v97, v98, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v98, v73, v70, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v98, v90, v98, s70                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
v_add_co_u32 v88, vcc_lo, v70, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v88, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v100, v72, v88, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v100, v90, v100, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v99, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v100, v73, v88, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v100, v90, v100, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
v_add_co_u32 v71, vcc_lo, v71, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 8                 // scale stride
v_add_nc_i32 v72, v72, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 8                 // scale stride
v_add_nc_i32 v73, v73, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v70, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v102, v72, v70, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v102, v90, v102, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v101, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v102, v73, v70, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v102, v90, v102, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
v_add_co_u32 v88, vcc_lo, v70, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v88, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v104, v72, v88, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v104, v90, v104, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v103, v104, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v104, v73, v88, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v104, v90, v104, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
v_add_co_u32 v71, vcc_lo, v71, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 8                 // scale stride
v_add_nc_i32 v72, v72, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 8                 // scale stride
v_add_nc_i32 v73, v73, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v70, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v106, v72, v70, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v106, v90, v106, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v105, v106, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v106, v73, v70, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v106, v90, v106, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
v_add_co_u32 v88, vcc_lo, v70, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v88, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v108, v72, v88, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v108, v90, v108, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v107, v108, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v108, v73, v88, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v108, v90, v108, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
v_add_co_u32 v71, vcc_lo, v71, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 8                 // scale stride
v_add_nc_i32 v72, v72, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 8                 // scale stride
v_add_nc_i32 v73, v73, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v70, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v110, v72, v70, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v110, v90, v110, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v109, v110, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v110, v73, v70, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v110, v90, v110, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(16,0,1,0) */
v_add_co_u32 v88, vcc_lo, v70, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v88, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v112, v72, v88, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v112, v90, v112, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v111, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v112, v73, v88, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v112, v90, v112, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
v_add_co_u32 v71, vcc_lo, v71, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 8                 // scale stride
v_add_nc_i32 v72, v72, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 8                 // scale stride
v_add_nc_i32 v73, v73, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v70, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v114, v72, v70, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v114, v90, v114, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v113, v114, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v114, v73, v70, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v114, v90, v114, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(20,0,1,0) */
v_add_co_u32 v88, vcc_lo, v70, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v88, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v71, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v116, v72, v88, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v116, v90, v116, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v115, v116, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v116, v73, v88, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v116, v90, v116, s70                 // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (12, 0, 0, 0), (12, 1, 0, 0), (16, 0, 0, 0), (16, 1, 0, 0), (20, 0, 0, 0), (20, 1, 0, 0)] */
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
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_mov_b32 v92, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v93, 0x7fff0000                          // fp32 Nan
v_mov_b32 v94, 0x7fff                              // rounding bias for bfloat16
v_lshlrev_b32 v88, 16, v74                         // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+0], v88, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+0], v[vgprValuC+0]    // check Nan
v_bfe_u32 v91, v[vgprValuC+0], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+0], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+0], v91, v93, s68
v_lshrrev_b32 v76, 16, v[vgprValuC+0]              // convert C to bf16
buffer_store_b16 v76, v75, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v88, 16, v95                         // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+1], v88, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+1], v[vgprValuC+1]    // check Nan
v_bfe_u32 v91, v[vgprValuC+1], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+1], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+1], v91, v93, s68
v_lshrrev_b32 v77, 16, v[vgprValuC+1]              // convert C to bf16
buffer_store_b16 v77, v96, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v88, 16, v97                         // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+2], v88, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+2], v[vgprValuC+2]    // check Nan
v_bfe_u32 v91, v[vgprValuC+2], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+2], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+2], v91, v93, s68
v_lshrrev_b32 v78, 16, v[vgprValuC+2]              // convert C to bf16
buffer_store_b16 v78, v98, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v88, 16, v99                         // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+3], v88, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+3], v[vgprValuC+3]    // check Nan
v_bfe_u32 v91, v[vgprValuC+3], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+3], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+3], v91, v93, s68
v_lshrrev_b32 v79, 16, v[vgprValuC+3]              // convert C to bf16
buffer_store_b16 v79, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v88, 16, v101                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+4], v88, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+4], v[vgprValuC+4]    // check Nan
v_bfe_u32 v91, v[vgprValuC+4], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+4], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+4], v91, v93, s68
v_lshrrev_b32 v80, 16, v[vgprValuC+4]              // convert C to bf16
buffer_store_b16 v80, v102, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v88, 16, v103                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+5], v88, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+5], v[vgprValuC+5]    // check Nan
v_bfe_u32 v91, v[vgprValuC+5], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+5], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+5], v91, v93, s68
v_lshrrev_b32 v81, 16, v[vgprValuC+5]              // convert C to bf16
buffer_store_b16 v81, v104, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v88, 16, v105                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+6], v88, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+6], v[vgprValuC+6]    // check Nan
v_bfe_u32 v91, v[vgprValuC+6], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+6], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+6], v91, v93, s68
v_lshrrev_b32 v82, 16, v[vgprValuC+6]              // convert C to bf16
buffer_store_b16 v82, v106, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v88, 16, v107                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+7], v88, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+7], v[vgprValuC+7]    // check Nan
v_bfe_u32 v91, v[vgprValuC+7], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+7], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+7], v91, v93, s68
v_lshrrev_b32 v83, 16, v[vgprValuC+7]              // convert C to bf16
buffer_store_b16 v83, v108, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v88, 16, v109                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+8], v88, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+8], v[vgprValuC+8]    // check Nan
v_bfe_u32 v91, v[vgprValuC+8], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+8], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+8], v91, v93, s68
v_lshrrev_b32 v84, 16, v[vgprValuC+8]              // convert C to bf16
buffer_store_b16 v84, v110, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v88, 16, v111                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+9], v88, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+9], v[vgprValuC+9]    // check Nan
v_bfe_u32 v91, v[vgprValuC+9], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+9], v91, v94           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+9], v91, v93, s68
v_lshrrev_b32 v85, 16, v[vgprValuC+9]              // convert C to bf16
buffer_store_b16 v85, v112, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v88, 16, v113                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+10], v88, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+10], v[vgprValuC+10]  // check Nan
v_bfe_u32 v91, v[vgprValuC+10], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+10], v91, v94          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+10], v91, v93, s68
v_lshrrev_b32 v86, 16, v[vgprValuC+10]             // convert C to bf16
buffer_store_b16 v86, v114, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v88, 16, v115                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+11], v88, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+11], v[vgprValuC+11]  // check Nan
v_bfe_u32 v91, v[vgprValuC+11], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v91, v[vgprValuC+11], v91, v94          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+11], v91, v93, s68
v_lshrrev_b32 v87, 16, v[vgprValuC+11]             // convert C to bf16
buffer_store_b16 v87, v116, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_SK_Partials_1:
label_GW_Partials_E0:
s_mov_b64 s[sgprSrdWS:sgprSrdWS+1], s[sgprAddressWS:sgprAddressWS+1]
s_mov_b32 s[sgprSrdWS+2], BufferOOB
s_mov_b32 s[sgprSrdWS+3], Srd127_96

s_mul_i32 s8, 0x1800, s[sgprPersistentWorkGroupIndex] // Offset to correct partials tile (low word)
s_mul_hi_u32 s9, 0x1800, s[sgprPersistentWorkGroupIndex] // partials tile offset (high word) for 64-bit SRD
s_add_u32 s[sgprSrdWS+0], s[sgprSrdWS+0], s8       // add lo to SRD
s_addc_u32 s[sgprSrdWS+1], s[sgprSrdWS+1], s9      // add hi (offset high word + lo carry) to SRD

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=160 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Partials Write Batch #0 (d1,d0,vc1,vc0) = */
/*      (0,0,0,0:vw1); (0,1,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (12,0,0,0:vw1); (12,1,0,0:vw1); (16,0,0,0:vw1); (16,1,0,0:vw1); (20,0,0,0:vw1); (20,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* apply mask, calc new C and issue writes */
v_mov_b32 v92, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v93, 0x7fff0000                          // fp32 Nan
v_mov_b32 v94, 0x7fff                              // rounding bias for bfloat16
v_lshlrev_b32 v74, 2, v[vgprSerial]                // v74 = v[vgprSerial] * 4
s_mov_b32 s8, 0                                    // Init sgpr offset
buffer_store_b32 v76, v74, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v77, v74, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v78, v74, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v79, v74, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v80, v74, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v81, v74, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v82, v74, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v83, v74, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v84, v74, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v85, v74, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v86, v74, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 512                              // Inc sgpr offset
buffer_store_b32 v87, v74, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_waitcnt_vscnt null 0                             // wait for data store
s_barrier                                          // store all data before setting flag
s_lshl_b32 s8, s[sgprPersistentWorkGroupIndex], 2  // flag offset based on physical partial slot
v_readfirstlane_b32 s9, v[vgprSerial]              // Wave 0 updates flags
s_cmp_eq_u32 s9, 0                                 // Check for wave 0
s_cbranch_scc0 label_SK_SkipFlagSet                // Skip flag set
s_mov_b32 s9, 1                                    // flag data
v_mov_b32 v95, s9                                  // move flag value to vgpr
v_mov_b32 v96, 0                                   // zero vaddr offset
s_mov_b64 s[68:69], s[sgprAddressFlags:sgprAddressFlags+1]
s_mov_b32 s70, BufferOOB
s_mov_b32 s71, Srd127_96
buffer_store_b32 v95, v96, s[68:71], s8 offen offset:0 glc dlc // set flag
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
