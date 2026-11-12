
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
.text
.protected RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBG_UserArgs_MT160x64x128_MI16x16x1_SK_gfx1151
.globl RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBG_UserArgs_MT160x64x128_MI16x16x1_SK_gfx1151
.p2align 8
.type RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBG_UserArgs_MT160x64x128_MI16x16x1_SK_gfx1151,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBG_UserArgs_MT160x64x128_MI16x16x1_SK_gfx1151
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 256 // vgprs
  .amdhsa_next_free_sgpr 88 // sgprs
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
/* Num SGPR   =88 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 40 x 1 */
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
  - .name: RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBG_UserArgs_MT160x64x128_MI16x16x1_SK_gfx1151
    .symbol: 'RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBG_UserArgs_MT160x64x128_MI16x16x1_SK_gfx1151.kd'
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
    .group_segment_fixed_size:   60928
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
RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBG_UserArgs_MT160x64x128_MI16x16x1_SK_gfx1151:
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
/* ValuC range: [0-40), serializedStore enabled */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprBase, 78
.set vgprLocalWriteAddrA, 74
.set vgprLocalWriteAddrB, 75
.set vgprGlobalReadOffsetA, 40
.set vgprGlobalReadOffsetB, 50
.set vgprGlobalReadOffsetScaleA, 54
.set vgprG2LScaleA, 64
.set vgprLocalReadAddrA, 76
.set vgprLocalReadAddrB, 77
.set vgprSerial, 232

/******************************************/
/* VGPR Macro Assignments                 */
/******************************************/
.set vgprValuA_X0_I0_BASE, vgprBase+0
.set vgprValuB_X0_I0_BASE, vgprBase+81
.set vgprG2LA_BASE, vgprBase+98
.set vgprG2LB_BASE, vgprBase+138
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+40
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X1_I0, vgprValuB_X0_I0_BASE+8
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

.set MT0, 160
.set MT1, 64
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
s_setprio 3                                        // optimization store
s_mov_b32 m0, 0xee00                               // LDS clamp at 60928 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id
v_mov_b32 v[vgprG2LScaleA+0], 0                    // scaleA: low half of the f32 scale
v_mov_b32 v[vgprG2LScaleA+1], 0                    // scaleA: low half of the f32 scale
v_mov_b32 v[vgprG2LScaleA+2], 0                    // scaleA: low half of the f32 scale
v_mov_b32 v[vgprG2LScaleA+3], 0                    // scaleA: low half of the f32 scale
v_mov_b32 v[vgprG2LScaleA+4], 0                    // scaleA: low half of the f32 scale
v_mov_b32 v[vgprG2LScaleA+5], 0                    // scaleA: low half of the f32 scale
v_mov_b32 v[vgprG2LScaleA+6], 0                    // scaleA: low half of the f32 scale
v_mov_b32 v[vgprG2LScaleA+7], 0                    // scaleA: low half of the f32 scale
v_mov_b32 v[vgprG2LScaleA+8], 0                    // scaleA: low half of the f32 scale
v_mov_b32 v[vgprG2LScaleA+9], 0                    // scaleA: low half of the f32 scale
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
/* init: add vgpr [78...253) to pool */
/* init: add vgpr [0...40) to pool */
/* init: add agpr [0...0) to pool */
s_waitcnt lgkmcnt(0)                               // wait for 128/0 bytes of kern args over preload
s_mul_hi_u32 s77, s[sgprSizesFree+0], 53687092     // STATIC_DIV: divisor=160
s_mul_i32 s76, s[sgprSizesFree+0], 53687092        // tmp = dividend * magic
s_lshr_b64 s[76:77], s[76:77], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s[sgprNumWorkGroups0], s76               // quotient
s_mul_i32 s76, s[sgprNumWorkGroups0], 160          // quotient*divisor
s_sub_u32 s45, s[sgprSizesFree+0], s76             // rReg = dividend - quotient*divisor
s_cmp_gt_u32 s45, 0                                // Check if remainder > 0
s_addc_u32 s[sgprNumWorkGroups0], s[sgprNumWorkGroups0], 0 // Increment if remainder > 0
s_lshr_b32 s[sgprNumWorkGroups1], s[sgprSizesFree+1], 6 // s[sgprNumWorkGroups1] = s[sgprSizesFree+1] / 64
s_and_b32 s45, 63, s[sgprSizesFree+1]              // s45 = s[sgprSizesFree+1] % 64
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
s_mul_hi_u32 s79, s12, 53687092                    // STATIC_DIV: divisor=160
s_mul_i32 s78, s12, 53687092                       // tmp = dividend * magic
s_lshr_b64 s[78:79], s[78:79], 33                  // tmp0 = quotient
s_mul_i32 s79, s78, 160                            // tmp1 = quotient * divisor
s_cmp_lg_u32 s79, s12                              // if (quotient * divisor != dividend), result+=1
s_addc_u32 s80, s78, 0                             // if (quotient * divisor != dividend), result+=1
s_lshr_b32 s81, s13, 6                             // s81 = s13 / 64
s_and_b32 s78, 63, s13                             // s78 = s13 % 64
s_addc_u32 s81, s81, 0
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
s_mul_hi_u32 s79, s12, 53687092                    // STATIC_DIV: divisor=160
s_mul_i32 s78, s12, 53687092                       // tmp = dividend * magic
s_lshr_b64 s[78:79], s[78:79], 33                  // tmp0 = quotient
s_mul_i32 s79, s78, 160                            // tmp1 = quotient * divisor
s_cmp_lg_u32 s79, s12                              // if (quotient * divisor != dividend), result+=1
s_addc_u32 s80, s78, 0                             // if (quotient * divisor != dividend), result+=1
s_lshr_b32 s81, s13, 6                             // s81 = s13 / 64
s_and_b32 s78, 63, s13                             // s78 = s13 % 64
s_addc_u32 s81, s81, 0
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
/* init: add vgpr [78...253) to pool */
/* init: add vgpr [0...40) to pool */
/* init: add agpr [0...0) to pool */
s_waitcnt lgkmcnt(0)                               // wait for 128/0 bytes of kern args over preload
s_mul_hi_u32 s77, s[sgprSizesFree+0], 53687092     // STATIC_DIV: divisor=160
s_mul_i32 s76, s[sgprSizesFree+0], 53687092        // tmp = dividend * magic
s_lshr_b64 s[76:77], s[76:77], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s[sgprNumWorkGroups0], s76               // quotient
s_mul_i32 s76, s[sgprNumWorkGroups0], 160          // quotient*divisor
s_sub_u32 s45, s[sgprSizesFree+0], s76             // rReg = dividend - quotient*divisor
s_cmp_gt_u32 s45, 0                                // Check if remainder > 0
s_addc_u32 s[sgprNumWorkGroups0], s[sgprNumWorkGroups0], 0 // Increment if remainder > 0
s_lshr_b32 s[sgprNumWorkGroups1], s[sgprSizesFree+1], 6 // s[sgprNumWorkGroups1] = s[sgprSizesFree+1] / 64
s_and_b32 s45, 63, s[sgprSizesFree+1]              // s45 = s[sgprSizesFree+1] % 64
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
v_and_b32 v3, 3, v3                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(4)
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
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, 0xaa00, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

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
v_lshrrev_b32 v6, 8, v[vgprLocalWriteAddrA]        // padding 16 per block 256
v_lshl_add_u32 v[vgprLocalWriteAddrA], v6, 4, v[vgprLocalWriteAddrA] // padding 16 per block 256

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x80, v2     // lwBL**(DepthU_Compute + PAD)
v_add_nc_u32 v[vgprLocalWriteAddrB], v5, v[vgprLocalWriteAddrB] // lwFOB = (lwBB + lwBL*(DepthU+PAD))
v_lshlrev_b32 v[vgprLocalWriteAddrB], 1, v[vgprLocalWriteAddrB] //  (multiple bpe)
v_lshrrev_b32 v6, 8, v[vgprLocalWriteAddrB]        // padding 16 per block 256
v_lshl_add_u32 v[vgprLocalWriteAddrB], v6, 4, v[vgprLocalWriteAddrB] // padding 16 per block 256
v_add_co_u32 v[vgprLocalWriteAddrB], vcc_lo, 0xaa00, v[vgprLocalWriteAddrB] // lwFOB = lw1J + lwL*MT1J + LDS_OFFSET_B=43520

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
v_add_co_u32 v10, vcc_lo, 16, v9                   // groA0I_4 += LSPA
v_add_co_u32 v11, vcc_lo, 16, v10                  // groA0I_5 += LSPA
v_add_co_u32 v12, vcc_lo, 16, v11                  // groA0I_6 += LSPA
v_add_co_u32 v13, vcc_lo, 16, v12                  // groA0I_7 += LSPA
v_add_co_u32 v14, vcc_lo, 16, v13                  // groA0I_8 += LSPA
v_add_co_u32 v15, vcc_lo, 16, v14                  // groA0I_9 += LSPA

/* global read addresses: tile offsets b */
v_mov_b32 v16, v2                                  // groB1J_0
v_add_co_u32 v17, vcc_lo, 16, v16                  // groB1J_1 += LSPB
v_add_co_u32 v18, vcc_lo, 16, v17                  // groB1J_2 += LSPB
v_add_co_u32 v19, vcc_lo, 16, v18                  // groB1J_3 += LSPB

/* global read addresses: unroll offsets a */
v_mov_b32 v20, v1                                  // groAL_0

/* global read addresses: unroll offsets b */
v_mov_b32 v21, v3                                  // groBL_0

/* global read addresses: addresses a */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s47, s[sgprWorkGroup0], 160           // WorkGroup[01] * MT
s_mul_i32 s46, s[sgprWorkGroup0], 160              // WorkGroup[01] * MT
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
s_mul_i32 s44, s[sgprWorkGroup0], 160              // scaleA: workgroup row origin
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
s_mul_i32 s44, s[sgprStreamKLocalStart], 128       // scaleA: first K of this range
s_lshr_b32 s44, s44, s[sgprScaleAGShift]           // scaleA: groups before this range
s_mul_i32 s45, s44, 2                              // scaleA: skipped bytes
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], s45 // scaleA: skip groups before the split (lower)
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA: skip groups before the split (upper)
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], s45 // scaleA: limit shrinks by the skip

/* global read addresses: addresses b */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s47, s[sgprWorkGroup1], 64            // WorkGroup[01] * MT
s_mul_i32 s46, s[sgprWorkGroup1], 64               // WorkGroup[01] * MT
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
v_mul_lo_u32 v22, s[sgprStrideA0I], v[6]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+0+0], vcc_lo, v[20], v[22+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+0+0], 0x8, v[vgprGlobalReadOffsetA+0+0] // add prepad for pointer shift
v_lshrrev_b32 v22, s[sgprScaleAGShift], v20        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+0], s[sgprStrideScaleA], v6 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+0], v22, v[vgprGlobalReadOffsetScaleA+0] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+0], 1, v[vgprGlobalReadOffsetScaleA+0] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetA+0], 1, v[vgprGlobalReadOffsetA+0] //  (multiple bpe)
v_mul_lo_u32 v22, s[sgprStrideA0I], v[7]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+1+0], vcc_lo, v[20], v[22+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+1+0], 0x8, v[vgprGlobalReadOffsetA+1+0] // add prepad for pointer shift
v_lshrrev_b32 v22, s[sgprScaleAGShift], v20        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+1], s[sgprStrideScaleA], v7 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+1], v22, v[vgprGlobalReadOffsetScaleA+1] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+1], 1, v[vgprGlobalReadOffsetScaleA+1] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetA+1], 1, v[vgprGlobalReadOffsetA+1] //  (multiple bpe)
v_mul_lo_u32 v22, s[sgprStrideA0I], v[8]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+2+0], vcc_lo, v[20], v[22+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+2+0], 0x8, v[vgprGlobalReadOffsetA+2+0] // add prepad for pointer shift
v_lshrrev_b32 v22, s[sgprScaleAGShift], v20        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+2], s[sgprStrideScaleA], v8 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+2], v22, v[vgprGlobalReadOffsetScaleA+2] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+2], 1, v[vgprGlobalReadOffsetScaleA+2] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetA+2], 1, v[vgprGlobalReadOffsetA+2] //  (multiple bpe)
v_mul_lo_u32 v22, s[sgprStrideA0I], v[9]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+3+0], vcc_lo, v[20], v[22+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+3+0], 0x8, v[vgprGlobalReadOffsetA+3+0] // add prepad for pointer shift
v_lshrrev_b32 v22, s[sgprScaleAGShift], v20        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+3], s[sgprStrideScaleA], v9 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+3], v22, v[vgprGlobalReadOffsetScaleA+3] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+3], 1, v[vgprGlobalReadOffsetScaleA+3] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetA+3], 1, v[vgprGlobalReadOffsetA+3] //  (multiple bpe)
v_mul_lo_u32 v22, s[sgprStrideA0I], v[10]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+4+0], vcc_lo, v[20], v[22+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+4+0], 0x8, v[vgprGlobalReadOffsetA+4+0] // add prepad for pointer shift
v_lshrrev_b32 v22, s[sgprScaleAGShift], v20        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+4], s[sgprStrideScaleA], v10 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+4], v22, v[vgprGlobalReadOffsetScaleA+4] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+4], 1, v[vgprGlobalReadOffsetScaleA+4] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetA+4], 1, v[vgprGlobalReadOffsetA+4] //  (multiple bpe)
v_mul_lo_u32 v22, s[sgprStrideA0I], v[11]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+5+0], vcc_lo, v[20], v[22+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+5+0], 0x8, v[vgprGlobalReadOffsetA+5+0] // add prepad for pointer shift
v_lshrrev_b32 v22, s[sgprScaleAGShift], v20        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+5], s[sgprStrideScaleA], v11 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+5], v22, v[vgprGlobalReadOffsetScaleA+5] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+5], 1, v[vgprGlobalReadOffsetScaleA+5] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetA+5], 1, v[vgprGlobalReadOffsetA+5] //  (multiple bpe)
v_mul_lo_u32 v22, s[sgprStrideA0I], v[12]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+6+0], vcc_lo, v[20], v[22+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+6+0], 0x8, v[vgprGlobalReadOffsetA+6+0] // add prepad for pointer shift
v_lshrrev_b32 v22, s[sgprScaleAGShift], v20        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+6], s[sgprStrideScaleA], v12 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+6], v22, v[vgprGlobalReadOffsetScaleA+6] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+6], 1, v[vgprGlobalReadOffsetScaleA+6] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetA+6], 1, v[vgprGlobalReadOffsetA+6] //  (multiple bpe)
v_mul_lo_u32 v22, s[sgprStrideA0I], v[13]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+7+0], vcc_lo, v[20], v[22+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+7+0], 0x8, v[vgprGlobalReadOffsetA+7+0] // add prepad for pointer shift
v_lshrrev_b32 v22, s[sgprScaleAGShift], v20        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+7], s[sgprStrideScaleA], v13 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+7], v22, v[vgprGlobalReadOffsetScaleA+7] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+7], 1, v[vgprGlobalReadOffsetScaleA+7] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetA+7], 1, v[vgprGlobalReadOffsetA+7] //  (multiple bpe)
v_mul_lo_u32 v22, s[sgprStrideA0I], v[14]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+8+0], vcc_lo, v[20], v[22+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+8+0], 0x8, v[vgprGlobalReadOffsetA+8+0] // add prepad for pointer shift
v_lshrrev_b32 v22, s[sgprScaleAGShift], v20        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+8], s[sgprStrideScaleA], v14 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+8], v22, v[vgprGlobalReadOffsetScaleA+8] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+8], 1, v[vgprGlobalReadOffsetScaleA+8] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetA+8], 1, v[vgprGlobalReadOffsetA+8] //  (multiple bpe)
v_mul_lo_u32 v22, s[sgprStrideA0I], v[15]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+9+0], vcc_lo, v[20], v[22+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+9+0], 0x8, v[vgprGlobalReadOffsetA+9+0] // add prepad for pointer shift
v_lshrrev_b32 v22, s[sgprScaleAGShift], v20        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+9], s[sgprStrideScaleA], v15 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+9], v22, v[vgprGlobalReadOffsetScaleA+9] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+9], 1, v[vgprGlobalReadOffsetScaleA+9] // scaleA: elements -> bytes
v_lshrrev_b32 v[vgprGlobalReadOffsetA+9], 1, v[vgprGlobalReadOffsetA+9] //  (multiple bpe)
/* ============================================================= */

/* global read addresses: final offsets b */
/* ============================================================= */
v_mul_lo_u32 v6, s[sgprStrideB1J], v[16]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+0+0], vcc_lo, v[21], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+0+0], 0x8, v[vgprGlobalReadOffsetB+0+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+0], 1, v[vgprGlobalReadOffsetB+0] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[17]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+1+0], vcc_lo, v[21], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+1+0], 0x8, v[vgprGlobalReadOffsetB+1+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+1], 1, v[vgprGlobalReadOffsetB+1] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[18]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+2+0], vcc_lo, v[21], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+2+0], 0x8, v[vgprGlobalReadOffsetB+2+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+2], 1, v[vgprGlobalReadOffsetB+2] //  (multiple bpe)
v_mul_lo_u32 v6, s[sgprStrideB1J], v[19]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+3+0], vcc_lo, v[21], v[6+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetB+3+0], 0x8, v[vgprGlobalReadOffsetB+3+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+3], 1, v[vgprGlobalReadOffsetB+3] //  (multiple bpe)
/* ============================================================= */

/* global read addresses: increments a */
.set GlobalReadIncsA, 64

/* global read addresses: increments b */
.set GlobalReadIncsB, 256
/* declare loop num iterations */

/* initC: remove ValuC vgpr buffer [0...40) from pool */

/* initC: remove acc vgpr buffer [0...0) from pool */

/* initC: remove ValuA/B vgpr buffer [78...175) from pool */
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
buffer_load_b32 v[vgprG2LA+14], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_3_0
buffer_load_b32 v[vgprG2LA+18], v[vgprGlobalReadOffsetA+4], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_4_0
buffer_load_b32 v[vgprG2LA+22], v[vgprGlobalReadOffsetA+5], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_5_0
buffer_load_b32 v[vgprG2LA+26], v[vgprGlobalReadOffsetA+6], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_6_0
buffer_load_b32 v[vgprG2LA+30], v[vgprGlobalReadOffsetA+7], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_7_0
buffer_load_b32 v[vgprG2LA+34], v[vgprGlobalReadOffsetA+8], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_8_0
buffer_load_b32 v[vgprG2LA+38], v[vgprGlobalReadOffsetA+9], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_9_0

/* global read block-scale A */
buffer_load_d16_hi_b16 v[vgprG2LScaleA+0], v[vgprGlobalReadOffsetScaleA+0], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 0 as f32
buffer_load_d16_hi_b16 v[vgprG2LScaleA+1], v[vgprGlobalReadOffsetScaleA+1], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 1 as f32
buffer_load_d16_hi_b16 v[vgprG2LScaleA+2], v[vgprGlobalReadOffsetScaleA+2], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 2 as f32
buffer_load_d16_hi_b16 v[vgprG2LScaleA+3], v[vgprGlobalReadOffsetScaleA+3], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 3 as f32
buffer_load_d16_hi_b16 v[vgprG2LScaleA+4], v[vgprGlobalReadOffsetScaleA+4], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 4 as f32
buffer_load_d16_hi_b16 v[vgprG2LScaleA+5], v[vgprGlobalReadOffsetScaleA+5], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 5 as f32
buffer_load_d16_hi_b16 v[vgprG2LScaleA+6], v[vgprGlobalReadOffsetScaleA+6], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 6 as f32
buffer_load_d16_hi_b16 v[vgprG2LScaleA+7], v[vgprGlobalReadOffsetScaleA+7], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 7 as f32
buffer_load_d16_hi_b16 v[vgprG2LScaleA+8], v[vgprGlobalReadOffsetScaleA+8], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 8 as f32
buffer_load_d16_hi_b16 v[vgprG2LScaleA+9], v[vgprGlobalReadOffsetScaleA+9], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 9 as f32
buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b128 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b128 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b128 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0

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
v_mov_b32 v234, v[vgprG2LScaleA+0]                 // scaleA: copy on the other VGPR bank
v_xor_b32 v235, 0x88888888, v[vgprG2LA+2+0]        // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v236, v235, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v237, v235, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v239, v235, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v238, v235, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v236, s[sgprScaleALiftMagic+6], v236 :: v_dual_subrev_f32 v237, s[sgprScaleALiftMagic+7], v237 // w4a16: n - c, exact
v_dual_subrev_f32 v239, s[sgprScaleALiftMagic+8], v239 :: v_dual_subrev_f32 v238, s[sgprScaleALiftMagic+9], v238 // w4a16: n - c, exact
v_dual_mul_f32 v236, v64, v236 :: v_dual_mul_f32 v237, v234, v237 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v239, v64, v239 :: v_dual_add_nc_u32 v236, s[sgprScaleALiftMagic+5], v236 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v238, v234, v238 :: v_dual_add_nc_u32 v237, s[sgprScaleALiftMagic+5], v237 // w4a16: (n - c) * s :: round
v_and_or_b32 v240, v235, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v235, 20, v235                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v241, v235, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v243, v235, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v242, v235, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v240, s[sgprScaleALiftMagic+10], v240 :: v_dual_add_nc_u32 v239, s[sgprScaleALiftMagic+5], v239 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v241, s[sgprScaleALiftMagic+6], v241 :: v_dual_add_nc_u32 v238, s[sgprScaleALiftMagic+5], v238 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v243, s[sgprScaleALiftMagic+7], v243 :: v_dual_subrev_f32 v242, s[sgprScaleALiftMagic+8], v242 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+0+0], v237, v236, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+1], v238, v239, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v240, v64, v240 :: v_dual_mul_f32 v241, v234, v241 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v243, v64, v243 :: v_dual_add_nc_u32 v240, s[sgprScaleALiftMagic+5], v240 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v242, v234, v242 :: v_dual_add_nc_u32 v241, s[sgprScaleALiftMagic+5], v241 // w4a16: (n - c) * s :: round
v_add_nc_u32 v243, s[sgprScaleALiftMagic+5], v243  // w4a16: round
v_add_nc_u32 v242, s[sgprScaleALiftMagic+5], v242  // w4a16: round
v_perm_b32 v[vgprG2LA+0+2], v241, v240, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+3], v242, v243, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
v_mov_b32 v234, v[vgprG2LScaleA+1]                 // scaleA: copy on the other VGPR bank
v_xor_b32 v235, 0x88888888, v[vgprG2LA+6+0]        // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v236, v235, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v237, v235, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v239, v235, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v238, v235, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v236, s[sgprScaleALiftMagic+6], v236 :: v_dual_subrev_f32 v237, s[sgprScaleALiftMagic+7], v237 // w4a16: n - c, exact
v_dual_subrev_f32 v239, s[sgprScaleALiftMagic+8], v239 :: v_dual_subrev_f32 v238, s[sgprScaleALiftMagic+9], v238 // w4a16: n - c, exact
v_dual_mul_f32 v236, v65, v236 :: v_dual_mul_f32 v237, v234, v237 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v239, v65, v239 :: v_dual_add_nc_u32 v236, s[sgprScaleALiftMagic+5], v236 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v238, v234, v238 :: v_dual_add_nc_u32 v237, s[sgprScaleALiftMagic+5], v237 // w4a16: (n - c) * s :: round
v_and_or_b32 v240, v235, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v235, 20, v235                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v241, v235, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v243, v235, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v242, v235, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v240, s[sgprScaleALiftMagic+10], v240 :: v_dual_add_nc_u32 v239, s[sgprScaleALiftMagic+5], v239 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v241, s[sgprScaleALiftMagic+6], v241 :: v_dual_add_nc_u32 v238, s[sgprScaleALiftMagic+5], v238 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v243, s[sgprScaleALiftMagic+7], v243 :: v_dual_subrev_f32 v242, s[sgprScaleALiftMagic+8], v242 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+4+0], v237, v236, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+4+1], v238, v239, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v240, v65, v240 :: v_dual_mul_f32 v241, v234, v241 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v243, v65, v243 :: v_dual_add_nc_u32 v240, s[sgprScaleALiftMagic+5], v240 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v242, v234, v242 :: v_dual_add_nc_u32 v241, s[sgprScaleALiftMagic+5], v241 // w4a16: (n - c) * s :: round
v_add_nc_u32 v243, s[sgprScaleALiftMagic+5], v243  // w4a16: round
v_add_nc_u32 v242, s[sgprScaleALiftMagic+5], v242  // w4a16: round
v_perm_b32 v[vgprG2LA+4+2], v241, v240, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+4+3], v242, v243, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:4352 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 4352 sync LDS0
v_mov_b32 v235, v[vgprG2LScaleA+2]                 // scaleA: copy on the other VGPR bank
v_xor_b32 v234, 0x88888888, v[vgprG2LA+10+0]       // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v236, v234, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v237, v234, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v239, v234, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v238, v234, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v236, s[sgprScaleALiftMagic+6], v236 :: v_dual_subrev_f32 v237, s[sgprScaleALiftMagic+7], v237 // w4a16: n - c, exact
v_dual_subrev_f32 v239, s[sgprScaleALiftMagic+8], v239 :: v_dual_subrev_f32 v238, s[sgprScaleALiftMagic+9], v238 // w4a16: n - c, exact
v_dual_mul_f32 v236, v66, v236 :: v_dual_mul_f32 v237, v235, v237 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v239, v66, v239 :: v_dual_add_nc_u32 v236, s[sgprScaleALiftMagic+5], v236 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v238, v235, v238 :: v_dual_add_nc_u32 v237, s[sgprScaleALiftMagic+5], v237 // w4a16: (n - c) * s :: round
v_and_or_b32 v240, v234, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v234, 20, v234                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v241, v234, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v243, v234, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v242, v234, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v240, s[sgprScaleALiftMagic+10], v240 :: v_dual_add_nc_u32 v239, s[sgprScaleALiftMagic+5], v239 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v241, s[sgprScaleALiftMagic+6], v241 :: v_dual_add_nc_u32 v238, s[sgprScaleALiftMagic+5], v238 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v243, s[sgprScaleALiftMagic+7], v243 :: v_dual_subrev_f32 v242, s[sgprScaleALiftMagic+8], v242 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+8+0], v237, v236, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+1], v238, v239, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v240, v66, v240 :: v_dual_mul_f32 v241, v235, v241 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v243, v66, v243 :: v_dual_add_nc_u32 v240, s[sgprScaleALiftMagic+5], v240 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v242, v235, v242 :: v_dual_add_nc_u32 v241, s[sgprScaleALiftMagic+5], v241 // w4a16: (n - c) * s :: round
v_add_nc_u32 v243, s[sgprScaleALiftMagic+5], v243  // w4a16: round
v_add_nc_u32 v242, s[sgprScaleALiftMagic+5], v242  // w4a16: round
v_perm_b32 v[vgprG2LA+8+2], v241, v240, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+3], v242, v243, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:8704 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 8704 sync LDS0
v_mov_b32 v234, v[vgprG2LScaleA+3]                 // scaleA: copy on the other VGPR bank
v_xor_b32 v235, 0x88888888, v[vgprG2LA+14+0]       // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v236, v235, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v237, v235, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v239, v235, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v238, v235, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v236, s[sgprScaleALiftMagic+6], v236 :: v_dual_subrev_f32 v237, s[sgprScaleALiftMagic+7], v237 // w4a16: n - c, exact
v_dual_subrev_f32 v239, s[sgprScaleALiftMagic+8], v239 :: v_dual_subrev_f32 v238, s[sgprScaleALiftMagic+9], v238 // w4a16: n - c, exact
v_dual_mul_f32 v236, v67, v236 :: v_dual_mul_f32 v237, v234, v237 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v239, v67, v239 :: v_dual_add_nc_u32 v236, s[sgprScaleALiftMagic+5], v236 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v238, v234, v238 :: v_dual_add_nc_u32 v237, s[sgprScaleALiftMagic+5], v237 // w4a16: (n - c) * s :: round
v_and_or_b32 v240, v235, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v235, 20, v235                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v241, v235, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v243, v235, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v242, v235, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v240, s[sgprScaleALiftMagic+10], v240 :: v_dual_add_nc_u32 v239, s[sgprScaleALiftMagic+5], v239 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v241, s[sgprScaleALiftMagic+6], v241 :: v_dual_add_nc_u32 v238, s[sgprScaleALiftMagic+5], v238 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v243, s[sgprScaleALiftMagic+7], v243 :: v_dual_subrev_f32 v242, s[sgprScaleALiftMagic+8], v242 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+12+0], v237, v236, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+12+1], v238, v239, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v240, v67, v240 :: v_dual_mul_f32 v241, v234, v241 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v243, v67, v243 :: v_dual_add_nc_u32 v240, s[sgprScaleALiftMagic+5], v240 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v242, v234, v242 :: v_dual_add_nc_u32 v241, s[sgprScaleALiftMagic+5], v241 // w4a16: (n - c) * s :: round
v_add_nc_u32 v243, s[sgprScaleALiftMagic+5], v243  // w4a16: round
v_add_nc_u32 v242, s[sgprScaleALiftMagic+5], v242  // w4a16: round
v_perm_b32 v[vgprG2LA+12+2], v241, v240, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+12+3], v242, v243, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+12:vgprG2LA+12+3] offset:13056 // lwoA_0_0_3_0 = (0*LSCA)*(MT0I+PAD) + (3*LSPA) = 13056 sync LDS0
v_mov_b32 v234, v[vgprG2LScaleA+4]                 // scaleA: copy on the other VGPR bank
v_xor_b32 v235, 0x88888888, v[vgprG2LA+18+0]       // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v236, v235, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v237, v235, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v239, v235, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v238, v235, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v236, s[sgprScaleALiftMagic+6], v236 :: v_dual_subrev_f32 v237, s[sgprScaleALiftMagic+7], v237 // w4a16: n - c, exact
v_dual_subrev_f32 v239, s[sgprScaleALiftMagic+8], v239 :: v_dual_subrev_f32 v238, s[sgprScaleALiftMagic+9], v238 // w4a16: n - c, exact
v_dual_mul_f32 v236, v68, v236 :: v_dual_mul_f32 v237, v234, v237 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v239, v68, v239 :: v_dual_add_nc_u32 v236, s[sgprScaleALiftMagic+5], v236 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v238, v234, v238 :: v_dual_add_nc_u32 v237, s[sgprScaleALiftMagic+5], v237 // w4a16: (n - c) * s :: round
v_and_or_b32 v240, v235, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v235, 20, v235                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v241, v235, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v243, v235, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v242, v235, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v240, s[sgprScaleALiftMagic+10], v240 :: v_dual_add_nc_u32 v239, s[sgprScaleALiftMagic+5], v239 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v241, s[sgprScaleALiftMagic+6], v241 :: v_dual_add_nc_u32 v238, s[sgprScaleALiftMagic+5], v238 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v243, s[sgprScaleALiftMagic+7], v243 :: v_dual_subrev_f32 v242, s[sgprScaleALiftMagic+8], v242 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+16+0], v237, v236, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+16+1], v238, v239, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v240, v68, v240 :: v_dual_mul_f32 v241, v234, v241 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v243, v68, v243 :: v_dual_add_nc_u32 v240, s[sgprScaleALiftMagic+5], v240 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v242, v234, v242 :: v_dual_add_nc_u32 v241, s[sgprScaleALiftMagic+5], v241 // w4a16: (n - c) * s :: round
v_add_nc_u32 v243, s[sgprScaleALiftMagic+5], v243  // w4a16: round
v_add_nc_u32 v242, s[sgprScaleALiftMagic+5], v242  // w4a16: round
v_perm_b32 v[vgprG2LA+16+2], v241, v240, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+16+3], v242, v243, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+16:vgprG2LA+16+3] offset:17408 // lwoA_0_0_4_0 = (0*LSCA)*(MT0I+PAD) + (4*LSPA) = 17408 sync LDS0
v_mov_b32 v234, v[vgprG2LScaleA+5]                 // scaleA: copy on the other VGPR bank
v_xor_b32 v235, 0x88888888, v[vgprG2LA+22+0]       // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v236, v235, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v237, v235, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v239, v235, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v238, v235, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v236, s[sgprScaleALiftMagic+6], v236 :: v_dual_subrev_f32 v237, s[sgprScaleALiftMagic+7], v237 // w4a16: n - c, exact
v_dual_subrev_f32 v239, s[sgprScaleALiftMagic+8], v239 :: v_dual_subrev_f32 v238, s[sgprScaleALiftMagic+9], v238 // w4a16: n - c, exact
v_dual_mul_f32 v236, v69, v236 :: v_dual_mul_f32 v237, v234, v237 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v239, v69, v239 :: v_dual_add_nc_u32 v236, s[sgprScaleALiftMagic+5], v236 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v238, v234, v238 :: v_dual_add_nc_u32 v237, s[sgprScaleALiftMagic+5], v237 // w4a16: (n - c) * s :: round
v_and_or_b32 v240, v235, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v235, 20, v235                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v241, v235, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v243, v235, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v242, v235, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v240, s[sgprScaleALiftMagic+10], v240 :: v_dual_add_nc_u32 v239, s[sgprScaleALiftMagic+5], v239 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v241, s[sgprScaleALiftMagic+6], v241 :: v_dual_add_nc_u32 v238, s[sgprScaleALiftMagic+5], v238 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v243, s[sgprScaleALiftMagic+7], v243 :: v_dual_subrev_f32 v242, s[sgprScaleALiftMagic+8], v242 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+20+0], v237, v236, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+20+1], v238, v239, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v240, v69, v240 :: v_dual_mul_f32 v241, v234, v241 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v243, v69, v243 :: v_dual_add_nc_u32 v240, s[sgprScaleALiftMagic+5], v240 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v242, v234, v242 :: v_dual_add_nc_u32 v241, s[sgprScaleALiftMagic+5], v241 // w4a16: (n - c) * s :: round
v_add_nc_u32 v243, s[sgprScaleALiftMagic+5], v243  // w4a16: round
v_add_nc_u32 v242, s[sgprScaleALiftMagic+5], v242  // w4a16: round
v_perm_b32 v[vgprG2LA+20+2], v241, v240, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+20+3], v242, v243, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+20:vgprG2LA+20+3] offset:21760 // lwoA_0_0_5_0 = (0*LSCA)*(MT0I+PAD) + (5*LSPA) = 21760 sync LDS0
v_mov_b32 v235, v[vgprG2LScaleA+6]                 // scaleA: copy on the other VGPR bank
v_xor_b32 v234, 0x88888888, v[vgprG2LA+26+0]       // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v236, v234, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v237, v234, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v239, v234, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v238, v234, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v236, s[sgprScaleALiftMagic+6], v236 :: v_dual_subrev_f32 v237, s[sgprScaleALiftMagic+7], v237 // w4a16: n - c, exact
v_dual_subrev_f32 v239, s[sgprScaleALiftMagic+8], v239 :: v_dual_subrev_f32 v238, s[sgprScaleALiftMagic+9], v238 // w4a16: n - c, exact
v_dual_mul_f32 v236, v70, v236 :: v_dual_mul_f32 v237, v235, v237 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v239, v70, v239 :: v_dual_add_nc_u32 v236, s[sgprScaleALiftMagic+5], v236 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v238, v235, v238 :: v_dual_add_nc_u32 v237, s[sgprScaleALiftMagic+5], v237 // w4a16: (n - c) * s :: round
v_and_or_b32 v240, v234, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v234, 20, v234                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v241, v234, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v243, v234, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v242, v234, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v240, s[sgprScaleALiftMagic+10], v240 :: v_dual_add_nc_u32 v239, s[sgprScaleALiftMagic+5], v239 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v241, s[sgprScaleALiftMagic+6], v241 :: v_dual_add_nc_u32 v238, s[sgprScaleALiftMagic+5], v238 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v243, s[sgprScaleALiftMagic+7], v243 :: v_dual_subrev_f32 v242, s[sgprScaleALiftMagic+8], v242 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+24+0], v237, v236, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+24+1], v238, v239, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v240, v70, v240 :: v_dual_mul_f32 v241, v235, v241 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v243, v70, v243 :: v_dual_add_nc_u32 v240, s[sgprScaleALiftMagic+5], v240 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v242, v235, v242 :: v_dual_add_nc_u32 v241, s[sgprScaleALiftMagic+5], v241 // w4a16: (n - c) * s :: round
v_add_nc_u32 v243, s[sgprScaleALiftMagic+5], v243  // w4a16: round
v_add_nc_u32 v242, s[sgprScaleALiftMagic+5], v242  // w4a16: round
v_perm_b32 v[vgprG2LA+24+2], v241, v240, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+24+3], v242, v243, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+24:vgprG2LA+24+3] offset:26112 // lwoA_0_0_6_0 = (0*LSCA)*(MT0I+PAD) + (6*LSPA) = 26112 sync LDS0
v_mov_b32 v234, v[vgprG2LScaleA+7]                 // scaleA: copy on the other VGPR bank
v_xor_b32 v235, 0x88888888, v[vgprG2LA+30+0]       // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v236, v235, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v237, v235, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v239, v235, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v238, v235, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v236, s[sgprScaleALiftMagic+6], v236 :: v_dual_subrev_f32 v237, s[sgprScaleALiftMagic+7], v237 // w4a16: n - c, exact
v_dual_subrev_f32 v239, s[sgprScaleALiftMagic+8], v239 :: v_dual_subrev_f32 v238, s[sgprScaleALiftMagic+9], v238 // w4a16: n - c, exact
v_dual_mul_f32 v236, v71, v236 :: v_dual_mul_f32 v237, v234, v237 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v239, v71, v239 :: v_dual_add_nc_u32 v236, s[sgprScaleALiftMagic+5], v236 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v238, v234, v238 :: v_dual_add_nc_u32 v237, s[sgprScaleALiftMagic+5], v237 // w4a16: (n - c) * s :: round
v_and_or_b32 v240, v235, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v235, 20, v235                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v241, v235, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v243, v235, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v242, v235, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v240, s[sgprScaleALiftMagic+10], v240 :: v_dual_add_nc_u32 v239, s[sgprScaleALiftMagic+5], v239 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v241, s[sgprScaleALiftMagic+6], v241 :: v_dual_add_nc_u32 v238, s[sgprScaleALiftMagic+5], v238 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v243, s[sgprScaleALiftMagic+7], v243 :: v_dual_subrev_f32 v242, s[sgprScaleALiftMagic+8], v242 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+28+0], v237, v236, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+28+1], v238, v239, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v240, v71, v240 :: v_dual_mul_f32 v241, v234, v241 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v243, v71, v243 :: v_dual_add_nc_u32 v240, s[sgprScaleALiftMagic+5], v240 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v242, v234, v242 :: v_dual_add_nc_u32 v241, s[sgprScaleALiftMagic+5], v241 // w4a16: (n - c) * s :: round
v_add_nc_u32 v243, s[sgprScaleALiftMagic+5], v243  // w4a16: round
v_add_nc_u32 v242, s[sgprScaleALiftMagic+5], v242  // w4a16: round
v_perm_b32 v[vgprG2LA+28+2], v241, v240, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+28+3], v242, v243, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+28:vgprG2LA+28+3] offset:30464 // lwoA_0_0_7_0 = (0*LSCA)*(MT0I+PAD) + (7*LSPA) = 30464 sync LDS0
v_mov_b32 v234, v[vgprG2LScaleA+8]                 // scaleA: copy on the other VGPR bank
v_xor_b32 v235, 0x88888888, v[vgprG2LA+34+0]       // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v236, v235, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v237, v235, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v239, v235, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v238, v235, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v236, s[sgprScaleALiftMagic+6], v236 :: v_dual_subrev_f32 v237, s[sgprScaleALiftMagic+7], v237 // w4a16: n - c, exact
v_dual_subrev_f32 v239, s[sgprScaleALiftMagic+8], v239 :: v_dual_subrev_f32 v238, s[sgprScaleALiftMagic+9], v238 // w4a16: n - c, exact
v_dual_mul_f32 v236, v72, v236 :: v_dual_mul_f32 v237, v234, v237 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v239, v72, v239 :: v_dual_add_nc_u32 v236, s[sgprScaleALiftMagic+5], v236 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v238, v234, v238 :: v_dual_add_nc_u32 v237, s[sgprScaleALiftMagic+5], v237 // w4a16: (n - c) * s :: round
v_and_or_b32 v240, v235, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v235, 20, v235                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v241, v235, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v243, v235, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v242, v235, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v240, s[sgprScaleALiftMagic+10], v240 :: v_dual_add_nc_u32 v239, s[sgprScaleALiftMagic+5], v239 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v241, s[sgprScaleALiftMagic+6], v241 :: v_dual_add_nc_u32 v238, s[sgprScaleALiftMagic+5], v238 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v243, s[sgprScaleALiftMagic+7], v243 :: v_dual_subrev_f32 v242, s[sgprScaleALiftMagic+8], v242 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+32+0], v237, v236, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+32+1], v238, v239, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v240, v72, v240 :: v_dual_mul_f32 v241, v234, v241 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v243, v72, v243 :: v_dual_add_nc_u32 v240, s[sgprScaleALiftMagic+5], v240 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v242, v234, v242 :: v_dual_add_nc_u32 v241, s[sgprScaleALiftMagic+5], v241 // w4a16: (n - c) * s :: round
v_add_nc_u32 v243, s[sgprScaleALiftMagic+5], v243  // w4a16: round
v_add_nc_u32 v242, s[sgprScaleALiftMagic+5], v242  // w4a16: round
v_perm_b32 v[vgprG2LA+32+2], v241, v240, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+32+3], v242, v243, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+32:vgprG2LA+32+3] offset:34816 // lwoA_0_0_8_0 = (0*LSCA)*(MT0I+PAD) + (8*LSPA) = 34816 sync LDS0
v_mov_b32 v234, v[vgprG2LScaleA+9]                 // scaleA: copy on the other VGPR bank
v_xor_b32 v235, 0x88888888, v[vgprG2LA+38+0]       // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v236, v235, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v237, v235, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v239, v235, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v238, v235, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v236, s[sgprScaleALiftMagic+6], v236 :: v_dual_subrev_f32 v237, s[sgprScaleALiftMagic+7], v237 // w4a16: n - c, exact
v_dual_subrev_f32 v239, s[sgprScaleALiftMagic+8], v239 :: v_dual_subrev_f32 v238, s[sgprScaleALiftMagic+9], v238 // w4a16: n - c, exact
v_dual_mul_f32 v236, v73, v236 :: v_dual_mul_f32 v237, v234, v237 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v239, v73, v239 :: v_dual_add_nc_u32 v236, s[sgprScaleALiftMagic+5], v236 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v238, v234, v238 :: v_dual_add_nc_u32 v237, s[sgprScaleALiftMagic+5], v237 // w4a16: (n - c) * s :: round
v_and_or_b32 v240, v235, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v235, 20, v235                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v241, v235, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v243, v235, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v242, v235, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v240, s[sgprScaleALiftMagic+10], v240 :: v_dual_add_nc_u32 v239, s[sgprScaleALiftMagic+5], v239 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v241, s[sgprScaleALiftMagic+6], v241 :: v_dual_add_nc_u32 v238, s[sgprScaleALiftMagic+5], v238 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v243, s[sgprScaleALiftMagic+7], v243 :: v_dual_subrev_f32 v242, s[sgprScaleALiftMagic+8], v242 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+36+0], v237, v236, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+36+1], v238, v239, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v240, v73, v240 :: v_dual_mul_f32 v241, v234, v241 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v243, v73, v243 :: v_dual_add_nc_u32 v240, s[sgprScaleALiftMagic+5], v240 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v242, v234, v242 :: v_dual_add_nc_u32 v241, s[sgprScaleALiftMagic+5], v241 // w4a16: (n - c) * s :: round
v_add_nc_u32 v243, s[sgprScaleALiftMagic+5], v243  // w4a16: round
v_add_nc_u32 v242, s[sgprScaleALiftMagic+5], v242  // w4a16: round
v_perm_b32 v[vgprG2LA+36+2], v241, v240, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+36+3], v242, v243, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+36:vgprG2LA+36+3] offset:39168 // lwoA_0_0_9_0 = (0*LSCA)*(MT0I+PAD) + (9*LSPA) = 39168 sync LDS0

/* local write b */
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4:vgprG2LB+4+3] offset:4352 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 4352 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8:vgprG2LB+8+3] offset:8704 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 8704 sync LDS0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12:vgprG2LB+12+3] offset:13056 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 13056 sync LDS0

/* local write swap a */

/* local write swap b */
s_waitcnt lgkmcnt(0)                               // 0prefetch wait for local write
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier                                          // LW to PLR, sync

/* local read prefetch a */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:8704 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:8720 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA+0] offset:17408 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA+0] offset:17424 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], v[vgprLocalReadAddrA+0] offset:26112 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[vgprLocalReadAddrA+0] offset:26128 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+32+3], v[vgprLocalReadAddrA+0] offset:34816 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+36+3], v[vgprLocalReadAddrA+0] offset:34832 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read prefetch b */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

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
s_cmp_le_u32 s[sgprLoopCounterL], 0x1              // LoopCounterL < EndCounter
s_cbranch_scc1 label_LoopEndL                      // do not enter LoopL
.align 16
label_LoopBeginL:

/******************************************/
/* Unrolled Loop 1/1 - Begin              */
/******************************************/

/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */

/* iter 0 */
/*  grEndMfmaIndex:14, lwStartMfmaIndex:32, lwEndMfmaIndex:32  */
/*  numMfmaForLR:4, syncPlrMfmaIndex:35 , sync1LdsMfmaIndex:31 */
/*  mfmaIndex:0  */
buffer_load_b32 v[vgprG2LA+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0 for iteration == 0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:1  */
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
buffer_load_b32 v[vgprG2LA+6], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_1_0
s_setprio 3                                        // store optimization
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:2  */
ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA+0] offset:8736 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA+0] offset:8752 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
buffer_load_b32 v[vgprG2LA+10], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_2_0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:3  */
ds_load_b128 v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], v[vgprLocalReadAddrA+0] offset:17440 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+20:vgprValuA_X1_I0+20+3], v[vgprLocalReadAddrA+0] offset:17456 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], v[vgprLocalReadAddrA+0] offset:26144 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
buffer_load_b32 v[vgprG2LA+14], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_3_0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:4  */
ds_load_b128 v[vgprValuA_X1_I0+28:vgprValuA_X1_I0+28+3], v[vgprLocalReadAddrA+0] offset:26160 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+32:vgprValuA_X1_I0+32+3], v[vgprLocalReadAddrA+0] offset:34848 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+36:vgprValuA_X1_I0+36+3], v[vgprLocalReadAddrA+0] offset:34864 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
buffer_load_b32 v[vgprG2LA+18], v[vgprGlobalReadOffsetA+4], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_4_0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+32+0+0:vgprValuA_X0_I0+32+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=10 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=2 */

/* iter 1 */
/*  grEndMfmaIndex:14, lwStartMfmaIndex:32, lwEndMfmaIndex:32  */
/*  numMfmaForLR:4, syncPlrMfmaIndex:35 , sync1LdsMfmaIndex:31 */
/*  mfmaIndex:5  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b32 v[vgprG2LA+22], v[vgprGlobalReadOffsetA+5], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_5_0
s_waitcnt lgkmcnt(3)                               // wait for prior local read local write old=0, new=3 newLW=0 newLR=3
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:6  */
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:8768 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:8784 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b32 v[vgprG2LA+26], v[vgprGlobalReadOffsetA+6], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_6_0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:7  */
ds_load_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA+0] offset:17472 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA+0] offset:17488 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b32 v[vgprG2LA+30], v[vgprGlobalReadOffsetA+7], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_7_0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:8  */
ds_load_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], v[vgprLocalReadAddrA+0] offset:26176 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[vgprLocalReadAddrA+0] offset:26192 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b32 v[vgprG2LA+34], v[vgprGlobalReadOffsetA+8], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_8_0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+24+0+0:vgprValuA_X1_I0+24+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:9  */
ds_load_b128 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+32+3], v[vgprLocalReadAddrA+0] offset:34880 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+36+3], v[vgprLocalReadAddrA+0] offset:34896 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
buffer_load_b32 v[vgprG2LA+38], v[vgprGlobalReadOffsetA+9], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_9_0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+32+0+0:vgprValuA_X1_I0+32+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=10 */
/* dataAtIterB=0 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=2 */

/* iter 2 */
/*  grEndMfmaIndex:14, lwStartMfmaIndex:32, lwEndMfmaIndex:32  */
/*  numMfmaForLR:4, syncPlrMfmaIndex:35 , sync1LdsMfmaIndex:31 */
/*  mfmaIndex:10  */
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0

/* global read block-scale A */
buffer_load_d16_hi_b16 v[vgprG2LScaleA+0], v[vgprGlobalReadOffsetScaleA+0], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 0 as f32
buffer_load_d16_hi_b16 v[vgprG2LScaleA+1], v[vgprGlobalReadOffsetScaleA+1], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 1 as f32
buffer_load_d16_hi_b16 v[vgprG2LScaleA+2], v[vgprGlobalReadOffsetScaleA+2], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 2 as f32
buffer_load_d16_hi_b16 v[vgprG2LScaleA+3], v[vgprGlobalReadOffsetScaleA+3], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 3 as f32
buffer_load_d16_hi_b16 v[vgprG2LScaleA+4], v[vgprGlobalReadOffsetScaleA+4], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 4 as f32
buffer_load_d16_hi_b16 v[vgprG2LScaleA+5], v[vgprGlobalReadOffsetScaleA+5], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 5 as f32
buffer_load_d16_hi_b16 v[vgprG2LScaleA+6], v[vgprGlobalReadOffsetScaleA+6], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 6 as f32
buffer_load_d16_hi_b16 v[vgprG2LScaleA+7], v[vgprGlobalReadOffsetScaleA+7], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 7 as f32
buffer_load_d16_hi_b16 v[vgprG2LScaleA+8], v[vgprGlobalReadOffsetScaleA+8], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 8 as f32
buffer_load_d16_hi_b16 v[vgprG2LScaleA+9], v[vgprGlobalReadOffsetScaleA+9], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 9 as f32
s_waitcnt lgkmcnt(3)                               // wait for prior local read local write old=0, new=3 newLW=0 newLR=3
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:11  */
ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA+0] offset:8800 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA+0] offset:8816 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:12  */
ds_load_b128 v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], v[vgprLocalReadAddrA+0] offset:17504 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+20:vgprValuA_X1_I0+20+3], v[vgprLocalReadAddrA+0] offset:17520 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
buffer_load_b128 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:13  */
ds_load_b128 v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], v[vgprLocalReadAddrA+0] offset:26208 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+28:vgprValuA_X1_I0+28+3], v[vgprLocalReadAddrA+0] offset:26224 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
buffer_load_b128 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:14  */
ds_load_b128 v[vgprValuA_X1_I0+32:vgprValuA_X1_I0+32+3], v[vgprLocalReadAddrA+0] offset:34912 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+36:vgprValuA_X1_I0+36+3], v[vgprLocalReadAddrA+0] offset:34928 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
buffer_load_b128 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+32+0+0:vgprValuA_X0_I0+32+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=3 skipReadsIterA=1 readsPerIterA=10 */
/* dataAtIterB=1 numReadsIterB=3 skipReadsIterB=1 readsPerIterB=2 */

/* iter 3 */
/*  grEndMfmaIndex:14, lwStartMfmaIndex:32, lwEndMfmaIndex:32  */
/*  numMfmaForLR:4, syncPlrMfmaIndex:35 , sync1LdsMfmaIndex:31 */
/*  mfmaIndex:15  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:144 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], GlobalReadIncsA // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], GlobalReadIncsA // limit -= inc)
s_waitcnt lgkmcnt(3)                               // wait for prior local read local write old=0, new=3 newLW=0 newLR=3
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:16  */
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:144 // L -> Reg lro=64 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:8832 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:8848 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:17  */
ds_load_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA+0] offset:17536 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA+0] offset:17552 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* global read inc block-scale A (2 bytes every 1 iters) */
s_add_u32 s[sgprScaleAKCnt], s[sgprScaleAKCnt], 1  // scaleA: one more K iteration done
s_and_b32 s11, s[sgprScaleAKCnt], s[sgprScaleAKMask] // scaleA: SCC = counter has not wrapped
s_cselect_b32 s11, 0, s[sgprScaleAInc]             // scaleA: advance only on a wrap
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:18  */
ds_load_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], v[vgprLocalReadAddrA+0] offset:26240 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[vgprLocalReadAddrA+0] offset:26256 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], s11 // scaleA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], s11 // scaleA limit -= inc
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+24+0+0:vgprValuA_X1_I0+24+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:19  */
ds_load_b128 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+32+3], v[vgprLocalReadAddrA+0] offset:34944 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+36+3], v[vgprLocalReadAddrA+0] offset:34960 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], GlobalReadIncsB // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], GlobalReadIncsB // limit -= inc)
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+32+0+0:vgprValuA_X1_I0+32+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=4 skipReadsIterA=1 readsPerIterA=10 */
/* dataAtIterB=2 numReadsIterB=4 skipReadsIterB=1 readsPerIterB=2 */

/* iter 4 */
/*  grEndMfmaIndex:14, lwStartMfmaIndex:32, lwEndMfmaIndex:32  */
/*  numMfmaForLR:4, syncPlrMfmaIndex:35 , sync1LdsMfmaIndex:31 */
/*  mfmaIndex:20  */
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:160 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:176 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:160 // L -> Reg lro=80 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32
s_waitcnt lgkmcnt(3)                               // wait for prior local read local write old=0, new=3 newLW=0 newLR=3
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:21  */
ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:176 // L -> Reg lro=80 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA+0] offset:8864 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA+0] offset:8880 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:22  */
ds_load_b128 v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], v[vgprLocalReadAddrA+0] offset:17568 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+20:vgprValuA_X1_I0+20+3], v[vgprLocalReadAddrA+0] offset:17584 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:23  */
ds_load_b128 v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], v[vgprLocalReadAddrA+0] offset:26272 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+28:vgprValuA_X1_I0+28+3], v[vgprLocalReadAddrA+0] offset:26288 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:24  */
ds_load_b128 v[vgprValuA_X1_I0+32:vgprValuA_X1_I0+32+3], v[vgprLocalReadAddrA+0] offset:34976 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+36:vgprValuA_X1_I0+36+3], v[vgprLocalReadAddrA+0] offset:34992 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+32+0+0:vgprValuA_X0_I0+32+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/* numPrefetchIter=0 */
/* dataAtIterA=3 numReadsIterA=5 skipReadsIterA=1 readsPerIterA=10 */
/* dataAtIterB=3 numReadsIterB=5 skipReadsIterB=1 readsPerIterB=2 */

/* iter 5 */
/*  grEndMfmaIndex:14, lwStartMfmaIndex:32, lwEndMfmaIndex:32  */
/*  numMfmaForLR:4, syncPlrMfmaIndex:35 , sync1LdsMfmaIndex:31 */
/*  mfmaIndex:25  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:208 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(3)                               // wait for prior local read local write old=0, new=3 newLW=0 newLR=3
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:26  */
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:208 // L -> Reg lro=96 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:8896 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:8912 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:27  */
ds_load_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA+0] offset:17600 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA+0] offset:17616 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:28  */
ds_load_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], v[vgprLocalReadAddrA+0] offset:26304 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[vgprLocalReadAddrA+0] offset:26320 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+24+0+0:vgprValuA_X1_I0+24+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:29  */
ds_load_b128 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+32+3], v[vgprLocalReadAddrA+0] offset:35008 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+36+3], v[vgprLocalReadAddrA+0] offset:35024 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+32+0+0:vgprValuA_X1_I0+32+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/* numPrefetchIter=0 */
/* dataAtIterA=4 numReadsIterA=6 skipReadsIterA=1 readsPerIterA=10 */
/* dataAtIterB=4 numReadsIterB=6 skipReadsIterB=1 readsPerIterB=2 */

/* iter 6 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:14, lwStartMfmaIndex:32, lwEndMfmaIndex:32  */
/*  numMfmaForLR:4, syncPlrMfmaIndex:35 , sync1LdsMfmaIndex:31 */
/*  mfmaIndex:30  */
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:224 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:240 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:224 // L -> Reg lro=112 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
s_waitcnt lgkmcnt(3)                               // wait for prior local read local write old=0, new=3 newLW=0 newLR=3
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:31  */
/* schedule remaining localreads for one buffer scheduling */
ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:240 // L -> Reg lro=112 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA+0] offset:8928 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA+0] offset:8944 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], v[vgprLocalReadAddrA+0] offset:17632 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+20:vgprValuA_X1_I0+20+3], v[vgprLocalReadAddrA+0] offset:17648 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], v[vgprLocalReadAddrA+0] offset:26336 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+28:vgprValuA_X1_I0+28+3], v[vgprLocalReadAddrA+0] offset:26352 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+32:vgprValuA_X1_I0+32+3], v[vgprLocalReadAddrA+0] offset:35040 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+36:vgprValuA_X1_I0+36+3], v[vgprLocalReadAddrA+0] offset:35056 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
s_setprio 0                                        // store optimization
/*  mfmaIndex:32  */
/* sched write - iter 6 writesPerItem=1 */
s_waitcnt vmcnt(13)                                // wait for global read before writing to local
v_mov_b32 v234, v[vgprG2LScaleA+0]                 // scaleA: copy on the other VGPR bank
v_xor_b32 v235, 0x88888888, v[vgprG2LA+2+0]        // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v236, v235, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v237, v235, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v239, v235, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v238, v235, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v236, s[sgprScaleALiftMagic+6], v236 :: v_dual_subrev_f32 v237, s[sgprScaleALiftMagic+7], v237 // w4a16: n - c, exact
v_dual_subrev_f32 v239, s[sgprScaleALiftMagic+8], v239 :: v_dual_subrev_f32 v238, s[sgprScaleALiftMagic+9], v238 // w4a16: n - c, exact
v_dual_mul_f32 v236, v64, v236 :: v_dual_mul_f32 v237, v234, v237 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v239, v64, v239 :: v_dual_add_nc_u32 v236, s[sgprScaleALiftMagic+5], v236 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v238, v234, v238 :: v_dual_add_nc_u32 v237, s[sgprScaleALiftMagic+5], v237 // w4a16: (n - c) * s :: round
v_and_or_b32 v240, v235, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v235, 20, v235                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v241, v235, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v243, v235, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v242, v235, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v240, s[sgprScaleALiftMagic+10], v240 :: v_dual_add_nc_u32 v239, s[sgprScaleALiftMagic+5], v239 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v241, s[sgprScaleALiftMagic+6], v241 :: v_dual_add_nc_u32 v238, s[sgprScaleALiftMagic+5], v238 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v243, s[sgprScaleALiftMagic+7], v243 :: v_dual_subrev_f32 v242, s[sgprScaleALiftMagic+8], v242 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+0+0], v237, v236, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+1], v238, v239, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v240, v64, v240 :: v_dual_mul_f32 v241, v234, v241 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v243, v64, v243 :: v_dual_add_nc_u32 v240, s[sgprScaleALiftMagic+5], v240 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v242, v234, v242 :: v_dual_add_nc_u32 v241, s[sgprScaleALiftMagic+5], v241 // w4a16: (n - c) * s :: round
v_add_nc_u32 v243, s[sgprScaleALiftMagic+5], v243  // w4a16: round
v_add_nc_u32 v242, s[sgprScaleALiftMagic+5], v242  // w4a16: round
v_perm_b32 v[vgprG2LA+0+2], v241, v240, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+0+3], v242, v243, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0 sync LDS0
/* sched write - iter 6 writesPerItem=1 */
s_waitcnt vmcnt(12)                                // wait for global read before writing to local
v_mov_b32 v234, v[vgprG2LScaleA+1]                 // scaleA: copy on the other VGPR bank
v_xor_b32 v235, 0x88888888, v[vgprG2LA+6+0]        // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v236, v235, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v237, v235, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v239, v235, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v238, v235, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v236, s[sgprScaleALiftMagic+6], v236 :: v_dual_subrev_f32 v237, s[sgprScaleALiftMagic+7], v237 // w4a16: n - c, exact
v_dual_subrev_f32 v239, s[sgprScaleALiftMagic+8], v239 :: v_dual_subrev_f32 v238, s[sgprScaleALiftMagic+9], v238 // w4a16: n - c, exact
v_dual_mul_f32 v236, v65, v236 :: v_dual_mul_f32 v237, v234, v237 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v239, v65, v239 :: v_dual_add_nc_u32 v236, s[sgprScaleALiftMagic+5], v236 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v238, v234, v238 :: v_dual_add_nc_u32 v237, s[sgprScaleALiftMagic+5], v237 // w4a16: (n - c) * s :: round
v_and_or_b32 v240, v235, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v235, 20, v235                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v241, v235, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v243, v235, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v242, v235, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v240, s[sgprScaleALiftMagic+10], v240 :: v_dual_add_nc_u32 v239, s[sgprScaleALiftMagic+5], v239 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v241, s[sgprScaleALiftMagic+6], v241 :: v_dual_add_nc_u32 v238, s[sgprScaleALiftMagic+5], v238 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v243, s[sgprScaleALiftMagic+7], v243 :: v_dual_subrev_f32 v242, s[sgprScaleALiftMagic+8], v242 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+4+0], v237, v236, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+4+1], v238, v239, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v240, v65, v240 :: v_dual_mul_f32 v241, v234, v241 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v243, v65, v243 :: v_dual_add_nc_u32 v240, s[sgprScaleALiftMagic+5], v240 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v242, v234, v242 :: v_dual_add_nc_u32 v241, s[sgprScaleALiftMagic+5], v241 // w4a16: (n - c) * s :: round
v_add_nc_u32 v243, s[sgprScaleALiftMagic+5], v243  // w4a16: round
v_add_nc_u32 v242, s[sgprScaleALiftMagic+5], v242  // w4a16: round
v_perm_b32 v[vgprG2LA+4+2], v241, v240, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+4+3], v242, v243, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:4352 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 4352 sync LDS0
/* sched write - iter 6 writesPerItem=1 */
s_waitcnt vmcnt(11)                                // wait for global read before writing to local
v_mov_b32 v235, v[vgprG2LScaleA+2]                 // scaleA: copy on the other VGPR bank
v_xor_b32 v234, 0x88888888, v[vgprG2LA+10+0]       // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v236, v234, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v237, v234, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v239, v234, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v238, v234, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v236, s[sgprScaleALiftMagic+6], v236 :: v_dual_subrev_f32 v237, s[sgprScaleALiftMagic+7], v237 // w4a16: n - c, exact
v_dual_subrev_f32 v239, s[sgprScaleALiftMagic+8], v239 :: v_dual_subrev_f32 v238, s[sgprScaleALiftMagic+9], v238 // w4a16: n - c, exact
v_dual_mul_f32 v236, v66, v236 :: v_dual_mul_f32 v237, v235, v237 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v239, v66, v239 :: v_dual_add_nc_u32 v236, s[sgprScaleALiftMagic+5], v236 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v238, v235, v238 :: v_dual_add_nc_u32 v237, s[sgprScaleALiftMagic+5], v237 // w4a16: (n - c) * s :: round
v_and_or_b32 v240, v234, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v234, 20, v234                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v241, v234, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v243, v234, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v242, v234, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v240, s[sgprScaleALiftMagic+10], v240 :: v_dual_add_nc_u32 v239, s[sgprScaleALiftMagic+5], v239 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v241, s[sgprScaleALiftMagic+6], v241 :: v_dual_add_nc_u32 v238, s[sgprScaleALiftMagic+5], v238 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v243, s[sgprScaleALiftMagic+7], v243 :: v_dual_subrev_f32 v242, s[sgprScaleALiftMagic+8], v242 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+8+0], v237, v236, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+1], v238, v239, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v240, v66, v240 :: v_dual_mul_f32 v241, v235, v241 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v243, v66, v243 :: v_dual_add_nc_u32 v240, s[sgprScaleALiftMagic+5], v240 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v242, v235, v242 :: v_dual_add_nc_u32 v241, s[sgprScaleALiftMagic+5], v241 // w4a16: (n - c) * s :: round
v_add_nc_u32 v243, s[sgprScaleALiftMagic+5], v243  // w4a16: round
v_add_nc_u32 v242, s[sgprScaleALiftMagic+5], v242  // w4a16: round
v_perm_b32 v[vgprG2LA+8+2], v241, v240, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+8+3], v242, v243, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:8704 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 8704 sync LDS0
/* sched write - iter 6 writesPerItem=1 */
s_waitcnt vmcnt(10)                                // wait for global read before writing to local
v_mov_b32 v234, v[vgprG2LScaleA+3]                 // scaleA: copy on the other VGPR bank
v_xor_b32 v235, 0x88888888, v[vgprG2LA+14+0]       // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v236, v235, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v237, v235, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v239, v235, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v238, v235, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v236, s[sgprScaleALiftMagic+6], v236 :: v_dual_subrev_f32 v237, s[sgprScaleALiftMagic+7], v237 // w4a16: n - c, exact
v_dual_subrev_f32 v239, s[sgprScaleALiftMagic+8], v239 :: v_dual_subrev_f32 v238, s[sgprScaleALiftMagic+9], v238 // w4a16: n - c, exact
v_dual_mul_f32 v236, v67, v236 :: v_dual_mul_f32 v237, v234, v237 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v239, v67, v239 :: v_dual_add_nc_u32 v236, s[sgprScaleALiftMagic+5], v236 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v238, v234, v238 :: v_dual_add_nc_u32 v237, s[sgprScaleALiftMagic+5], v237 // w4a16: (n - c) * s :: round
v_and_or_b32 v240, v235, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v235, 20, v235                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v241, v235, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v243, v235, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v242, v235, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v240, s[sgprScaleALiftMagic+10], v240 :: v_dual_add_nc_u32 v239, s[sgprScaleALiftMagic+5], v239 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v241, s[sgprScaleALiftMagic+6], v241 :: v_dual_add_nc_u32 v238, s[sgprScaleALiftMagic+5], v238 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v243, s[sgprScaleALiftMagic+7], v243 :: v_dual_subrev_f32 v242, s[sgprScaleALiftMagic+8], v242 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+12+0], v237, v236, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+12+1], v238, v239, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v240, v67, v240 :: v_dual_mul_f32 v241, v234, v241 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v243, v67, v243 :: v_dual_add_nc_u32 v240, s[sgprScaleALiftMagic+5], v240 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v242, v234, v242 :: v_dual_add_nc_u32 v241, s[sgprScaleALiftMagic+5], v241 // w4a16: (n - c) * s :: round
v_add_nc_u32 v243, s[sgprScaleALiftMagic+5], v243  // w4a16: round
v_add_nc_u32 v242, s[sgprScaleALiftMagic+5], v242  // w4a16: round
v_perm_b32 v[vgprG2LA+12+2], v241, v240, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+12+3], v242, v243, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+12:vgprG2LA+12+3] offset:13056 // lwoA_0_0_3_0 = (0*LSCA)*(MT0I+PAD) + (3*LSPA) = 13056 sync LDS0
/* sched write - iter 6 writesPerItem=1 */
s_waitcnt vmcnt(9)                                 // wait for global read before writing to local
v_mov_b32 v234, v[vgprG2LScaleA+4]                 // scaleA: copy on the other VGPR bank
v_xor_b32 v235, 0x88888888, v[vgprG2LA+18+0]       // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v236, v235, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v237, v235, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v239, v235, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v238, v235, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v236, s[sgprScaleALiftMagic+6], v236 :: v_dual_subrev_f32 v237, s[sgprScaleALiftMagic+7], v237 // w4a16: n - c, exact
v_dual_subrev_f32 v239, s[sgprScaleALiftMagic+8], v239 :: v_dual_subrev_f32 v238, s[sgprScaleALiftMagic+9], v238 // w4a16: n - c, exact
v_dual_mul_f32 v236, v68, v236 :: v_dual_mul_f32 v237, v234, v237 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v239, v68, v239 :: v_dual_add_nc_u32 v236, s[sgprScaleALiftMagic+5], v236 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v238, v234, v238 :: v_dual_add_nc_u32 v237, s[sgprScaleALiftMagic+5], v237 // w4a16: (n - c) * s :: round
v_and_or_b32 v240, v235, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v235, 20, v235                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v241, v235, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v243, v235, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v242, v235, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v240, s[sgprScaleALiftMagic+10], v240 :: v_dual_add_nc_u32 v239, s[sgprScaleALiftMagic+5], v239 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v241, s[sgprScaleALiftMagic+6], v241 :: v_dual_add_nc_u32 v238, s[sgprScaleALiftMagic+5], v238 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v243, s[sgprScaleALiftMagic+7], v243 :: v_dual_subrev_f32 v242, s[sgprScaleALiftMagic+8], v242 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+16+0], v237, v236, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+16+1], v238, v239, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v240, v68, v240 :: v_dual_mul_f32 v241, v234, v241 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v243, v68, v243 :: v_dual_add_nc_u32 v240, s[sgprScaleALiftMagic+5], v240 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v242, v234, v242 :: v_dual_add_nc_u32 v241, s[sgprScaleALiftMagic+5], v241 // w4a16: (n - c) * s :: round
v_add_nc_u32 v243, s[sgprScaleALiftMagic+5], v243  // w4a16: round
v_add_nc_u32 v242, s[sgprScaleALiftMagic+5], v242  // w4a16: round
v_perm_b32 v[vgprG2LA+16+2], v241, v240, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+16+3], v242, v243, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+16:vgprG2LA+16+3] offset:17408 // lwoA_0_0_4_0 = (0*LSCA)*(MT0I+PAD) + (4*LSPA) = 17408 sync LDS0
/* sched write - iter 6 writesPerItem=1 */
s_waitcnt vmcnt(8)                                 // wait for global read before writing to local
v_mov_b32 v234, v[vgprG2LScaleA+5]                 // scaleA: copy on the other VGPR bank
v_xor_b32 v235, 0x88888888, v[vgprG2LA+22+0]       // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v236, v235, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v237, v235, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v239, v235, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v238, v235, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v236, s[sgprScaleALiftMagic+6], v236 :: v_dual_subrev_f32 v237, s[sgprScaleALiftMagic+7], v237 // w4a16: n - c, exact
v_dual_subrev_f32 v239, s[sgprScaleALiftMagic+8], v239 :: v_dual_subrev_f32 v238, s[sgprScaleALiftMagic+9], v238 // w4a16: n - c, exact
v_dual_mul_f32 v236, v69, v236 :: v_dual_mul_f32 v237, v234, v237 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v239, v69, v239 :: v_dual_add_nc_u32 v236, s[sgprScaleALiftMagic+5], v236 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v238, v234, v238 :: v_dual_add_nc_u32 v237, s[sgprScaleALiftMagic+5], v237 // w4a16: (n - c) * s :: round
v_and_or_b32 v240, v235, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v235, 20, v235                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v241, v235, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v243, v235, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v242, v235, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v240, s[sgprScaleALiftMagic+10], v240 :: v_dual_add_nc_u32 v239, s[sgprScaleALiftMagic+5], v239 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v241, s[sgprScaleALiftMagic+6], v241 :: v_dual_add_nc_u32 v238, s[sgprScaleALiftMagic+5], v238 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v243, s[sgprScaleALiftMagic+7], v243 :: v_dual_subrev_f32 v242, s[sgprScaleALiftMagic+8], v242 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+20+0], v237, v236, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+20+1], v238, v239, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v240, v69, v240 :: v_dual_mul_f32 v241, v234, v241 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v243, v69, v243 :: v_dual_add_nc_u32 v240, s[sgprScaleALiftMagic+5], v240 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v242, v234, v242 :: v_dual_add_nc_u32 v241, s[sgprScaleALiftMagic+5], v241 // w4a16: (n - c) * s :: round
v_add_nc_u32 v243, s[sgprScaleALiftMagic+5], v243  // w4a16: round
v_add_nc_u32 v242, s[sgprScaleALiftMagic+5], v242  // w4a16: round
v_perm_b32 v[vgprG2LA+20+2], v241, v240, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+20+3], v242, v243, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+20:vgprG2LA+20+3] offset:21760 // lwoA_0_0_5_0 = (0*LSCA)*(MT0I+PAD) + (5*LSPA) = 21760 sync LDS0
/* sched write - iter 6 writesPerItem=1 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
v_mov_b32 v235, v[vgprG2LScaleA+6]                 // scaleA: copy on the other VGPR bank
v_xor_b32 v234, 0x88888888, v[vgprG2LA+26+0]       // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v236, v234, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v237, v234, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v239, v234, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v238, v234, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v236, s[sgprScaleALiftMagic+6], v236 :: v_dual_subrev_f32 v237, s[sgprScaleALiftMagic+7], v237 // w4a16: n - c, exact
v_dual_subrev_f32 v239, s[sgprScaleALiftMagic+8], v239 :: v_dual_subrev_f32 v238, s[sgprScaleALiftMagic+9], v238 // w4a16: n - c, exact
v_dual_mul_f32 v236, v70, v236 :: v_dual_mul_f32 v237, v235, v237 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v239, v70, v239 :: v_dual_add_nc_u32 v236, s[sgprScaleALiftMagic+5], v236 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v238, v235, v238 :: v_dual_add_nc_u32 v237, s[sgprScaleALiftMagic+5], v237 // w4a16: (n - c) * s :: round
v_and_or_b32 v240, v234, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v234, 20, v234                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v241, v234, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v243, v234, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v242, v234, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v240, s[sgprScaleALiftMagic+10], v240 :: v_dual_add_nc_u32 v239, s[sgprScaleALiftMagic+5], v239 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v241, s[sgprScaleALiftMagic+6], v241 :: v_dual_add_nc_u32 v238, s[sgprScaleALiftMagic+5], v238 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v243, s[sgprScaleALiftMagic+7], v243 :: v_dual_subrev_f32 v242, s[sgprScaleALiftMagic+8], v242 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+24+0], v237, v236, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+24+1], v238, v239, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v240, v70, v240 :: v_dual_mul_f32 v241, v235, v241 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v243, v70, v243 :: v_dual_add_nc_u32 v240, s[sgprScaleALiftMagic+5], v240 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v242, v235, v242 :: v_dual_add_nc_u32 v241, s[sgprScaleALiftMagic+5], v241 // w4a16: (n - c) * s :: round
v_add_nc_u32 v243, s[sgprScaleALiftMagic+5], v243  // w4a16: round
v_add_nc_u32 v242, s[sgprScaleALiftMagic+5], v242  // w4a16: round
v_perm_b32 v[vgprG2LA+24+2], v241, v240, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+24+3], v242, v243, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+24:vgprG2LA+24+3] offset:26112 // lwoA_0_0_6_0 = (0*LSCA)*(MT0I+PAD) + (6*LSPA) = 26112 sync LDS0
/* sched write - iter 6 writesPerItem=1 */
s_waitcnt vmcnt(6)                                 // wait for global read before writing to local
v_mov_b32 v234, v[vgprG2LScaleA+7]                 // scaleA: copy on the other VGPR bank
v_xor_b32 v235, 0x88888888, v[vgprG2LA+30+0]       // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v236, v235, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v237, v235, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v239, v235, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v238, v235, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v236, s[sgprScaleALiftMagic+6], v236 :: v_dual_subrev_f32 v237, s[sgprScaleALiftMagic+7], v237 // w4a16: n - c, exact
v_dual_subrev_f32 v239, s[sgprScaleALiftMagic+8], v239 :: v_dual_subrev_f32 v238, s[sgprScaleALiftMagic+9], v238 // w4a16: n - c, exact
v_dual_mul_f32 v236, v71, v236 :: v_dual_mul_f32 v237, v234, v237 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v239, v71, v239 :: v_dual_add_nc_u32 v236, s[sgprScaleALiftMagic+5], v236 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v238, v234, v238 :: v_dual_add_nc_u32 v237, s[sgprScaleALiftMagic+5], v237 // w4a16: (n - c) * s :: round
v_and_or_b32 v240, v235, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v235, 20, v235                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v241, v235, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v243, v235, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v242, v235, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v240, s[sgprScaleALiftMagic+10], v240 :: v_dual_add_nc_u32 v239, s[sgprScaleALiftMagic+5], v239 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v241, s[sgprScaleALiftMagic+6], v241 :: v_dual_add_nc_u32 v238, s[sgprScaleALiftMagic+5], v238 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v243, s[sgprScaleALiftMagic+7], v243 :: v_dual_subrev_f32 v242, s[sgprScaleALiftMagic+8], v242 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+28+0], v237, v236, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+28+1], v238, v239, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v240, v71, v240 :: v_dual_mul_f32 v241, v234, v241 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v243, v71, v243 :: v_dual_add_nc_u32 v240, s[sgprScaleALiftMagic+5], v240 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v242, v234, v242 :: v_dual_add_nc_u32 v241, s[sgprScaleALiftMagic+5], v241 // w4a16: (n - c) * s :: round
v_add_nc_u32 v243, s[sgprScaleALiftMagic+5], v243  // w4a16: round
v_add_nc_u32 v242, s[sgprScaleALiftMagic+5], v242  // w4a16: round
v_perm_b32 v[vgprG2LA+28+2], v241, v240, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+28+3], v242, v243, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+28:vgprG2LA+28+3] offset:30464 // lwoA_0_0_7_0 = (0*LSCA)*(MT0I+PAD) + (7*LSPA) = 30464 sync LDS0
/* sched write - iter 6 writesPerItem=1 */
s_waitcnt vmcnt(5)                                 // wait for global read before writing to local
v_mov_b32 v234, v[vgprG2LScaleA+8]                 // scaleA: copy on the other VGPR bank
v_xor_b32 v235, 0x88888888, v[vgprG2LA+34+0]       // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v236, v235, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v237, v235, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v239, v235, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v238, v235, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v236, s[sgprScaleALiftMagic+6], v236 :: v_dual_subrev_f32 v237, s[sgprScaleALiftMagic+7], v237 // w4a16: n - c, exact
v_dual_subrev_f32 v239, s[sgprScaleALiftMagic+8], v239 :: v_dual_subrev_f32 v238, s[sgprScaleALiftMagic+9], v238 // w4a16: n - c, exact
v_dual_mul_f32 v236, v72, v236 :: v_dual_mul_f32 v237, v234, v237 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v239, v72, v239 :: v_dual_add_nc_u32 v236, s[sgprScaleALiftMagic+5], v236 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v238, v234, v238 :: v_dual_add_nc_u32 v237, s[sgprScaleALiftMagic+5], v237 // w4a16: (n - c) * s :: round
v_and_or_b32 v240, v235, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v235, 20, v235                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v241, v235, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v243, v235, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v242, v235, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v240, s[sgprScaleALiftMagic+10], v240 :: v_dual_add_nc_u32 v239, s[sgprScaleALiftMagic+5], v239 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v241, s[sgprScaleALiftMagic+6], v241 :: v_dual_add_nc_u32 v238, s[sgprScaleALiftMagic+5], v238 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v243, s[sgprScaleALiftMagic+7], v243 :: v_dual_subrev_f32 v242, s[sgprScaleALiftMagic+8], v242 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+32+0], v237, v236, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+32+1], v238, v239, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v240, v72, v240 :: v_dual_mul_f32 v241, v234, v241 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v243, v72, v243 :: v_dual_add_nc_u32 v240, s[sgprScaleALiftMagic+5], v240 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v242, v234, v242 :: v_dual_add_nc_u32 v241, s[sgprScaleALiftMagic+5], v241 // w4a16: (n - c) * s :: round
v_add_nc_u32 v243, s[sgprScaleALiftMagic+5], v243  // w4a16: round
v_add_nc_u32 v242, s[sgprScaleALiftMagic+5], v242  // w4a16: round
v_perm_b32 v[vgprG2LA+32+2], v241, v240, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+32+3], v242, v243, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+32:vgprG2LA+32+3] offset:34816 // lwoA_0_0_8_0 = (0*LSCA)*(MT0I+PAD) + (8*LSPA) = 34816 sync LDS0
/* sched write - iter 6 writesPerItem=1 */
s_waitcnt vmcnt(4)                                 // wait for global read before writing to local
v_mov_b32 v234, v[vgprG2LScaleA+9]                 // scaleA: copy on the other VGPR bank
v_xor_b32 v235, 0x88888888, v[vgprG2LA+38+0]       // w4a16: save packed int4 dword 0, biased to unsigned
v_and_or_b32 v236, v235, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v237, v235, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v239, v235, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v238, v235, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v236, s[sgprScaleALiftMagic+6], v236 :: v_dual_subrev_f32 v237, s[sgprScaleALiftMagic+7], v237 // w4a16: n - c, exact
v_dual_subrev_f32 v239, s[sgprScaleALiftMagic+8], v239 :: v_dual_subrev_f32 v238, s[sgprScaleALiftMagic+9], v238 // w4a16: n - c, exact
v_dual_mul_f32 v236, v73, v236 :: v_dual_mul_f32 v237, v234, v237 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v239, v73, v239 :: v_dual_add_nc_u32 v236, s[sgprScaleALiftMagic+5], v236 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v238, v234, v238 :: v_dual_add_nc_u32 v237, s[sgprScaleALiftMagic+5], v237 // w4a16: (n - c) * s :: round
v_and_or_b32 v240, v235, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v235, 20, v235                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v241, v235, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v243, v235, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v242, v235, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v240, s[sgprScaleALiftMagic+10], v240 :: v_dual_add_nc_u32 v239, s[sgprScaleALiftMagic+5], v239 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v241, s[sgprScaleALiftMagic+6], v241 :: v_dual_add_nc_u32 v238, s[sgprScaleALiftMagic+5], v238 // w4a16: n - c, exact :: round (first half)
v_dual_subrev_f32 v243, s[sgprScaleALiftMagic+7], v243 :: v_dual_subrev_f32 v242, s[sgprScaleALiftMagic+8], v242 // w4a16: n - c, exact
v_perm_b32 v[vgprG2LA+36+0], v237, v236, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+36+1], v238, v239, 0x07060302 // w4a16: pack 2 bf16
v_dual_mul_f32 v240, v73, v240 :: v_dual_mul_f32 v241, v234, v241 // w4a16: (n - c) * s, exact
v_dual_mul_f32 v243, v73, v243 :: v_dual_add_nc_u32 v240, s[sgprScaleALiftMagic+5], v240 // w4a16: (n - c) * s :: round
v_dual_mul_f32 v242, v234, v242 :: v_dual_add_nc_u32 v241, s[sgprScaleALiftMagic+5], v241 // w4a16: (n - c) * s :: round
v_add_nc_u32 v243, s[sgprScaleALiftMagic+5], v243  // w4a16: round
v_add_nc_u32 v242, s[sgprScaleALiftMagic+5], v242  // w4a16: round
v_perm_b32 v[vgprG2LA+36+2], v241, v240, 0x07060302 // w4a16: pack 2 bf16
v_perm_b32 v[vgprG2LA+36+3], v242, v243, 0x07060302 // w4a16: pack 2 bf16
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+36:vgprG2LA+36+3] offset:39168 // lwoA_0_0_9_0 = (0*LSCA)*(MT0I+PAD) + (9*LSPA) = 39168 sync LDS0
/* sched write - iter 6 writesPerItem=1 */
s_waitcnt vmcnt(3)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0
/* sched write - iter 6 writesPerItem=1 */
s_waitcnt vmcnt(2)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4:vgprG2LB+4+3] offset:4352 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 4352 sync LDS0
/* sched write - iter 6 writesPerItem=1 */
s_waitcnt vmcnt(1)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8:vgprG2LB+8+3] offset:8704 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 8704 sync LDS0
/* sched write - iter 6 writesPerItem=1 */
s_waitcnt vmcnt(0)                                 // wait for global read before writing to local
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12:vgprG2LB+12+3] offset:13056 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 13056 sync LDS0

/* local write swap offsets a */

/* local write swap offsets b */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:33  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:34  */

/* local read swap offsets a */

/* local read swap offsets b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+32+0+0:vgprValuA_X0_I0+32+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/* numPrefetchIter=0 */
/* dataAtIterA=5 numReadsIterA=7 skipReadsIterA=1 readsPerIterA=10 */
/* dataAtIterB=5 numReadsIterB=7 skipReadsIterB=1 readsPerIterB=2 */

/* iter 7 */
/*  grEndMfmaIndex:14, lwStartMfmaIndex:32, lwEndMfmaIndex:32  */
/*  numMfmaForLR:4, syncPlrMfmaIndex:35 , sync1LdsMfmaIndex:31 */
/*  mfmaIndex:35  */
s_waitcnt lgkmcnt(0)                               // 3wait for local write
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier                                          // PGR, and wait until LW done to sync
s_waitcnt lgkmcnt(14)                              // wait for prior local read local write old=0, new=14 newLW=14 newLR=0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:36  */
s_setprio 3                                        // store optimization
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:37  */
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:8704 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:8720 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:38  */
ds_load_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA+0] offset:17408 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA+0] offset:17424 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], v[vgprLocalReadAddrA+0] offset:26112 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+24+0+0:vgprValuA_X1_I0+24+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:39  */
ds_load_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[vgprLocalReadAddrA+0] offset:26128 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+32+3], v[vgprLocalReadAddrA+0] offset:34816 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+36+3], v[vgprLocalReadAddrA+0] offset:34832 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+32+0+0:vgprValuA_X1_I0+32+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
s_setprio 0                                        // store optimization
/* numPrefetchIter=1 */
/* dataAtIterA=6 numReadsIterA=7 skipReadsIterA=1 readsPerIterA=10 */
/* dataAtIterB=6 numReadsIterB=7 skipReadsIterB=1 readsPerIterB=2 */

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

/* iter 0 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:32, lwEndMfmaIndex:32  */
/*  numMfmaForLR:4, syncPlrMfmaIndex:35 , sync1LdsMfmaIndex:31 */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0 for iteration == 0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:1  */
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
s_setprio 3                                        // store optimization
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:2  */
ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:48 // L -> Reg lro=16 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA+0] offset:8736 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA+0] offset:8752 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:3  */
ds_load_b128 v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], v[vgprLocalReadAddrA+0] offset:17440 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+20:vgprValuA_X1_I0+20+3], v[vgprLocalReadAddrA+0] offset:17456 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], v[vgprLocalReadAddrA+0] offset:26144 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:4  */
ds_load_b128 v[vgprValuA_X1_I0+28:vgprValuA_X1_I0+28+3], v[vgprLocalReadAddrA+0] offset:26160 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+32:vgprValuA_X1_I0+32+3], v[vgprLocalReadAddrA+0] offset:34848 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+36:vgprValuA_X1_I0+36+3], v[vgprLocalReadAddrA+0] offset:34864 // L -> Reg lro=16 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+32+0+0:vgprValuA_X0_I0+32+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=10 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=2 */

/* iter 1 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:32, lwEndMfmaIndex:32  */
/*  numMfmaForLR:4, syncPlrMfmaIndex:35 , sync1LdsMfmaIndex:31 */
/*  mfmaIndex:5  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(3)                               // wait for prior local read local write old=0, new=3 newLW=0 newLR=3
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:6  */
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:80 // L -> Reg lro=32 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:8768 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:8784 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:7  */
ds_load_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA+0] offset:17472 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA+0] offset:17488 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:8  */
ds_load_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], v[vgprLocalReadAddrA+0] offset:26176 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[vgprLocalReadAddrA+0] offset:26192 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+24+0+0:vgprValuA_X1_I0+24+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:9  */
ds_load_b128 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+32+3], v[vgprLocalReadAddrA+0] offset:34880 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+36+3], v[vgprLocalReadAddrA+0] offset:34896 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+32+0+0:vgprValuA_X1_I0+32+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=10 */
/* dataAtIterB=0 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=2 */

/* iter 2 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:32, lwEndMfmaIndex:32  */
/*  numMfmaForLR:4, syncPlrMfmaIndex:35 , sync1LdsMfmaIndex:31 */
/*  mfmaIndex:10  */
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:96 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
s_waitcnt lgkmcnt(3)                               // wait for prior local read local write old=0, new=3 newLW=0 newLR=3
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:11  */
ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:112 // L -> Reg lro=48 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA+0] offset:8800 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA+0] offset:8816 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:12  */
ds_load_b128 v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], v[vgprLocalReadAddrA+0] offset:17504 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+20:vgprValuA_X1_I0+20+3], v[vgprLocalReadAddrA+0] offset:17520 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:13  */
ds_load_b128 v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], v[vgprLocalReadAddrA+0] offset:26208 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+28:vgprValuA_X1_I0+28+3], v[vgprLocalReadAddrA+0] offset:26224 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:14  */
ds_load_b128 v[vgprValuA_X1_I0+32:vgprValuA_X1_I0+32+3], v[vgprLocalReadAddrA+0] offset:34912 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+36:vgprValuA_X1_I0+36+3], v[vgprLocalReadAddrA+0] offset:34928 // L -> Reg lro=48 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+32+0+0:vgprValuA_X0_I0+32+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=3 skipReadsIterA=1 readsPerIterA=10 */
/* dataAtIterB=1 numReadsIterB=3 skipReadsIterB=1 readsPerIterB=2 */

/* iter 3 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:32, lwEndMfmaIndex:32  */
/*  numMfmaForLR:4, syncPlrMfmaIndex:35 , sync1LdsMfmaIndex:31 */
/*  mfmaIndex:15  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:144 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(3)                               // wait for prior local read local write old=0, new=3 newLW=0 newLR=3
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:16  */
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:144 // L -> Reg lro=64 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:8832 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:8848 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:17  */
ds_load_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA+0] offset:17536 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA+0] offset:17552 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:18  */
ds_load_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], v[vgprLocalReadAddrA+0] offset:26240 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[vgprLocalReadAddrA+0] offset:26256 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+24+0+0:vgprValuA_X1_I0+24+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:19  */
ds_load_b128 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+32+3], v[vgprLocalReadAddrA+0] offset:34944 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+36+3], v[vgprLocalReadAddrA+0] offset:34960 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+32+0+0:vgprValuA_X1_I0+32+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
s_setprio 0                                        // store optimization
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=4 skipReadsIterA=1 readsPerIterA=10 */
/* dataAtIterB=2 numReadsIterB=4 skipReadsIterB=1 readsPerIterB=2 */

/* iter 4 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:32, lwEndMfmaIndex:32  */
/*  numMfmaForLR:4, syncPlrMfmaIndex:35 , sync1LdsMfmaIndex:31 */
/*  mfmaIndex:20  */
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:160 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:176 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:160 // L -> Reg lro=80 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
s_waitcnt lgkmcnt(3)                               // wait for prior local read local write old=0, new=3 newLW=0 newLR=3
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:21  */
ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:176 // L -> Reg lro=80 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA+0] offset:8864 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA+0] offset:8880 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:22  */
ds_load_b128 v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], v[vgprLocalReadAddrA+0] offset:17568 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+20:vgprValuA_X1_I0+20+3], v[vgprLocalReadAddrA+0] offset:17584 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:23  */
ds_load_b128 v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], v[vgprLocalReadAddrA+0] offset:26272 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+28:vgprValuA_X1_I0+28+3], v[vgprLocalReadAddrA+0] offset:26288 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:24  */
ds_load_b128 v[vgprValuA_X1_I0+32:vgprValuA_X1_I0+32+3], v[vgprLocalReadAddrA+0] offset:34976 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+36:vgprValuA_X1_I0+36+3], v[vgprLocalReadAddrA+0] offset:34992 // L -> Reg lro=80 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+32+0+0:vgprValuA_X0_I0+32+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/* numPrefetchIter=0 */
/* dataAtIterA=3 numReadsIterA=5 skipReadsIterA=1 readsPerIterA=10 */
/* dataAtIterB=3 numReadsIterB=5 skipReadsIterB=1 readsPerIterB=2 */

/* iter 5 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:32, lwEndMfmaIndex:32  */
/*  numMfmaForLR:4, syncPlrMfmaIndex:35 , sync1LdsMfmaIndex:31 */
/*  mfmaIndex:25  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:208 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(3)                               // wait for prior local read local write old=0, new=3 newLW=0 newLR=3
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:26  */
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:208 // L -> Reg lro=96 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:8896 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:8912 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:27  */
ds_load_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA+0] offset:17600 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA+0] offset:17616 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:28  */
ds_load_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], v[vgprLocalReadAddrA+0] offset:26304 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[vgprLocalReadAddrA+0] offset:26320 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+24+0+0:vgprValuA_X1_I0+24+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:29  */
ds_load_b128 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+32+3], v[vgprLocalReadAddrA+0] offset:35008 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+36+3], v[vgprLocalReadAddrA+0] offset:35024 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+32+0+0:vgprValuA_X1_I0+32+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/* numPrefetchIter=0 */
/* dataAtIterA=4 numReadsIterA=6 skipReadsIterA=1 readsPerIterA=10 */
/* dataAtIterB=4 numReadsIterB=6 skipReadsIterB=1 readsPerIterB=2 */

/* iter 6 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:32, lwEndMfmaIndex:32  */
/*  numMfmaForLR:4, syncPlrMfmaIndex:35 , sync1LdsMfmaIndex:31 */
/*  mfmaIndex:30  */
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:224 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:240 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:224 // L -> Reg lro=112 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
s_waitcnt lgkmcnt(3)                               // wait for prior local read local write old=0, new=3 newLW=0 newLR=3
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:31  */
/* schedule remaining localreads for one buffer scheduling */
ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:240 // L -> Reg lro=112 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA+0] offset:8928 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA+0] offset:8944 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], v[vgprLocalReadAddrA+0] offset:17632 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+20:vgprValuA_X1_I0+20+3], v[vgprLocalReadAddrA+0] offset:17648 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], v[vgprLocalReadAddrA+0] offset:26336 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+28:vgprValuA_X1_I0+28+3], v[vgprLocalReadAddrA+0] offset:26352 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+32:vgprValuA_X1_I0+32+3], v[vgprLocalReadAddrA+0] offset:35040 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+36:vgprValuA_X1_I0+36+3], v[vgprLocalReadAddrA+0] offset:35056 // L -> Reg lro=112 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:32  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:33  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:34  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+32+0+0:vgprValuA_X0_I0+32+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/* numPrefetchIter=0 */
/* dataAtIterA=5 numReadsIterA=7 skipReadsIterA=1 readsPerIterA=10 */
/* dataAtIterB=5 numReadsIterB=7 skipReadsIterB=1 readsPerIterB=2 */

/* iter 7 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:32, lwEndMfmaIndex:32  */
/*  numMfmaForLR:4, syncPlrMfmaIndex:35 , sync1LdsMfmaIndex:31 */
/*  mfmaIndex:35  */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_wmma_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
/*  mfmaIndex:36  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
/*  mfmaIndex:37  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+16+0+0:vgprValuA_X1_I0+16+0+0+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
/*  mfmaIndex:38  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+24+0+0:vgprValuA_X1_I0+24+0+0+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
/*  mfmaIndex:39  */
v_wmma_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuA_X1_I0+32+0+0:vgprValuA_X1_I0+32+0+0+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
/* numPrefetchIter=0 */
/* dataAtIterA=6 numReadsIterA=7 skipReadsIterA=0 readsPerIterA=10 */
/* dataAtIterB=6 numReadsIterB=7 skipReadsIterB=0 readsPerIterB=2 */
label_toPGR1end_OrdNLL:
label_PrefetchGlobalLastIterEnd:

/* Tail: add ValuA/B vgpr buffer [78...175) to pool */

/* Tail: add address/G2L vgpr [175...232) to pool */
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
v_lshrrev_b32 v82, 5, v[vgprSerial]                // 82 = Serial / 32
v_lshrrev_b32 v83, 1, v82                          // 83 = 82 / 2
v_mul_lo_u32 v83, 0x10, v83                        // wave coordination offset 1
v_and_b32 v79, 31, v[vgprSerial]                   // v79 = v[vgprSerial] % 32
v_lshrrev_b32 v79, 4, v79                          // 79 = 79 / 16
                                                   // thread0 * continuous_output (multiplier is 1, do nothing)
v_add_lshl_u32 v79, v83, v79, 0                    // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v80, v79, s[sgprStrideC1J]            //  offset 1
v_mul_lo_u32 v81, v79, s[sgprStrideD1J]            //  offset 1
v_and_b32 v78, 1, v82                              // v78 = v82 % 2
v_mul_lo_u32 v78, 0x10, v78                        // wave coordination offset 0
v_and_b32 v83, 15, v[vgprSerial]                   // v83 = v[vgprSerial] % 16
v_add_lshl_u32 v78, v83, v78, 0                    // coordination 0 = vwA * (wave_id0 + tid0)
s_mul_i32 s7, 160, s[sgprWorkGroup0]               // wgp0 * MT0
v_add_nc_u32 v78, s7, v78                          // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s7, 64, s[sgprWorkGroup1]                // wgp1 * MT1
v_add_nc_u32 v79, s7, v79                          // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1

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

/* Edge/NonEdge store path check (M): Size % 160 > 0 -> Edge store; else -> NonEdge store */
s_mul_hi_u32 s71, s[sgprSizeI], 53687092           // STATIC_DIV: divisor=160
s_mul_i32 s70, s[sgprSizeI], 53687092              // tmp = dividend * magic
s_lshr_b64 s[70:71], s[70:71], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s69, s70                                 // quotient
s_mul_i32 s70, s69, 160                            // quotient*divisor
s_sub_u32 s68, s[sgprSizeI], s70                   // rReg = dividend - quotient*divisor
s_add_u32 s69, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s69                // wg0 >= nwg0-1 ?
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_MB_Else         // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 64 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s68, 63, s[sgprSizeJ]                    // s68 = s[sgprSizeJ] % 64
s_add_u32 s69, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s69                // wg1 >= nwg1-1
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_MB_Then         // jump if edges required
label_GW_B0_FD0_VW1_MB_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=164 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,4,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (1,4,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (2,4,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1); (3,4,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (4,3,0,0:vw1); (4,4,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (5,3,0,0:vw1); (5,4,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (6,3,0,0:vw1); (6,4,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (7,3,0,0:vw1); (7,4,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
/* (d1,vc1,d0,vc0)=(1,0,4,0) */
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
/* (d1,vc1,d0,vc0)=(2,0,4,0) */
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
/* (d1,vc1,d0,vc0)=(3,0,4,0) */
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
/* (d1,vc1,d0,vc0)=(4,0,2,0) */
/* (d1,vc1,d0,vc0)=(4,0,3,0) */
/* (d1,vc1,d0,vc0)=(4,0,4,0) */
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
/* (d1,vc1,d0,vc0)=(5,0,2,0) */
/* (d1,vc1,d0,vc0)=(5,0,3,0) */
/* (d1,vc1,d0,vc0)=(5,0,4,0) */
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
/* (d1,vc1,d0,vc0)=(6,0,2,0) */
/* (d1,vc1,d0,vc0)=(6,0,3,0) */
/* (d1,vc1,d0,vc0)=(6,0,4,0) */
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
/* (d1,vc1,d0,vc0)=(7,0,2,0) */
/* (d1,vc1,d0,vc0)=(7,0,3,0) */
/* (d1,vc1,d0,vc0)=(7,0,4,0) */
v_add_lshl_u32 v89, v81, v78, 2                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=78, coord0Vgpr=78 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (1, 4, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (2, 3, 0, 0), (2, 4, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (3, 3, 0, 0), (3, 4, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0), (4, 3, 0, 0), (4, 4, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (5, 2, 0, 0), (5, 3, 0, 0), (5, 4, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (6, 3, 0, 0), (6, 4, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (7, 2, 0, 0), (7, 3, 0, 0), (7, 4, 0, 0)] */
v_mov_b32 v[vgprValuC+91], v[vgprValuC+0]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+92], v[vgprValuC+8]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+93], v[vgprValuC+16]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+94], v[vgprValuC+24]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+95], v[vgprValuC+32]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+96], v[vgprValuC+1]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+97], v[vgprValuC+9]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+98], v[vgprValuC+17]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+99], v[vgprValuC+25]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+100], v[vgprValuC+33]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+101], v[vgprValuC+2]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+102], v[vgprValuC+10]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+103], v[vgprValuC+18]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+104], v[vgprValuC+26]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+105], v[vgprValuC+34]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+106], v[vgprValuC+3]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+107], v[vgprValuC+11]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+108], v[vgprValuC+19]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+109], v[vgprValuC+27]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+110], v[vgprValuC+35]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+111], v[vgprValuC+4]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+112], v[vgprValuC+12]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+113], v[vgprValuC+20]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+114], v[vgprValuC+28]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+115], v[vgprValuC+36]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+116], v[vgprValuC+5]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+117], v[vgprValuC+13]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+118], v[vgprValuC+21]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+119], v[vgprValuC+29]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+120], v[vgprValuC+37]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+121], v[vgprValuC+6]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+122], v[vgprValuC+14]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+123], v[vgprValuC+22]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+124], v[vgprValuC+30]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+125], v[vgprValuC+38]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+126], v[vgprValuC+7]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+127], v[vgprValuC+15]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+128], v[vgprValuC+23]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+129], v[vgprValuC+31]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+130], v[vgprValuC+39]        // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b32 v91, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v92, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b32 v93, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v94, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:384 // store D
buffer_store_b32 v95, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v96, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v97, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b32 v98, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v99, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:384 // store D
buffer_store_b32 v100, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v101, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v102, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b32 v103, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v104, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:384 // store D
buffer_store_b32 v105, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v106, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v107, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b32 v108, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v109, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:384 // store D
buffer_store_b32 v110, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v111, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v112, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b32 v113, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v114, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:384 // store D
buffer_store_b32 v115, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v116, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v117, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b32 v118, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v119, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:384 // store D
buffer_store_b32 v120, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v121, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v122, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b32 v123, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v124, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:384 // store D
buffer_store_b32 v125, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_mul_i32 s8, s[sgprStrideD1J], 8                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b32 v126, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v127, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b32 v128, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_b32 v129, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:384 // store D
buffer_store_b32 v130, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:512 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_B0_FD0_VW1_MB_NonEdgeEnd:
label_GW_B0_FD0_VW1_MB_Else:
label_GW_B0_FD0_VW1_MB_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=82 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,4,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (1,4,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (2,4,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1); (3,4,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (4,3,0,0:vw1); (4,4,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (5,3,0,0:vw1); (5,4,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (6,3,0,0:vw1); (6,4,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (7,3,0,0:vw1); (7,4,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v84, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s68, v78, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v129, v81, v78, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v129, v84, v129, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v82, vcc_lo, v78, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v130, v81, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v84, v130, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
v_add_co_u32 v82, vcc_lo, v78, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v131, v81, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v131, v84, v131, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
s_mov_b32 s68, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v132, v81, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v84, v132, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
s_mov_b32 s68, 128                                 // coordOffset0 d0=4 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v133, v81, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v133, v84, v133, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v79, vcc_lo, v79, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v80, v80, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v81, v81, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v78, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v134, v81, v78, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v84, v134, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v82, vcc_lo, v78, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v135, v81, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v135, v84, v135, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
v_add_co_u32 v82, vcc_lo, v78, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v136, v81, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v84, v136, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
s_mov_b32 s68, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v137, v81, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v137, v84, v137, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,4,0) */
s_mov_b32 s68, 128                                 // coordOffset0 d0=4 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v138, v81, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v84, v138, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v79, vcc_lo, v79, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v80, v80, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v81, v81, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v78, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v139, v81, v78, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v139, v84, v139, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
v_add_co_u32 v82, vcc_lo, v78, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v140, v81, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v84, v140, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
v_add_co_u32 v82, vcc_lo, v78, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v141, v81, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v141, v84, v141, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
s_mov_b32 s68, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v142, v81, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v84, v142, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,4,0) */
s_mov_b32 s68, 128                                 // coordOffset0 d0=4 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v143, v81, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v143, v84, v143, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v79, vcc_lo, v79, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v80, v80, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v81, v81, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v78, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v144, v81, v78, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v144, v84, v144, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
v_add_co_u32 v82, vcc_lo, v78, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v145, v81, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v145, v84, v145, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
v_add_co_u32 v82, vcc_lo, v78, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v146, v81, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v146, v84, v146, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
s_mov_b32 s68, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v147, v81, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v147, v84, v147, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,4,0) */
s_mov_b32 s68, 128                                 // coordOffset0 d0=4 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v148, v81, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v148, v84, v148, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v79, vcc_lo, v79, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v80, v80, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v81, v81, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v78, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v149, v81, v78, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v149, v84, v149, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
v_add_co_u32 v82, vcc_lo, v78, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v150, v81, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v150, v84, v150, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,2,0) */
v_add_co_u32 v82, vcc_lo, v78, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v151, v81, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v151, v84, v151, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,3,0) */
s_mov_b32 s68, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v152, v81, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v152, v84, v152, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,4,0) */
s_mov_b32 s68, 128                                 // coordOffset0 d0=4 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v153, v81, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v153, v84, v153, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
v_add_co_u32 v79, vcc_lo, v79, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v80, v80, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v81, v81, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v78, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v154, v81, v78, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v84, v154, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
v_add_co_u32 v82, vcc_lo, v78, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v155, v81, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v155, v84, v155, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,2,0) */
v_add_co_u32 v82, vcc_lo, v78, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v156, v81, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v84, v156, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,3,0) */
s_mov_b32 s68, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v157, v81, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v157, v84, v157, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,4,0) */
s_mov_b32 s68, 128                                 // coordOffset0 d0=4 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v158, v81, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v84, v158, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
v_add_co_u32 v79, vcc_lo, v79, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v80, v80, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v81, v81, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v78, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v159, v81, v78, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v159, v84, v159, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
v_add_co_u32 v82, vcc_lo, v78, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v160, v81, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v160, v84, v160, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,2,0) */
v_add_co_u32 v82, vcc_lo, v78, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v161, v81, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v161, v84, v161, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,3,0) */
s_mov_b32 s68, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v162, v81, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v162, v84, v162, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,4,0) */
s_mov_b32 s68, 128                                 // coordOffset0 d0=4 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v163, v81, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v163, v84, v163, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
v_add_co_u32 v79, vcc_lo, v79, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v80, v80, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v81, v81, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v78, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v164, v81, v78, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v164, v84, v164, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
v_add_co_u32 v82, vcc_lo, v78, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v165, v81, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v165, v84, v165, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,2,0) */
v_add_co_u32 v82, vcc_lo, v78, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v166, v81, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v166, v84, v166, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,3,0) */
s_mov_b32 s68, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v167, v81, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v167, v84, v167, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,4,0) */
s_mov_b32 s68, 128                                 // coordOffset0 d0=4 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v168, v81, v82, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v168, v84, v168, s70                 // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (1, 4, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (2, 3, 0, 0), (2, 4, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (3, 3, 0, 0), (3, 4, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0), (4, 3, 0, 0), (4, 4, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (5, 2, 0, 0), (5, 3, 0, 0), (5, 4, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (6, 3, 0, 0), (6, 4, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (7, 2, 0, 0), (7, 3, 0, 0), (7, 4, 0, 0)] */
v_mov_b32 v[vgprValuC+89], v[vgprValuC+0]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+90], v[vgprValuC+8]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+91], v[vgprValuC+16]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+92], v[vgprValuC+24]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+93], v[vgprValuC+32]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+94], v[vgprValuC+1]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+95], v[vgprValuC+9]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+96], v[vgprValuC+17]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+97], v[vgprValuC+25]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+98], v[vgprValuC+33]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+99], v[vgprValuC+2]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+100], v[vgprValuC+10]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+101], v[vgprValuC+18]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+102], v[vgprValuC+26]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+103], v[vgprValuC+34]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+104], v[vgprValuC+3]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+105], v[vgprValuC+11]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+106], v[vgprValuC+19]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+107], v[vgprValuC+27]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+108], v[vgprValuC+35]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+109], v[vgprValuC+4]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+110], v[vgprValuC+12]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+111], v[vgprValuC+20]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+112], v[vgprValuC+28]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+113], v[vgprValuC+36]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+114], v[vgprValuC+5]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+115], v[vgprValuC+13]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+116], v[vgprValuC+21]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+117], v[vgprValuC+29]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+118], v[vgprValuC+37]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+119], v[vgprValuC+6]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+120], v[vgprValuC+14]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+121], v[vgprValuC+22]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+122], v[vgprValuC+30]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+123], v[vgprValuC+38]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+124], v[vgprValuC+7]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+125], v[vgprValuC+15]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+126], v[vgprValuC+23]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+127], v[vgprValuC+31]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+128], v[vgprValuC+39]        // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b32 v89, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v90, v130, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v91, v131, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v92, v132, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v93, v133, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v94, v134, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v95, v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v96, v136, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v97, v137, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v98, v138, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v99, v139, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v100, v140, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v101, v141, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v102, v142, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v103, v143, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v104, v144, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v105, v145, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v106, v146, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v107, v147, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v108, v148, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v109, v149, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v110, v150, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v111, v151, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v112, v152, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v113, v153, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v114, v154, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v115, v155, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v116, v156, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v117, v157, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v118, v158, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v119, v159, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v120, v160, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v121, v161, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v122, v162, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v123, v163, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v124, v164, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v125, v165, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v126, v166, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v127, v167, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v128, v168, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
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
v_mov_b32 v89, s70                                 // move flag value to vgpr
v_mov_b32 v90, 0                                   // zero vaddr offset
s_mov_b64 s[72:73], s[sgprAddressFlags:sgprAddressFlags+1]
s_mov_b32 s74, BufferOOB
s_mov_b32 s75, Srd127_96
buffer_store_b32 v89, v90, s[72:75], s68 offen offset:0 glc dlc // reset flag
s_waitcnt_vscnt null 0                             // wait for data store
label_SK_SkipFlagReset:
label_Fixup_E0:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=80 */
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
/*      (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,4,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (1,4,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (2,4,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1); (3,4,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (4,3,0,0:vw1); (4,4,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (5,3,0,0:vw1); (5,4,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (6,3,0,0:vw1); (6,4,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (7,3,0,0:vw1); (7,4,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_lshlrev_b32 v90, 2, v[vgprSerial]                // v90 = v[vgprSerial] * 4
s_mov_b32 s62, 0                                   // Init sgpr offset
buffer_load_b32 v131, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v132, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v133, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v134, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v135, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v136, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v137, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v138, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v139, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v140, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v141, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v142, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v143, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v144, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v145, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v146, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v147, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v148, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v149, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v150, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v151, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v152, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v153, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v154, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v155, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v156, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v157, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v158, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v159, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v160, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v161, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v162, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v163, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v164, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v165, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v166, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v167, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v168, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v169, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
s_add_u32 s62, s62, 1024                           // Inc sgpr offset
buffer_load_b32 v170, v90, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 glc slc // load WS
v_mov_b32 v[vgprValuC+91], v[vgprValuC+0]          // copy MI out reg to vreg[0]
v_mov_b32 v[vgprValuC+92], v[vgprValuC+8]          // copy MI out reg to vreg[1]
v_mov_b32 v[vgprValuC+93], v[vgprValuC+16]         // copy MI out reg to vreg[2]
v_mov_b32 v[vgprValuC+94], v[vgprValuC+24]         // copy MI out reg to vreg[3]
v_mov_b32 v[vgprValuC+95], v[vgprValuC+32]         // copy MI out reg to vreg[4]
v_mov_b32 v[vgprValuC+96], v[vgprValuC+1]          // copy MI out reg to vreg[5]
v_mov_b32 v[vgprValuC+97], v[vgprValuC+9]          // copy MI out reg to vreg[6]
v_mov_b32 v[vgprValuC+98], v[vgprValuC+17]         // copy MI out reg to vreg[7]
v_mov_b32 v[vgprValuC+99], v[vgprValuC+25]         // copy MI out reg to vreg[8]
v_mov_b32 v[vgprValuC+100], v[vgprValuC+33]        // copy MI out reg to vreg[9]
v_mov_b32 v[vgprValuC+101], v[vgprValuC+2]         // copy MI out reg to vreg[10]
v_mov_b32 v[vgprValuC+102], v[vgprValuC+10]        // copy MI out reg to vreg[11]
v_mov_b32 v[vgprValuC+103], v[vgprValuC+18]        // copy MI out reg to vreg[12]
v_mov_b32 v[vgprValuC+104], v[vgprValuC+26]        // copy MI out reg to vreg[13]
v_mov_b32 v[vgprValuC+105], v[vgprValuC+34]        // copy MI out reg to vreg[14]
v_mov_b32 v[vgprValuC+106], v[vgprValuC+3]         // copy MI out reg to vreg[15]
v_mov_b32 v[vgprValuC+107], v[vgprValuC+11]        // copy MI out reg to vreg[16]
v_mov_b32 v[vgprValuC+108], v[vgprValuC+19]        // copy MI out reg to vreg[17]
v_mov_b32 v[vgprValuC+109], v[vgprValuC+27]        // copy MI out reg to vreg[18]
v_mov_b32 v[vgprValuC+110], v[vgprValuC+35]        // copy MI out reg to vreg[19]
v_mov_b32 v[vgprValuC+111], v[vgprValuC+4]         // copy MI out reg to vreg[20]
v_mov_b32 v[vgprValuC+112], v[vgprValuC+12]        // copy MI out reg to vreg[21]
v_mov_b32 v[vgprValuC+113], v[vgprValuC+20]        // copy MI out reg to vreg[22]
v_mov_b32 v[vgprValuC+114], v[vgprValuC+28]        // copy MI out reg to vreg[23]
v_mov_b32 v[vgprValuC+115], v[vgprValuC+36]        // copy MI out reg to vreg[24]
v_mov_b32 v[vgprValuC+116], v[vgprValuC+5]         // copy MI out reg to vreg[25]
v_mov_b32 v[vgprValuC+117], v[vgprValuC+13]        // copy MI out reg to vreg[26]
v_mov_b32 v[vgprValuC+118], v[vgprValuC+21]        // copy MI out reg to vreg[27]
v_mov_b32 v[vgprValuC+119], v[vgprValuC+29]        // copy MI out reg to vreg[28]
v_mov_b32 v[vgprValuC+120], v[vgprValuC+37]        // copy MI out reg to vreg[29]
v_mov_b32 v[vgprValuC+121], v[vgprValuC+6]         // copy MI out reg to vreg[30]
v_mov_b32 v[vgprValuC+122], v[vgprValuC+14]        // copy MI out reg to vreg[31]
v_mov_b32 v[vgprValuC+123], v[vgprValuC+22]        // copy MI out reg to vreg[32]
v_mov_b32 v[vgprValuC+124], v[vgprValuC+30]        // copy MI out reg to vreg[33]
v_mov_b32 v[vgprValuC+125], v[vgprValuC+38]        // copy MI out reg to vreg[34]
v_mov_b32 v[vgprValuC+126], v[vgprValuC+7]         // copy MI out reg to vreg[35]
v_mov_b32 v[vgprValuC+127], v[vgprValuC+15]        // copy MI out reg to vreg[36]
v_mov_b32 v[vgprValuC+128], v[vgprValuC+23]        // copy MI out reg to vreg[37]
v_mov_b32 v[vgprValuC+129], v[vgprValuC+31]        // copy MI out reg to vreg[38]
v_mov_b32 v[vgprValuC+130], v[vgprValuC+39]        // copy MI out reg to vreg[39]

/* apply mask, calc new C and issue writes */
v_mov_b32 v86, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v87, 0x7fff0000                          // fp32 Nan
v_mov_b32 v88, 0x7fff                              // rounding bias for bfloat16

s_waitcnt vmcnt(39)                                // wait C (interleaved) 39 = 40 - 0 - 1
v_add_f32 v[vgprValuC+91], v[vgprValuC+91], v131   // accum partials

s_waitcnt vmcnt(38)                                // wait C (interleaved) 38 = 40 - 1 - 1
v_add_f32 v[vgprValuC+92], v[vgprValuC+92], v132   // accum partials

s_waitcnt vmcnt(37)                                // wait C (interleaved) 37 = 40 - 2 - 1
v_add_f32 v[vgprValuC+93], v[vgprValuC+93], v133   // accum partials

s_waitcnt vmcnt(36)                                // wait C (interleaved) 36 = 40 - 3 - 1
v_add_f32 v[vgprValuC+94], v[vgprValuC+94], v134   // accum partials

s_waitcnt vmcnt(35)                                // wait C (interleaved) 35 = 40 - 4 - 1
v_add_f32 v[vgprValuC+95], v[vgprValuC+95], v135   // accum partials

s_waitcnt vmcnt(34)                                // wait C (interleaved) 34 = 40 - 5 - 1
v_add_f32 v[vgprValuC+96], v[vgprValuC+96], v136   // accum partials

s_waitcnt vmcnt(33)                                // wait C (interleaved) 33 = 40 - 6 - 1
v_add_f32 v[vgprValuC+97], v[vgprValuC+97], v137   // accum partials

s_waitcnt vmcnt(32)                                // wait C (interleaved) 32 = 40 - 7 - 1
v_add_f32 v[vgprValuC+98], v[vgprValuC+98], v138   // accum partials

s_waitcnt vmcnt(31)                                // wait C (interleaved) 31 = 40 - 8 - 1
v_add_f32 v[vgprValuC+99], v[vgprValuC+99], v139   // accum partials

s_waitcnt vmcnt(30)                                // wait C (interleaved) 30 = 40 - 9 - 1
v_add_f32 v[vgprValuC+100], v[vgprValuC+100], v140 // accum partials

s_waitcnt vmcnt(29)                                // wait C (interleaved) 29 = 40 - 10 - 1
v_add_f32 v[vgprValuC+101], v[vgprValuC+101], v141 // accum partials

s_waitcnt vmcnt(28)                                // wait C (interleaved) 28 = 40 - 11 - 1
v_add_f32 v[vgprValuC+102], v[vgprValuC+102], v142 // accum partials

s_waitcnt vmcnt(27)                                // wait C (interleaved) 27 = 40 - 12 - 1
v_add_f32 v[vgprValuC+103], v[vgprValuC+103], v143 // accum partials

s_waitcnt vmcnt(26)                                // wait C (interleaved) 26 = 40 - 13 - 1
v_add_f32 v[vgprValuC+104], v[vgprValuC+104], v144 // accum partials

s_waitcnt vmcnt(25)                                // wait C (interleaved) 25 = 40 - 14 - 1
v_add_f32 v[vgprValuC+105], v[vgprValuC+105], v145 // accum partials

s_waitcnt vmcnt(24)                                // wait C (interleaved) 24 = 40 - 15 - 1
v_add_f32 v[vgprValuC+106], v[vgprValuC+106], v146 // accum partials

s_waitcnt vmcnt(23)                                // wait C (interleaved) 23 = 40 - 16 - 1
v_add_f32 v[vgprValuC+107], v[vgprValuC+107], v147 // accum partials

s_waitcnt vmcnt(22)                                // wait C (interleaved) 22 = 40 - 17 - 1
v_add_f32 v[vgprValuC+108], v[vgprValuC+108], v148 // accum partials

s_waitcnt vmcnt(21)                                // wait C (interleaved) 21 = 40 - 18 - 1
v_add_f32 v[vgprValuC+109], v[vgprValuC+109], v149 // accum partials

s_waitcnt vmcnt(20)                                // wait C (interleaved) 20 = 40 - 19 - 1
v_add_f32 v[vgprValuC+110], v[vgprValuC+110], v150 // accum partials

s_waitcnt vmcnt(19)                                // wait C (interleaved) 19 = 40 - 20 - 1
v_add_f32 v[vgprValuC+111], v[vgprValuC+111], v151 // accum partials

s_waitcnt vmcnt(18)                                // wait C (interleaved) 18 = 40 - 21 - 1
v_add_f32 v[vgprValuC+112], v[vgprValuC+112], v152 // accum partials

s_waitcnt vmcnt(17)                                // wait C (interleaved) 17 = 40 - 22 - 1
v_add_f32 v[vgprValuC+113], v[vgprValuC+113], v153 // accum partials

s_waitcnt vmcnt(16)                                // wait C (interleaved) 16 = 40 - 23 - 1
v_add_f32 v[vgprValuC+114], v[vgprValuC+114], v154 // accum partials

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 40 - 24 - 1
v_add_f32 v[vgprValuC+115], v[vgprValuC+115], v155 // accum partials

s_waitcnt vmcnt(14)                                // wait C (interleaved) 14 = 40 - 25 - 1
v_add_f32 v[vgprValuC+116], v[vgprValuC+116], v156 // accum partials

s_waitcnt vmcnt(13)                                // wait C (interleaved) 13 = 40 - 26 - 1
v_add_f32 v[vgprValuC+117], v[vgprValuC+117], v157 // accum partials

s_waitcnt vmcnt(12)                                // wait C (interleaved) 12 = 40 - 27 - 1
v_add_f32 v[vgprValuC+118], v[vgprValuC+118], v158 // accum partials

s_waitcnt vmcnt(11)                                // wait C (interleaved) 11 = 40 - 28 - 1
v_add_f32 v[vgprValuC+119], v[vgprValuC+119], v159 // accum partials

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 40 - 29 - 1
v_add_f32 v[vgprValuC+120], v[vgprValuC+120], v160 // accum partials

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 40 - 30 - 1
v_add_f32 v[vgprValuC+121], v[vgprValuC+121], v161 // accum partials

s_waitcnt vmcnt(8)                                 // wait C (interleaved) 8 = 40 - 31 - 1
v_add_f32 v[vgprValuC+122], v[vgprValuC+122], v162 // accum partials

s_waitcnt vmcnt(7)                                 // wait C (interleaved) 7 = 40 - 32 - 1
v_add_f32 v[vgprValuC+123], v[vgprValuC+123], v163 // accum partials

s_waitcnt vmcnt(6)                                 // wait C (interleaved) 6 = 40 - 33 - 1
v_add_f32 v[vgprValuC+124], v[vgprValuC+124], v164 // accum partials

s_waitcnt vmcnt(5)                                 // wait C (interleaved) 5 = 40 - 34 - 1
v_add_f32 v[vgprValuC+125], v[vgprValuC+125], v165 // accum partials

s_waitcnt vmcnt(4)                                 // wait C (interleaved) 4 = 40 - 35 - 1
v_add_f32 v[vgprValuC+126], v[vgprValuC+126], v166 // accum partials

s_waitcnt vmcnt(3)                                 // wait C (interleaved) 3 = 40 - 36 - 1
v_add_f32 v[vgprValuC+127], v[vgprValuC+127], v167 // accum partials

s_waitcnt vmcnt(2)                                 // wait C (interleaved) 2 = 40 - 37 - 1
v_add_f32 v[vgprValuC+128], v[vgprValuC+128], v168 // accum partials

s_waitcnt vmcnt(1)                                 // wait C (interleaved) 1 = 40 - 38 - 1
v_add_f32 v[vgprValuC+129], v[vgprValuC+129], v169 // accum partials

s_waitcnt vmcnt(0)                                 // wait C (interleaved) 0 = 40 - 39 - 1
v_add_f32 v[vgprValuC+130], v[vgprValuC+130], v170 // accum partials
v_mov_b32 v[vgprValuC+0], v[vgprValuC+91]          // copy vreg[0] to MI out reg
v_mov_b32 v[vgprValuC+8], v[vgprValuC+92]          // copy vreg[1] to MI out reg
v_mov_b32 v[vgprValuC+16], v[vgprValuC+93]         // copy vreg[2] to MI out reg
v_mov_b32 v[vgprValuC+24], v[vgprValuC+94]         // copy vreg[3] to MI out reg
v_mov_b32 v[vgprValuC+32], v[vgprValuC+95]         // copy vreg[4] to MI out reg
v_mov_b32 v[vgprValuC+1], v[vgprValuC+96]          // copy vreg[5] to MI out reg
v_mov_b32 v[vgprValuC+9], v[vgprValuC+97]          // copy vreg[6] to MI out reg
v_mov_b32 v[vgprValuC+17], v[vgprValuC+98]         // copy vreg[7] to MI out reg
v_mov_b32 v[vgprValuC+25], v[vgprValuC+99]         // copy vreg[8] to MI out reg
v_mov_b32 v[vgprValuC+33], v[vgprValuC+100]        // copy vreg[9] to MI out reg
v_mov_b32 v[vgprValuC+2], v[vgprValuC+101]         // copy vreg[10] to MI out reg
v_mov_b32 v[vgprValuC+10], v[vgprValuC+102]        // copy vreg[11] to MI out reg
v_mov_b32 v[vgprValuC+18], v[vgprValuC+103]        // copy vreg[12] to MI out reg
v_mov_b32 v[vgprValuC+26], v[vgprValuC+104]        // copy vreg[13] to MI out reg
v_mov_b32 v[vgprValuC+34], v[vgprValuC+105]        // copy vreg[14] to MI out reg
v_mov_b32 v[vgprValuC+3], v[vgprValuC+106]         // copy vreg[15] to MI out reg
v_mov_b32 v[vgprValuC+11], v[vgprValuC+107]        // copy vreg[16] to MI out reg
v_mov_b32 v[vgprValuC+19], v[vgprValuC+108]        // copy vreg[17] to MI out reg
v_mov_b32 v[vgprValuC+27], v[vgprValuC+109]        // copy vreg[18] to MI out reg
v_mov_b32 v[vgprValuC+35], v[vgprValuC+110]        // copy vreg[19] to MI out reg
v_mov_b32 v[vgprValuC+4], v[vgprValuC+111]         // copy vreg[20] to MI out reg
v_mov_b32 v[vgprValuC+12], v[vgprValuC+112]        // copy vreg[21] to MI out reg
v_mov_b32 v[vgprValuC+20], v[vgprValuC+113]        // copy vreg[22] to MI out reg
v_mov_b32 v[vgprValuC+28], v[vgprValuC+114]        // copy vreg[23] to MI out reg
v_mov_b32 v[vgprValuC+36], v[vgprValuC+115]        // copy vreg[24] to MI out reg
v_mov_b32 v[vgprValuC+5], v[vgprValuC+116]         // copy vreg[25] to MI out reg
v_mov_b32 v[vgprValuC+13], v[vgprValuC+117]        // copy vreg[26] to MI out reg
v_mov_b32 v[vgprValuC+21], v[vgprValuC+118]        // copy vreg[27] to MI out reg
v_mov_b32 v[vgprValuC+29], v[vgprValuC+119]        // copy vreg[28] to MI out reg
v_mov_b32 v[vgprValuC+37], v[vgprValuC+120]        // copy vreg[29] to MI out reg
v_mov_b32 v[vgprValuC+6], v[vgprValuC+121]         // copy vreg[30] to MI out reg
v_mov_b32 v[vgprValuC+14], v[vgprValuC+122]        // copy vreg[31] to MI out reg
v_mov_b32 v[vgprValuC+22], v[vgprValuC+123]        // copy vreg[32] to MI out reg
v_mov_b32 v[vgprValuC+30], v[vgprValuC+124]        // copy vreg[33] to MI out reg
v_mov_b32 v[vgprValuC+38], v[vgprValuC+125]        // copy vreg[34] to MI out reg
v_mov_b32 v[vgprValuC+7], v[vgprValuC+126]         // copy vreg[35] to MI out reg
v_mov_b32 v[vgprValuC+15], v[vgprValuC+127]        // copy vreg[36] to MI out reg
v_mov_b32 v[vgprValuC+23], v[vgprValuC+128]        // copy vreg[37] to MI out reg
v_mov_b32 v[vgprValuC+31], v[vgprValuC+129]        // copy vreg[38] to MI out reg
v_mov_b32 v[vgprValuC+39], v[vgprValuC+130]        // copy vreg[39] to MI out reg
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_mul_i32 s11, s[sgprskTiles], s[sgprItersPerTile]
s_mul_i32 s53, s[sgprSKItersPerWG], s[sgprskGrid]
s_sub_u32 s11, s11, s53                            // skTiles * ItersPerTile - SKItersPerWG * skGrid
s_bitcmp1_b32 s[sgprMagicShiftItersPerTile], 29    // USO on? (bit 29 of MagicShiftItersPerTile); off -> historical global peer size
s_cbranch_scc0 label_SK_PeerGlobal                 // USO off -> historical global mapping
s_cmp_eq_u32 s[sgprskTiles], 0                     // skTiles == 0?
s_cbranch_scc1 label_SK_PeerNoTiles                // global peer size
v_cvt_f32_u32 v89, s[sgprskTiles]                  // skGrid % skTiles
v_rcp_iflag_f32 v89, v89                           // skGrid % skTiles
v_mov_b32 v90, 0x4f7ffffe                          // 2^32 - 512
v_mul_f32 v89, v90, v89                            // 2^32 / divisor
v_cvt_u32_f32 v89, v89                             // skGrid % skTiles
v_mul_lo_u32 v90, v89, s[sgprskTiles]              // refine 2^32 / divisor
v_sub_nc_u32 v90, 0, v90                           // refine 2^32 / divisor
v_mul_hi_u32 v90, v89, v90                         // refine 2^32 / divisor
v_add_nc_u32 v89, v89, v90                         // refine 2^32 / divisor
v_mul_hi_u32 v89, s[sgprskGrid], v89               // skGrid % skTiles
v_mul_lo_u32 v90, v89, s[sgprskTiles]              // skGrid % skTiles
v_sub_nc_u32 v90, s[sgprskGrid], v90               // skGrid % skTiles
v_cmp_ge_u32 vcc_lo, v90, s[sgprskTiles]           // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v89, 1, v89                           // quotient + 1
v_sub_nc_u32 v90, v90, s[sgprskTiles]              // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v90, s[sgprskTiles]           // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v89, 1, v89                           // quotient + 1
v_sub_nc_u32 v90, v90, s[sgprskTiles]              // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s53, v89                       // quotient
v_readfirstlane_b32 s53, v90                       // remainder
s_cmp_eq_u32 s53, 0                                // skGrid % skTiles == 0?
s_cbranch_scc0 label_SK_PeerGlobal                 // ragged -> global peer
label_SK_PeerPerTile:
v_cvt_f32_u32 v89, s[sgprskTiles]                  // F = skGrid / skTiles
v_rcp_iflag_f32 v89, v89                           // F = skGrid / skTiles
v_mov_b32 v90, 0x4f7ffffe                          // 2^32 - 512
v_mul_f32 v89, v90, v89                            // 2^32 / divisor
v_cvt_u32_f32 v89, v89                             // F = skGrid / skTiles
v_mul_lo_u32 v90, v89, s[sgprskTiles]              // refine 2^32 / divisor
v_sub_nc_u32 v90, 0, v90                           // refine 2^32 / divisor
v_mul_hi_u32 v90, v89, v90                         // refine 2^32 / divisor
v_add_nc_u32 v89, v89, v90                         // refine 2^32 / divisor
v_mul_hi_u32 v89, s[sgprskGrid], v89               // F = skGrid / skTiles
v_mul_lo_u32 v90, v89, s[sgprskTiles]              // F = skGrid / skTiles
v_sub_nc_u32 v90, s[sgprskGrid], v90               // F = skGrid / skTiles
v_cmp_ge_u32 vcc_lo, v90, s[sgprskTiles]           // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v89, 1, v89                           // quotient + 1
v_sub_nc_u32 v90, v90, s[sgprskTiles]              // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v90, s[sgprskTiles]           // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v89, 1, v89                           // quotient + 1
v_sub_nc_u32 v90, v90, s[sgprskTiles]              // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s53, v89                       // quotient
s_mov_b32 s11, s53                                 // F = skGrid / skTiles
v_cvt_f32_u32 v89, s11                             // s = cta % F
v_rcp_iflag_f32 v89, v89                           // s = cta % F
v_mov_b32 v90, 0x4f7ffffe                          // 2^32 - 512
v_mul_f32 v89, v90, v89                            // 2^32 / divisor
v_cvt_u32_f32 v89, v89                             // s = cta % F
v_mul_lo_u32 v90, v89, s11                         // refine 2^32 / divisor
v_sub_nc_u32 v90, 0, v90                           // refine 2^32 / divisor
v_mul_hi_u32 v90, v89, v90                         // refine 2^32 / divisor
v_add_nc_u32 v89, v89, v90                         // refine 2^32 / divisor
v_mul_hi_u32 v89, s7, v89                          // s = cta % F
v_mul_lo_u32 v90, v89, s11                         // s = cta % F
v_sub_nc_u32 v90, s7, v90                          // s = cta % F
v_cmp_ge_u32 vcc_lo, v90, s11                      // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v89, 1, v89                           // quotient + 1
v_sub_nc_u32 v90, v90, s11                         // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v90, s11                      // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v89, 1, v89                           // quotient + 1
v_sub_nc_u32 v90, v90, s11                         // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s53, v89                       // quotient
v_readfirstlane_b32 s53, v90                       // remainder
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

/* Edge/NonEdge store path check (M): Size % 160 > 0 -> Edge store; else -> NonEdge store */
s_mul_hi_u32 s71, s[sgprSizeI], 53687092           // STATIC_DIV: divisor=160
s_mul_i32 s70, s[sgprSizeI], 53687092              // tmp = dividend * magic
s_lshr_b64 s[70:71], s[70:71], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s69, s70                                 // quotient
s_mul_i32 s70, s69, 160                            // quotient*divisor
s_sub_u32 s68, s[sgprSizeI], s70                   // rReg = dividend - quotient*divisor
s_add_u32 s69, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s69                // wg0 >= nwg0-1 ?
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_GSU1_Else       // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 64 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s68, 63, s[sgprSizeJ]                    // s68 = s[sgprSizeJ] % 64
s_add_u32 s69, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s69                // wg1 >= nwg1-1
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_GSU1_Then       // jump if edges required
label_GW_B0_FD0_VW1_GSU1_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=164 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,4,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (1,4,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (2,4,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1); (3,4,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (4,3,0,0:vw1); (4,4,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (5,3,0,0:vw1); (5,4,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (6,3,0,0:vw1); (6,4,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (7,3,0,0:vw1); (7,4,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
/* (d1,vc1,d0,vc0)=(1,0,4,0) */
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
/* (d1,vc1,d0,vc0)=(2,0,4,0) */
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
/* (d1,vc1,d0,vc0)=(3,0,4,0) */
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
/* (d1,vc1,d0,vc0)=(4,0,2,0) */
/* (d1,vc1,d0,vc0)=(4,0,3,0) */
/* (d1,vc1,d0,vc0)=(4,0,4,0) */
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
/* (d1,vc1,d0,vc0)=(5,0,2,0) */
/* (d1,vc1,d0,vc0)=(5,0,3,0) */
/* (d1,vc1,d0,vc0)=(5,0,4,0) */
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
/* (d1,vc1,d0,vc0)=(6,0,2,0) */
/* (d1,vc1,d0,vc0)=(6,0,3,0) */
/* (d1,vc1,d0,vc0)=(6,0,4,0) */
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
/* (d1,vc1,d0,vc0)=(7,0,2,0) */
/* (d1,vc1,d0,vc0)=(7,0,3,0) */
/* (d1,vc1,d0,vc0)=(7,0,4,0) */
v_add_lshl_u32 v89, v81, v78, 1                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=78, coord0Vgpr=78 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (1, 4, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (2, 3, 0, 0), (2, 4, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (3, 3, 0, 0), (3, 4, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0), (4, 3, 0, 0), (4, 4, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (5, 2, 0, 0), (5, 3, 0, 0), (5, 4, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (6, 3, 0, 0), (6, 4, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (7, 2, 0, 0), (7, 3, 0, 0), (7, 4, 0, 0)] */
v_mul_f32 v[vgprValuC+91], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+92], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+93], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+94], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+95], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+96], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+97], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+98], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+100], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+102], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+103], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+104], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+105], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+106], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+107], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+108], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+109], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+110], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+111], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+112], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+128], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+129], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+130], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v86, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v87, 0x7fff0000                          // fp32 Nan
v_mov_b32 v88, 0x7fff                              // rounding bias for bfloat16
v_cmp_u_f32 s8, v[vgprValuC+91], v[vgprValuC+91]   // check Nan
v_bfe_u32 v85, v[vgprValuC+91], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+91], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+91], v85, v87, s8
v_lshrrev_b32 v91, 16, v[vgprValuC+91]             // convert C to bf16
buffer_store_b16 v91, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+92], v[vgprValuC+92]   // check Nan
v_bfe_u32 v85, v[vgprValuC+92], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+92], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+92], v85, v87, s8
v_lshrrev_b32 v92, 16, v[vgprValuC+92]             // convert C to bf16
buffer_store_b16 v92, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+93], v[vgprValuC+93]   // check Nan
v_bfe_u32 v85, v[vgprValuC+93], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+93], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+93], v85, v87, s8
v_lshrrev_b32 v93, 16, v[vgprValuC+93]             // convert C to bf16
buffer_store_b16 v93, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+94], v[vgprValuC+94]   // check Nan
v_bfe_u32 v85, v[vgprValuC+94], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+94], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+94], v85, v87, s8
v_lshrrev_b32 v94, 16, v[vgprValuC+94]             // convert C to bf16
buffer_store_b16 v94, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D
v_cmp_u_f32 s8, v[vgprValuC+95], v[vgprValuC+95]   // check Nan
v_bfe_u32 v85, v[vgprValuC+95], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+95], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+95], v85, v87, s8
v_lshrrev_b32 v95, 16, v[vgprValuC+95]             // convert C to bf16
buffer_store_b16 v95, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+96], v[vgprValuC+96]   // check Nan
v_bfe_u32 v85, v[vgprValuC+96], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+96], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+96], v85, v87, s8
v_lshrrev_b32 v96, 16, v[vgprValuC+96]             // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v96, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+97], v[vgprValuC+97]   // check Nan
v_bfe_u32 v85, v[vgprValuC+97], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+97], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+97], v85, v87, s8
v_lshrrev_b32 v97, 16, v[vgprValuC+97]             // convert C to bf16
buffer_store_b16 v97, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+98], v[vgprValuC+98]   // check Nan
v_bfe_u32 v85, v[vgprValuC+98], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+98], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+98], v85, v87, s8
v_lshrrev_b32 v98, 16, v[vgprValuC+98]             // convert C to bf16
buffer_store_b16 v98, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+99], v[vgprValuC+99]   // check Nan
v_bfe_u32 v85, v[vgprValuC+99], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+99], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+99], v85, v87, s8
v_lshrrev_b32 v99, 16, v[vgprValuC+99]             // convert C to bf16
buffer_store_b16 v99, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D
v_cmp_u_f32 s8, v[vgprValuC+100], v[vgprValuC+100] // check Nan
v_bfe_u32 v85, v[vgprValuC+100], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+100], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+100], v85, v87, s8
v_lshrrev_b32 v100, 16, v[vgprValuC+100]           // convert C to bf16
buffer_store_b16 v100, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+101], v[vgprValuC+101] // check Nan
v_bfe_u32 v85, v[vgprValuC+101], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+101], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+101], v85, v87, s8
v_lshrrev_b32 v101, 16, v[vgprValuC+101]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v101, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+102], v[vgprValuC+102] // check Nan
v_bfe_u32 v85, v[vgprValuC+102], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+102], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+102], v85, v87, s8
v_lshrrev_b32 v102, 16, v[vgprValuC+102]           // convert C to bf16
buffer_store_b16 v102, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+103], v[vgprValuC+103] // check Nan
v_bfe_u32 v85, v[vgprValuC+103], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+103], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+103], v85, v87, s8
v_lshrrev_b32 v103, 16, v[vgprValuC+103]           // convert C to bf16
buffer_store_b16 v103, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+104], v[vgprValuC+104] // check Nan
v_bfe_u32 v85, v[vgprValuC+104], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+104], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+104], v85, v87, s8
v_lshrrev_b32 v104, 16, v[vgprValuC+104]           // convert C to bf16
buffer_store_b16 v104, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D
v_cmp_u_f32 s8, v[vgprValuC+105], v[vgprValuC+105] // check Nan
v_bfe_u32 v85, v[vgprValuC+105], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+105], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+105], v85, v87, s8
v_lshrrev_b32 v105, 16, v[vgprValuC+105]           // convert C to bf16
buffer_store_b16 v105, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+106], v[vgprValuC+106] // check Nan
v_bfe_u32 v85, v[vgprValuC+106], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+106], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+106], v85, v87, s8
v_lshrrev_b32 v106, 16, v[vgprValuC+106]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v106, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+107], v[vgprValuC+107] // check Nan
v_bfe_u32 v85, v[vgprValuC+107], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+107], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+107], v85, v87, s8
v_lshrrev_b32 v107, 16, v[vgprValuC+107]           // convert C to bf16
buffer_store_b16 v107, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+108], v[vgprValuC+108] // check Nan
v_bfe_u32 v85, v[vgprValuC+108], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+108], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+108], v85, v87, s8
v_lshrrev_b32 v108, 16, v[vgprValuC+108]           // convert C to bf16
buffer_store_b16 v108, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+109], v[vgprValuC+109] // check Nan
v_bfe_u32 v85, v[vgprValuC+109], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+109], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+109], v85, v87, s8
v_lshrrev_b32 v109, 16, v[vgprValuC+109]           // convert C to bf16
buffer_store_b16 v109, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D
v_cmp_u_f32 s8, v[vgprValuC+110], v[vgprValuC+110] // check Nan
v_bfe_u32 v85, v[vgprValuC+110], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+110], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+110], v85, v87, s8
v_lshrrev_b32 v110, 16, v[vgprValuC+110]           // convert C to bf16
buffer_store_b16 v110, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+111], v[vgprValuC+111] // check Nan
v_bfe_u32 v85, v[vgprValuC+111], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+111], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+111], v85, v87, s8
v_lshrrev_b32 v111, 16, v[vgprValuC+111]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v111, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+112], v[vgprValuC+112] // check Nan
v_bfe_u32 v85, v[vgprValuC+112], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+112], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+112], v85, v87, s8
v_lshrrev_b32 v112, 16, v[vgprValuC+112]           // convert C to bf16
buffer_store_b16 v112, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+113], v[vgprValuC+113] // check Nan
v_bfe_u32 v85, v[vgprValuC+113], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+113], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+113], v85, v87, s8
v_lshrrev_b32 v113, 16, v[vgprValuC+113]           // convert C to bf16
buffer_store_b16 v113, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+114], v[vgprValuC+114] // check Nan
v_bfe_u32 v85, v[vgprValuC+114], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+114], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+114], v85, v87, s8
v_lshrrev_b32 v114, 16, v[vgprValuC+114]           // convert C to bf16
buffer_store_b16 v114, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D
v_cmp_u_f32 s8, v[vgprValuC+115], v[vgprValuC+115] // check Nan
v_bfe_u32 v85, v[vgprValuC+115], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+115], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+115], v85, v87, s8
v_lshrrev_b32 v115, 16, v[vgprValuC+115]           // convert C to bf16
buffer_store_b16 v115, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+116], v[vgprValuC+116] // check Nan
v_bfe_u32 v85, v[vgprValuC+116], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+116], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+116], v85, v87, s8
v_lshrrev_b32 v116, 16, v[vgprValuC+116]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v116, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+117], v[vgprValuC+117] // check Nan
v_bfe_u32 v85, v[vgprValuC+117], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+117], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+117], v85, v87, s8
v_lshrrev_b32 v117, 16, v[vgprValuC+117]           // convert C to bf16
buffer_store_b16 v117, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+118], v[vgprValuC+118] // check Nan
v_bfe_u32 v85, v[vgprValuC+118], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+118], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+118], v85, v87, s8
v_lshrrev_b32 v118, 16, v[vgprValuC+118]           // convert C to bf16
buffer_store_b16 v118, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+119], v[vgprValuC+119] // check Nan
v_bfe_u32 v85, v[vgprValuC+119], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+119], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+119], v85, v87, s8
v_lshrrev_b32 v119, 16, v[vgprValuC+119]           // convert C to bf16
buffer_store_b16 v119, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D
v_cmp_u_f32 s8, v[vgprValuC+120], v[vgprValuC+120] // check Nan
v_bfe_u32 v85, v[vgprValuC+120], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+120], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+120], v85, v87, s8
v_lshrrev_b32 v120, 16, v[vgprValuC+120]           // convert C to bf16
buffer_store_b16 v120, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+121], v[vgprValuC+121] // check Nan
v_bfe_u32 v85, v[vgprValuC+121], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+121], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+121], v85, v87, s8
v_lshrrev_b32 v121, 16, v[vgprValuC+121]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v121, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+122], v[vgprValuC+122] // check Nan
v_bfe_u32 v85, v[vgprValuC+122], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+122], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+122], v85, v87, s8
v_lshrrev_b32 v122, 16, v[vgprValuC+122]           // convert C to bf16
buffer_store_b16 v122, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+123], v[vgprValuC+123] // check Nan
v_bfe_u32 v85, v[vgprValuC+123], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+123], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+123], v85, v87, s8
v_lshrrev_b32 v123, 16, v[vgprValuC+123]           // convert C to bf16
buffer_store_b16 v123, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+124], v[vgprValuC+124] // check Nan
v_bfe_u32 v85, v[vgprValuC+124], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+124], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+124], v85, v87, s8
v_lshrrev_b32 v124, 16, v[vgprValuC+124]           // convert C to bf16
buffer_store_b16 v124, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D
v_cmp_u_f32 s8, v[vgprValuC+125], v[vgprValuC+125] // check Nan
v_bfe_u32 v85, v[vgprValuC+125], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+125], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+125], v85, v87, s8
v_lshrrev_b32 v125, 16, v[vgprValuC+125]           // convert C to bf16
buffer_store_b16 v125, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
v_cmp_u_f32 s8, v[vgprValuC+126], v[vgprValuC+126] // check Nan
v_bfe_u32 v85, v[vgprValuC+126], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+126], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+126], v85, v87, s8
v_lshrrev_b32 v126, 16, v[vgprValuC+126]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v126, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+127], v[vgprValuC+127] // check Nan
v_bfe_u32 v85, v[vgprValuC+127], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+127], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+127], v85, v87, s8
v_lshrrev_b32 v127, 16, v[vgprValuC+127]           // convert C to bf16
buffer_store_b16 v127, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cmp_u_f32 s8, v[vgprValuC+128], v[vgprValuC+128] // check Nan
v_bfe_u32 v85, v[vgprValuC+128], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+128], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+128], v85, v87, s8
v_lshrrev_b32 v128, 16, v[vgprValuC+128]           // convert C to bf16
buffer_store_b16 v128, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_cmp_u_f32 s8, v[vgprValuC+129], v[vgprValuC+129] // check Nan
v_bfe_u32 v85, v[vgprValuC+129], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+129], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+129], v85, v87, s8
v_lshrrev_b32 v129, 16, v[vgprValuC+129]           // convert C to bf16
buffer_store_b16 v129, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D
v_cmp_u_f32 s8, v[vgprValuC+130], v[vgprValuC+130] // check Nan
v_bfe_u32 v85, v[vgprValuC+130], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+130], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+130], v85, v87, s8
v_lshrrev_b32 v130, 16, v[vgprValuC+130]           // convert C to bf16
buffer_store_b16 v130, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW1_GSU1_NonEdgeEnd:
label_GW_B0_FD0_VW1_GSU1_Else:
label_GW_B0_FD0_VW1_GSU1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=82 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,4,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (1,4,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (2,4,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1); (3,4,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (4,3,0,0:vw1); (4,4,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (5,3,0,0:vw1); (5,4,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (6,3,0,0:vw1); (6,4,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (7,3,0,0:vw1); (7,4,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v84, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s68, v78, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v129, v81, v78, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v129, v84, v129, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v82, vcc_lo, v78, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v130, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v84, v130, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
v_add_co_u32 v82, vcc_lo, v78, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v131, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v131, v84, v131, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
s_mov_b32 s68, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v132, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v84, v132, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
s_mov_b32 s68, 128                                 // coordOffset0 d0=4 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v133, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v133, v84, v133, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v79, vcc_lo, v79, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v80, v80, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v81, v81, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v78, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v134, v81, v78, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v84, v134, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v82, vcc_lo, v78, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v135, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v135, v84, v135, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
v_add_co_u32 v82, vcc_lo, v78, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v136, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v84, v136, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
s_mov_b32 s68, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v137, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v137, v84, v137, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,4,0) */
s_mov_b32 s68, 128                                 // coordOffset0 d0=4 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v138, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v84, v138, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v79, vcc_lo, v79, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v80, v80, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v81, v81, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v78, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v139, v81, v78, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v139, v84, v139, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
v_add_co_u32 v82, vcc_lo, v78, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v140, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v84, v140, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
v_add_co_u32 v82, vcc_lo, v78, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v141, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v141, v84, v141, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
s_mov_b32 s68, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v142, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v84, v142, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,4,0) */
s_mov_b32 s68, 128                                 // coordOffset0 d0=4 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v143, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v143, v84, v143, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v79, vcc_lo, v79, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v80, v80, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v81, v81, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v78, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v144, v81, v78, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v144, v84, v144, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
v_add_co_u32 v82, vcc_lo, v78, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v145, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v145, v84, v145, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
v_add_co_u32 v82, vcc_lo, v78, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v146, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v146, v84, v146, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
s_mov_b32 s68, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v147, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v147, v84, v147, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,4,0) */
s_mov_b32 s68, 128                                 // coordOffset0 d0=4 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v148, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v148, v84, v148, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v79, vcc_lo, v79, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v80, v80, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v81, v81, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v78, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v149, v81, v78, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v149, v84, v149, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
v_add_co_u32 v82, vcc_lo, v78, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v150, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v150, v84, v150, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,2,0) */
v_add_co_u32 v82, vcc_lo, v78, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v151, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v151, v84, v151, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,3,0) */
s_mov_b32 s68, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v152, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v152, v84, v152, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,4,0) */
s_mov_b32 s68, 128                                 // coordOffset0 d0=4 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v153, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v153, v84, v153, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
v_add_co_u32 v79, vcc_lo, v79, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v80, v80, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v81, v81, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v78, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v154, v81, v78, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v84, v154, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
v_add_co_u32 v82, vcc_lo, v78, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v155, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v155, v84, v155, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,2,0) */
v_add_co_u32 v82, vcc_lo, v78, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v156, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v84, v156, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,3,0) */
s_mov_b32 s68, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v157, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v157, v84, v157, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,4,0) */
s_mov_b32 s68, 128                                 // coordOffset0 d0=4 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v158, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v84, v158, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
v_add_co_u32 v79, vcc_lo, v79, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v80, v80, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v81, v81, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v78, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v159, v81, v78, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v159, v84, v159, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
v_add_co_u32 v82, vcc_lo, v78, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v160, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v160, v84, v160, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,2,0) */
v_add_co_u32 v82, vcc_lo, v78, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v161, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v161, v84, v161, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,3,0) */
s_mov_b32 s68, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v162, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v162, v84, v162, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,4,0) */
s_mov_b32 s68, 128                                 // coordOffset0 d0=4 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v163, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v163, v84, v163, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
v_add_co_u32 v79, vcc_lo, v79, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v80, v80, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v81, v81, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v78, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v164, v81, v78, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v164, v84, v164, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
v_add_co_u32 v82, vcc_lo, v78, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v165, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v165, v84, v165, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,2,0) */
v_add_co_u32 v82, vcc_lo, v78, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v166, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v166, v84, v166, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,3,0) */
s_mov_b32 s68, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v167, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v167, v84, v167, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,4,0) */
s_mov_b32 s68, 128                                 // coordOffset0 d0=4 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v168, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v168, v84, v168, s70                 // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (1, 4, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (2, 3, 0, 0), (2, 4, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (3, 3, 0, 0), (3, 4, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0), (4, 3, 0, 0), (4, 4, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (5, 2, 0, 0), (5, 3, 0, 0), (5, 4, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (6, 3, 0, 0), (6, 4, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (7, 2, 0, 0), (7, 3, 0, 0), (7, 4, 0, 0)] */
v_mul_f32 v[vgprValuC+89], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+90], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+91], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+92], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+93], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+94], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+95], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+96], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+97], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+98], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+100], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+102], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+103], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+104], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+105], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+106], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+107], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+108], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+109], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+110], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+111], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+112], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+128], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v86, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v87, 0x7fff0000                          // fp32 Nan
v_mov_b32 v88, 0x7fff                              // rounding bias for bfloat16
v_cmp_u_f32 s68, v[vgprValuC+89], v[vgprValuC+89]  // check Nan
v_bfe_u32 v85, v[vgprValuC+89], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+89], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+89], v85, v87, s68
v_lshrrev_b32 v89, 16, v[vgprValuC+89]             // convert C to bf16
buffer_store_b16 v89, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+90], v[vgprValuC+90]  // check Nan
v_bfe_u32 v85, v[vgprValuC+90], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+90], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+90], v85, v87, s68
v_lshrrev_b32 v90, 16, v[vgprValuC+90]             // convert C to bf16
buffer_store_b16 v90, v130, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+91], v[vgprValuC+91]  // check Nan
v_bfe_u32 v85, v[vgprValuC+91], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+91], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+91], v85, v87, s68
v_lshrrev_b32 v91, 16, v[vgprValuC+91]             // convert C to bf16
buffer_store_b16 v91, v131, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+92], v[vgprValuC+92]  // check Nan
v_bfe_u32 v85, v[vgprValuC+92], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+92], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+92], v85, v87, s68
v_lshrrev_b32 v92, 16, v[vgprValuC+92]             // convert C to bf16
buffer_store_b16 v92, v132, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+93], v[vgprValuC+93]  // check Nan
v_bfe_u32 v85, v[vgprValuC+93], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+93], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+93], v85, v87, s68
v_lshrrev_b32 v93, 16, v[vgprValuC+93]             // convert C to bf16
buffer_store_b16 v93, v133, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+94], v[vgprValuC+94]  // check Nan
v_bfe_u32 v85, v[vgprValuC+94], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+94], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+94], v85, v87, s68
v_lshrrev_b32 v94, 16, v[vgprValuC+94]             // convert C to bf16
buffer_store_b16 v94, v134, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+95], v[vgprValuC+95]  // check Nan
v_bfe_u32 v85, v[vgprValuC+95], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+95], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+95], v85, v87, s68
v_lshrrev_b32 v95, 16, v[vgprValuC+95]             // convert C to bf16
buffer_store_b16 v95, v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+96], v[vgprValuC+96]  // check Nan
v_bfe_u32 v85, v[vgprValuC+96], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+96], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+96], v85, v87, s68
v_lshrrev_b32 v96, 16, v[vgprValuC+96]             // convert C to bf16
buffer_store_b16 v96, v136, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+97], v[vgprValuC+97]  // check Nan
v_bfe_u32 v85, v[vgprValuC+97], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+97], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+97], v85, v87, s68
v_lshrrev_b32 v97, 16, v[vgprValuC+97]             // convert C to bf16
buffer_store_b16 v97, v137, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+98], v[vgprValuC+98]  // check Nan
v_bfe_u32 v85, v[vgprValuC+98], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+98], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+98], v85, v87, s68
v_lshrrev_b32 v98, 16, v[vgprValuC+98]             // convert C to bf16
buffer_store_b16 v98, v138, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+99], v[vgprValuC+99]  // check Nan
v_bfe_u32 v85, v[vgprValuC+99], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+99], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+99], v85, v87, s68
v_lshrrev_b32 v99, 16, v[vgprValuC+99]             // convert C to bf16
buffer_store_b16 v99, v139, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+100], v[vgprValuC+100] // check Nan
v_bfe_u32 v85, v[vgprValuC+100], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+100], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+100], v85, v87, s68
v_lshrrev_b32 v100, 16, v[vgprValuC+100]           // convert C to bf16
buffer_store_b16 v100, v140, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+101], v[vgprValuC+101] // check Nan
v_bfe_u32 v85, v[vgprValuC+101], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+101], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+101], v85, v87, s68
v_lshrrev_b32 v101, 16, v[vgprValuC+101]           // convert C to bf16
buffer_store_b16 v101, v141, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+102], v[vgprValuC+102] // check Nan
v_bfe_u32 v85, v[vgprValuC+102], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+102], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+102], v85, v87, s68
v_lshrrev_b32 v102, 16, v[vgprValuC+102]           // convert C to bf16
buffer_store_b16 v102, v142, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+103], v[vgprValuC+103] // check Nan
v_bfe_u32 v85, v[vgprValuC+103], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+103], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+103], v85, v87, s68
v_lshrrev_b32 v103, 16, v[vgprValuC+103]           // convert C to bf16
buffer_store_b16 v103, v143, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+104], v[vgprValuC+104] // check Nan
v_bfe_u32 v85, v[vgprValuC+104], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+104], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+104], v85, v87, s68
v_lshrrev_b32 v104, 16, v[vgprValuC+104]           // convert C to bf16
buffer_store_b16 v104, v144, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+105], v[vgprValuC+105] // check Nan
v_bfe_u32 v85, v[vgprValuC+105], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+105], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+105], v85, v87, s68
v_lshrrev_b32 v105, 16, v[vgprValuC+105]           // convert C to bf16
buffer_store_b16 v105, v145, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+106], v[vgprValuC+106] // check Nan
v_bfe_u32 v85, v[vgprValuC+106], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+106], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+106], v85, v87, s68
v_lshrrev_b32 v106, 16, v[vgprValuC+106]           // convert C to bf16
buffer_store_b16 v106, v146, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+107], v[vgprValuC+107] // check Nan
v_bfe_u32 v85, v[vgprValuC+107], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+107], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+107], v85, v87, s68
v_lshrrev_b32 v107, 16, v[vgprValuC+107]           // convert C to bf16
buffer_store_b16 v107, v147, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+108], v[vgprValuC+108] // check Nan
v_bfe_u32 v85, v[vgprValuC+108], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+108], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+108], v85, v87, s68
v_lshrrev_b32 v108, 16, v[vgprValuC+108]           // convert C to bf16
buffer_store_b16 v108, v148, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+109], v[vgprValuC+109] // check Nan
v_bfe_u32 v85, v[vgprValuC+109], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+109], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+109], v85, v87, s68
v_lshrrev_b32 v109, 16, v[vgprValuC+109]           // convert C to bf16
buffer_store_b16 v109, v149, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+110], v[vgprValuC+110] // check Nan
v_bfe_u32 v85, v[vgprValuC+110], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+110], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+110], v85, v87, s68
v_lshrrev_b32 v110, 16, v[vgprValuC+110]           // convert C to bf16
buffer_store_b16 v110, v150, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+111], v[vgprValuC+111] // check Nan
v_bfe_u32 v85, v[vgprValuC+111], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+111], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+111], v85, v87, s68
v_lshrrev_b32 v111, 16, v[vgprValuC+111]           // convert C to bf16
buffer_store_b16 v111, v151, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+112], v[vgprValuC+112] // check Nan
v_bfe_u32 v85, v[vgprValuC+112], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+112], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+112], v85, v87, s68
v_lshrrev_b32 v112, 16, v[vgprValuC+112]           // convert C to bf16
buffer_store_b16 v112, v152, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+113], v[vgprValuC+113] // check Nan
v_bfe_u32 v85, v[vgprValuC+113], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+113], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+113], v85, v87, s68
v_lshrrev_b32 v113, 16, v[vgprValuC+113]           // convert C to bf16
buffer_store_b16 v113, v153, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+114], v[vgprValuC+114] // check Nan
v_bfe_u32 v85, v[vgprValuC+114], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+114], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+114], v85, v87, s68
v_lshrrev_b32 v114, 16, v[vgprValuC+114]           // convert C to bf16
buffer_store_b16 v114, v154, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+115], v[vgprValuC+115] // check Nan
v_bfe_u32 v85, v[vgprValuC+115], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+115], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+115], v85, v87, s68
v_lshrrev_b32 v115, 16, v[vgprValuC+115]           // convert C to bf16
buffer_store_b16 v115, v155, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+116], v[vgprValuC+116] // check Nan
v_bfe_u32 v85, v[vgprValuC+116], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+116], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+116], v85, v87, s68
v_lshrrev_b32 v116, 16, v[vgprValuC+116]           // convert C to bf16
buffer_store_b16 v116, v156, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+117], v[vgprValuC+117] // check Nan
v_bfe_u32 v85, v[vgprValuC+117], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+117], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+117], v85, v87, s68
v_lshrrev_b32 v117, 16, v[vgprValuC+117]           // convert C to bf16
buffer_store_b16 v117, v157, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+118], v[vgprValuC+118] // check Nan
v_bfe_u32 v85, v[vgprValuC+118], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+118], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+118], v85, v87, s68
v_lshrrev_b32 v118, 16, v[vgprValuC+118]           // convert C to bf16
buffer_store_b16 v118, v158, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+119], v[vgprValuC+119] // check Nan
v_bfe_u32 v85, v[vgprValuC+119], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+119], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+119], v85, v87, s68
v_lshrrev_b32 v119, 16, v[vgprValuC+119]           // convert C to bf16
buffer_store_b16 v119, v159, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+120], v[vgprValuC+120] // check Nan
v_bfe_u32 v85, v[vgprValuC+120], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+120], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+120], v85, v87, s68
v_lshrrev_b32 v120, 16, v[vgprValuC+120]           // convert C to bf16
buffer_store_b16 v120, v160, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+121], v[vgprValuC+121] // check Nan
v_bfe_u32 v85, v[vgprValuC+121], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+121], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+121], v85, v87, s68
v_lshrrev_b32 v121, 16, v[vgprValuC+121]           // convert C to bf16
buffer_store_b16 v121, v161, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+122], v[vgprValuC+122] // check Nan
v_bfe_u32 v85, v[vgprValuC+122], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+122], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+122], v85, v87, s68
v_lshrrev_b32 v122, 16, v[vgprValuC+122]           // convert C to bf16
buffer_store_b16 v122, v162, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+123], v[vgprValuC+123] // check Nan
v_bfe_u32 v85, v[vgprValuC+123], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+123], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+123], v85, v87, s68
v_lshrrev_b32 v123, 16, v[vgprValuC+123]           // convert C to bf16
buffer_store_b16 v123, v163, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+124], v[vgprValuC+124] // check Nan
v_bfe_u32 v85, v[vgprValuC+124], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+124], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+124], v85, v87, s68
v_lshrrev_b32 v124, 16, v[vgprValuC+124]           // convert C to bf16
buffer_store_b16 v124, v164, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+125], v[vgprValuC+125] // check Nan
v_bfe_u32 v85, v[vgprValuC+125], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+125], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+125], v85, v87, s68
v_lshrrev_b32 v125, 16, v[vgprValuC+125]           // convert C to bf16
buffer_store_b16 v125, v165, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+126], v[vgprValuC+126] // check Nan
v_bfe_u32 v85, v[vgprValuC+126], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+126], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+126], v85, v87, s68
v_lshrrev_b32 v126, 16, v[vgprValuC+126]           // convert C to bf16
buffer_store_b16 v126, v166, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+127], v[vgprValuC+127] // check Nan
v_bfe_u32 v85, v[vgprValuC+127], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+127], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+127], v85, v87, s68
v_lshrrev_b32 v127, 16, v[vgprValuC+127]           // convert C to bf16
buffer_store_b16 v127, v167, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s68, v[vgprValuC+128], v[vgprValuC+128] // check Nan
v_bfe_u32 v85, v[vgprValuC+128], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+128], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+128], v85, v87, s68
v_lshrrev_b32 v128, 16, v[vgprValuC+128]           // convert C to bf16
buffer_store_b16 v128, v168, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_GSU1:
label_GW_B1_FD0_GSU1:

/* Edge/NonEdge store path check (M): Size % 160 > 0 -> Edge store; else -> NonEdge store */
s_mul_hi_u32 s71, s[sgprSizeI], 53687092           // STATIC_DIV: divisor=160
s_mul_i32 s70, s[sgprSizeI], 53687092              // tmp = dividend * magic
s_lshr_b64 s[70:71], s[70:71], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s69, s70                                 // quotient
s_mul_i32 s70, s69, 160                            // quotient*divisor
s_sub_u32 s68, s[sgprSizeI], s70                   // rReg = dividend - quotient*divisor
s_add_u32 s69, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s69                // wg0 >= nwg0-1 ?
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW1_GSU1_Else       // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 64 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s68, 63, s[sgprSizeJ]                    // s68 = s[sgprSizeJ] % 64
s_add_u32 s69, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s69                // wg1 >= nwg1-1
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW1_GSU1_Then       // jump if edges required
label_GW_B1_FD0_VW1_GSU1_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=80 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (1, 4, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (2, 3, 0, 0), (2, 4, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (3, 3, 0, 0), (3, 4, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0), (4, 3, 0, 0), (4, 4, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (5, 2, 0, 0), (5, 3, 0, 0), (5, 4, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (6, 3, 0, 0), (6, 4, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (7, 2, 0, 0), (7, 3, 0, 0), (7, 4, 0, 0)] */
v_mul_f32 v[vgprValuC+91], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+92], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+93], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+94], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+95], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+96], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+97], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+98], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+100], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+102], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+103], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+104], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+105], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+106], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+107], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+108], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+109], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+110], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+111], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+112], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+128], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+129], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+130], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,4,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (1,4,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (2,4,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1); (3,4,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (4,3,0,0:vw1); (4,4,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (5,3,0,0:vw1); (5,4,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (6,3,0,0:vw1); (6,4,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (7,3,0,0:vw1); (7,4,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v90, v80, v78, 1                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=78, coord0Vgpr=78 (multiple bpe)
buffer_load_d16_b16 v131, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
buffer_load_d16_b16 v132, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
buffer_load_d16_b16 v133, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
buffer_load_d16_b16 v134, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:192 // load C
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
buffer_load_d16_b16 v135, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v136, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
buffer_load_d16_b16 v137, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
buffer_load_d16_b16 v138, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
buffer_load_d16_b16 v139, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:192 // load C
/* (d1,vc1,d0,vc0)=(1,0,4,0) */
buffer_load_d16_b16 v140, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v141, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
buffer_load_d16_b16 v142, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
buffer_load_d16_b16 v143, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
buffer_load_d16_b16 v144, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:192 // load C
/* (d1,vc1,d0,vc0)=(2,0,4,0) */
buffer_load_d16_b16 v145, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v146, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
buffer_load_d16_b16 v147, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
buffer_load_d16_b16 v148, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
buffer_load_d16_b16 v149, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:192 // load C
/* (d1,vc1,d0,vc0)=(3,0,4,0) */
buffer_load_d16_b16 v150, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v151, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
buffer_load_d16_b16 v152, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(4,0,2,0) */
buffer_load_d16_b16 v153, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(4,0,3,0) */
buffer_load_d16_b16 v154, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:192 // load C
/* (d1,vc1,d0,vc0)=(4,0,4,0) */
buffer_load_d16_b16 v155, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v156, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
buffer_load_d16_b16 v157, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(5,0,2,0) */
buffer_load_d16_b16 v158, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(5,0,3,0) */
buffer_load_d16_b16 v159, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:192 // load C
/* (d1,vc1,d0,vc0)=(5,0,4,0) */
buffer_load_d16_b16 v160, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v161, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
buffer_load_d16_b16 v162, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(6,0,2,0) */
buffer_load_d16_b16 v163, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(6,0,3,0) */
buffer_load_d16_b16 v164, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:192 // load C
/* (d1,vc1,d0,vc0)=(6,0,4,0) */
buffer_load_d16_b16 v165, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 4                  // scale StrideC *= numRows(2) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_load_d16_b16 v166, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
buffer_load_d16_b16 v167, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(7,0,2,0) */
buffer_load_d16_b16 v168, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(7,0,3,0) */
buffer_load_d16_b16 v169, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:192 // load C
/* (d1,vc1,d0,vc0)=(7,0,4,0) */
buffer_load_d16_b16 v170, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:256 // load C
v_add_lshl_u32 v89, v81, v78, 1                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=78, coord0Vgpr=78 (multiple bpe)

/* apply mask, calc new C and issue writes */
v_mov_b32 v86, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v87, 0x7fff0000                          // fp32 Nan
v_mov_b32 v88, 0x7fff                              // rounding bias for bfloat16

s_waitcnt vmcnt(39)                                // vlcnt(39) = 40 - 1 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v131                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+91], v82, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+91], v[vgprValuC+91]   // check Nan
v_bfe_u32 v85, v[vgprValuC+91], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+91], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+91], v85, v87, s8
v_lshrrev_b32 v91, 16, v[vgprValuC+91]             // convert C to bf16
buffer_store_b16 v91, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(38)                                // vlcnt(38) = 40 - 2 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v132                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+92], v82, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+92], v[vgprValuC+92]   // check Nan
v_bfe_u32 v85, v[vgprValuC+92], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+92], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+92], v85, v87, s8
v_lshrrev_b32 v92, 16, v[vgprValuC+92]             // convert C to bf16
buffer_store_b16 v92, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(37)                                // vlcnt(37) = 40 - 3 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v133                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+93], v82, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+93], v[vgprValuC+93]   // check Nan
v_bfe_u32 v85, v[vgprValuC+93], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+93], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+93], v85, v87, s8
v_lshrrev_b32 v93, 16, v[vgprValuC+93]             // convert C to bf16
buffer_store_b16 v93, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(36)                                // vlcnt(36) = 40 - 4 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v134                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+94], v82, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+94], v[vgprValuC+94]   // check Nan
v_bfe_u32 v85, v[vgprValuC+94], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+94], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+94], v85, v87, s8
v_lshrrev_b32 v94, 16, v[vgprValuC+94]             // convert C to bf16
buffer_store_b16 v94, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D

s_waitcnt vmcnt(35)                                // vlcnt(35) = 40 - 5 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v135                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+95], v82, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+95], v[vgprValuC+95]   // check Nan
v_bfe_u32 v85, v[vgprValuC+95], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+95], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+95], v85, v87, s8
v_lshrrev_b32 v95, 16, v[vgprValuC+95]             // convert C to bf16
buffer_store_b16 v95, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(34)                                // vlcnt(34) = 40 - 6 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v136                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+96], v82, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+96], v[vgprValuC+96]   // check Nan
v_bfe_u32 v85, v[vgprValuC+96], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+96], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+96], v85, v87, s8
v_lshrrev_b32 v96, 16, v[vgprValuC+96]             // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v96, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(33)                                // vlcnt(33) = 40 - 7 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v137                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+97], v82, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+97], v[vgprValuC+97]   // check Nan
v_bfe_u32 v85, v[vgprValuC+97], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+97], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+97], v85, v87, s8
v_lshrrev_b32 v97, 16, v[vgprValuC+97]             // convert C to bf16
buffer_store_b16 v97, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(32)                                // vlcnt(32) = 40 - 8 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v138                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+98], v82, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+98], v[vgprValuC+98]   // check Nan
v_bfe_u32 v85, v[vgprValuC+98], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+98], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+98], v85, v87, s8
v_lshrrev_b32 v98, 16, v[vgprValuC+98]             // convert C to bf16
buffer_store_b16 v98, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(31)                                // vlcnt(31) = 40 - 9 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v139                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+99], v82, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+99], v[vgprValuC+99]   // check Nan
v_bfe_u32 v85, v[vgprValuC+99], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+99], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+99], v85, v87, s8
v_lshrrev_b32 v99, 16, v[vgprValuC+99]             // convert C to bf16
buffer_store_b16 v99, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D

s_waitcnt vmcnt(30)                                // vlcnt(30) = 40 - 10 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v140                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+100], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+100], v[vgprValuC+100] // check Nan
v_bfe_u32 v85, v[vgprValuC+100], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+100], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+100], v85, v87, s8
v_lshrrev_b32 v100, 16, v[vgprValuC+100]           // convert C to bf16
buffer_store_b16 v100, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(29)                                // vlcnt(29) = 40 - 11 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v141                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+101], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+101], v[vgprValuC+101] // check Nan
v_bfe_u32 v85, v[vgprValuC+101], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+101], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+101], v85, v87, s8
v_lshrrev_b32 v101, 16, v[vgprValuC+101]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v101, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(28)                                // vlcnt(28) = 40 - 12 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v142                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+102], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+102], v[vgprValuC+102] // check Nan
v_bfe_u32 v85, v[vgprValuC+102], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+102], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+102], v85, v87, s8
v_lshrrev_b32 v102, 16, v[vgprValuC+102]           // convert C to bf16
buffer_store_b16 v102, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(27)                                // vlcnt(27) = 40 - 13 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v143                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+103], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+103], v[vgprValuC+103] // check Nan
v_bfe_u32 v85, v[vgprValuC+103], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+103], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+103], v85, v87, s8
v_lshrrev_b32 v103, 16, v[vgprValuC+103]           // convert C to bf16
buffer_store_b16 v103, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(26)                                // vlcnt(26) = 40 - 14 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v144                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+104], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+104], v[vgprValuC+104] // check Nan
v_bfe_u32 v85, v[vgprValuC+104], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+104], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+104], v85, v87, s8
v_lshrrev_b32 v104, 16, v[vgprValuC+104]           // convert C to bf16
buffer_store_b16 v104, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D

s_waitcnt vmcnt(25)                                // vlcnt(25) = 40 - 15 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v145                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+105], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+105], v[vgprValuC+105] // check Nan
v_bfe_u32 v85, v[vgprValuC+105], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+105], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+105], v85, v87, s8
v_lshrrev_b32 v105, 16, v[vgprValuC+105]           // convert C to bf16
buffer_store_b16 v105, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(24)                                // vlcnt(24) = 40 - 16 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v146                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+106], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+106], v[vgprValuC+106] // check Nan
v_bfe_u32 v85, v[vgprValuC+106], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+106], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+106], v85, v87, s8
v_lshrrev_b32 v106, 16, v[vgprValuC+106]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v106, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(23)                                // vlcnt(23) = 40 - 17 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v147                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+107], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+107], v[vgprValuC+107] // check Nan
v_bfe_u32 v85, v[vgprValuC+107], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+107], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+107], v85, v87, s8
v_lshrrev_b32 v107, 16, v[vgprValuC+107]           // convert C to bf16
buffer_store_b16 v107, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(22)                                // vlcnt(22) = 40 - 18 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v148                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+108], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+108], v[vgprValuC+108] // check Nan
v_bfe_u32 v85, v[vgprValuC+108], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+108], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+108], v85, v87, s8
v_lshrrev_b32 v108, 16, v[vgprValuC+108]           // convert C to bf16
buffer_store_b16 v108, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(21)                                // vlcnt(21) = 40 - 19 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v149                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+109], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+109], v[vgprValuC+109] // check Nan
v_bfe_u32 v85, v[vgprValuC+109], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+109], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+109], v85, v87, s8
v_lshrrev_b32 v109, 16, v[vgprValuC+109]           // convert C to bf16
buffer_store_b16 v109, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D

s_waitcnt vmcnt(20)                                // vlcnt(20) = 40 - 20 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v150                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+110], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+110], v[vgprValuC+110] // check Nan
v_bfe_u32 v85, v[vgprValuC+110], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+110], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+110], v85, v87, s8
v_lshrrev_b32 v110, 16, v[vgprValuC+110]           // convert C to bf16
buffer_store_b16 v110, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(19)                                // vlcnt(19) = 40 - 21 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v151                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+111], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+111], v[vgprValuC+111] // check Nan
v_bfe_u32 v85, v[vgprValuC+111], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+111], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+111], v85, v87, s8
v_lshrrev_b32 v111, 16, v[vgprValuC+111]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v111, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(18)                                // vlcnt(18) = 40 - 22 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v152                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+112], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+112], v[vgprValuC+112] // check Nan
v_bfe_u32 v85, v[vgprValuC+112], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+112], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+112], v85, v87, s8
v_lshrrev_b32 v112, 16, v[vgprValuC+112]           // convert C to bf16
buffer_store_b16 v112, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(17)                                // vlcnt(17) = 40 - 23 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v153                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+113], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+113], v[vgprValuC+113] // check Nan
v_bfe_u32 v85, v[vgprValuC+113], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+113], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+113], v85, v87, s8
v_lshrrev_b32 v113, 16, v[vgprValuC+113]           // convert C to bf16
buffer_store_b16 v113, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(16)                                // vlcnt(16) = 40 - 24 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v154                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+114], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+114], v[vgprValuC+114] // check Nan
v_bfe_u32 v85, v[vgprValuC+114], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+114], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+114], v85, v87, s8
v_lshrrev_b32 v114, 16, v[vgprValuC+114]           // convert C to bf16
buffer_store_b16 v114, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D

s_waitcnt vmcnt(15)                                // vlcnt(15) = 40 - 25 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v155                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+115], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+115], v[vgprValuC+115] // check Nan
v_bfe_u32 v85, v[vgprValuC+115], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+115], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+115], v85, v87, s8
v_lshrrev_b32 v115, 16, v[vgprValuC+115]           // convert C to bf16
buffer_store_b16 v115, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(14)                                // vlcnt(14) = 40 - 26 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v156                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+116], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+116], v[vgprValuC+116] // check Nan
v_bfe_u32 v85, v[vgprValuC+116], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+116], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+116], v85, v87, s8
v_lshrrev_b32 v116, 16, v[vgprValuC+116]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v116, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(13)                                // vlcnt(13) = 40 - 27 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v157                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+117], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+117], v[vgprValuC+117] // check Nan
v_bfe_u32 v85, v[vgprValuC+117], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+117], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+117], v85, v87, s8
v_lshrrev_b32 v117, 16, v[vgprValuC+117]           // convert C to bf16
buffer_store_b16 v117, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(12)                                // vlcnt(12) = 40 - 28 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v158                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+118], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+118], v[vgprValuC+118] // check Nan
v_bfe_u32 v85, v[vgprValuC+118], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+118], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+118], v85, v87, s8
v_lshrrev_b32 v118, 16, v[vgprValuC+118]           // convert C to bf16
buffer_store_b16 v118, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(11)                                // vlcnt(11) = 40 - 29 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v159                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+119], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+119], v[vgprValuC+119] // check Nan
v_bfe_u32 v85, v[vgprValuC+119], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+119], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+119], v85, v87, s8
v_lshrrev_b32 v119, 16, v[vgprValuC+119]           // convert C to bf16
buffer_store_b16 v119, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D

s_waitcnt vmcnt(10)                                // vlcnt(10) = 40 - 30 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v160                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+120], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+120], v[vgprValuC+120] // check Nan
v_bfe_u32 v85, v[vgprValuC+120], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+120], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+120], v85, v87, s8
v_lshrrev_b32 v120, 16, v[vgprValuC+120]           // convert C to bf16
buffer_store_b16 v120, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(9)                                 // vlcnt(9) = 40 - 31 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v161                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+121], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+121], v[vgprValuC+121] // check Nan
v_bfe_u32 v85, v[vgprValuC+121], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+121], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+121], v85, v87, s8
v_lshrrev_b32 v121, 16, v[vgprValuC+121]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v121, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(8)                                 // vlcnt(8) = 40 - 32 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v162                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+122], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+122], v[vgprValuC+122] // check Nan
v_bfe_u32 v85, v[vgprValuC+122], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+122], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+122], v85, v87, s8
v_lshrrev_b32 v122, 16, v[vgprValuC+122]           // convert C to bf16
buffer_store_b16 v122, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(7)                                 // vlcnt(7) = 40 - 33 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v163                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+123], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+123], v[vgprValuC+123] // check Nan
v_bfe_u32 v85, v[vgprValuC+123], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+123], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+123], v85, v87, s8
v_lshrrev_b32 v123, 16, v[vgprValuC+123]           // convert C to bf16
buffer_store_b16 v123, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(6)                                 // vlcnt(6) = 40 - 34 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v164                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+124], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+124], v[vgprValuC+124] // check Nan
v_bfe_u32 v85, v[vgprValuC+124], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+124], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+124], v85, v87, s8
v_lshrrev_b32 v124, 16, v[vgprValuC+124]           // convert C to bf16
buffer_store_b16 v124, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D

s_waitcnt vmcnt(5)                                 // vlcnt(5) = 40 - 35 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v165                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+125], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+125], v[vgprValuC+125] // check Nan
v_bfe_u32 v85, v[vgprValuC+125], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+125], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+125], v85, v87, s8
v_lshrrev_b32 v125, 16, v[vgprValuC+125]           // convert C to bf16
buffer_store_b16 v125, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D

s_waitcnt vmcnt(4)                                 // vlcnt(4) = 40 - 36 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v166                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+126], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+126], v[vgprValuC+126] // check Nan
v_bfe_u32 v85, v[vgprValuC+126], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+126], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+126], v85, v87, s8
v_lshrrev_b32 v126, 16, v[vgprValuC+126]           // convert C to bf16
s_mul_i32 s8, s[sgprStrideD1J], 4                  // scale StrideD *= numRows(2) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(2): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(2): gra SRD += inc(upper)
buffer_store_b16 v126, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(3)                                 // vlcnt(3) = 40 - 37 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v167                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+127], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+127], v[vgprValuC+127] // check Nan
v_bfe_u32 v85, v[vgprValuC+127], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+127], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+127], v85, v87, s8
v_lshrrev_b32 v127, 16, v[vgprValuC+127]           // convert C to bf16
buffer_store_b16 v127, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(2)                                 // vlcnt(2) = 40 - 38 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v168                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+128], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+128], v[vgprValuC+128] // check Nan
v_bfe_u32 v85, v[vgprValuC+128], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+128], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+128], v85, v87, s8
v_lshrrev_b32 v128, 16, v[vgprValuC+128]           // convert C to bf16
buffer_store_b16 v128, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(1)                                 // vlcnt(1) = 40 - 39 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v169                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+129], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+129], v[vgprValuC+129] // check Nan
v_bfe_u32 v85, v[vgprValuC+129], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+129], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+129], v85, v87, s8
v_lshrrev_b32 v129, 16, v[vgprValuC+129]           // convert C to bf16
buffer_store_b16 v129, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D

s_waitcnt vmcnt(0)                                 // vlcnt(0) = 40 - 40 (beta) (interleaved)
v_lshlrev_b32 v82, 16, v170                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+130], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+130], v[vgprValuC+130] // check Nan
v_bfe_u32 v85, v[vgprValuC+130], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+130], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+130], v85, v87, s8
v_lshrrev_b32 v130, 16, v[vgprValuC+130]           // convert C to bf16
buffer_store_b16 v130, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_FD0_VW1_GSU1_NonEdgeEnd:
label_GW_B1_FD0_VW1_GSU1_Else:
label_GW_B1_FD0_VW1_GSU1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=54 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (1, 4, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (2, 3, 0, 0), (2, 4, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (3, 3, 0, 0), (3, 4, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0), (4, 3, 0, 0), (4, 4, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (5, 2, 0, 0), (5, 3, 0, 0), (5, 4, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (6, 3, 0, 0), (6, 4, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (7, 2, 0, 0), (7, 3, 0, 0), (7, 4, 0, 0)] */
v_mul_f32 v[vgprValuC+89], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+90], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+91], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+92], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+93], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+94], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+95], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+96], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+97], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+98], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+100], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+102], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+103], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+104], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+105], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+106], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+107], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+108], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+109], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+110], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+111], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+112], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+128], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,4,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (1,4,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (2,4,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1); (3,4,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (4,3,0,0:vw1); (4,4,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (5,3,0,0:vw1); (5,4,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (6,3,0,0:vw1); (6,4,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (7,3,0,0:vw1); (7,4,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v84, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s68, v78, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v130, v80, v78, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v84, v130, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v129, v130, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v130, v81, v78, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v84, v130, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v82, vcc_lo, v78, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v132, v80, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v84, v132, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v131, v132, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v132, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v84, v132, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
v_add_co_u32 v82, vcc_lo, v78, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v134, v80, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v84, v134, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v133, v134, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v134, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v84, v134, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
s_mov_b32 s68, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v136, v80, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v84, v136, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v135, v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v136, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v84, v136, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
s_mov_b32 s68, 128                                 // coordOffset0 d0=4 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v138, v80, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v84, v138, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v137, v138, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v138, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v84, v138, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v79, vcc_lo, v79, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v80, v80, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v81, v81, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v78, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v140, v80, v78, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v84, v140, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v139, v140, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v140, v81, v78, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v84, v140, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v82, vcc_lo, v78, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v142, v80, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v84, v142, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v141, v142, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v142, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v84, v142, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
v_add_co_u32 v82, vcc_lo, v78, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v144, v80, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v144, v84, v144, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v143, v144, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v144, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v144, v84, v144, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
s_mov_b32 s68, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v146, v80, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v146, v84, v146, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v145, v146, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v146, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v146, v84, v146, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,4,0) */
s_mov_b32 s68, 128                                 // coordOffset0 d0=4 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v148, v80, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v148, v84, v148, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v147, v148, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v148, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v148, v84, v148, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v79, vcc_lo, v79, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v80, v80, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v81, v81, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v78, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v150, v80, v78, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v150, v84, v150, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v149, v150, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v150, v81, v78, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v150, v84, v150, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
v_add_co_u32 v82, vcc_lo, v78, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v152, v80, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v152, v84, v152, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v151, v152, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v152, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v152, v84, v152, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
v_add_co_u32 v82, vcc_lo, v78, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v154, v80, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v84, v154, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v153, v154, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v154, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v84, v154, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
s_mov_b32 s68, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v156, v80, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v84, v156, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v155, v156, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v156, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v84, v156, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,4,0) */
s_mov_b32 s68, 128                                 // coordOffset0 d0=4 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v158, v80, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v84, v158, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v157, v158, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v158, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v84, v158, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v79, vcc_lo, v79, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v80, v80, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v81, v81, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v78, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v160, v80, v78, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v160, v84, v160, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v159, v160, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v160, v81, v78, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v160, v84, v160, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
v_add_co_u32 v82, vcc_lo, v78, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v162, v80, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v162, v84, v162, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v161, v162, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v162, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v162, v84, v162, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
v_add_co_u32 v82, vcc_lo, v78, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v164, v80, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v164, v84, v164, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v163, v164, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v164, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v164, v84, v164, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
s_mov_b32 s68, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v166, v80, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v166, v84, v166, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v165, v166, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v166, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v166, v84, v166, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,4,0) */
s_mov_b32 s68, 128                                 // coordOffset0 d0=4 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v168, v80, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v168, v84, v168, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v167, v168, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v168, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v168, v84, v168, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v79, vcc_lo, v79, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v80, v80, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v81, v81, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v78, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v170, v80, v78, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v170, v84, v170, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v169, v170, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v170, v81, v78, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v170, v84, v170, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
v_add_co_u32 v82, vcc_lo, v78, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v172, v80, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v172, v84, v172, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v171, v172, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v172, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v172, v84, v172, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,2,0) */
v_add_co_u32 v82, vcc_lo, v78, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v174, v80, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v174, v84, v174, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v173, v174, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v174, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v174, v84, v174, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,3,0) */
s_mov_b32 s68, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v176, v80, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v176, v84, v176, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v175, v176, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v176, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v176, v84, v176, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,4,0) */
s_mov_b32 s68, 128                                 // coordOffset0 d0=4 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v178, v80, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v178, v84, v178, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v177, v178, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v178, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v178, v84, v178, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
v_add_co_u32 v79, vcc_lo, v79, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v80, v80, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v81, v81, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v78, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v180, v80, v78, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v180, v84, v180, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v179, v180, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v180, v81, v78, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v180, v84, v180, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
v_add_co_u32 v82, vcc_lo, v78, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v182, v80, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v182, v84, v182, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v181, v182, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v182, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v182, v84, v182, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,2,0) */
v_add_co_u32 v82, vcc_lo, v78, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v184, v80, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v184, v84, v184, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v183, v184, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v184, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v184, v84, v184, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,3,0) */
s_mov_b32 s68, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v186, v80, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v186, v84, v186, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v185, v186, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v186, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v186, v84, v186, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,4,0) */
s_mov_b32 s68, 128                                 // coordOffset0 d0=4 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v188, v80, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v188, v84, v188, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v187, v188, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v188, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v188, v84, v188, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
v_add_co_u32 v79, vcc_lo, v79, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v80, v80, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v81, v81, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v78, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v190, v80, v78, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v190, v84, v190, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v189, v190, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v190, v81, v78, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v190, v84, v190, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
v_add_co_u32 v82, vcc_lo, v78, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v192, v80, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v192, v84, v192, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v191, v192, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v192, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v192, v84, v192, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,2,0) */
v_add_co_u32 v82, vcc_lo, v78, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v194, v80, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v194, v84, v194, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v193, v194, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v194, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v194, v84, v194, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,3,0) */
s_mov_b32 s68, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v196, v80, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v196, v84, v196, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v195, v196, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v196, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v196, v84, v196, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,4,0) */
s_mov_b32 s68, 128                                 // coordOffset0 d0=4 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v198, v80, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v198, v84, v198, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v197, v198, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v198, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v198, v84, v198, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
v_add_co_u32 v79, vcc_lo, v79, 2                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 2                 // scale stride
v_add_nc_i32 v80, v80, s68                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 2                 // scale stride
v_add_nc_i32 v81, v81, s68                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s68, v78, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v200, v80, v78, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v200, v84, v200, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v199, v200, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v200, v81, v78, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v200, v84, v200, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
v_add_co_u32 v82, vcc_lo, v78, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v202, v80, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v202, v84, v202, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v201, v202, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v202, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v202, v84, v202, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,2,0) */
v_add_co_u32 v82, vcc_lo, v78, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v204, v80, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v204, v84, v204, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v203, v204, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v204, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v204, v84, v204, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,3,0) */
s_mov_b32 s68, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v206, v80, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v206, v84, v206, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v205, v206, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v206, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v206, v84, v206, s70                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,4,0) */
s_mov_b32 s68, 128                                 // coordOffset0 d0=4 vc0=0
v_add_co_u32 v82, vcc_lo, v78, s68                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s68, v82, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s70, v79, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s70, s68, s70                            // in0 && in1
v_add_lshl_u32 v208, v80, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v208, v84, v208, s70                 // LDC clip if OOB. offset
buffer_load_d16_b16 v207, v208, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v208, v81, v82, 1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v208, v84, v208, s70                 // LDD clip if OOB. offset
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_mov_b32 v86, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v87, 0x7fff0000                          // fp32 Nan
v_mov_b32 v88, 0x7fff                              // rounding bias for bfloat16
v_lshlrev_b32 v82, 16, v129                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+89], v82, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+89], v[vgprValuC+89]  // check Nan
v_bfe_u32 v85, v[vgprValuC+89], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+89], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+89], v85, v87, s68
v_lshrrev_b32 v89, 16, v[vgprValuC+89]             // convert C to bf16
buffer_store_b16 v89, v130, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v131                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+90], v82, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+90], v[vgprValuC+90]  // check Nan
v_bfe_u32 v85, v[vgprValuC+90], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+90], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+90], v85, v87, s68
v_lshrrev_b32 v90, 16, v[vgprValuC+90]             // convert C to bf16
buffer_store_b16 v90, v132, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v133                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+91], v82, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+91], v[vgprValuC+91]  // check Nan
v_bfe_u32 v85, v[vgprValuC+91], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+91], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+91], v85, v87, s68
v_lshrrev_b32 v91, 16, v[vgprValuC+91]             // convert C to bf16
buffer_store_b16 v91, v134, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v135                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+92], v82, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+92], v[vgprValuC+92]  // check Nan
v_bfe_u32 v85, v[vgprValuC+92], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+92], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+92], v85, v87, s68
v_lshrrev_b32 v92, 16, v[vgprValuC+92]             // convert C to bf16
buffer_store_b16 v92, v136, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v137                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+93], v82, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+93], v[vgprValuC+93]  // check Nan
v_bfe_u32 v85, v[vgprValuC+93], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+93], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+93], v85, v87, s68
v_lshrrev_b32 v93, 16, v[vgprValuC+93]             // convert C to bf16
buffer_store_b16 v93, v138, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v139                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+94], v82, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+94], v[vgprValuC+94]  // check Nan
v_bfe_u32 v85, v[vgprValuC+94], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+94], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+94], v85, v87, s68
v_lshrrev_b32 v94, 16, v[vgprValuC+94]             // convert C to bf16
buffer_store_b16 v94, v140, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v141                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+95], v82, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+95], v[vgprValuC+95]  // check Nan
v_bfe_u32 v85, v[vgprValuC+95], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+95], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+95], v85, v87, s68
v_lshrrev_b32 v95, 16, v[vgprValuC+95]             // convert C to bf16
buffer_store_b16 v95, v142, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v143                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+96], v82, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+96], v[vgprValuC+96]  // check Nan
v_bfe_u32 v85, v[vgprValuC+96], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+96], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+96], v85, v87, s68
v_lshrrev_b32 v96, 16, v[vgprValuC+96]             // convert C to bf16
buffer_store_b16 v96, v144, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v145                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+97], v82, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+97], v[vgprValuC+97]  // check Nan
v_bfe_u32 v85, v[vgprValuC+97], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+97], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+97], v85, v87, s68
v_lshrrev_b32 v97, 16, v[vgprValuC+97]             // convert C to bf16
buffer_store_b16 v97, v146, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v147                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+98], v82, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+98], v[vgprValuC+98]  // check Nan
v_bfe_u32 v85, v[vgprValuC+98], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+98], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+98], v85, v87, s68
v_lshrrev_b32 v98, 16, v[vgprValuC+98]             // convert C to bf16
buffer_store_b16 v98, v148, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v149                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+99], v82, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+99], v[vgprValuC+99]  // check Nan
v_bfe_u32 v85, v[vgprValuC+99], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+99], v85, v88          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+99], v85, v87, s68
v_lshrrev_b32 v99, 16, v[vgprValuC+99]             // convert C to bf16
buffer_store_b16 v99, v150, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v151                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+100], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+100], v[vgprValuC+100] // check Nan
v_bfe_u32 v85, v[vgprValuC+100], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+100], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+100], v85, v87, s68
v_lshrrev_b32 v100, 16, v[vgprValuC+100]           // convert C to bf16
buffer_store_b16 v100, v152, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v153                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+101], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+101], v[vgprValuC+101] // check Nan
v_bfe_u32 v85, v[vgprValuC+101], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+101], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+101], v85, v87, s68
v_lshrrev_b32 v101, 16, v[vgprValuC+101]           // convert C to bf16
buffer_store_b16 v101, v154, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v155                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+102], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+102], v[vgprValuC+102] // check Nan
v_bfe_u32 v85, v[vgprValuC+102], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+102], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+102], v85, v87, s68
v_lshrrev_b32 v102, 16, v[vgprValuC+102]           // convert C to bf16
buffer_store_b16 v102, v156, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v157                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+103], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+103], v[vgprValuC+103] // check Nan
v_bfe_u32 v85, v[vgprValuC+103], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+103], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+103], v85, v87, s68
v_lshrrev_b32 v103, 16, v[vgprValuC+103]           // convert C to bf16
buffer_store_b16 v103, v158, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v159                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+104], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+104], v[vgprValuC+104] // check Nan
v_bfe_u32 v85, v[vgprValuC+104], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+104], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+104], v85, v87, s68
v_lshrrev_b32 v104, 16, v[vgprValuC+104]           // convert C to bf16
buffer_store_b16 v104, v160, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v161                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+105], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+105], v[vgprValuC+105] // check Nan
v_bfe_u32 v85, v[vgprValuC+105], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+105], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+105], v85, v87, s68
v_lshrrev_b32 v105, 16, v[vgprValuC+105]           // convert C to bf16
buffer_store_b16 v105, v162, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v163                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+106], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+106], v[vgprValuC+106] // check Nan
v_bfe_u32 v85, v[vgprValuC+106], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+106], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+106], v85, v87, s68
v_lshrrev_b32 v106, 16, v[vgprValuC+106]           // convert C to bf16
buffer_store_b16 v106, v164, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v165                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+107], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+107], v[vgprValuC+107] // check Nan
v_bfe_u32 v85, v[vgprValuC+107], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+107], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+107], v85, v87, s68
v_lshrrev_b32 v107, 16, v[vgprValuC+107]           // convert C to bf16
buffer_store_b16 v107, v166, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v167                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+108], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+108], v[vgprValuC+108] // check Nan
v_bfe_u32 v85, v[vgprValuC+108], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+108], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+108], v85, v87, s68
v_lshrrev_b32 v108, 16, v[vgprValuC+108]           // convert C to bf16
buffer_store_b16 v108, v168, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v169                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+109], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+109], v[vgprValuC+109] // check Nan
v_bfe_u32 v85, v[vgprValuC+109], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+109], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+109], v85, v87, s68
v_lshrrev_b32 v109, 16, v[vgprValuC+109]           // convert C to bf16
buffer_store_b16 v109, v170, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v171                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+110], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+110], v[vgprValuC+110] // check Nan
v_bfe_u32 v85, v[vgprValuC+110], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+110], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+110], v85, v87, s68
v_lshrrev_b32 v110, 16, v[vgprValuC+110]           // convert C to bf16
buffer_store_b16 v110, v172, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v173                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+111], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+111], v[vgprValuC+111] // check Nan
v_bfe_u32 v85, v[vgprValuC+111], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+111], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+111], v85, v87, s68
v_lshrrev_b32 v111, 16, v[vgprValuC+111]           // convert C to bf16
buffer_store_b16 v111, v174, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v175                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+112], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+112], v[vgprValuC+112] // check Nan
v_bfe_u32 v85, v[vgprValuC+112], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+112], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+112], v85, v87, s68
v_lshrrev_b32 v112, 16, v[vgprValuC+112]           // convert C to bf16
buffer_store_b16 v112, v176, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v177                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+113], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+113], v[vgprValuC+113] // check Nan
v_bfe_u32 v85, v[vgprValuC+113], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+113], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+113], v85, v87, s68
v_lshrrev_b32 v113, 16, v[vgprValuC+113]           // convert C to bf16
buffer_store_b16 v113, v178, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v179                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+114], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+114], v[vgprValuC+114] // check Nan
v_bfe_u32 v85, v[vgprValuC+114], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+114], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+114], v85, v87, s68
v_lshrrev_b32 v114, 16, v[vgprValuC+114]           // convert C to bf16
buffer_store_b16 v114, v180, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v181                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+115], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+115], v[vgprValuC+115] // check Nan
v_bfe_u32 v85, v[vgprValuC+115], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+115], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+115], v85, v87, s68
v_lshrrev_b32 v115, 16, v[vgprValuC+115]           // convert C to bf16
buffer_store_b16 v115, v182, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v183                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+116], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+116], v[vgprValuC+116] // check Nan
v_bfe_u32 v85, v[vgprValuC+116], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+116], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+116], v85, v87, s68
v_lshrrev_b32 v116, 16, v[vgprValuC+116]           // convert C to bf16
buffer_store_b16 v116, v184, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v185                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+117], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+117], v[vgprValuC+117] // check Nan
v_bfe_u32 v85, v[vgprValuC+117], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+117], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+117], v85, v87, s68
v_lshrrev_b32 v117, 16, v[vgprValuC+117]           // convert C to bf16
buffer_store_b16 v117, v186, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v187                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+118], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+118], v[vgprValuC+118] // check Nan
v_bfe_u32 v85, v[vgprValuC+118], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+118], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+118], v85, v87, s68
v_lshrrev_b32 v118, 16, v[vgprValuC+118]           // convert C to bf16
buffer_store_b16 v118, v188, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v189                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+119], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+119], v[vgprValuC+119] // check Nan
v_bfe_u32 v85, v[vgprValuC+119], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+119], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+119], v85, v87, s68
v_lshrrev_b32 v119, 16, v[vgprValuC+119]           // convert C to bf16
buffer_store_b16 v119, v190, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v191                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+120], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+120], v[vgprValuC+120] // check Nan
v_bfe_u32 v85, v[vgprValuC+120], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+120], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+120], v85, v87, s68
v_lshrrev_b32 v120, 16, v[vgprValuC+120]           // convert C to bf16
buffer_store_b16 v120, v192, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v193                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+121], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+121], v[vgprValuC+121] // check Nan
v_bfe_u32 v85, v[vgprValuC+121], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+121], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+121], v85, v87, s68
v_lshrrev_b32 v121, 16, v[vgprValuC+121]           // convert C to bf16
buffer_store_b16 v121, v194, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v195                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+122], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+122], v[vgprValuC+122] // check Nan
v_bfe_u32 v85, v[vgprValuC+122], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+122], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+122], v85, v87, s68
v_lshrrev_b32 v122, 16, v[vgprValuC+122]           // convert C to bf16
buffer_store_b16 v122, v196, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v197                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+123], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+123], v[vgprValuC+123] // check Nan
v_bfe_u32 v85, v[vgprValuC+123], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+123], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+123], v85, v87, s68
v_lshrrev_b32 v123, 16, v[vgprValuC+123]           // convert C to bf16
buffer_store_b16 v123, v198, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v199                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+124], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+124], v[vgprValuC+124] // check Nan
v_bfe_u32 v85, v[vgprValuC+124], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+124], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+124], v85, v87, s68
v_lshrrev_b32 v124, 16, v[vgprValuC+124]           // convert C to bf16
buffer_store_b16 v124, v200, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v201                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+125], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+125], v[vgprValuC+125] // check Nan
v_bfe_u32 v85, v[vgprValuC+125], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+125], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+125], v85, v87, s68
v_lshrrev_b32 v125, 16, v[vgprValuC+125]           // convert C to bf16
buffer_store_b16 v125, v202, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v203                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+126], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+126], v[vgprValuC+126] // check Nan
v_bfe_u32 v85, v[vgprValuC+126], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+126], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+126], v85, v87, s68
v_lshrrev_b32 v126, 16, v[vgprValuC+126]           // convert C to bf16
buffer_store_b16 v126, v204, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v205                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+127], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+127], v[vgprValuC+127] // check Nan
v_bfe_u32 v85, v[vgprValuC+127], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+127], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+127], v85, v87, s68
v_lshrrev_b32 v127, 16, v[vgprValuC+127]           // convert C to bf16
buffer_store_b16 v127, v206, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v82, 16, v207                        // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+128], v82, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s68, v[vgprValuC+128], v[vgprValuC+128] // check Nan
v_bfe_u32 v85, v[vgprValuC+128], 16, 1             // Non-Nan case: store lsb of bf16
v_add3_u32 v85, v[vgprValuC+128], v85, v88         // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+128], v85, v87, s68
v_lshrrev_b32 v128, 16, v[vgprValuC+128]           // convert C to bf16
buffer_store_b16 v128, v208, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
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

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=164 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Partials Write Batch #0 (d1,d0,vc1,vc0) = */
/*      (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,4,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (1,4,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (2,4,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1); (3,4,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (4,3,0,0:vw1); (4,4,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (5,3,0,0:vw1); (5,4,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (6,3,0,0:vw1); (6,4,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (7,3,0,0:vw1); (7,4,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v[vgprValuC+91], v[vgprValuC+0]          // copy MI out reg to vreg[0]
v_mov_b32 v[vgprValuC+92], v[vgprValuC+8]          // copy MI out reg to vreg[1]
v_mov_b32 v[vgprValuC+93], v[vgprValuC+16]         // copy MI out reg to vreg[2]
v_mov_b32 v[vgprValuC+94], v[vgprValuC+24]         // copy MI out reg to vreg[3]
v_mov_b32 v[vgprValuC+95], v[vgprValuC+32]         // copy MI out reg to vreg[4]
v_mov_b32 v[vgprValuC+96], v[vgprValuC+1]          // copy MI out reg to vreg[5]
v_mov_b32 v[vgprValuC+97], v[vgprValuC+9]          // copy MI out reg to vreg[6]
v_mov_b32 v[vgprValuC+98], v[vgprValuC+17]         // copy MI out reg to vreg[7]
v_mov_b32 v[vgprValuC+99], v[vgprValuC+25]         // copy MI out reg to vreg[8]
v_mov_b32 v[vgprValuC+100], v[vgprValuC+33]        // copy MI out reg to vreg[9]
v_mov_b32 v[vgprValuC+101], v[vgprValuC+2]         // copy MI out reg to vreg[10]
v_mov_b32 v[vgprValuC+102], v[vgprValuC+10]        // copy MI out reg to vreg[11]
v_mov_b32 v[vgprValuC+103], v[vgprValuC+18]        // copy MI out reg to vreg[12]
v_mov_b32 v[vgprValuC+104], v[vgprValuC+26]        // copy MI out reg to vreg[13]
v_mov_b32 v[vgprValuC+105], v[vgprValuC+34]        // copy MI out reg to vreg[14]
v_mov_b32 v[vgprValuC+106], v[vgprValuC+3]         // copy MI out reg to vreg[15]
v_mov_b32 v[vgprValuC+107], v[vgprValuC+11]        // copy MI out reg to vreg[16]
v_mov_b32 v[vgprValuC+108], v[vgprValuC+19]        // copy MI out reg to vreg[17]
v_mov_b32 v[vgprValuC+109], v[vgprValuC+27]        // copy MI out reg to vreg[18]
v_mov_b32 v[vgprValuC+110], v[vgprValuC+35]        // copy MI out reg to vreg[19]
v_mov_b32 v[vgprValuC+111], v[vgprValuC+4]         // copy MI out reg to vreg[20]
v_mov_b32 v[vgprValuC+112], v[vgprValuC+12]        // copy MI out reg to vreg[21]
v_mov_b32 v[vgprValuC+113], v[vgprValuC+20]        // copy MI out reg to vreg[22]
v_mov_b32 v[vgprValuC+114], v[vgprValuC+28]        // copy MI out reg to vreg[23]
v_mov_b32 v[vgprValuC+115], v[vgprValuC+36]        // copy MI out reg to vreg[24]
v_mov_b32 v[vgprValuC+116], v[vgprValuC+5]         // copy MI out reg to vreg[25]
v_mov_b32 v[vgprValuC+117], v[vgprValuC+13]        // copy MI out reg to vreg[26]
v_mov_b32 v[vgprValuC+118], v[vgprValuC+21]        // copy MI out reg to vreg[27]
v_mov_b32 v[vgprValuC+119], v[vgprValuC+29]        // copy MI out reg to vreg[28]
v_mov_b32 v[vgprValuC+120], v[vgprValuC+37]        // copy MI out reg to vreg[29]
v_mov_b32 v[vgprValuC+121], v[vgprValuC+6]         // copy MI out reg to vreg[30]
v_mov_b32 v[vgprValuC+122], v[vgprValuC+14]        // copy MI out reg to vreg[31]
v_mov_b32 v[vgprValuC+123], v[vgprValuC+22]        // copy MI out reg to vreg[32]
v_mov_b32 v[vgprValuC+124], v[vgprValuC+30]        // copy MI out reg to vreg[33]
v_mov_b32 v[vgprValuC+125], v[vgprValuC+38]        // copy MI out reg to vreg[34]
v_mov_b32 v[vgprValuC+126], v[vgprValuC+7]         // copy MI out reg to vreg[35]
v_mov_b32 v[vgprValuC+127], v[vgprValuC+15]        // copy MI out reg to vreg[36]
v_mov_b32 v[vgprValuC+128], v[vgprValuC+23]        // copy MI out reg to vreg[37]
v_mov_b32 v[vgprValuC+129], v[vgprValuC+31]        // copy MI out reg to vreg[38]
v_mov_b32 v[vgprValuC+130], v[vgprValuC+39]        // copy MI out reg to vreg[39]

/* apply mask, calc new C and issue writes */
v_mov_b32 v86, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v87, 0x7fff0000                          // fp32 Nan
v_mov_b32 v88, 0x7fff                              // rounding bias for bfloat16
v_lshlrev_b32 v89, 2, v[vgprSerial]                // v89 = v[vgprSerial] * 4
s_mov_b32 s8, 0                                    // Init sgpr offset
buffer_store_b32 v91, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v92, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v93, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v94, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v95, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v96, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v97, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v98, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v99, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v100, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v101, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v102, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v103, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v104, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v105, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v106, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v107, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v108, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v109, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v110, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v111, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v112, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v113, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v114, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v115, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v116, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v117, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v118, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v119, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v120, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v121, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v122, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v123, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v124, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v125, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v126, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v127, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v128, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v129, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_b32 v130, v89, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 glc slc dlc // addStore
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_waitcnt_vscnt null 0                             // wait for data store
s_barrier                                          // store all data before setting flag
s_lshl_b32 s8, s[sgprPersistentWorkGroupIndex], 2  // flag offset based on physical partial slot
v_readfirstlane_b32 s7, v[vgprSerial]              // Wave 0 updates flags
s_cmp_eq_u32 s7, 0                                 // Check for wave 0
s_cbranch_scc0 label_SK_SkipFlagSet                // Skip flag set
s_mov_b32 s7, 1                                    // flag data
v_mov_b32 v91, s7                                  // move flag value to vgpr
v_mov_b32 v92, 0                                   // zero vaddr offset
s_mov_b64 s[68:69], s[sgprAddressFlags:sgprAddressFlags+1]
s_mov_b32 s70, BufferOOB
s_mov_b32 s71, Srd127_96
buffer_store_b32 v91, v92, s[68:71], s8 offen offset:0 glc dlc // set flag
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
