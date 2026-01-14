// Custom kernel using LDS triple buffer
// LDS A : 128*(128+4) = 16896 bytes
// LDS B : 128*(128+4) = 16896 bytes
// Total : 33792 bytes
// Triple buffering : 33792*3=101376
.set LDSSize, 101376
.set LDSBufferSize, 33792 
.set LDSBufferOffsetB, 16896
.set TotalLDSBufferSize, 3*LDSBufferSize 


/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.protected Custom_Cijk_Ailk_Bljk_BBS_BH_Bias_SAV_UserArgs_MT128x128x64_MI16x16x1_UserArgs_shortname1_gfx950
.globl Custom_Cijk_Ailk_Bljk_BBS_BH_Bias_SAV_UserArgs_MT128x128x64_MI16x16x1_UserArgs_shortname1_gfx950
.p2align 8
.type Custom_Cijk_Ailk_Bljk_BBS_BH_Bias_SAV_UserArgs_MT128x128x64_MI16x16x1_UserArgs_shortname1_gfx950,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Custom_Cijk_Ailk_Bljk_BBS_BH_Bias_SAV_UserArgs_MT128x128x64_MI16x16x1_UserArgs_shortname1_gfx950
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_accum_offset 256 // accvgpr offset
  .amdhsa_next_free_vgpr 320 // vgprs
  .amdhsa_next_free_sgpr 94 // sgprs
  .amdhsa_group_segment_fixed_size LDSSize // lds bytes
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
  .amdhsa_user_sgpr_count 13
  .amdhsa_user_sgpr_kernarg_preload_length 11
  .amdhsa_user_sgpr_kernarg_preload_offset 0
.end_amdhsa_kernel
.text
/* Num VGPR   =256 */
/* Num AccVGPR=64 */
/* Num SGPR   =94 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 16 x 4 */
/* SubGroup= 8 x 32 */
/* VectorWidthA=1 */
/* VectorWidthB=4 */
/* GlobalReadVectorWidthA=8, GlobalReadVectorWidthB=8 */
/* DirectToLdsA=True */
/* DirectToLdsB=True */
/* UseSgprForGRO=0 */
.amdgpu_metadata
---
custom.config:
  InternalSupportParams:
    KernArgsVersion: 2
amdhsa.version:
  - 1
  - 1
amdhsa.kernels:
  - .name: Custom_Cijk_Ailk_Bljk_BBS_BH_Bias_SAV_UserArgs_MT128x128x64_MI16x16x1_UserArgs_shortname1_gfx950
    .symbol: 'Custom_Cijk_Ailk_Bljk_BBS_BH_Bias_SAV_UserArgs_MT128x128x64_MI16x16x1_UserArgs_shortname1_gfx950.kd'
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
      - .name:            D
        .size:            8
        .offset:          32
        .value_kind:      global_buffer
        .value_type:      bf16
        .address_space:   generic
      - .name:            C
        .size:            8
        .offset:          40
        .value_kind:      global_buffer
        .value_type:      bf16
        .address_space:   generic
      - .name:            A
        .size:            8
        .offset:          48
        .value_kind:      global_buffer
        .value_type:      bf16
        .address_space:   generic
      - .name:            B
        .size:            8
        .offset:          56
        .value_kind:      global_buffer
        .value_type:      bf16
        .address_space:   generic
      - .name:            AddressWS
        .size:            8
        .offset:          64
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            AddressFlags
        .size:            8
        .offset:          72
        .value_kind:      global_buffer
        .value_type:      bf16
        .address_space:   generic
      - .name:            strideD0
        .size:            4
        .offset:          80
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideD1
        .size:            4
        .offset:          84
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC0
        .size:            4
        .offset:          88
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC1
        .size:            4
        .offset:          92
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA0
        .size:            4
        .offset:          96
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA1
        .size:            4
        .offset:          100
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB0
        .size:            4
        .offset:          104
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB1
        .size:            4
        .offset:          108
        .value_kind:      by_value
        .value_type:      u32
      - .name:            alpha
        .size:            4
        .offset:          112
        .value_kind:      by_value
        .value_type:      f32
      - .name:            beta
        .size:            4
        .offset:          116
        .value_kind:      by_value
        .value_type:      f32
      - .name:            ItersPerTile
        .size:            4
        .offset:          120
        .value_kind:      by_value
        .value_type:      u32
      - .name:            MagicNumberItersPerTile
        .size:            4
        .offset:          124
        .value_kind:      by_value
        .value_type:      u32
      - .name:            MagicShiftItersPerTile
        .size:            4
        .offset:          128
        .value_kind:      by_value
        .value_type:      u32
      - .name:            TotalIters
        .size:            4
        .offset:          132
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SKItersPerWG
        .size:            4
        .offset:          136
        .value_kind:      by_value
        .value_type:      u32
      - .name:            skGrid
        .size:            4
        .offset:          140
        .value_kind:      by_value
        .value_type:      u32
      - .name:            skTiles
        .size:            4
        .offset:          144
        .value_kind:      by_value
        .value_type:      u32
      - .name:            AddressScaleAlphaVec
        .size:            8
        .offset:          148
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            bias
        .size:            8
        .offset:          156
        .value_kind:      global_buffer
        .value_type:      void
        .address_space:   generic
      - .name:            biasType
        .size:            4
        .offset:          164
        .value_kind:      by_value
        .value_type:      u32
      - .name:            StrideBias
        .size:            4
        .offset:          168
        .value_kind:      by_value
        .value_type:      u32
    .group_segment_fixed_size:   101376
    .kernarg_segment_align:      8
    .kernarg_segment_size:       176
    .max_flat_workgroup_size:    256
    .private_segment_fixed_size: 0
    .sgpr_count:                 94
    .sgpr_spill_count:           0
    .vgpr_count:                 256
    .vgpr_spill_count:           0
    .wavefront_size:             64
...
.end_amdgpu_metadata
Custom_Cijk_Ailk_Bljk_BBS_BH_Bias_SAV_UserArgs_MT128x128x64_MI16x16x1_UserArgs_shortname1_gfx950:
label_ASM_Start:  /// Main body of the asm kernel
.macro V_MAGIC_DIV vgprDstIdx:req, dividend:req, magicNumber:req, magicShift:req, magicA:req
    v_mul_hi_u32 v[\vgprDstIdx+1], \dividend, \magicNumber
    v_mul_lo_u32 v[\vgprDstIdx+0], \dividend, \magicA
    v_add_u32 v[\vgprDstIdx+0], v[\vgprDstIdx+0], v[\vgprDstIdx+1]
    v_lshrrev_b32 v[\vgprDstIdx+0], \magicShift, v[\vgprDstIdx+0]
.endm

/******************************************/
/* VGPR Assignments                       */
/******************************************/
/* ValuC range: [0-0), serializedStore enabled */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprBase, 10
.set vgprGlobalReadOffsetA, 0
.set vgprGlobalReadOffsetB, 4
.set vgprLocalReadAddrA, 8
.set vgprLocalReadAddrB, 9
.set vgprSerial, 74

/******************************************/
/* VGPR Macro Assignments                 */
/******************************************/
.set vgprValuA_X0_I0_BASE, vgprBase+0
.set vgprValuA_X0_I0_D0_PACK, vgprBase+-11
.set vgprValuB_X0_I0_BASE, vgprBase+32
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+16
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X1_I0, vgprValuB_X0_I0_BASE+16
.set vgprValuA_X0_I0_D1, vgprValuA_X0_I0_D0_PACK+0
.set vgprValuA_X1_I0_D1, vgprValuA_X0_I0_D0_PACK+16
.set vgprTmp0, 100 
.set vgprLocalReadAddrARef, 101 
.set vgprLocalReadAddrBRef, 102 
/******************************************/
/* SGPR Assignments                       */
/******************************************/
.set sgprKernArgAddress, 0
.set sgprWorkGroup0, 2
.set sgprWorkGroup1, 3
.set sgprWorkGroup2, 4
.set sgprArgType, 5
.set sgprStaggerU, 6
.set sgprWGM, 7
.set sgprLoopCounterL, 8
.set sgprOrigLoopCounter, 9
.set sgprSrdD, 12
.set sgprSrdC, 16
.set sgprNumWorkGroups0, 10
.set sgprNumWorkGroups1, 11
.set sgprSizesFree, 20
.set sgprSizesSum, 23
.set sgprAddressD, 24
.set sgprAddressC, 26
.set sgprAddressA, 28
.set sgprAddressB, 30
.set sgprAddressWS, 32
.set sgprAddressFlags, 34
.set sgprStridesD, 36
.set sgprStridesC, 38
.set sgprStridesA, 40
.set sgprStridesB, 42
.set sgprAlpha, 44
.set sgprBeta, 45
.set sgprItersPerTile, 46
.set sgprMagicNumberItersPerTile, 47
.set sgprMagicShiftItersPerTile, 48
.set sgprTotalIters, 49
.set sgprSKItersPerWG, 50
.set sgprskGrid, 51
.set sgprskTiles, 52
.set sgprLocalWriteAddrA, 53
.set sgprLocalWriteAddrB, 54
.set sgprStreamKIdx, 55
.set sgprStreamKIter, 56
.set sgprStreamKIterEnd, 57
.set sgprStreamKLocalStart, 58
.set sgprStreamKLocalEnd, 59
.set sgprSrdWS, 60
.set sgprTmp0, 88
.set sgprTmp1, 89
.set sgprAddrARef, 90
.set sgprAddrBRef, 91
.set sgprOffsetW, 92
.set sgprOffsetR, 93

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
.set constStrideA0I, 1
.set sgprStrideAL, sgprStridesA+0
.set sgprStrideAK, sgprStridesA+1
.set constStrideBL, 1
.set sgprStrideB1J, sgprStridesB+0
.set sgprStrideBK, sgprStridesB+1

.set MT0, 128
.set MT1, 128
.set DepthU, 64
.set BpeA, 2
.set BpeALog2, 1
.set BpeB, 2
.set BpeBLog2, 1
.set BpeAGR, 2
.set BpeAGRLog2, 1
.set BpeBGR, 2
.set BpeBGRLog2, 1
/* Number of elements to shift-left SRD */
.set SrdShiftLeftA, 8
.set SrdShiftLeftB, 8
/* 2GB limit - set offsets to -1 to exceed this and clamp */
.set BufferLimit, 0xffffffff
.set BufferOOB, 0x80000000

/******************************************/
/* Bits 127:96 of SRD.                    */
/* hex: 0x20000                           */
/* dst_sel_x (3b): 0                      */
/* dst_sel_y (3b): 0                      */
/* dst_sel_z (3b): 0                      */
/* dst_sel_w (3b): 0                      */
/* num_format (3b): 0                     */
/* data_format (4b): 4                    */
/* user_vm_enable (1b): 0                 */
/* user_vm_mode (1b): 0                   */
/* index_stride (2b): 0                   */
/* add_tid_enable (1b): 0                 */
/* _unusedA (3b): 0                       */
/* nv (1b): 0                             */
/* _unusedB (2b): 0                       */
/* type (2b): 0                           */
/******************************************/
.set Srd127_96, 0x20000

/* Global Offset A */
.macro GLOBAL_OFFSET_A vgprAddr:req, vgprOffset0I:req, vgprOffsetL:req, vgprTmp:req
    v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideAL], v[\vgprOffsetL] // mul d1 lower
    v_add_co_u32 v[\vgprAddr+0], vcc, v[\vgprOffset0I], v[\vgprTmp+0] // accumulate K lower
    v_add_u32 v[\vgprAddr+0], 0x8, v[\vgprAddr+0]      // add prepad for pointer shift
    v_lshlrev_b32 v[\vgprAddr+0], 1, v[\vgprAddr+0]    // offset *= bytes/element
.endm

/* Global Offset B */
.macro GLOBAL_OFFSET_B vgprAddr:req, vgprOffsetL:req, vgprOffset1J:req, vgprTmp:req
    v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideB1J], v[\vgprOffset1J] // mul d1 lower
    v_add_co_u32 v[\vgprAddr+0], vcc, v[\vgprOffsetL], v[\vgprTmp+0] // accumulate K lower
    v_add_u32 v[\vgprAddr+0], 0x8, v[\vgprAddr+0]      // add prepad for pointer shift
    v_lshlrev_b32 v[\vgprAddr+0], 1, v[\vgprAddr+0]    // offset *= bytes/element
.endm

/******************************************/
/* Allocate Resources                     */
/******************************************/

/* Load num of Gemms */
s_load_dword s16, s[sgprKernArgAddress:sgprKernArgAddress+1], 0

/* Load packed kernel args (StaggerU/GSU) */
s_load_dword s18, s[sgprKernArgAddress:sgprKernArgAddress+1], 4

/* Load WGM data */
s_load_dword s[sgprWGM], s[sgprKernArgAddress:sgprKernArgAddress+1], 8

/* Load num of WGs */
s_load_dword s19, s[sgprKernArgAddress:sgprKernArgAddress+1], 12
s_waitcnt lgkmcnt(0)                               // load args
s_lshr_b32 s17, s16, 0x1e                          // Get arg type
s_and_b32 s16, 0x3fffffff, s16                     // Get nums of gemm
s_cmp_eq_u32 s17, 0                                // Is kernel args
s_cbranch_scc0 label_HBMArgs
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], 0x10 // Shift common args
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_dwordx16 s[20:35], s[sgprKernArgAddress:sgprKernArgAddress+1], 0 // 0
s_load_dwordx16 s[36:51], s[sgprKernArgAddress:sgprKernArgAddress+1], 64 // 64
s_load_dword s52, s[sgprKernArgAddress:sgprKernArgAddress+1], 128 // 128
s_waitcnt lgkmcnt(0)                               // preload
s_branch label_LoadArgsEnd
label_HBMArgs:

/* Load address of kernel arguments */
s_load_dwordx2 s[sgprKernArgAddress:sgprKernArgAddress+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 16
s_waitcnt lgkmcnt(0)                               // wait for args to load
label_LoadArgsEnd:
s_branch label_common_kernel_entry

/* pad 37 snops to satisfy 0x100 code size for Preload Backward Compatibility Prologue */
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
label_Preload_Offset_Start:
s_and_b32 s16, 0x3fffffff, s2                      // Get nums of gemm
s_lshr_b32 s17, s2, 0x1e                           // Get arg type
s_mov_b32 s18, s3                                  // Preload internal args
s_cmp_eq_u32 s17, 0                                // Is kernel args
s_cbranch_scc0 label_Preload_HBMArgs
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], 0x10 // Shift common args
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_dword s27, s[sgprKernArgAddress:sgprKernArgAddress+1], 28 // 28
s_load_dwordx16 s[28:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 32 // 32
s_load_dwordx8 s[44:51], s[sgprKernArgAddress:sgprKernArgAddress+1], 96 // 96
s_load_dword s52, s[sgprKernArgAddress:sgprKernArgAddress+1], 128 // 128
s_mov_b64 s[20:21], s[6:7]                         // move preload data to correct sgpr
s_mov_b64 s[22:23], s[8:9]                         // move preload data to correct sgpr
s_mov_b64 s[24:25], s[10:11]                       // move preload data to correct sgpr
s_mov_b32 s26, s12                                 // move preload data to correct sgpr
s_branch label_Preload_LoadArgsEnd
label_Preload_HBMArgs:
s_mov_b64 s[sgprKernArgAddress:sgprKernArgAddress+1], s[6:7] // Load address of kernel arguments
label_Preload_LoadArgsEnd:
s_mov_b32 s[sgprWGM], s4                           // Preload internal args2
s_mov_b32 s19, s5                                  // Load num of WGs
label_common_kernel_entry:  /// for both preload/non-preload common code
s_mov_b32 s[sgprWorkGroup0+0], s13                 // restore workgroup id
s_mov_b32 s[sgprWorkGroup0+1], s14                 // restore workgroup id
s_mov_b32 s[sgprWorkGroup0+2], s15                 // restore workgroup id
s_and_b32 s[sgprStaggerU], s18, 0xffff0000         // Restore StaggerU related vars
s_lshr_b32 s[sgprStaggerU], s[sgprStaggerU], 0x10
s_mov_b32 s[sgprArgType], s17
s_mov_b32 m0, LDSSize                              // LDS clamp at 99328 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id

/* remap workgroup to XCCs */
s_lshr_b32 s64, s[sgprWGM], 0x10                   // Get WGMXCC
s_ff1_i32_b32 s64, s64                             // Get log(WGMXCC)
s_lshr_b32 s65, s[sgprWGM], 0x16                   // Get CU_Count
/* remap WGs if WGMXCC > 1 ( log(WGMXCC) > 0 ) */
s_cmp_gt_i32 s64, 0
s_cbranch_scc0 label_skip_WGMXCC
/* only remap WGs in the range */
s_lshr_b32 s61, s19, s64
s_lshl_b32 s61, s61, s64
s_cmp_ge_u32 s[sgprWorkGroup0], s61
s_cbranch_scc1 label_skip_WGMXCC
s_cmp_eq_u32 s65, 0                                // CU_Count == 0 ?
s_cbranch_scc0 label_XCCG_nonzero
s_lshr_b32 s61, s[sgprWorkGroup0], s64
s_bfm_b32 s62, s64, 0
s_and_b32 s62, s[sgprWorkGroup0], s62
s_lshr_b32 s63, s19, s64
s_mul_i32 s62, s62, s63
s_add_u32 s[sgprWorkGroup0], s61, s62
s_branch label_skip_WGMXCC
label_XCCG_nonzero:
/* temp0 = (wg//CU_Count)*CU_Count */
v_cvt_f64_u32 v[10:11], s65                        // s61 = s[sgprWorkGroup0] / s65
v_rcp_f64 v[10:11], v[10:11]                       // s61 = s[sgprWorkGroup0] / s65
v_cvt_f64_u32 v[12:13], s[sgprWorkGroup0]          // s61 = s[sgprWorkGroup0] / s65
v_mul_f64 v[10:11], v[10:11], v[12:13]             // s61 = s[sgprWorkGroup0] / s65
v_cvt_u32_f64 v10, v[10:11]                        // s61 = s[sgprWorkGroup0] / s65
v_mul_lo_u32 v11, v10, s65                         // s61 = s[sgprWorkGroup0] / s65
v_sub_u32 v12, s[sgprWorkGroup0], v11              // s61 = s[sgprWorkGroup0] / s65
v_cmpx_ge_u32 exec, v12, s65                       // s61 = s[sgprWorkGroup0] / s65
v_add_u32 v10, v10, 1                              // s61 = s[sgprWorkGroup0] / s65
s_mov_b64 exec, -1                                 // Reset exec
v_mul_lo_u32 v11, v10, s65                         // s61 = s[sgprWorkGroup0] / s65
v_sub_u32 v12, s[sgprWorkGroup0], v11              // s61 = s[sgprWorkGroup0] / s65
v_readfirstlane_b32 s61, v10                       // quotient
v_readfirstlane_b32 s62, v12                       // remainder
s_mul_i32 s61, s61, s65
/* temp1 = (wg%CU_Count)//WGMXCC */
s_lshr_b32 s62, s62, s64
/* temp0 = temp0 + temp1 */
s_add_u32 s61, s61, s62
/* temp1 = (wg%WGMXCC) * ((WGs - (WGs//CU_Count) * CU_Count) if (wg > (WGs//CU_Count) * CU_Count) else CU_Count)//WGMXCC */
v_cvt_f64_u32 v[10:11], s65                        // s62 = s19 / s65
v_rcp_f64 v[10:11], v[10:11]                       // s62 = s19 / s65
v_cvt_f64_u32 v[12:13], s19                        // s62 = s19 / s65
v_mul_f64 v[10:11], v[10:11], v[12:13]             // s62 = s19 / s65
v_cvt_u32_f64 v10, v[10:11]                        // s62 = s19 / s65
v_mul_lo_u32 v11, v10, s65                         // s62 = s19 / s65
v_sub_u32 v12, s19, v11                            // s62 = s19 / s65
v_cmpx_ge_u32 exec, v12, s65                       // s62 = s19 / s65
v_add_u32 v10, v10, 1                              // s62 = s19 / s65
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s62, v10                       // quotient
s_mul_i32 s62, s62, s65
s_sub_u32 s63, s19, s62
s_cmp_gt_u32 s[sgprWorkGroup0], s62
s_cselect_b32 s62, s63, s65
s_lshr_b32 s62, s62, s64
s_bfm_b32 s63, s64, 0
s_and_b32 s63, s[sgprWorkGroup0], s63
s_mul_i32 s62, s62, s63
/* WorkGroup0 = temp0 + temp1 */
s_add_u32 s[sgprWorkGroup0], s61, s62
label_skip_WGMXCC:  /// skip WGMXCC if no enough WGs to remap
s_cmp_eq_u32 s17, 0
s_cbranch_scc0 label_MultiGemm
/* init: add vgpr [10...84) to pool */
/* init: add vgpr [0...0) to pool */
/* init: add agpr [0...64) to pool */
v_mov_b32 v12, MT0                                 // set MT0 into sgpr
v_mov_b32 v11, s[sgprSizesFree+0]                  // set Free0 size
v_cvt_f32_u32 v10, v12                             // v10 = ceil(v11 / v12)
v_rcp_iflag_f32 v10, v10                           // v10 = ceil(v11 / v12)
v_cvt_f32_u32 v13, v11                             // v10 = ceil(v11 / v12)
v_mul_f32 v10, v10, v13                            // v10 = ceil(v11 / v12)
v_cvt_u32_f32 v10, v10                             // v10 = ceil(v11 / v12)
v_mul_u32_u24 v13, v10, v12                        // v10 = ceil(v11 / v12)
v_sub_u32 v13, v11, v13                            // v10 = ceil(v11 / v12)
v_cmp_ne_u32 vcc, v13, 0                           // v10 = ceil(v11 / v12)
v_addc_co_u32 v10, vcc, v10, 0, vcc                // ceil
v_mov_b32 v12, MT1                                 // set MT1 into sgpr
v_mov_b32 v11, s[sgprSizesFree+1]                  // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v10     // set back to numWorkGroup0
v_cvt_f32_u32 v10, v12                             // v10 = ceil(v11 / v12)
v_rcp_iflag_f32 v10, v10                           // v10 = ceil(v11 / v12)
v_cvt_f32_u32 v13, v11                             // v10 = ceil(v11 / v12)
v_mul_f32 v10, v10, v13                            // v10 = ceil(v11 / v12)
v_cvt_u32_f32 v10, v10                             // v10 = ceil(v11 / v12)
v_mul_u32_u24 v13, v10, v12                        // v10 = ceil(v11 / v12)
v_sub_u32 v13, v11, v13                            // v10 = ceil(v11 / v12)
v_cmp_ne_u32 vcc, v13, 0                           // v10 = ceil(v11 / v12)
v_addc_co_u32 v10, vcc, v10, 0, vcc                // ceil
s_nop 0                                            // 1 wait states
v_readfirstlane_b32 s[sgprNumWorkGroups1], v10     // set back to numWorkGroup1
s_waitcnt lgkmcnt(0)                               // wait for 88/0 bytes of kern args
s_branch label_MultiGemmEnd
label_MultiGemm:

/* Check if custom structure pointer is null */
s_cmp_eq_u32 s[sgprArgType], 2                     // ArgType == 2 ?
s_cbranch_scc1 label_IsExternalValid               // branch if ArgType == 2
s_mov_b32 s11, 156
s_mul_i32 s66, s16, 4
s_mov_b64 s[60:61], s[sgprKernArgAddress:sgprKernArgAddress+1]
s_branch label_IsExternalValidEnd
label_IsExternalValid:
s_mov_b32 s11, 224
s_mov_b32 s66, 0
s_mov_b64 s[60:61], s[sgprKernArgAddress:sgprKernArgAddress+1]
label_IsExternalValidEnd:

/* Grouped Gemm:: prefetch 1 arg load */
s_mov_b32 s10, 1
s_mov_b32 s67, 0
s_load_dwordx4 s[20:23], s[60:61], s66
s_cmpk_eq_u32 s16, 1                               // if gemm_count is 1?
s_cbranch_scc1 label_wgTable_noLoadLoop

/* Grouped Gemm:: accumulate numTiles for each gemm */
/* Grouped Gemm:: loop start */
label_Loop_GemmCount:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s64, s20, 7                             // s64 = s20 / 128
s_and_b32 s62, 127, s20                            // s62 = s20 % 128
s_addc_u32 s64, s64, 0
s_lshr_b32 s65, s21, 7                             // s65 = s21 / 128
s_and_b32 s62, 127, s21                            // s62 = s21 % 128
s_addc_u32 s65, s65, 0
s_mul_i32 s64, s64, s65
s_mul_i32 s64, s64, s22
s_add_u32 s67, s67, s64
s_cmp_lt_u32 s[sgprWorkGroup0], s67
s_cbranch_scc1 label_FOUND
s_add_u32 s66, s66, s11
s_load_dwordx4 s[20:23], s[60:61], s66
s_add_u32 s10, s10, 1
s_cmp_lt_u32 s10, s16
s_cbranch_scc1 label_Loop_GemmCount

/* Grouped Gemm:: noLoadLoop */
label_wgTable_noLoadLoop:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s64, s20, 7                             // s64 = s20 / 128
s_and_b32 s62, 127, s20                            // s62 = s20 % 128
s_addc_u32 s64, s64, 0
s_lshr_b32 s65, s21, 7                             // s65 = s21 / 128
s_and_b32 s62, 127, s21                            // s62 = s21 % 128
s_addc_u32 s65, s65, 0
s_mul_i32 s64, s64, s65
s_mul_i32 s64, s64, s22
s_add_u32 s67, s67, s64

/* Grouped Gemm:: gemmIndex found */
label_FOUND:
s_sub_u32 s61, s10, 1
s_sub_u32 s60, s67, s64
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s60
/* Check if custom structure pointer is null */
s_cmp_eq_u32 s[sgprArgType], 2                     // ArgType == 2 ?
s_cbranch_scc1 label_LoadExternalStruct            // branch if ArgType == 2

/* Grouped Gemm: offset argument address to gemm */
/* Grouped Gemm: offset address from wg_table_start to args_start */
s_lshl2_add_u32 s[sgprKernArgAddress], s16, s[sgprKernArgAddress]
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s61, s61, 156
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s61
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_dwordx16 s[24:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_dwordx8 s[40:47], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
s_load_dwordx4 s[48:51], s[sgprKernArgAddress:sgprKernArgAddress+1], 112 // 112
s_load_dword s52, s[sgprKernArgAddress:sgprKernArgAddress+1], 128 // 128
s_branch label_LoadExternalStructEnd
label_LoadExternalStruct:
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s61, s61, 224
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s61
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
s_load_dwordx16 s[24:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_dwordx8 s[40:47], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
s_load_dwordx4 s[48:51], s[sgprKernArgAddress:sgprKernArgAddress+1], 112 // 112
// Read Beta
s_load_dword s45, s[sgprKernArgAddress:sgprKernArgAddress+1], 140 // 140
label_LoadExternalStructEnd:
/* init: add vgpr [10...84) to pool */
/* init: add vgpr [0...0) to pool */
/* init: add agpr [0...64) to pool */
v_mov_b32 v12, MT0                                 // set MT0 into sgpr
v_mov_b32 v11, s[sgprSizesFree+0]                  // set Free0 size
v_cvt_f32_u32 v10, v12                             // v10 = ceil(v11 / v12)
v_rcp_iflag_f32 v10, v10                           // v10 = ceil(v11 / v12)
v_cvt_f32_u32 v13, v11                             // v10 = ceil(v11 / v12)
v_mul_f32 v10, v10, v13                            // v10 = ceil(v11 / v12)
v_cvt_u32_f32 v10, v10                             // v10 = ceil(v11 / v12)
v_mul_u32_u24 v13, v10, v12                        // v10 = ceil(v11 / v12)
v_sub_u32 v13, v11, v13                            // v10 = ceil(v11 / v12)
v_cmp_ne_u32 vcc, v13, 0                           // v10 = ceil(v11 / v12)
v_addc_co_u32 v10, vcc, v10, 0, vcc                // ceil
v_mov_b32 v12, MT1                                 // set MT1 into sgpr
v_mov_b32 v11, s[sgprSizesFree+1]                  // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v10     // set back to numWorkGroup0
v_cvt_f32_u32 v10, v12                             // v10 = ceil(v11 / v12)
v_rcp_iflag_f32 v10, v10                           // v10 = ceil(v11 / v12)
v_cvt_f32_u32 v13, v11                             // v10 = ceil(v11 / v12)
v_mul_f32 v10, v10, v13                            // v10 = ceil(v11 / v12)
v_cvt_u32_f32 v10, v10                             // v10 = ceil(v11 / v12)
v_mul_u32_u24 v13, v10, v12                        // v10 = ceil(v11 / v12)
v_sub_u32 v13, v11, v13                            // v10 = ceil(v11 / v12)
v_cmp_ne_u32 vcc, v13, 0                           // v10 = ceil(v11 / v12)
v_addc_co_u32 v10, vcc, v10, 0, vcc                // ceil
s_nop 0                                            // 1 wait states
v_readfirstlane_b32 s[sgprNumWorkGroups1], v10     // set back to numWorkGroup1
s_waitcnt lgkmcnt(0)                               // wait for 88/0 bytes of kern args

/* Early stop if N(SizeFreeJ) == 0 */
s_cmp_eq_u32 s[sgprSizeJ], 0
s_cbranch_scc0 label_NoEarlyStop_N0
label_EarlyStop_if_N_is_0:
s_endpgm
label_NoEarlyStop_N0:

label_MultiGemmEnd:
.set sgprSrdA, 64
.set sgprSrdB, 68
.set sgprShadowLimitA, 72
.set sgprShadowLimitB, 74
.set sgprStaggerUIter, 76
.set sgprWrapUA, 77
.set sgprWrapUB, 79
.set sgprGlobalReadIncsA, 81
.set sgprGlobalReadIncsB, 82
s_sub_u32 s[sgprAddressA+0], s[sgprAddressA+0], 16 // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressA+1], s[sgprAddressA+1], 0 // pre-pad to make room for possible pointer shift
s_sub_u32 s[sgprAddressB+0], s[sgprAddressB+0], 16 // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressB+1], s[sgprAddressB+1], 0 // pre-pad to make room for possible pointer shift

/* Short circuit condition if Alpha == 0, then sumDims=0 */
v_cmp_eq_f32 vcc, s[sgprAlpha], 0.0                // s[Alpha] == 0.0f ?
s_cbranch_vccz label_AlphaNonZero                  // branch if s[Alpha] != 0
s_mov_b32 s[sgprSizesSum+0], 0                     // Set summation dim=0 if Alpha == 0
label_AlphaNonZero:
s_mov_b32 s[sgprStreamKIdx], s[sgprWorkGroup0]     // Save original StreamK index
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_SK_SplitInit                  // Jump to single kernel init
v_cvt_f32_u32 v10, s[sgprSkSplit]                  // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_rcp_iflag_f32 v10, v10                           // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_cvt_f32_u32 v11, s[sgprStreamKIdx]               // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_mul_f32 v10, v10, v11                            // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_cvt_u32_f32 v10, v10                             // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_mul_u32_u24 v11, v10, s[sgprSkSplit]             // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_sub_u32 v11, s[sgprStreamKIdx], v11              // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_cmpx_eq_u32 exec, v11, s[sgprSkSplit]            // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_add_u32 v10, 1, v10                              // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_mov_b32 v11, 0                                   // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v11, s[sgprSkSplit]            // overflow happened in remainder
v_sub_u32 v10, v10, 1                              // quotient - 1
v_mul_u32_u24 v11, v10, s[sgprSkSplit]             // re-calculate remainder
v_sub_u32 v11, s[sgprStreamKIdx], v11              // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s12, v10                       // quotient
v_readfirstlane_b32 s13, v11                       // remainder
s_mul_i32 s14, s[sgprSkSplit], s[sgprSKItersPerWG]
s_sub_u32 s14, s[sgprItersPerTile], s14            // extraIters = itersPerTile - SkSplit * skItersPerWG
s_mul_i32 s[sgprStreamKIter], s13, s[sgprSKItersPerWG] // StreamK starting iteration (case: after extra iters)
s_cmp_lt_u32 s13, s14                              // Check if WG gets an extra iteration
s_cbranch_scc1 label_SK_HasExtra                   // Has extra iter
s_add_u32 s[sgprStreamKIter], s[sgprStreamKIter], s14 // This WG does not have an extra iteration
s_add_u32 s[sgprStreamKIterEnd], s[sgprStreamKIter], s[sgprSKItersPerWG] // StreamK ending iteration (case: after extra iters)
s_branch label_SK_DoneExtra                        // Done init for parallel reduction
label_SK_HasExtra:
s_add_u32 s[sgprStreamKIter], s[sgprStreamKIter], s13 // This WG has an extra iteration
s_add_u32 s[sgprStreamKIterEnd], s[sgprStreamKIter], s[sgprSKItersPerWG] // StreamK ending iteration (case: after extra iters)
s_add_u32 s[sgprStreamKIterEnd], s[sgprStreamKIterEnd], 1 // StreamK ending iteration (case: after extra iters)
label_SK_DoneExtra:
s_mul_i32 s12, s12, s[sgprItersPerTile]            // Tile offset = tilesIdx * itersPerTile
s_add_u32 s[sgprStreamKIter], s[sgprStreamKIter], s12 // Offset to correct tile
s_add_u32 s[sgprStreamKIterEnd], s[sgprStreamKIterEnd], s12 // Offset to correct tile
s_mov_b32 s[sgprSkPartialIdx], s13                 // Save partial idx for SrdD calculation
s_branch label_SK_InitDone                         // Done init for parallel reduction
label_SK_SplitInit:
s_mul_i32 s[sgprStreamKIter], s[sgprStreamKIdx], s[sgprItersPerTile] // DP starting iteration (case: DP work to do)
s_mov_b32 s[sgprStreamKIterEnd], s[sgprTotalIters] // DP ending iteration (case: only DP work to do)
s_mul_i32 s12, s[sgprskTiles], s[sgprItersPerTile] // Total SK iters
s_cmp_lt_u32 s12, s[sgprTotalIters]                // Check if there are DP tiles to do
s_cbranch_scc1 label_SK_InitDone                   // Done init
s_mul_i32 s12, s[sgprskTiles], s[sgprItersPerTile]
s_mul_i32 s13, s[sgprSKItersPerWG], s[sgprskGrid]
s_sub_u32 s12, s12, s13                            // skTiles * ItersPerTile - SKItersPerWG * skGrid
s_mul_i32 s[sgprStreamKIter], s[sgprStreamKIdx], s[sgprSKItersPerWG] // StreamK starting iteration (case: after extra iters)
s_add_u32 s[sgprStreamKIter], s[sgprStreamKIter], s12 // Add extra iters
s_add_u32 s[sgprStreamKIterEnd], s[sgprStreamKIter], s[sgprSKItersPerWG] // StreamK ending iteration (case: after extra iters)
s_add_u32 s14, s[sgprSKItersPerWG], 1              // Spread out extra iterations
s_mul_i32 s13, s[sgprStreamKIdx], s14              // StreamK starting iteration (case: before extra iters)
s_add_u32 s14, s13, s14                            // StreamK ending iteration (case: before extra iters)
s_cmp_lt_u32 s[sgprStreamKIdx], s12                // Check if lane gets an extra iteration
s_cselect_b32 s[sgprStreamKIter], s13, s[sgprStreamKIter] // Set start iter
s_cselect_b32 s[sgprStreamKIterEnd], s14, s[sgprStreamKIterEnd] // Set end iter
s_mul_i32 s12, s[sgprskTiles], s[sgprItersPerTile] // Total SK iters
s_min_u32 s[sgprStreamKIterEnd], s[sgprStreamKIterEnd], s12 // Cap ending iter at total SK iters
label_SK_InitDone:
s_cmp_lt_u32 s[sgprStreamKIter], s[sgprTotalIters] // Make sure there's work to do
s_cbranch_scc1 label_NoBranch_T8JHFHKM7BO5OHXW     // Only branch on scc0
s_getpc_b64 s[12:13]                               // addr of next instr
s_add_i32 s14, label_KernelEnd, 4                  // target branch offset
s_add_u32 s12, s12, s14                            // add target branch offset
s_addc_u32 s13, s13, 0                             // add high and carry
s_setpc_b64 s[12:13]                               // branch to label_KernelEnd
label_NoBranch_T8JHFHKM7BO5OHXW:

/******************************************/
/* Persistent Loop Start                  */
/******************************************/
label_PersistentLoopStart:

/******************************************/
/* Begin setupNewTile                     */
/******************************************/

/* global read addresses: work-group */
/* graWorkGroup mapping */

/* localReadResetOffsets */
/* handled internally */
v_and_b32 v[vgprLocalReadAddrA+0], 0xffff, v[vgprLocalReadAddrA+0] // reset Red,Blk -> Red

/* localReadResetOffsets */
/* handled internally */
v_and_b32 v[vgprLocalReadAddrB+0], 0xffff, v[vgprLocalReadAddrB+0] // reset Red,Blk -> Red
/* StreamK calculate tile idx and map to WG */
s_mul_hi_u32 s13, s[sgprStreamKIter], s[sgprMagicNumberItersPerTile] // s_magic mul, div alg 2
s_lshr_b32 s14, s[sgprMagicShiftItersPerTile], 31  // tmpS = extract abit
s_mul_i32 s12, s[sgprStreamKIter], s14             // s_magic mul, div alg 2
s_add_u32 s12, s12, s13
s_and_b32 s14, s[sgprMagicShiftItersPerTile], 2147483647 // tmpS = remove abit to final shift
s_lshr_b32 s12, s12, s14                           // sMagicDiv Alg 2
s_mul_i32 s13, s12, s[sgprItersPerTile]            // Tile start iteration
s_add_u32 s14, s13, s[sgprItersPerTile]            // Tile end iteration
s_sub_u32 s[sgprStreamKLocalStart], s[sgprStreamKIter], s13 // Local iteration start
s_min_u32 s[sgprStreamKLocalEnd], s[sgprStreamKIterEnd], s14 // 1. (Local) iteration end (SK tile)
s_sub_u32 s[sgprStreamKLocalEnd], s[sgprStreamKLocalEnd], s13 // 2. Local iteration end (SK tile)
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_SK_SplitUpdate                // Jump to single kernel update
s_mov_b32 s13, s[sgprStreamKIterEnd]               // Parallel reduction, work contained to single partial tile
s_branch label_SK_UpdateDone                       // Done update for parallel reduction
label_SK_SplitUpdate:
s_mul_i32 s15, s[sgprskTiles], s[sgprItersPerTile] // Total SK iters
s_sub_u32 s15, s[sgprTotalIters], s15              // Offset to first SK tile
s_mul_i32 s13, s[sgprskGrid], s[sgprItersPerTile]  // DP iterations shift
s_add_u32 s13, s13, s[sgprStreamKIter]             // Add DP shift
s_cmp_lt_u32 s13, s15                              // Check if still in DP section
s_cbranch_scc1 label_SK_UpdateDone                 // Done update
s_mov_b32 s13, s14                                 // SK iterations shift
s_cmp_le_u32 s15, s[sgprStreamKIter]               // Check if continuing in SK section
s_cbranch_scc1 label_SK_UpdateDone                 // Done update
s_mul_i32 s16, s[sgprskTiles], s[sgprItersPerTile]
s_mul_i32 s17, s[sgprSKItersPerWG], s[sgprskGrid]
s_sub_u32 s16, s16, s17                            // skTiles * ItersPerTile - SKItersPerWG * skGrid
s_mul_i32 s[sgprStreamKIter], s[sgprStreamKIdx], s[sgprSKItersPerWG] // StreamK starting iteration (case: after extra iters)
s_add_u32 s[sgprStreamKIter], s[sgprStreamKIter], s16 // Add extra iters
s_add_u32 s[sgprStreamKIterEnd], s[sgprStreamKIter], s[sgprSKItersPerWG] // StreamK ending iteration (case: after extra iters)
s_add_u32 s18, s[sgprSKItersPerWG], 1              // Spread out extra iterations
s_mul_i32 s17, s[sgprStreamKIdx], s18              // StreamK starting iteration (case: before extra iters)
s_add_u32 s18, s17, s18                            // StreamK ending iteration (case: before extra iters)
s_cmp_lt_u32 s[sgprStreamKIdx], s16                // Check if lane gets an extra iteration
s_cselect_b32 s[sgprStreamKIter], s17, s[sgprStreamKIter] // Set start iter
s_cselect_b32 s[sgprStreamKIterEnd], s18, s[sgprStreamKIterEnd] // Set end iter
s_add_u32 s13, s[sgprStreamKIter], s15             // Offset to start of SK section
s_add_u32 s[sgprStreamKIterEnd], s[sgprStreamKIterEnd], s15 // Offset to start of SK section
s_min_u32 s[sgprStreamKIterEnd], s[sgprStreamKIterEnd], s[sgprTotalIters] // Cap ending iter at total SK iters
s_cmp_lt_u32 s[sgprStreamKIter], s[sgprTotalIters] // Make sure there's work to do
s_cbranch_scc1 label_NoBranch_S4FDBQ587JJL6NOU     // Only branch on scc0
s_getpc_b64 s[16:17]                               // addr of next instr
s_add_i32 s18, label_KernelEnd, 4                  // target branch offset
s_add_u32 s16, s16, s18                            // add target branch offset
s_addc_u32 s17, s17, 0                             // add high and carry
s_setpc_b64 s[16:17]                               // branch to label_KernelEnd
label_NoBranch_S4FDBQ587JJL6NOU:
label_SK_UpdateDone:
s_mov_b32 s[sgprStreamKIter], s13                  // Store current iteration
/* Map StreamK tile index to wg0/1/2 */
s_mul_i32 s13, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1] // Total tiles
v_cvt_f32_u32 v10, s13                             // TileID // nWG0*nWG1
v_rcp_iflag_f32 v10, v10                           // TileID // nWG0*nWG1
v_cvt_f32_u32 v11, s12                             // TileID // nWG0*nWG1
v_mul_f32 v10, v10, v11                            // TileID // nWG0*nWG1
v_cvt_u32_f32 v10, v10                             // TileID // nWG0*nWG1
v_mul_u32_u24 v11, v10, s13                        // TileID // nWG0*nWG1
v_sub_u32 v11, s12, v11                            // TileID // nWG0*nWG1
v_cmpx_eq_u32 exec, v11, s13                       // TileID // nWG0*nWG1
v_add_u32 v10, 1, v10                              // TileID // nWG0*nWG1
v_mov_b32 v11, 0                                   // TileID // nWG0*nWG1
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v11, s13                       // overflow happened in remainder
v_sub_u32 v10, v10, 1                              // quotient - 1
v_mul_u32_u24 v11, v10, s13                        // re-calculate remainder
v_sub_u32 v11, s12, v11                            // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s[sgprWorkGroup2], v10         // quotient
v_readfirstlane_b32 s14, v11                       // remainder
v_cvt_f32_u32 v10, s[sgprNumWorkGroups0]           // TileID // nWG0
v_rcp_iflag_f32 v10, v10                           // TileID // nWG0
v_cvt_f32_u32 v11, s14                             // TileID // nWG0
v_mul_f32 v10, v10, v11                            // TileID // nWG0
v_cvt_u32_f32 v10, v10                             // TileID // nWG0
v_mul_u32_u24 v11, v10, s[sgprNumWorkGroups0]      // TileID // nWG0
v_sub_u32 v11, s14, v11                            // TileID // nWG0
v_cmpx_eq_u32 exec, v11, s[sgprNumWorkGroups0]     // TileID // nWG0
v_add_u32 v10, 1, v10                              // TileID // nWG0
v_mov_b32 v11, 0                                   // TileID // nWG0
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v11, s[sgprNumWorkGroups0]     // overflow happened in remainder
v_sub_u32 v10, v10, 1                              // quotient - 1
v_mul_u32_u24 v11, v10, s[sgprNumWorkGroups0]      // re-calculate remainder
v_sub_u32 v11, s14, v11                            // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s[sgprWorkGroup1], v10         // quotient
v_readfirstlane_b32 s[sgprWorkGroup0], v11         // remainder

v_cmp_eq_f32 vcc, s[sgprAlpha], 0.0                // s[Alpha] == 0.0f ?
s_cbranch_vccz label_SKAlphaCheck                  // branch if s[Alpha] != 0
s_cmp_eq_u32 s[sgprStreamKLocalStart], 0           // does wg start tile?
s_cbranch_scc1 label_NoBranch_UR8VN3A1SJCPC6PO     // Only branch on scc0
s_getpc_b64 s[16:17]                               // addr of next instr
s_add_i32 s18, label_SK_CloseLoop, 4               // target branch offset
s_add_u32 s16, s16, s18                            // add target branch offset
s_addc_u32 s17, s17, 0                             // add high and carry
s_setpc_b64 s[16:17]                               // branch to label_SK_CloseLoop
label_NoBranch_UR8VN3A1SJCPC6PO:
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
v_cvt_f64_u32 v[10:11], s12                        // s13 = s[sgprWorkGroup0] / s12
v_rcp_f64 v[10:11], v[10:11]                       // s13 = s[sgprWorkGroup0] / s12
v_cvt_f64_u32 v[12:13], s[sgprWorkGroup0]          // s13 = s[sgprWorkGroup0] / s12
v_mul_f64 v[10:11], v[10:11], v[12:13]             // s13 = s[sgprWorkGroup0] / s12
v_cvt_u32_f64 v10, v[10:11]                        // s13 = s[sgprWorkGroup0] / s12
v_mul_lo_u32 v11, v10, s12                         // s13 = s[sgprWorkGroup0] / s12
v_sub_u32 v12, s[sgprWorkGroup0], v11              // s13 = s[sgprWorkGroup0] / s12
v_cmpx_ge_u32 exec, v12, s12                       // s13 = s[sgprWorkGroup0] / s12
v_add_u32 v10, v10, 1                              // s13 = s[sgprWorkGroup0] / s12
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s13, v10                       // quotient
s_mul_i32 s16, s13, s12                            // quotient * non-magic divisor
s_sub_u32 s16, s[sgprWorkGroup0], s16              // WorkGroup0=remainder
s_mul_i32 s16, s16, s[sgprNumWorkGroups1]          // (wg1 % WGM)*NumWorkGroups1
s_add_u32 s16, s16, s[sgprWorkGroup1]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups1
v_cvt_f64_u32 v[10:11], s12                        // s14 = s[sgprNumWorkGroups0] / s12
v_rcp_f64 v[10:11], v[10:11]                       // s14 = s[sgprNumWorkGroups0] / s12
v_cvt_f64_u32 v[12:13], s[sgprNumWorkGroups0]      // s14 = s[sgprNumWorkGroups0] / s12
v_mul_f64 v[10:11], v[10:11], v[12:13]             // s14 = s[sgprNumWorkGroups0] / s12
v_cvt_u32_f64 v10, v[10:11]                        // s14 = s[sgprNumWorkGroups0] / s12
v_mul_lo_u32 v11, v10, s12                         // s14 = s[sgprNumWorkGroups0] / s12
v_sub_u32 v12, s[sgprNumWorkGroups0], v11          // s14 = s[sgprNumWorkGroups0] / s12
v_cmpx_ge_u32 exec, v12, s12                       // s14 = s[sgprNumWorkGroups0] / s12
v_add_u32 v10, v10, 1                              // s14 = s[sgprNumWorkGroups0] / s12
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s14, v10                       // quotient
s_mul_i32 s15, s12, s14                            // quotient * non-magic divisor
s_sub_u32 s15, s[sgprNumWorkGroups0], s15          // NumWorkGroups0=remainder
s_cmp_eq_u32 s15, 0                                // remainder == 0 ?
s_cmov_b32 s15, s12                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s13, s14                              // blockId >= numFullBlocks ?
s_cselect_b32 s14, s15, s12
v_cvt_f64_u32 v[10:11], s14                        // s[sgprWorkGroup1] = s16 / s14
v_rcp_f64 v[10:11], v[10:11]                       // s[sgprWorkGroup1] = s16 / s14
v_cvt_f64_u32 v[12:13], s16                        // s[sgprWorkGroup1] = s16 / s14
v_mul_f64 v[10:11], v[10:11], v[12:13]             // s[sgprWorkGroup1] = s16 / s14
v_cvt_u32_f64 v10, v[10:11]                        // s[sgprWorkGroup1] = s16 / s14
v_mul_lo_u32 v11, v10, s14                         // s[sgprWorkGroup1] = s16 / s14
v_sub_u32 v12, s16, v11                            // s[sgprWorkGroup1] = s16 / s14
v_cmpx_ge_u32 exec, v12, s14                       // s[sgprWorkGroup1] = s16 / s14
v_add_u32 v10, v10, 1                              // s[sgprWorkGroup1] = s16 / s14
s_mov_b64 exec, -1                                 // Reset exec
v_mul_lo_u32 v11, v10, s14                         // s[sgprWorkGroup1] = s16 / s14
v_sub_u32 v12, s16, v11                            // s[sgprWorkGroup1] = s16 / s14
v_readfirstlane_b32 s[sgprWorkGroup1], v10         // quotient
v_readfirstlane_b32 s[sgprWorkGroup0], v12         // remainder
s_mul_i32 s[sgprWorkGroup0], s[sgprWorkGroup1], s14 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup0], s16, s[sgprWorkGroup0] // WorkGroup0=remainder
s_mul_i32 s13, s13, s12                            // blockId * WGM
s_add_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s13 // wg1 += blockId * WGM
s_branch label_WGM
label_WGMPositive:
s_mov_b32 s12, s12                                 // WGM
v_cvt_f64_u32 v[10:11], s12                        // s13 = s[sgprWorkGroup1] / s12
v_rcp_f64 v[10:11], v[10:11]                       // s13 = s[sgprWorkGroup1] / s12
v_cvt_f64_u32 v[12:13], s[sgprWorkGroup1]          // s13 = s[sgprWorkGroup1] / s12
v_mul_f64 v[10:11], v[10:11], v[12:13]             // s13 = s[sgprWorkGroup1] / s12
v_cvt_u32_f64 v10, v[10:11]                        // s13 = s[sgprWorkGroup1] / s12
v_mul_lo_u32 v11, v10, s12                         // s13 = s[sgprWorkGroup1] / s12
v_sub_u32 v12, s[sgprWorkGroup1], v11              // s13 = s[sgprWorkGroup1] / s12
v_cmpx_ge_u32 exec, v12, s12                       // s13 = s[sgprWorkGroup1] / s12
v_add_u32 v10, v10, 1                              // s13 = s[sgprWorkGroup1] / s12
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s13, v10                       // quotient
s_mul_i32 s16, s13, s12                            // quotient * non-magic divisor
s_sub_u32 s16, s[sgprWorkGroup1], s16              // WorkGroup1=remainder
s_mul_i32 s16, s16, s[sgprNumWorkGroups0]          // (wg1 % WGM)*NumWorkGroups0
s_add_u32 s16, s16, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups0
v_cvt_f64_u32 v[10:11], s12                        // s14 = s[sgprNumWorkGroups1] / s12
v_rcp_f64 v[10:11], v[10:11]                       // s14 = s[sgprNumWorkGroups1] / s12
v_cvt_f64_u32 v[12:13], s[sgprNumWorkGroups1]      // s14 = s[sgprNumWorkGroups1] / s12
v_mul_f64 v[10:11], v[10:11], v[12:13]             // s14 = s[sgprNumWorkGroups1] / s12
v_cvt_u32_f64 v10, v[10:11]                        // s14 = s[sgprNumWorkGroups1] / s12
v_mul_lo_u32 v11, v10, s12                         // s14 = s[sgprNumWorkGroups1] / s12
v_sub_u32 v12, s[sgprNumWorkGroups1], v11          // s14 = s[sgprNumWorkGroups1] / s12
v_cmpx_ge_u32 exec, v12, s12                       // s14 = s[sgprNumWorkGroups1] / s12
v_add_u32 v10, v10, 1                              // s14 = s[sgprNumWorkGroups1] / s12
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s14, v10                       // quotient
s_mul_i32 s15, s12, s14                            // quotient * non-magic divisor
s_sub_u32 s15, s[sgprNumWorkGroups1], s15          // NumWorkGroups1=remainder
s_cmp_eq_u32 s15, 0                                // remainder == 0 ?
s_cmov_b32 s15, s12                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s13, s14                              // blockId >= numFullBlocks ?
s_cselect_b32 s14, s15, s12
v_cvt_f64_u32 v[10:11], s14                        // s[sgprWorkGroup0] = s16 / s14
v_rcp_f64 v[10:11], v[10:11]                       // s[sgprWorkGroup0] = s16 / s14
v_cvt_f64_u32 v[12:13], s16                        // s[sgprWorkGroup0] = s16 / s14
v_mul_f64 v[10:11], v[10:11], v[12:13]             // s[sgprWorkGroup0] = s16 / s14
v_cvt_u32_f64 v10, v[10:11]                        // s[sgprWorkGroup0] = s16 / s14
v_mul_lo_u32 v11, v10, s14                         // s[sgprWorkGroup0] = s16 / s14
v_sub_u32 v12, s16, v11                            // s[sgprWorkGroup0] = s16 / s14
v_cmpx_ge_u32 exec, v12, s14                       // s[sgprWorkGroup0] = s16 / s14
v_add_u32 v10, v10, 1                              // s[sgprWorkGroup0] = s16 / s14
s_mov_b64 exec, -1                                 // Reset exec
v_mul_lo_u32 v11, v10, s14                         // s[sgprWorkGroup0] = s16 / s14
v_sub_u32 v12, s16, v11                            // s[sgprWorkGroup0] = s16 / s14
v_readfirstlane_b32 s[sgprWorkGroup0], v10         // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v12         // remainder
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
v_and_b32 v11, 63, v[vgprSerial]                   // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v10, 3, v11                              // 1. N offset: nIdx = wtid %% 4
v_and_b32 v15, 15, v11                             // 1. N offset: nIdx = wtid % MI_M(16)
v_lshrrev_b32 v15, 4, v15                          // 1. thread id in wave: k1Idx = mtid // 16
v_lshlrev_b32 v15, 4, v15                          // 1. K1 offset: lrK1Offset = k1Idx * mStride(128)
v_lshlrev_b32 v10, 2, v10                          // 1. N offset: nOffset = nIdx * nStride(4)
v_add_u32 v10, v15, v10                            // 1. offset in wave: lrOffset = bnOffset + lrKOffset
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_and_b32 v16, 15, v11                             // 5.1 thread id in wave: mtid = wtid %% 16
v_lshrrev_b32 v16, 2, v16                          // 5.2 thread id in wave: k1Idx = mtid // 4
v_lshrrev_b32 v11, 4, v11                          // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshlrev_b32 v11, 3, v11                          // 5. K offset: lrKOffset = kIdx * mStride(8)
v_add_u32 v11, v16, v11                            // 5.1 offset in wave: lrOffset = bnOffset + lrKOffset
/* Computing strided(8) perp indicies */
v_and_b32 v17, 3, v11                              // r0 = I % (32 // 8)
v_lshlrev_b32 v17, 3, v17                          // r0 = 8 * r0
/* Computing r1 = (I % 32) // (32 // 8) */
v_and_b32 v18, 31, v11                             // r1 = I % (32)
v_lshrrev_b32 v18, 2, v18                          // r1 = (r1) // (32 // 8)
v_add_u32 v17, v17, v18                            // r0 = r0 + r1
v_lshrrev_b32 v18, 5, v11                          // r1 = I // 32
v_lshl_add_u32 v11, v18, 5, v17                    // v11 = v18 * 32
/* Done computing strided(8) perp indices */
v_lshlrev_b32 v11, 7, v11                          // 5.2 K1 offset: lrK1Offset = k1Idx * mStride(128)
v_add_u32 v10, v11, v10                            // 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v14, 6, v[vgprSerial]                // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
v_and_b32 v14, 1, v14                              // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v10, v14, 4, v10                    // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(16); 7. final local read offset: flrOffset = lrOffset + WOffset
/* lr1J */
v_and_b32 v12, 63, v[vgprSerial]                   // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v11, 15, v12                             // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v11, 6, v11                          // 1. N offset: nOffset = nIdx * nStride(64)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
v_lshlrev_b32 v11, 2, v11                          // 4. apply VectorWidth: bnOffset = bnOffset * vw(4)
v_lshrrev_b32 v12, 4, v12                          // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshl_add_u32 v11, v12, 3, v11                    // 5. K offset: lrKOffset = kIdx * mStride(8); 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v13, 7, v[vgprSerial]                // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(128)
v_and_b32 v13, 1, v13                              // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v11, v13, 12, v11                   // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(4096); 7. final local read offset: flrOffset = lrOffset + WOffset

/* local read addresses: final offsets a */
v_lshrrev_b32 v12, 6, v[vgprSerial]                // 12 = Serial / 64
v_lshrrev_b32 v12, 2, v12                          // LSU offset: Get LSU wave_id
s_mov_b32 s12, 8192                                // LSU offset: stride = lsuStride(64)*(MT0(128) + PAD0(0))
v_mul_lo_u32 v12, s12, v12                         // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_lshl_u32 v[vgprLocalReadAddrA], v12, v10, 0x1 // Final Offset: offset = (lro0+lsuoffset)*bpeDS
v_lshrrev_b32 v13, 10, v[vgprLocalReadAddrA]       // Final Offset: padding 32 per block 1024
v_lshl_add_u32 v[vgprLocalReadAddrA], v13, 5, v[vgprLocalReadAddrA] // Final Offset: padding 32 per block 1024

/* local read addresses: final offsets b */
v_lshrrev_b32 v10, 6, v[vgprSerial]                // 10 = Serial / 64
v_lshrrev_b32 v10, 2, v10                          // LSU offset: Get LSU wave_id
s_mov_b32 s12, 64                                  // LSU offset: stride = lsuStride(64) when umlds==True
v_mul_lo_u32 v10, s12, v10                         // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_lshl_u32 v[vgprLocalReadAddrB], v10, v11, 0x1 // Final Offset: offset = (lro1+lsuoffset)*bpeDS
v_lshrrev_b32 v12, 10, v[vgprLocalReadAddrB]       // Final Offset: padding 32 per block 1024
v_lshl_add_u32 v[vgprLocalReadAddrB], v12, 5, v[vgprLocalReadAddrB] // Final Offset: padding 32 per block 1024

/* local read addresses: declare addresses a */
/* N/A */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, 0x4200, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

// Save LDS local address
v_mov_b32 v[vgprLocalReadAddrARef], v[vgprLocalReadAddrA]        
v_mov_b32 v[vgprLocalReadAddrBRef], v[vgprLocalReadAddrB]        


/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 16 */
/* v11 = A-unroll = serial/LVCA */
v_lshrrev_b32 v11, 4, v[vgprSerial]                // 11 = Serial / 16
v_and_b32 v10, 15, v[vgprSerial]                   // 10 = Serial % 16
/* tile *= glvw */
v_lshlrev_b32 v10, 3, v10                          // v10 = v10 * 8
v_mov_b32 v14, v11                                 // copy for GlobalSplitU
/* LVCB = 8 */
/* v13 = B-unroll = serial%LVCB */
v_lshrrev_b32 v12, 3, v[vgprSerial]                // 12 = Serial / 8
v_and_b32 v13, 7, v[vgprSerial]                    // 13 = Serial % 8
/* unroll *= glvw */
v_lshlrev_b32 v13, 3, v13                          // v13 = v13 * 8
v_mov_b32 v15, v13                                 // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v14 */
/* lwaUnrollAssignmentB = v15 */

/* local write addresses: first offset a */
v_mul_u32_u24 v16, 0x80, v14                       // lwAL**(MTA + PAD)
v_add_lshl_u32 v16, v10, v16, 0x1                  // lwFOA = (lwAA + lwAL*(MT0I+PAD))*bpeDS
v_lshrrev_b32 v18, 10, v16                         // padding 32 per block 1024
v_lshl_add_u32 v16, v18, 5, v16                    // padding 32 per block 1024
v_lshrrev_b32 v17, 6, v[vgprSerial]                // Compute waveID
s_nop 0                                            // 1 wait states required before reading vgpr by lane
v_readfirstlane_b32 s[sgprLocalWriteAddrA], v17    // Copy lds write address VGPR to SGPR
s_mul_i32 s[sgprLocalWriteAddrA], s[sgprLocalWriteAddrA], 1056

/* local write addresses: first offset b */
v_mul_u32_u24 v16, 0x40, v12                       // lwBL**(DepthU_Compute + PAD)
v_add_lshl_u32 v16, v15, v16, 0x1                  // lwFOB = (lwBB + lwBL*(DepthU+PAD))*bpeDS
v_lshrrev_b32 v18, 10, v16                         // padding 32 per block 1024
v_lshl_add_u32 v16, v18, 5, v16                    // padding 32 per block 1024
v_add_co_u32 v16, vcc, 0x4200, v16                 // lwFOB = lwB1J + lwBL*MT1J + LDS_OFFSET_B=16896
v_lshrrev_b32 v17, 6, v[vgprSerial]                // Compute waveID
s_nop 0                                            // 1 wait states required before reading vgpr by lane
v_readfirstlane_b32 s[sgprLocalWriteAddrB], v17    // Copy lds write address VGPR to SGPR
s_mul_i32 s[sgprLocalWriteAddrB], s[sgprLocalWriteAddrB], 1056
s_add_u32 s[sgprLocalWriteAddrB], s[sgprLocalWriteAddrB], 16896

// Save base LDS addresses
s_mov_b32 s[sgprAddrARef], s[sgprLocalWriteAddrA]
s_mov_b32 s[sgprAddrBRef], s[sgprLocalWriteAddrB]
s_mov_b32 s[sgprOffsetW], 0
s_mov_b32 s[sgprOffsetR], 0

/* global read addresses: tile offset assignment a */
/* graTileAssignmentA = v10 */

/* global read addresses: tile offset assignment b */
/* graTileAssignmentB = v12 */

/* global read addresses: unroll assignment a */
/* v11 */

/* global read addresses: unroll assignment b */
/* v13 */

/* global read addresses: other free assignments */
/* s[sgprWorkGroup2] */

/* global read addresses: tile offsets a */
v_mov_b32 v16, v10                                 // groA0I_0

/* global read addresses: tile offsets b */
v_mov_b32 v17, v12                                 // groB1J_0
v_add_co_u32 v18, vcc, 32, v17                     // groB1J_1 += LSPB
v_add_co_u32 v19, vcc, 32, v18                     // groB1J_2 += LSPB
v_add_co_u32 v20, vcc, 32, v19                     // groB1J_3 += LSPB

/* global read addresses: unroll offsets a */
v_mov_b32 v21, v11                                 // groAL_0
v_add_co_u32 v22, vcc, 16, v21                     // groAL_1 + LSPA
v_add_co_u32 v23, vcc, 16, v22                     // groAL_2 + LSPA
v_add_co_u32 v24, vcc, 16, v23                     // groAL_3 + LSPA

/* global read addresses: unroll offsets b */
v_mov_b32 v25, v13                                 // groBL_0

/* global read addresses: addresses a */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s15, s[sgprWorkGroup0], 128           // WorkGroup[01] * MT
s_mul_i32 s14, s[sgprWorkGroup0], 128              // WorkGroup[01] * MT
s_mul_i32 s12, s[sgprStreamKLocalStart], DepthU    // StreamK tile start offset
s_mul_hi_u32 s13, s12, s[sgprStrideAL]             // StreamK tile start offset
s_mul_i32 s12, s12, s[sgprStrideAL]                // StreamK tile start offset
s_add_u32 s14, s14, s12                            // accum GsuOffset term to tilestart
s_addc_u32 s15, s15, s13                           // accum GsuOffset term to tilestart
s_mov_b64 s[sgprShadowLimitA+0:sgprShadowLimitA+0+1], 1 // Init tensor size
s_sub_u32 s12, s[sgprSizeI], 1                     // (size-1)
s_mul_hi_u32 s13, constStrideA0I, s12              // stride x (size-1)
s_mul_i32 s12, constStrideA0I, s12                 // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s12 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s13 // sum tensor size
s_sub_u32 s12, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s13, s[sgprStrideAL], s12             // stride x (size-1)
s_mul_i32 s12, s[sgprStrideAL], s12                // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s12 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s13 // sum tensor size
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s14 // sub tileStart
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s15 // sub tileStart
s_lshl_b64 s[sgprShadowLimitA:sgprShadowLimitA+1], s[sgprShadowLimitA:sgprShadowLimitA+1], 0x1 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_mul_hi_u32 s13, s[sgprStrideAK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s12, s[sgprStrideAK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s14, s14, s12                            // accum wg term to tilestart
s_addc_u32 s15, s15, s13                           // accum wg term to tilestart
s_lshl_b64 s[14:15], s[14:15], 1                   // tileStart *= BPE
s_add_u32 s[sgprSrdA+0], s[sgprAddressA+0], s14    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprAddressA+1], s15   // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdA+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: addresses b */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s15, s[sgprWorkGroup1], 128           // WorkGroup[01] * MT
s_mul_i32 s14, s[sgprWorkGroup1], 128              // WorkGroup[01] * MT
s_mul_hi_u32 s15, s14, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s14, s14, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_mul_i32 s12, s[sgprStreamKLocalStart], DepthU    // StreamK tile start offset
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
s_lshl_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], 0x1 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_mul_hi_u32 s13, s[sgprStrideBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s12, s[sgprStrideBK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s14, s14, s12                            // accum wg term to tilestart
s_addc_u32 s15, s15, s13                           // accum wg term to tilestart
s_lshl_b64 s[14:15], s[14:15], 1                   // tileStart *= BPE
s_add_u32 s[sgprSrdB+0], s[sgprAddressB+0], s14    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprAddressB+1], s15   // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdB+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: final offsets a */
// Using GLNC for A
/* NumThreadsCoalescedA = 16, 256 total threads, 8 thread groups */
v_mov_b32 v[vgprGlobalReadOffsetA+0], v[vgprSerial]
v_add_u32 v[vgprGlobalReadOffsetA+1], 256, v[vgprGlobalReadOffsetA+0] //  = vgprSerial + 1 * 256
v_add_u32 v[vgprGlobalReadOffsetA+2], 256, v[vgprGlobalReadOffsetA+1] //  = vgprSerial + 2 * 256
v_add_u32 v[vgprGlobalReadOffsetA+3], 256, v[vgprGlobalReadOffsetA+2] //  = vgprSerial + 3 * 256
s_mul_i32 s14, s[sgprWorkGroup0], 128              // WorkGroup[01] * MT
s_sub_u32 s14, s[sgprSizeI], s14                   // edge = Size0I - WG*MT
s_sub_u32 s14, s14, 8                              // edge -= margin(8)
v_lshrrev_b32 v30, 4, v[vgprGlobalReadOffsetA+0]   // division
v_and_b32 v29, 0xf, v[vgprGlobalReadOffsetA+0]
/* Computing strided(4) perp indicies */
v_and_b32 v32, 7, v30                              // r0 = I % (32 // 4)
v_lshlrev_b32 v32, 2, v32                          // r0 = 4 * r0
/* Computing r1 = (I % 32) // (32 // 4) */
v_and_b32 v33, 31, v30                             // r1 = I % (32)
v_lshrrev_b32 v33, 3, v33                          // r1 = (r1) // (32 // 4)
v_add_u32 v32, v32, v33                            // r0 = r0 + r1
v_lshrrev_b32 v33, 5, v30                          // r1 = I // 32
v_lshl_add_u32 v30, v33, 5, v32                    // v30 = v33 * 32
/* Done computing strided(4) perp indices */
v_lshlrev_b32 v[vgprGlobalReadOffsetA+0], 3, v29
v_mul_lo_u32 v30, s[sgprStridesA], v30
v_min_i32 v[vgprGlobalReadOffsetA+0], s14, v[vgprGlobalReadOffsetA+0]
v_add_u32 v[vgprGlobalReadOffsetA+0], v30, v[vgprGlobalReadOffsetA+0] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetA+0], 1, v[vgprGlobalReadOffsetA+0]
v_add_u32 v[vgprGlobalReadOffsetA+0], 16, v[vgprGlobalReadOffsetA+0] // ptr-shift
v_lshrrev_b32 v30, 4, v[vgprGlobalReadOffsetA+1]   // division
v_and_b32 v29, 0xf, v[vgprGlobalReadOffsetA+1]
/* Computing strided(4) perp indicies */
v_and_b32 v32, 7, v30                              // r0 = I % (32 // 4)
v_lshlrev_b32 v32, 2, v32                          // r0 = 4 * r0
/* Computing r1 = (I % 32) // (32 // 4) */
v_and_b32 v33, 31, v30                             // r1 = I % (32)
v_lshrrev_b32 v33, 3, v33                          // r1 = (r1) // (32 // 4)
v_add_u32 v32, v32, v33                            // r0 = r0 + r1
v_lshrrev_b32 v33, 5, v30                          // r1 = I // 32
v_lshl_add_u32 v30, v33, 5, v32                    // v30 = v33 * 32
/* Done computing strided(4) perp indices */
v_lshlrev_b32 v[vgprGlobalReadOffsetA+1], 3, v29
v_mul_lo_u32 v30, s[sgprStridesA], v30
v_min_i32 v[vgprGlobalReadOffsetA+1], s14, v[vgprGlobalReadOffsetA+1]
v_add_u32 v[vgprGlobalReadOffsetA+1], v30, v[vgprGlobalReadOffsetA+1] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetA+1], 1, v[vgprGlobalReadOffsetA+1]
v_add_u32 v[vgprGlobalReadOffsetA+1], 16, v[vgprGlobalReadOffsetA+1] // ptr-shift
v_lshrrev_b32 v30, 4, v[vgprGlobalReadOffsetA+2]   // division
v_and_b32 v29, 0xf, v[vgprGlobalReadOffsetA+2]
/* Computing strided(4) perp indicies */
v_and_b32 v32, 7, v30                              // r0 = I % (32 // 4)
v_lshlrev_b32 v32, 2, v32                          // r0 = 4 * r0
/* Computing r1 = (I % 32) // (32 // 4) */
v_and_b32 v33, 31, v30                             // r1 = I % (32)
v_lshrrev_b32 v33, 3, v33                          // r1 = (r1) // (32 // 4)
v_add_u32 v32, v32, v33                            // r0 = r0 + r1
v_lshrrev_b32 v33, 5, v30                          // r1 = I // 32
v_lshl_add_u32 v30, v33, 5, v32                    // v30 = v33 * 32
/* Done computing strided(4) perp indices */
v_lshlrev_b32 v[vgprGlobalReadOffsetA+2], 3, v29
v_mul_lo_u32 v30, s[sgprStridesA], v30
v_min_i32 v[vgprGlobalReadOffsetA+2], s14, v[vgprGlobalReadOffsetA+2]
v_add_u32 v[vgprGlobalReadOffsetA+2], v30, v[vgprGlobalReadOffsetA+2] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetA+2], 1, v[vgprGlobalReadOffsetA+2]
v_add_u32 v[vgprGlobalReadOffsetA+2], 16, v[vgprGlobalReadOffsetA+2] // ptr-shift
v_lshrrev_b32 v30, 4, v[vgprGlobalReadOffsetA+3]   // division
v_and_b32 v29, 0xf, v[vgprGlobalReadOffsetA+3]
/* Computing strided(4) perp indicies */
v_and_b32 v32, 7, v30                              // r0 = I % (32 // 4)
v_lshlrev_b32 v32, 2, v32                          // r0 = 4 * r0
/* Computing r1 = (I % 32) // (32 // 4) */
v_and_b32 v33, 31, v30                             // r1 = I % (32)
v_lshrrev_b32 v33, 3, v33                          // r1 = (r1) // (32 // 4)
v_add_u32 v32, v32, v33                            // r0 = r0 + r1
v_lshrrev_b32 v33, 5, v30                          // r1 = I // 32
v_lshl_add_u32 v30, v33, 5, v32                    // v30 = v33 * 32
/* Done computing strided(4) perp indices */
v_lshlrev_b32 v[vgprGlobalReadOffsetA+3], 3, v29
v_mul_lo_u32 v30, s[sgprStridesA], v30
v_min_i32 v[vgprGlobalReadOffsetA+3], s14, v[vgprGlobalReadOffsetA+3]
v_add_u32 v[vgprGlobalReadOffsetA+3], v30, v[vgprGlobalReadOffsetA+3] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetA+3], 1, v[vgprGlobalReadOffsetA+3]
v_add_u32 v[vgprGlobalReadOffsetA+3], 16, v[vgprGlobalReadOffsetA+3] // ptr-shift

/* global read addresses: final offsets b */
// Using GLNC for B
/* NumThreadsCoalescedB = 8, 256 total threads, 2 thread groups */
v_mov_b32 v[vgprGlobalReadOffsetB+0], v[vgprSerial]
v_add_u32 v[vgprGlobalReadOffsetB+1], 256, v[vgprGlobalReadOffsetB+0] //  = vgprSerial + 1 * 256
v_add_u32 v[vgprGlobalReadOffsetB+2], 256, v[vgprGlobalReadOffsetB+1] //  = vgprSerial + 2 * 256
v_add_u32 v[vgprGlobalReadOffsetB+3], 256, v[vgprGlobalReadOffsetB+2] //  = vgprSerial + 3 * 256
v_lshrrev_b32 v10, 3, v[vgprGlobalReadOffsetB+0]   // division
v_and_b32 v14, 0x7, v[vgprGlobalReadOffsetB+0]
v_lshlrev_b32 v[vgprGlobalReadOffsetB+0], 3, v14
v_mul_lo_u32 v10, s[sgprStridesB], v10
v_add_u32 v[vgprGlobalReadOffsetB+0], v10, v[vgprGlobalReadOffsetB+0] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetB+0], 1, v[vgprGlobalReadOffsetB+0]
v_add_u32 v[vgprGlobalReadOffsetB+0], 16, v[vgprGlobalReadOffsetB+0] // ptr-shift
v_lshrrev_b32 v10, 3, v[vgprGlobalReadOffsetB+1]   // division
v_and_b32 v14, 0x7, v[vgprGlobalReadOffsetB+1]
v_lshlrev_b32 v[vgprGlobalReadOffsetB+1], 3, v14
v_mul_lo_u32 v10, s[sgprStridesB], v10
v_add_u32 v[vgprGlobalReadOffsetB+1], v10, v[vgprGlobalReadOffsetB+1] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetB+1], 1, v[vgprGlobalReadOffsetB+1]
v_add_u32 v[vgprGlobalReadOffsetB+1], 16, v[vgprGlobalReadOffsetB+1] // ptr-shift
v_lshrrev_b32 v10, 3, v[vgprGlobalReadOffsetB+2]   // division
v_and_b32 v14, 0x7, v[vgprGlobalReadOffsetB+2]
v_lshlrev_b32 v[vgprGlobalReadOffsetB+2], 3, v14
v_mul_lo_u32 v10, s[sgprStridesB], v10
v_add_u32 v[vgprGlobalReadOffsetB+2], v10, v[vgprGlobalReadOffsetB+2] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetB+2], 1, v[vgprGlobalReadOffsetB+2]
v_add_u32 v[vgprGlobalReadOffsetB+2], 16, v[vgprGlobalReadOffsetB+2] // ptr-shift
v_lshrrev_b32 v10, 3, v[vgprGlobalReadOffsetB+3]   // division
v_and_b32 v14, 0x7, v[vgprGlobalReadOffsetB+3]
v_lshlrev_b32 v[vgprGlobalReadOffsetB+3], 3, v14
v_mul_lo_u32 v10, s[sgprStridesB], v10
v_add_u32 v[vgprGlobalReadOffsetB+3], v10, v[vgprGlobalReadOffsetB+3] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetB+3], 1, v[vgprGlobalReadOffsetB+3]
v_add_u32 v[vgprGlobalReadOffsetB+3], 16, v[vgprGlobalReadOffsetB+3] // ptr-shift

/* global read addresses: increments a */
s_mul_i32 s[sgprGlobalReadIncsA+0], DepthU*BpeAGR, s[sgprStrideAL] // incrA unrollIdx)

/* global read addresses: increments b */
s_mov_b32 s[sgprGlobalReadIncsB+0], DepthU*BpeBGR  // incrB (unrollIdx)
/* declare loop num iterations */
s_sub_u32 s[sgprLoopCounterL], s[sgprStreamKLocalEnd], s[sgprStreamKLocalStart] // StreamK loop counter = localEnd - localStart
v_cmp_eq_f32 vcc, s[sgprAlpha], 0.0                // s[Alpha] == 0.0f ?
s_cbranch_vccz label_SKAlphaCheck2                 // branch if s[Alpha] != 0
s_mov_b32 s[sgprLoopCounterL], 0                   // Skip iterations
label_SKAlphaCheck2:
s_and_b32 s13, 63, s[sgprSizesSum+0]               // s13 = s[sgprSizesSum+0] % 64
s_cmp_eq_u32 s13, 0                                // numIterL == 0
s_cselect_b32 s12, 0, 1                            // check if size uses tail loop
s_cmp_eq_u32 s[sgprStreamKLocalEnd], s[sgprItersPerTile] // Check if WG processes final iteration of tile
s_cselect_b32 s12, s12, 0                          // this WG runs tail loop
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], s12 // Adjust loop counter for tail loop
s_mov_b32 s[sgprOrigLoopCounter], s[sgprLoopCounterL] // copy loop counter
s_and_b32 s14, s[sgprStaggerU], 0x1f00
s_lshr_b32 s14, s14, 0x8
s_and_b32 s15, s[sgprStaggerU], 0xe000
s_and_b32 s[sgprStaggerU], s[sgprStaggerU], 0xff
s_mov_b32 s12, s[sgprStaggerU]                     // init staggerU
label_beginStaggerUIter:
s_lshl_b32 s13, s12, s14                           // shift by StaggerUStride
s_cmp_ge_u32 s[sgprOrigLoopCounter], s13           // loopCount >= current shift Count
s_cbranch_scc1 label_endStaggerUIter               // jump to end
s_lshr_b32 s12, s12, 1                             // step down to smaller stagger
s_branch label_beginStaggerUIter                   // jump to begin
label_endStaggerUIter:
s_sub_u32 s13, s12, 1                              // staggerU mask
s_cmp_ge_u32 s12, 1                                // if current staggerU >= 1
s_cselect_b32 s[sgprStaggerUIter], s13, 0          // set Mask
s_cmp_eq_u32 s15, 0x0
s_cbranch_scc1 label_StaggerUMapping_1
s_mov_b32 s12, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping_1:
s_cmp_eq_u32 s15, 0x2000
s_cbranch_scc1 label_StaggerUMapping_2
s_mov_b32 s12, s[sgprWorkGroup1]
s_branch label_staggerInputEnd
label_StaggerUMapping_2:
s_cmp_eq_u32 s15, 0x4000
s_cbranch_scc1 label_StaggerUMapping_3
s_mov_b32 s12, -0x1
s_branch label_staggerInputEnd
label_StaggerUMapping_3:
s_cmp_eq_u32 s15, 0x6000
s_cbranch_scc1 label_StaggerUMapping_4
s_mul_i32 s13, s[sgprNumWorkGroups0], s[sgprWorkGroup1]
s_add_u32 s12, s12, s13
s_add_u32 s12, s12, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping_4:
s_cmp_eq_u32 s15, 0x8000
s_cbranch_scc1 label_staggerInputEnd
s_mov_b32 s12, -0x1
s_branch label_staggerInputEnd
label_staggerInputEnd:
s_and_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s12 // Compute actual stagger start for this tile
s_lshl_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s14 // shift by StaggerUStride
s_cmp_gt_u32 s[sgprStreamKLocalStart], 0           // does wg start tile?
s_cmov_b32 s[sgprStaggerUIter], 0                  // set stagger=0 for partial tiles
s_cmp_lt_u32 s[sgprStreamKLocalEnd], s[sgprItersPerTile] // does wg finish tile?
s_cmov_b32 s[sgprStaggerUIter], 0                  // set stagger=0 for partial tiles

/* SRDs += (StaggerUIter) * GlobalReadIncsA+0 */
s_mul_hi_i32 s13, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_i32 s12, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_hi_i32 s[sgprWrapUA+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUA+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0], s[sgprWrapUA+0] // remove one iteration
s_subb_u32 s[sgprWrapUA+1], 0, s[sgprWrapUA+1]     // remove one iteration
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s12        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s13       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s12 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s13 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* SRDs += (StaggerUIter) * GlobalReadIncsB+0 */
s_mul_hi_i32 s13, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_i32 s12, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_hi_i32 s[sgprWrapUB+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUB+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0], s[sgprWrapUB+0] // remove one iteration
s_subb_u32 s[sgprWrapUB+1], 0, s[sgprWrapUB+1]     // remove one iteration
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s12        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s13       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s12 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s13 // limit -= inc)
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
s_mov_b32 m0, s[sgprLocalWriteAddrA]               // m0 <- LDS write address
/* before DirectToLds load, ensure prior ds_reads have finished */
s_waitcnt lgkmcnt(0)
s_barrier
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_0_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_1_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_2_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_3_0
s_mov_b32 m0, s[sgprLocalWriteAddrB]               // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_0_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_1_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_2_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_3_0

/* global read inc A loopL */
s_add_u32 s14, s[sgprLoopCounterL], 1              // remove pf(1)
s_cmp_eq_u32 s[sgprStaggerUIter], s14              // Is this wrapIter? (pf)
s_cselect_b32 s12, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s13, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s12        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s13       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s12 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s13 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s14, s[sgprLoopCounterL], 1              // remove pf(1)
s_cmp_eq_u32 s[sgprStaggerUIter], s14              // Is this wrapIter? (pf)
s_cselect_b32 s12, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s13, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s12        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s13       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s12 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s13 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32

/******************************************/
/* End setupNewTile                       */
/******************************************/
label_ShadowInitStart:
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], s[sgprAddressD+0:sgprAddressD+0+1] // init SRD base address
s_mov_b32 s[sgprSrdD+2], BufferOOB
s_mov_b32 s[sgprSrdD+3], Srd127_96                 // Set bits 127_96 in post-loop SRD

s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], s[sgprAddressC+0:sgprAddressC+0+1] // init SRD base address
s_mov_b32 s[sgprSrdC+2], BufferOOB
s_mov_b32 s[sgprSrdC+3], Srd127_96                 // Set bits 127_96 in post-loop SRD

s_mov_b32 s60, 1
s_mov_b32 s61, 1
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_BPEDone                       // If synchronizer, use regular output BPE
s_cmp_eq_u32 s[sgprskTiles], 1                     // split == 1 ?
s_cbranch_scc1 label_BPEDone                       // If split == 1, use reguler output BPE
s_mov_b32 s60, 1
s_mov_b32 s61, 2
label_BPEDone:

s_mul_i32 s86, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_mul_hi_u32 s85, s86, s[sgprStrideC1J]            // ScaleC s86 by Stride
s_mul_i32 s84, s86, s[sgprStrideC1J]               // ScaleC s86 by Stride
s_lshl_b64 s[84:85], s[84:85], s60                 // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprAddressC+0], s84    // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprAddressC+1], s85   // add hi to SRD
s_mul_hi_u32 s85, s86, s[sgprStrideD1J]            // ScaleD s86 by Stride
s_mul_i32 s84, s86, s[sgprStrideD1J]               // ScaleD s86 by Stride
s_lshl_b64 s[84:85], s[84:85], s61                 // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprAddressD+0], s84    // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprAddressD+1], s85   // add hi to SRD

s_mul_hi_u32 s85, s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride
s_mul_i32 s84, s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride
s_lshl_b64 s[84:85], s[84:85], s60                 // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s84        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s85       // add hi to SRD
s_mul_hi_u32 s85, s[sgprWorkGroup2], s[sgprStrideDK] // ScaleD s[sgprWorkGroup2] by Stride
s_mul_i32 s84, s[sgprWorkGroup2], s[sgprStrideDK]  // ScaleD s[sgprWorkGroup2] by Stride
s_lshl_b64 s[84:85], s[84:85], s61                 // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s84        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s85       // add hi to SRD

s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_SK_SplitSrd                   // Skip this block if using single-kernel stream-k fixup
s_cmp_eq_u32 s[sgprskTiles], 1                     // split == 1 ?
s_cbranch_scc1 label_SK_SplitSrd                   // branch if split == 1
// Split Output Buffer offset: Free0 + (Free1-1)*StrideC1J + (Free2-1)*StrideCK * SplitIdx * bpe%s
s_mul_hi_u32 s85, s[sgprSizesFree+0], s[sgprSkPartialIdx] // Free0
s_mul_i32 s84, s[sgprSizesFree+0], s[sgprSkPartialIdx] // Free0
s_sub_u32 s83, s[sgprSizesFree+1], 1               // Free1
s_mul_i32 s83, s83, s[sgprSkPartialIdx]            // Free1
s_mul_hi_u32 s86, s83, s[sgprStrideC1J]            // Free1
s_mul_i32 s83, s83, s[sgprStrideC1J]               // Free1
s_add_u32 s84, s84, s83                            // Free1
s_addc_u32 s85, s85, s86                           // Free1
s_sub_u32 s83, s[sgprSizesFree+2], 1               // Free2
s_mul_i32 s83, s83, s[sgprSkPartialIdx]            // Free2
s_mul_hi_u32 s86, s83, s[sgprStrideCK]             // Free2
s_mul_i32 s83, s83, s[sgprStrideCK]                // Free2
s_add_u32 s84, s84, s83                            // Free2
s_addc_u32 s85, s85, s86                           // Free2
s_lshl_b64 s[84:85], s[84:85], 2                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s84        // add lo GSU offset to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s85       // add hi GSU offset to SRD
label_SK_SplitSrd:

/* initC: remove ValuC vgpr buffer [0...0) from pool */

/* initC: remove acc vgpr buffer [0...64) from pool */

/* initC: remove ValuA/B vgpr buffer [10...74) from pool */
v_accvgpr_write acc0, 0                            // initC
v_accvgpr_write acc1, 0                            // initC
v_accvgpr_write acc2, 0                            // initC
v_accvgpr_write acc3, 0                            // initC
v_accvgpr_write acc4, 0                            // initC
v_accvgpr_write acc5, 0                            // initC
v_accvgpr_write acc6, 0                            // initC
v_accvgpr_write acc7, 0                            // initC
v_accvgpr_write acc8, 0                            // initC
v_accvgpr_write acc9, 0                            // initC
v_accvgpr_write acc10, 0                           // initC
v_accvgpr_write acc11, 0                           // initC
v_accvgpr_write acc12, 0                           // initC
v_accvgpr_write acc13, 0                           // initC
v_accvgpr_write acc14, 0                           // initC
v_accvgpr_write acc15, 0                           // initC
v_accvgpr_write acc16, 0                           // initC
v_accvgpr_write acc17, 0                           // initC
v_accvgpr_write acc18, 0                           // initC
v_accvgpr_write acc19, 0                           // initC
v_accvgpr_write acc20, 0                           // initC
v_accvgpr_write acc21, 0                           // initC
v_accvgpr_write acc22, 0                           // initC
v_accvgpr_write acc23, 0                           // initC
v_accvgpr_write acc24, 0                           // initC
v_accvgpr_write acc25, 0                           // initC
v_accvgpr_write acc26, 0                           // initC
v_accvgpr_write acc27, 0                           // initC
v_accvgpr_write acc28, 0                           // initC
v_accvgpr_write acc29, 0                           // initC
v_accvgpr_write acc30, 0                           // initC
v_accvgpr_write acc31, 0                           // initC
v_accvgpr_write acc32, 0                           // initC
v_accvgpr_write acc33, 0                           // initC
v_accvgpr_write acc34, 0                           // initC
v_accvgpr_write acc35, 0                           // initC
v_accvgpr_write acc36, 0                           // initC
v_accvgpr_write acc37, 0                           // initC
v_accvgpr_write acc38, 0                           // initC
v_accvgpr_write acc39, 0                           // initC
v_accvgpr_write acc40, 0                           // initC
v_accvgpr_write acc41, 0                           // initC
v_accvgpr_write acc42, 0                           // initC
v_accvgpr_write acc43, 0                           // initC
v_accvgpr_write acc44, 0                           // initC
v_accvgpr_write acc45, 0                           // initC
v_accvgpr_write acc46, 0                           // initC
v_accvgpr_write acc47, 0                           // initC
v_accvgpr_write acc48, 0                           // initC
v_accvgpr_write acc49, 0                           // initC
v_accvgpr_write acc50, 0                           // initC
v_accvgpr_write acc51, 0                           // initC
v_accvgpr_write acc52, 0                           // initC
v_accvgpr_write acc53, 0                           // initC
v_accvgpr_write acc54, 0                           // initC
v_accvgpr_write acc55, 0                           // initC
v_accvgpr_write acc56, 0                           // initC
v_accvgpr_write acc57, 0                           // initC
v_accvgpr_write acc58, 0                           // initC
v_accvgpr_write acc59, 0                           // initC
v_accvgpr_write acc60, 0                           // initC
v_accvgpr_write acc61, 0                           // initC
v_accvgpr_write acc62, 0                           // initC
v_accvgpr_write acc63, 0                           // initC
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?

/* after InitC, skip to end of prefetch last iter if numIter==0 */
s_cbranch_scc0 label_NoBranch_8S4L1KCK9VFC7AQU     // Only branch on scc1
s_getpc_b64 s[60:61]                               // addr of next instr
s_add_i32 s62, label_PrefetchGlobalLastIterEnd, 4  // target branch offset
s_add_u32 s60, s60, s62                            // add target branch offset
s_addc_u32 s61, s61, 0                             // add high and carry
s_setpc_b64 s[60:61]                               // branch to label_PrefetchGlobalLastIterEnd
label_NoBranch_8S4L1KCK9VFC7AQU:
s_waitcnt vmcnt(0)                                 // wait for global read
s_barrier                                          // For stream-k / persistent loop

/* local write a */

/* local write b */

/* local write swap a */
/* local write swap b */
s_mov_b32 s[sgprOffsetR], s[sgprOffsetW]
s_add_u32 s[sgprOffsetW], s[sgprOffsetW], LDSBufferSize
s_cmp_ge_u32 s[sgprOffsetW], 3*LDSBufferSize
s_cselect_b32 s[sgprOffsetW], 0, s[sgprOffsetW]

s_add_u32 s[sgprLocalWriteAddrA], s[sgprAddrARef], s[sgprOffsetW]
s_add_u32 s[sgprLocalWriteAddrB], s[sgprAddrBRef], s[sgprOffsetW]


s_cmp_eq_u32 s[sgprLoopCounterL], 0x1              // PGR=2 but only 1 loop
s_cbranch_scc1 label_skipPGR2                      // PGR=2 but only 1 loop
s_mov_b32 m0, s[sgprLocalWriteAddrA]               // m0 <- LDS write address
/* before DirectToLds load, ensure prior ds_reads have finished */
s_waitcnt lgkmcnt(0)
s_barrier
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_0_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_1_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_2_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_3_0
s_mov_b32 m0, s[sgprLocalWriteAddrB]               // m0 <- LDS write address
/* before DirectToLds load, ensure prior ds_reads have finished */
s_waitcnt lgkmcnt(0)
s_barrier
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_0_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_1_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_2_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_3_0

/* local write swap a */
/* local write swap b */

s_mov_b32 s[sgprOffsetR], s[sgprOffsetW]
s_add_u32 s[sgprOffsetW], s[sgprOffsetW], LDSBufferSize
s_cmp_ge_u32 s[sgprOffsetW], 3*LDSBufferSize
s_cselect_b32 s[sgprOffsetW], 0, s[sgprOffsetW]
s_add_u32 s[sgprLocalWriteAddrA], s[sgprAddrARef], s[sgprOffsetW]
s_add_u32 s[sgprLocalWriteAddrB], s[sgprAddrBRef], s[sgprOffsetW]

label_skipPGR2:
// Skip force waitcnt0
s_barrier

/* local read prefetch a */
ds_read_b64_tr_b16 v[vgprValuA_X0_I0+0+0:vgprValuA_X0_I0+0+0+1], v[vgprLocalReadAddrA+0] offset:0 // LDS Transpose
ds_read_b64_tr_b16 v[vgprValuA_X0_I0+0+2:vgprValuA_X0_I0+0+2+1], v[vgprLocalReadAddrA+0] offset:256 // LDS Transpose
ds_read_b64_tr_b16 v[vgprValuA_X0_I0+4+0:vgprValuA_X0_I0+4+0+1], v[vgprLocalReadAddrA+0] offset:64 // LDS Transpose
ds_read_b64_tr_b16 v[vgprValuA_X0_I0+4+2:vgprValuA_X0_I0+4+2+1], v[vgprLocalReadAddrA+0] offset:320 // LDS Transpose
ds_read_b64_tr_b16 v[vgprValuA_X0_I0+8+0:vgprValuA_X0_I0+8+0+1], v[vgprLocalReadAddrA+0] offset:128 // LDS Transpose
ds_read_b64_tr_b16 v[vgprValuA_X0_I0+8+2:vgprValuA_X0_I0+8+2+1], v[vgprLocalReadAddrA+0] offset:384 // LDS Transpose
ds_read_b64_tr_b16 v[vgprValuA_X0_I0+12+0:vgprValuA_X0_I0+12+0+1], v[vgprLocalReadAddrA+0] offset:192 // LDS Transpose
ds_read_b64_tr_b16 v[vgprValuA_X0_I0+12+2:vgprValuA_X0_I0+12+2+1], v[vgprLocalReadAddrA+0] offset:448 // LDS Transpose

/* local read prefetch b */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB] offset:384 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read inc a */
/* N/A, lro->4096 */
/* self.localReadDoCntA 1 self.localReadDoCntB 1 */

/* local read inc b */
/* N/A, lro->32 */
/* self.localReadDoCntA 1 self.localReadDoCntB 1 */

/******************************************/
/* Unrolled Loop(s) - Begin               */
/******************************************/
label_openLoopL:
s_cmp_eq_u32 s[sgprLoopCounterL], 0x1              // LoopCounterL < EndCounter
s_cbranch_scc1 label_toPGR1                        // PGR=2 but only 1 loop, toPGR1
s_cmp_le_u32 s[sgprLoopCounterL], 0x2              // LoopCounterL < EndCounter
s_cbranch_scc1 label_LoopEndL                      // do not enter LoopL
.align 16
label_LoopBeginL:

/******************************************/
/* Unrolled Loop 1/1 - Begin              */
/******************************************/

// Wait LRA1 & LRB1
s_waitcnt lgkmcnt(3) // wait for 1 B / 8 A
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[0:3] // left value = acc[0+0:3+0]
    // GRInc A
    s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
    s_cselect_b32 s60, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
    s_cselect_b32 s61, s[sgprWrapUA+1], 0              // incUpper <- ?
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[4:7] // left value = acc[4+0:7+0]
    s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s60        // gra SRD += inc(lower)
    s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s61       // gra SRD += inc(upper)
    s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s60 // limit -= inc)
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[8:11] // left value = acc[8+0:11+0]
    s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s61 // limit -= inc)
    s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
    s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[12:15] // left value = acc[12+0:15+0]
    // GRA
    s_mov_b32 m0, s[sgprLocalWriteAddrA]               // m0 <- LDS write address
    buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_0_0
    s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[16:19] // left value = acc[16+0:19+0]
    ds_read_b64_tr_b16 v[vgprValuA_X1_I0+0+0:vgprValuA_X1_I0+0+0+1], v[vgprLocalReadAddrA+0] offset:8448 // LDS Transpose
    ds_read_b64_tr_b16 v[vgprValuA_X1_I0+0+2:vgprValuA_X1_I0+0+2+1], v[vgprLocalReadAddrA+0] offset:8704 // LDS Transpose
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[20:23] // left value = acc[20+0:23+0]
    s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
    buffer_load_dwordx4 v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_1_0
    ds_read_b64_tr_b16 v[vgprValuA_X1_I0+4+0:vgprValuA_X1_I0+4+0+1], v[vgprLocalReadAddrA+0] offset:8512 // LDS Transpose
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[24:27] // left value = acc[24+0:27+0]
    ds_read_b64_tr_b16 v[vgprValuA_X1_I0+4+2:vgprValuA_X1_I0+4+2+1], v[vgprLocalReadAddrA+0] offset:8768 // LDS Transpose
    ds_read_b64_tr_b16 v[vgprValuA_X1_I0+8+0:vgprValuA_X1_I0+8+0+1], v[vgprLocalReadAddrA+0] offset:8576 // LDS Transpose
    



v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[28:31] // left value = acc[28+0:31+0]
    s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
    buffer_load_dwordx4 v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_2_0
    ds_read_b64_tr_b16 v[vgprValuA_X1_I0+8+2:vgprValuA_X1_I0+8+2+1], v[vgprLocalReadAddrA+0] offset:8832 // LDS Transpose
    // LRA0
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[32:35] // left value = acc[32+0:35+0]
    ds_read_b64_tr_b16 v[vgprValuA_X1_I0+12+0:vgprValuA_X1_I0+12+0+1], v[vgprLocalReadAddrA+0] offset:8640 // LDS Transpose
    s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[36:39] // left value = acc[36+0:39+0]
    buffer_load_dwordx4 v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_3_0
    s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
    s_cselect_b32 s60, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[40:43] // left value = acc[40+0:43+0]
    ds_read_b64_tr_b16 v[vgprValuA_X1_I0+12+2:vgprValuA_X1_I0+12+2+1], v[vgprLocalReadAddrA+0] offset:8896 // LDS Transpose
    s_cselect_b32 s61, s[sgprWrapUB+1], 0              // incUpper <- ?
    s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s60        // gra SRD += inc(lower)
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[44:47] // left value = acc[44+0:47+0]
    ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
    s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s61       // gra SRD += inc(upper)
    // LBA0
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[48:51] // left value = acc[48+0:51+0]
    ds_read_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:192 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
    s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s60 // limit -= inc)
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[52:55] // left value = acc[52+0:55+0]
    ds_read_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB] offset:320 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0
    s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s61 // limit -= inc)
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[56:59] // left value = acc[56+0:59+0]
    ds_read_b128 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[vgprLocalReadAddrB] offset:448 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0
    // GRInc B
    s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[60:63] // left value = acc[60+0:63+0]
    s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32



v_add_u32 v[vgprLocalReadAddrA], s[sgprOffsetR], v[vgprLocalReadAddrARef]
v_add_u32 v[vgprLocalReadAddrB], s[sgprOffsetR], v[vgprLocalReadAddrBRef]

;s_trap 1


;s_waitcnt 
s_waitcnt lgkmcnt(3) vmcnt(8)

 

// 2nd 16 MFMAs
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[0:3] // left value = acc[0+0:3+0]
    s_mov_b32 m0, s[sgprLocalWriteAddrB]               // m0 <- LDS write address
    buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_0_0
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[4:7] // left value = acc[4+0:7+0]
  s_mov_b32 s[sgprOffsetR], s[sgprOffsetW]
  s_add_u32 s[sgprOffsetW], s[sgprOffsetW], LDSBufferSize
  s_cmp_ge_u32 s[sgprOffsetW], TotalLDSBufferSize
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[8:11] // left value = acc[8+0:11+0]
  s_cselect_b32 s[sgprOffsetW], 0, s[sgprOffsetW]
    s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
    buffer_load_dwordx4 v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_1_0
  s_add_u32 s[sgprLocalWriteAddrA], s[sgprAddrARef], s[sgprOffsetW]
  s_add_u32 s[sgprLocalWriteAddrB], s[sgprAddrBRef], s[sgprOffsetW]
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[12:15] // left value = acc[12+0:15+0]
    s_waitcnt lgkmcnt(0)
s_barrier
    ds_read_b64_tr_b16 v[vgprValuA_X0_I0+0+0:vgprValuA_X0_I0+0+0+1], v[vgprLocalReadAddrA+0] offset:0 // LDS Transpose
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[16:19] // left value = acc[16+0:19+0]
    ds_read_b64_tr_b16 v[vgprValuA_X0_I0+0+2:vgprValuA_X0_I0+0+2+1], v[vgprLocalReadAddrA+0] offset:256 // LDS Transpose
    s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
    buffer_load_dwordx4 v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_2_0
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[20:23] // left value = acc[20+0:23+0]
    ds_read_b64_tr_b16 v[vgprValuA_X0_I0+4+0:vgprValuA_X0_I0+4+0+1], v[vgprLocalReadAddrA+0] offset:64 // LDS Transpose
    
    

v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[24:27] // left value = acc[24+0:27+0]
    ds_read_b64_tr_b16 v[vgprValuA_X0_I0+4+2:vgprValuA_X0_I0+4+2+1], v[vgprLocalReadAddrA+0] offset:320 // LDS Transpose
    s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
    buffer_load_dwordx4 v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_3_0
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[28:31] // left value = acc[28+0:31+0]
    ds_read_b64_tr_b16 v[vgprValuA_X0_I0+8+0:vgprValuA_X0_I0+8+0+1], v[vgprLocalReadAddrA+0] offset:128 // LDS Transpose

v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[32:35] // left value = acc[32+0:35+0]
    ds_read_b64_tr_b16 v[vgprValuA_X0_I0+8+2:vgprValuA_X0_I0+8+2+1], v[vgprLocalReadAddrA+0] offset:384 // LDS Transpose
    // SWAP GR addresses

v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[36:39] // left value = acc[36+0:39+0]
    ds_read_b64_tr_b16 v[vgprValuA_X0_I0+12+0:vgprValuA_X0_I0+12+0+1], v[vgprLocalReadAddrA+0] offset:192 // LDS Transpose
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[40:43] // left value = acc[40+0:43+0]
    ds_read_b64_tr_b16 v[vgprValuA_X0_I0+12+2:vgprValuA_X0_I0+12+2+1], v[vgprLocalReadAddrA+0] offset:448 // LDS Transpose
    
    s_waitcnt vmcnt(8)
s_barrier
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[44:47] // left value = acc[44+0:47+0]
    ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X1_I0+12+0+0:vgprValuB_X1_I0+12+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[48:51] // left value = acc[48+0:51+0]
    ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
    

v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X1_I0+12+0+0:vgprValuB_X1_I0+12+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[52:55] // left value = acc[52+0:55+0]
    ds_read_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X1_I0+12+0+0:vgprValuB_X1_I0+12+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[56:59] // left value = acc[56+0:59+0]
    ds_read_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB] offset:384 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X1_I0+12+0+0:vgprValuB_X1_I0+12+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[60:63] // left value = acc[60+0:63+0]
 



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
/* Ord. NoGlobalLoadLoop - Begin          */
/******************************************/

/* iter 0 (reset local read pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:14, lwEndMfmaIndex:19  */
/*  numMfmaForLR:10, syncPlrMfmaIndex:21  */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=3 newLW=0 newLR=3 for iteration == 0
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:1  */
ds_read_b64_tr_b16 v[vgprValuA_X1_I0+0+0:vgprValuA_X1_I0+0+0+1], v[vgprLocalReadAddrA+0] offset:8448 // LDS Transpose

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s60, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s61, s[sgprWrapUA+1], 0              // incUpper <- ?
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[4:7] // left value = acc[4+0:7+0]
/*  mfmaIndex:2  */
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s60        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s61       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s60 // limit -= inc)
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[8:11] // left value = acc[8+0:11+0]
/*  mfmaIndex:3  */
ds_read_b64_tr_b16 v[vgprValuA_X1_I0+0+2:vgprValuA_X1_I0+0+2+1], v[vgprLocalReadAddrA+0] offset:8704 // LDS Transpose
ds_read_b64_tr_b16 v[vgprValuA_X1_I0+4+0:vgprValuA_X1_I0+4+0+1], v[vgprLocalReadAddrA+0] offset:8512 // LDS Transpose
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s61 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[12:15] // left value = acc[12+0:15+0]
/*  mfmaIndex:4  */
ds_read_b64_tr_b16 v[vgprValuA_X1_I0+4+2:vgprValuA_X1_I0+4+2+1], v[vgprLocalReadAddrA+0] offset:8768 // LDS Transpose
ds_read_b64_tr_b16 v[vgprValuA_X1_I0+8+0:vgprValuA_X1_I0+8+0+1], v[vgprLocalReadAddrA+0] offset:8576 // LDS Transpose

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s60, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s61, s[sgprWrapUB+1], 0              // incUpper <- ?
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[16:19] // left value = acc[16+0:19+0]
/*  mfmaIndex:5  */
ds_read_b64_tr_b16 v[vgprValuA_X1_I0+8+2:vgprValuA_X1_I0+8+2+1], v[vgprLocalReadAddrA+0] offset:8832 // LDS Transpose
ds_read_b64_tr_b16 v[vgprValuA_X1_I0+12+0:vgprValuA_X1_I0+12+0+1], v[vgprLocalReadAddrA+0] offset:8640 // LDS Transpose
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s60        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s61       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s60 // limit -= inc)
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[20:23] // left value = acc[20+0:23+0]
/*  mfmaIndex:6  */
ds_read_b64_tr_b16 v[vgprValuA_X1_I0+12+2:vgprValuA_X1_I0+12+2+1], v[vgprLocalReadAddrA+0] offset:8896 // LDS Transpose
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s61 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[24:27] // left value = acc[24+0:27+0]
/*  mfmaIndex:7  */
ds_read_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:192 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[28:31] // left value = acc[28+0:31+0]
/*  mfmaIndex:8  */
ds_read_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB] offset:320 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[32:35] // left value = acc[32+0:35+0]
/*  mfmaIndex:9  */
ds_read_b128 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[vgprLocalReadAddrB] offset:448 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0
/* localReadsVacancy: latencyLeft 1 */
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[36:39] // left value = acc[36+0:39+0]
/*  mfmaIndex:10  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[40:43] // left value = acc[40+0:43+0]
/*  mfmaIndex:11  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[44:47] // left value = acc[44+0:47+0]
/*  mfmaIndex:12  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[48:51] // left value = acc[48+0:51+0]
/*  mfmaIndex:13  */
/* schedule remaining localreads for one buffer scheduling */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[52:55] // left value = acc[52+0:55+0]
/*  mfmaIndex:14  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[56:59] // left value = acc[56+0:59+0]
/*  mfmaIndex:15  */
/* localReadsVacancy: latencyLeft 5 */

/* local read swap offsets a */

v_add_u32 v[vgprLocalReadAddrA], LDSBufferSize, v[vgprLocalReadAddrA]
v_subrev_u32 v[vgprTmp0], TotalLDSBufferSize, v[vgprLocalReadAddrA]
v_cmp_le_u32_e32 vcc, TotalLDSBufferSize, v[vgprLocalReadAddrA]
s_nop 1
v_cndmask_b32_e32 v[vgprLocalReadAddrA], v[vgprLocalReadAddrA], v[vgprTmp0], vcc

v_add_u32 v[vgprLocalReadAddrB], LDSBufferSize, v[vgprLocalReadAddrB]
v_subrev_u32 v[vgprTmp0], TotalLDSBufferSize, v[vgprLocalReadAddrB]
v_cmp_le_u32_e32 vcc, TotalLDSBufferSize, v[vgprLocalReadAddrB]
s_nop 1
v_cndmask_b32_e32 v[vgprLocalReadAddrB], v[vgprLocalReadAddrB], v[vgprTmp0], vcc


/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[60:63] // left value = acc[60+0:63+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=8 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=4 */

/* iter 1 (swap and reset local write pointers iteration)  */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:14, lwEndMfmaIndex:19  */
/*  numMfmaForLR:10, syncPlrMfmaIndex:21  */
/*  mfmaIndex:16  */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:17  */
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[4:7] // left value = acc[4+0:7+0]
/*  mfmaIndex:18  */
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[8:11] // left value = acc[8+0:11+0]
/*  mfmaIndex:19  */

/* local write swap offsets a */
s_add_u32 s[sgprOffsetW], s[sgprOffsetW], LDSBufferSize
s_cmp_ge_u32 s[sgprOffsetW], 3*LDSBufferSize
s_cselect_b32 s[sgprOffsetW], 0, s[sgprOffsetW]

s_add_u32 s[sgprLocalWriteAddrA], s[sgprAddrARef], s[sgprOffsetW]
s_add_u32 s[sgprLocalWriteAddrB], s[sgprAddrBRef], s[sgprOffsetW]

v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[12:15] // left value = acc[12+0:15+0]
/*  mfmaIndex:20  */
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[16:19] // left value = acc[16+0:19+0]
/*  mfmaIndex:21  */
s_waitcnt vmcnt(0)                                 // wait for global reads with lds
// Skip force waitcnt0
s_barrier
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[20:23] // left value = acc[20+0:23+0]
/*  mfmaIndex:22  */
ds_read_b64_tr_b16 v[vgprValuA_X0_I0+0+0:vgprValuA_X0_I0+0+0+1], v[vgprLocalReadAddrA+0] offset:0 // LDS Transpose
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[24:27] // left value = acc[24+0:27+0]
/*  mfmaIndex:23  */
ds_read_b64_tr_b16 v[vgprValuA_X0_I0+0+2:vgprValuA_X0_I0+0+2+1], v[vgprLocalReadAddrA+0] offset:256 // LDS Transpose
ds_read_b64_tr_b16 v[vgprValuA_X0_I0+4+0:vgprValuA_X0_I0+4+0+1], v[vgprLocalReadAddrA+0] offset:64 // LDS Transpose
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[28:31] // left value = acc[28+0:31+0]
/*  mfmaIndex:24  */
ds_read_b64_tr_b16 v[vgprValuA_X0_I0+4+2:vgprValuA_X0_I0+4+2+1], v[vgprLocalReadAddrA+0] offset:320 // LDS Transpose
ds_read_b64_tr_b16 v[vgprValuA_X0_I0+8+0:vgprValuA_X0_I0+8+0+1], v[vgprLocalReadAddrA+0] offset:128 // LDS Transpose
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[32:35] // left value = acc[32+0:35+0]
/*  mfmaIndex:25  */
ds_read_b64_tr_b16 v[vgprValuA_X0_I0+8+2:vgprValuA_X0_I0+8+2+1], v[vgprLocalReadAddrA+0] offset:384 // LDS Transpose
ds_read_b64_tr_b16 v[vgprValuA_X0_I0+12+0:vgprValuA_X0_I0+12+0+1], v[vgprLocalReadAddrA+0] offset:192 // LDS Transpose
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[36:39] // left value = acc[36+0:39+0]
/*  mfmaIndex:26  */
ds_read_b64_tr_b16 v[vgprValuA_X0_I0+12+2:vgprValuA_X0_I0+12+2+1], v[vgprLocalReadAddrA+0] offset:448 // LDS Transpose
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[40:43] // left value = acc[40+0:43+0]
/*  mfmaIndex:27  */
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[44:47] // left value = acc[44+0:47+0]
/*  mfmaIndex:28  */
ds_read_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X1_I0+12+0+0:vgprValuB_X1_I0+12+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[48:51] // left value = acc[48+0:51+0]
/*  mfmaIndex:29  */
ds_read_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB] offset:384 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X1_I0+12+0+0:vgprValuB_X1_I0+12+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[52:55] // left value = acc[52+0:55+0]
/*  mfmaIndex:30  */
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X1_I0+12+0+0:vgprValuB_X1_I0+12+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[56:59] // left value = acc[56+0:59+0]
/*  mfmaIndex:31  */
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X1_I0+12+0+0:vgprValuB_X1_I0+12+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[60:63] // left value = acc[60+0:63+0]
/* numPrefetchIter=1 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=8 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=4 */
label_toPGR1:

/******************************************/
/* Ord. NoLoadLoop - Begin                */
/******************************************/

/* iter 0 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:15, lwEndMfmaIndex:15  */
/*  numMfmaForLR:10, syncPlrMfmaIndex:21  */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=3 newLW=0 newLR=3 for iteration == 0
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:1  */
ds_read_b64_tr_b16 v[vgprValuA_X1_I0+0+0:vgprValuA_X1_I0+0+0+1], v[vgprLocalReadAddrA+0] offset:8448 // LDS Transpose
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[4:7] // left value = acc[4+0:7+0]
/*  mfmaIndex:2  */
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[8:11] // left value = acc[8+0:11+0]
/*  mfmaIndex:3  */
ds_read_b64_tr_b16 v[vgprValuA_X1_I0+0+2:vgprValuA_X1_I0+0+2+1], v[vgprLocalReadAddrA+0] offset:8704 // LDS Transpose
ds_read_b64_tr_b16 v[vgprValuA_X1_I0+4+0:vgprValuA_X1_I0+4+0+1], v[vgprLocalReadAddrA+0] offset:8512 // LDS Transpose
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[12:15] // left value = acc[12+0:15+0]
/*  mfmaIndex:4  */
ds_read_b64_tr_b16 v[vgprValuA_X1_I0+4+2:vgprValuA_X1_I0+4+2+1], v[vgprLocalReadAddrA+0] offset:8768 // LDS Transpose
ds_read_b64_tr_b16 v[vgprValuA_X1_I0+8+0:vgprValuA_X1_I0+8+0+1], v[vgprLocalReadAddrA+0] offset:8576 // LDS Transpose
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[16:19] // left value = acc[16+0:19+0]
/*  mfmaIndex:5  */
ds_read_b64_tr_b16 v[vgprValuA_X1_I0+8+2:vgprValuA_X1_I0+8+2+1], v[vgprLocalReadAddrA+0] offset:8832 // LDS Transpose
ds_read_b64_tr_b16 v[vgprValuA_X1_I0+12+0:vgprValuA_X1_I0+12+0+1], v[vgprLocalReadAddrA+0] offset:8640 // LDS Transpose
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[20:23] // left value = acc[20+0:23+0]
/*  mfmaIndex:6  */
ds_read_b64_tr_b16 v[vgprValuA_X1_I0+12+2:vgprValuA_X1_I0+12+2+1], v[vgprLocalReadAddrA+0] offset:8896 // LDS Transpose
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[24:27] // left value = acc[24+0:27+0]
/*  mfmaIndex:7  */
ds_read_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:192 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[28:31] // left value = acc[28+0:31+0]
/*  mfmaIndex:8  */
ds_read_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB] offset:320 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[32:35] // left value = acc[32+0:35+0]
/*  mfmaIndex:9  */
ds_read_b128 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[vgprLocalReadAddrB] offset:448 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0
/* localReadsVacancy: latencyLeft 1 */
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[36:39] // left value = acc[36+0:39+0]
/*  mfmaIndex:10  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[40:43] // left value = acc[40+0:43+0]
/*  mfmaIndex:11  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[44:47] // left value = acc[44+0:47+0]
/*  mfmaIndex:12  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[48:51] // left value = acc[48+0:51+0]
/*  mfmaIndex:13  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[52:55] // left value = acc[52+0:55+0]
/*  mfmaIndex:14  */
/* schedule remaining localreads for one buffer scheduling */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[56:59] // left value = acc[56+0:59+0]
/*  mfmaIndex:15  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[60:63] // left value = acc[60+0:63+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=8 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=4 */

/* iter 1 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:15, lwEndMfmaIndex:15  */
/*  numMfmaForLR:10, syncPlrMfmaIndex:21  */
/*  mfmaIndex:16  */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:17  */
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[4:7] // left value = acc[4+0:7+0]
/*  mfmaIndex:18  */
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[8:11] // left value = acc[8+0:11+0]
/*  mfmaIndex:19  */
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[12:15] // left value = acc[12+0:15+0]
/*  mfmaIndex:20  */
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[16:19] // left value = acc[16+0:19+0]
/*  mfmaIndex:21  */
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[20:23] // left value = acc[20+0:23+0]
/*  mfmaIndex:22  */
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[24:27] // left value = acc[24+0:27+0]
/*  mfmaIndex:23  */
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[28:31] // left value = acc[28+0:31+0]
/*  mfmaIndex:24  */
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[32:35] // left value = acc[32+0:35+0]
/*  mfmaIndex:25  */
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[36:39] // left value = acc[36+0:39+0]
/*  mfmaIndex:26  */
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[40:43] // left value = acc[40+0:43+0]
/*  mfmaIndex:27  */
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[44:47] // left value = acc[44+0:47+0]
/*  mfmaIndex:28  */
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X1_I0+12+0+0:vgprValuB_X1_I0+12+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[48:51] // left value = acc[48+0:51+0]
/*  mfmaIndex:29  */
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X1_I0+12+0+0:vgprValuB_X1_I0+12+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[52:55] // left value = acc[52+0:55+0]
/*  mfmaIndex:30  */
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X1_I0+12+0+0:vgprValuB_X1_I0+12+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[56:59] // left value = acc[56+0:59+0]
/*  mfmaIndex:31  */
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X1_I0+12+0+0:vgprValuB_X1_I0+12+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[60:63] // left value = acc[60+0:63+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=8 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=4 */
label_toPGR1end_OrdNLL:
label_PrefetchGlobalLastIterEnd:

/* Tail: add ValuA/B vgpr buffer [10...74) to pool */

/* Tail: add address/G2L vgpr [74...74) to pool */

/******************************************/
/* Tail Loop                              */
/******************************************/


s_mov_b32 s[sgprLocalWriteAddrA], s[sgprAddrARef]
s_mov_b32 s[sgprLocalWriteAddrB], s[sgprAddrBRef]


/* Check out VGPR (numG2LA,numG2LB,numG2LMetadata) = (16,16,0) */
.set vgprG2LA_BASE, 10
.set vgprG2LA, vgprG2LA_BASE+0
.set vgprG2LB_BASE, 26
.set vgprG2LB, vgprG2LB_BASE+0
/* Check out VGPR (numLWA,numLWB) = (1,1) */
.set vgprLocalWriteAddrA, 42
.set vgprLocalWriteAddrB, 43

// numIterL = LOCAL_SPLITU * min(sizeL % LOCAL_DEPTHU, DEPTHU / LOCAL_SPLITU)
s_and_b32 s[sgprLoopCounterL], 63, s[sgprSizesSum+0] // s[sgprLoopCounterL] = s[sgprSizesSum+0] % 64
s_cmp_lt_u32 s[sgprStreamKLocalEnd], s[sgprItersPerTile] // Check if WG processes final iteration of tile
s_cmov_b32 s[sgprLoopCounterL], 0                  // This WG not completing tile
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // numIterL == 0
s_mov_b32 s[sgprOrigLoopCounter], 0                // repurpose to count each localRead increment
s_cbranch_scc1 label_SkipTailLoopL                 // skip to end of tail loop b/c numIter==0

/* remove stagger offsets for tail loop */
s_sub_i32 s84, 3, s[sgprStaggerUIter]
s_cmp_ge_i32 s84, 0
s_cbranch_scc0 label_Negative_J5DQFVGFWLXU2DUR
s_mul_hi_u32 s85, s84, s[sgprGlobalReadIncsA+0]    // start offset S in bytes
s_mul_i32 s84, s84, s[sgprGlobalReadIncsA+0]       // start offset S in bytes
s_branch label_MultiplyDone_DLSAQLEVYLOBCPNL
label_Negative_J5DQFVGFWLXU2DUR:
s_abs_i32 s84, s84
s_mul_hi_u32 s85, s84, s[sgprGlobalReadIncsA+0]    // start offset S in bytes
s_mul_i32 s84, s84, s[sgprGlobalReadIncsA+0]       // start offset S in bytes
s_xor_b32 s84, s84, 0xffffffff
s_xor_b32 s85, s85, 0xffffffff
s_add_u32 s84, s84, 0x1
s_addc_u32 s85, s85, 0
label_MultiplyDone_DLSAQLEVYLOBCPNL:
s_sub_u32 s84, s84, s[sgprWrapUA]                  // S - WrapU
s_subb_u32 s85, s85, s[sgprWrapUA+1]               // S - WrapU
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s84        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s85       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s84 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s85 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_sub_i32 s84, 3, s[sgprStaggerUIter]
s_cmp_ge_i32 s84, 0
s_cbranch_scc0 label_Negative_LQI6BOBE0EY8XIP1
s_mul_hi_u32 s85, s84, s[sgprGlobalReadIncsB+0]    // start offset S in bytes
s_mul_i32 s84, s84, s[sgprGlobalReadIncsB+0]       // start offset S in bytes
s_branch label_MultiplyDone_9N1QELR2XL4Z0HRB
label_Negative_LQI6BOBE0EY8XIP1:
s_abs_i32 s84, s84
s_mul_hi_u32 s85, s84, s[sgprGlobalReadIncsB+0]    // start offset S in bytes
s_mul_i32 s84, s84, s[sgprGlobalReadIncsB+0]       // start offset S in bytes
s_xor_b32 s84, s84, 0xffffffff
s_xor_b32 s85, s85, 0xffffffff
s_add_u32 s84, s84, 0x1
s_addc_u32 s85, s85, 0
label_MultiplyDone_9N1QELR2XL4Z0HRB:
s_sub_u32 s84, s84, s[sgprWrapUB]                  // S - WrapU
s_subb_u32 s85, s85, s[sgprWrapUB+1]               // S - WrapU
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s84        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s85       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s84 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s85 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32

/* Update M0 for DTLDS */
s_mov_b32 m0, s[sgprLocalWriteAddrA]               // m0 <- LDS write address
/* before DirectToLds load, ensure prior ds_reads have finished */
s_waitcnt lgkmcnt(0)
s_barrier

/* Tail global read A */
/* g2l=0, load component 0 */
buffer_load_short_d16 v[vgprG2LA+0+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
buffer_load_short_d16_hi v44, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:2 // load one buffer value
/* g2l=0, load component 2 */
buffer_load_short_d16 v[vgprG2LA+0+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:4 // load one buffer value
/* g2l=0, load component 3 */
buffer_load_short_d16_hi v45, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:6 // load one buffer value
/* g2l=0, load component 4 */
buffer_load_short_d16 v[vgprG2LA+0+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:8 // load one buffer value
/* g2l=0, load component 5 */
buffer_load_short_d16_hi v46, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:10 // load one buffer value
/* g2l=0, load component 6 */
buffer_load_short_d16 v[vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:12 // load one buffer value
/* g2l=0, load component 7 */
buffer_load_short_d16_hi v47, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:14 // load one buffer value
/* g2l=4, load component 0 */
buffer_load_short_d16 v[vgprG2LA+4+0], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // load one buffer value
/* g2l=4, load component 1 */
buffer_load_short_d16_hi v48, v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:2 // load one buffer value
/* g2l=4, load component 2 */
buffer_load_short_d16 v[vgprG2LA+4+1], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:4 // load one buffer value
/* g2l=4, load component 3 */
buffer_load_short_d16_hi v49, v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:6 // load one buffer value
/* g2l=4, load component 4 */
buffer_load_short_d16 v[vgprG2LA+4+2], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:8 // load one buffer value
/* g2l=4, load component 5 */
buffer_load_short_d16_hi v50, v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:10 // load one buffer value
/* g2l=4, load component 6 */
buffer_load_short_d16 v[vgprG2LA+4+3], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:12 // load one buffer value
/* g2l=4, load component 7 */
buffer_load_short_d16_hi v51, v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:14 // load one buffer value
/* g2l=8, load component 0 */
buffer_load_short_d16 v[vgprG2LA+8+0], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // load one buffer value
/* g2l=8, load component 1 */
buffer_load_short_d16_hi v52, v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:2 // load one buffer value
/* g2l=8, load component 2 */
buffer_load_short_d16 v[vgprG2LA+8+1], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:4 // load one buffer value
/* g2l=8, load component 3 */
buffer_load_short_d16_hi v53, v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:6 // load one buffer value
/* g2l=8, load component 4 */
buffer_load_short_d16 v[vgprG2LA+8+2], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:8 // load one buffer value
/* g2l=8, load component 5 */
buffer_load_short_d16_hi v54, v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:10 // load one buffer value
/* g2l=8, load component 6 */
buffer_load_short_d16 v[vgprG2LA+8+3], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:12 // load one buffer value
/* g2l=8, load component 7 */
buffer_load_short_d16_hi v55, v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:14 // load one buffer value
/* g2l=12, load component 0 */
buffer_load_short_d16 v[vgprG2LA+12+0], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // load one buffer value
/* g2l=12, load component 1 */
buffer_load_short_d16_hi v56, v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:2 // load one buffer value
/* g2l=12, load component 2 */
buffer_load_short_d16 v[vgprG2LA+12+1], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:4 // load one buffer value
/* g2l=12, load component 3 */
buffer_load_short_d16_hi v57, v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:6 // load one buffer value
/* g2l=12, load component 4 */
buffer_load_short_d16 v[vgprG2LA+12+2], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:8 // load one buffer value
/* g2l=12, load component 5 */
buffer_load_short_d16_hi v58, v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:10 // load one buffer value
/* g2l=12, load component 6 */
buffer_load_short_d16 v[vgprG2LA+12+3], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:12 // load one buffer value
/* g2l=12, load component 7 */
buffer_load_short_d16_hi v59, v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:14 // load one buffer value
s_waitcnt vmcnt(0)                                 // Wait for previous GR to finish
v_or_b32 v[vgprG2LA+0+0], v[vgprG2LA+0+0], v44     // HasEccHalf: pack
v_or_b32 v[vgprG2LA+0+1], v[vgprG2LA+0+1], v45     // HasEccHalf: pack
v_or_b32 v[vgprG2LA+0+2], v[vgprG2LA+0+2], v46     // HasEccHalf: pack
v_or_b32 v[vgprG2LA+0+3], v[vgprG2LA+0+3], v47     // HasEccHalf: pack
v_or_b32 v[vgprG2LA+4+0], v[vgprG2LA+4+0], v48     // HasEccHalf: pack
v_or_b32 v[vgprG2LA+4+1], v[vgprG2LA+4+1], v49     // HasEccHalf: pack
v_or_b32 v[vgprG2LA+4+2], v[vgprG2LA+4+2], v50     // HasEccHalf: pack
v_or_b32 v[vgprG2LA+4+3], v[vgprG2LA+4+3], v51     // HasEccHalf: pack
v_or_b32 v[vgprG2LA+8+0], v[vgprG2LA+8+0], v52     // HasEccHalf: pack
v_or_b32 v[vgprG2LA+8+1], v[vgprG2LA+8+1], v53     // HasEccHalf: pack
v_or_b32 v[vgprG2LA+8+2], v[vgprG2LA+8+2], v54     // HasEccHalf: pack
v_or_b32 v[vgprG2LA+8+3], v[vgprG2LA+8+3], v55     // HasEccHalf: pack
v_or_b32 v[vgprG2LA+12+0], v[vgprG2LA+12+0], v56   // HasEccHalf: pack
v_or_b32 v[vgprG2LA+12+1], v[vgprG2LA+12+1], v57   // HasEccHalf: pack
v_or_b32 v[vgprG2LA+12+2], v[vgprG2LA+12+2], v58   // HasEccHalf: pack
v_or_b32 v[vgprG2LA+12+3], v[vgprG2LA+12+3], v59   // HasEccHalf: pack

/* Update M0 for DTLDS */
s_mov_b32 m0, s[sgprLocalWriteAddrB]               // m0 <- LDS write address

/* Tail global read B */
/* g2l=0, load component 0 */
buffer_load_short_d16 v[vgprG2LB+0+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
buffer_load_short_d16_hi v44, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:2 // load one buffer value
/* g2l=0, load component 2 */
buffer_load_short_d16 v[vgprG2LB+0+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:4 // load one buffer value
/* g2l=0, load component 3 */
buffer_load_short_d16_hi v45, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:6 // load one buffer value
/* g2l=0, load component 4 */
buffer_load_short_d16 v[vgprG2LB+0+2], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:8 // load one buffer value
/* g2l=0, load component 5 */
buffer_load_short_d16_hi v46, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:10 // load one buffer value
/* g2l=0, load component 6 */
buffer_load_short_d16 v[vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:12 // load one buffer value
/* g2l=0, load component 7 */
buffer_load_short_d16_hi v47, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:14 // load one buffer value
/* g2l=4, load component 0 */
buffer_load_short_d16 v[vgprG2LB+4+0], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // load one buffer value
/* g2l=4, load component 1 */
buffer_load_short_d16_hi v48, v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:2 // load one buffer value
/* g2l=4, load component 2 */
buffer_load_short_d16 v[vgprG2LB+4+1], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:4 // load one buffer value
/* g2l=4, load component 3 */
buffer_load_short_d16_hi v49, v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:6 // load one buffer value
/* g2l=4, load component 4 */
buffer_load_short_d16 v[vgprG2LB+4+2], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:8 // load one buffer value
/* g2l=4, load component 5 */
buffer_load_short_d16_hi v50, v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:10 // load one buffer value
/* g2l=4, load component 6 */
buffer_load_short_d16 v[vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:12 // load one buffer value
/* g2l=4, load component 7 */
buffer_load_short_d16_hi v51, v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:14 // load one buffer value
/* g2l=8, load component 0 */
buffer_load_short_d16 v[vgprG2LB+8+0], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // load one buffer value
/* g2l=8, load component 1 */
buffer_load_short_d16_hi v52, v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:2 // load one buffer value
/* g2l=8, load component 2 */
buffer_load_short_d16 v[vgprG2LB+8+1], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:4 // load one buffer value
/* g2l=8, load component 3 */
buffer_load_short_d16_hi v53, v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:6 // load one buffer value
/* g2l=8, load component 4 */
buffer_load_short_d16 v[vgprG2LB+8+2], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:8 // load one buffer value
/* g2l=8, load component 5 */
buffer_load_short_d16_hi v54, v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:10 // load one buffer value
/* g2l=8, load component 6 */
buffer_load_short_d16 v[vgprG2LB+8+3], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:12 // load one buffer value
/* g2l=8, load component 7 */
buffer_load_short_d16_hi v55, v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:14 // load one buffer value
/* g2l=12, load component 0 */
buffer_load_short_d16 v[vgprG2LB+12+0], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // load one buffer value
/* g2l=12, load component 1 */
buffer_load_short_d16_hi v56, v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:2 // load one buffer value
/* g2l=12, load component 2 */
buffer_load_short_d16 v[vgprG2LB+12+1], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:4 // load one buffer value
/* g2l=12, load component 3 */
buffer_load_short_d16_hi v57, v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:6 // load one buffer value
/* g2l=12, load component 4 */
buffer_load_short_d16 v[vgprG2LB+12+2], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:8 // load one buffer value
/* g2l=12, load component 5 */
buffer_load_short_d16_hi v58, v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:10 // load one buffer value
/* g2l=12, load component 6 */
buffer_load_short_d16 v[vgprG2LB+12+3], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:12 // load one buffer value
/* g2l=12, load component 7 */
buffer_load_short_d16_hi v59, v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:14 // load one buffer value
s_waitcnt vmcnt(0)                                 // Wait for previous GR to finish
v_or_b32 v[vgprG2LB+0+0], v[vgprG2LB+0+0], v44     // HasEccHalf: pack
v_or_b32 v[vgprG2LB+0+1], v[vgprG2LB+0+1], v45     // HasEccHalf: pack
v_or_b32 v[vgprG2LB+0+2], v[vgprG2LB+0+2], v46     // HasEccHalf: pack
v_or_b32 v[vgprG2LB+0+3], v[vgprG2LB+0+3], v47     // HasEccHalf: pack
v_or_b32 v[vgprG2LB+4+0], v[vgprG2LB+4+0], v48     // HasEccHalf: pack
v_or_b32 v[vgprG2LB+4+1], v[vgprG2LB+4+1], v49     // HasEccHalf: pack
v_or_b32 v[vgprG2LB+4+2], v[vgprG2LB+4+2], v50     // HasEccHalf: pack
v_or_b32 v[vgprG2LB+4+3], v[vgprG2LB+4+3], v51     // HasEccHalf: pack
v_or_b32 v[vgprG2LB+8+0], v[vgprG2LB+8+0], v52     // HasEccHalf: pack
v_or_b32 v[vgprG2LB+8+1], v[vgprG2LB+8+1], v53     // HasEccHalf: pack
v_or_b32 v[vgprG2LB+8+2], v[vgprG2LB+8+2], v54     // HasEccHalf: pack
v_or_b32 v[vgprG2LB+8+3], v[vgprG2LB+8+3], v55     // HasEccHalf: pack
v_or_b32 v[vgprG2LB+12+0], v[vgprG2LB+12+0], v56   // HasEccHalf: pack
v_or_b32 v[vgprG2LB+12+1], v[vgprG2LB+12+1], v57   // HasEccHalf: pack
v_or_b32 v[vgprG2LB+12+2], v[vgprG2LB+12+2], v58   // HasEccHalf: pack
v_or_b32 v[vgprG2LB+12+3], v[vgprG2LB+12+3], v59   // HasEccHalf: pack
s_waitcnt vmcnt(0)                                 // 2wait for global read
// Skip force waitcnt0
s_barrier
/* Set local write offsets for A to be same as DTL 16B load */
v_and_b32 v[vgprLocalWriteAddrA], 63, v[vgprSerial] // Serial % wavesize
v_lshlrev_b32 v[vgprLocalWriteAddrA], 0x4, v[vgprLocalWriteAddrA]
v_add_u32 v[vgprLocalWriteAddrA], s[sgprLocalWriteAddrA], v[vgprLocalWriteAddrA]
/* Set local write offsets for B to be same as DTL 16B load */
v_and_b32 v[vgprLocalWriteAddrB], 63, v[vgprSerial] // Serial % wavesize
v_lshlrev_b32 v[vgprLocalWriteAddrB], 0x4, v[vgprLocalWriteAddrB]
v_add_u32 v[vgprLocalWriteAddrB], s[sgprLocalWriteAddrB], v[vgprLocalWriteAddrB]

/* local write a */
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+4:vgprG2LA+4+3] offset:4224 // lwoA_0_0_1_0 = (0*LSCA) + (1*LSPA)(*MT0I+PAD) = 4224
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+8:vgprG2LA+8+3] offset:8448 // lwoA_0_0_2_0 = (0*LSCA) + (2*LSPA)(*MT0I+PAD) = 8448
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+12:vgprG2LA+12+3] offset:12672 // lwoA_0_0_3_0 = (0*LSCA) + (3*LSPA)(*MT0I+PAD) = 12672

/* local write b */
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+4:vgprG2LB+4+3] offset:4224 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 4224
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+8:vgprG2LB+8+3] offset:8448 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 8448
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+12:vgprG2LB+12+3] offset:12672 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 12672

/* Recalc local read offsets */
s_waitcnt lgkmcnt(0)                               // 5wait for local write
// Skip force waitcnt0
s_barrier
.set vgprG2LA_BASE, UNDEF
.set vgprG2LA, UNDEF
.set vgprG2LB_BASE, UNDEF
.set vgprG2LB, UNDEF
.set vgprLocalWriteAddrA, UNDEF
.set vgprLocalWriteAddrB, UNDEF
.set vgprValuA_X0_I0_BASE, 10
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+16
.set vgprValuB_X0_I0_BASE, 42
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X1_I0, vgprValuB_X0_I0_BASE+16

/* Tail: local read reset offsets a */

/* localReadResetOffsets */
/* handled internally */

/* Tail: local read reset offsets b */

/* localReadResetOffsets */
/* handled internally */
v_mov_b32 v[vgprLocalReadAddrA], v[vgprLocalReadAddrARef]
v_mov_b32 v[vgprLocalReadAddrB], v[vgprLocalReadAddrBRef]

/* Tail: local read init pointers a */

/* localReadInitPointers */

/* Tail: local read init pointers b */

/* localReadInitPointers */

/* tail loop: macs */
.align 16
label_TailLoopBeginL:

/* tail loop unroll iter 0 */

/* local read a */
ds_read_b64_tr_b16 v[vgprValuA_X0_I0+0+0:vgprValuA_X0_I0+0+0+1], v[vgprLocalReadAddrA+0] offset:0 // LDS Transpose
ds_read_b64_tr_b16 v[vgprValuA_X0_I0+0+2:vgprValuA_X0_I0+0+2+1], v[vgprLocalReadAddrA+0] offset:256 // LDS Transpose
ds_read_b64_tr_b16 v[vgprValuA_X0_I0+4+0:vgprValuA_X0_I0+4+0+1], v[vgprLocalReadAddrA+0] offset:64 // LDS Transpose
ds_read_b64_tr_b16 v[vgprValuA_X0_I0+4+2:vgprValuA_X0_I0+4+2+1], v[vgprLocalReadAddrA+0] offset:320 // LDS Transpose
ds_read_b64_tr_b16 v[vgprValuA_X0_I0+8+0:vgprValuA_X0_I0+8+0+1], v[vgprLocalReadAddrA+0] offset:128 // LDS Transpose
ds_read_b64_tr_b16 v[vgprValuA_X0_I0+8+2:vgprValuA_X0_I0+8+2+1], v[vgprLocalReadAddrA+0] offset:384 // LDS Transpose
ds_read_b64_tr_b16 v[vgprValuA_X0_I0+12+0:vgprValuA_X0_I0+12+0+1], v[vgprLocalReadAddrA+0] offset:192 // LDS Transpose
ds_read_b64_tr_b16 v[vgprValuA_X0_I0+12+2:vgprValuA_X0_I0+12+2+1], v[vgprLocalReadAddrA+0] offset:448 // LDS Transpose

/* local read b */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB] offset:384 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read inc a */
/* Adding additional 256 pad since cumulative inc has reached 1024 */
s_mov_b32 s83, 8448                                // inc
v_add_co_u32 v[vgprLocalReadAddrA+0], vcc, s83, v[vgprLocalReadAddrA+0] // lrA += 8192 ((MT+PAD)*bpeDS)

/* local read inc b */
s_mov_b32 s83, 64                                  // inc
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, s83, v[vgprLocalReadAddrB+0] // lrB += 64 (bpeDS)
s_waitcnt lgkmcnt(0)                               // 4wait for local read
v_and_b32 v75, 63, v[vgprSerial]                   // v75 = v[vgprSerial] % 64
v_lshrrev_b32 v75, 4, v75                          // 75 = 75 / 16
v_lshlrev_b32 v75, 3, v75                          // v75 = v75 * 8
v_add_u32 v76, v75, 0
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+0], v[vgprValuA_X0_I0+0+0+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+4+0+0+0], v[vgprValuA_X0_I0+4+0+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+8+0+0+0], v[vgprValuA_X0_I0+8+0+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+12+0+0+0], v[vgprValuA_X0_I0+12+0+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0+1], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+4+0+0+1], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+8+0+0+1], v[vgprValuA_X0_I0+8+0+0+1], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+12+0+0+1], v[vgprValuA_X0_I0+12+0+0+1], 0, s[84:85] // set 0 if K_idx >= sizeL
v_add_u32 v76, v76, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+2], v[vgprValuA_X0_I0+0+0+0+2], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+4+0+0+2], v[vgprValuA_X0_I0+4+0+0+2], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+8+0+0+2], v[vgprValuA_X0_I0+8+0+0+2], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+12+0+0+2], v[vgprValuA_X0_I0+12+0+0+2], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+0+0+0+3], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+4+0+0+3], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+8+0+0+3], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+12+0+0+3], v[vgprValuA_X0_I0+12+0+0+3], 0, s[84:85] // set 0 if K_idx >= sizeL
v_and_b32 v75, 63, v[vgprSerial]                   // v75 = v[vgprSerial] % 64
v_lshrrev_b32 v75, 4, v75                          // 75 = 75 / 16
v_lshlrev_b32 v75, 3, v75                          // v75 = v75 * 8
v_add_u32 v76, v75, 0
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+0], v[vgprValuB_X0_I0+0+0+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+4+0+0+0], v[vgprValuB_X0_I0+4+0+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0+0], v[vgprValuB_X0_I0+8+0+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+12+0+0+0], v[vgprValuB_X0_I0+12+0+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0+1], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0+1], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0+1], v[vgprValuB_X0_I0+8+0+0+1], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+12+0+0+1], v[vgprValuB_X0_I0+12+0+0+1], 0, s[84:85] // set 0 if K_idx >= sizeL
v_add_u32 v76, v76, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+2], v[vgprValuB_X0_I0+0+0+0+2], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+4+0+0+2], v[vgprValuB_X0_I0+4+0+0+2], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0+2], v[vgprValuB_X0_I0+8+0+0+2], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+12+0+0+2], v[vgprValuB_X0_I0+12+0+0+2], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+3], v[vgprValuB_X0_I0+0+0+0+3], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+4+0+0+3], v[vgprValuB_X0_I0+4+0+0+3], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0+3], v[vgprValuB_X0_I0+8+0+0+3], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+12+0+0+3], v[vgprValuB_X0_I0+12+0+0+3], 0, s[84:85] // set 0 if K_idx >= sizeL
s_and_b32 s83, s[sgprSizeL], 7                     // if summation is multiple of 8, skip masking
s_cmp_eq_u32 s83, 0
s_cbranch_scc1 label_TailLoop_SkipZeroOutMask_DZOUDPYJU2HHRCOQ // skip mask
s_and_b32 s83, s[sgprLoopCounterL], 7              // get inputs for edge thread
s_sub_u32 s83, 8, s83                              // use shift to fill 0 for outside element
s_lshl_b32 s83, s83, 4                             // use shift to fill 0 for outside element
v_lshlrev_b64 v[78:79], s83, v[vgprValuA_X0_I0+0+0+0+0:vgprValuA_X0_I0+0+0+0+0+1]
v_lshlrev_b64 v[80:81], s83, v[vgprValuA_X0_I0+0+0+0+2:vgprValuA_X0_I0+0+0+0+2+1]
v_add_u32 v76, v75, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+0], v[vgprValuA_X0_I0+0+0+0+0], v78, s[84:85]
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0+1], v79, s[84:85]
v_add_u32 v76, v76, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+2], v[vgprValuA_X0_I0+0+0+0+2], v80, s[84:85]
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+0+0+0+3], v81, s[84:85]
v_lshlrev_b64 v[78:79], s83, v[vgprValuA_X0_I0+4+0+0+0:vgprValuA_X0_I0+4+0+0+0+1]
v_lshlrev_b64 v[80:81], s83, v[vgprValuA_X0_I0+4+0+0+2:vgprValuA_X0_I0+4+0+0+2+1]
v_add_u32 v76, v75, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+4+0+0+0], v[vgprValuA_X0_I0+4+0+0+0], v78, s[84:85]
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+4+0+0+1], v79, s[84:85]
v_add_u32 v76, v76, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+4+0+0+2], v[vgprValuA_X0_I0+4+0+0+2], v80, s[84:85]
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+4+0+0+3], v81, s[84:85]
v_lshlrev_b64 v[78:79], s83, v[vgprValuA_X0_I0+8+0+0+0:vgprValuA_X0_I0+8+0+0+0+1]
v_lshlrev_b64 v[80:81], s83, v[vgprValuA_X0_I0+8+0+0+2:vgprValuA_X0_I0+8+0+0+2+1]
v_add_u32 v76, v75, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+8+0+0+0], v[vgprValuA_X0_I0+8+0+0+0], v78, s[84:85]
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+8+0+0+1], v[vgprValuA_X0_I0+8+0+0+1], v79, s[84:85]
v_add_u32 v76, v76, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+8+0+0+2], v[vgprValuA_X0_I0+8+0+0+2], v80, s[84:85]
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+8+0+0+3], v81, s[84:85]
v_lshlrev_b64 v[78:79], s83, v[vgprValuA_X0_I0+12+0+0+0:vgprValuA_X0_I0+12+0+0+0+1]
v_lshlrev_b64 v[80:81], s83, v[vgprValuA_X0_I0+12+0+0+2:vgprValuA_X0_I0+12+0+0+2+1]
v_add_u32 v76, v75, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+12+0+0+0], v[vgprValuA_X0_I0+12+0+0+0], v78, s[84:85]
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+12+0+0+1], v[vgprValuA_X0_I0+12+0+0+1], v79, s[84:85]
v_add_u32 v76, v76, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+12+0+0+2], v[vgprValuA_X0_I0+12+0+0+2], v80, s[84:85]
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+12+0+0+3], v[vgprValuA_X0_I0+12+0+0+3], v81, s[84:85]
v_lshlrev_b64 v[78:79], s83, v[vgprValuB_X0_I0+0+0+0+0:vgprValuB_X0_I0+0+0+0+0+1]
v_lshlrev_b64 v[80:81], s83, v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1]
v_add_u32 v76, v75, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+0], v[vgprValuB_X0_I0+0+0+0+0], v78, s[84:85]
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0+1], v79, s[84:85]
v_add_u32 v76, v76, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+2], v[vgprValuB_X0_I0+0+0+0+2], v80, s[84:85]
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+3], v[vgprValuB_X0_I0+0+0+0+3], v81, s[84:85]
v_lshlrev_b64 v[78:79], s83, v[vgprValuB_X0_I0+4+0+0+0:vgprValuB_X0_I0+4+0+0+0+1]
v_lshlrev_b64 v[80:81], s83, v[vgprValuB_X0_I0+4+0+0+2:vgprValuB_X0_I0+4+0+0+2+1]
v_add_u32 v76, v75, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+4+0+0+0], v[vgprValuB_X0_I0+4+0+0+0], v78, s[84:85]
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0+1], v79, s[84:85]
v_add_u32 v76, v76, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+4+0+0+2], v[vgprValuB_X0_I0+4+0+0+2], v80, s[84:85]
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+4+0+0+3], v[vgprValuB_X0_I0+4+0+0+3], v81, s[84:85]
v_lshlrev_b64 v[78:79], s83, v[vgprValuB_X0_I0+8+0+0+0:vgprValuB_X0_I0+8+0+0+0+1]
v_lshlrev_b64 v[80:81], s83, v[vgprValuB_X0_I0+8+0+0+2:vgprValuB_X0_I0+8+0+0+2+1]
v_add_u32 v76, v75, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0+0], v[vgprValuB_X0_I0+8+0+0+0], v78, s[84:85]
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0+1], v[vgprValuB_X0_I0+8+0+0+1], v79, s[84:85]
v_add_u32 v76, v76, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0+2], v[vgprValuB_X0_I0+8+0+0+2], v80, s[84:85]
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0+3], v[vgprValuB_X0_I0+8+0+0+3], v81, s[84:85]
v_lshlrev_b64 v[78:79], s83, v[vgprValuB_X0_I0+12+0+0+0:vgprValuB_X0_I0+12+0+0+0+1]
v_lshlrev_b64 v[80:81], s83, v[vgprValuB_X0_I0+12+0+0+2:vgprValuB_X0_I0+12+0+0+2+1]
v_add_u32 v76, v75, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+12+0+0+0], v[vgprValuB_X0_I0+12+0+0+0], v78, s[84:85]
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+12+0+0+1], v[vgprValuB_X0_I0+12+0+0+1], v79, s[84:85]
v_add_u32 v76, v76, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+12+0+0+2], v[vgprValuB_X0_I0+12+0+0+2], v80, s[84:85]
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+12+0+0+3], v[vgprValuB_X0_I0+12+0+0+3], v81, s[84:85]
label_TailLoop_SkipZeroOutMask_DZOUDPYJU2HHRCOQ:
s_nop 1
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[0:3] // left value = acc[0+0:3+0]
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[4:7] // left value = acc[4+0:7+0]
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[8:11] // left value = acc[8+0:11+0]
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[12:15] // left value = acc[12+0:15+0]
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[16:19] // left value = acc[16+0:19+0]
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[20:23] // left value = acc[20+0:23+0]
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[24:27] // left value = acc[24+0:27+0]
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[28:31] // left value = acc[28+0:31+0]
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[32:35] // left value = acc[32+0:35+0]
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[36:39] // left value = acc[36+0:39+0]
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[40:43] // left value = acc[40+0:43+0]
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[44:47] // left value = acc[44+0:47+0]
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], acc[48:51] // left value = acc[48+0:51+0]
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+3], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+3], acc[52:55] // left value = acc[52+0:55+0]
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+3], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+3], acc[56:59] // left value = acc[56+0:59+0]
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+3], v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+3], acc[60:63] // left value = acc[60+0:63+0]

/* closeLoop loopL finalLoop=0 tailLoop=1 */
s_sub_i32 s[sgprLoopCounterL], s[sgprLoopCounterL], 0x20 // dec counterL (tailLoop)
s_add_u32 s[sgprOrigLoopCounter], s[sgprOrigLoopCounter], 0x20 // inc counterL
s_cmp_le_i32 s[sgprLoopCounterL], 0x0              // counterL<=0
s_cbranch_scc1 label_TailLoopEndL                  // exit LoopL

/* tail loop unroll iter 1 */

/* local read a */
ds_read_b64_tr_b16 v[vgprValuA_X1_I0+0+0:vgprValuA_X1_I0+0+0+1], v[vgprLocalReadAddrA+0] offset:0 // LDS Transpose
ds_read_b64_tr_b16 v[vgprValuA_X1_I0+0+2:vgprValuA_X1_I0+0+2+1], v[vgprLocalReadAddrA+0] offset:256 // LDS Transpose
ds_read_b64_tr_b16 v[vgprValuA_X1_I0+4+0:vgprValuA_X1_I0+4+0+1], v[vgprLocalReadAddrA+0] offset:64 // LDS Transpose
ds_read_b64_tr_b16 v[vgprValuA_X1_I0+4+2:vgprValuA_X1_I0+4+2+1], v[vgprLocalReadAddrA+0] offset:320 // LDS Transpose
ds_read_b64_tr_b16 v[vgprValuA_X1_I0+8+0:vgprValuA_X1_I0+8+0+1], v[vgprLocalReadAddrA+0] offset:128 // LDS Transpose
ds_read_b64_tr_b16 v[vgprValuA_X1_I0+8+2:vgprValuA_X1_I0+8+2+1], v[vgprLocalReadAddrA+0] offset:384 // LDS Transpose
ds_read_b64_tr_b16 v[vgprValuA_X1_I0+12+0:vgprValuA_X1_I0+12+0+1], v[vgprLocalReadAddrA+0] offset:192 // LDS Transpose
ds_read_b64_tr_b16 v[vgprValuA_X1_I0+12+2:vgprValuA_X1_I0+12+2+1], v[vgprLocalReadAddrA+0] offset:448 // LDS Transpose

/* local read b */
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[vgprLocalReadAddrB] offset:384 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read inc a */
/* Adding additional 256 pad since cumulative inc has reached 1024 */
s_mov_b32 s83, 8448                                // inc
v_add_co_u32 v[vgprLocalReadAddrA+0], vcc, s83, v[vgprLocalReadAddrA+0] // lrA += 8192 ((MT+PAD)*bpeDS)

/* local read inc b */
s_mov_b32 s83, 64                                  // inc
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, s83, v[vgprLocalReadAddrB+0] // lrB += 64 (bpeDS)
s_waitcnt lgkmcnt(0)                               // 4wait for local read
v_and_b32 v75, 63, v[vgprSerial]                   // v75 = v[vgprSerial] % 64
v_lshrrev_b32 v75, 4, v75                          // 75 = 75 / 16
v_lshlrev_b32 v75, 3, v75                          // v75 = v75 * 8
v_add_u32 v76, v75, 0
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+0+0+0+0], v[vgprValuA_X1_I0+0+0+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X1_I0+4+0+0+0], v[vgprValuA_X1_I0+4+0+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X1_I0+8+0+0+0], v[vgprValuA_X1_I0+8+0+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X1_I0+12+0+0+0], v[vgprValuA_X1_I0+12+0+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X1_I0+0+0+0+1], v[vgprValuA_X1_I0+0+0+0+1], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X1_I0+4+0+0+1], v[vgprValuA_X1_I0+4+0+0+1], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X1_I0+8+0+0+1], v[vgprValuA_X1_I0+8+0+0+1], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X1_I0+12+0+0+1], v[vgprValuA_X1_I0+12+0+0+1], 0, s[84:85] // set 0 if K_idx >= sizeL
v_add_u32 v76, v76, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+0+0+0+2], v[vgprValuA_X1_I0+0+0+0+2], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X1_I0+4+0+0+2], v[vgprValuA_X1_I0+4+0+0+2], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X1_I0+8+0+0+2], v[vgprValuA_X1_I0+8+0+0+2], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X1_I0+12+0+0+2], v[vgprValuA_X1_I0+12+0+0+2], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+0+0+0+3], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+4+0+0+3], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+8+0+0+3], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X1_I0+12+0+0+3], v[vgprValuA_X1_I0+12+0+0+3], 0, s[84:85] // set 0 if K_idx >= sizeL
v_and_b32 v75, 63, v[vgprSerial]                   // v75 = v[vgprSerial] % 64
v_lshrrev_b32 v75, 4, v75                          // 75 = 75 / 16
v_lshlrev_b32 v75, 3, v75                          // v75 = v75 * 8
v_add_u32 v76, v75, 0
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+0+0+0+0], v[vgprValuB_X1_I0+0+0+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+4+0+0+0], v[vgprValuB_X1_I0+4+0+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+8+0+0+0], v[vgprValuB_X1_I0+8+0+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+12+0+0+0], v[vgprValuB_X1_I0+12+0+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+0+0+0+1], v[vgprValuB_X1_I0+0+0+0+1], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+4+0+0+1], v[vgprValuB_X1_I0+4+0+0+1], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+8+0+0+1], v[vgprValuB_X1_I0+8+0+0+1], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+12+0+0+1], v[vgprValuB_X1_I0+12+0+0+1], 0, s[84:85] // set 0 if K_idx >= sizeL
v_add_u32 v76, v76, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+0+0+0+2], v[vgprValuB_X1_I0+0+0+0+2], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+4+0+0+2], v[vgprValuB_X1_I0+4+0+0+2], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+8+0+0+2], v[vgprValuB_X1_I0+8+0+0+2], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+12+0+0+2], v[vgprValuB_X1_I0+12+0+0+2], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+0+0+0+3], v[vgprValuB_X1_I0+0+0+0+3], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+4+0+0+3], v[vgprValuB_X1_I0+4+0+0+3], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+8+0+0+3], v[vgprValuB_X1_I0+8+0+0+3], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+12+0+0+3], v[vgprValuB_X1_I0+12+0+0+3], 0, s[84:85] // set 0 if K_idx >= sizeL
s_and_b32 s83, s[sgprSizeL], 7                     // if summation is multiple of 8, skip masking
s_cmp_eq_u32 s83, 0
s_cbranch_scc1 label_TailLoop_SkipZeroOutMask_QWMA7J3AUDGL0X23 // skip mask
s_and_b32 s83, s[sgprLoopCounterL], 7              // get inputs for edge thread
s_sub_u32 s83, 8, s83                              // use shift to fill 0 for outside element
s_lshl_b32 s83, s83, 4                             // use shift to fill 0 for outside element
v_lshlrev_b64 v[78:79], s83, v[vgprValuA_X1_I0+0+0+0+0:vgprValuA_X1_I0+0+0+0+0+1]
v_lshlrev_b64 v[80:81], s83, v[vgprValuA_X1_I0+0+0+0+2:vgprValuA_X1_I0+0+0+0+2+1]
v_add_u32 v76, v75, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+0+0+0+0], v[vgprValuA_X1_I0+0+0+0+0], v78, s[84:85]
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+0+0+0+1], v[vgprValuA_X1_I0+0+0+0+1], v79, s[84:85]
v_add_u32 v76, v76, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+0+0+0+2], v[vgprValuA_X1_I0+0+0+0+2], v80, s[84:85]
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+0+0+0+3], v81, s[84:85]
v_lshlrev_b64 v[78:79], s83, v[vgprValuA_X1_I0+4+0+0+0:vgprValuA_X1_I0+4+0+0+0+1]
v_lshlrev_b64 v[80:81], s83, v[vgprValuA_X1_I0+4+0+0+2:vgprValuA_X1_I0+4+0+0+2+1]
v_add_u32 v76, v75, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+4+0+0+0], v[vgprValuA_X1_I0+4+0+0+0], v78, s[84:85]
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+4+0+0+1], v[vgprValuA_X1_I0+4+0+0+1], v79, s[84:85]
v_add_u32 v76, v76, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+4+0+0+2], v[vgprValuA_X1_I0+4+0+0+2], v80, s[84:85]
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+4+0+0+3], v81, s[84:85]
v_lshlrev_b64 v[78:79], s83, v[vgprValuA_X1_I0+8+0+0+0:vgprValuA_X1_I0+8+0+0+0+1]
v_lshlrev_b64 v[80:81], s83, v[vgprValuA_X1_I0+8+0+0+2:vgprValuA_X1_I0+8+0+0+2+1]
v_add_u32 v76, v75, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+8+0+0+0], v[vgprValuA_X1_I0+8+0+0+0], v78, s[84:85]
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+8+0+0+1], v[vgprValuA_X1_I0+8+0+0+1], v79, s[84:85]
v_add_u32 v76, v76, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+8+0+0+2], v[vgprValuA_X1_I0+8+0+0+2], v80, s[84:85]
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+8+0+0+3], v81, s[84:85]
v_lshlrev_b64 v[78:79], s83, v[vgprValuA_X1_I0+12+0+0+0:vgprValuA_X1_I0+12+0+0+0+1]
v_lshlrev_b64 v[80:81], s83, v[vgprValuA_X1_I0+12+0+0+2:vgprValuA_X1_I0+12+0+0+2+1]
v_add_u32 v76, v75, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+12+0+0+0], v[vgprValuA_X1_I0+12+0+0+0], v78, s[84:85]
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+12+0+0+1], v[vgprValuA_X1_I0+12+0+0+1], v79, s[84:85]
v_add_u32 v76, v76, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+12+0+0+2], v[vgprValuA_X1_I0+12+0+0+2], v80, s[84:85]
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+12+0+0+3], v[vgprValuA_X1_I0+12+0+0+3], v81, s[84:85]
v_lshlrev_b64 v[78:79], s83, v[vgprValuB_X1_I0+0+0+0+0:vgprValuB_X1_I0+0+0+0+0+1]
v_lshlrev_b64 v[80:81], s83, v[vgprValuB_X1_I0+0+0+0+2:vgprValuB_X1_I0+0+0+0+2+1]
v_add_u32 v76, v75, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+0+0+0+0], v[vgprValuB_X1_I0+0+0+0+0], v78, s[84:85]
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+0+0+0+1], v[vgprValuB_X1_I0+0+0+0+1], v79, s[84:85]
v_add_u32 v76, v76, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+0+0+0+2], v[vgprValuB_X1_I0+0+0+0+2], v80, s[84:85]
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+0+0+0+3], v[vgprValuB_X1_I0+0+0+0+3], v81, s[84:85]
v_lshlrev_b64 v[78:79], s83, v[vgprValuB_X1_I0+4+0+0+0:vgprValuB_X1_I0+4+0+0+0+1]
v_lshlrev_b64 v[80:81], s83, v[vgprValuB_X1_I0+4+0+0+2:vgprValuB_X1_I0+4+0+0+2+1]
v_add_u32 v76, v75, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+4+0+0+0], v[vgprValuB_X1_I0+4+0+0+0], v78, s[84:85]
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+4+0+0+1], v[vgprValuB_X1_I0+4+0+0+1], v79, s[84:85]
v_add_u32 v76, v76, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+4+0+0+2], v[vgprValuB_X1_I0+4+0+0+2], v80, s[84:85]
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+4+0+0+3], v[vgprValuB_X1_I0+4+0+0+3], v81, s[84:85]
v_lshlrev_b64 v[78:79], s83, v[vgprValuB_X1_I0+8+0+0+0:vgprValuB_X1_I0+8+0+0+0+1]
v_lshlrev_b64 v[80:81], s83, v[vgprValuB_X1_I0+8+0+0+2:vgprValuB_X1_I0+8+0+0+2+1]
v_add_u32 v76, v75, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+8+0+0+0], v[vgprValuB_X1_I0+8+0+0+0], v78, s[84:85]
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+8+0+0+1], v[vgprValuB_X1_I0+8+0+0+1], v79, s[84:85]
v_add_u32 v76, v76, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+8+0+0+2], v[vgprValuB_X1_I0+8+0+0+2], v80, s[84:85]
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+8+0+0+3], v[vgprValuB_X1_I0+8+0+0+3], v81, s[84:85]
v_lshlrev_b64 v[78:79], s83, v[vgprValuB_X1_I0+12+0+0+0:vgprValuB_X1_I0+12+0+0+0+1]
v_lshlrev_b64 v[80:81], s83, v[vgprValuB_X1_I0+12+0+0+2:vgprValuB_X1_I0+12+0+0+2+1]
v_add_u32 v76, v75, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+12+0+0+0], v[vgprValuB_X1_I0+12+0+0+0], v78, s[84:85]
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+12+0+0+1], v[vgprValuB_X1_I0+12+0+0+1], v79, s[84:85]
v_add_u32 v76, v76, 4                              // add part of K
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+12+0+0+2], v[vgprValuB_X1_I0+12+0+0+2], v80, s[84:85]
v_cmp_ge_i32 s[84:85], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+12+0+0+3], v[vgprValuB_X1_I0+12+0+0+3], v81, s[84:85]
label_TailLoop_SkipZeroOutMask_QWMA7J3AUDGL0X23:
s_nop 1
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[0:3] // left value = acc[0+0:3+0]
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[4:7] // left value = acc[4+0:7+0]
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[8:11] // left value = acc[8+0:11+0]
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[12:15] // left value = acc[12+0:15+0]
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[16:19] // left value = acc[16+0:19+0]
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[20:23] // left value = acc[20+0:23+0]
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[24:27] // left value = acc[24+0:27+0]
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[28:31] // left value = acc[28+0:31+0]
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[32:35] // left value = acc[32+0:35+0]
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[36:39] // left value = acc[36+0:39+0]
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[40:43] // left value = acc[40+0:43+0]
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X1_I0+8+0+0:vgprValuB_X1_I0+8+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[44:47] // left value = acc[44+0:47+0]
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X1_I0+12+0+0:vgprValuB_X1_I0+12+0+0+3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], acc[48:51] // left value = acc[48+0:51+0]
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X1_I0+12+0+0:vgprValuB_X1_I0+12+0+0+3], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+3], acc[52:55] // left value = acc[52+0:55+0]
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X1_I0+12+0+0:vgprValuB_X1_I0+12+0+0+3], v[vgprValuA_X1_I0+8+0+0:vgprValuA_X1_I0+8+0+0+3], acc[56:59] // left value = acc[56+0:59+0]
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X1_I0+12+0+0:vgprValuB_X1_I0+12+0+0+3], v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+3], acc[60:63] // left value = acc[60+0:63+0]

/* closeLoop loopL finalLoop=1 tailLoop=1 */
s_sub_i32 s[sgprLoopCounterL], s[sgprLoopCounterL], 0x20 // dec counterL (tailLoop)
s_add_u32 s[sgprOrigLoopCounter], s[sgprOrigLoopCounter], 0x20 // inc counterL
s_cmp_le_i32 s[sgprLoopCounterL], 0x0              // counterL<=0
s_cbranch_scc0 label_TailLoopBeginL                // restart LoopL
label_TailLoopEndL:
s_mov_b32 s83, 256                                 // tailloop lds offset
s_mul_i32 s83, s[sgprOrigLoopCounter], s83         // scale by mul
v_sub_u32 v[vgprLocalReadAddrA], v[vgprLocalReadAddrA], s83 // remove lro damage
s_mov_b32 s83, 2                                   // tailloop lds offset
s_mul_i32 s83, s[sgprOrigLoopCounter], s83         // scale by mul
v_sub_u32 v[vgprLocalReadAddrB], v[vgprLocalReadAddrB], s83 // remove lro damage
label_SkipTailLoopL:
.set vgprValuA_X0_I0_BASE, UNDEF
.set vgprValuA_X0_I0, UNDEF
.set vgprValuA_X1_I0, UNDEF
.set vgprValuB_X0_I0_BASE, UNDEF
.set vgprValuB_X0_I0, UNDEF
.set vgprValuB_X1_I0, UNDEF
label_Summation_End_2G3LC8VCGIZD1EUX:
.set sgprLoopCounterL, UNDEF
.set sgprOrigLoopCounter, UNDEF
.set sgprSrdA, UNDEF
.set sgprSrdB, UNDEF
.set sgprShadowLimitA, UNDEF
.set sgprShadowLimitB, UNDEF
.set sgprStaggerUIter, UNDEF
.set sgprWrapUA, UNDEF
.set sgprWrapUB, UNDEF
.set sgprGlobalReadIncsA, UNDEF
.set sgprGlobalReadIncsB, UNDEF
/* load store sgprs */
.set sgprAddressScaleAlphaVec, 64
.set sgprAddressBias, 66
.set sgprBiasType, 68
.set sgprBiasStride, 69
/* Check if custom structure pointer is null */
s_cmp_eq_u32 s[sgprArgType], 2                     // ArgType == 2 ?
s_cbranch_scc1 label_LoadExternalEpilogueStruct    // branch if ArgType == 2
s_load_dwordx4 s[64:67], s[sgprKernArgAddress:sgprKernArgAddress+1], 132 // 132
s_load_dwordx2 s[68:69], s[sgprKernArgAddress:sgprKernArgAddress+1], 148 // 148
s_branch label_LoadExternalEpilogueStructEnd
label_LoadExternalEpilogueStruct:
s_load_dwordx4 s[64:67], s[sgprKernArgAddress:sgprKernArgAddress+1], 188 // 188
s_load_dwordx2 s[68:69], s[sgprKernArgAddress:sgprKernArgAddress+1], 204 // 204
label_LoadExternalEpilogueStructEnd:
.set sgprSrdScaleAlphaVec, 72
.set sgprSrdBias, 76

/* Mapping of Acc register -> C Vgpr register */

/* shift vector components d0 */
v_mov_b32 v13, s[sgprWorkGroup0]
v_mul_i32_i24 v13, -0x80, v13                      // wg*MT
v_add_co_u32 v13, vcc, s[sgprSizesFree+0], v13     // wgMT = Size - wg*MT
v_mov_b32 v14, 0x80                                // MT
v_cmp_lt_u32 s[8:9], v13, v14                      // wgMT < MT
v_cndmask_b32 v13, v14, v13, s[8:9]                // wgMT = (wgMT < MT) ? wgMT : MT
v_lshrrev_b32 v15, 6, v[vgprSerial]                // 15 = Serial / 64
v_and_b32 v15, 1, v15                              // v15 = v15 % 2
v_lshrrev_b32 v16, 4, v13                          // 16 = 13 / 16
v_and_b32 v16, 1, v16                              // v16 = v16 % 2
v_cmp_eq_u32 s[8:9], v16, v15                      // wave_id == block_belong_to_wave?
v_cndmask_b32 v13, v14, v13, s[8:9]                // wgMT = (wgMT < MT) ? wgMT : MT

/* mbReg: which mb block need to shift, mb(matrixInstCoal(16) * VectorWidth(1)) */
v_lshrrev_b32 v14, 4, v13                          // 14 = 13 / 16
v_lshlrev_b32 v16, 0, v15                          // v16 = v15 * 1
v_sub_u32 v14, v14, v16

/* gbReg: glvw block id */
v_lshrrev_b32 v16, 3, v13                          // 16 = 13 / 8

/* tgbReg: glvw block id */
v_lshrrev_b32 v17, 0, v[vgprSerial]                // 17 = Serial / 1
v_and_b32 v17, 15, v17                             // v17 = v17 % 16
                                                   // v17 = v17 * 1 (multiplier is 1, do nothing)
v_lshrrev_b32 v17, 3, v17                          // 17 = 17 / 8
v_lshlrev_b32 v15, 1, v15                          // v15 = v15 * 2
v_add_co_u32 v17, vcc, v15, v17                    // tgbReg = (tid_coal * continOut) / GLVW
v_sub_u32 v16, v16, v17

/* vwReg: glvw in which vw block? */
v_and_b32 v15, 0, v13                              // permute register between threads
v_lshrrev_b32 v15, 3, v15                          // permute register between threads

/* rReg : reminder of M_size % GlobalReadVectorWidth */
v_and_b32 v17, 7, v13                              // v17 = v13 % 8
v_cmp_eq_u32 vcc, v17, 0x1                         // wgMT%VW == 1
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1 // branch to shift d0 r=1
v_cmp_eq_u32 vcc, v17, 0x2                         // wgMT%VW == 2
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW2 // branch to shift d0 r=2
v_cmp_eq_u32 vcc, v17, 0x3                         // wgMT%VW == 3
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW3 // branch to shift d0 r=3
v_cmp_eq_u32 vcc, v17, 0x4                         // wgMT%VW == 4
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW4 // branch to shift d0 r=4
v_cmp_eq_u32 vcc, v17, 0x5                         // wgMT%VW == 5
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW5 // branch to shift d0 r=5
v_cmp_eq_u32 vcc, v17, 0x6                         // wgMT%VW == 6
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW6 // branch to shift d0 r=6
v_cmp_eq_u32 vcc, v17, 0x7                         // wgMT%VW == 7
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW7 // branch to shift d0 r=7

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0

/******************************************/
/* shift d0 r=1                           */
/******************************************/
label_ShiftVectorComponents0_GLVW1:
v_cmp_eq_u32 vcc, v14, 0x0

/* branch to shift d0 r1 mb0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM0
v_cmp_eq_u32 vcc, v14, 0x2

/* branch to shift d0 r1 mb1 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM1
v_cmp_eq_u32 vcc, v14, 0x4

/* branch to shift d0 r1 mb2 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM2
v_cmp_eq_u32 vcc, v14, 0x6

/* branch to shift d0 r1 mb3 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM3

/******************************************/
/* shift d0 r=2                           */
/******************************************/
label_ShiftVectorComponents0_GLVW2:
v_cmp_eq_u32 vcc, v14, 0x0

/* branch to shift d0 r2 mb0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW2_BM0
v_cmp_eq_u32 vcc, v14, 0x2

/* branch to shift d0 r2 mb1 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW2_BM1
v_cmp_eq_u32 vcc, v14, 0x4

/* branch to shift d0 r2 mb2 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW2_BM2
v_cmp_eq_u32 vcc, v14, 0x6

/* branch to shift d0 r2 mb3 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW2_BM3

/******************************************/
/* shift d0 r=3                           */
/******************************************/
label_ShiftVectorComponents0_GLVW3:
v_cmp_eq_u32 vcc, v14, 0x0

/* branch to shift d0 r3 mb0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW3_BM0
v_cmp_eq_u32 vcc, v14, 0x2

/* branch to shift d0 r3 mb1 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW3_BM1
v_cmp_eq_u32 vcc, v14, 0x4

/* branch to shift d0 r3 mb2 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW3_BM2
v_cmp_eq_u32 vcc, v14, 0x6

/* branch to shift d0 r3 mb3 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW3_BM3

/******************************************/
/* shift d0 r=4                           */
/******************************************/
label_ShiftVectorComponents0_GLVW4:
v_cmp_eq_u32 vcc, v14, 0x0

/* branch to shift d0 r4 mb0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW4_BM0
v_cmp_eq_u32 vcc, v14, 0x2

/* branch to shift d0 r4 mb1 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW4_BM1
v_cmp_eq_u32 vcc, v14, 0x4

/* branch to shift d0 r4 mb2 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW4_BM2
v_cmp_eq_u32 vcc, v14, 0x6

/* branch to shift d0 r4 mb3 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW4_BM3

/******************************************/
/* shift d0 r=5                           */
/******************************************/
label_ShiftVectorComponents0_GLVW5:
v_cmp_eq_u32 vcc, v14, 0x0

/* branch to shift d0 r5 mb0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW5_BM0
v_cmp_eq_u32 vcc, v14, 0x2

/* branch to shift d0 r5 mb1 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW5_BM1
v_cmp_eq_u32 vcc, v14, 0x4

/* branch to shift d0 r5 mb2 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW5_BM2
v_cmp_eq_u32 vcc, v14, 0x6

/* branch to shift d0 r5 mb3 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW5_BM3

/******************************************/
/* shift d0 r=6                           */
/******************************************/
label_ShiftVectorComponents0_GLVW6:
v_cmp_eq_u32 vcc, v14, 0x0

/* branch to shift d0 r6 mb0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW6_BM0
v_cmp_eq_u32 vcc, v14, 0x2

/* branch to shift d0 r6 mb1 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW6_BM1
v_cmp_eq_u32 vcc, v14, 0x4

/* branch to shift d0 r6 mb2 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW6_BM2
v_cmp_eq_u32 vcc, v14, 0x6

/* branch to shift d0 r6 mb3 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW6_BM3

/******************************************/
/* shift d0 r=7                           */
/******************************************/
label_ShiftVectorComponents0_GLVW7:
v_cmp_eq_u32 vcc, v14, 0x0

/* branch to shift d0 r7 mb0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW7_BM0
v_cmp_eq_u32 vcc, v14, 0x2

/* branch to shift d0 r7 mb1 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW7_BM1
v_cmp_eq_u32 vcc, v14, 0x4

/* branch to shift d0 r7 mb2 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW7_BM2
v_cmp_eq_u32 vcc, v14, 0x6

/* branch to shift d0 r7 mb3 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW7_BM3

/******************************************/
/* shift d0 r=1 mb=0                      */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM0:  /// r1 mb0
v_cmp_eq_u32 vcc, v15, 0x0

/* branch to shift d0 r1 mb0 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM0_VW0

/******************************************/
/* shift d0 r=1 mb=1                      */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM1:  /// r1 mb1
v_cmp_eq_u32 vcc, v15, 0x0

/* branch to shift d0 r1 mb1 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM1_VW0

/******************************************/
/* shift d0 r=1 mb=2                      */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM2:  /// r1 mb2
v_cmp_eq_u32 vcc, v15, 0x0

/* branch to shift d0 r1 mb2 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM2_VW0

/******************************************/
/* shift d0 r=1 mb=3                      */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM3:  /// r1 mb3
v_cmp_eq_u32 vcc, v15, 0x0

/* branch to shift d0 r1 mb3 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM3_VW0

/******************************************/
/* shift d0 r=2 mb=0                      */
/******************************************/
label_ShiftVectorComponents0_GLVW2_BM0:  /// r2 mb0
v_cmp_eq_u32 vcc, v15, 0x0

/* branch to shift d0 r2 mb0 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW2_BM0_VW0

/******************************************/
/* shift d0 r=2 mb=1                      */
/******************************************/
label_ShiftVectorComponents0_GLVW2_BM1:  /// r2 mb1
v_cmp_eq_u32 vcc, v15, 0x0

/* branch to shift d0 r2 mb1 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW2_BM1_VW0

/******************************************/
/* shift d0 r=2 mb=2                      */
/******************************************/
label_ShiftVectorComponents0_GLVW2_BM2:  /// r2 mb2
v_cmp_eq_u32 vcc, v15, 0x0

/* branch to shift d0 r2 mb2 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW2_BM2_VW0

/******************************************/
/* shift d0 r=2 mb=3                      */
/******************************************/
label_ShiftVectorComponents0_GLVW2_BM3:  /// r2 mb3
v_cmp_eq_u32 vcc, v15, 0x0

/* branch to shift d0 r2 mb3 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW2_BM3_VW0

/******************************************/
/* shift d0 r=3 mb=0                      */
/******************************************/
label_ShiftVectorComponents0_GLVW3_BM0:  /// r3 mb0
v_cmp_eq_u32 vcc, v15, 0x0

/* branch to shift d0 r3 mb0 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW3_BM0_VW0

/******************************************/
/* shift d0 r=3 mb=1                      */
/******************************************/
label_ShiftVectorComponents0_GLVW3_BM1:  /// r3 mb1
v_cmp_eq_u32 vcc, v15, 0x0

/* branch to shift d0 r3 mb1 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW3_BM1_VW0

/******************************************/
/* shift d0 r=3 mb=2                      */
/******************************************/
label_ShiftVectorComponents0_GLVW3_BM2:  /// r3 mb2
v_cmp_eq_u32 vcc, v15, 0x0

/* branch to shift d0 r3 mb2 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW3_BM2_VW0

/******************************************/
/* shift d0 r=3 mb=3                      */
/******************************************/
label_ShiftVectorComponents0_GLVW3_BM3:  /// r3 mb3
v_cmp_eq_u32 vcc, v15, 0x0

/* branch to shift d0 r3 mb3 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW3_BM3_VW0

/******************************************/
/* shift d0 r=4 mb=0                      */
/******************************************/
label_ShiftVectorComponents0_GLVW4_BM0:  /// r4 mb0
v_cmp_eq_u32 vcc, v15, 0x0

/* branch to shift d0 r4 mb0 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW4_BM0_VW0

/******************************************/
/* shift d0 r=4 mb=1                      */
/******************************************/
label_ShiftVectorComponents0_GLVW4_BM1:  /// r4 mb1
v_cmp_eq_u32 vcc, v15, 0x0

/* branch to shift d0 r4 mb1 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW4_BM1_VW0

/******************************************/
/* shift d0 r=4 mb=2                      */
/******************************************/
label_ShiftVectorComponents0_GLVW4_BM2:  /// r4 mb2
v_cmp_eq_u32 vcc, v15, 0x0

/* branch to shift d0 r4 mb2 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW4_BM2_VW0

/******************************************/
/* shift d0 r=4 mb=3                      */
/******************************************/
label_ShiftVectorComponents0_GLVW4_BM3:  /// r4 mb3
v_cmp_eq_u32 vcc, v15, 0x0

/* branch to shift d0 r4 mb3 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW4_BM3_VW0

/******************************************/
/* shift d0 r=5 mb=0                      */
/******************************************/
label_ShiftVectorComponents0_GLVW5_BM0:  /// r5 mb0
v_cmp_eq_u32 vcc, v15, 0x0

/* branch to shift d0 r5 mb0 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW5_BM0_VW0

/******************************************/
/* shift d0 r=5 mb=1                      */
/******************************************/
label_ShiftVectorComponents0_GLVW5_BM1:  /// r5 mb1
v_cmp_eq_u32 vcc, v15, 0x0

/* branch to shift d0 r5 mb1 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW5_BM1_VW0

/******************************************/
/* shift d0 r=5 mb=2                      */
/******************************************/
label_ShiftVectorComponents0_GLVW5_BM2:  /// r5 mb2
v_cmp_eq_u32 vcc, v15, 0x0

/* branch to shift d0 r5 mb2 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW5_BM2_VW0

/******************************************/
/* shift d0 r=5 mb=3                      */
/******************************************/
label_ShiftVectorComponents0_GLVW5_BM3:  /// r5 mb3
v_cmp_eq_u32 vcc, v15, 0x0

/* branch to shift d0 r5 mb3 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW5_BM3_VW0

/******************************************/
/* shift d0 r=6 mb=0                      */
/******************************************/
label_ShiftVectorComponents0_GLVW6_BM0:  /// r6 mb0
v_cmp_eq_u32 vcc, v15, 0x0

/* branch to shift d0 r6 mb0 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW6_BM0_VW0

/******************************************/
/* shift d0 r=6 mb=1                      */
/******************************************/
label_ShiftVectorComponents0_GLVW6_BM1:  /// r6 mb1
v_cmp_eq_u32 vcc, v15, 0x0

/* branch to shift d0 r6 mb1 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW6_BM1_VW0

/******************************************/
/* shift d0 r=6 mb=2                      */
/******************************************/
label_ShiftVectorComponents0_GLVW6_BM2:  /// r6 mb2
v_cmp_eq_u32 vcc, v15, 0x0

/* branch to shift d0 r6 mb2 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW6_BM2_VW0

/******************************************/
/* shift d0 r=6 mb=3                      */
/******************************************/
label_ShiftVectorComponents0_GLVW6_BM3:  /// r6 mb3
v_cmp_eq_u32 vcc, v15, 0x0

/* branch to shift d0 r6 mb3 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW6_BM3_VW0

/******************************************/
/* shift d0 r=7 mb=0                      */
/******************************************/
label_ShiftVectorComponents0_GLVW7_BM0:  /// r7 mb0
v_cmp_eq_u32 vcc, v15, 0x0

/* branch to shift d0 r7 mb0 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW7_BM0_VW0

/******************************************/
/* shift d0 r=7 mb=1                      */
/******************************************/
label_ShiftVectorComponents0_GLVW7_BM1:  /// r7 mb1
v_cmp_eq_u32 vcc, v15, 0x0

/* branch to shift d0 r7 mb1 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW7_BM1_VW0

/******************************************/
/* shift d0 r=7 mb=2                      */
/******************************************/
label_ShiftVectorComponents0_GLVW7_BM2:  /// r7 mb2
v_cmp_eq_u32 vcc, v15, 0x0

/* branch to shift d0 r7 mb2 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW7_BM2_VW0

/******************************************/
/* shift d0 r=7 mb=3                      */
/******************************************/
label_ShiftVectorComponents0_GLVW7_BM3:  /// r7 mb3
v_cmp_eq_u32 vcc, v15, 0x0

/* branch to shift d0 r7 mb3 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW7_BM3_VW0

/******************************************/
/* shift d0 r=1 mb=0 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM0_VW0:  /// r1 mb0 vw0
s_mov_b32 s8, 0
v_cmpx_eq_u32 s[8:9], v16, s8                      // is thread in edge glvw region
v_and_b32 v10, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v10, 2, v10                          // permute register between threads
v_accvgpr_read_b32 v17, acc0                       // glvw 1 mb 0 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc0, v17
v_accvgpr_read_b32 v17, acc16                      // glvw 1 mb 0 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc16, v17
v_accvgpr_read_b32 v17, acc32                      // glvw 1 mb 0 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc32, v17
v_accvgpr_read_b32 v17, acc48                      // glvw 1 mb 0 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc48, v17
v_accvgpr_read_b32 v17, acc1                       // glvw 1 mb 0 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc1, v17
v_accvgpr_read_b32 v17, acc17                      // glvw 1 mb 0 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc17, v17
v_accvgpr_read_b32 v17, acc33                      // glvw 1 mb 0 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc33, v17
v_accvgpr_read_b32 v17, acc49                      // glvw 1 mb 0 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc49, v17
v_accvgpr_read_b32 v17, acc2                       // glvw 1 mb 0 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc2, v17
v_accvgpr_read_b32 v17, acc18                      // glvw 1 mb 0 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc18, v17
v_accvgpr_read_b32 v17, acc34                      // glvw 1 mb 0 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc34, v17
v_accvgpr_read_b32 v17, acc50                      // glvw 1 mb 0 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc50, v17
v_accvgpr_read_b32 v17, acc3                       // glvw 1 mb 0 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc3, v17
v_accvgpr_read_b32 v17, acc19                      // glvw 1 mb 0 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc19, v17
v_accvgpr_read_b32 v17, acc35                      // glvw 1 mb 0 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc35, v17
v_accvgpr_read_b32 v17, acc51                      // glvw 1 mb 0 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc51, v17
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=1 mb=1 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM1_VW0:  /// r1 mb1 vw0
s_mov_b32 s8, 4
v_cmpx_eq_u32 s[8:9], v16, s8                      // is thread in edge glvw region
v_and_b32 v10, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v10, 2, v10                          // permute register between threads
v_accvgpr_read_b32 v17, acc4                       // glvw 1 mb 1 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc4, v17
v_accvgpr_read_b32 v17, acc20                      // glvw 1 mb 1 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc20, v17
v_accvgpr_read_b32 v17, acc36                      // glvw 1 mb 1 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc36, v17
v_accvgpr_read_b32 v17, acc52                      // glvw 1 mb 1 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc52, v17
v_accvgpr_read_b32 v17, acc5                       // glvw 1 mb 1 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc5, v17
v_accvgpr_read_b32 v17, acc21                      // glvw 1 mb 1 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc21, v17
v_accvgpr_read_b32 v17, acc37                      // glvw 1 mb 1 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc37, v17
v_accvgpr_read_b32 v17, acc53                      // glvw 1 mb 1 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc53, v17
v_accvgpr_read_b32 v17, acc6                       // glvw 1 mb 1 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc6, v17
v_accvgpr_read_b32 v17, acc22                      // glvw 1 mb 1 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc22, v17
v_accvgpr_read_b32 v17, acc38                      // glvw 1 mb 1 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc38, v17
v_accvgpr_read_b32 v17, acc54                      // glvw 1 mb 1 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc54, v17
v_accvgpr_read_b32 v17, acc7                       // glvw 1 mb 1 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc7, v17
v_accvgpr_read_b32 v17, acc23                      // glvw 1 mb 1 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc23, v17
v_accvgpr_read_b32 v17, acc39                      // glvw 1 mb 1 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc39, v17
v_accvgpr_read_b32 v17, acc55                      // glvw 1 mb 1 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc55, v17
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=1 mb=2 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM2_VW0:  /// r1 mb2 vw0
s_mov_b32 s8, 8
v_cmpx_eq_u32 s[8:9], v16, s8                      // is thread in edge glvw region
v_and_b32 v10, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v10, 2, v10                          // permute register between threads
v_accvgpr_read_b32 v17, acc8                       // glvw 1 mb 2 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc8, v17
v_accvgpr_read_b32 v17, acc24                      // glvw 1 mb 2 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc24, v17
v_accvgpr_read_b32 v17, acc40                      // glvw 1 mb 2 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc40, v17
v_accvgpr_read_b32 v17, acc56                      // glvw 1 mb 2 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc56, v17
v_accvgpr_read_b32 v17, acc9                       // glvw 1 mb 2 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc9, v17
v_accvgpr_read_b32 v17, acc25                      // glvw 1 mb 2 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc25, v17
v_accvgpr_read_b32 v17, acc41                      // glvw 1 mb 2 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc41, v17
v_accvgpr_read_b32 v17, acc57                      // glvw 1 mb 2 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc57, v17
v_accvgpr_read_b32 v17, acc10                      // glvw 1 mb 2 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc10, v17
v_accvgpr_read_b32 v17, acc26                      // glvw 1 mb 2 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc26, v17
v_accvgpr_read_b32 v17, acc42                      // glvw 1 mb 2 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc42, v17
v_accvgpr_read_b32 v17, acc58                      // glvw 1 mb 2 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc58, v17
v_accvgpr_read_b32 v17, acc11                      // glvw 1 mb 2 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc11, v17
v_accvgpr_read_b32 v17, acc27                      // glvw 1 mb 2 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc27, v17
v_accvgpr_read_b32 v17, acc43                      // glvw 1 mb 2 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc43, v17
v_accvgpr_read_b32 v17, acc59                      // glvw 1 mb 2 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc59, v17
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=1 mb=3 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM3_VW0:  /// r1 mb3 vw0
s_mov_b32 s8, 12
v_cmpx_eq_u32 s[8:9], v16, s8                      // is thread in edge glvw region
v_and_b32 v10, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v10, 2, v10                          // permute register between threads
v_accvgpr_read_b32 v17, acc12                      // glvw 1 mb 3 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc12, v17
v_accvgpr_read_b32 v17, acc28                      // glvw 1 mb 3 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc28, v17
v_accvgpr_read_b32 v17, acc44                      // glvw 1 mb 3 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc44, v17
v_accvgpr_read_b32 v17, acc60                      // glvw 1 mb 3 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc60, v17
v_accvgpr_read_b32 v17, acc13                      // glvw 1 mb 3 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc13, v17
v_accvgpr_read_b32 v17, acc29                      // glvw 1 mb 3 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc29, v17
v_accvgpr_read_b32 v17, acc45                      // glvw 1 mb 3 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc45, v17
v_accvgpr_read_b32 v17, acc61                      // glvw 1 mb 3 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc61, v17
v_accvgpr_read_b32 v17, acc14                      // glvw 1 mb 3 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc14, v17
v_accvgpr_read_b32 v17, acc30                      // glvw 1 mb 3 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc30, v17
v_accvgpr_read_b32 v17, acc46                      // glvw 1 mb 3 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc46, v17
v_accvgpr_read_b32 v17, acc62                      // glvw 1 mb 3 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc62, v17
v_accvgpr_read_b32 v17, acc15                      // glvw 1 mb 3 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc15, v17
v_accvgpr_read_b32 v17, acc31                      // glvw 1 mb 3 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc31, v17
v_accvgpr_read_b32 v17, acc47                      // glvw 1 mb 3 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc47, v17
v_accvgpr_read_b32 v17, acc63                      // glvw 1 mb 3 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:28            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc63, v17
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=2 mb=0 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW2_BM0_VW0:  /// r2 mb0 vw0
s_mov_b32 s8, 0
v_cmpx_eq_u32 s[8:9], v16, s8                      // is thread in edge glvw region
v_and_b32 v10, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v10, 2, v10                          // permute register between threads
v_accvgpr_read_b32 v17, acc0                       // glvw 2 mb 0 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc0, v17
v_accvgpr_read_b32 v17, acc16                      // glvw 2 mb 0 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc16, v17
v_accvgpr_read_b32 v17, acc32                      // glvw 2 mb 0 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc32, v17
v_accvgpr_read_b32 v17, acc48                      // glvw 2 mb 0 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc48, v17
v_accvgpr_read_b32 v17, acc1                       // glvw 2 mb 0 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc1, v17
v_accvgpr_read_b32 v17, acc17                      // glvw 2 mb 0 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc17, v17
v_accvgpr_read_b32 v17, acc33                      // glvw 2 mb 0 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc33, v17
v_accvgpr_read_b32 v17, acc49                      // glvw 2 mb 0 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc49, v17
v_accvgpr_read_b32 v17, acc2                       // glvw 2 mb 0 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc2, v17
v_accvgpr_read_b32 v17, acc18                      // glvw 2 mb 0 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc18, v17
v_accvgpr_read_b32 v17, acc34                      // glvw 2 mb 0 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc34, v17
v_accvgpr_read_b32 v17, acc50                      // glvw 2 mb 0 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc50, v17
v_accvgpr_read_b32 v17, acc3                       // glvw 2 mb 0 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc3, v17
v_accvgpr_read_b32 v17, acc19                      // glvw 2 mb 0 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc19, v17
v_accvgpr_read_b32 v17, acc35                      // glvw 2 mb 0 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc35, v17
v_accvgpr_read_b32 v17, acc51                      // glvw 2 mb 0 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc51, v17
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=2 mb=1 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW2_BM1_VW0:  /// r2 mb1 vw0
s_mov_b32 s8, 4
v_cmpx_eq_u32 s[8:9], v16, s8                      // is thread in edge glvw region
v_and_b32 v10, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v10, 2, v10                          // permute register between threads
v_accvgpr_read_b32 v17, acc4                       // glvw 2 mb 1 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc4, v17
v_accvgpr_read_b32 v17, acc20                      // glvw 2 mb 1 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc20, v17
v_accvgpr_read_b32 v17, acc36                      // glvw 2 mb 1 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc36, v17
v_accvgpr_read_b32 v17, acc52                      // glvw 2 mb 1 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc52, v17
v_accvgpr_read_b32 v17, acc5                       // glvw 2 mb 1 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc5, v17
v_accvgpr_read_b32 v17, acc21                      // glvw 2 mb 1 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc21, v17
v_accvgpr_read_b32 v17, acc37                      // glvw 2 mb 1 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc37, v17
v_accvgpr_read_b32 v17, acc53                      // glvw 2 mb 1 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc53, v17
v_accvgpr_read_b32 v17, acc6                       // glvw 2 mb 1 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc6, v17
v_accvgpr_read_b32 v17, acc22                      // glvw 2 mb 1 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc22, v17
v_accvgpr_read_b32 v17, acc38                      // glvw 2 mb 1 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc38, v17
v_accvgpr_read_b32 v17, acc54                      // glvw 2 mb 1 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc54, v17
v_accvgpr_read_b32 v17, acc7                       // glvw 2 mb 1 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc7, v17
v_accvgpr_read_b32 v17, acc23                      // glvw 2 mb 1 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc23, v17
v_accvgpr_read_b32 v17, acc39                      // glvw 2 mb 1 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc39, v17
v_accvgpr_read_b32 v17, acc55                      // glvw 2 mb 1 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc55, v17
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=2 mb=2 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW2_BM2_VW0:  /// r2 mb2 vw0
s_mov_b32 s8, 8
v_cmpx_eq_u32 s[8:9], v16, s8                      // is thread in edge glvw region
v_and_b32 v10, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v10, 2, v10                          // permute register between threads
v_accvgpr_read_b32 v17, acc8                       // glvw 2 mb 2 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc8, v17
v_accvgpr_read_b32 v17, acc24                      // glvw 2 mb 2 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc24, v17
v_accvgpr_read_b32 v17, acc40                      // glvw 2 mb 2 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc40, v17
v_accvgpr_read_b32 v17, acc56                      // glvw 2 mb 2 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc56, v17
v_accvgpr_read_b32 v17, acc9                       // glvw 2 mb 2 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc9, v17
v_accvgpr_read_b32 v17, acc25                      // glvw 2 mb 2 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc25, v17
v_accvgpr_read_b32 v17, acc41                      // glvw 2 mb 2 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc41, v17
v_accvgpr_read_b32 v17, acc57                      // glvw 2 mb 2 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc57, v17
v_accvgpr_read_b32 v17, acc10                      // glvw 2 mb 2 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc10, v17
v_accvgpr_read_b32 v17, acc26                      // glvw 2 mb 2 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc26, v17
v_accvgpr_read_b32 v17, acc42                      // glvw 2 mb 2 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc42, v17
v_accvgpr_read_b32 v17, acc58                      // glvw 2 mb 2 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc58, v17
v_accvgpr_read_b32 v17, acc11                      // glvw 2 mb 2 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc11, v17
v_accvgpr_read_b32 v17, acc27                      // glvw 2 mb 2 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc27, v17
v_accvgpr_read_b32 v17, acc43                      // glvw 2 mb 2 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc43, v17
v_accvgpr_read_b32 v17, acc59                      // glvw 2 mb 2 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc59, v17
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=2 mb=3 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW2_BM3_VW0:  /// r2 mb3 vw0
s_mov_b32 s8, 12
v_cmpx_eq_u32 s[8:9], v16, s8                      // is thread in edge glvw region
v_and_b32 v10, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v10, 2, v10                          // permute register between threads
v_accvgpr_read_b32 v17, acc12                      // glvw 2 mb 3 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc12, v17
v_accvgpr_read_b32 v17, acc28                      // glvw 2 mb 3 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc28, v17
v_accvgpr_read_b32 v17, acc44                      // glvw 2 mb 3 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc44, v17
v_accvgpr_read_b32 v17, acc60                      // glvw 2 mb 3 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc60, v17
v_accvgpr_read_b32 v17, acc13                      // glvw 2 mb 3 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc13, v17
v_accvgpr_read_b32 v17, acc29                      // glvw 2 mb 3 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc29, v17
v_accvgpr_read_b32 v17, acc45                      // glvw 2 mb 3 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc45, v17
v_accvgpr_read_b32 v17, acc61                      // glvw 2 mb 3 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc61, v17
v_accvgpr_read_b32 v17, acc14                      // glvw 2 mb 3 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc14, v17
v_accvgpr_read_b32 v17, acc30                      // glvw 2 mb 3 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc30, v17
v_accvgpr_read_b32 v17, acc46                      // glvw 2 mb 3 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc46, v17
v_accvgpr_read_b32 v17, acc62                      // glvw 2 mb 3 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc62, v17
v_accvgpr_read_b32 v17, acc15                      // glvw 2 mb 3 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc15, v17
v_accvgpr_read_b32 v17, acc31                      // glvw 2 mb 3 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc31, v17
v_accvgpr_read_b32 v17, acc47                      // glvw 2 mb 3 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc47, v17
v_accvgpr_read_b32 v17, acc63                      // glvw 2 mb 3 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:24            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc63, v17
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=3 mb=0 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW3_BM0_VW0:  /// r3 mb0 vw0
s_mov_b32 s8, 0
v_cmpx_eq_u32 s[8:9], v16, s8                      // is thread in edge glvw region
v_and_b32 v10, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v10, 2, v10                          // permute register between threads
v_accvgpr_read_b32 v17, acc0                       // glvw 3 mb 0 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc0, v17
v_accvgpr_read_b32 v17, acc16                      // glvw 3 mb 0 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc16, v17
v_accvgpr_read_b32 v17, acc32                      // glvw 3 mb 0 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc32, v17
v_accvgpr_read_b32 v17, acc48                      // glvw 3 mb 0 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc48, v17
v_accvgpr_read_b32 v17, acc1                       // glvw 3 mb 0 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc1, v17
v_accvgpr_read_b32 v17, acc17                      // glvw 3 mb 0 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc17, v17
v_accvgpr_read_b32 v17, acc33                      // glvw 3 mb 0 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc33, v17
v_accvgpr_read_b32 v17, acc49                      // glvw 3 mb 0 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc49, v17
v_accvgpr_read_b32 v17, acc2                       // glvw 3 mb 0 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc2, v17
v_accvgpr_read_b32 v17, acc18                      // glvw 3 mb 0 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc18, v17
v_accvgpr_read_b32 v17, acc34                      // glvw 3 mb 0 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc34, v17
v_accvgpr_read_b32 v17, acc50                      // glvw 3 mb 0 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc50, v17
v_accvgpr_read_b32 v17, acc3                       // glvw 3 mb 0 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc3, v17
v_accvgpr_read_b32 v17, acc19                      // glvw 3 mb 0 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc19, v17
v_accvgpr_read_b32 v17, acc35                      // glvw 3 mb 0 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc35, v17
v_accvgpr_read_b32 v17, acc51                      // glvw 3 mb 0 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc51, v17
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=3 mb=1 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW3_BM1_VW0:  /// r3 mb1 vw0
s_mov_b32 s8, 4
v_cmpx_eq_u32 s[8:9], v16, s8                      // is thread in edge glvw region
v_and_b32 v10, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v10, 2, v10                          // permute register between threads
v_accvgpr_read_b32 v17, acc4                       // glvw 3 mb 1 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc4, v17
v_accvgpr_read_b32 v17, acc20                      // glvw 3 mb 1 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc20, v17
v_accvgpr_read_b32 v17, acc36                      // glvw 3 mb 1 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc36, v17
v_accvgpr_read_b32 v17, acc52                      // glvw 3 mb 1 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc52, v17
v_accvgpr_read_b32 v17, acc5                       // glvw 3 mb 1 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc5, v17
v_accvgpr_read_b32 v17, acc21                      // glvw 3 mb 1 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc21, v17
v_accvgpr_read_b32 v17, acc37                      // glvw 3 mb 1 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc37, v17
v_accvgpr_read_b32 v17, acc53                      // glvw 3 mb 1 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc53, v17
v_accvgpr_read_b32 v17, acc6                       // glvw 3 mb 1 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc6, v17
v_accvgpr_read_b32 v17, acc22                      // glvw 3 mb 1 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc22, v17
v_accvgpr_read_b32 v17, acc38                      // glvw 3 mb 1 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc38, v17
v_accvgpr_read_b32 v17, acc54                      // glvw 3 mb 1 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc54, v17
v_accvgpr_read_b32 v17, acc7                       // glvw 3 mb 1 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc7, v17
v_accvgpr_read_b32 v17, acc23                      // glvw 3 mb 1 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc23, v17
v_accvgpr_read_b32 v17, acc39                      // glvw 3 mb 1 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc39, v17
v_accvgpr_read_b32 v17, acc55                      // glvw 3 mb 1 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc55, v17
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=3 mb=2 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW3_BM2_VW0:  /// r3 mb2 vw0
s_mov_b32 s8, 8
v_cmpx_eq_u32 s[8:9], v16, s8                      // is thread in edge glvw region
v_and_b32 v10, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v10, 2, v10                          // permute register between threads
v_accvgpr_read_b32 v17, acc8                       // glvw 3 mb 2 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc8, v17
v_accvgpr_read_b32 v17, acc24                      // glvw 3 mb 2 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc24, v17
v_accvgpr_read_b32 v17, acc40                      // glvw 3 mb 2 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc40, v17
v_accvgpr_read_b32 v17, acc56                      // glvw 3 mb 2 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc56, v17
v_accvgpr_read_b32 v17, acc9                       // glvw 3 mb 2 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc9, v17
v_accvgpr_read_b32 v17, acc25                      // glvw 3 mb 2 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc25, v17
v_accvgpr_read_b32 v17, acc41                      // glvw 3 mb 2 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc41, v17
v_accvgpr_read_b32 v17, acc57                      // glvw 3 mb 2 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc57, v17
v_accvgpr_read_b32 v17, acc10                      // glvw 3 mb 2 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc10, v17
v_accvgpr_read_b32 v17, acc26                      // glvw 3 mb 2 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc26, v17
v_accvgpr_read_b32 v17, acc42                      // glvw 3 mb 2 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc42, v17
v_accvgpr_read_b32 v17, acc58                      // glvw 3 mb 2 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc58, v17
v_accvgpr_read_b32 v17, acc11                      // glvw 3 mb 2 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc11, v17
v_accvgpr_read_b32 v17, acc27                      // glvw 3 mb 2 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc27, v17
v_accvgpr_read_b32 v17, acc43                      // glvw 3 mb 2 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc43, v17
v_accvgpr_read_b32 v17, acc59                      // glvw 3 mb 2 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc59, v17
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=3 mb=3 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW3_BM3_VW0:  /// r3 mb3 vw0
s_mov_b32 s8, 12
v_cmpx_eq_u32 s[8:9], v16, s8                      // is thread in edge glvw region
v_and_b32 v10, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v10, 2, v10                          // permute register between threads
v_accvgpr_read_b32 v17, acc12                      // glvw 3 mb 3 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc12, v17
v_accvgpr_read_b32 v17, acc28                      // glvw 3 mb 3 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc28, v17
v_accvgpr_read_b32 v17, acc44                      // glvw 3 mb 3 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc44, v17
v_accvgpr_read_b32 v17, acc60                      // glvw 3 mb 3 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc60, v17
v_accvgpr_read_b32 v17, acc13                      // glvw 3 mb 3 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc13, v17
v_accvgpr_read_b32 v17, acc29                      // glvw 3 mb 3 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc29, v17
v_accvgpr_read_b32 v17, acc45                      // glvw 3 mb 3 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc45, v17
v_accvgpr_read_b32 v17, acc61                      // glvw 3 mb 3 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc61, v17
v_accvgpr_read_b32 v17, acc14                      // glvw 3 mb 3 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc14, v17
v_accvgpr_read_b32 v17, acc30                      // glvw 3 mb 3 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc30, v17
v_accvgpr_read_b32 v17, acc46                      // glvw 3 mb 3 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc46, v17
v_accvgpr_read_b32 v17, acc62                      // glvw 3 mb 3 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc62, v17
v_accvgpr_read_b32 v17, acc15                      // glvw 3 mb 3 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc15, v17
v_accvgpr_read_b32 v17, acc31                      // glvw 3 mb 3 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc31, v17
v_accvgpr_read_b32 v17, acc47                      // glvw 3 mb 3 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc47, v17
v_accvgpr_read_b32 v17, acc63                      // glvw 3 mb 3 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:20            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc63, v17
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=4 mb=0 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW4_BM0_VW0:  /// r4 mb0 vw0
s_mov_b32 s8, 0
v_cmpx_eq_u32 s[8:9], v16, s8                      // is thread in edge glvw region
v_and_b32 v10, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v10, 2, v10                          // permute register between threads
v_accvgpr_read_b32 v17, acc0                       // glvw 4 mb 0 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc0, v17
v_accvgpr_read_b32 v17, acc16                      // glvw 4 mb 0 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc16, v17
v_accvgpr_read_b32 v17, acc32                      // glvw 4 mb 0 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc32, v17
v_accvgpr_read_b32 v17, acc48                      // glvw 4 mb 0 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc48, v17
v_accvgpr_read_b32 v17, acc1                       // glvw 4 mb 0 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc1, v17
v_accvgpr_read_b32 v17, acc17                      // glvw 4 mb 0 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc17, v17
v_accvgpr_read_b32 v17, acc33                      // glvw 4 mb 0 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc33, v17
v_accvgpr_read_b32 v17, acc49                      // glvw 4 mb 0 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc49, v17
v_accvgpr_read_b32 v17, acc2                       // glvw 4 mb 0 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc2, v17
v_accvgpr_read_b32 v17, acc18                      // glvw 4 mb 0 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc18, v17
v_accvgpr_read_b32 v17, acc34                      // glvw 4 mb 0 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc34, v17
v_accvgpr_read_b32 v17, acc50                      // glvw 4 mb 0 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc50, v17
v_accvgpr_read_b32 v17, acc3                       // glvw 4 mb 0 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc3, v17
v_accvgpr_read_b32 v17, acc19                      // glvw 4 mb 0 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc19, v17
v_accvgpr_read_b32 v17, acc35                      // glvw 4 mb 0 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc35, v17
v_accvgpr_read_b32 v17, acc51                      // glvw 4 mb 0 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc51, v17
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=4 mb=1 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW4_BM1_VW0:  /// r4 mb1 vw0
s_mov_b32 s8, 4
v_cmpx_eq_u32 s[8:9], v16, s8                      // is thread in edge glvw region
v_and_b32 v10, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v10, 2, v10                          // permute register between threads
v_accvgpr_read_b32 v17, acc4                       // glvw 4 mb 1 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc4, v17
v_accvgpr_read_b32 v17, acc20                      // glvw 4 mb 1 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc20, v17
v_accvgpr_read_b32 v17, acc36                      // glvw 4 mb 1 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc36, v17
v_accvgpr_read_b32 v17, acc52                      // glvw 4 mb 1 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc52, v17
v_accvgpr_read_b32 v17, acc5                       // glvw 4 mb 1 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc5, v17
v_accvgpr_read_b32 v17, acc21                      // glvw 4 mb 1 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc21, v17
v_accvgpr_read_b32 v17, acc37                      // glvw 4 mb 1 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc37, v17
v_accvgpr_read_b32 v17, acc53                      // glvw 4 mb 1 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc53, v17
v_accvgpr_read_b32 v17, acc6                       // glvw 4 mb 1 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc6, v17
v_accvgpr_read_b32 v17, acc22                      // glvw 4 mb 1 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc22, v17
v_accvgpr_read_b32 v17, acc38                      // glvw 4 mb 1 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc38, v17
v_accvgpr_read_b32 v17, acc54                      // glvw 4 mb 1 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc54, v17
v_accvgpr_read_b32 v17, acc7                       // glvw 4 mb 1 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc7, v17
v_accvgpr_read_b32 v17, acc23                      // glvw 4 mb 1 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc23, v17
v_accvgpr_read_b32 v17, acc39                      // glvw 4 mb 1 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc39, v17
v_accvgpr_read_b32 v17, acc55                      // glvw 4 mb 1 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc55, v17
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=4 mb=2 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW4_BM2_VW0:  /// r4 mb2 vw0
s_mov_b32 s8, 8
v_cmpx_eq_u32 s[8:9], v16, s8                      // is thread in edge glvw region
v_and_b32 v10, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v10, 2, v10                          // permute register between threads
v_accvgpr_read_b32 v17, acc8                       // glvw 4 mb 2 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc8, v17
v_accvgpr_read_b32 v17, acc24                      // glvw 4 mb 2 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc24, v17
v_accvgpr_read_b32 v17, acc40                      // glvw 4 mb 2 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc40, v17
v_accvgpr_read_b32 v17, acc56                      // glvw 4 mb 2 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc56, v17
v_accvgpr_read_b32 v17, acc9                       // glvw 4 mb 2 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc9, v17
v_accvgpr_read_b32 v17, acc25                      // glvw 4 mb 2 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc25, v17
v_accvgpr_read_b32 v17, acc41                      // glvw 4 mb 2 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc41, v17
v_accvgpr_read_b32 v17, acc57                      // glvw 4 mb 2 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc57, v17
v_accvgpr_read_b32 v17, acc10                      // glvw 4 mb 2 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc10, v17
v_accvgpr_read_b32 v17, acc26                      // glvw 4 mb 2 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc26, v17
v_accvgpr_read_b32 v17, acc42                      // glvw 4 mb 2 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc42, v17
v_accvgpr_read_b32 v17, acc58                      // glvw 4 mb 2 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc58, v17
v_accvgpr_read_b32 v17, acc11                      // glvw 4 mb 2 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc11, v17
v_accvgpr_read_b32 v17, acc27                      // glvw 4 mb 2 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc27, v17
v_accvgpr_read_b32 v17, acc43                      // glvw 4 mb 2 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc43, v17
v_accvgpr_read_b32 v17, acc59                      // glvw 4 mb 2 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc59, v17
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=4 mb=3 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW4_BM3_VW0:  /// r4 mb3 vw0
s_mov_b32 s8, 12
v_cmpx_eq_u32 s[8:9], v16, s8                      // is thread in edge glvw region
v_and_b32 v10, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v10, 2, v10                          // permute register between threads
v_accvgpr_read_b32 v17, acc12                      // glvw 4 mb 3 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc12, v17
v_accvgpr_read_b32 v17, acc28                      // glvw 4 mb 3 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc28, v17
v_accvgpr_read_b32 v17, acc44                      // glvw 4 mb 3 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc44, v17
v_accvgpr_read_b32 v17, acc60                      // glvw 4 mb 3 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc60, v17
v_accvgpr_read_b32 v17, acc13                      // glvw 4 mb 3 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc13, v17
v_accvgpr_read_b32 v17, acc29                      // glvw 4 mb 3 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc29, v17
v_accvgpr_read_b32 v17, acc45                      // glvw 4 mb 3 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc45, v17
v_accvgpr_read_b32 v17, acc61                      // glvw 4 mb 3 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc61, v17
v_accvgpr_read_b32 v17, acc14                      // glvw 4 mb 3 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc14, v17
v_accvgpr_read_b32 v17, acc30                      // glvw 4 mb 3 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc30, v17
v_accvgpr_read_b32 v17, acc46                      // glvw 4 mb 3 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc46, v17
v_accvgpr_read_b32 v17, acc62                      // glvw 4 mb 3 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc62, v17
v_accvgpr_read_b32 v17, acc15                      // glvw 4 mb 3 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc15, v17
v_accvgpr_read_b32 v17, acc31                      // glvw 4 mb 3 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc31, v17
v_accvgpr_read_b32 v17, acc47                      // glvw 4 mb 3 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc47, v17
v_accvgpr_read_b32 v17, acc63                      // glvw 4 mb 3 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:16            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc63, v17
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=5 mb=0 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW5_BM0_VW0:  /// r5 mb0 vw0
s_mov_b32 s8, 0
v_cmpx_eq_u32 s[8:9], v16, s8                      // is thread in edge glvw region
v_and_b32 v10, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v10, 2, v10                          // permute register between threads
v_accvgpr_read_b32 v17, acc0                       // glvw 5 mb 0 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc0, v17
v_accvgpr_read_b32 v17, acc16                      // glvw 5 mb 0 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc16, v17
v_accvgpr_read_b32 v17, acc32                      // glvw 5 mb 0 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc32, v17
v_accvgpr_read_b32 v17, acc48                      // glvw 5 mb 0 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc48, v17
v_accvgpr_read_b32 v17, acc1                       // glvw 5 mb 0 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc1, v17
v_accvgpr_read_b32 v17, acc17                      // glvw 5 mb 0 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc17, v17
v_accvgpr_read_b32 v17, acc33                      // glvw 5 mb 0 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc33, v17
v_accvgpr_read_b32 v17, acc49                      // glvw 5 mb 0 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc49, v17
v_accvgpr_read_b32 v17, acc2                       // glvw 5 mb 0 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc2, v17
v_accvgpr_read_b32 v17, acc18                      // glvw 5 mb 0 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc18, v17
v_accvgpr_read_b32 v17, acc34                      // glvw 5 mb 0 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc34, v17
v_accvgpr_read_b32 v17, acc50                      // glvw 5 mb 0 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc50, v17
v_accvgpr_read_b32 v17, acc3                       // glvw 5 mb 0 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc3, v17
v_accvgpr_read_b32 v17, acc19                      // glvw 5 mb 0 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc19, v17
v_accvgpr_read_b32 v17, acc35                      // glvw 5 mb 0 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc35, v17
v_accvgpr_read_b32 v17, acc51                      // glvw 5 mb 0 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc51, v17
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=5 mb=1 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW5_BM1_VW0:  /// r5 mb1 vw0
s_mov_b32 s8, 4
v_cmpx_eq_u32 s[8:9], v16, s8                      // is thread in edge glvw region
v_and_b32 v10, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v10, 2, v10                          // permute register between threads
v_accvgpr_read_b32 v17, acc4                       // glvw 5 mb 1 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc4, v17
v_accvgpr_read_b32 v17, acc20                      // glvw 5 mb 1 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc20, v17
v_accvgpr_read_b32 v17, acc36                      // glvw 5 mb 1 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc36, v17
v_accvgpr_read_b32 v17, acc52                      // glvw 5 mb 1 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc52, v17
v_accvgpr_read_b32 v17, acc5                       // glvw 5 mb 1 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc5, v17
v_accvgpr_read_b32 v17, acc21                      // glvw 5 mb 1 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc21, v17
v_accvgpr_read_b32 v17, acc37                      // glvw 5 mb 1 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc37, v17
v_accvgpr_read_b32 v17, acc53                      // glvw 5 mb 1 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc53, v17
v_accvgpr_read_b32 v17, acc6                       // glvw 5 mb 1 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc6, v17
v_accvgpr_read_b32 v17, acc22                      // glvw 5 mb 1 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc22, v17
v_accvgpr_read_b32 v17, acc38                      // glvw 5 mb 1 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc38, v17
v_accvgpr_read_b32 v17, acc54                      // glvw 5 mb 1 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc54, v17
v_accvgpr_read_b32 v17, acc7                       // glvw 5 mb 1 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc7, v17
v_accvgpr_read_b32 v17, acc23                      // glvw 5 mb 1 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc23, v17
v_accvgpr_read_b32 v17, acc39                      // glvw 5 mb 1 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc39, v17
v_accvgpr_read_b32 v17, acc55                      // glvw 5 mb 1 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc55, v17
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=5 mb=2 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW5_BM2_VW0:  /// r5 mb2 vw0
s_mov_b32 s8, 8
v_cmpx_eq_u32 s[8:9], v16, s8                      // is thread in edge glvw region
v_and_b32 v10, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v10, 2, v10                          // permute register between threads
v_accvgpr_read_b32 v17, acc8                       // glvw 5 mb 2 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc8, v17
v_accvgpr_read_b32 v17, acc24                      // glvw 5 mb 2 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc24, v17
v_accvgpr_read_b32 v17, acc40                      // glvw 5 mb 2 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc40, v17
v_accvgpr_read_b32 v17, acc56                      // glvw 5 mb 2 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc56, v17
v_accvgpr_read_b32 v17, acc9                       // glvw 5 mb 2 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc9, v17
v_accvgpr_read_b32 v17, acc25                      // glvw 5 mb 2 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc25, v17
v_accvgpr_read_b32 v17, acc41                      // glvw 5 mb 2 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc41, v17
v_accvgpr_read_b32 v17, acc57                      // glvw 5 mb 2 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc57, v17
v_accvgpr_read_b32 v17, acc10                      // glvw 5 mb 2 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc10, v17
v_accvgpr_read_b32 v17, acc26                      // glvw 5 mb 2 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc26, v17
v_accvgpr_read_b32 v17, acc42                      // glvw 5 mb 2 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc42, v17
v_accvgpr_read_b32 v17, acc58                      // glvw 5 mb 2 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc58, v17
v_accvgpr_read_b32 v17, acc11                      // glvw 5 mb 2 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc11, v17
v_accvgpr_read_b32 v17, acc27                      // glvw 5 mb 2 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc27, v17
v_accvgpr_read_b32 v17, acc43                      // glvw 5 mb 2 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc43, v17
v_accvgpr_read_b32 v17, acc59                      // glvw 5 mb 2 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc59, v17
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=5 mb=3 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW5_BM3_VW0:  /// r5 mb3 vw0
s_mov_b32 s8, 12
v_cmpx_eq_u32 s[8:9], v16, s8                      // is thread in edge glvw region
v_and_b32 v10, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v10, 2, v10                          // permute register between threads
v_accvgpr_read_b32 v17, acc12                      // glvw 5 mb 3 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc12, v17
v_accvgpr_read_b32 v17, acc28                      // glvw 5 mb 3 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc28, v17
v_accvgpr_read_b32 v17, acc44                      // glvw 5 mb 3 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc44, v17
v_accvgpr_read_b32 v17, acc60                      // glvw 5 mb 3 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc60, v17
v_accvgpr_read_b32 v17, acc13                      // glvw 5 mb 3 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc13, v17
v_accvgpr_read_b32 v17, acc29                      // glvw 5 mb 3 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc29, v17
v_accvgpr_read_b32 v17, acc45                      // glvw 5 mb 3 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc45, v17
v_accvgpr_read_b32 v17, acc61                      // glvw 5 mb 3 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc61, v17
v_accvgpr_read_b32 v17, acc14                      // glvw 5 mb 3 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc14, v17
v_accvgpr_read_b32 v17, acc30                      // glvw 5 mb 3 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc30, v17
v_accvgpr_read_b32 v17, acc46                      // glvw 5 mb 3 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc46, v17
v_accvgpr_read_b32 v17, acc62                      // glvw 5 mb 3 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc62, v17
v_accvgpr_read_b32 v17, acc15                      // glvw 5 mb 3 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc15, v17
v_accvgpr_read_b32 v17, acc31                      // glvw 5 mb 3 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc31, v17
v_accvgpr_read_b32 v17, acc47                      // glvw 5 mb 3 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc47, v17
v_accvgpr_read_b32 v17, acc63                      // glvw 5 mb 3 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc63, v17
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=6 mb=0 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW6_BM0_VW0:  /// r6 mb0 vw0
s_mov_b32 s8, 0
v_cmpx_eq_u32 s[8:9], v16, s8                      // is thread in edge glvw region
v_and_b32 v10, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v10, 2, v10                          // permute register between threads
v_accvgpr_read_b32 v17, acc0                       // glvw 6 mb 0 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc0, v17
v_accvgpr_read_b32 v17, acc16                      // glvw 6 mb 0 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc16, v17
v_accvgpr_read_b32 v17, acc32                      // glvw 6 mb 0 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc32, v17
v_accvgpr_read_b32 v17, acc48                      // glvw 6 mb 0 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc48, v17
v_accvgpr_read_b32 v17, acc1                       // glvw 6 mb 0 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc1, v17
v_accvgpr_read_b32 v17, acc17                      // glvw 6 mb 0 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc17, v17
v_accvgpr_read_b32 v17, acc33                      // glvw 6 mb 0 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc33, v17
v_accvgpr_read_b32 v17, acc49                      // glvw 6 mb 0 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc49, v17
v_accvgpr_read_b32 v17, acc2                       // glvw 6 mb 0 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc2, v17
v_accvgpr_read_b32 v17, acc18                      // glvw 6 mb 0 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc18, v17
v_accvgpr_read_b32 v17, acc34                      // glvw 6 mb 0 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc34, v17
v_accvgpr_read_b32 v17, acc50                      // glvw 6 mb 0 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc50, v17
v_accvgpr_read_b32 v17, acc3                       // glvw 6 mb 0 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc3, v17
v_accvgpr_read_b32 v17, acc19                      // glvw 6 mb 0 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc19, v17
v_accvgpr_read_b32 v17, acc35                      // glvw 6 mb 0 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc35, v17
v_accvgpr_read_b32 v17, acc51                      // glvw 6 mb 0 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc51, v17
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=6 mb=1 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW6_BM1_VW0:  /// r6 mb1 vw0
s_mov_b32 s8, 4
v_cmpx_eq_u32 s[8:9], v16, s8                      // is thread in edge glvw region
v_and_b32 v10, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v10, 2, v10                          // permute register between threads
v_accvgpr_read_b32 v17, acc4                       // glvw 6 mb 1 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc4, v17
v_accvgpr_read_b32 v17, acc20                      // glvw 6 mb 1 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc20, v17
v_accvgpr_read_b32 v17, acc36                      // glvw 6 mb 1 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc36, v17
v_accvgpr_read_b32 v17, acc52                      // glvw 6 mb 1 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc52, v17
v_accvgpr_read_b32 v17, acc5                       // glvw 6 mb 1 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc5, v17
v_accvgpr_read_b32 v17, acc21                      // glvw 6 mb 1 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc21, v17
v_accvgpr_read_b32 v17, acc37                      // glvw 6 mb 1 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc37, v17
v_accvgpr_read_b32 v17, acc53                      // glvw 6 mb 1 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc53, v17
v_accvgpr_read_b32 v17, acc6                       // glvw 6 mb 1 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc6, v17
v_accvgpr_read_b32 v17, acc22                      // glvw 6 mb 1 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc22, v17
v_accvgpr_read_b32 v17, acc38                      // glvw 6 mb 1 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc38, v17
v_accvgpr_read_b32 v17, acc54                      // glvw 6 mb 1 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc54, v17
v_accvgpr_read_b32 v17, acc7                       // glvw 6 mb 1 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc7, v17
v_accvgpr_read_b32 v17, acc23                      // glvw 6 mb 1 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc23, v17
v_accvgpr_read_b32 v17, acc39                      // glvw 6 mb 1 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc39, v17
v_accvgpr_read_b32 v17, acc55                      // glvw 6 mb 1 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc55, v17
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=6 mb=2 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW6_BM2_VW0:  /// r6 mb2 vw0
s_mov_b32 s8, 8
v_cmpx_eq_u32 s[8:9], v16, s8                      // is thread in edge glvw region
v_and_b32 v10, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v10, 2, v10                          // permute register between threads
v_accvgpr_read_b32 v17, acc8                       // glvw 6 mb 2 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc8, v17
v_accvgpr_read_b32 v17, acc24                      // glvw 6 mb 2 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc24, v17
v_accvgpr_read_b32 v17, acc40                      // glvw 6 mb 2 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc40, v17
v_accvgpr_read_b32 v17, acc56                      // glvw 6 mb 2 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc56, v17
v_accvgpr_read_b32 v17, acc9                       // glvw 6 mb 2 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc9, v17
v_accvgpr_read_b32 v17, acc25                      // glvw 6 mb 2 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc25, v17
v_accvgpr_read_b32 v17, acc41                      // glvw 6 mb 2 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc41, v17
v_accvgpr_read_b32 v17, acc57                      // glvw 6 mb 2 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc57, v17
v_accvgpr_read_b32 v17, acc10                      // glvw 6 mb 2 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc10, v17
v_accvgpr_read_b32 v17, acc26                      // glvw 6 mb 2 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc26, v17
v_accvgpr_read_b32 v17, acc42                      // glvw 6 mb 2 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc42, v17
v_accvgpr_read_b32 v17, acc58                      // glvw 6 mb 2 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc58, v17
v_accvgpr_read_b32 v17, acc11                      // glvw 6 mb 2 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc11, v17
v_accvgpr_read_b32 v17, acc27                      // glvw 6 mb 2 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc27, v17
v_accvgpr_read_b32 v17, acc43                      // glvw 6 mb 2 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc43, v17
v_accvgpr_read_b32 v17, acc59                      // glvw 6 mb 2 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc59, v17
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=6 mb=3 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW6_BM3_VW0:  /// r6 mb3 vw0
s_mov_b32 s8, 12
v_cmpx_eq_u32 s[8:9], v16, s8                      // is thread in edge glvw region
v_and_b32 v10, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v10, 2, v10                          // permute register between threads
v_accvgpr_read_b32 v17, acc12                      // glvw 6 mb 3 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc12, v17
v_accvgpr_read_b32 v17, acc28                      // glvw 6 mb 3 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc28, v17
v_accvgpr_read_b32 v17, acc44                      // glvw 6 mb 3 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc44, v17
v_accvgpr_read_b32 v17, acc60                      // glvw 6 mb 3 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc60, v17
v_accvgpr_read_b32 v17, acc13                      // glvw 6 mb 3 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc13, v17
v_accvgpr_read_b32 v17, acc29                      // glvw 6 mb 3 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc29, v17
v_accvgpr_read_b32 v17, acc45                      // glvw 6 mb 3 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc45, v17
v_accvgpr_read_b32 v17, acc61                      // glvw 6 mb 3 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc61, v17
v_accvgpr_read_b32 v17, acc14                      // glvw 6 mb 3 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc14, v17
v_accvgpr_read_b32 v17, acc30                      // glvw 6 mb 3 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc30, v17
v_accvgpr_read_b32 v17, acc46                      // glvw 6 mb 3 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc46, v17
v_accvgpr_read_b32 v17, acc62                      // glvw 6 mb 3 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc62, v17
v_accvgpr_read_b32 v17, acc15                      // glvw 6 mb 3 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc15, v17
v_accvgpr_read_b32 v17, acc31                      // glvw 6 mb 3 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc31, v17
v_accvgpr_read_b32 v17, acc47                      // glvw 6 mb 3 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc47, v17
v_accvgpr_read_b32 v17, acc63                      // glvw 6 mb 3 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc63, v17
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=7 mb=0 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW7_BM0_VW0:  /// r7 mb0 vw0
s_mov_b32 s8, 0
v_cmpx_eq_u32 s[8:9], v16, s8                      // is thread in edge glvw region
v_and_b32 v10, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v10, 2, v10                          // permute register between threads
v_accvgpr_read_b32 v17, acc0                       // glvw 7 mb 0 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc0, v17
v_accvgpr_read_b32 v17, acc16                      // glvw 7 mb 0 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc16, v17
v_accvgpr_read_b32 v17, acc32                      // glvw 7 mb 0 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc32, v17
v_accvgpr_read_b32 v17, acc48                      // glvw 7 mb 0 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc48, v17
v_accvgpr_read_b32 v17, acc1                       // glvw 7 mb 0 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc1, v17
v_accvgpr_read_b32 v17, acc17                      // glvw 7 mb 0 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc17, v17
v_accvgpr_read_b32 v17, acc33                      // glvw 7 mb 0 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc33, v17
v_accvgpr_read_b32 v17, acc49                      // glvw 7 mb 0 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc49, v17
v_accvgpr_read_b32 v17, acc2                       // glvw 7 mb 0 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc2, v17
v_accvgpr_read_b32 v17, acc18                      // glvw 7 mb 0 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc18, v17
v_accvgpr_read_b32 v17, acc34                      // glvw 7 mb 0 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc34, v17
v_accvgpr_read_b32 v17, acc50                      // glvw 7 mb 0 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc50, v17
v_accvgpr_read_b32 v17, acc3                       // glvw 7 mb 0 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc3, v17
v_accvgpr_read_b32 v17, acc19                      // glvw 7 mb 0 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc19, v17
v_accvgpr_read_b32 v17, acc35                      // glvw 7 mb 0 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc35, v17
v_accvgpr_read_b32 v17, acc51                      // glvw 7 mb 0 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc51, v17
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=7 mb=1 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW7_BM1_VW0:  /// r7 mb1 vw0
s_mov_b32 s8, 4
v_cmpx_eq_u32 s[8:9], v16, s8                      // is thread in edge glvw region
v_and_b32 v10, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v10, 2, v10                          // permute register between threads
v_accvgpr_read_b32 v17, acc4                       // glvw 7 mb 1 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc4, v17
v_accvgpr_read_b32 v17, acc20                      // glvw 7 mb 1 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc20, v17
v_accvgpr_read_b32 v17, acc36                      // glvw 7 mb 1 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc36, v17
v_accvgpr_read_b32 v17, acc52                      // glvw 7 mb 1 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc52, v17
v_accvgpr_read_b32 v17, acc5                       // glvw 7 mb 1 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc5, v17
v_accvgpr_read_b32 v17, acc21                      // glvw 7 mb 1 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc21, v17
v_accvgpr_read_b32 v17, acc37                      // glvw 7 mb 1 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc37, v17
v_accvgpr_read_b32 v17, acc53                      // glvw 7 mb 1 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc53, v17
v_accvgpr_read_b32 v17, acc6                       // glvw 7 mb 1 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc6, v17
v_accvgpr_read_b32 v17, acc22                      // glvw 7 mb 1 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc22, v17
v_accvgpr_read_b32 v17, acc38                      // glvw 7 mb 1 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc38, v17
v_accvgpr_read_b32 v17, acc54                      // glvw 7 mb 1 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc54, v17
v_accvgpr_read_b32 v17, acc7                       // glvw 7 mb 1 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc7, v17
v_accvgpr_read_b32 v17, acc23                      // glvw 7 mb 1 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc23, v17
v_accvgpr_read_b32 v17, acc39                      // glvw 7 mb 1 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc39, v17
v_accvgpr_read_b32 v17, acc55                      // glvw 7 mb 1 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc55, v17
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=7 mb=2 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW7_BM2_VW0:  /// r7 mb2 vw0
s_mov_b32 s8, 8
v_cmpx_eq_u32 s[8:9], v16, s8                      // is thread in edge glvw region
v_and_b32 v10, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v10, 2, v10                          // permute register between threads
v_accvgpr_read_b32 v17, acc8                       // glvw 7 mb 2 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc8, v17
v_accvgpr_read_b32 v17, acc24                      // glvw 7 mb 2 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc24, v17
v_accvgpr_read_b32 v17, acc40                      // glvw 7 mb 2 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc40, v17
v_accvgpr_read_b32 v17, acc56                      // glvw 7 mb 2 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc56, v17
v_accvgpr_read_b32 v17, acc9                       // glvw 7 mb 2 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc9, v17
v_accvgpr_read_b32 v17, acc25                      // glvw 7 mb 2 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc25, v17
v_accvgpr_read_b32 v17, acc41                      // glvw 7 mb 2 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc41, v17
v_accvgpr_read_b32 v17, acc57                      // glvw 7 mb 2 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc57, v17
v_accvgpr_read_b32 v17, acc10                      // glvw 7 mb 2 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc10, v17
v_accvgpr_read_b32 v17, acc26                      // glvw 7 mb 2 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc26, v17
v_accvgpr_read_b32 v17, acc42                      // glvw 7 mb 2 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc42, v17
v_accvgpr_read_b32 v17, acc58                      // glvw 7 mb 2 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc58, v17
v_accvgpr_read_b32 v17, acc11                      // glvw 7 mb 2 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc11, v17
v_accvgpr_read_b32 v17, acc27                      // glvw 7 mb 2 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc27, v17
v_accvgpr_read_b32 v17, acc43                      // glvw 7 mb 2 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc43, v17
v_accvgpr_read_b32 v17, acc59                      // glvw 7 mb 2 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc59, v17
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=7 mb=3 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW7_BM3_VW0:  /// r7 mb3 vw0
s_mov_b32 s8, 12
v_cmpx_eq_u32 s[8:9], v16, s8                      // is thread in edge glvw region
v_and_b32 v10, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v10, 2, v10                          // permute register between threads
v_accvgpr_read_b32 v17, acc12                      // glvw 7 mb 3 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc12, v17
v_accvgpr_read_b32 v17, acc28                      // glvw 7 mb 3 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc28, v17
v_accvgpr_read_b32 v17, acc44                      // glvw 7 mb 3 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc44, v17
v_accvgpr_read_b32 v17, acc60                      // glvw 7 mb 3 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc60, v17
v_accvgpr_read_b32 v17, acc13                      // glvw 7 mb 3 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc13, v17
v_accvgpr_read_b32 v17, acc29                      // glvw 7 mb 3 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc29, v17
v_accvgpr_read_b32 v17, acc45                      // glvw 7 mb 3 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc45, v17
v_accvgpr_read_b32 v17, acc61                      // glvw 7 mb 3 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc61, v17
v_accvgpr_read_b32 v17, acc14                      // glvw 7 mb 3 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc14, v17
v_accvgpr_read_b32 v17, acc30                      // glvw 7 mb 3 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc30, v17
v_accvgpr_read_b32 v17, acc46                      // glvw 7 mb 3 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc46, v17
v_accvgpr_read_b32 v17, acc62                      // glvw 7 mb 3 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc62, v17
v_accvgpr_read_b32 v17, acc15                      // glvw 7 mb 3 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc15, v17
v_accvgpr_read_b32 v17, acc31                      // glvw 7 mb 3 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc31, v17
v_accvgpr_read_b32 v17, acc47                      // glvw 7 mb 3 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc47, v17
v_accvgpr_read_b32 v17, acc63                      // glvw 7 mb 3 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v17, v10, v17 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc63, v17
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0

label_ShiftVectorComponents0_GLVW0:  /// end shift0

/* not-LocalSplitU: global write indices */
/* computeStoreVgprs */
v_lshrrev_b32 v14, 6, v[vgprSerial]                // 14 = Serial / 64
v_lshrrev_b32 v15, 1, v14                          // 15 = 14 / 2
v_mul_lo_u32 v15, 0x10, v15                        // wave coordination offset 1
v_and_b32 v11, 63, v[vgprSerial]                   // v11 = v[vgprSerial] % 64
v_lshrrev_b32 v11, 4, v11                          // 11 = 11 / 16
v_lshlrev_b32 v11, 2, v11                          // thread0 * continuous_output
v_add_lshl_u32 v11, v15, v11, 2                    // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v12, v11, s[sgprStrideC1J]            //  offset 1
v_mul_lo_u32 v13, v11, s[sgprStrideD1J]            //  offset 1
v_and_b32 v10, 1, v14                              // v10 = v14 % 2
v_mul_lo_u32 v10, 0x10, v10                        // wave coordination offset 0
v_and_b32 v15, 15, v[vgprSerial]                   // v15 = v[vgprSerial] % 16
v_add_lshl_u32 v10, v15, v10, 0                    // coordination 0 = vwA * (wave_id0 + tid0)
s_mul_i32 s8, 128, s[sgprWorkGroup0]               // wgp0 * MT0
v_add_u32 v10, s8, v10                             // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s8, 128, s[sgprWorkGroup1]               // wgp1 * MT1
v_add_u32 v11, s8, v11                             // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1

/* not-LocalSplitU: global write */

/******************************************/
/* Global Write Elements                  */
/******************************************/
s_waitcnt lgkmcnt(0)                               // wait for 24 bytes of kern args.
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_GSU                           // Branch to stream-k store code
s_cmp_eq_u32 s[sgprskTiles], 1                     // split == 1 ?
s_cbranch_scc1 label_GSU                           // branch if split == 1
.set sgprAddressScaleAlphaVec, UNDEF
.set sgprSrdScaleAlphaVec, UNDEF
s_and_b32 s70, 127, s[sgprSizeI]                   // s70 = s[sgprSizeI] % 128
s_add_u32 s71, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s71                // wg0 >= nwg0-1 ?
s_cselect_b32 s70, s70, 0                          // set rMT0
s_cmpk_gt_u32 s70, 0                               // rMT0 > 0
s_cbranch_scc1 label_GW_B0_E1                      // jump if edges required
s_and_b32 s70, 127, s[sgprSizeJ]                   // s70 = s[sgprSizeJ] % 128
s_add_u32 s71, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s71                // wg1 >= nwg1-1
s_cselect_b32 s70, s70, 0                          // set rMT1
s_cmpk_gt_u32 s70, 0                               // rMT1 > 0
s_cbranch_scc1 label_GW_B0_E1                      // jump if edges required
label_GW_B0_E0:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=16 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,0,1,0:vw1); (0,1,1,0:vw1); (0,2,1,0:vw1); (0,3,1,0:vw1); (0,0,2,0:vw1); (0,1,2,0:vw1); (0,2,2,0:vw1); (0,3,2,0:vw1); (0,0,3,0:vw1); (0,1,3,0:vw1); (0,2,3,0:vw1); (0,3,3,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
/* (d1,vc1,d0,vc0)=(0,1,2,0) */
/* (d1,vc1,d0,vc0)=(0,1,3,0) */
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
/* (d1,vc1,d0,vc0)=(0,2,1,0) */
/* (d1,vc1,d0,vc0)=(0,2,2,0) */
/* (d1,vc1,d0,vc0)=(0,2,3,0) */
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
/* (d1,vc1,d0,vc0)=(0,3,1,0) */
/* (d1,vc1,d0,vc0)=(0,3,2,0) */
/* (d1,vc1,d0,vc0)=(0,3,3,0) */
v_add_lshl_u32 v21, v13, v10, 0x2                  // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=10, coord0Vgpr=10
v_accvgpr_read_b32 v[vgprValuC+23], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+24], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+25], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+26], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+27], acc16          // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+28], acc20          // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+29], acc24          // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+30], acc28          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+31], acc32          // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+32], acc36          // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+33], acc40          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+34], acc44          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+35], acc48          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+36], acc52          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+37], acc56          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+38], acc60          // copy acc to vreg[15]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 0, 1, 0), (0, 1, 1, 0), (0, 2, 1, 0), (0, 3, 1, 0), (0, 0, 2, 0), (0, 1, 2, 0), (0, 2, 2, 0), (0, 3, 2, 0), (0, 0, 3, 0), (0, 1, 3, 0), (0, 2, 3, 0), (0, 3, 3, 0)] */

/* apply mask, calc new C and issue writes */
buffer_store_dword v23, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v24, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_dword v25, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_dword v26, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:384 // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v27, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v28, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_dword v29, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_dword v30, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:384 // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v31, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v32, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_dword v33, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_dword v34, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:384 // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v35, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v36, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_dword v37, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_dword v38, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:384 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #1 (d1,d0,vc1,vc0) = */
/*    (0,0,4,0:vw1); (0,1,4,0:vw1); (0,2,4,0:vw1); (0,3,4,0:vw1); (0,0,5,0:vw1); (0,1,5,0:vw1); (0,2,5,0:vw1); (0,3,5,0:vw1); (0,0,6,0:vw1); (0,1,6,0:vw1); (0,2,6,0:vw1); (0,3,6,0:vw1); (0,0,7,0:vw1); (0,1,7,0:vw1); (0,2,7,0:vw1); (0,3,7,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,4,0,0) */
/* (d1,vc1,d0,vc0)=(0,4,1,0) */
/* (d1,vc1,d0,vc0)=(0,4,2,0) */
/* (d1,vc1,d0,vc0)=(0,4,3,0) */
/* (d1,vc1,d0,vc0)=(0,5,0,0) */
/* (d1,vc1,d0,vc0)=(0,5,1,0) */
/* (d1,vc1,d0,vc0)=(0,5,2,0) */
/* (d1,vc1,d0,vc0)=(0,5,3,0) */
/* (d1,vc1,d0,vc0)=(0,6,0,0) */
/* (d1,vc1,d0,vc0)=(0,6,1,0) */
/* (d1,vc1,d0,vc0)=(0,6,2,0) */
/* (d1,vc1,d0,vc0)=(0,6,3,0) */
/* (d1,vc1,d0,vc0)=(0,7,0,0) */
/* (d1,vc1,d0,vc0)=(0,7,1,0) */
/* (d1,vc1,d0,vc0)=(0,7,2,0) */
/* (d1,vc1,d0,vc0)=(0,7,3,0) */
v_accvgpr_read_b32 v[vgprValuC+23], acc1           // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+24], acc5           // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+25], acc9           // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+26], acc13          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+27], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+28], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+29], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+30], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+31], acc33          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+32], acc37          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+33], acc41          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+34], acc45          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+35], acc49          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+36], acc53          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+37], acc57          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+38], acc61          // copy acc to vreg[31]

/* rC *= alpha batchElements=[(0, 0, 4, 0), (0, 1, 4, 0), (0, 2, 4, 0), (0, 3, 4, 0), (0, 0, 5, 0), (0, 1, 5, 0), (0, 2, 5, 0), (0, 3, 5, 0), (0, 0, 6, 0), (0, 1, 6, 0), (0, 2, 6, 0), (0, 3, 6, 0), (0, 0, 7, 0), (0, 1, 7, 0), (0, 2, 7, 0), (0, 3, 7, 0)] */

/* apply mask, calc new C and issue writes */
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v23, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v24, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_dword v25, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_dword v26, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:384 // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v27, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v28, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_dword v29, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_dword v30, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:384 // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v31, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v32, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_dword v33, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_dword v34, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:384 // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v35, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v36, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_dword v37, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_dword v38, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:384 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #2 (d1,d0,vc1,vc0) = */
/*    (0,0,8,0:vw1); (0,1,8,0:vw1); (0,2,8,0:vw1); (0,3,8,0:vw1); (0,0,9,0:vw1); (0,1,9,0:vw1); (0,2,9,0:vw1); (0,3,9,0:vw1); (0,0,10,0:vw1); (0,1,10,0:vw1); (0,2,10,0:vw1); (0,3,10,0:vw1); (0,0,11,0:vw1); (0,1,11,0:vw1); (0,2,11,0:vw1); (0,3,11,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,8,0,0) */
/* (d1,vc1,d0,vc0)=(0,8,1,0) */
/* (d1,vc1,d0,vc0)=(0,8,2,0) */
/* (d1,vc1,d0,vc0)=(0,8,3,0) */
/* (d1,vc1,d0,vc0)=(0,9,0,0) */
/* (d1,vc1,d0,vc0)=(0,9,1,0) */
/* (d1,vc1,d0,vc0)=(0,9,2,0) */
/* (d1,vc1,d0,vc0)=(0,9,3,0) */
/* (d1,vc1,d0,vc0)=(0,10,0,0) */
/* (d1,vc1,d0,vc0)=(0,10,1,0) */
/* (d1,vc1,d0,vc0)=(0,10,2,0) */
/* (d1,vc1,d0,vc0)=(0,10,3,0) */
/* (d1,vc1,d0,vc0)=(0,11,0,0) */
/* (d1,vc1,d0,vc0)=(0,11,1,0) */
/* (d1,vc1,d0,vc0)=(0,11,2,0) */
/* (d1,vc1,d0,vc0)=(0,11,3,0) */
v_accvgpr_read_b32 v[vgprValuC+23], acc2           // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+24], acc6           // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+25], acc10          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+26], acc14          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+27], acc18          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+28], acc22          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+29], acc26          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+30], acc30          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+31], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+32], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+33], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+34], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+35], acc50          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+36], acc54          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+37], acc58          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+38], acc62          // copy acc to vreg[47]

/* rC *= alpha batchElements=[(0, 0, 8, 0), (0, 1, 8, 0), (0, 2, 8, 0), (0, 3, 8, 0), (0, 0, 9, 0), (0, 1, 9, 0), (0, 2, 9, 0), (0, 3, 9, 0), (0, 0, 10, 0), (0, 1, 10, 0), (0, 2, 10, 0), (0, 3, 10, 0), (0, 0, 11, 0), (0, 1, 11, 0), (0, 2, 11, 0), (0, 3, 11, 0)] */

/* apply mask, calc new C and issue writes */
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v23, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v24, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_dword v25, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_dword v26, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:384 // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v27, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v28, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_dword v29, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_dword v30, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:384 // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v31, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v32, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_dword v33, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_dword v34, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:384 // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v35, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v36, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_dword v37, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_dword v38, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:384 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #3 (d1,d0,vc1,vc0) = */
/*    (0,0,12,0:vw1); (0,1,12,0:vw1); (0,2,12,0:vw1); (0,3,12,0:vw1); (0,0,13,0:vw1); (0,1,13,0:vw1); (0,2,13,0:vw1); (0,3,13,0:vw1); (0,0,14,0:vw1); (0,1,14,0:vw1); (0,2,14,0:vw1); (0,3,14,0:vw1); (0,0,15,0:vw1); (0,1,15,0:vw1); (0,2,15,0:vw1); (0,3,15,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,12,0,0) */
/* (d1,vc1,d0,vc0)=(0,12,1,0) */
/* (d1,vc1,d0,vc0)=(0,12,2,0) */
/* (d1,vc1,d0,vc0)=(0,12,3,0) */
/* (d1,vc1,d0,vc0)=(0,13,0,0) */
/* (d1,vc1,d0,vc0)=(0,13,1,0) */
/* (d1,vc1,d0,vc0)=(0,13,2,0) */
/* (d1,vc1,d0,vc0)=(0,13,3,0) */
/* (d1,vc1,d0,vc0)=(0,14,0,0) */
/* (d1,vc1,d0,vc0)=(0,14,1,0) */
/* (d1,vc1,d0,vc0)=(0,14,2,0) */
/* (d1,vc1,d0,vc0)=(0,14,3,0) */
/* (d1,vc1,d0,vc0)=(0,15,0,0) */
/* (d1,vc1,d0,vc0)=(0,15,1,0) */
/* (d1,vc1,d0,vc0)=(0,15,2,0) */
/* (d1,vc1,d0,vc0)=(0,15,3,0) */
v_accvgpr_read_b32 v[vgprValuC+23], acc3           // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+24], acc7           // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+25], acc11          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+26], acc15          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+27], acc19          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+28], acc23          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+29], acc27          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+30], acc31          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+31], acc35          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+32], acc39          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+33], acc43          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+34], acc47          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+35], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+36], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+37], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+38], acc63          // copy acc to vreg[63]

/* rC *= alpha batchElements=[(0, 0, 12, 0), (0, 1, 12, 0), (0, 2, 12, 0), (0, 3, 12, 0), (0, 0, 13, 0), (0, 1, 13, 0), (0, 2, 13, 0), (0, 3, 13, 0), (0, 0, 14, 0), (0, 1, 14, 0), (0, 2, 14, 0), (0, 3, 14, 0), (0, 0, 15, 0), (0, 1, 15, 0), (0, 2, 15, 0), (0, 3, 15, 0)] */

/* apply mask, calc new C and issue writes */
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v23, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v24, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_dword v25, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_dword v26, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:384 // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v27, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v28, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_dword v29, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_dword v30, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:384 // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v31, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v32, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_dword v33, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_dword v34, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:384 // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v35, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v36, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_dword v37, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:256 // store D
buffer_store_dword v38, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:384 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_B0_E1:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=16 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,0,1,0:vw1); (0,1,1,0:vw1); (0,2,1,0:vw1); (0,3,1,0:vw1); (0,0,2,0:vw1); (0,1,2,0:vw1); (0,2,2,0:vw1); (0,3,2,0:vw1); (0,0,3,0:vw1); (0,1,3,0:vw1); (0,2,3,0:vw1); (0,3,3,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v16, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v37, v13, v10, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v37, v16, v37, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v38, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v38, v16, v38, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v39, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v39, v16, v39, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v40, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v40, v16, v40, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v41, v13, v10, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v41, v16, v41, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v42, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v42, v16, v42, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v43, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v43, v16, v43, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v44, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v44, v16, v44, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v45, v13, v10, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v45, v16, v45, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v46, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v46, v16, v46, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v47, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v47, v16, v47, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v48, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v48, v16, v48, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v49, v13, v10, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v49, v16, v49, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v50, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v50, v16, v50, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v51, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v51, v16, v51, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v52, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v52, v16, v52, s[74:75]              // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+21], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+22], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+23], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+24], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+25], acc16          // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+26], acc20          // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+27], acc24          // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+28], acc28          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+29], acc32          // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+30], acc36          // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+31], acc40          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+32], acc44          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+33], acc48          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+34], acc52          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+35], acc56          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+36], acc60          // copy acc to vreg[15]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 0, 1, 0), (0, 1, 1, 0), (0, 2, 1, 0), (0, 3, 1, 0), (0, 0, 2, 0), (0, 1, 2, 0), (0, 2, 2, 0), (0, 3, 2, 0), (0, 0, 3, 0), (0, 1, 3, 0), (0, 2, 3, 0), (0, 3, 3, 0)] */

/* apply mask, calc new C and issue writes */
buffer_store_dword v21, v37, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v22, v38, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v23, v39, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v24, v40, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v25, v41, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v26, v42, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v27, v43, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v28, v44, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v29, v45, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v30, v46, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v31, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v32, v48, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v33, v49, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v34, v50, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v35, v51, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v36, v52, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (0,0,4,0:vw1); (0,1,4,0:vw1); (0,2,4,0:vw1); (0,3,4,0:vw1); (0,0,5,0:vw1); (0,1,5,0:vw1); (0,2,5,0:vw1); (0,3,5,0:vw1); (0,0,6,0:vw1); (0,1,6,0:vw1); (0,2,6,0:vw1); (0,3,6,0:vw1); (0,0,7,0:vw1); (0,1,7,0:vw1); (0,2,7,0:vw1); (0,3,7,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v16, BufferOOB
/* (d1,vc1,d0,vc0)=(0,4,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v37, v13, v10, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v37, v16, v37, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v38, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v38, v16, v38, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v39, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v39, v16, v39, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v40, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v40, v16, v40, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v41, v13, v10, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v41, v16, v41, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v42, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v42, v16, v42, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v43, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v43, v16, v43, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v44, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v44, v16, v44, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v45, v13, v10, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v45, v16, v45, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v46, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v46, v16, v46, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v47, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v47, v16, v47, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v48, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v48, v16, v48, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v49, v13, v10, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v49, v16, v49, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v50, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v50, v16, v50, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v51, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v51, v16, v51, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v52, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v52, v16, v52, s[74:75]              // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+21], acc1           // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+22], acc5           // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+23], acc9           // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+24], acc13          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+25], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+26], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+27], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+28], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+29], acc33          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+30], acc37          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+31], acc41          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+32], acc45          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+33], acc49          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+34], acc53          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+35], acc57          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+36], acc61          // copy acc to vreg[31]

/* rC *= alpha batchElements=[(0, 0, 4, 0), (0, 1, 4, 0), (0, 2, 4, 0), (0, 3, 4, 0), (0, 0, 5, 0), (0, 1, 5, 0), (0, 2, 5, 0), (0, 3, 5, 0), (0, 0, 6, 0), (0, 1, 6, 0), (0, 2, 6, 0), (0, 3, 6, 0), (0, 0, 7, 0), (0, 1, 7, 0), (0, 2, 7, 0), (0, 3, 7, 0)] */

/* apply mask, calc new C and issue writes */
buffer_store_dword v21, v37, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v22, v38, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v23, v39, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v24, v40, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v25, v41, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v26, v42, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v27, v43, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v28, v44, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v29, v45, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v30, v46, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v31, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v32, v48, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v33, v49, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v34, v50, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v35, v51, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v36, v52, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #2 (d1,d0,vc1,vc0) = */
/*    (0,0,8,0:vw1); (0,1,8,0:vw1); (0,2,8,0:vw1); (0,3,8,0:vw1); (0,0,9,0:vw1); (0,1,9,0:vw1); (0,2,9,0:vw1); (0,3,9,0:vw1); (0,0,10,0:vw1); (0,1,10,0:vw1); (0,2,10,0:vw1); (0,3,10,0:vw1); (0,0,11,0:vw1); (0,1,11,0:vw1); (0,2,11,0:vw1); (0,3,11,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v16, BufferOOB
/* (d1,vc1,d0,vc0)=(0,8,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v37, v13, v10, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v37, v16, v37, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v38, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v38, v16, v38, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v39, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v39, v16, v39, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v40, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v40, v16, v40, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v41, v13, v10, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v41, v16, v41, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v42, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v42, v16, v42, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v43, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v43, v16, v43, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v44, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v44, v16, v44, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v45, v13, v10, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v45, v16, v45, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v46, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v46, v16, v46, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v47, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v47, v16, v47, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v48, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v48, v16, v48, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v49, v13, v10, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v49, v16, v49, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v50, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v50, v16, v50, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v51, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v51, v16, v51, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v52, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v52, v16, v52, s[74:75]              // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+21], acc2           // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+22], acc6           // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+23], acc10          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+24], acc14          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+25], acc18          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+26], acc22          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+27], acc26          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+28], acc30          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+29], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+30], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+31], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+32], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+33], acc50          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+34], acc54          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+35], acc58          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+36], acc62          // copy acc to vreg[47]

/* rC *= alpha batchElements=[(0, 0, 8, 0), (0, 1, 8, 0), (0, 2, 8, 0), (0, 3, 8, 0), (0, 0, 9, 0), (0, 1, 9, 0), (0, 2, 9, 0), (0, 3, 9, 0), (0, 0, 10, 0), (0, 1, 10, 0), (0, 2, 10, 0), (0, 3, 10, 0), (0, 0, 11, 0), (0, 1, 11, 0), (0, 2, 11, 0), (0, 3, 11, 0)] */

/* apply mask, calc new C and issue writes */
buffer_store_dword v21, v37, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v22, v38, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v23, v39, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v24, v40, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v25, v41, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v26, v42, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v27, v43, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v28, v44, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v29, v45, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v30, v46, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v31, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v32, v48, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v33, v49, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v34, v50, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v35, v51, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v36, v52, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #3 (d1,d0,vc1,vc0) = */
/*    (0,0,12,0:vw1); (0,1,12,0:vw1); (0,2,12,0:vw1); (0,3,12,0:vw1); (0,0,13,0:vw1); (0,1,13,0:vw1); (0,2,13,0:vw1); (0,3,13,0:vw1); (0,0,14,0:vw1); (0,1,14,0:vw1); (0,2,14,0:vw1); (0,3,14,0:vw1); (0,0,15,0:vw1); (0,1,15,0:vw1); (0,2,15,0:vw1); (0,3,15,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v16, BufferOOB
/* (d1,vc1,d0,vc0)=(0,12,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v37, v13, v10, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v37, v16, v37, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v38, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v38, v16, v38, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v39, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v39, v16, v39, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v40, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v40, v16, v40, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v41, v13, v10, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v41, v16, v41, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v42, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v42, v16, v42, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v43, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v43, v16, v43, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v44, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v44, v16, v44, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v45, v13, v10, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v45, v16, v45, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v46, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v46, v16, v46, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v47, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v47, v16, v47, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v48, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v48, v16, v48, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v49, v13, v10, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v49, v16, v49, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v50, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v50, v16, v50, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v51, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v51, v16, v51, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v52, v13, v14, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v52, v16, v52, s[74:75]              // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+21], acc3           // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+22], acc7           // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+23], acc11          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+24], acc15          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+25], acc19          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+26], acc23          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+27], acc27          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+28], acc31          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+29], acc35          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+30], acc39          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+31], acc43          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+32], acc47          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+33], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+34], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+35], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+36], acc63          // copy acc to vreg[63]

/* rC *= alpha batchElements=[(0, 0, 12, 0), (0, 1, 12, 0), (0, 2, 12, 0), (0, 3, 12, 0), (0, 0, 13, 0), (0, 1, 13, 0), (0, 2, 13, 0), (0, 3, 13, 0), (0, 0, 14, 0), (0, 1, 14, 0), (0, 2, 14, 0), (0, 3, 14, 0), (0, 0, 15, 0), (0, 1, 15, 0), (0, 2, 15, 0), (0, 3, 15, 0)] */

/* apply mask, calc new C and issue writes */
buffer_store_dword v21, v37, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v22, v38, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v23, v39, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v24, v40, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v25, v41, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v26, v42, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v27, v43, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v28, v44, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v29, v45, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v30, v46, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v31, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v32, v48, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v33, v49, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v34, v50, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v35, v51, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v36, v52, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_End:
s_getpc_b64 s[70:71]                               // addr of next instr
s_add_i32 s72, label_KernelEnd, 4                  // target branch offset
s_add_u32 s70, s70, s72                            // add target branch offset
s_addc_u32 s71, s71, 0                             // add high and carry
s_setpc_b64 s[70:71]                               // branch to label_KernelEnd
label_GSU:
.set sgprAddressScaleAlphaVec, 64
.set sgprSrdScaleAlphaVec, 72
s_mov_b64 s[sgprSrdScaleAlphaVec+0:sgprSrdScaleAlphaVec+0+1], s[sgprAddressScaleAlphaVec+0:sgprAddressScaleAlphaVec+0+1] // init SRD base address
s_mov_b32 s[sgprSrdScaleAlphaVec+3], Srd127_96     // Set bits 127_96 in post-loop SRD
s_cmp_eq_u64 s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1], 0 // s[AddressScaleAlphaVec] == 0 ?
s_cbranch_scc0 label_ScaleAlphaVecAddrValid        // branch if s[AddressScaleAlphaVec] != 0
s_mov_b32 s[sgprSrdScaleAlphaVec+2], 0
s_branch label_ScaleAlphaVecAddrValid_End
label_ScaleAlphaVecAddrValid:
s_mov_b32 s[sgprSrdScaleAlphaVec+2], s[sgprSizeI]
label_ScaleAlphaVecAddrValid_End:

s_mul_i32 s[sgprSrdScaleAlphaVec+2], 0x4, s[sgprSrdScaleAlphaVec+2] // ScaleAlphaVec scaled by BPE
s_add_u32 s8, s[sgprWorkGroup2], 0x1
s_mul_i32 s8, s[sgprBiasStride], s8                // stride * (wg+1)
s_cmp_eq_u32 s8, 0                                 // bias stride = 0?
s_cselect_b32 s8, s[sgprSizeI], s8
s_mov_b64 s[sgprSrdBias+0:sgprSrdBias+0+1], s[sgprAddressBias+0:sgprAddressBias+0+1] // init SRD base address
s_mov_b32 s[sgprSrdBias+3], Srd127_96              // Set bits 127_96 in post-loop SRD
s_cmp_eq_u64 s[sgprAddressBias:sgprAddressBias+1], 0 // s[AddressBias] == 0 ?
s_cbranch_scc0 label_BiasAddrValid                 // branch if s[AddressBias] != 0
s_mov_b32 s[sgprSrdBias+2], 0
s_branch label_BiasAddrValid_End
label_BiasAddrValid:
s_mov_b32 s[sgprSrdBias+2], s8
label_BiasAddrValid_End:

label_Load_Biasf32_0:
s_cmpk_lg_u32 s[sgprBiasType], 0                   // BiasType != 0
s_cbranch_scc1 label_Load_Biasbf16_0               // Branch if true

/******************************************/
/* Read vector to LDS                     */
/******************************************/
s_mul_i32 s8, 128, s[sgprWorkGroup0]               // wgp0 * MT0
v_add_u32 v18, s8, v[vgprSerial]                   // coord 0 = wgp0 * MT0 + thread offset
s_mul_i32 s[sgprSrdBias+2], 0x4, s[sgprSrdBias+2]  // scaled by BPE
s_mul_i32 s8, s[sgprBiasStride], s[sgprWorkGroup2] // Stride * WG
v_add_u32 v16, s8, v18                             // coord 0 = wgp0 * MT0 + thread offset + Stride * WG
v_lshlrev_b32 v16, 0x2, v16                        // Global bias address scaled by BPE
v_lshlrev_b32 v17, 0x2, v18                        // Global scaleAlpha address scaled by BPE
s_mul_i32 s8, 128, s[sgprWorkGroup1]               // wgp1 * MT1
v_add_u32 v18, s8, v[vgprSerial]                   // coord 1 = wgp1 * MT1 + thread offset
buffer_load_dword v14, v16, s[sgprSrdBias:sgprSrdBias+3], 0 offen offset:0 // Load Bias
buffer_load_dword v15, v17, s[sgprSrdScaleAlphaVec:sgprSrdScaleAlphaVec+3], 0 offen offset:0 // Load ScaleAlphaVec
v_lshlrev_b32 v18, 0x2, v[vgprSerial]              // Local address scaled by BPE
s_barrier                                          // wait for all global loads.
s_waitcnt vmcnt(1)                                 // wait for global load
ds_write_b32 v18, v14 offset:0                     // store bias
v_cmp_gt_u32 s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1], s[sgprSrdScaleAlphaVec+2], 0 //  == 0 ?
s_waitcnt vmcnt(0)                                 // wait for global load
v_cndmask_b32 v15, 1.0, v15, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
ds_write_b32 v18, v15 offset:1024                  // store scaleAlpha
s_branch label_Load_Bias_End                       // Branch to load bias end
label_Load_Biasbf16_0:
s_cmpk_lg_u32 s[sgprBiasType], 7                   // BiasType != 7
s_cbranch_scc1 label_Load_Bias_End                 // Branch if true

/******************************************/
/* Read vector to LDS                     */
/******************************************/
s_mul_i32 s8, 128, s[sgprWorkGroup0]               // wgp0 * MT0
v_add_u32 v18, s8, v[vgprSerial]                   // coord 0 = wgp0 * MT0 + thread offset
s_mul_i32 s[sgprSrdBias+2], 0x2, s[sgprSrdBias+2]  // scaled by BPE
s_mul_i32 s8, s[sgprBiasStride], s[sgprWorkGroup2] // Stride * WG
v_add_u32 v16, s8, v18                             // coord 0 = wgp0 * MT0 + thread offset + Stride * WG
v_lshlrev_b32 v16, 0x1, v16                        // Global bias address scaled by BPE
v_lshlrev_b32 v17, 0x2, v18                        // Global scaleAlpha address scaled by BPE
s_mul_i32 s8, 128, s[sgprWorkGroup1]               // wgp1 * MT1
v_add_u32 v18, s8, v[vgprSerial]                   // coord 1 = wgp1 * MT1 + thread offset
buffer_load_short_d16 v14, v16, s[sgprSrdBias:sgprSrdBias+3], 0 offen offset:0 // Load Bias
buffer_load_dword v15, v17, s[sgprSrdScaleAlphaVec:sgprSrdScaleAlphaVec+3], 0 offen offset:0 // Load ScaleAlphaVec
v_lshlrev_b32 v18, 0x2, v[vgprSerial]              // Local address scaled by BPE
s_barrier                                          // wait for all global loads.
s_waitcnt vmcnt(1)                                 // wait for global load
v_cvt_f32_bf16 v14, v14 src0_sel:WORD_0            // cvt bf16 to f32
ds_write_b32 v18, v14 offset:0                     // store bias
v_cmp_gt_u32 s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1], s[sgprSrdScaleAlphaVec+2], 0 //  == 0 ?
s_waitcnt vmcnt(0)                                 // wait for global load
v_cndmask_b32 v15, 1.0, v15, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
ds_write_b32 v18, v15 offset:1024                  // store scaleAlpha
s_branch label_Load_Bias_End                       // Branch to load bias end
label_Load_Bias_End:
.set sgprAddressScaleAlphaVec, UNDEF
.set sgprSrdScaleAlphaVec, UNDEF
s_cmp_eq_u32 s[sgprStreamKLocalStart], 0           // does wg start tile?
s_cbranch_scc1 label_NoBranch_0MXDW6EW9K7ZNG8F     // Only branch on scc0
s_getpc_b64 s[80:81]                               // addr of next instr
s_add_i32 s82, label_SK_Partials_1, 4              // target branch offset
s_add_u32 s80, s80, s82                            // add target branch offset
s_addc_u32 s81, s81, 0                             // add high and carry
s_setpc_b64 s[80:81]                               // branch to label_SK_Partials_1
label_NoBranch_0MXDW6EW9K7ZNG8F:
s_cmp_eq_u32 s[sgprStreamKLocalEnd], s[sgprItersPerTile] // does wg finish tile?
s_cbranch_scc1 label_SK_Store                      // Branch if started and finished tile, go to regular store code
s_add_u32 s8, s[sgprStreamKIdx], 1                 // input partial tile index
s_mul_hi_u32 s71, s[sgprStreamKIterEnd], s[sgprMagicNumberItersPerTile] // s_magic mul, div alg 2
s_lshr_b32 s72, s[sgprMagicShiftItersPerTile], 31  // tmpS = extract abit
s_mul_i32 s70, s[sgprStreamKIterEnd], s72          // s_magic mul, div alg 2
s_add_u32 s70, s70, s71
s_and_b32 s72, s[sgprMagicShiftItersPerTile], 2147483647 // tmpS = remove abit to final shift
s_lshr_b32 s70, s70, s72                           // sMagicDiv Alg 2
s_mul_i32 s70, s70, s[sgprItersPerTile]            // start iteration of partial tile
s_sub_u32 s9, s[sgprStreamKIterEnd], s70           // calc iterations completed by this WG
label_SK_Fixup:
s_lshl_b32 s70, s8, 2                              // flag offset based on CTA index
s_load_dword s72, s[sgprAddressFlags:sgprAddressFlags+1], s70 glc // get flag
s_waitcnt lgkmcnt(0)                               // wait for flag load
s_cmp_eq_u32 s72, 1                                // check if ready
s_cbranch_scc0 label_SK_Fixup                      // if flag not set, wait and check again
s_barrier                                          // wait for all workgroups before resetting flag
v_readfirstlane_b32 s72, v[vgprSerial]             // Wave 0 updates flags
s_cmp_eq_u32 s72, 0                                // Check for wave 0
s_cbranch_scc0 label_SK_SkipFlagReset              // Skip flag reset
s_store_dword s72, s[sgprAddressFlags:sgprAddressFlags+1], s70 glc // reset flag
label_SK_SkipFlagReset:
label_Fixup_E0:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=16 */
s_mov_b64 s[sgprSrdWS+0:sgprSrdWS+0+1], s[sgprAddressWS+0:sgprAddressWS+0+1] // init SRD base address
s_mov_b32 s[sgprSrdWS+2], BufferOOB
s_mov_b32 s[sgprSrdWS+3], Srd127_96                // Set bits 127_96 in post-loop SRD

s_mul_i32 s64, 0x10000, s8                         // Offset to correct partials tile
s_add_u32 s[sgprSrdWS+0], s[sgprSrdWS+0], s64      // add lo to SRD
s_addc_u32 s[sgprSrdWS+1], s[sgprSrdWS+1], 0       // add hi to SRD
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Fixup Batch #0 (d1,d0,vc1,vc0) =       */
/*      (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,0,1,0:vw1); (0,1,1,0:vw1); (0,2,1,0:vw1); (0,3,1,0:vw1); (0,0,2,0:vw1); (0,1,2,0:vw1); (0,2,2,0:vw1); (0,3,2,0:vw1); (0,0,3,0:vw1); (0,1,3,0:vw1); (0,2,3,0:vw1); (0,3,3,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_lshlrev_b32 v22, 2, v[vgprSerial]                // v22 = v[vgprSerial] * 4
s_mov_b32 s64, 0                                   // Init sgpr offset
buffer_load_dword v39, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v40, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v41, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v42, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v43, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v44, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v45, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v46, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v47, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v48, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v49, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v50, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v51, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v52, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v53, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v54, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
v_accvgpr_read_b32 v[vgprValuC+23], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+24], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+25], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+26], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+27], acc16          // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+28], acc20          // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+29], acc24          // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+30], acc28          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+31], acc32          // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+32], acc36          // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+33], acc40          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+34], acc44          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+35], acc48          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+36], acc52          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+37], acc56          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+38], acc60          // copy acc to vreg[15]
s_nop 1                                            // 2 wait states required before reading vgpr

/* apply mask, calc new C and issue writes */
v_mov_b32 v18, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v19, 0x7fff0000                          // fp32 Nan
v_mov_b32 v20, 0x7fff                              // rounding bias for bfloat16

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 0 + 0 - 1
v_add_f32 v[vgprValuC+23], v[vgprValuC+23], v39    // accum partials

s_waitcnt vmcnt(14)                                // wait C (interleaved) 14 = 16 - 1 + 0 - 1
v_add_f32 v[vgprValuC+24], v[vgprValuC+24], v40    // accum partials

s_waitcnt vmcnt(13)                                // wait C (interleaved) 13 = 16 - 2 + 0 - 1
v_add_f32 v[vgprValuC+25], v[vgprValuC+25], v41    // accum partials

s_waitcnt vmcnt(12)                                // wait C (interleaved) 12 = 16 - 3 + 0 - 1
v_add_f32 v[vgprValuC+26], v[vgprValuC+26], v42    // accum partials

s_waitcnt vmcnt(11)                                // wait C (interleaved) 11 = 16 - 4 + 0 - 1
v_add_f32 v[vgprValuC+27], v[vgprValuC+27], v43    // accum partials

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 16 - 5 + 0 - 1
v_add_f32 v[vgprValuC+28], v[vgprValuC+28], v44    // accum partials

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 16 - 6 + 0 - 1
v_add_f32 v[vgprValuC+29], v[vgprValuC+29], v45    // accum partials

s_waitcnt vmcnt(8)                                 // wait C (interleaved) 8 = 16 - 7 + 0 - 1
v_add_f32 v[vgprValuC+30], v[vgprValuC+30], v46    // accum partials

s_waitcnt vmcnt(7)                                 // wait C (interleaved) 7 = 16 - 8 + 0 - 1
v_add_f32 v[vgprValuC+31], v[vgprValuC+31], v47    // accum partials

s_waitcnt vmcnt(6)                                 // wait C (interleaved) 6 = 16 - 9 + 0 - 1
v_add_f32 v[vgprValuC+32], v[vgprValuC+32], v48    // accum partials

s_waitcnt vmcnt(5)                                 // wait C (interleaved) 5 = 16 - 10 + 0 - 1
v_add_f32 v[vgprValuC+33], v[vgprValuC+33], v49    // accum partials

s_waitcnt vmcnt(4)                                 // wait C (interleaved) 4 = 16 - 11 + 0 - 1
v_add_f32 v[vgprValuC+34], v[vgprValuC+34], v50    // accum partials

s_waitcnt vmcnt(3)                                 // wait C (interleaved) 3 = 16 - 12 + 0 - 1
v_add_f32 v[vgprValuC+35], v[vgprValuC+35], v51    // accum partials

s_waitcnt vmcnt(2)                                 // wait C (interleaved) 2 = 16 - 13 + 0 - 1
v_add_f32 v[vgprValuC+36], v[vgprValuC+36], v52    // accum partials

s_waitcnt vmcnt(1)                                 // wait C (interleaved) 1 = 16 - 14 + 0 - 1
v_add_f32 v[vgprValuC+37], v[vgprValuC+37], v53    // accum partials

s_waitcnt vmcnt(0)                                 // wait C (interleaved) 0 = 16 - 15 + 0 - 1
v_add_f32 v[vgprValuC+38], v[vgprValuC+38], v54    // accum partials
v_accvgpr_write_b32 acc0, v[vgprValuC+23]          // copy vreg[0] to acc
v_accvgpr_write_b32 acc4, v[vgprValuC+24]          // copy vreg[1] to acc
v_accvgpr_write_b32 acc8, v[vgprValuC+25]          // copy vreg[2] to acc
v_accvgpr_write_b32 acc12, v[vgprValuC+26]         // copy vreg[3] to acc
v_accvgpr_write_b32 acc16, v[vgprValuC+27]         // copy vreg[4] to acc
v_accvgpr_write_b32 acc20, v[vgprValuC+28]         // copy vreg[5] to acc
v_accvgpr_write_b32 acc24, v[vgprValuC+29]         // copy vreg[6] to acc
v_accvgpr_write_b32 acc28, v[vgprValuC+30]         // copy vreg[7] to acc
v_accvgpr_write_b32 acc32, v[vgprValuC+31]         // copy vreg[8] to acc
v_accvgpr_write_b32 acc36, v[vgprValuC+32]         // copy vreg[9] to acc
v_accvgpr_write_b32 acc40, v[vgprValuC+33]         // copy vreg[10] to acc
v_accvgpr_write_b32 acc44, v[vgprValuC+34]         // copy vreg[11] to acc
v_accvgpr_write_b32 acc48, v[vgprValuC+35]         // copy vreg[12] to acc
v_accvgpr_write_b32 acc52, v[vgprValuC+36]         // copy vreg[13] to acc
v_accvgpr_write_b32 acc56, v[vgprValuC+37]         // copy vreg[14] to acc
v_accvgpr_write_b32 acc60, v[vgprValuC+38]         // copy vreg[15] to acc
s_nop 1                                            // 2 wait states required before reading vgpr
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Fixup Batch #1 (d1,d0,vc1,vc0) =       */
/*      (0,0,4,0:vw1); (0,1,4,0:vw1); (0,2,4,0:vw1); (0,3,4,0:vw1); (0,0,5,0:vw1); (0,1,5,0:vw1); (0,2,5,0:vw1); (0,3,5,0:vw1); (0,0,6,0:vw1); (0,1,6,0:vw1); (0,2,6,0:vw1); (0,3,6,0:vw1); (0,0,7,0:vw1); (0,1,7,0:vw1); (0,2,7,0:vw1); (0,3,7,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v39, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v40, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v41, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v42, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v43, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v44, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v45, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v46, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v47, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v48, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v49, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v50, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v51, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v52, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v53, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v54, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
v_accvgpr_read_b32 v[vgprValuC+23], acc1           // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+24], acc5           // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+25], acc9           // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+26], acc13          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+27], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+28], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+29], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+30], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+31], acc33          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+32], acc37          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+33], acc41          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+34], acc45          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+35], acc49          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+36], acc53          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+37], acc57          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+38], acc61          // copy acc to vreg[31]
s_nop 1                                            // 2 wait states required before reading vgpr

/* apply mask, calc new C and issue writes */
v_mov_b32 v18, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v19, 0x7fff0000                          // fp32 Nan
v_mov_b32 v20, 0x7fff                              // rounding bias for bfloat16

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 0 + 0 - 1
v_add_f32 v[vgprValuC+23], v[vgprValuC+23], v39    // accum partials

s_waitcnt vmcnt(14)                                // wait C (interleaved) 14 = 16 - 1 + 0 - 1
v_add_f32 v[vgprValuC+24], v[vgprValuC+24], v40    // accum partials

s_waitcnt vmcnt(13)                                // wait C (interleaved) 13 = 16 - 2 + 0 - 1
v_add_f32 v[vgprValuC+25], v[vgprValuC+25], v41    // accum partials

s_waitcnt vmcnt(12)                                // wait C (interleaved) 12 = 16 - 3 + 0 - 1
v_add_f32 v[vgprValuC+26], v[vgprValuC+26], v42    // accum partials

s_waitcnt vmcnt(11)                                // wait C (interleaved) 11 = 16 - 4 + 0 - 1
v_add_f32 v[vgprValuC+27], v[vgprValuC+27], v43    // accum partials

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 16 - 5 + 0 - 1
v_add_f32 v[vgprValuC+28], v[vgprValuC+28], v44    // accum partials

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 16 - 6 + 0 - 1
v_add_f32 v[vgprValuC+29], v[vgprValuC+29], v45    // accum partials

s_waitcnt vmcnt(8)                                 // wait C (interleaved) 8 = 16 - 7 + 0 - 1
v_add_f32 v[vgprValuC+30], v[vgprValuC+30], v46    // accum partials

s_waitcnt vmcnt(7)                                 // wait C (interleaved) 7 = 16 - 8 + 0 - 1
v_add_f32 v[vgprValuC+31], v[vgprValuC+31], v47    // accum partials

s_waitcnt vmcnt(6)                                 // wait C (interleaved) 6 = 16 - 9 + 0 - 1
v_add_f32 v[vgprValuC+32], v[vgprValuC+32], v48    // accum partials

s_waitcnt vmcnt(5)                                 // wait C (interleaved) 5 = 16 - 10 + 0 - 1
v_add_f32 v[vgprValuC+33], v[vgprValuC+33], v49    // accum partials

s_waitcnt vmcnt(4)                                 // wait C (interleaved) 4 = 16 - 11 + 0 - 1
v_add_f32 v[vgprValuC+34], v[vgprValuC+34], v50    // accum partials

s_waitcnt vmcnt(3)                                 // wait C (interleaved) 3 = 16 - 12 + 0 - 1
v_add_f32 v[vgprValuC+35], v[vgprValuC+35], v51    // accum partials

s_waitcnt vmcnt(2)                                 // wait C (interleaved) 2 = 16 - 13 + 0 - 1
v_add_f32 v[vgprValuC+36], v[vgprValuC+36], v52    // accum partials

s_waitcnt vmcnt(1)                                 // wait C (interleaved) 1 = 16 - 14 + 0 - 1
v_add_f32 v[vgprValuC+37], v[vgprValuC+37], v53    // accum partials

s_waitcnt vmcnt(0)                                 // wait C (interleaved) 0 = 16 - 15 + 0 - 1
v_add_f32 v[vgprValuC+38], v[vgprValuC+38], v54    // accum partials
v_accvgpr_write_b32 acc1, v[vgprValuC+23]          // copy vreg[16] to acc
v_accvgpr_write_b32 acc5, v[vgprValuC+24]          // copy vreg[17] to acc
v_accvgpr_write_b32 acc9, v[vgprValuC+25]          // copy vreg[18] to acc
v_accvgpr_write_b32 acc13, v[vgprValuC+26]         // copy vreg[19] to acc
v_accvgpr_write_b32 acc17, v[vgprValuC+27]         // copy vreg[20] to acc
v_accvgpr_write_b32 acc21, v[vgprValuC+28]         // copy vreg[21] to acc
v_accvgpr_write_b32 acc25, v[vgprValuC+29]         // copy vreg[22] to acc
v_accvgpr_write_b32 acc29, v[vgprValuC+30]         // copy vreg[23] to acc
v_accvgpr_write_b32 acc33, v[vgprValuC+31]         // copy vreg[24] to acc
v_accvgpr_write_b32 acc37, v[vgprValuC+32]         // copy vreg[25] to acc
v_accvgpr_write_b32 acc41, v[vgprValuC+33]         // copy vreg[26] to acc
v_accvgpr_write_b32 acc45, v[vgprValuC+34]         // copy vreg[27] to acc
v_accvgpr_write_b32 acc49, v[vgprValuC+35]         // copy vreg[28] to acc
v_accvgpr_write_b32 acc53, v[vgprValuC+36]         // copy vreg[29] to acc
v_accvgpr_write_b32 acc57, v[vgprValuC+37]         // copy vreg[30] to acc
v_accvgpr_write_b32 acc61, v[vgprValuC+38]         // copy vreg[31] to acc
s_nop 1                                            // 2 wait states required before reading vgpr
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Fixup Batch #2 (d1,d0,vc1,vc0) =       */
/*      (0,0,8,0:vw1); (0,1,8,0:vw1); (0,2,8,0:vw1); (0,3,8,0:vw1); (0,0,9,0:vw1); (0,1,9,0:vw1); (0,2,9,0:vw1); (0,3,9,0:vw1); (0,0,10,0:vw1); (0,1,10,0:vw1); (0,2,10,0:vw1); (0,3,10,0:vw1); (0,0,11,0:vw1); (0,1,11,0:vw1); (0,2,11,0:vw1); (0,3,11,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v39, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v40, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v41, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v42, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v43, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v44, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v45, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v46, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v47, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v48, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v49, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v50, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v51, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v52, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v53, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v54, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
v_accvgpr_read_b32 v[vgprValuC+23], acc2           // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+24], acc6           // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+25], acc10          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+26], acc14          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+27], acc18          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+28], acc22          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+29], acc26          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+30], acc30          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+31], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+32], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+33], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+34], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+35], acc50          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+36], acc54          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+37], acc58          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+38], acc62          // copy acc to vreg[47]
s_nop 1                                            // 2 wait states required before reading vgpr

/* apply mask, calc new C and issue writes */
v_mov_b32 v18, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v19, 0x7fff0000                          // fp32 Nan
v_mov_b32 v20, 0x7fff                              // rounding bias for bfloat16

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 0 + 0 - 1
v_add_f32 v[vgprValuC+23], v[vgprValuC+23], v39    // accum partials

s_waitcnt vmcnt(14)                                // wait C (interleaved) 14 = 16 - 1 + 0 - 1
v_add_f32 v[vgprValuC+24], v[vgprValuC+24], v40    // accum partials

s_waitcnt vmcnt(13)                                // wait C (interleaved) 13 = 16 - 2 + 0 - 1
v_add_f32 v[vgprValuC+25], v[vgprValuC+25], v41    // accum partials

s_waitcnt vmcnt(12)                                // wait C (interleaved) 12 = 16 - 3 + 0 - 1
v_add_f32 v[vgprValuC+26], v[vgprValuC+26], v42    // accum partials

s_waitcnt vmcnt(11)                                // wait C (interleaved) 11 = 16 - 4 + 0 - 1
v_add_f32 v[vgprValuC+27], v[vgprValuC+27], v43    // accum partials

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 16 - 5 + 0 - 1
v_add_f32 v[vgprValuC+28], v[vgprValuC+28], v44    // accum partials

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 16 - 6 + 0 - 1
v_add_f32 v[vgprValuC+29], v[vgprValuC+29], v45    // accum partials

s_waitcnt vmcnt(8)                                 // wait C (interleaved) 8 = 16 - 7 + 0 - 1
v_add_f32 v[vgprValuC+30], v[vgprValuC+30], v46    // accum partials

s_waitcnt vmcnt(7)                                 // wait C (interleaved) 7 = 16 - 8 + 0 - 1
v_add_f32 v[vgprValuC+31], v[vgprValuC+31], v47    // accum partials

s_waitcnt vmcnt(6)                                 // wait C (interleaved) 6 = 16 - 9 + 0 - 1
v_add_f32 v[vgprValuC+32], v[vgprValuC+32], v48    // accum partials

s_waitcnt vmcnt(5)                                 // wait C (interleaved) 5 = 16 - 10 + 0 - 1
v_add_f32 v[vgprValuC+33], v[vgprValuC+33], v49    // accum partials

s_waitcnt vmcnt(4)                                 // wait C (interleaved) 4 = 16 - 11 + 0 - 1
v_add_f32 v[vgprValuC+34], v[vgprValuC+34], v50    // accum partials

s_waitcnt vmcnt(3)                                 // wait C (interleaved) 3 = 16 - 12 + 0 - 1
v_add_f32 v[vgprValuC+35], v[vgprValuC+35], v51    // accum partials

s_waitcnt vmcnt(2)                                 // wait C (interleaved) 2 = 16 - 13 + 0 - 1
v_add_f32 v[vgprValuC+36], v[vgprValuC+36], v52    // accum partials

s_waitcnt vmcnt(1)                                 // wait C (interleaved) 1 = 16 - 14 + 0 - 1
v_add_f32 v[vgprValuC+37], v[vgprValuC+37], v53    // accum partials

s_waitcnt vmcnt(0)                                 // wait C (interleaved) 0 = 16 - 15 + 0 - 1
v_add_f32 v[vgprValuC+38], v[vgprValuC+38], v54    // accum partials
v_accvgpr_write_b32 acc2, v[vgprValuC+23]          // copy vreg[32] to acc
v_accvgpr_write_b32 acc6, v[vgprValuC+24]          // copy vreg[33] to acc
v_accvgpr_write_b32 acc10, v[vgprValuC+25]         // copy vreg[34] to acc
v_accvgpr_write_b32 acc14, v[vgprValuC+26]         // copy vreg[35] to acc
v_accvgpr_write_b32 acc18, v[vgprValuC+27]         // copy vreg[36] to acc
v_accvgpr_write_b32 acc22, v[vgprValuC+28]         // copy vreg[37] to acc
v_accvgpr_write_b32 acc26, v[vgprValuC+29]         // copy vreg[38] to acc
v_accvgpr_write_b32 acc30, v[vgprValuC+30]         // copy vreg[39] to acc
v_accvgpr_write_b32 acc34, v[vgprValuC+31]         // copy vreg[40] to acc
v_accvgpr_write_b32 acc38, v[vgprValuC+32]         // copy vreg[41] to acc
v_accvgpr_write_b32 acc42, v[vgprValuC+33]         // copy vreg[42] to acc
v_accvgpr_write_b32 acc46, v[vgprValuC+34]         // copy vreg[43] to acc
v_accvgpr_write_b32 acc50, v[vgprValuC+35]         // copy vreg[44] to acc
v_accvgpr_write_b32 acc54, v[vgprValuC+36]         // copy vreg[45] to acc
v_accvgpr_write_b32 acc58, v[vgprValuC+37]         // copy vreg[46] to acc
v_accvgpr_write_b32 acc62, v[vgprValuC+38]         // copy vreg[47] to acc
s_nop 1                                            // 2 wait states required before reading vgpr
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Fixup Batch #3 (d1,d0,vc1,vc0) =       */
/*      (0,0,12,0:vw1); (0,1,12,0:vw1); (0,2,12,0:vw1); (0,3,12,0:vw1); (0,0,13,0:vw1); (0,1,13,0:vw1); (0,2,13,0:vw1); (0,3,13,0:vw1); (0,0,14,0:vw1); (0,1,14,0:vw1); (0,2,14,0:vw1); (0,3,14,0:vw1); (0,0,15,0:vw1); (0,1,15,0:vw1); (0,2,15,0:vw1); (0,3,15,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v39, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v40, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v41, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v42, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v43, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v44, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v45, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v46, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v47, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v48, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v49, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v50, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v51, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v52, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v53, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
s_add_u32 s64, s64, 1024                           // Inc sgpr offset
buffer_load_dword v54, v22, s[sgprSrdWS:sgprSrdWS+3], s64 offen offset:0 // load WS
v_accvgpr_read_b32 v[vgprValuC+23], acc3           // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+24], acc7           // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+25], acc11          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+26], acc15          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+27], acc19          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+28], acc23          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+29], acc27          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+30], acc31          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+31], acc35          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+32], acc39          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+33], acc43          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+34], acc47          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+35], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+36], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+37], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+38], acc63          // copy acc to vreg[63]
s_nop 1                                            // 2 wait states required before reading vgpr

/* apply mask, calc new C and issue writes */
v_mov_b32 v18, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v19, 0x7fff0000                          // fp32 Nan
v_mov_b32 v20, 0x7fff                              // rounding bias for bfloat16

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 0 + 0 - 1
v_add_f32 v[vgprValuC+23], v[vgprValuC+23], v39    // accum partials

s_waitcnt vmcnt(14)                                // wait C (interleaved) 14 = 16 - 1 + 0 - 1
v_add_f32 v[vgprValuC+24], v[vgprValuC+24], v40    // accum partials

s_waitcnt vmcnt(13)                                // wait C (interleaved) 13 = 16 - 2 + 0 - 1
v_add_f32 v[vgprValuC+25], v[vgprValuC+25], v41    // accum partials

s_waitcnt vmcnt(12)                                // wait C (interleaved) 12 = 16 - 3 + 0 - 1
v_add_f32 v[vgprValuC+26], v[vgprValuC+26], v42    // accum partials

s_waitcnt vmcnt(11)                                // wait C (interleaved) 11 = 16 - 4 + 0 - 1
v_add_f32 v[vgprValuC+27], v[vgprValuC+27], v43    // accum partials

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 16 - 5 + 0 - 1
v_add_f32 v[vgprValuC+28], v[vgprValuC+28], v44    // accum partials

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 16 - 6 + 0 - 1
v_add_f32 v[vgprValuC+29], v[vgprValuC+29], v45    // accum partials

s_waitcnt vmcnt(8)                                 // wait C (interleaved) 8 = 16 - 7 + 0 - 1
v_add_f32 v[vgprValuC+30], v[vgprValuC+30], v46    // accum partials

s_waitcnt vmcnt(7)                                 // wait C (interleaved) 7 = 16 - 8 + 0 - 1
v_add_f32 v[vgprValuC+31], v[vgprValuC+31], v47    // accum partials

s_waitcnt vmcnt(6)                                 // wait C (interleaved) 6 = 16 - 9 + 0 - 1
v_add_f32 v[vgprValuC+32], v[vgprValuC+32], v48    // accum partials

s_waitcnt vmcnt(5)                                 // wait C (interleaved) 5 = 16 - 10 + 0 - 1
v_add_f32 v[vgprValuC+33], v[vgprValuC+33], v49    // accum partials

s_waitcnt vmcnt(4)                                 // wait C (interleaved) 4 = 16 - 11 + 0 - 1
v_add_f32 v[vgprValuC+34], v[vgprValuC+34], v50    // accum partials

s_waitcnt vmcnt(3)                                 // wait C (interleaved) 3 = 16 - 12 + 0 - 1
v_add_f32 v[vgprValuC+35], v[vgprValuC+35], v51    // accum partials

s_waitcnt vmcnt(2)                                 // wait C (interleaved) 2 = 16 - 13 + 0 - 1
v_add_f32 v[vgprValuC+36], v[vgprValuC+36], v52    // accum partials

s_waitcnt vmcnt(1)                                 // wait C (interleaved) 1 = 16 - 14 + 0 - 1
v_add_f32 v[vgprValuC+37], v[vgprValuC+37], v53    // accum partials

s_waitcnt vmcnt(0)                                 // wait C (interleaved) 0 = 16 - 15 + 0 - 1
v_add_f32 v[vgprValuC+38], v[vgprValuC+38], v54    // accum partials
v_accvgpr_write_b32 acc3, v[vgprValuC+23]          // copy vreg[48] to acc
v_accvgpr_write_b32 acc7, v[vgprValuC+24]          // copy vreg[49] to acc
v_accvgpr_write_b32 acc11, v[vgprValuC+25]         // copy vreg[50] to acc
v_accvgpr_write_b32 acc15, v[vgprValuC+26]         // copy vreg[51] to acc
v_accvgpr_write_b32 acc19, v[vgprValuC+27]         // copy vreg[52] to acc
v_accvgpr_write_b32 acc23, v[vgprValuC+28]         // copy vreg[53] to acc
v_accvgpr_write_b32 acc27, v[vgprValuC+29]         // copy vreg[54] to acc
v_accvgpr_write_b32 acc31, v[vgprValuC+30]         // copy vreg[55] to acc
v_accvgpr_write_b32 acc35, v[vgprValuC+31]         // copy vreg[56] to acc
v_accvgpr_write_b32 acc39, v[vgprValuC+32]         // copy vreg[57] to acc
v_accvgpr_write_b32 acc43, v[vgprValuC+33]         // copy vreg[58] to acc
v_accvgpr_write_b32 acc47, v[vgprValuC+34]         // copy vreg[59] to acc
v_accvgpr_write_b32 acc51, v[vgprValuC+35]         // copy vreg[60] to acc
v_accvgpr_write_b32 acc55, v[vgprValuC+36]         // copy vreg[61] to acc
v_accvgpr_write_b32 acc59, v[vgprValuC+37]         // copy vreg[62] to acc
v_accvgpr_write_b32 acc63, v[vgprValuC+38]         // copy vreg[63] to acc
s_nop 1                                            // 2 wait states required before reading vgpr
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_mul_i32 s64, s[sgprskTiles], s[sgprItersPerTile]
s_mul_i32 s65, s[sgprSKItersPerWG], s[sgprskGrid]
s_sub_u32 s64, s64, s65                            // skTiles * ItersPerTile - SKItersPerWG * skGrid
s_add_u32 s65, s[sgprSKItersPerWG], 1              // Add extra iter
s_cmp_lt_u32 s8, s64                               // Check if next WG had an extra iteration
s_cselect_b32 s65, s65, s[sgprSKItersPerWG]        // Select correct number of iterations for next WG
s_add_u32 s9, s9, s65                              // next partial tile iteration
s_add_u32 s8, s8, 1                                // next partial tile index
s_cmp_lt_u32 s9, s[sgprItersPerTile]               // done loading partial tiles?
s_cbranch_scc1 label_SK_Fixup                      // Branch to continue fixup loop
label_SK_Store:
s_cmpk_eq_u32 s[sgprBeta], 0                       // Beta == 0
s_cbranch_scc0 label_GW_Beta_1                     // Branch if Beta is not zero

s_and_b32 s70, 127, s[sgprSizeI]                   // s70 = s[sgprSizeI] % 128
s_add_u32 s71, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s71                // wg0 >= nwg0-1 ?
s_cselect_b32 s70, s70, 0                          // set rMT0
s_cmpk_gt_u32 s70, 0                               // rMT0 > 0
s_cbranch_scc1 label_GW_B0_E1_1                    // jump if edges required
s_and_b32 s70, 127, s[sgprSizeJ]                   // s70 = s[sgprSizeJ] % 128
s_add_u32 s71, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s71                // wg1 >= nwg1-1
s_cselect_b32 s70, s70, 0                          // set rMT1
s_cmpk_gt_u32 s70, 0                               // rMT1 > 0
s_cbranch_scc1 label_GW_B0_E1_1                    // jump if edges required
label_GW_B0_E0_1:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=16 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,0,1,0:vw1); (0,1,1,0:vw1); (0,2,1,0:vw1); (0,3,1,0:vw1); (0,0,2,0:vw1); (0,1,2,0:vw1); (0,2,2,0:vw1); (0,3,2,0:vw1); (0,0,3,0:vw1); (0,1,3,0:vw1); (0,2,3,0:vw1); (0,3,3,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
s_mul_i32 s8, 128, s[sgprWorkGroup0]               // wgp0 * MT0
v_sub_u32 v23, v10, s8
v_lshlrev_b32 v23, 0x2, v23                        // Bias address scaled by BPE
s_waitcnt lgkmcnt(0)                               // Wait for LDS write
s_barrier                                          // LDS write barrier
ds_read_b32 v40, v23 offset:0                      // load Bias
ds_read_b32 v41, v23 offset:1024                   // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
ds_read_b32 v42, v23 offset:128                    // load Bias
ds_read_b32 v43, v23 offset:1152                   // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
ds_read_b32 v44, v23 offset:256                    // load Bias
ds_read_b32 v45, v23 offset:1280                   // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
ds_read_b32 v46, v23 offset:384                    // load Bias
ds_read_b32 v47, v23 offset:1408                   // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
/* (d1,vc1,d0,vc0)=(0,1,2,0) */
/* (d1,vc1,d0,vc0)=(0,1,3,0) */
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
/* (d1,vc1,d0,vc0)=(0,2,1,0) */
/* (d1,vc1,d0,vc0)=(0,2,2,0) */
/* (d1,vc1,d0,vc0)=(0,2,3,0) */
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
/* (d1,vc1,d0,vc0)=(0,3,1,0) */
/* (d1,vc1,d0,vc0)=(0,3,2,0) */
/* (d1,vc1,d0,vc0)=(0,3,3,0) */
v_add_lshl_u32 v21, v13, v10, 0x1                  // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=10, coord0Vgpr=10
v_accvgpr_read_b32 v[vgprValuC+24], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+25], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+26], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+27], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+28], acc16          // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+29], acc20          // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+30], acc24          // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+31], acc28          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+32], acc32          // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+33], acc36          // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+34], acc40          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+35], acc44          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+36], acc48          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+37], acc52          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+38], acc56          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+39], acc60          // copy acc to vreg[15]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 0, 1, 0), (0, 1, 1, 0), (0, 2, 1, 0), (0, 3, 1, 0), (0, 0, 2, 0), (0, 1, 2, 0), (0, 2, 2, 0), (0, 3, 2, 0), (0, 0, 3, 0), (0, 1, 3, 0), (0, 2, 3, 0), (0, 3, 3, 0)] */
v_pk_mul_f32 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+24:vgprValuC+24+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+26:vgprValuC+26+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+28:vgprValuC+28+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+30:vgprValuC+30+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+32:vgprValuC+32+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+34:vgprValuC+34+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+36:vgprValuC+36+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+36:vgprValuC+36+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+38:vgprValuC+38+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+38:vgprValuC+38+1] op_sel_hi:[0,1,1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */
v_mov_b32 v18, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v19, 0x7fff0000                          // fp32 Nan
v_mov_b32 v20, 0x7fff                              // rounding bias for bfloat16

s_waitcnt lgkmcnt(6)                               // dscnt(6) = 8 - 1 (bias) - 1 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+24], v41, v[vgprValuC+24]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+24], v40, v[vgprValuC+24]    // C += bias
v_cvt_pk_bf16_f32 v24, v[vgprValuC+24], v[vgprValuC+24] // convert C to bf16 in gwvw==1
buffer_store_short v24, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt lgkmcnt(4)                               // dscnt(4) = 8 - 2 (bias) - 2 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+25], v43, v[vgprValuC+25]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+25], v42, v[vgprValuC+25]    // C += bias
v_cvt_pk_bf16_f32 v25, v[vgprValuC+25], v[vgprValuC+25] // convert C to bf16 in gwvw==1
buffer_store_short v25, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt lgkmcnt(2)                               // dscnt(2) = 8 - 3 (bias) - 3 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+26], v45, v[vgprValuC+26]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+26], v44, v[vgprValuC+26]    // C += bias
v_cvt_pk_bf16_f32 v26, v[vgprValuC+26], v[vgprValuC+26] // convert C to bf16 in gwvw==1
buffer_store_short v26, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt lgkmcnt(0)                               // dscnt(0) = 8 - 4 (bias) - 4 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+27], v47, v[vgprValuC+27]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+27], v46, v[vgprValuC+27]    // C += bias
v_cvt_pk_bf16_f32 v27, v[vgprValuC+27], v[vgprValuC+27] // convert C to bf16 in gwvw==1
buffer_store_short v27, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D
v_mul_f32 v[vgprValuC+28], v41, v[vgprValuC+28]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+28], v40, v[vgprValuC+28]    // C += bias
v_cvt_pk_bf16_f32 v28, v[vgprValuC+28], v[vgprValuC+28] // convert C to bf16 in gwvw==1
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v28, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+29], v43, v[vgprValuC+29]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+29], v42, v[vgprValuC+29]    // C += bias
v_cvt_pk_bf16_f32 v29, v[vgprValuC+29], v[vgprValuC+29] // convert C to bf16 in gwvw==1
buffer_store_short v29, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_mul_f32 v[vgprValuC+30], v45, v[vgprValuC+30]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+30], v44, v[vgprValuC+30]    // C += bias
v_cvt_pk_bf16_f32 v30, v[vgprValuC+30], v[vgprValuC+30] // convert C to bf16 in gwvw==1
buffer_store_short v30, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_mul_f32 v[vgprValuC+31], v47, v[vgprValuC+31]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+31], v46, v[vgprValuC+31]    // C += bias
v_cvt_pk_bf16_f32 v31, v[vgprValuC+31], v[vgprValuC+31] // convert C to bf16 in gwvw==1
buffer_store_short v31, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D
v_mul_f32 v[vgprValuC+32], v41, v[vgprValuC+32]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+32], v40, v[vgprValuC+32]    // C += bias
v_cvt_pk_bf16_f32 v32, v[vgprValuC+32], v[vgprValuC+32] // convert C to bf16 in gwvw==1
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v32, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+33], v43, v[vgprValuC+33]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+33], v42, v[vgprValuC+33]    // C += bias
v_cvt_pk_bf16_f32 v33, v[vgprValuC+33], v[vgprValuC+33] // convert C to bf16 in gwvw==1
buffer_store_short v33, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_mul_f32 v[vgprValuC+34], v45, v[vgprValuC+34]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+34], v44, v[vgprValuC+34]    // C += bias
v_cvt_pk_bf16_f32 v34, v[vgprValuC+34], v[vgprValuC+34] // convert C to bf16 in gwvw==1
buffer_store_short v34, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_mul_f32 v[vgprValuC+35], v47, v[vgprValuC+35]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+35], v46, v[vgprValuC+35]    // C += bias
v_cvt_pk_bf16_f32 v35, v[vgprValuC+35], v[vgprValuC+35] // convert C to bf16 in gwvw==1
buffer_store_short v35, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D
v_mul_f32 v[vgprValuC+36], v41, v[vgprValuC+36]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+36], v40, v[vgprValuC+36]    // C += bias
v_cvt_pk_bf16_f32 v36, v[vgprValuC+36], v[vgprValuC+36] // convert C to bf16 in gwvw==1
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v36, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+37], v43, v[vgprValuC+37]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+37], v42, v[vgprValuC+37]    // C += bias
v_cvt_pk_bf16_f32 v37, v[vgprValuC+37], v[vgprValuC+37] // convert C to bf16 in gwvw==1
buffer_store_short v37, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_mul_f32 v[vgprValuC+38], v45, v[vgprValuC+38]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+38], v44, v[vgprValuC+38]    // C += bias
v_cvt_pk_bf16_f32 v38, v[vgprValuC+38], v[vgprValuC+38] // convert C to bf16 in gwvw==1
buffer_store_short v38, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_mul_f32 v[vgprValuC+39], v47, v[vgprValuC+39]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+39], v46, v[vgprValuC+39]    // C += bias
v_cvt_pk_bf16_f32 v39, v[vgprValuC+39], v[vgprValuC+39] // convert C to bf16 in gwvw==1
buffer_store_short v39, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #1 (d1,d0,vc1,vc0) = */
/*    (0,0,4,0:vw1); (0,1,4,0:vw1); (0,2,4,0:vw1); (0,3,4,0:vw1); (0,0,5,0:vw1); (0,1,5,0:vw1); (0,2,5,0:vw1); (0,3,5,0:vw1); (0,0,6,0:vw1); (0,1,6,0:vw1); (0,2,6,0:vw1); (0,3,6,0:vw1); (0,0,7,0:vw1); (0,1,7,0:vw1); (0,2,7,0:vw1); (0,3,7,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,4,0,0) */
ds_read_b32 v40, v23 offset:0                      // load Bias
ds_read_b32 v41, v23 offset:1024                   // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,4,1,0) */
ds_read_b32 v42, v23 offset:128                    // load Bias
ds_read_b32 v43, v23 offset:1152                   // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,4,2,0) */
ds_read_b32 v44, v23 offset:256                    // load Bias
ds_read_b32 v45, v23 offset:1280                   // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,4,3,0) */
ds_read_b32 v46, v23 offset:384                    // load Bias
ds_read_b32 v47, v23 offset:1408                   // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,5,0,0) */
/* (d1,vc1,d0,vc0)=(0,5,1,0) */
/* (d1,vc1,d0,vc0)=(0,5,2,0) */
/* (d1,vc1,d0,vc0)=(0,5,3,0) */
/* (d1,vc1,d0,vc0)=(0,6,0,0) */
/* (d1,vc1,d0,vc0)=(0,6,1,0) */
/* (d1,vc1,d0,vc0)=(0,6,2,0) */
/* (d1,vc1,d0,vc0)=(0,6,3,0) */
/* (d1,vc1,d0,vc0)=(0,7,0,0) */
/* (d1,vc1,d0,vc0)=(0,7,1,0) */
/* (d1,vc1,d0,vc0)=(0,7,2,0) */
/* (d1,vc1,d0,vc0)=(0,7,3,0) */
v_accvgpr_read_b32 v[vgprValuC+24], acc1           // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+25], acc5           // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+26], acc9           // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+27], acc13          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+28], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+29], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+30], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+31], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+32], acc33          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+33], acc37          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+34], acc41          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+35], acc45          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+36], acc49          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+37], acc53          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+38], acc57          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+39], acc61          // copy acc to vreg[31]

/* rC *= alpha batchElements=[(0, 0, 4, 0), (0, 1, 4, 0), (0, 2, 4, 0), (0, 3, 4, 0), (0, 0, 5, 0), (0, 1, 5, 0), (0, 2, 5, 0), (0, 3, 5, 0), (0, 0, 6, 0), (0, 1, 6, 0), (0, 2, 6, 0), (0, 3, 6, 0), (0, 0, 7, 0), (0, 1, 7, 0), (0, 2, 7, 0), (0, 3, 7, 0)] */
v_pk_mul_f32 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+24:vgprValuC+24+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+26:vgprValuC+26+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+28:vgprValuC+28+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+30:vgprValuC+30+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+32:vgprValuC+32+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+34:vgprValuC+34+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+36:vgprValuC+36+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+36:vgprValuC+36+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+38:vgprValuC+38+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+38:vgprValuC+38+1] op_sel_hi:[0,1,1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */
v_mov_b32 v18, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v19, 0x7fff0000                          // fp32 Nan
v_mov_b32 v20, 0x7fff                              // rounding bias for bfloat16

s_waitcnt lgkmcnt(6)                               // dscnt(6) = 8 - 1 (bias) - 1 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+24], v41, v[vgprValuC+24]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+24], v40, v[vgprValuC+24]    // C += bias
v_cvt_pk_bf16_f32 v24, v[vgprValuC+24], v[vgprValuC+24] // convert C to bf16 in gwvw==1
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v24, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt lgkmcnt(4)                               // dscnt(4) = 8 - 2 (bias) - 2 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+25], v43, v[vgprValuC+25]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+25], v42, v[vgprValuC+25]    // C += bias
v_cvt_pk_bf16_f32 v25, v[vgprValuC+25], v[vgprValuC+25] // convert C to bf16 in gwvw==1
buffer_store_short v25, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt lgkmcnt(2)                               // dscnt(2) = 8 - 3 (bias) - 3 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+26], v45, v[vgprValuC+26]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+26], v44, v[vgprValuC+26]    // C += bias
v_cvt_pk_bf16_f32 v26, v[vgprValuC+26], v[vgprValuC+26] // convert C to bf16 in gwvw==1
buffer_store_short v26, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt lgkmcnt(0)                               // dscnt(0) = 8 - 4 (bias) - 4 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+27], v47, v[vgprValuC+27]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+27], v46, v[vgprValuC+27]    // C += bias
v_cvt_pk_bf16_f32 v27, v[vgprValuC+27], v[vgprValuC+27] // convert C to bf16 in gwvw==1
buffer_store_short v27, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D
v_mul_f32 v[vgprValuC+28], v41, v[vgprValuC+28]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+28], v40, v[vgprValuC+28]    // C += bias
v_cvt_pk_bf16_f32 v28, v[vgprValuC+28], v[vgprValuC+28] // convert C to bf16 in gwvw==1
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v28, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+29], v43, v[vgprValuC+29]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+29], v42, v[vgprValuC+29]    // C += bias
v_cvt_pk_bf16_f32 v29, v[vgprValuC+29], v[vgprValuC+29] // convert C to bf16 in gwvw==1
buffer_store_short v29, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_mul_f32 v[vgprValuC+30], v45, v[vgprValuC+30]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+30], v44, v[vgprValuC+30]    // C += bias
v_cvt_pk_bf16_f32 v30, v[vgprValuC+30], v[vgprValuC+30] // convert C to bf16 in gwvw==1
buffer_store_short v30, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_mul_f32 v[vgprValuC+31], v47, v[vgprValuC+31]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+31], v46, v[vgprValuC+31]    // C += bias
v_cvt_pk_bf16_f32 v31, v[vgprValuC+31], v[vgprValuC+31] // convert C to bf16 in gwvw==1
buffer_store_short v31, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D
v_mul_f32 v[vgprValuC+32], v41, v[vgprValuC+32]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+32], v40, v[vgprValuC+32]    // C += bias
v_cvt_pk_bf16_f32 v32, v[vgprValuC+32], v[vgprValuC+32] // convert C to bf16 in gwvw==1
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v32, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+33], v43, v[vgprValuC+33]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+33], v42, v[vgprValuC+33]    // C += bias
v_cvt_pk_bf16_f32 v33, v[vgprValuC+33], v[vgprValuC+33] // convert C to bf16 in gwvw==1
buffer_store_short v33, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_mul_f32 v[vgprValuC+34], v45, v[vgprValuC+34]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+34], v44, v[vgprValuC+34]    // C += bias
v_cvt_pk_bf16_f32 v34, v[vgprValuC+34], v[vgprValuC+34] // convert C to bf16 in gwvw==1
buffer_store_short v34, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_mul_f32 v[vgprValuC+35], v47, v[vgprValuC+35]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+35], v46, v[vgprValuC+35]    // C += bias
v_cvt_pk_bf16_f32 v35, v[vgprValuC+35], v[vgprValuC+35] // convert C to bf16 in gwvw==1
buffer_store_short v35, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D
v_mul_f32 v[vgprValuC+36], v41, v[vgprValuC+36]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+36], v40, v[vgprValuC+36]    // C += bias
v_cvt_pk_bf16_f32 v36, v[vgprValuC+36], v[vgprValuC+36] // convert C to bf16 in gwvw==1
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v36, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+37], v43, v[vgprValuC+37]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+37], v42, v[vgprValuC+37]    // C += bias
v_cvt_pk_bf16_f32 v37, v[vgprValuC+37], v[vgprValuC+37] // convert C to bf16 in gwvw==1
buffer_store_short v37, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_mul_f32 v[vgprValuC+38], v45, v[vgprValuC+38]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+38], v44, v[vgprValuC+38]    // C += bias
v_cvt_pk_bf16_f32 v38, v[vgprValuC+38], v[vgprValuC+38] // convert C to bf16 in gwvw==1
buffer_store_short v38, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_mul_f32 v[vgprValuC+39], v47, v[vgprValuC+39]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+39], v46, v[vgprValuC+39]    // C += bias
v_cvt_pk_bf16_f32 v39, v[vgprValuC+39], v[vgprValuC+39] // convert C to bf16 in gwvw==1
buffer_store_short v39, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #2 (d1,d0,vc1,vc0) = */
/*    (0,0,8,0:vw1); (0,1,8,0:vw1); (0,2,8,0:vw1); (0,3,8,0:vw1); (0,0,9,0:vw1); (0,1,9,0:vw1); (0,2,9,0:vw1); (0,3,9,0:vw1); (0,0,10,0:vw1); (0,1,10,0:vw1); (0,2,10,0:vw1); (0,3,10,0:vw1); (0,0,11,0:vw1); (0,1,11,0:vw1); (0,2,11,0:vw1); (0,3,11,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,8,0,0) */
ds_read_b32 v40, v23 offset:0                      // load Bias
ds_read_b32 v41, v23 offset:1024                   // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,8,1,0) */
ds_read_b32 v42, v23 offset:128                    // load Bias
ds_read_b32 v43, v23 offset:1152                   // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,8,2,0) */
ds_read_b32 v44, v23 offset:256                    // load Bias
ds_read_b32 v45, v23 offset:1280                   // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,8,3,0) */
ds_read_b32 v46, v23 offset:384                    // load Bias
ds_read_b32 v47, v23 offset:1408                   // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,9,0,0) */
/* (d1,vc1,d0,vc0)=(0,9,1,0) */
/* (d1,vc1,d0,vc0)=(0,9,2,0) */
/* (d1,vc1,d0,vc0)=(0,9,3,0) */
/* (d1,vc1,d0,vc0)=(0,10,0,0) */
/* (d1,vc1,d0,vc0)=(0,10,1,0) */
/* (d1,vc1,d0,vc0)=(0,10,2,0) */
/* (d1,vc1,d0,vc0)=(0,10,3,0) */
/* (d1,vc1,d0,vc0)=(0,11,0,0) */
/* (d1,vc1,d0,vc0)=(0,11,1,0) */
/* (d1,vc1,d0,vc0)=(0,11,2,0) */
/* (d1,vc1,d0,vc0)=(0,11,3,0) */
v_accvgpr_read_b32 v[vgprValuC+24], acc2           // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+25], acc6           // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+26], acc10          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+27], acc14          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+28], acc18          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+29], acc22          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+30], acc26          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+31], acc30          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+32], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+33], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+34], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+35], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+36], acc50          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+37], acc54          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+38], acc58          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+39], acc62          // copy acc to vreg[47]

/* rC *= alpha batchElements=[(0, 0, 8, 0), (0, 1, 8, 0), (0, 2, 8, 0), (0, 3, 8, 0), (0, 0, 9, 0), (0, 1, 9, 0), (0, 2, 9, 0), (0, 3, 9, 0), (0, 0, 10, 0), (0, 1, 10, 0), (0, 2, 10, 0), (0, 3, 10, 0), (0, 0, 11, 0), (0, 1, 11, 0), (0, 2, 11, 0), (0, 3, 11, 0)] */
v_pk_mul_f32 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+24:vgprValuC+24+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+26:vgprValuC+26+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+28:vgprValuC+28+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+30:vgprValuC+30+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+32:vgprValuC+32+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+34:vgprValuC+34+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+36:vgprValuC+36+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+36:vgprValuC+36+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+38:vgprValuC+38+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+38:vgprValuC+38+1] op_sel_hi:[0,1,1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */
v_mov_b32 v18, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v19, 0x7fff0000                          // fp32 Nan
v_mov_b32 v20, 0x7fff                              // rounding bias for bfloat16

s_waitcnt lgkmcnt(6)                               // dscnt(6) = 8 - 1 (bias) - 1 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+24], v41, v[vgprValuC+24]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+24], v40, v[vgprValuC+24]    // C += bias
v_cvt_pk_bf16_f32 v24, v[vgprValuC+24], v[vgprValuC+24] // convert C to bf16 in gwvw==1
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v24, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt lgkmcnt(4)                               // dscnt(4) = 8 - 2 (bias) - 2 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+25], v43, v[vgprValuC+25]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+25], v42, v[vgprValuC+25]    // C += bias
v_cvt_pk_bf16_f32 v25, v[vgprValuC+25], v[vgprValuC+25] // convert C to bf16 in gwvw==1
buffer_store_short v25, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt lgkmcnt(2)                               // dscnt(2) = 8 - 3 (bias) - 3 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+26], v45, v[vgprValuC+26]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+26], v44, v[vgprValuC+26]    // C += bias
v_cvt_pk_bf16_f32 v26, v[vgprValuC+26], v[vgprValuC+26] // convert C to bf16 in gwvw==1
buffer_store_short v26, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt lgkmcnt(0)                               // dscnt(0) = 8 - 4 (bias) - 4 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+27], v47, v[vgprValuC+27]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+27], v46, v[vgprValuC+27]    // C += bias
v_cvt_pk_bf16_f32 v27, v[vgprValuC+27], v[vgprValuC+27] // convert C to bf16 in gwvw==1
buffer_store_short v27, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D
v_mul_f32 v[vgprValuC+28], v41, v[vgprValuC+28]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+28], v40, v[vgprValuC+28]    // C += bias
v_cvt_pk_bf16_f32 v28, v[vgprValuC+28], v[vgprValuC+28] // convert C to bf16 in gwvw==1
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v28, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+29], v43, v[vgprValuC+29]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+29], v42, v[vgprValuC+29]    // C += bias
v_cvt_pk_bf16_f32 v29, v[vgprValuC+29], v[vgprValuC+29] // convert C to bf16 in gwvw==1
buffer_store_short v29, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_mul_f32 v[vgprValuC+30], v45, v[vgprValuC+30]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+30], v44, v[vgprValuC+30]    // C += bias
v_cvt_pk_bf16_f32 v30, v[vgprValuC+30], v[vgprValuC+30] // convert C to bf16 in gwvw==1
buffer_store_short v30, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_mul_f32 v[vgprValuC+31], v47, v[vgprValuC+31]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+31], v46, v[vgprValuC+31]    // C += bias
v_cvt_pk_bf16_f32 v31, v[vgprValuC+31], v[vgprValuC+31] // convert C to bf16 in gwvw==1
buffer_store_short v31, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D
v_mul_f32 v[vgprValuC+32], v41, v[vgprValuC+32]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+32], v40, v[vgprValuC+32]    // C += bias
v_cvt_pk_bf16_f32 v32, v[vgprValuC+32], v[vgprValuC+32] // convert C to bf16 in gwvw==1
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v32, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+33], v43, v[vgprValuC+33]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+33], v42, v[vgprValuC+33]    // C += bias
v_cvt_pk_bf16_f32 v33, v[vgprValuC+33], v[vgprValuC+33] // convert C to bf16 in gwvw==1
buffer_store_short v33, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_mul_f32 v[vgprValuC+34], v45, v[vgprValuC+34]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+34], v44, v[vgprValuC+34]    // C += bias
v_cvt_pk_bf16_f32 v34, v[vgprValuC+34], v[vgprValuC+34] // convert C to bf16 in gwvw==1
buffer_store_short v34, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_mul_f32 v[vgprValuC+35], v47, v[vgprValuC+35]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+35], v46, v[vgprValuC+35]    // C += bias
v_cvt_pk_bf16_f32 v35, v[vgprValuC+35], v[vgprValuC+35] // convert C to bf16 in gwvw==1
buffer_store_short v35, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D
v_mul_f32 v[vgprValuC+36], v41, v[vgprValuC+36]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+36], v40, v[vgprValuC+36]    // C += bias
v_cvt_pk_bf16_f32 v36, v[vgprValuC+36], v[vgprValuC+36] // convert C to bf16 in gwvw==1
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v36, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+37], v43, v[vgprValuC+37]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+37], v42, v[vgprValuC+37]    // C += bias
v_cvt_pk_bf16_f32 v37, v[vgprValuC+37], v[vgprValuC+37] // convert C to bf16 in gwvw==1
buffer_store_short v37, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_mul_f32 v[vgprValuC+38], v45, v[vgprValuC+38]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+38], v44, v[vgprValuC+38]    // C += bias
v_cvt_pk_bf16_f32 v38, v[vgprValuC+38], v[vgprValuC+38] // convert C to bf16 in gwvw==1
buffer_store_short v38, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_mul_f32 v[vgprValuC+39], v47, v[vgprValuC+39]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+39], v46, v[vgprValuC+39]    // C += bias
v_cvt_pk_bf16_f32 v39, v[vgprValuC+39], v[vgprValuC+39] // convert C to bf16 in gwvw==1
buffer_store_short v39, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #3 (d1,d0,vc1,vc0) = */
/*    (0,0,12,0:vw1); (0,1,12,0:vw1); (0,2,12,0:vw1); (0,3,12,0:vw1); (0,0,13,0:vw1); (0,1,13,0:vw1); (0,2,13,0:vw1); (0,3,13,0:vw1); (0,0,14,0:vw1); (0,1,14,0:vw1); (0,2,14,0:vw1); (0,3,14,0:vw1); (0,0,15,0:vw1); (0,1,15,0:vw1); (0,2,15,0:vw1); (0,3,15,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,12,0,0) */
ds_read_b32 v40, v23 offset:0                      // load Bias
ds_read_b32 v41, v23 offset:1024                   // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,12,1,0) */
ds_read_b32 v42, v23 offset:128                    // load Bias
ds_read_b32 v43, v23 offset:1152                   // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,12,2,0) */
ds_read_b32 v44, v23 offset:256                    // load Bias
ds_read_b32 v45, v23 offset:1280                   // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,12,3,0) */
ds_read_b32 v46, v23 offset:384                    // load Bias
ds_read_b32 v47, v23 offset:1408                   // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,13,0,0) */
/* (d1,vc1,d0,vc0)=(0,13,1,0) */
/* (d1,vc1,d0,vc0)=(0,13,2,0) */
/* (d1,vc1,d0,vc0)=(0,13,3,0) */
/* (d1,vc1,d0,vc0)=(0,14,0,0) */
/* (d1,vc1,d0,vc0)=(0,14,1,0) */
/* (d1,vc1,d0,vc0)=(0,14,2,0) */
/* (d1,vc1,d0,vc0)=(0,14,3,0) */
/* (d1,vc1,d0,vc0)=(0,15,0,0) */
/* (d1,vc1,d0,vc0)=(0,15,1,0) */
/* (d1,vc1,d0,vc0)=(0,15,2,0) */
/* (d1,vc1,d0,vc0)=(0,15,3,0) */
v_accvgpr_read_b32 v[vgprValuC+24], acc3           // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+25], acc7           // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+26], acc11          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+27], acc15          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+28], acc19          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+29], acc23          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+30], acc27          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+31], acc31          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+32], acc35          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+33], acc39          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+34], acc43          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+35], acc47          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+36], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+37], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+38], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+39], acc63          // copy acc to vreg[63]

/* rC *= alpha batchElements=[(0, 0, 12, 0), (0, 1, 12, 0), (0, 2, 12, 0), (0, 3, 12, 0), (0, 0, 13, 0), (0, 1, 13, 0), (0, 2, 13, 0), (0, 3, 13, 0), (0, 0, 14, 0), (0, 1, 14, 0), (0, 2, 14, 0), (0, 3, 14, 0), (0, 0, 15, 0), (0, 1, 15, 0), (0, 2, 15, 0), (0, 3, 15, 0)] */
v_pk_mul_f32 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+24:vgprValuC+24+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+26:vgprValuC+26+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+28:vgprValuC+28+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+30:vgprValuC+30+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+32:vgprValuC+32+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+34:vgprValuC+34+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+36:vgprValuC+36+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+36:vgprValuC+36+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+38:vgprValuC+38+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+38:vgprValuC+38+1] op_sel_hi:[0,1,1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */
v_mov_b32 v18, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v19, 0x7fff0000                          // fp32 Nan
v_mov_b32 v20, 0x7fff                              // rounding bias for bfloat16

s_waitcnt lgkmcnt(6)                               // dscnt(6) = 8 - 1 (bias) - 1 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+24], v41, v[vgprValuC+24]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+24], v40, v[vgprValuC+24]    // C += bias
v_cvt_pk_bf16_f32 v24, v[vgprValuC+24], v[vgprValuC+24] // convert C to bf16 in gwvw==1
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v24, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt lgkmcnt(4)                               // dscnt(4) = 8 - 2 (bias) - 2 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+25], v43, v[vgprValuC+25]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+25], v42, v[vgprValuC+25]    // C += bias
v_cvt_pk_bf16_f32 v25, v[vgprValuC+25], v[vgprValuC+25] // convert C to bf16 in gwvw==1
buffer_store_short v25, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt lgkmcnt(2)                               // dscnt(2) = 8 - 3 (bias) - 3 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+26], v45, v[vgprValuC+26]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+26], v44, v[vgprValuC+26]    // C += bias
v_cvt_pk_bf16_f32 v26, v[vgprValuC+26], v[vgprValuC+26] // convert C to bf16 in gwvw==1
buffer_store_short v26, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt lgkmcnt(0)                               // dscnt(0) = 8 - 4 (bias) - 4 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+27], v47, v[vgprValuC+27]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+27], v46, v[vgprValuC+27]    // C += bias
v_cvt_pk_bf16_f32 v27, v[vgprValuC+27], v[vgprValuC+27] // convert C to bf16 in gwvw==1
buffer_store_short v27, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D
v_mul_f32 v[vgprValuC+28], v41, v[vgprValuC+28]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+28], v40, v[vgprValuC+28]    // C += bias
v_cvt_pk_bf16_f32 v28, v[vgprValuC+28], v[vgprValuC+28] // convert C to bf16 in gwvw==1
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v28, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+29], v43, v[vgprValuC+29]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+29], v42, v[vgprValuC+29]    // C += bias
v_cvt_pk_bf16_f32 v29, v[vgprValuC+29], v[vgprValuC+29] // convert C to bf16 in gwvw==1
buffer_store_short v29, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_mul_f32 v[vgprValuC+30], v45, v[vgprValuC+30]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+30], v44, v[vgprValuC+30]    // C += bias
v_cvt_pk_bf16_f32 v30, v[vgprValuC+30], v[vgprValuC+30] // convert C to bf16 in gwvw==1
buffer_store_short v30, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_mul_f32 v[vgprValuC+31], v47, v[vgprValuC+31]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+31], v46, v[vgprValuC+31]    // C += bias
v_cvt_pk_bf16_f32 v31, v[vgprValuC+31], v[vgprValuC+31] // convert C to bf16 in gwvw==1
buffer_store_short v31, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D
v_mul_f32 v[vgprValuC+32], v41, v[vgprValuC+32]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+32], v40, v[vgprValuC+32]    // C += bias
v_cvt_pk_bf16_f32 v32, v[vgprValuC+32], v[vgprValuC+32] // convert C to bf16 in gwvw==1
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v32, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+33], v43, v[vgprValuC+33]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+33], v42, v[vgprValuC+33]    // C += bias
v_cvt_pk_bf16_f32 v33, v[vgprValuC+33], v[vgprValuC+33] // convert C to bf16 in gwvw==1
buffer_store_short v33, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_mul_f32 v[vgprValuC+34], v45, v[vgprValuC+34]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+34], v44, v[vgprValuC+34]    // C += bias
v_cvt_pk_bf16_f32 v34, v[vgprValuC+34], v[vgprValuC+34] // convert C to bf16 in gwvw==1
buffer_store_short v34, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_mul_f32 v[vgprValuC+35], v47, v[vgprValuC+35]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+35], v46, v[vgprValuC+35]    // C += bias
v_cvt_pk_bf16_f32 v35, v[vgprValuC+35], v[vgprValuC+35] // convert C to bf16 in gwvw==1
buffer_store_short v35, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D
v_mul_f32 v[vgprValuC+36], v41, v[vgprValuC+36]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+36], v40, v[vgprValuC+36]    // C += bias
v_cvt_pk_bf16_f32 v36, v[vgprValuC+36], v[vgprValuC+36] // convert C to bf16 in gwvw==1
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v36, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+37], v43, v[vgprValuC+37]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+37], v42, v[vgprValuC+37]    // C += bias
v_cvt_pk_bf16_f32 v37, v[vgprValuC+37], v[vgprValuC+37] // convert C to bf16 in gwvw==1
buffer_store_short v37, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_mul_f32 v[vgprValuC+38], v45, v[vgprValuC+38]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+38], v44, v[vgprValuC+38]    // C += bias
v_cvt_pk_bf16_f32 v38, v[vgprValuC+38], v[vgprValuC+38] // convert C to bf16 in gwvw==1
buffer_store_short v38, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
v_mul_f32 v[vgprValuC+39], v47, v[vgprValuC+39]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+39], v46, v[vgprValuC+39]    // C += bias
v_cvt_pk_bf16_f32 v39, v[vgprValuC+39], v[vgprValuC+39] // convert C to bf16 in gwvw==1
buffer_store_short v39, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_E1_1:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=16 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,0,1,0:vw1); (0,1,1,0:vw1); (0,2,1,0:vw1); (0,3,1,0:vw1); (0,0,2,0:vw1); (0,1,2,0:vw1); (0,2,2,0:vw1); (0,3,2,0:vw1); (0,0,3,0:vw1); (0,1,3,0:vw1); (0,2,3,0:vw1); (0,3,3,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v16, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v40, v10, s70
v_lshlrev_b32 v40, 0x2, v40                        // Bias address scaled by BPE
s_waitcnt lgkmcnt(0)                               // Wait for LDS write
s_barrier                                          // LDS write barrier
ds_read_b32 v37, v40 offset:0                      // load Bias
ds_read_b32 v38, v40 offset:1024                   // load scaleAlpha
v_add_lshl_u32 v39, v13, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v39, v16, v39, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v44, v14, s70
v_lshlrev_b32 v44, 0x2, v44                        // Bias address scaled by BPE
ds_read_b32 v41, v44 offset:0                      // load Bias
ds_read_b32 v42, v44 offset:1024                   // load scaleAlpha
v_add_lshl_u32 v43, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v43, v16, v43, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v48, v14, s70
v_lshlrev_b32 v48, 0x2, v48                        // Bias address scaled by BPE
ds_read_b32 v45, v48 offset:0                      // load Bias
ds_read_b32 v46, v48 offset:1024                   // load scaleAlpha
v_add_lshl_u32 v47, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v47, v16, v47, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v52, v14, s70
v_lshlrev_b32 v52, 0x2, v52                        // Bias address scaled by BPE
ds_read_b32 v49, v52 offset:0                      // load Bias
ds_read_b32 v50, v52 offset:1024                   // load scaleAlpha
v_add_lshl_u32 v51, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v51, v16, v51, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v54, v10, s70
v_lshlrev_b32 v54, 0x2, v54                        // Bias address scaled by BPE
v_add_lshl_u32 v53, v13, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v53, v16, v53, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v56, v14, s70
v_lshlrev_b32 v56, 0x2, v56                        // Bias address scaled by BPE
v_add_lshl_u32 v55, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v55, v16, v55, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v58, v14, s70
v_lshlrev_b32 v58, 0x2, v58                        // Bias address scaled by BPE
v_add_lshl_u32 v57, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v57, v16, v57, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v60, v14, s70
v_lshlrev_b32 v60, 0x2, v60                        // Bias address scaled by BPE
v_add_lshl_u32 v59, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v59, v16, v59, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v62, v10, s70
v_lshlrev_b32 v62, 0x2, v62                        // Bias address scaled by BPE
v_add_lshl_u32 v61, v13, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v61, v16, v61, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v64, v14, s70
v_lshlrev_b32 v64, 0x2, v64                        // Bias address scaled by BPE
v_add_lshl_u32 v63, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v63, v16, v63, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v66, v14, s70
v_lshlrev_b32 v66, 0x2, v66                        // Bias address scaled by BPE
v_add_lshl_u32 v65, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v65, v16, v65, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v68, v14, s70
v_lshlrev_b32 v68, 0x2, v68                        // Bias address scaled by BPE
v_add_lshl_u32 v67, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v67, v16, v67, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v70, v10, s70
v_lshlrev_b32 v70, 0x2, v70                        // Bias address scaled by BPE
v_add_lshl_u32 v69, v13, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v69, v16, v69, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v72, v14, s70
v_lshlrev_b32 v72, 0x2, v72                        // Bias address scaled by BPE
v_add_lshl_u32 v71, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v71, v16, v71, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v75, v14, s70
v_lshlrev_b32 v75, 0x2, v75                        // Bias address scaled by BPE
v_add_lshl_u32 v73, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v73, v16, v73, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v77, v14, s70
v_lshlrev_b32 v77, 0x2, v77                        // Bias address scaled by BPE
v_add_lshl_u32 v76, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v76, v16, v76, s[74:75]              // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+21], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+22], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+23], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+24], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+25], acc16          // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+26], acc20          // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+27], acc24          // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+28], acc28          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+29], acc32          // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+30], acc36          // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+31], acc40          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+32], acc44          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+33], acc48          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+34], acc52          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+35], acc56          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+36], acc60          // copy acc to vreg[15]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 0, 1, 0), (0, 1, 1, 0), (0, 2, 1, 0), (0, 3, 1, 0), (0, 0, 2, 0), (0, 1, 2, 0), (0, 2, 2, 0), (0, 3, 2, 0), (0, 0, 3, 0), (0, 1, 3, 0), (0, 2, 3, 0), (0, 3, 3, 0)] */
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+21] // *= alpha
v_pk_mul_f32 v[vgprValuC+22:vgprValuC+22+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+22:vgprValuC+22+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+24:vgprValuC+24+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+26:vgprValuC+26+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+28:vgprValuC+28+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+30:vgprValuC+30+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+32:vgprValuC+32+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+34:vgprValuC+34+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+36], s[sgprAlpha], v[vgprValuC+36] // *= alpha
s_waitcnt lgkmcnt(0)                               // wait for Bias LDS, ScaleAlphaVec

/* apply mask, calc new C and issue writes */
v_mov_b32 v18, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v19, 0x7fff0000                          // fp32 Nan
v_mov_b32 v20, 0x7fff                              // rounding bias for bfloat16
v_mul_f32 v[vgprValuC+21], v38, v[vgprValuC+21]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+21], v37, v[vgprValuC+21]    // C += bias
v_cvt_pk_bf16_f32 v21, v[vgprValuC+21], v[vgprValuC+21] // convert C to bf16 in gwvw==1
buffer_store_short v21, v39, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+22], v42, v[vgprValuC+22]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+22], v41, v[vgprValuC+22]    // C += bias
v_cvt_pk_bf16_f32 v22, v[vgprValuC+22], v[vgprValuC+22] // convert C to bf16 in gwvw==1
buffer_store_short v22, v43, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+23], v46, v[vgprValuC+23]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+23], v45, v[vgprValuC+23]    // C += bias
v_cvt_pk_bf16_f32 v23, v[vgprValuC+23], v[vgprValuC+23] // convert C to bf16 in gwvw==1
buffer_store_short v23, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+24], v50, v[vgprValuC+24]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+24], v49, v[vgprValuC+24]    // C += bias
v_cvt_pk_bf16_f32 v24, v[vgprValuC+24], v[vgprValuC+24] // convert C to bf16 in gwvw==1
buffer_store_short v24, v51, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+25], v38, v[vgprValuC+25]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+25], v37, v[vgprValuC+25]    // C += bias
v_cvt_pk_bf16_f32 v25, v[vgprValuC+25], v[vgprValuC+25] // convert C to bf16 in gwvw==1
buffer_store_short v25, v53, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+26], v42, v[vgprValuC+26]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+26], v41, v[vgprValuC+26]    // C += bias
v_cvt_pk_bf16_f32 v26, v[vgprValuC+26], v[vgprValuC+26] // convert C to bf16 in gwvw==1
buffer_store_short v26, v55, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+27], v46, v[vgprValuC+27]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+27], v45, v[vgprValuC+27]    // C += bias
v_cvt_pk_bf16_f32 v27, v[vgprValuC+27], v[vgprValuC+27] // convert C to bf16 in gwvw==1
buffer_store_short v27, v57, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+28], v50, v[vgprValuC+28]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+28], v49, v[vgprValuC+28]    // C += bias
v_cvt_pk_bf16_f32 v28, v[vgprValuC+28], v[vgprValuC+28] // convert C to bf16 in gwvw==1
buffer_store_short v28, v59, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+29], v38, v[vgprValuC+29]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+29], v37, v[vgprValuC+29]    // C += bias
v_cvt_pk_bf16_f32 v29, v[vgprValuC+29], v[vgprValuC+29] // convert C to bf16 in gwvw==1
buffer_store_short v29, v61, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+30], v42, v[vgprValuC+30]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+30], v41, v[vgprValuC+30]    // C += bias
v_cvt_pk_bf16_f32 v30, v[vgprValuC+30], v[vgprValuC+30] // convert C to bf16 in gwvw==1
buffer_store_short v30, v63, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+31], v46, v[vgprValuC+31]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+31], v45, v[vgprValuC+31]    // C += bias
v_cvt_pk_bf16_f32 v31, v[vgprValuC+31], v[vgprValuC+31] // convert C to bf16 in gwvw==1
buffer_store_short v31, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+32], v50, v[vgprValuC+32]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+32], v49, v[vgprValuC+32]    // C += bias
v_cvt_pk_bf16_f32 v32, v[vgprValuC+32], v[vgprValuC+32] // convert C to bf16 in gwvw==1
buffer_store_short v32, v67, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+33], v38, v[vgprValuC+33]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+33], v37, v[vgprValuC+33]    // C += bias
v_cvt_pk_bf16_f32 v33, v[vgprValuC+33], v[vgprValuC+33] // convert C to bf16 in gwvw==1
buffer_store_short v33, v69, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+34], v42, v[vgprValuC+34]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+34], v41, v[vgprValuC+34]    // C += bias
v_cvt_pk_bf16_f32 v34, v[vgprValuC+34], v[vgprValuC+34] // convert C to bf16 in gwvw==1
buffer_store_short v34, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+35], v46, v[vgprValuC+35]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+35], v45, v[vgprValuC+35]    // C += bias
v_cvt_pk_bf16_f32 v35, v[vgprValuC+35], v[vgprValuC+35] // convert C to bf16 in gwvw==1
buffer_store_short v35, v73, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+36], v50, v[vgprValuC+36]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+36], v49, v[vgprValuC+36]    // C += bias
v_cvt_pk_bf16_f32 v36, v[vgprValuC+36], v[vgprValuC+36] // convert C to bf16 in gwvw==1
buffer_store_short v36, v76, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (0,0,4,0:vw1); (0,1,4,0:vw1); (0,2,4,0:vw1); (0,3,4,0:vw1); (0,0,5,0:vw1); (0,1,5,0:vw1); (0,2,5,0:vw1); (0,3,5,0:vw1); (0,0,6,0:vw1); (0,1,6,0:vw1); (0,2,6,0:vw1); (0,3,6,0:vw1); (0,0,7,0:vw1); (0,1,7,0:vw1); (0,2,7,0:vw1); (0,3,7,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v16, BufferOOB
/* (d1,vc1,d0,vc0)=(0,4,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v40, v10, s70
v_lshlrev_b32 v40, 0x2, v40                        // Bias address scaled by BPE
ds_read_b32 v37, v40 offset:0                      // load Bias
ds_read_b32 v38, v40 offset:1024                   // load scaleAlpha
v_add_lshl_u32 v39, v13, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v39, v16, v39, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v44, v14, s70
v_lshlrev_b32 v44, 0x2, v44                        // Bias address scaled by BPE
ds_read_b32 v41, v44 offset:0                      // load Bias
ds_read_b32 v42, v44 offset:1024                   // load scaleAlpha
v_add_lshl_u32 v43, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v43, v16, v43, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v48, v14, s70
v_lshlrev_b32 v48, 0x2, v48                        // Bias address scaled by BPE
ds_read_b32 v45, v48 offset:0                      // load Bias
ds_read_b32 v46, v48 offset:1024                   // load scaleAlpha
v_add_lshl_u32 v47, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v47, v16, v47, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v52, v14, s70
v_lshlrev_b32 v52, 0x2, v52                        // Bias address scaled by BPE
ds_read_b32 v49, v52 offset:0                      // load Bias
ds_read_b32 v50, v52 offset:1024                   // load scaleAlpha
v_add_lshl_u32 v51, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v51, v16, v51, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v54, v10, s70
v_lshlrev_b32 v54, 0x2, v54                        // Bias address scaled by BPE
v_add_lshl_u32 v53, v13, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v53, v16, v53, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v56, v14, s70
v_lshlrev_b32 v56, 0x2, v56                        // Bias address scaled by BPE
v_add_lshl_u32 v55, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v55, v16, v55, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v58, v14, s70
v_lshlrev_b32 v58, 0x2, v58                        // Bias address scaled by BPE
v_add_lshl_u32 v57, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v57, v16, v57, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v60, v14, s70
v_lshlrev_b32 v60, 0x2, v60                        // Bias address scaled by BPE
v_add_lshl_u32 v59, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v59, v16, v59, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v62, v10, s70
v_lshlrev_b32 v62, 0x2, v62                        // Bias address scaled by BPE
v_add_lshl_u32 v61, v13, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v61, v16, v61, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v64, v14, s70
v_lshlrev_b32 v64, 0x2, v64                        // Bias address scaled by BPE
v_add_lshl_u32 v63, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v63, v16, v63, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v66, v14, s70
v_lshlrev_b32 v66, 0x2, v66                        // Bias address scaled by BPE
v_add_lshl_u32 v65, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v65, v16, v65, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v68, v14, s70
v_lshlrev_b32 v68, 0x2, v68                        // Bias address scaled by BPE
v_add_lshl_u32 v67, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v67, v16, v67, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v70, v10, s70
v_lshlrev_b32 v70, 0x2, v70                        // Bias address scaled by BPE
v_add_lshl_u32 v69, v13, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v69, v16, v69, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v72, v14, s70
v_lshlrev_b32 v72, 0x2, v72                        // Bias address scaled by BPE
v_add_lshl_u32 v71, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v71, v16, v71, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v75, v14, s70
v_lshlrev_b32 v75, 0x2, v75                        // Bias address scaled by BPE
v_add_lshl_u32 v73, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v73, v16, v73, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v77, v14, s70
v_lshlrev_b32 v77, 0x2, v77                        // Bias address scaled by BPE
v_add_lshl_u32 v76, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v76, v16, v76, s[74:75]              // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+21], acc1           // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+22], acc5           // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+23], acc9           // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+24], acc13          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+25], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+26], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+27], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+28], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+29], acc33          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+30], acc37          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+31], acc41          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+32], acc45          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+33], acc49          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+34], acc53          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+35], acc57          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+36], acc61          // copy acc to vreg[31]

/* rC *= alpha batchElements=[(0, 0, 4, 0), (0, 1, 4, 0), (0, 2, 4, 0), (0, 3, 4, 0), (0, 0, 5, 0), (0, 1, 5, 0), (0, 2, 5, 0), (0, 3, 5, 0), (0, 0, 6, 0), (0, 1, 6, 0), (0, 2, 6, 0), (0, 3, 6, 0), (0, 0, 7, 0), (0, 1, 7, 0), (0, 2, 7, 0), (0, 3, 7, 0)] */
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+21] // *= alpha
v_pk_mul_f32 v[vgprValuC+22:vgprValuC+22+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+22:vgprValuC+22+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+24:vgprValuC+24+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+26:vgprValuC+26+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+28:vgprValuC+28+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+30:vgprValuC+30+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+32:vgprValuC+32+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+34:vgprValuC+34+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+36], s[sgprAlpha], v[vgprValuC+36] // *= alpha
s_waitcnt lgkmcnt(0)                               // wait for Bias LDS, ScaleAlphaVec

/* apply mask, calc new C and issue writes */
v_mov_b32 v18, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v19, 0x7fff0000                          // fp32 Nan
v_mov_b32 v20, 0x7fff                              // rounding bias for bfloat16
v_mul_f32 v[vgprValuC+21], v38, v[vgprValuC+21]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+21], v37, v[vgprValuC+21]    // C += bias
v_cvt_pk_bf16_f32 v21, v[vgprValuC+21], v[vgprValuC+21] // convert C to bf16 in gwvw==1
buffer_store_short v21, v39, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+22], v42, v[vgprValuC+22]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+22], v41, v[vgprValuC+22]    // C += bias
v_cvt_pk_bf16_f32 v22, v[vgprValuC+22], v[vgprValuC+22] // convert C to bf16 in gwvw==1
buffer_store_short v22, v43, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+23], v46, v[vgprValuC+23]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+23], v45, v[vgprValuC+23]    // C += bias
v_cvt_pk_bf16_f32 v23, v[vgprValuC+23], v[vgprValuC+23] // convert C to bf16 in gwvw==1
buffer_store_short v23, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+24], v50, v[vgprValuC+24]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+24], v49, v[vgprValuC+24]    // C += bias
v_cvt_pk_bf16_f32 v24, v[vgprValuC+24], v[vgprValuC+24] // convert C to bf16 in gwvw==1
buffer_store_short v24, v51, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+25], v38, v[vgprValuC+25]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+25], v37, v[vgprValuC+25]    // C += bias
v_cvt_pk_bf16_f32 v25, v[vgprValuC+25], v[vgprValuC+25] // convert C to bf16 in gwvw==1
buffer_store_short v25, v53, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+26], v42, v[vgprValuC+26]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+26], v41, v[vgprValuC+26]    // C += bias
v_cvt_pk_bf16_f32 v26, v[vgprValuC+26], v[vgprValuC+26] // convert C to bf16 in gwvw==1
buffer_store_short v26, v55, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+27], v46, v[vgprValuC+27]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+27], v45, v[vgprValuC+27]    // C += bias
v_cvt_pk_bf16_f32 v27, v[vgprValuC+27], v[vgprValuC+27] // convert C to bf16 in gwvw==1
buffer_store_short v27, v57, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+28], v50, v[vgprValuC+28]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+28], v49, v[vgprValuC+28]    // C += bias
v_cvt_pk_bf16_f32 v28, v[vgprValuC+28], v[vgprValuC+28] // convert C to bf16 in gwvw==1
buffer_store_short v28, v59, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+29], v38, v[vgprValuC+29]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+29], v37, v[vgprValuC+29]    // C += bias
v_cvt_pk_bf16_f32 v29, v[vgprValuC+29], v[vgprValuC+29] // convert C to bf16 in gwvw==1
buffer_store_short v29, v61, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+30], v42, v[vgprValuC+30]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+30], v41, v[vgprValuC+30]    // C += bias
v_cvt_pk_bf16_f32 v30, v[vgprValuC+30], v[vgprValuC+30] // convert C to bf16 in gwvw==1
buffer_store_short v30, v63, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+31], v46, v[vgprValuC+31]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+31], v45, v[vgprValuC+31]    // C += bias
v_cvt_pk_bf16_f32 v31, v[vgprValuC+31], v[vgprValuC+31] // convert C to bf16 in gwvw==1
buffer_store_short v31, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+32], v50, v[vgprValuC+32]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+32], v49, v[vgprValuC+32]    // C += bias
v_cvt_pk_bf16_f32 v32, v[vgprValuC+32], v[vgprValuC+32] // convert C to bf16 in gwvw==1
buffer_store_short v32, v67, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+33], v38, v[vgprValuC+33]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+33], v37, v[vgprValuC+33]    // C += bias
v_cvt_pk_bf16_f32 v33, v[vgprValuC+33], v[vgprValuC+33] // convert C to bf16 in gwvw==1
buffer_store_short v33, v69, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+34], v42, v[vgprValuC+34]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+34], v41, v[vgprValuC+34]    // C += bias
v_cvt_pk_bf16_f32 v34, v[vgprValuC+34], v[vgprValuC+34] // convert C to bf16 in gwvw==1
buffer_store_short v34, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+35], v46, v[vgprValuC+35]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+35], v45, v[vgprValuC+35]    // C += bias
v_cvt_pk_bf16_f32 v35, v[vgprValuC+35], v[vgprValuC+35] // convert C to bf16 in gwvw==1
buffer_store_short v35, v73, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+36], v50, v[vgprValuC+36]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+36], v49, v[vgprValuC+36]    // C += bias
v_cvt_pk_bf16_f32 v36, v[vgprValuC+36], v[vgprValuC+36] // convert C to bf16 in gwvw==1
buffer_store_short v36, v76, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #2 (d1,d0,vc1,vc0) = */
/*    (0,0,8,0:vw1); (0,1,8,0:vw1); (0,2,8,0:vw1); (0,3,8,0:vw1); (0,0,9,0:vw1); (0,1,9,0:vw1); (0,2,9,0:vw1); (0,3,9,0:vw1); (0,0,10,0:vw1); (0,1,10,0:vw1); (0,2,10,0:vw1); (0,3,10,0:vw1); (0,0,11,0:vw1); (0,1,11,0:vw1); (0,2,11,0:vw1); (0,3,11,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v16, BufferOOB
/* (d1,vc1,d0,vc0)=(0,8,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v40, v10, s70
v_lshlrev_b32 v40, 0x2, v40                        // Bias address scaled by BPE
ds_read_b32 v37, v40 offset:0                      // load Bias
ds_read_b32 v38, v40 offset:1024                   // load scaleAlpha
v_add_lshl_u32 v39, v13, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v39, v16, v39, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v44, v14, s70
v_lshlrev_b32 v44, 0x2, v44                        // Bias address scaled by BPE
ds_read_b32 v41, v44 offset:0                      // load Bias
ds_read_b32 v42, v44 offset:1024                   // load scaleAlpha
v_add_lshl_u32 v43, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v43, v16, v43, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v48, v14, s70
v_lshlrev_b32 v48, 0x2, v48                        // Bias address scaled by BPE
ds_read_b32 v45, v48 offset:0                      // load Bias
ds_read_b32 v46, v48 offset:1024                   // load scaleAlpha
v_add_lshl_u32 v47, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v47, v16, v47, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v52, v14, s70
v_lshlrev_b32 v52, 0x2, v52                        // Bias address scaled by BPE
ds_read_b32 v49, v52 offset:0                      // load Bias
ds_read_b32 v50, v52 offset:1024                   // load scaleAlpha
v_add_lshl_u32 v51, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v51, v16, v51, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v54, v10, s70
v_lshlrev_b32 v54, 0x2, v54                        // Bias address scaled by BPE
v_add_lshl_u32 v53, v13, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v53, v16, v53, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v56, v14, s70
v_lshlrev_b32 v56, 0x2, v56                        // Bias address scaled by BPE
v_add_lshl_u32 v55, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v55, v16, v55, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v58, v14, s70
v_lshlrev_b32 v58, 0x2, v58                        // Bias address scaled by BPE
v_add_lshl_u32 v57, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v57, v16, v57, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v60, v14, s70
v_lshlrev_b32 v60, 0x2, v60                        // Bias address scaled by BPE
v_add_lshl_u32 v59, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v59, v16, v59, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v62, v10, s70
v_lshlrev_b32 v62, 0x2, v62                        // Bias address scaled by BPE
v_add_lshl_u32 v61, v13, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v61, v16, v61, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v64, v14, s70
v_lshlrev_b32 v64, 0x2, v64                        // Bias address scaled by BPE
v_add_lshl_u32 v63, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v63, v16, v63, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v66, v14, s70
v_lshlrev_b32 v66, 0x2, v66                        // Bias address scaled by BPE
v_add_lshl_u32 v65, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v65, v16, v65, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v68, v14, s70
v_lshlrev_b32 v68, 0x2, v68                        // Bias address scaled by BPE
v_add_lshl_u32 v67, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v67, v16, v67, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v70, v10, s70
v_lshlrev_b32 v70, 0x2, v70                        // Bias address scaled by BPE
v_add_lshl_u32 v69, v13, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v69, v16, v69, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v72, v14, s70
v_lshlrev_b32 v72, 0x2, v72                        // Bias address scaled by BPE
v_add_lshl_u32 v71, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v71, v16, v71, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v75, v14, s70
v_lshlrev_b32 v75, 0x2, v75                        // Bias address scaled by BPE
v_add_lshl_u32 v73, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v73, v16, v73, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v77, v14, s70
v_lshlrev_b32 v77, 0x2, v77                        // Bias address scaled by BPE
v_add_lshl_u32 v76, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v76, v16, v76, s[74:75]              // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+21], acc2           // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+22], acc6           // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+23], acc10          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+24], acc14          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+25], acc18          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+26], acc22          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+27], acc26          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+28], acc30          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+29], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+30], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+31], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+32], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+33], acc50          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+34], acc54          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+35], acc58          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+36], acc62          // copy acc to vreg[47]

/* rC *= alpha batchElements=[(0, 0, 8, 0), (0, 1, 8, 0), (0, 2, 8, 0), (0, 3, 8, 0), (0, 0, 9, 0), (0, 1, 9, 0), (0, 2, 9, 0), (0, 3, 9, 0), (0, 0, 10, 0), (0, 1, 10, 0), (0, 2, 10, 0), (0, 3, 10, 0), (0, 0, 11, 0), (0, 1, 11, 0), (0, 2, 11, 0), (0, 3, 11, 0)] */
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+21] // *= alpha
v_pk_mul_f32 v[vgprValuC+22:vgprValuC+22+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+22:vgprValuC+22+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+24:vgprValuC+24+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+26:vgprValuC+26+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+28:vgprValuC+28+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+30:vgprValuC+30+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+32:vgprValuC+32+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+34:vgprValuC+34+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+36], s[sgprAlpha], v[vgprValuC+36] // *= alpha
s_waitcnt lgkmcnt(0)                               // wait for Bias LDS, ScaleAlphaVec

/* apply mask, calc new C and issue writes */
v_mov_b32 v18, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v19, 0x7fff0000                          // fp32 Nan
v_mov_b32 v20, 0x7fff                              // rounding bias for bfloat16
v_mul_f32 v[vgprValuC+21], v38, v[vgprValuC+21]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+21], v37, v[vgprValuC+21]    // C += bias
v_cvt_pk_bf16_f32 v21, v[vgprValuC+21], v[vgprValuC+21] // convert C to bf16 in gwvw==1
buffer_store_short v21, v39, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+22], v42, v[vgprValuC+22]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+22], v41, v[vgprValuC+22]    // C += bias
v_cvt_pk_bf16_f32 v22, v[vgprValuC+22], v[vgprValuC+22] // convert C to bf16 in gwvw==1
buffer_store_short v22, v43, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+23], v46, v[vgprValuC+23]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+23], v45, v[vgprValuC+23]    // C += bias
v_cvt_pk_bf16_f32 v23, v[vgprValuC+23], v[vgprValuC+23] // convert C to bf16 in gwvw==1
buffer_store_short v23, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+24], v50, v[vgprValuC+24]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+24], v49, v[vgprValuC+24]    // C += bias
v_cvt_pk_bf16_f32 v24, v[vgprValuC+24], v[vgprValuC+24] // convert C to bf16 in gwvw==1
buffer_store_short v24, v51, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+25], v38, v[vgprValuC+25]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+25], v37, v[vgprValuC+25]    // C += bias
v_cvt_pk_bf16_f32 v25, v[vgprValuC+25], v[vgprValuC+25] // convert C to bf16 in gwvw==1
buffer_store_short v25, v53, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+26], v42, v[vgprValuC+26]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+26], v41, v[vgprValuC+26]    // C += bias
v_cvt_pk_bf16_f32 v26, v[vgprValuC+26], v[vgprValuC+26] // convert C to bf16 in gwvw==1
buffer_store_short v26, v55, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+27], v46, v[vgprValuC+27]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+27], v45, v[vgprValuC+27]    // C += bias
v_cvt_pk_bf16_f32 v27, v[vgprValuC+27], v[vgprValuC+27] // convert C to bf16 in gwvw==1
buffer_store_short v27, v57, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+28], v50, v[vgprValuC+28]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+28], v49, v[vgprValuC+28]    // C += bias
v_cvt_pk_bf16_f32 v28, v[vgprValuC+28], v[vgprValuC+28] // convert C to bf16 in gwvw==1
buffer_store_short v28, v59, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+29], v38, v[vgprValuC+29]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+29], v37, v[vgprValuC+29]    // C += bias
v_cvt_pk_bf16_f32 v29, v[vgprValuC+29], v[vgprValuC+29] // convert C to bf16 in gwvw==1
buffer_store_short v29, v61, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+30], v42, v[vgprValuC+30]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+30], v41, v[vgprValuC+30]    // C += bias
v_cvt_pk_bf16_f32 v30, v[vgprValuC+30], v[vgprValuC+30] // convert C to bf16 in gwvw==1
buffer_store_short v30, v63, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+31], v46, v[vgprValuC+31]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+31], v45, v[vgprValuC+31]    // C += bias
v_cvt_pk_bf16_f32 v31, v[vgprValuC+31], v[vgprValuC+31] // convert C to bf16 in gwvw==1
buffer_store_short v31, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+32], v50, v[vgprValuC+32]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+32], v49, v[vgprValuC+32]    // C += bias
v_cvt_pk_bf16_f32 v32, v[vgprValuC+32], v[vgprValuC+32] // convert C to bf16 in gwvw==1
buffer_store_short v32, v67, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+33], v38, v[vgprValuC+33]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+33], v37, v[vgprValuC+33]    // C += bias
v_cvt_pk_bf16_f32 v33, v[vgprValuC+33], v[vgprValuC+33] // convert C to bf16 in gwvw==1
buffer_store_short v33, v69, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+34], v42, v[vgprValuC+34]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+34], v41, v[vgprValuC+34]    // C += bias
v_cvt_pk_bf16_f32 v34, v[vgprValuC+34], v[vgprValuC+34] // convert C to bf16 in gwvw==1
buffer_store_short v34, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+35], v46, v[vgprValuC+35]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+35], v45, v[vgprValuC+35]    // C += bias
v_cvt_pk_bf16_f32 v35, v[vgprValuC+35], v[vgprValuC+35] // convert C to bf16 in gwvw==1
buffer_store_short v35, v73, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+36], v50, v[vgprValuC+36]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+36], v49, v[vgprValuC+36]    // C += bias
v_cvt_pk_bf16_f32 v36, v[vgprValuC+36], v[vgprValuC+36] // convert C to bf16 in gwvw==1
buffer_store_short v36, v76, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #3 (d1,d0,vc1,vc0) = */
/*    (0,0,12,0:vw1); (0,1,12,0:vw1); (0,2,12,0:vw1); (0,3,12,0:vw1); (0,0,13,0:vw1); (0,1,13,0:vw1); (0,2,13,0:vw1); (0,3,13,0:vw1); (0,0,14,0:vw1); (0,1,14,0:vw1); (0,2,14,0:vw1); (0,3,14,0:vw1); (0,0,15,0:vw1); (0,1,15,0:vw1); (0,2,15,0:vw1); (0,3,15,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v16, BufferOOB
/* (d1,vc1,d0,vc0)=(0,12,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v40, v10, s70
v_lshlrev_b32 v40, 0x2, v40                        // Bias address scaled by BPE
ds_read_b32 v37, v40 offset:0                      // load Bias
ds_read_b32 v38, v40 offset:1024                   // load scaleAlpha
v_add_lshl_u32 v39, v13, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v39, v16, v39, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v44, v14, s70
v_lshlrev_b32 v44, 0x2, v44                        // Bias address scaled by BPE
ds_read_b32 v41, v44 offset:0                      // load Bias
ds_read_b32 v42, v44 offset:1024                   // load scaleAlpha
v_add_lshl_u32 v43, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v43, v16, v43, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v48, v14, s70
v_lshlrev_b32 v48, 0x2, v48                        // Bias address scaled by BPE
ds_read_b32 v45, v48 offset:0                      // load Bias
ds_read_b32 v46, v48 offset:1024                   // load scaleAlpha
v_add_lshl_u32 v47, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v47, v16, v47, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v52, v14, s70
v_lshlrev_b32 v52, 0x2, v52                        // Bias address scaled by BPE
ds_read_b32 v49, v52 offset:0                      // load Bias
ds_read_b32 v50, v52 offset:1024                   // load scaleAlpha
v_add_lshl_u32 v51, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v51, v16, v51, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v54, v10, s70
v_lshlrev_b32 v54, 0x2, v54                        // Bias address scaled by BPE
v_add_lshl_u32 v53, v13, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v53, v16, v53, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v56, v14, s70
v_lshlrev_b32 v56, 0x2, v56                        // Bias address scaled by BPE
v_add_lshl_u32 v55, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v55, v16, v55, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v58, v14, s70
v_lshlrev_b32 v58, 0x2, v58                        // Bias address scaled by BPE
v_add_lshl_u32 v57, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v57, v16, v57, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v60, v14, s70
v_lshlrev_b32 v60, 0x2, v60                        // Bias address scaled by BPE
v_add_lshl_u32 v59, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v59, v16, v59, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v62, v10, s70
v_lshlrev_b32 v62, 0x2, v62                        // Bias address scaled by BPE
v_add_lshl_u32 v61, v13, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v61, v16, v61, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v64, v14, s70
v_lshlrev_b32 v64, 0x2, v64                        // Bias address scaled by BPE
v_add_lshl_u32 v63, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v63, v16, v63, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v66, v14, s70
v_lshlrev_b32 v66, 0x2, v66                        // Bias address scaled by BPE
v_add_lshl_u32 v65, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v65, v16, v65, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v68, v14, s70
v_lshlrev_b32 v68, 0x2, v68                        // Bias address scaled by BPE
v_add_lshl_u32 v67, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v67, v16, v67, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v70, v10, s70
v_lshlrev_b32 v70, 0x2, v70                        // Bias address scaled by BPE
v_add_lshl_u32 v69, v13, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v69, v16, v69, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v72, v14, s70
v_lshlrev_b32 v72, 0x2, v72                        // Bias address scaled by BPE
v_add_lshl_u32 v71, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v71, v16, v71, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v75, v14, s70
v_lshlrev_b32 v75, 0x2, v75                        // Bias address scaled by BPE
v_add_lshl_u32 v73, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v73, v16, v73, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v77, v14, s70
v_lshlrev_b32 v77, 0x2, v77                        // Bias address scaled by BPE
v_add_lshl_u32 v76, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v76, v16, v76, s[74:75]              // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+21], acc3           // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+22], acc7           // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+23], acc11          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+24], acc15          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+25], acc19          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+26], acc23          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+27], acc27          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+28], acc31          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+29], acc35          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+30], acc39          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+31], acc43          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+32], acc47          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+33], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+34], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+35], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+36], acc63          // copy acc to vreg[63]

/* rC *= alpha batchElements=[(0, 0, 12, 0), (0, 1, 12, 0), (0, 2, 12, 0), (0, 3, 12, 0), (0, 0, 13, 0), (0, 1, 13, 0), (0, 2, 13, 0), (0, 3, 13, 0), (0, 0, 14, 0), (0, 1, 14, 0), (0, 2, 14, 0), (0, 3, 14, 0), (0, 0, 15, 0), (0, 1, 15, 0), (0, 2, 15, 0), (0, 3, 15, 0)] */
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+21] // *= alpha
v_pk_mul_f32 v[vgprValuC+22:vgprValuC+22+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+22:vgprValuC+22+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+24:vgprValuC+24+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+26:vgprValuC+26+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+28:vgprValuC+28+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+30:vgprValuC+30+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+32:vgprValuC+32+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+34:vgprValuC+34+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+36], s[sgprAlpha], v[vgprValuC+36] // *= alpha
s_waitcnt lgkmcnt(0)                               // wait for Bias LDS, ScaleAlphaVec

/* apply mask, calc new C and issue writes */
v_mov_b32 v18, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v19, 0x7fff0000                          // fp32 Nan
v_mov_b32 v20, 0x7fff                              // rounding bias for bfloat16
v_mul_f32 v[vgprValuC+21], v38, v[vgprValuC+21]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+21], v37, v[vgprValuC+21]    // C += bias
v_cvt_pk_bf16_f32 v21, v[vgprValuC+21], v[vgprValuC+21] // convert C to bf16 in gwvw==1
buffer_store_short v21, v39, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+22], v42, v[vgprValuC+22]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+22], v41, v[vgprValuC+22]    // C += bias
v_cvt_pk_bf16_f32 v22, v[vgprValuC+22], v[vgprValuC+22] // convert C to bf16 in gwvw==1
buffer_store_short v22, v43, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+23], v46, v[vgprValuC+23]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+23], v45, v[vgprValuC+23]    // C += bias
v_cvt_pk_bf16_f32 v23, v[vgprValuC+23], v[vgprValuC+23] // convert C to bf16 in gwvw==1
buffer_store_short v23, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+24], v50, v[vgprValuC+24]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+24], v49, v[vgprValuC+24]    // C += bias
v_cvt_pk_bf16_f32 v24, v[vgprValuC+24], v[vgprValuC+24] // convert C to bf16 in gwvw==1
buffer_store_short v24, v51, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+25], v38, v[vgprValuC+25]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+25], v37, v[vgprValuC+25]    // C += bias
v_cvt_pk_bf16_f32 v25, v[vgprValuC+25], v[vgprValuC+25] // convert C to bf16 in gwvw==1
buffer_store_short v25, v53, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+26], v42, v[vgprValuC+26]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+26], v41, v[vgprValuC+26]    // C += bias
v_cvt_pk_bf16_f32 v26, v[vgprValuC+26], v[vgprValuC+26] // convert C to bf16 in gwvw==1
buffer_store_short v26, v55, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+27], v46, v[vgprValuC+27]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+27], v45, v[vgprValuC+27]    // C += bias
v_cvt_pk_bf16_f32 v27, v[vgprValuC+27], v[vgprValuC+27] // convert C to bf16 in gwvw==1
buffer_store_short v27, v57, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+28], v50, v[vgprValuC+28]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+28], v49, v[vgprValuC+28]    // C += bias
v_cvt_pk_bf16_f32 v28, v[vgprValuC+28], v[vgprValuC+28] // convert C to bf16 in gwvw==1
buffer_store_short v28, v59, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+29], v38, v[vgprValuC+29]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+29], v37, v[vgprValuC+29]    // C += bias
v_cvt_pk_bf16_f32 v29, v[vgprValuC+29], v[vgprValuC+29] // convert C to bf16 in gwvw==1
buffer_store_short v29, v61, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+30], v42, v[vgprValuC+30]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+30], v41, v[vgprValuC+30]    // C += bias
v_cvt_pk_bf16_f32 v30, v[vgprValuC+30], v[vgprValuC+30] // convert C to bf16 in gwvw==1
buffer_store_short v30, v63, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+31], v46, v[vgprValuC+31]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+31], v45, v[vgprValuC+31]    // C += bias
v_cvt_pk_bf16_f32 v31, v[vgprValuC+31], v[vgprValuC+31] // convert C to bf16 in gwvw==1
buffer_store_short v31, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+32], v50, v[vgprValuC+32]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+32], v49, v[vgprValuC+32]    // C += bias
v_cvt_pk_bf16_f32 v32, v[vgprValuC+32], v[vgprValuC+32] // convert C to bf16 in gwvw==1
buffer_store_short v32, v67, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+33], v38, v[vgprValuC+33]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+33], v37, v[vgprValuC+33]    // C += bias
v_cvt_pk_bf16_f32 v33, v[vgprValuC+33], v[vgprValuC+33] // convert C to bf16 in gwvw==1
buffer_store_short v33, v69, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+34], v42, v[vgprValuC+34]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+34], v41, v[vgprValuC+34]    // C += bias
v_cvt_pk_bf16_f32 v34, v[vgprValuC+34], v[vgprValuC+34] // convert C to bf16 in gwvw==1
buffer_store_short v34, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+35], v46, v[vgprValuC+35]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+35], v45, v[vgprValuC+35]    // C += bias
v_cvt_pk_bf16_f32 v35, v[vgprValuC+35], v[vgprValuC+35] // convert C to bf16 in gwvw==1
buffer_store_short v35, v73, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+36], v50, v[vgprValuC+36]    // *= ScaleAlphaVecVMul
v_add_f32 v[vgprValuC+36], v49, v[vgprValuC+36]    // C += bias
v_cvt_pk_bf16_f32 v36, v[vgprValuC+36], v[vgprValuC+36] // convert C to bf16 in gwvw==1
buffer_store_short v36, v76, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_Beta_1:
s_and_b32 s70, 127, s[sgprSizeI]                   // s70 = s[sgprSizeI] % 128
s_add_u32 s71, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s71                // wg0 >= nwg0-1 ?
s_cselect_b32 s70, s70, 0                          // set rMT0
s_cmpk_gt_u32 s70, 0                               // rMT0 > 0
s_cbranch_scc1 label_GW_B1_E1                      // jump if edges required
s_and_b32 s70, 127, s[sgprSizeJ]                   // s70 = s[sgprSizeJ] % 128
s_add_u32 s71, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s71                // wg1 >= nwg1-1
s_cselect_b32 s70, s70, 0                          // set rMT1
s_cmpk_gt_u32 s70, 0                               // rMT1 > 0
s_cbranch_scc1 label_GW_B1_E1                      // jump if edges required
label_GW_B1_E0:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=16 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,0,1,0:vw1); (0,1,1,0:vw1); (0,2,1,0:vw1); (0,3,1,0:vw1); (0,0,2,0:vw1); (0,1,2,0:vw1); (0,2,2,0:vw1); (0,3,2,0:vw1); (0,0,3,0:vw1); (0,1,3,0:vw1); (0,2,3,0:vw1); (0,3,3,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v22, v12, v10, 0x1                  // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=10, coord0Vgpr=10
buffer_load_short_d16 v40, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s8, 128, s[sgprWorkGroup0]               // wgp0 * MT0
v_sub_u32 v23, v10, s8
v_lshlrev_b32 v23, 0x2, v23                        // Bias address scaled by BPE
s_waitcnt lgkmcnt(0)                               // Wait for LDS write
s_barrier                                          // LDS write barrier
ds_read_b32 v41, v23 offset:0                      // load Bias
ds_read_b32 v42, v23 offset:1024                   // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
buffer_load_short_d16 v43, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
ds_read_b32 v44, v23 offset:128                    // load Bias
ds_read_b32 v45, v23 offset:1152                   // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
buffer_load_short_d16 v46, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
ds_read_b32 v47, v23 offset:256                    // load Bias
ds_read_b32 v48, v23 offset:1280                   // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
buffer_load_short_d16 v49, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:192 // load C
ds_read_b32 v50, v23 offset:384                    // load Bias
ds_read_b32 v51, v23 offset:1408                   // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v52, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
buffer_load_short_d16 v53, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(0,1,2,0) */
buffer_load_short_d16 v54, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(0,1,3,0) */
buffer_load_short_d16 v55, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:192 // load C
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v56, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,2,1,0) */
buffer_load_short_d16 v57, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(0,2,2,0) */
buffer_load_short_d16 v58, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(0,2,3,0) */
buffer_load_short_d16 v59, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:192 // load C
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v60, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,3,1,0) */
buffer_load_short_d16 v61, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(0,3,2,0) */
buffer_load_short_d16 v62, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(0,3,3,0) */
buffer_load_short_d16 v63, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:192 // load C
v_add_lshl_u32 v21, v13, v10, 0x1                  // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=10, coord0Vgpr=10
v_accvgpr_read_b32 v[vgprValuC+24], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+25], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+26], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+27], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+28], acc16          // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+29], acc20          // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+30], acc24          // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+31], acc28          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+32], acc32          // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+33], acc36          // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+34], acc40          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+35], acc44          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+36], acc48          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+37], acc52          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+38], acc56          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+39], acc60          // copy acc to vreg[15]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 0, 1, 0), (0, 1, 1, 0), (0, 2, 1, 0), (0, 3, 1, 0), (0, 0, 2, 0), (0, 1, 2, 0), (0, 2, 2, 0), (0, 3, 2, 0), (0, 0, 3, 0), (0, 1, 3, 0), (0, 2, 3, 0), (0, 3, 3, 0)] */
v_pk_mul_f32 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+24:vgprValuC+24+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+26:vgprValuC+26+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+28:vgprValuC+28+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+30:vgprValuC+30+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+32:vgprValuC+32+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+34:vgprValuC+34+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+36:vgprValuC+36+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+36:vgprValuC+36+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+38:vgprValuC+38+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+38:vgprValuC+38+1] op_sel_hi:[0,1,1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */
v_mov_b32 v18, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v19, 0x7fff0000                          // fp32 Nan
v_mov_b32 v20, 0x7fff                              // rounding bias for bfloat16

s_waitcnt lgkmcnt(6), vmcnt(15)                    // vlcnt(15) = 16 - 1 (beta) vscnt(0) dscnt(6) = 8 - 1 (bias) - 1 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+24], v42, v[vgprValuC+24]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v40 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+24], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+24], v41, v[vgprValuC+24]    // C += bias
v_cvt_pk_bf16_f32 v24, v[vgprValuC+24], v[vgprValuC+24] // convert C to bf16 in gwvw==1
buffer_store_short v24, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt lgkmcnt(4), vmcnt(15)                    // vlcnt(14) = 16 - 2 (beta) vscnt(1) dscnt(4) = 8 - 2 (bias) - 2 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+25], v45, v[vgprValuC+25]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v43 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+25], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+25], v44, v[vgprValuC+25]    // C += bias
v_cvt_pk_bf16_f32 v25, v[vgprValuC+25], v[vgprValuC+25] // convert C to bf16 in gwvw==1
buffer_store_short v25, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt lgkmcnt(2), vmcnt(15)                    // vlcnt(13) = 16 - 3 (beta) vscnt(2) dscnt(2) = 8 - 3 (bias) - 3 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+26], v48, v[vgprValuC+26]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v46 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+26], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+26], v47, v[vgprValuC+26]    // C += bias
v_cvt_pk_bf16_f32 v26, v[vgprValuC+26], v[vgprValuC+26] // convert C to bf16 in gwvw==1
buffer_store_short v26, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt lgkmcnt(0), vmcnt(15)                    // vlcnt(12) = 16 - 4 (beta) vscnt(3) dscnt(0) = 8 - 4 (bias) - 4 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+27], v51, v[vgprValuC+27]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v49 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+27], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+27], v50, v[vgprValuC+27]    // C += bias
v_cvt_pk_bf16_f32 v27, v[vgprValuC+27], v[vgprValuC+27] // convert C to bf16 in gwvw==1
buffer_store_short v27, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D

s_waitcnt vmcnt(15)                                // vlcnt(11) = 16 - 5 (beta) vscnt(4) (interleaved)
v_mul_f32 v[vgprValuC+28], v42, v[vgprValuC+28]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v52 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+28], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+28], v41, v[vgprValuC+28]    // C += bias
v_cvt_pk_bf16_f32 v28, v[vgprValuC+28], v[vgprValuC+28] // convert C to bf16 in gwvw==1
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v28, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(10) = 16 - 6 (beta) vscnt(5) (interleaved)
v_mul_f32 v[vgprValuC+29], v45, v[vgprValuC+29]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v53 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+29], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+29], v44, v[vgprValuC+29]    // C += bias
v_cvt_pk_bf16_f32 v29, v[vgprValuC+29], v[vgprValuC+29] // convert C to bf16 in gwvw==1
buffer_store_short v29, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(15)                                // vlcnt(9) = 16 - 7 (beta) vscnt(6) (interleaved)
v_mul_f32 v[vgprValuC+30], v48, v[vgprValuC+30]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v54 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+30], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+30], v47, v[vgprValuC+30]    // C += bias
v_cvt_pk_bf16_f32 v30, v[vgprValuC+30], v[vgprValuC+30] // convert C to bf16 in gwvw==1
buffer_store_short v30, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(15)                                // vlcnt(8) = 16 - 8 (beta) vscnt(7) (interleaved)
v_mul_f32 v[vgprValuC+31], v51, v[vgprValuC+31]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v55 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+31], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+31], v50, v[vgprValuC+31]    // C += bias
v_cvt_pk_bf16_f32 v31, v[vgprValuC+31], v[vgprValuC+31] // convert C to bf16 in gwvw==1
buffer_store_short v31, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D

s_waitcnt vmcnt(15)                                // vlcnt(7) = 16 - 9 (beta) vscnt(8) (interleaved)
v_mul_f32 v[vgprValuC+32], v42, v[vgprValuC+32]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v56 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+32], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+32], v41, v[vgprValuC+32]    // C += bias
v_cvt_pk_bf16_f32 v32, v[vgprValuC+32], v[vgprValuC+32] // convert C to bf16 in gwvw==1
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v32, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(6) = 16 - 10 (beta) vscnt(9) (interleaved)
v_mul_f32 v[vgprValuC+33], v45, v[vgprValuC+33]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v57 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+33], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+33], v44, v[vgprValuC+33]    // C += bias
v_cvt_pk_bf16_f32 v33, v[vgprValuC+33], v[vgprValuC+33] // convert C to bf16 in gwvw==1
buffer_store_short v33, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(15)                                // vlcnt(5) = 16 - 11 (beta) vscnt(10) (interleaved)
v_mul_f32 v[vgprValuC+34], v48, v[vgprValuC+34]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v58 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+34], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+34], v47, v[vgprValuC+34]    // C += bias
v_cvt_pk_bf16_f32 v34, v[vgprValuC+34], v[vgprValuC+34] // convert C to bf16 in gwvw==1
buffer_store_short v34, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(15)                                // vlcnt(4) = 16 - 12 (beta) vscnt(11) (interleaved)
v_mul_f32 v[vgprValuC+35], v51, v[vgprValuC+35]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v59 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+35], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+35], v50, v[vgprValuC+35]    // C += bias
v_cvt_pk_bf16_f32 v35, v[vgprValuC+35], v[vgprValuC+35] // convert C to bf16 in gwvw==1
buffer_store_short v35, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D

s_waitcnt vmcnt(15)                                // vlcnt(3) = 16 - 13 (beta) vscnt(12) (interleaved)
v_mul_f32 v[vgprValuC+36], v42, v[vgprValuC+36]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v60 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+36], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+36], v41, v[vgprValuC+36]    // C += bias
v_cvt_pk_bf16_f32 v36, v[vgprValuC+36], v[vgprValuC+36] // convert C to bf16 in gwvw==1
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v36, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(2) = 16 - 14 (beta) vscnt(13) (interleaved)
v_mul_f32 v[vgprValuC+37], v45, v[vgprValuC+37]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v61 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+37], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+37], v44, v[vgprValuC+37]    // C += bias
v_cvt_pk_bf16_f32 v37, v[vgprValuC+37], v[vgprValuC+37] // convert C to bf16 in gwvw==1
buffer_store_short v37, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(15)                                // vlcnt(1) = 16 - 15 (beta) vscnt(14) (interleaved)
v_mul_f32 v[vgprValuC+38], v48, v[vgprValuC+38]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v62 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+38], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+38], v47, v[vgprValuC+38]    // C += bias
v_cvt_pk_bf16_f32 v38, v[vgprValuC+38], v[vgprValuC+38] // convert C to bf16 in gwvw==1
buffer_store_short v38, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(15)                                // vlcnt(0) = 16 - 16 (beta) vscnt(15) (interleaved)
v_mul_f32 v[vgprValuC+39], v51, v[vgprValuC+39]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v63 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+39], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+39], v50, v[vgprValuC+39]    // C += bias
v_cvt_pk_bf16_f32 v39, v[vgprValuC+39], v[vgprValuC+39] // convert C to bf16 in gwvw==1
buffer_store_short v39, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #1 (d1,d0,vc1,vc0) = */
/*    (0,0,4,0:vw1); (0,1,4,0:vw1); (0,2,4,0:vw1); (0,3,4,0:vw1); (0,0,5,0:vw1); (0,1,5,0:vw1); (0,2,5,0:vw1); (0,3,5,0:vw1); (0,0,6,0:vw1); (0,1,6,0:vw1); (0,2,6,0:vw1); (0,3,6,0:vw1); (0,0,7,0:vw1); (0,1,7,0:vw1); (0,2,7,0:vw1); (0,3,7,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,4,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v40, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
ds_read_b32 v41, v23 offset:0                      // load Bias
ds_read_b32 v42, v23 offset:1024                   // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,4,1,0) */
buffer_load_short_d16 v43, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
ds_read_b32 v44, v23 offset:128                    // load Bias
ds_read_b32 v45, v23 offset:1152                   // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,4,2,0) */
buffer_load_short_d16 v46, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
ds_read_b32 v47, v23 offset:256                    // load Bias
ds_read_b32 v48, v23 offset:1280                   // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,4,3,0) */
buffer_load_short_d16 v49, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:192 // load C
ds_read_b32 v50, v23 offset:384                    // load Bias
ds_read_b32 v51, v23 offset:1408                   // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,5,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v52, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,5,1,0) */
buffer_load_short_d16 v53, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(0,5,2,0) */
buffer_load_short_d16 v54, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(0,5,3,0) */
buffer_load_short_d16 v55, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:192 // load C
/* (d1,vc1,d0,vc0)=(0,6,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v56, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,6,1,0) */
buffer_load_short_d16 v57, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(0,6,2,0) */
buffer_load_short_d16 v58, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(0,6,3,0) */
buffer_load_short_d16 v59, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:192 // load C
/* (d1,vc1,d0,vc0)=(0,7,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v60, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,7,1,0) */
buffer_load_short_d16 v61, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(0,7,2,0) */
buffer_load_short_d16 v62, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(0,7,3,0) */
buffer_load_short_d16 v63, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:192 // load C
v_accvgpr_read_b32 v[vgprValuC+24], acc1           // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+25], acc5           // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+26], acc9           // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+27], acc13          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+28], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+29], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+30], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+31], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+32], acc33          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+33], acc37          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+34], acc41          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+35], acc45          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+36], acc49          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+37], acc53          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+38], acc57          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+39], acc61          // copy acc to vreg[31]

/* rC *= alpha batchElements=[(0, 0, 4, 0), (0, 1, 4, 0), (0, 2, 4, 0), (0, 3, 4, 0), (0, 0, 5, 0), (0, 1, 5, 0), (0, 2, 5, 0), (0, 3, 5, 0), (0, 0, 6, 0), (0, 1, 6, 0), (0, 2, 6, 0), (0, 3, 6, 0), (0, 0, 7, 0), (0, 1, 7, 0), (0, 2, 7, 0), (0, 3, 7, 0)] */
v_pk_mul_f32 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+24:vgprValuC+24+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+26:vgprValuC+26+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+28:vgprValuC+28+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+30:vgprValuC+30+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+32:vgprValuC+32+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+34:vgprValuC+34+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+36:vgprValuC+36+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+36:vgprValuC+36+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+38:vgprValuC+38+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+38:vgprValuC+38+1] op_sel_hi:[0,1,1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */
v_mov_b32 v18, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v19, 0x7fff0000                          // fp32 Nan
v_mov_b32 v20, 0x7fff                              // rounding bias for bfloat16

s_waitcnt lgkmcnt(6), vmcnt(15)                    // vlcnt(15) = 16 - 1 (beta) vscnt(0) dscnt(6) = 8 - 1 (bias) - 1 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+24], v42, v[vgprValuC+24]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v40 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+24], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+24], v41, v[vgprValuC+24]    // C += bias
v_cvt_pk_bf16_f32 v24, v[vgprValuC+24], v[vgprValuC+24] // convert C to bf16 in gwvw==1
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v24, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt lgkmcnt(4), vmcnt(15)                    // vlcnt(14) = 16 - 2 (beta) vscnt(1) dscnt(4) = 8 - 2 (bias) - 2 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+25], v45, v[vgprValuC+25]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v43 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+25], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+25], v44, v[vgprValuC+25]    // C += bias
v_cvt_pk_bf16_f32 v25, v[vgprValuC+25], v[vgprValuC+25] // convert C to bf16 in gwvw==1
buffer_store_short v25, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt lgkmcnt(2), vmcnt(15)                    // vlcnt(13) = 16 - 3 (beta) vscnt(2) dscnt(2) = 8 - 3 (bias) - 3 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+26], v48, v[vgprValuC+26]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v46 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+26], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+26], v47, v[vgprValuC+26]    // C += bias
v_cvt_pk_bf16_f32 v26, v[vgprValuC+26], v[vgprValuC+26] // convert C to bf16 in gwvw==1
buffer_store_short v26, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt lgkmcnt(0), vmcnt(15)                    // vlcnt(12) = 16 - 4 (beta) vscnt(3) dscnt(0) = 8 - 4 (bias) - 4 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+27], v51, v[vgprValuC+27]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v49 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+27], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+27], v50, v[vgprValuC+27]    // C += bias
v_cvt_pk_bf16_f32 v27, v[vgprValuC+27], v[vgprValuC+27] // convert C to bf16 in gwvw==1
buffer_store_short v27, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D

s_waitcnt vmcnt(15)                                // vlcnt(11) = 16 - 5 (beta) vscnt(4) (interleaved)
v_mul_f32 v[vgprValuC+28], v42, v[vgprValuC+28]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v52 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+28], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+28], v41, v[vgprValuC+28]    // C += bias
v_cvt_pk_bf16_f32 v28, v[vgprValuC+28], v[vgprValuC+28] // convert C to bf16 in gwvw==1
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v28, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(10) = 16 - 6 (beta) vscnt(5) (interleaved)
v_mul_f32 v[vgprValuC+29], v45, v[vgprValuC+29]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v53 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+29], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+29], v44, v[vgprValuC+29]    // C += bias
v_cvt_pk_bf16_f32 v29, v[vgprValuC+29], v[vgprValuC+29] // convert C to bf16 in gwvw==1
buffer_store_short v29, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(15)                                // vlcnt(9) = 16 - 7 (beta) vscnt(6) (interleaved)
v_mul_f32 v[vgprValuC+30], v48, v[vgprValuC+30]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v54 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+30], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+30], v47, v[vgprValuC+30]    // C += bias
v_cvt_pk_bf16_f32 v30, v[vgprValuC+30], v[vgprValuC+30] // convert C to bf16 in gwvw==1
buffer_store_short v30, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(15)                                // vlcnt(8) = 16 - 8 (beta) vscnt(7) (interleaved)
v_mul_f32 v[vgprValuC+31], v51, v[vgprValuC+31]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v55 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+31], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+31], v50, v[vgprValuC+31]    // C += bias
v_cvt_pk_bf16_f32 v31, v[vgprValuC+31], v[vgprValuC+31] // convert C to bf16 in gwvw==1
buffer_store_short v31, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D

s_waitcnt vmcnt(15)                                // vlcnt(7) = 16 - 9 (beta) vscnt(8) (interleaved)
v_mul_f32 v[vgprValuC+32], v42, v[vgprValuC+32]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v56 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+32], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+32], v41, v[vgprValuC+32]    // C += bias
v_cvt_pk_bf16_f32 v32, v[vgprValuC+32], v[vgprValuC+32] // convert C to bf16 in gwvw==1
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v32, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(6) = 16 - 10 (beta) vscnt(9) (interleaved)
v_mul_f32 v[vgprValuC+33], v45, v[vgprValuC+33]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v57 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+33], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+33], v44, v[vgprValuC+33]    // C += bias
v_cvt_pk_bf16_f32 v33, v[vgprValuC+33], v[vgprValuC+33] // convert C to bf16 in gwvw==1
buffer_store_short v33, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(15)                                // vlcnt(5) = 16 - 11 (beta) vscnt(10) (interleaved)
v_mul_f32 v[vgprValuC+34], v48, v[vgprValuC+34]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v58 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+34], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+34], v47, v[vgprValuC+34]    // C += bias
v_cvt_pk_bf16_f32 v34, v[vgprValuC+34], v[vgprValuC+34] // convert C to bf16 in gwvw==1
buffer_store_short v34, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(15)                                // vlcnt(4) = 16 - 12 (beta) vscnt(11) (interleaved)
v_mul_f32 v[vgprValuC+35], v51, v[vgprValuC+35]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v59 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+35], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+35], v50, v[vgprValuC+35]    // C += bias
v_cvt_pk_bf16_f32 v35, v[vgprValuC+35], v[vgprValuC+35] // convert C to bf16 in gwvw==1
buffer_store_short v35, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D

s_waitcnt vmcnt(15)                                // vlcnt(3) = 16 - 13 (beta) vscnt(12) (interleaved)
v_mul_f32 v[vgprValuC+36], v42, v[vgprValuC+36]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v60 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+36], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+36], v41, v[vgprValuC+36]    // C += bias
v_cvt_pk_bf16_f32 v36, v[vgprValuC+36], v[vgprValuC+36] // convert C to bf16 in gwvw==1
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v36, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(2) = 16 - 14 (beta) vscnt(13) (interleaved)
v_mul_f32 v[vgprValuC+37], v45, v[vgprValuC+37]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v61 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+37], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+37], v44, v[vgprValuC+37]    // C += bias
v_cvt_pk_bf16_f32 v37, v[vgprValuC+37], v[vgprValuC+37] // convert C to bf16 in gwvw==1
buffer_store_short v37, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(15)                                // vlcnt(1) = 16 - 15 (beta) vscnt(14) (interleaved)
v_mul_f32 v[vgprValuC+38], v48, v[vgprValuC+38]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v62 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+38], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+38], v47, v[vgprValuC+38]    // C += bias
v_cvt_pk_bf16_f32 v38, v[vgprValuC+38], v[vgprValuC+38] // convert C to bf16 in gwvw==1
buffer_store_short v38, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(15)                                // vlcnt(0) = 16 - 16 (beta) vscnt(15) (interleaved)
v_mul_f32 v[vgprValuC+39], v51, v[vgprValuC+39]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v63 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+39], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+39], v50, v[vgprValuC+39]    // C += bias
v_cvt_pk_bf16_f32 v39, v[vgprValuC+39], v[vgprValuC+39] // convert C to bf16 in gwvw==1
buffer_store_short v39, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #2 (d1,d0,vc1,vc0) = */
/*    (0,0,8,0:vw1); (0,1,8,0:vw1); (0,2,8,0:vw1); (0,3,8,0:vw1); (0,0,9,0:vw1); (0,1,9,0:vw1); (0,2,9,0:vw1); (0,3,9,0:vw1); (0,0,10,0:vw1); (0,1,10,0:vw1); (0,2,10,0:vw1); (0,3,10,0:vw1); (0,0,11,0:vw1); (0,1,11,0:vw1); (0,2,11,0:vw1); (0,3,11,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,8,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v40, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
ds_read_b32 v41, v23 offset:0                      // load Bias
ds_read_b32 v42, v23 offset:1024                   // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,8,1,0) */
buffer_load_short_d16 v43, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
ds_read_b32 v44, v23 offset:128                    // load Bias
ds_read_b32 v45, v23 offset:1152                   // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,8,2,0) */
buffer_load_short_d16 v46, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
ds_read_b32 v47, v23 offset:256                    // load Bias
ds_read_b32 v48, v23 offset:1280                   // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,8,3,0) */
buffer_load_short_d16 v49, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:192 // load C
ds_read_b32 v50, v23 offset:384                    // load Bias
ds_read_b32 v51, v23 offset:1408                   // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,9,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v52, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,9,1,0) */
buffer_load_short_d16 v53, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(0,9,2,0) */
buffer_load_short_d16 v54, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(0,9,3,0) */
buffer_load_short_d16 v55, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:192 // load C
/* (d1,vc1,d0,vc0)=(0,10,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v56, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,10,1,0) */
buffer_load_short_d16 v57, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(0,10,2,0) */
buffer_load_short_d16 v58, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(0,10,3,0) */
buffer_load_short_d16 v59, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:192 // load C
/* (d1,vc1,d0,vc0)=(0,11,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v60, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,11,1,0) */
buffer_load_short_d16 v61, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(0,11,2,0) */
buffer_load_short_d16 v62, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(0,11,3,0) */
buffer_load_short_d16 v63, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:192 // load C
v_accvgpr_read_b32 v[vgprValuC+24], acc2           // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+25], acc6           // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+26], acc10          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+27], acc14          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+28], acc18          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+29], acc22          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+30], acc26          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+31], acc30          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+32], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+33], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+34], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+35], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+36], acc50          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+37], acc54          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+38], acc58          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+39], acc62          // copy acc to vreg[47]

/* rC *= alpha batchElements=[(0, 0, 8, 0), (0, 1, 8, 0), (0, 2, 8, 0), (0, 3, 8, 0), (0, 0, 9, 0), (0, 1, 9, 0), (0, 2, 9, 0), (0, 3, 9, 0), (0, 0, 10, 0), (0, 1, 10, 0), (0, 2, 10, 0), (0, 3, 10, 0), (0, 0, 11, 0), (0, 1, 11, 0), (0, 2, 11, 0), (0, 3, 11, 0)] */
v_pk_mul_f32 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+24:vgprValuC+24+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+26:vgprValuC+26+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+28:vgprValuC+28+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+30:vgprValuC+30+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+32:vgprValuC+32+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+34:vgprValuC+34+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+36:vgprValuC+36+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+36:vgprValuC+36+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+38:vgprValuC+38+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+38:vgprValuC+38+1] op_sel_hi:[0,1,1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */
v_mov_b32 v18, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v19, 0x7fff0000                          // fp32 Nan
v_mov_b32 v20, 0x7fff                              // rounding bias for bfloat16

s_waitcnt lgkmcnt(6), vmcnt(15)                    // vlcnt(15) = 16 - 1 (beta) vscnt(0) dscnt(6) = 8 - 1 (bias) - 1 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+24], v42, v[vgprValuC+24]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v40 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+24], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+24], v41, v[vgprValuC+24]    // C += bias
v_cvt_pk_bf16_f32 v24, v[vgprValuC+24], v[vgprValuC+24] // convert C to bf16 in gwvw==1
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v24, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt lgkmcnt(4), vmcnt(15)                    // vlcnt(14) = 16 - 2 (beta) vscnt(1) dscnt(4) = 8 - 2 (bias) - 2 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+25], v45, v[vgprValuC+25]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v43 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+25], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+25], v44, v[vgprValuC+25]    // C += bias
v_cvt_pk_bf16_f32 v25, v[vgprValuC+25], v[vgprValuC+25] // convert C to bf16 in gwvw==1
buffer_store_short v25, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt lgkmcnt(2), vmcnt(15)                    // vlcnt(13) = 16 - 3 (beta) vscnt(2) dscnt(2) = 8 - 3 (bias) - 3 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+26], v48, v[vgprValuC+26]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v46 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+26], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+26], v47, v[vgprValuC+26]    // C += bias
v_cvt_pk_bf16_f32 v26, v[vgprValuC+26], v[vgprValuC+26] // convert C to bf16 in gwvw==1
buffer_store_short v26, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt lgkmcnt(0), vmcnt(15)                    // vlcnt(12) = 16 - 4 (beta) vscnt(3) dscnt(0) = 8 - 4 (bias) - 4 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+27], v51, v[vgprValuC+27]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v49 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+27], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+27], v50, v[vgprValuC+27]    // C += bias
v_cvt_pk_bf16_f32 v27, v[vgprValuC+27], v[vgprValuC+27] // convert C to bf16 in gwvw==1
buffer_store_short v27, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D

s_waitcnt vmcnt(15)                                // vlcnt(11) = 16 - 5 (beta) vscnt(4) (interleaved)
v_mul_f32 v[vgprValuC+28], v42, v[vgprValuC+28]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v52 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+28], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+28], v41, v[vgprValuC+28]    // C += bias
v_cvt_pk_bf16_f32 v28, v[vgprValuC+28], v[vgprValuC+28] // convert C to bf16 in gwvw==1
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v28, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(10) = 16 - 6 (beta) vscnt(5) (interleaved)
v_mul_f32 v[vgprValuC+29], v45, v[vgprValuC+29]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v53 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+29], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+29], v44, v[vgprValuC+29]    // C += bias
v_cvt_pk_bf16_f32 v29, v[vgprValuC+29], v[vgprValuC+29] // convert C to bf16 in gwvw==1
buffer_store_short v29, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(15)                                // vlcnt(9) = 16 - 7 (beta) vscnt(6) (interleaved)
v_mul_f32 v[vgprValuC+30], v48, v[vgprValuC+30]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v54 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+30], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+30], v47, v[vgprValuC+30]    // C += bias
v_cvt_pk_bf16_f32 v30, v[vgprValuC+30], v[vgprValuC+30] // convert C to bf16 in gwvw==1
buffer_store_short v30, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(15)                                // vlcnt(8) = 16 - 8 (beta) vscnt(7) (interleaved)
v_mul_f32 v[vgprValuC+31], v51, v[vgprValuC+31]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v55 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+31], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+31], v50, v[vgprValuC+31]    // C += bias
v_cvt_pk_bf16_f32 v31, v[vgprValuC+31], v[vgprValuC+31] // convert C to bf16 in gwvw==1
buffer_store_short v31, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D

s_waitcnt vmcnt(15)                                // vlcnt(7) = 16 - 9 (beta) vscnt(8) (interleaved)
v_mul_f32 v[vgprValuC+32], v42, v[vgprValuC+32]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v56 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+32], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+32], v41, v[vgprValuC+32]    // C += bias
v_cvt_pk_bf16_f32 v32, v[vgprValuC+32], v[vgprValuC+32] // convert C to bf16 in gwvw==1
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v32, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(6) = 16 - 10 (beta) vscnt(9) (interleaved)
v_mul_f32 v[vgprValuC+33], v45, v[vgprValuC+33]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v57 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+33], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+33], v44, v[vgprValuC+33]    // C += bias
v_cvt_pk_bf16_f32 v33, v[vgprValuC+33], v[vgprValuC+33] // convert C to bf16 in gwvw==1
buffer_store_short v33, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(15)                                // vlcnt(5) = 16 - 11 (beta) vscnt(10) (interleaved)
v_mul_f32 v[vgprValuC+34], v48, v[vgprValuC+34]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v58 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+34], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+34], v47, v[vgprValuC+34]    // C += bias
v_cvt_pk_bf16_f32 v34, v[vgprValuC+34], v[vgprValuC+34] // convert C to bf16 in gwvw==1
buffer_store_short v34, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(15)                                // vlcnt(4) = 16 - 12 (beta) vscnt(11) (interleaved)
v_mul_f32 v[vgprValuC+35], v51, v[vgprValuC+35]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v59 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+35], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+35], v50, v[vgprValuC+35]    // C += bias
v_cvt_pk_bf16_f32 v35, v[vgprValuC+35], v[vgprValuC+35] // convert C to bf16 in gwvw==1
buffer_store_short v35, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D

s_waitcnt vmcnt(15)                                // vlcnt(3) = 16 - 13 (beta) vscnt(12) (interleaved)
v_mul_f32 v[vgprValuC+36], v42, v[vgprValuC+36]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v60 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+36], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+36], v41, v[vgprValuC+36]    // C += bias
v_cvt_pk_bf16_f32 v36, v[vgprValuC+36], v[vgprValuC+36] // convert C to bf16 in gwvw==1
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v36, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(2) = 16 - 14 (beta) vscnt(13) (interleaved)
v_mul_f32 v[vgprValuC+37], v45, v[vgprValuC+37]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v61 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+37], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+37], v44, v[vgprValuC+37]    // C += bias
v_cvt_pk_bf16_f32 v37, v[vgprValuC+37], v[vgprValuC+37] // convert C to bf16 in gwvw==1
buffer_store_short v37, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(15)                                // vlcnt(1) = 16 - 15 (beta) vscnt(14) (interleaved)
v_mul_f32 v[vgprValuC+38], v48, v[vgprValuC+38]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v62 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+38], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+38], v47, v[vgprValuC+38]    // C += bias
v_cvt_pk_bf16_f32 v38, v[vgprValuC+38], v[vgprValuC+38] // convert C to bf16 in gwvw==1
buffer_store_short v38, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(15)                                // vlcnt(0) = 16 - 16 (beta) vscnt(15) (interleaved)
v_mul_f32 v[vgprValuC+39], v51, v[vgprValuC+39]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v63 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+39], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+39], v50, v[vgprValuC+39]    // C += bias
v_cvt_pk_bf16_f32 v39, v[vgprValuC+39], v[vgprValuC+39] // convert C to bf16 in gwvw==1
buffer_store_short v39, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #3 (d1,d0,vc1,vc0) = */
/*    (0,0,12,0:vw1); (0,1,12,0:vw1); (0,2,12,0:vw1); (0,3,12,0:vw1); (0,0,13,0:vw1); (0,1,13,0:vw1); (0,2,13,0:vw1); (0,3,13,0:vw1); (0,0,14,0:vw1); (0,1,14,0:vw1); (0,2,14,0:vw1); (0,3,14,0:vw1); (0,0,15,0:vw1); (0,1,15,0:vw1); (0,2,15,0:vw1); (0,3,15,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,12,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v40, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
ds_read_b32 v41, v23 offset:0                      // load Bias
ds_read_b32 v42, v23 offset:1024                   // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,12,1,0) */
buffer_load_short_d16 v43, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
ds_read_b32 v44, v23 offset:128                    // load Bias
ds_read_b32 v45, v23 offset:1152                   // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,12,2,0) */
buffer_load_short_d16 v46, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
ds_read_b32 v47, v23 offset:256                    // load Bias
ds_read_b32 v48, v23 offset:1280                   // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,12,3,0) */
buffer_load_short_d16 v49, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:192 // load C
ds_read_b32 v50, v23 offset:384                    // load Bias
ds_read_b32 v51, v23 offset:1408                   // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,13,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v52, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,13,1,0) */
buffer_load_short_d16 v53, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(0,13,2,0) */
buffer_load_short_d16 v54, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(0,13,3,0) */
buffer_load_short_d16 v55, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:192 // load C
/* (d1,vc1,d0,vc0)=(0,14,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v56, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,14,1,0) */
buffer_load_short_d16 v57, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(0,14,2,0) */
buffer_load_short_d16 v58, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(0,14,3,0) */
buffer_load_short_d16 v59, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:192 // load C
/* (d1,vc1,d0,vc0)=(0,15,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v60, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,15,1,0) */
buffer_load_short_d16 v61, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(0,15,2,0) */
buffer_load_short_d16 v62, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(0,15,3,0) */
buffer_load_short_d16 v63, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:192 // load C
v_accvgpr_read_b32 v[vgprValuC+24], acc3           // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+25], acc7           // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+26], acc11          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+27], acc15          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+28], acc19          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+29], acc23          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+30], acc27          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+31], acc31          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+32], acc35          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+33], acc39          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+34], acc43          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+35], acc47          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+36], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+37], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+38], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+39], acc63          // copy acc to vreg[63]

/* rC *= alpha batchElements=[(0, 0, 12, 0), (0, 1, 12, 0), (0, 2, 12, 0), (0, 3, 12, 0), (0, 0, 13, 0), (0, 1, 13, 0), (0, 2, 13, 0), (0, 3, 13, 0), (0, 0, 14, 0), (0, 1, 14, 0), (0, 2, 14, 0), (0, 3, 14, 0), (0, 0, 15, 0), (0, 1, 15, 0), (0, 2, 15, 0), (0, 3, 15, 0)] */
v_pk_mul_f32 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+24:vgprValuC+24+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+26:vgprValuC+26+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+28:vgprValuC+28+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+30:vgprValuC+30+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+32:vgprValuC+32+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+34:vgprValuC+34+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+36:vgprValuC+36+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+36:vgprValuC+36+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+38:vgprValuC+38+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+38:vgprValuC+38+1] op_sel_hi:[0,1,1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */
v_mov_b32 v18, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v19, 0x7fff0000                          // fp32 Nan
v_mov_b32 v20, 0x7fff                              // rounding bias for bfloat16

s_waitcnt lgkmcnt(6), vmcnt(15)                    // vlcnt(15) = 16 - 1 (beta) vscnt(0) dscnt(6) = 8 - 1 (bias) - 1 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+24], v42, v[vgprValuC+24]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v40 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+24], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+24], v41, v[vgprValuC+24]    // C += bias
v_cvt_pk_bf16_f32 v24, v[vgprValuC+24], v[vgprValuC+24] // convert C to bf16 in gwvw==1
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v24, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt lgkmcnt(4), vmcnt(15)                    // vlcnt(14) = 16 - 2 (beta) vscnt(1) dscnt(4) = 8 - 2 (bias) - 2 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+25], v45, v[vgprValuC+25]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v43 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+25], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+25], v44, v[vgprValuC+25]    // C += bias
v_cvt_pk_bf16_f32 v25, v[vgprValuC+25], v[vgprValuC+25] // convert C to bf16 in gwvw==1
buffer_store_short v25, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt lgkmcnt(2), vmcnt(15)                    // vlcnt(13) = 16 - 3 (beta) vscnt(2) dscnt(2) = 8 - 3 (bias) - 3 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+26], v48, v[vgprValuC+26]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v46 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+26], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+26], v47, v[vgprValuC+26]    // C += bias
v_cvt_pk_bf16_f32 v26, v[vgprValuC+26], v[vgprValuC+26] // convert C to bf16 in gwvw==1
buffer_store_short v26, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt lgkmcnt(0), vmcnt(15)                    // vlcnt(12) = 16 - 4 (beta) vscnt(3) dscnt(0) = 8 - 4 (bias) - 4 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+27], v51, v[vgprValuC+27]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v49 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+27], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+27], v50, v[vgprValuC+27]    // C += bias
v_cvt_pk_bf16_f32 v27, v[vgprValuC+27], v[vgprValuC+27] // convert C to bf16 in gwvw==1
buffer_store_short v27, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D

s_waitcnt vmcnt(15)                                // vlcnt(11) = 16 - 5 (beta) vscnt(4) (interleaved)
v_mul_f32 v[vgprValuC+28], v42, v[vgprValuC+28]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v52 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+28], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+28], v41, v[vgprValuC+28]    // C += bias
v_cvt_pk_bf16_f32 v28, v[vgprValuC+28], v[vgprValuC+28] // convert C to bf16 in gwvw==1
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v28, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(10) = 16 - 6 (beta) vscnt(5) (interleaved)
v_mul_f32 v[vgprValuC+29], v45, v[vgprValuC+29]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v53 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+29], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+29], v44, v[vgprValuC+29]    // C += bias
v_cvt_pk_bf16_f32 v29, v[vgprValuC+29], v[vgprValuC+29] // convert C to bf16 in gwvw==1
buffer_store_short v29, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(15)                                // vlcnt(9) = 16 - 7 (beta) vscnt(6) (interleaved)
v_mul_f32 v[vgprValuC+30], v48, v[vgprValuC+30]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v54 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+30], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+30], v47, v[vgprValuC+30]    // C += bias
v_cvt_pk_bf16_f32 v30, v[vgprValuC+30], v[vgprValuC+30] // convert C to bf16 in gwvw==1
buffer_store_short v30, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(15)                                // vlcnt(8) = 16 - 8 (beta) vscnt(7) (interleaved)
v_mul_f32 v[vgprValuC+31], v51, v[vgprValuC+31]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v55 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+31], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+31], v50, v[vgprValuC+31]    // C += bias
v_cvt_pk_bf16_f32 v31, v[vgprValuC+31], v[vgprValuC+31] // convert C to bf16 in gwvw==1
buffer_store_short v31, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D

s_waitcnt vmcnt(15)                                // vlcnt(7) = 16 - 9 (beta) vscnt(8) (interleaved)
v_mul_f32 v[vgprValuC+32], v42, v[vgprValuC+32]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v56 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+32], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+32], v41, v[vgprValuC+32]    // C += bias
v_cvt_pk_bf16_f32 v32, v[vgprValuC+32], v[vgprValuC+32] // convert C to bf16 in gwvw==1
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v32, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(6) = 16 - 10 (beta) vscnt(9) (interleaved)
v_mul_f32 v[vgprValuC+33], v45, v[vgprValuC+33]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v57 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+33], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+33], v44, v[vgprValuC+33]    // C += bias
v_cvt_pk_bf16_f32 v33, v[vgprValuC+33], v[vgprValuC+33] // convert C to bf16 in gwvw==1
buffer_store_short v33, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(15)                                // vlcnt(5) = 16 - 11 (beta) vscnt(10) (interleaved)
v_mul_f32 v[vgprValuC+34], v48, v[vgprValuC+34]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v58 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+34], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+34], v47, v[vgprValuC+34]    // C += bias
v_cvt_pk_bf16_f32 v34, v[vgprValuC+34], v[vgprValuC+34] // convert C to bf16 in gwvw==1
buffer_store_short v34, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(15)                                // vlcnt(4) = 16 - 12 (beta) vscnt(11) (interleaved)
v_mul_f32 v[vgprValuC+35], v51, v[vgprValuC+35]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v59 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+35], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+35], v50, v[vgprValuC+35]    // C += bias
v_cvt_pk_bf16_f32 v35, v[vgprValuC+35], v[vgprValuC+35] // convert C to bf16 in gwvw==1
buffer_store_short v35, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D

s_waitcnt vmcnt(15)                                // vlcnt(3) = 16 - 13 (beta) vscnt(12) (interleaved)
v_mul_f32 v[vgprValuC+36], v42, v[vgprValuC+36]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v60 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+36], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+36], v41, v[vgprValuC+36]    // C += bias
v_cvt_pk_bf16_f32 v36, v[vgprValuC+36], v[vgprValuC+36] // convert C to bf16 in gwvw==1
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v36, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(2) = 16 - 14 (beta) vscnt(13) (interleaved)
v_mul_f32 v[vgprValuC+37], v45, v[vgprValuC+37]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v61 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+37], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+37], v44, v[vgprValuC+37]    // C += bias
v_cvt_pk_bf16_f32 v37, v[vgprValuC+37], v[vgprValuC+37] // convert C to bf16 in gwvw==1
buffer_store_short v37, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(15)                                // vlcnt(1) = 16 - 15 (beta) vscnt(14) (interleaved)
v_mul_f32 v[vgprValuC+38], v48, v[vgprValuC+38]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v62 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+38], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+38], v47, v[vgprValuC+38]    // C += bias
v_cvt_pk_bf16_f32 v38, v[vgprValuC+38], v[vgprValuC+38] // convert C to bf16 in gwvw==1
buffer_store_short v38, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(15)                                // vlcnt(0) = 16 - 16 (beta) vscnt(15) (interleaved)
v_mul_f32 v[vgprValuC+39], v51, v[vgprValuC+39]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v63 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+39], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+39], v50, v[vgprValuC+39]    // C += bias
v_cvt_pk_bf16_f32 v39, v[vgprValuC+39], v[vgprValuC+39] // convert C to bf16 in gwvw==1
buffer_store_short v39, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_E1:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=16 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,0,1,0:vw1); (0,1,1,0:vw1); (0,2,1,0:vw1); (0,3,1,0:vw1); (0,0,2,0:vw1); (0,1,2,0:vw1); (0,2,2,0:vw1); (0,3,2,0:vw1); (0,0,3,0:vw1); (0,1,3,0:vw1); (0,2,3,0:vw1); (0,3,3,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v16, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v40, v12, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v40, v16, v40, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v37, v40, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v41, v10, s70
v_lshlrev_b32 v41, 0x2, v41                        // Bias address scaled by BPE
s_waitcnt lgkmcnt(0)                               // Wait for LDS write
s_barrier                                          // LDS write barrier
ds_read_b32 v38, v41 offset:0                      // load Bias
ds_read_b32 v39, v41 offset:1024                   // load scaleAlpha
v_add_lshl_u32 v40, v13, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v40, v16, v40, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v45, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v45, v16, v45, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v42, v45, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v46, v14, s70
v_lshlrev_b32 v46, 0x2, v46                        // Bias address scaled by BPE
ds_read_b32 v43, v46 offset:0                      // load Bias
ds_read_b32 v44, v46 offset:1024                   // load scaleAlpha
v_add_lshl_u32 v45, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v45, v16, v45, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v50, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v50, v16, v50, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v47, v50, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v51, v14, s70
v_lshlrev_b32 v51, 0x2, v51                        // Bias address scaled by BPE
ds_read_b32 v48, v51 offset:0                      // load Bias
ds_read_b32 v49, v51 offset:1024                   // load scaleAlpha
v_add_lshl_u32 v50, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v50, v16, v50, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v55, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v55, v16, v55, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v52, v55, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v56, v14, s70
v_lshlrev_b32 v56, 0x2, v56                        // Bias address scaled by BPE
ds_read_b32 v53, v56 offset:0                      // load Bias
ds_read_b32 v54, v56 offset:1024                   // load scaleAlpha
v_add_lshl_u32 v55, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v55, v16, v55, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v58, v12, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v58, v16, v58, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v57, v58, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v59, v10, s70
v_lshlrev_b32 v59, 0x2, v59                        // Bias address scaled by BPE
v_add_lshl_u32 v58, v13, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v58, v16, v58, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v61, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v61, v16, v61, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v60, v61, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v62, v14, s70
v_lshlrev_b32 v62, 0x2, v62                        // Bias address scaled by BPE
v_add_lshl_u32 v61, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v61, v16, v61, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v64, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v64, v16, v64, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v63, v64, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v65, v14, s70
v_lshlrev_b32 v65, 0x2, v65                        // Bias address scaled by BPE
v_add_lshl_u32 v64, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v64, v16, v64, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v67, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v67, v16, v67, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v66, v67, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v68, v14, s70
v_lshlrev_b32 v68, 0x2, v68                        // Bias address scaled by BPE
v_add_lshl_u32 v67, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v67, v16, v67, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v70, v12, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v70, v16, v70, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v69, v70, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v71, v10, s70
v_lshlrev_b32 v71, 0x2, v71                        // Bias address scaled by BPE
v_add_lshl_u32 v70, v13, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v70, v16, v70, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v73, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v73, v16, v73, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v72, v73, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v75, v14, s70
v_lshlrev_b32 v75, 0x2, v75                        // Bias address scaled by BPE
v_add_lshl_u32 v73, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v73, v16, v73, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v77, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v77, v16, v77, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v76, v77, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v78, v14, s70
v_lshlrev_b32 v78, 0x2, v78                        // Bias address scaled by BPE
v_add_lshl_u32 v77, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v77, v16, v77, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v80, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, v16, v80, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v79, v80, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v81, v14, s70
v_lshlrev_b32 v81, 0x2, v81                        // Bias address scaled by BPE
v_add_lshl_u32 v80, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, v16, v80, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v83, v12, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v83, v16, v83, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v82, v83, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v84, v10, s70
v_lshlrev_b32 v84, 0x2, v84                        // Bias address scaled by BPE
v_add_lshl_u32 v83, v13, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v83, v16, v83, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v86, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, v16, v86, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v85, v86, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v87, v14, s70
v_lshlrev_b32 v87, 0x2, v87                        // Bias address scaled by BPE
v_add_lshl_u32 v86, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, v16, v86, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v89, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v89, v16, v89, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v88, v89, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v90, v14, s70
v_lshlrev_b32 v90, 0x2, v90                        // Bias address scaled by BPE
v_add_lshl_u32 v89, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v89, v16, v89, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v92, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v92, v16, v92, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v91, v92, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v93, v14, s70
v_lshlrev_b32 v93, 0x2, v93                        // Bias address scaled by BPE
v_add_lshl_u32 v92, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v92, v16, v92, s[74:75]              // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+21], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+22], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+23], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+24], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+25], acc16          // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+26], acc20          // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+27], acc24          // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+28], acc28          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+29], acc32          // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+30], acc36          // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+31], acc40          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+32], acc44          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+33], acc48          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+34], acc52          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+35], acc56          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+36], acc60          // copy acc to vreg[15]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 0, 1, 0), (0, 1, 1, 0), (0, 2, 1, 0), (0, 3, 1, 0), (0, 0, 2, 0), (0, 1, 2, 0), (0, 2, 2, 0), (0, 3, 2, 0), (0, 0, 3, 0), (0, 1, 3, 0), (0, 2, 3, 0), (0, 3, 3, 0)] */
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+21] // *= alpha
v_pk_mul_f32 v[vgprValuC+22:vgprValuC+22+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+22:vgprValuC+22+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+24:vgprValuC+24+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+26:vgprValuC+26+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+28:vgprValuC+28+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+30:vgprValuC+30+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+32:vgprValuC+32+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+34:vgprValuC+34+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+36], s[sgprAlpha], v[vgprValuC+36] // *= alpha
s_waitcnt 0                                        // wait for Beta, Bias LDS, ScaleAlphaVec

/* apply mask, calc new C and issue writes */
v_mov_b32 v18, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v19, 0x7fff0000                          // fp32 Nan
v_mov_b32 v20, 0x7fff                              // rounding bias for bfloat16
v_mul_f32 v[vgprValuC+21], v39, v[vgprValuC+21]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v37 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+21], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+21], v38, v[vgprValuC+21]    // C += bias
v_cvt_pk_bf16_f32 v21, v[vgprValuC+21], v[vgprValuC+21] // convert C to bf16 in gwvw==1
buffer_store_short v21, v40, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+22], v44, v[vgprValuC+22]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v42 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+22], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+22], v43, v[vgprValuC+22]    // C += bias
v_cvt_pk_bf16_f32 v22, v[vgprValuC+22], v[vgprValuC+22] // convert C to bf16 in gwvw==1
buffer_store_short v22, v45, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+23], v49, v[vgprValuC+23]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v47 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+23], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+23], v48, v[vgprValuC+23]    // C += bias
v_cvt_pk_bf16_f32 v23, v[vgprValuC+23], v[vgprValuC+23] // convert C to bf16 in gwvw==1
buffer_store_short v23, v50, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+24], v54, v[vgprValuC+24]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v52 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+24], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+24], v53, v[vgprValuC+24]    // C += bias
v_cvt_pk_bf16_f32 v24, v[vgprValuC+24], v[vgprValuC+24] // convert C to bf16 in gwvw==1
buffer_store_short v24, v55, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+25], v39, v[vgprValuC+25]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v57 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+25], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+25], v38, v[vgprValuC+25]    // C += bias
v_cvt_pk_bf16_f32 v25, v[vgprValuC+25], v[vgprValuC+25] // convert C to bf16 in gwvw==1
buffer_store_short v25, v58, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+26], v44, v[vgprValuC+26]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v60 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+26], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+26], v43, v[vgprValuC+26]    // C += bias
v_cvt_pk_bf16_f32 v26, v[vgprValuC+26], v[vgprValuC+26] // convert C to bf16 in gwvw==1
buffer_store_short v26, v61, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+27], v49, v[vgprValuC+27]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v63 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+27], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+27], v48, v[vgprValuC+27]    // C += bias
v_cvt_pk_bf16_f32 v27, v[vgprValuC+27], v[vgprValuC+27] // convert C to bf16 in gwvw==1
buffer_store_short v27, v64, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+28], v54, v[vgprValuC+28]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v66 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+28], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+28], v53, v[vgprValuC+28]    // C += bias
v_cvt_pk_bf16_f32 v28, v[vgprValuC+28], v[vgprValuC+28] // convert C to bf16 in gwvw==1
buffer_store_short v28, v67, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+29], v39, v[vgprValuC+29]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v69 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+29], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+29], v38, v[vgprValuC+29]    // C += bias
v_cvt_pk_bf16_f32 v29, v[vgprValuC+29], v[vgprValuC+29] // convert C to bf16 in gwvw==1
buffer_store_short v29, v70, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+30], v44, v[vgprValuC+30]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v72 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+30], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+30], v43, v[vgprValuC+30]    // C += bias
v_cvt_pk_bf16_f32 v30, v[vgprValuC+30], v[vgprValuC+30] // convert C to bf16 in gwvw==1
buffer_store_short v30, v73, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+31], v49, v[vgprValuC+31]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v76 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+31], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+31], v48, v[vgprValuC+31]    // C += bias
v_cvt_pk_bf16_f32 v31, v[vgprValuC+31], v[vgprValuC+31] // convert C to bf16 in gwvw==1
buffer_store_short v31, v77, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+32], v54, v[vgprValuC+32]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v79 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+32], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+32], v53, v[vgprValuC+32]    // C += bias
v_cvt_pk_bf16_f32 v32, v[vgprValuC+32], v[vgprValuC+32] // convert C to bf16 in gwvw==1
buffer_store_short v32, v80, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+33], v39, v[vgprValuC+33]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v82 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+33], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+33], v38, v[vgprValuC+33]    // C += bias
v_cvt_pk_bf16_f32 v33, v[vgprValuC+33], v[vgprValuC+33] // convert C to bf16 in gwvw==1
buffer_store_short v33, v83, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+34], v44, v[vgprValuC+34]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v85 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+34], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+34], v43, v[vgprValuC+34]    // C += bias
v_cvt_pk_bf16_f32 v34, v[vgprValuC+34], v[vgprValuC+34] // convert C to bf16 in gwvw==1
buffer_store_short v34, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+35], v49, v[vgprValuC+35]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v88 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+35], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+35], v48, v[vgprValuC+35]    // C += bias
v_cvt_pk_bf16_f32 v35, v[vgprValuC+35], v[vgprValuC+35] // convert C to bf16 in gwvw==1
buffer_store_short v35, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+36], v54, v[vgprValuC+36]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v91 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+36], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+36], v53, v[vgprValuC+36]    // C += bias
v_cvt_pk_bf16_f32 v36, v[vgprValuC+36], v[vgprValuC+36] // convert C to bf16 in gwvw==1
buffer_store_short v36, v92, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (0,0,4,0:vw1); (0,1,4,0:vw1); (0,2,4,0:vw1); (0,3,4,0:vw1); (0,0,5,0:vw1); (0,1,5,0:vw1); (0,2,5,0:vw1); (0,3,5,0:vw1); (0,0,6,0:vw1); (0,1,6,0:vw1); (0,2,6,0:vw1); (0,3,6,0:vw1); (0,0,7,0:vw1); (0,1,7,0:vw1); (0,2,7,0:vw1); (0,3,7,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v16, BufferOOB
/* (d1,vc1,d0,vc0)=(0,4,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v40, v12, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v40, v16, v40, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v37, v40, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v41, v10, s70
v_lshlrev_b32 v41, 0x2, v41                        // Bias address scaled by BPE
ds_read_b32 v38, v41 offset:0                      // load Bias
ds_read_b32 v39, v41 offset:1024                   // load scaleAlpha
v_add_lshl_u32 v40, v13, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v40, v16, v40, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v45, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v45, v16, v45, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v42, v45, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v46, v14, s70
v_lshlrev_b32 v46, 0x2, v46                        // Bias address scaled by BPE
ds_read_b32 v43, v46 offset:0                      // load Bias
ds_read_b32 v44, v46 offset:1024                   // load scaleAlpha
v_add_lshl_u32 v45, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v45, v16, v45, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v50, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v50, v16, v50, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v47, v50, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v51, v14, s70
v_lshlrev_b32 v51, 0x2, v51                        // Bias address scaled by BPE
ds_read_b32 v48, v51 offset:0                      // load Bias
ds_read_b32 v49, v51 offset:1024                   // load scaleAlpha
v_add_lshl_u32 v50, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v50, v16, v50, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v55, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v55, v16, v55, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v52, v55, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v56, v14, s70
v_lshlrev_b32 v56, 0x2, v56                        // Bias address scaled by BPE
ds_read_b32 v53, v56 offset:0                      // load Bias
ds_read_b32 v54, v56 offset:1024                   // load scaleAlpha
v_add_lshl_u32 v55, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v55, v16, v55, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v58, v12, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v58, v16, v58, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v57, v58, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v59, v10, s70
v_lshlrev_b32 v59, 0x2, v59                        // Bias address scaled by BPE
v_add_lshl_u32 v58, v13, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v58, v16, v58, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v61, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v61, v16, v61, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v60, v61, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v62, v14, s70
v_lshlrev_b32 v62, 0x2, v62                        // Bias address scaled by BPE
v_add_lshl_u32 v61, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v61, v16, v61, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v64, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v64, v16, v64, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v63, v64, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v65, v14, s70
v_lshlrev_b32 v65, 0x2, v65                        // Bias address scaled by BPE
v_add_lshl_u32 v64, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v64, v16, v64, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v67, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v67, v16, v67, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v66, v67, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v68, v14, s70
v_lshlrev_b32 v68, 0x2, v68                        // Bias address scaled by BPE
v_add_lshl_u32 v67, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v67, v16, v67, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v70, v12, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v70, v16, v70, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v69, v70, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v71, v10, s70
v_lshlrev_b32 v71, 0x2, v71                        // Bias address scaled by BPE
v_add_lshl_u32 v70, v13, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v70, v16, v70, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v73, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v73, v16, v73, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v72, v73, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v75, v14, s70
v_lshlrev_b32 v75, 0x2, v75                        // Bias address scaled by BPE
v_add_lshl_u32 v73, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v73, v16, v73, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v77, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v77, v16, v77, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v76, v77, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v78, v14, s70
v_lshlrev_b32 v78, 0x2, v78                        // Bias address scaled by BPE
v_add_lshl_u32 v77, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v77, v16, v77, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v80, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, v16, v80, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v79, v80, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v81, v14, s70
v_lshlrev_b32 v81, 0x2, v81                        // Bias address scaled by BPE
v_add_lshl_u32 v80, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, v16, v80, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v83, v12, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v83, v16, v83, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v82, v83, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v84, v10, s70
v_lshlrev_b32 v84, 0x2, v84                        // Bias address scaled by BPE
v_add_lshl_u32 v83, v13, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v83, v16, v83, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v86, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, v16, v86, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v85, v86, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v87, v14, s70
v_lshlrev_b32 v87, 0x2, v87                        // Bias address scaled by BPE
v_add_lshl_u32 v86, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, v16, v86, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v89, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v89, v16, v89, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v88, v89, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v90, v14, s70
v_lshlrev_b32 v90, 0x2, v90                        // Bias address scaled by BPE
v_add_lshl_u32 v89, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v89, v16, v89, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v92, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v92, v16, v92, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v91, v92, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v93, v14, s70
v_lshlrev_b32 v93, 0x2, v93                        // Bias address scaled by BPE
v_add_lshl_u32 v92, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v92, v16, v92, s[74:75]              // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+21], acc1           // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+22], acc5           // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+23], acc9           // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+24], acc13          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+25], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+26], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+27], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+28], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+29], acc33          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+30], acc37          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+31], acc41          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+32], acc45          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+33], acc49          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+34], acc53          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+35], acc57          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+36], acc61          // copy acc to vreg[31]

/* rC *= alpha batchElements=[(0, 0, 4, 0), (0, 1, 4, 0), (0, 2, 4, 0), (0, 3, 4, 0), (0, 0, 5, 0), (0, 1, 5, 0), (0, 2, 5, 0), (0, 3, 5, 0), (0, 0, 6, 0), (0, 1, 6, 0), (0, 2, 6, 0), (0, 3, 6, 0), (0, 0, 7, 0), (0, 1, 7, 0), (0, 2, 7, 0), (0, 3, 7, 0)] */
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+21] // *= alpha
v_pk_mul_f32 v[vgprValuC+22:vgprValuC+22+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+22:vgprValuC+22+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+24:vgprValuC+24+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+26:vgprValuC+26+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+28:vgprValuC+28+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+30:vgprValuC+30+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+32:vgprValuC+32+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+34:vgprValuC+34+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+36], s[sgprAlpha], v[vgprValuC+36] // *= alpha
s_waitcnt 0                                        // wait for Beta, Bias LDS, ScaleAlphaVec

/* apply mask, calc new C and issue writes */
v_mov_b32 v18, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v19, 0x7fff0000                          // fp32 Nan
v_mov_b32 v20, 0x7fff                              // rounding bias for bfloat16
v_mul_f32 v[vgprValuC+21], v39, v[vgprValuC+21]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v37 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+21], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+21], v38, v[vgprValuC+21]    // C += bias
v_cvt_pk_bf16_f32 v21, v[vgprValuC+21], v[vgprValuC+21] // convert C to bf16 in gwvw==1
buffer_store_short v21, v40, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+22], v44, v[vgprValuC+22]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v42 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+22], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+22], v43, v[vgprValuC+22]    // C += bias
v_cvt_pk_bf16_f32 v22, v[vgprValuC+22], v[vgprValuC+22] // convert C to bf16 in gwvw==1
buffer_store_short v22, v45, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+23], v49, v[vgprValuC+23]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v47 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+23], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+23], v48, v[vgprValuC+23]    // C += bias
v_cvt_pk_bf16_f32 v23, v[vgprValuC+23], v[vgprValuC+23] // convert C to bf16 in gwvw==1
buffer_store_short v23, v50, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+24], v54, v[vgprValuC+24]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v52 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+24], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+24], v53, v[vgprValuC+24]    // C += bias
v_cvt_pk_bf16_f32 v24, v[vgprValuC+24], v[vgprValuC+24] // convert C to bf16 in gwvw==1
buffer_store_short v24, v55, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+25], v39, v[vgprValuC+25]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v57 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+25], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+25], v38, v[vgprValuC+25]    // C += bias
v_cvt_pk_bf16_f32 v25, v[vgprValuC+25], v[vgprValuC+25] // convert C to bf16 in gwvw==1
buffer_store_short v25, v58, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+26], v44, v[vgprValuC+26]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v60 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+26], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+26], v43, v[vgprValuC+26]    // C += bias
v_cvt_pk_bf16_f32 v26, v[vgprValuC+26], v[vgprValuC+26] // convert C to bf16 in gwvw==1
buffer_store_short v26, v61, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+27], v49, v[vgprValuC+27]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v63 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+27], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+27], v48, v[vgprValuC+27]    // C += bias
v_cvt_pk_bf16_f32 v27, v[vgprValuC+27], v[vgprValuC+27] // convert C to bf16 in gwvw==1
buffer_store_short v27, v64, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+28], v54, v[vgprValuC+28]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v66 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+28], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+28], v53, v[vgprValuC+28]    // C += bias
v_cvt_pk_bf16_f32 v28, v[vgprValuC+28], v[vgprValuC+28] // convert C to bf16 in gwvw==1
buffer_store_short v28, v67, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+29], v39, v[vgprValuC+29]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v69 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+29], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+29], v38, v[vgprValuC+29]    // C += bias
v_cvt_pk_bf16_f32 v29, v[vgprValuC+29], v[vgprValuC+29] // convert C to bf16 in gwvw==1
buffer_store_short v29, v70, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+30], v44, v[vgprValuC+30]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v72 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+30], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+30], v43, v[vgprValuC+30]    // C += bias
v_cvt_pk_bf16_f32 v30, v[vgprValuC+30], v[vgprValuC+30] // convert C to bf16 in gwvw==1
buffer_store_short v30, v73, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+31], v49, v[vgprValuC+31]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v76 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+31], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+31], v48, v[vgprValuC+31]    // C += bias
v_cvt_pk_bf16_f32 v31, v[vgprValuC+31], v[vgprValuC+31] // convert C to bf16 in gwvw==1
buffer_store_short v31, v77, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+32], v54, v[vgprValuC+32]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v79 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+32], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+32], v53, v[vgprValuC+32]    // C += bias
v_cvt_pk_bf16_f32 v32, v[vgprValuC+32], v[vgprValuC+32] // convert C to bf16 in gwvw==1
buffer_store_short v32, v80, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+33], v39, v[vgprValuC+33]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v82 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+33], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+33], v38, v[vgprValuC+33]    // C += bias
v_cvt_pk_bf16_f32 v33, v[vgprValuC+33], v[vgprValuC+33] // convert C to bf16 in gwvw==1
buffer_store_short v33, v83, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+34], v44, v[vgprValuC+34]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v85 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+34], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+34], v43, v[vgprValuC+34]    // C += bias
v_cvt_pk_bf16_f32 v34, v[vgprValuC+34], v[vgprValuC+34] // convert C to bf16 in gwvw==1
buffer_store_short v34, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+35], v49, v[vgprValuC+35]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v88 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+35], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+35], v48, v[vgprValuC+35]    // C += bias
v_cvt_pk_bf16_f32 v35, v[vgprValuC+35], v[vgprValuC+35] // convert C to bf16 in gwvw==1
buffer_store_short v35, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+36], v54, v[vgprValuC+36]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v91 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+36], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+36], v53, v[vgprValuC+36]    // C += bias
v_cvt_pk_bf16_f32 v36, v[vgprValuC+36], v[vgprValuC+36] // convert C to bf16 in gwvw==1
buffer_store_short v36, v92, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #2 (d1,d0,vc1,vc0) = */
/*    (0,0,8,0:vw1); (0,1,8,0:vw1); (0,2,8,0:vw1); (0,3,8,0:vw1); (0,0,9,0:vw1); (0,1,9,0:vw1); (0,2,9,0:vw1); (0,3,9,0:vw1); (0,0,10,0:vw1); (0,1,10,0:vw1); (0,2,10,0:vw1); (0,3,10,0:vw1); (0,0,11,0:vw1); (0,1,11,0:vw1); (0,2,11,0:vw1); (0,3,11,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v16, BufferOOB
/* (d1,vc1,d0,vc0)=(0,8,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v40, v12, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v40, v16, v40, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v37, v40, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v41, v10, s70
v_lshlrev_b32 v41, 0x2, v41                        // Bias address scaled by BPE
ds_read_b32 v38, v41 offset:0                      // load Bias
ds_read_b32 v39, v41 offset:1024                   // load scaleAlpha
v_add_lshl_u32 v40, v13, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v40, v16, v40, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v45, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v45, v16, v45, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v42, v45, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v46, v14, s70
v_lshlrev_b32 v46, 0x2, v46                        // Bias address scaled by BPE
ds_read_b32 v43, v46 offset:0                      // load Bias
ds_read_b32 v44, v46 offset:1024                   // load scaleAlpha
v_add_lshl_u32 v45, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v45, v16, v45, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v50, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v50, v16, v50, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v47, v50, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v51, v14, s70
v_lshlrev_b32 v51, 0x2, v51                        // Bias address scaled by BPE
ds_read_b32 v48, v51 offset:0                      // load Bias
ds_read_b32 v49, v51 offset:1024                   // load scaleAlpha
v_add_lshl_u32 v50, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v50, v16, v50, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v55, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v55, v16, v55, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v52, v55, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v56, v14, s70
v_lshlrev_b32 v56, 0x2, v56                        // Bias address scaled by BPE
ds_read_b32 v53, v56 offset:0                      // load Bias
ds_read_b32 v54, v56 offset:1024                   // load scaleAlpha
v_add_lshl_u32 v55, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v55, v16, v55, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v58, v12, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v58, v16, v58, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v57, v58, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v59, v10, s70
v_lshlrev_b32 v59, 0x2, v59                        // Bias address scaled by BPE
v_add_lshl_u32 v58, v13, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v58, v16, v58, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v61, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v61, v16, v61, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v60, v61, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v62, v14, s70
v_lshlrev_b32 v62, 0x2, v62                        // Bias address scaled by BPE
v_add_lshl_u32 v61, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v61, v16, v61, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v64, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v64, v16, v64, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v63, v64, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v65, v14, s70
v_lshlrev_b32 v65, 0x2, v65                        // Bias address scaled by BPE
v_add_lshl_u32 v64, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v64, v16, v64, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v67, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v67, v16, v67, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v66, v67, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v68, v14, s70
v_lshlrev_b32 v68, 0x2, v68                        // Bias address scaled by BPE
v_add_lshl_u32 v67, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v67, v16, v67, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v70, v12, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v70, v16, v70, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v69, v70, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v71, v10, s70
v_lshlrev_b32 v71, 0x2, v71                        // Bias address scaled by BPE
v_add_lshl_u32 v70, v13, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v70, v16, v70, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v73, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v73, v16, v73, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v72, v73, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v75, v14, s70
v_lshlrev_b32 v75, 0x2, v75                        // Bias address scaled by BPE
v_add_lshl_u32 v73, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v73, v16, v73, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v77, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v77, v16, v77, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v76, v77, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v78, v14, s70
v_lshlrev_b32 v78, 0x2, v78                        // Bias address scaled by BPE
v_add_lshl_u32 v77, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v77, v16, v77, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v80, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, v16, v80, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v79, v80, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v81, v14, s70
v_lshlrev_b32 v81, 0x2, v81                        // Bias address scaled by BPE
v_add_lshl_u32 v80, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, v16, v80, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v83, v12, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v83, v16, v83, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v82, v83, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v84, v10, s70
v_lshlrev_b32 v84, 0x2, v84                        // Bias address scaled by BPE
v_add_lshl_u32 v83, v13, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v83, v16, v83, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v86, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, v16, v86, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v85, v86, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v87, v14, s70
v_lshlrev_b32 v87, 0x2, v87                        // Bias address scaled by BPE
v_add_lshl_u32 v86, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, v16, v86, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v89, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v89, v16, v89, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v88, v89, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v90, v14, s70
v_lshlrev_b32 v90, 0x2, v90                        // Bias address scaled by BPE
v_add_lshl_u32 v89, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v89, v16, v89, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v92, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v92, v16, v92, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v91, v92, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v93, v14, s70
v_lshlrev_b32 v93, 0x2, v93                        // Bias address scaled by BPE
v_add_lshl_u32 v92, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v92, v16, v92, s[74:75]              // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+21], acc2           // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+22], acc6           // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+23], acc10          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+24], acc14          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+25], acc18          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+26], acc22          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+27], acc26          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+28], acc30          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+29], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+30], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+31], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+32], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+33], acc50          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+34], acc54          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+35], acc58          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+36], acc62          // copy acc to vreg[47]

/* rC *= alpha batchElements=[(0, 0, 8, 0), (0, 1, 8, 0), (0, 2, 8, 0), (0, 3, 8, 0), (0, 0, 9, 0), (0, 1, 9, 0), (0, 2, 9, 0), (0, 3, 9, 0), (0, 0, 10, 0), (0, 1, 10, 0), (0, 2, 10, 0), (0, 3, 10, 0), (0, 0, 11, 0), (0, 1, 11, 0), (0, 2, 11, 0), (0, 3, 11, 0)] */
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+21] // *= alpha
v_pk_mul_f32 v[vgprValuC+22:vgprValuC+22+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+22:vgprValuC+22+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+24:vgprValuC+24+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+26:vgprValuC+26+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+28:vgprValuC+28+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+30:vgprValuC+30+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+32:vgprValuC+32+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+34:vgprValuC+34+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+36], s[sgprAlpha], v[vgprValuC+36] // *= alpha
s_waitcnt 0                                        // wait for Beta, Bias LDS, ScaleAlphaVec

/* apply mask, calc new C and issue writes */
v_mov_b32 v18, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v19, 0x7fff0000                          // fp32 Nan
v_mov_b32 v20, 0x7fff                              // rounding bias for bfloat16
v_mul_f32 v[vgprValuC+21], v39, v[vgprValuC+21]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v37 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+21], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+21], v38, v[vgprValuC+21]    // C += bias
v_cvt_pk_bf16_f32 v21, v[vgprValuC+21], v[vgprValuC+21] // convert C to bf16 in gwvw==1
buffer_store_short v21, v40, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+22], v44, v[vgprValuC+22]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v42 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+22], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+22], v43, v[vgprValuC+22]    // C += bias
v_cvt_pk_bf16_f32 v22, v[vgprValuC+22], v[vgprValuC+22] // convert C to bf16 in gwvw==1
buffer_store_short v22, v45, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+23], v49, v[vgprValuC+23]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v47 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+23], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+23], v48, v[vgprValuC+23]    // C += bias
v_cvt_pk_bf16_f32 v23, v[vgprValuC+23], v[vgprValuC+23] // convert C to bf16 in gwvw==1
buffer_store_short v23, v50, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+24], v54, v[vgprValuC+24]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v52 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+24], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+24], v53, v[vgprValuC+24]    // C += bias
v_cvt_pk_bf16_f32 v24, v[vgprValuC+24], v[vgprValuC+24] // convert C to bf16 in gwvw==1
buffer_store_short v24, v55, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+25], v39, v[vgprValuC+25]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v57 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+25], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+25], v38, v[vgprValuC+25]    // C += bias
v_cvt_pk_bf16_f32 v25, v[vgprValuC+25], v[vgprValuC+25] // convert C to bf16 in gwvw==1
buffer_store_short v25, v58, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+26], v44, v[vgprValuC+26]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v60 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+26], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+26], v43, v[vgprValuC+26]    // C += bias
v_cvt_pk_bf16_f32 v26, v[vgprValuC+26], v[vgprValuC+26] // convert C to bf16 in gwvw==1
buffer_store_short v26, v61, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+27], v49, v[vgprValuC+27]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v63 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+27], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+27], v48, v[vgprValuC+27]    // C += bias
v_cvt_pk_bf16_f32 v27, v[vgprValuC+27], v[vgprValuC+27] // convert C to bf16 in gwvw==1
buffer_store_short v27, v64, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+28], v54, v[vgprValuC+28]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v66 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+28], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+28], v53, v[vgprValuC+28]    // C += bias
v_cvt_pk_bf16_f32 v28, v[vgprValuC+28], v[vgprValuC+28] // convert C to bf16 in gwvw==1
buffer_store_short v28, v67, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+29], v39, v[vgprValuC+29]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v69 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+29], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+29], v38, v[vgprValuC+29]    // C += bias
v_cvt_pk_bf16_f32 v29, v[vgprValuC+29], v[vgprValuC+29] // convert C to bf16 in gwvw==1
buffer_store_short v29, v70, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+30], v44, v[vgprValuC+30]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v72 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+30], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+30], v43, v[vgprValuC+30]    // C += bias
v_cvt_pk_bf16_f32 v30, v[vgprValuC+30], v[vgprValuC+30] // convert C to bf16 in gwvw==1
buffer_store_short v30, v73, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+31], v49, v[vgprValuC+31]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v76 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+31], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+31], v48, v[vgprValuC+31]    // C += bias
v_cvt_pk_bf16_f32 v31, v[vgprValuC+31], v[vgprValuC+31] // convert C to bf16 in gwvw==1
buffer_store_short v31, v77, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+32], v54, v[vgprValuC+32]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v79 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+32], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+32], v53, v[vgprValuC+32]    // C += bias
v_cvt_pk_bf16_f32 v32, v[vgprValuC+32], v[vgprValuC+32] // convert C to bf16 in gwvw==1
buffer_store_short v32, v80, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+33], v39, v[vgprValuC+33]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v82 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+33], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+33], v38, v[vgprValuC+33]    // C += bias
v_cvt_pk_bf16_f32 v33, v[vgprValuC+33], v[vgprValuC+33] // convert C to bf16 in gwvw==1
buffer_store_short v33, v83, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+34], v44, v[vgprValuC+34]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v85 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+34], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+34], v43, v[vgprValuC+34]    // C += bias
v_cvt_pk_bf16_f32 v34, v[vgprValuC+34], v[vgprValuC+34] // convert C to bf16 in gwvw==1
buffer_store_short v34, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+35], v49, v[vgprValuC+35]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v88 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+35], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+35], v48, v[vgprValuC+35]    // C += bias
v_cvt_pk_bf16_f32 v35, v[vgprValuC+35], v[vgprValuC+35] // convert C to bf16 in gwvw==1
buffer_store_short v35, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+36], v54, v[vgprValuC+36]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v91 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+36], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+36], v53, v[vgprValuC+36]    // C += bias
v_cvt_pk_bf16_f32 v36, v[vgprValuC+36], v[vgprValuC+36] // convert C to bf16 in gwvw==1
buffer_store_short v36, v92, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #3 (d1,d0,vc1,vc0) = */
/*    (0,0,12,0:vw1); (0,1,12,0:vw1); (0,2,12,0:vw1); (0,3,12,0:vw1); (0,0,13,0:vw1); (0,1,13,0:vw1); (0,2,13,0:vw1); (0,3,13,0:vw1); (0,0,14,0:vw1); (0,1,14,0:vw1); (0,2,14,0:vw1); (0,3,14,0:vw1); (0,0,15,0:vw1); (0,1,15,0:vw1); (0,2,15,0:vw1); (0,3,15,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v16, BufferOOB
/* (d1,vc1,d0,vc0)=(0,12,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v40, v12, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v40, v16, v40, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v37, v40, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v41, v10, s70
v_lshlrev_b32 v41, 0x2, v41                        // Bias address scaled by BPE
ds_read_b32 v38, v41 offset:0                      // load Bias
ds_read_b32 v39, v41 offset:1024                   // load scaleAlpha
v_add_lshl_u32 v40, v13, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v40, v16, v40, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v45, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v45, v16, v45, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v42, v45, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v46, v14, s70
v_lshlrev_b32 v46, 0x2, v46                        // Bias address scaled by BPE
ds_read_b32 v43, v46 offset:0                      // load Bias
ds_read_b32 v44, v46 offset:1024                   // load scaleAlpha
v_add_lshl_u32 v45, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v45, v16, v45, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v50, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v50, v16, v50, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v47, v50, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v51, v14, s70
v_lshlrev_b32 v51, 0x2, v51                        // Bias address scaled by BPE
ds_read_b32 v48, v51 offset:0                      // load Bias
ds_read_b32 v49, v51 offset:1024                   // load scaleAlpha
v_add_lshl_u32 v50, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v50, v16, v50, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v55, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v55, v16, v55, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v52, v55, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v56, v14, s70
v_lshlrev_b32 v56, 0x2, v56                        // Bias address scaled by BPE
ds_read_b32 v53, v56 offset:0                      // load Bias
ds_read_b32 v54, v56 offset:1024                   // load scaleAlpha
v_add_lshl_u32 v55, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v55, v16, v55, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v58, v12, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v58, v16, v58, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v57, v58, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v59, v10, s70
v_lshlrev_b32 v59, 0x2, v59                        // Bias address scaled by BPE
v_add_lshl_u32 v58, v13, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v58, v16, v58, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v61, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v61, v16, v61, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v60, v61, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v62, v14, s70
v_lshlrev_b32 v62, 0x2, v62                        // Bias address scaled by BPE
v_add_lshl_u32 v61, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v61, v16, v61, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v64, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v64, v16, v64, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v63, v64, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v65, v14, s70
v_lshlrev_b32 v65, 0x2, v65                        // Bias address scaled by BPE
v_add_lshl_u32 v64, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v64, v16, v64, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v67, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v67, v16, v67, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v66, v67, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v68, v14, s70
v_lshlrev_b32 v68, 0x2, v68                        // Bias address scaled by BPE
v_add_lshl_u32 v67, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v67, v16, v67, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v70, v12, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v70, v16, v70, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v69, v70, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v71, v10, s70
v_lshlrev_b32 v71, 0x2, v71                        // Bias address scaled by BPE
v_add_lshl_u32 v70, v13, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v70, v16, v70, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v73, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v73, v16, v73, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v72, v73, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v75, v14, s70
v_lshlrev_b32 v75, 0x2, v75                        // Bias address scaled by BPE
v_add_lshl_u32 v73, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v73, v16, v73, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v77, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v77, v16, v77, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v76, v77, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v78, v14, s70
v_lshlrev_b32 v78, 0x2, v78                        // Bias address scaled by BPE
v_add_lshl_u32 v77, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v77, v16, v77, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v80, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, v16, v80, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v79, v80, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v81, v14, s70
v_lshlrev_b32 v81, 0x2, v81                        // Bias address scaled by BPE
v_add_lshl_u32 v80, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, v16, v80, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,0) */
v_add_co_u32 v11, vcc, v11, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v12, v12, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v13, v13, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[70:71], v10, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v83, v12, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v83, v16, v83, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v82, v83, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v84, v10, s70
v_lshlrev_b32 v84, 0x2, v84                        // Bias address scaled by BPE
v_add_lshl_u32 v83, v13, v10, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v83, v16, v83, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,1,0) */
v_add_co_u32 v14, vcc, v10, 32                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v86, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, v16, v86, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v85, v86, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v87, v14, s70
v_lshlrev_b32 v87, 0x2, v87                        // Bias address scaled by BPE
v_add_lshl_u32 v86, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, v16, v86, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,2,0) */
v_add_co_u32 v14, vcc, v10, 64                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v89, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v89, v16, v89, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v88, v89, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v90, v14, s70
v_lshlrev_b32 v90, 0x2, v90                        // Bias address scaled by BPE
v_add_lshl_u32 v89, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v89, v16, v89, s[74:75]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,3,0) */
s_mov_b32 s70, 96                                  // coordOffset0 d0=3 vc0=0
v_add_co_u32 v14, vcc, v10, s70                    // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[70:71], v14, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v11, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[70:71], s[74:75]             // in0 && in1
v_add_lshl_u32 v92, v12, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v92, v16, v92, s[74:75]              // LDC clip if OOB. offset
buffer_load_short_d16 v91, v92, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s70, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v93, v14, s70
v_lshlrev_b32 v93, 0x2, v93                        // Bias address scaled by BPE
v_add_lshl_u32 v92, v13, v14, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v92, v16, v92, s[74:75]              // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+21], acc3           // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+22], acc7           // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+23], acc11          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+24], acc15          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+25], acc19          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+26], acc23          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+27], acc27          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+28], acc31          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+29], acc35          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+30], acc39          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+31], acc43          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+32], acc47          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+33], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+34], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+35], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+36], acc63          // copy acc to vreg[63]

/* rC *= alpha batchElements=[(0, 0, 12, 0), (0, 1, 12, 0), (0, 2, 12, 0), (0, 3, 12, 0), (0, 0, 13, 0), (0, 1, 13, 0), (0, 2, 13, 0), (0, 3, 13, 0), (0, 0, 14, 0), (0, 1, 14, 0), (0, 2, 14, 0), (0, 3, 14, 0), (0, 0, 15, 0), (0, 1, 15, 0), (0, 2, 15, 0), (0, 3, 15, 0)] */
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+21] // *= alpha
v_pk_mul_f32 v[vgprValuC+22:vgprValuC+22+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+22:vgprValuC+22+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+24:vgprValuC+24+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+26:vgprValuC+26+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+28:vgprValuC+28+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+30:vgprValuC+30+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+32:vgprValuC+32+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+34:vgprValuC+34+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+36], s[sgprAlpha], v[vgprValuC+36] // *= alpha
s_waitcnt 0                                        // wait for Beta, Bias LDS, ScaleAlphaVec

/* apply mask, calc new C and issue writes */
v_mov_b32 v18, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v19, 0x7fff0000                          // fp32 Nan
v_mov_b32 v20, 0x7fff                              // rounding bias for bfloat16
v_mul_f32 v[vgprValuC+21], v39, v[vgprValuC+21]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v37 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+21], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+21], v38, v[vgprValuC+21]    // C += bias
v_cvt_pk_bf16_f32 v21, v[vgprValuC+21], v[vgprValuC+21] // convert C to bf16 in gwvw==1
buffer_store_short v21, v40, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+22], v44, v[vgprValuC+22]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v42 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+22], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+22], v43, v[vgprValuC+22]    // C += bias
v_cvt_pk_bf16_f32 v22, v[vgprValuC+22], v[vgprValuC+22] // convert C to bf16 in gwvw==1
buffer_store_short v22, v45, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+23], v49, v[vgprValuC+23]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v47 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+23], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+23], v48, v[vgprValuC+23]    // C += bias
v_cvt_pk_bf16_f32 v23, v[vgprValuC+23], v[vgprValuC+23] // convert C to bf16 in gwvw==1
buffer_store_short v23, v50, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+24], v54, v[vgprValuC+24]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v52 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+24], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+24], v53, v[vgprValuC+24]    // C += bias
v_cvt_pk_bf16_f32 v24, v[vgprValuC+24], v[vgprValuC+24] // convert C to bf16 in gwvw==1
buffer_store_short v24, v55, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+25], v39, v[vgprValuC+25]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v57 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+25], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+25], v38, v[vgprValuC+25]    // C += bias
v_cvt_pk_bf16_f32 v25, v[vgprValuC+25], v[vgprValuC+25] // convert C to bf16 in gwvw==1
buffer_store_short v25, v58, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+26], v44, v[vgprValuC+26]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v60 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+26], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+26], v43, v[vgprValuC+26]    // C += bias
v_cvt_pk_bf16_f32 v26, v[vgprValuC+26], v[vgprValuC+26] // convert C to bf16 in gwvw==1
buffer_store_short v26, v61, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+27], v49, v[vgprValuC+27]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v63 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+27], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+27], v48, v[vgprValuC+27]    // C += bias
v_cvt_pk_bf16_f32 v27, v[vgprValuC+27], v[vgprValuC+27] // convert C to bf16 in gwvw==1
buffer_store_short v27, v64, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+28], v54, v[vgprValuC+28]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v66 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+28], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+28], v53, v[vgprValuC+28]    // C += bias
v_cvt_pk_bf16_f32 v28, v[vgprValuC+28], v[vgprValuC+28] // convert C to bf16 in gwvw==1
buffer_store_short v28, v67, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+29], v39, v[vgprValuC+29]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v69 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+29], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+29], v38, v[vgprValuC+29]    // C += bias
v_cvt_pk_bf16_f32 v29, v[vgprValuC+29], v[vgprValuC+29] // convert C to bf16 in gwvw==1
buffer_store_short v29, v70, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+30], v44, v[vgprValuC+30]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v72 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+30], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+30], v43, v[vgprValuC+30]    // C += bias
v_cvt_pk_bf16_f32 v30, v[vgprValuC+30], v[vgprValuC+30] // convert C to bf16 in gwvw==1
buffer_store_short v30, v73, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+31], v49, v[vgprValuC+31]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v76 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+31], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+31], v48, v[vgprValuC+31]    // C += bias
v_cvt_pk_bf16_f32 v31, v[vgprValuC+31], v[vgprValuC+31] // convert C to bf16 in gwvw==1
buffer_store_short v31, v77, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+32], v54, v[vgprValuC+32]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v79 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+32], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+32], v53, v[vgprValuC+32]    // C += bias
v_cvt_pk_bf16_f32 v32, v[vgprValuC+32], v[vgprValuC+32] // convert C to bf16 in gwvw==1
buffer_store_short v32, v80, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+33], v39, v[vgprValuC+33]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v82 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+33], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+33], v38, v[vgprValuC+33]    // C += bias
v_cvt_pk_bf16_f32 v33, v[vgprValuC+33], v[vgprValuC+33] // convert C to bf16 in gwvw==1
buffer_store_short v33, v83, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+34], v44, v[vgprValuC+34]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v85 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+34], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+34], v43, v[vgprValuC+34]    // C += bias
v_cvt_pk_bf16_f32 v34, v[vgprValuC+34], v[vgprValuC+34] // convert C to bf16 in gwvw==1
buffer_store_short v34, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+35], v49, v[vgprValuC+35]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v88 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+35], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+35], v48, v[vgprValuC+35]    // C += bias
v_cvt_pk_bf16_f32 v35, v[vgprValuC+35], v[vgprValuC+35] // convert C to bf16 in gwvw==1
buffer_store_short v35, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_f32 v[vgprValuC+36], v54, v[vgprValuC+36]    // *= ScaleAlphaVecVMul
v_cvt_f32_bf16 v14, v91 src0_sel:WORD_0            // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+36], v14, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v[vgprValuC+36], v53, v[vgprValuC+36]    // C += bias
v_cvt_pk_bf16_f32 v36, v[vgprValuC+36], v[vgprValuC+36] // convert C to bf16 in gwvw==1
buffer_store_short v36, v92, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_SK_Partials_1:
label_GW_Partials_E0:
s_mov_b64 s[sgprSrdWS+0:sgprSrdWS+0+1], s[sgprAddressWS+0:sgprAddressWS+0+1] // init SRD base address
s_mov_b32 s[sgprSrdWS+2], BufferOOB
s_mov_b32 s[sgprSrdWS+3], Srd127_96                // Set bits 127_96 in post-loop SRD

s_mul_i32 s8, 0x10000, s[sgprStreamKIdx]           // Offset to correct partials tile
s_add_u32 s[sgprSrdWS+0], s[sgprSrdWS+0], s8       // add lo to SRD
s_addc_u32 s[sgprSrdWS+1], s[sgprSrdWS+1], 0       // add hi to SRD

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=16 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Partials Write Batch #0 (d1,d0,vc1,vc0) = */
/*      (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,0,1,0:vw1); (0,1,1,0:vw1); (0,2,1,0:vw1); (0,3,1,0:vw1); (0,0,2,0:vw1); (0,1,2,0:vw1); (0,2,2,0:vw1); (0,3,2,0:vw1); (0,0,3,0:vw1); (0,1,3,0:vw1); (0,2,3,0:vw1); (0,3,3,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_accvgpr_read_b32 v[vgprValuC+23], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+24], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+25], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+26], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+27], acc16          // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+28], acc20          // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+29], acc24          // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+30], acc28          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+31], acc32          // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+32], acc36          // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+33], acc40          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+34], acc44          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+35], acc48          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+36], acc52          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+37], acc56          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+38], acc60          // copy acc to vreg[15]
s_nop 1                                            // 2 wait states required before reading vgpr

/* apply mask, calc new C and issue writes */
v_mov_b32 v18, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v19, 0x7fff0000                          // fp32 Nan
v_mov_b32 v20, 0x7fff                              // rounding bias for bfloat16
v_lshlrev_b32 v21, 2, v[vgprSerial]                // v21 = v[vgprSerial] * 4
s_mov_b32 s8, 0                                    // Init sgpr offset
buffer_store_dword v23, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v24, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v25, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v26, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v27, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v28, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v29, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v30, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v31, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v32, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v33, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v34, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v35, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v36, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v37, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v38, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Partials Write Batch #1 (d1,d0,vc1,vc0) = */
/*      (0,0,4,0:vw1); (0,1,4,0:vw1); (0,2,4,0:vw1); (0,3,4,0:vw1); (0,0,5,0:vw1); (0,1,5,0:vw1); (0,2,5,0:vw1); (0,3,5,0:vw1); (0,0,6,0:vw1); (0,1,6,0:vw1); (0,2,6,0:vw1); (0,3,6,0:vw1); (0,0,7,0:vw1); (0,1,7,0:vw1); (0,2,7,0:vw1); (0,3,7,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_accvgpr_read_b32 v[vgprValuC+23], acc1           // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+24], acc5           // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+25], acc9           // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+26], acc13          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+27], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+28], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+29], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+30], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+31], acc33          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+32], acc37          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+33], acc41          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+34], acc45          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+35], acc49          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+36], acc53          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+37], acc57          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+38], acc61          // copy acc to vreg[31]
s_nop 1                                            // 2 wait states required before reading vgpr

/* apply mask, calc new C and issue writes */
v_mov_b32 v18, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v19, 0x7fff0000                          // fp32 Nan
v_mov_b32 v20, 0x7fff                              // rounding bias for bfloat16
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v23, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v24, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v25, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v26, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v27, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v28, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v29, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v30, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v31, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v32, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v33, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v34, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v35, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v36, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v37, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v38, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Partials Write Batch #2 (d1,d0,vc1,vc0) = */
/*      (0,0,8,0:vw1); (0,1,8,0:vw1); (0,2,8,0:vw1); (0,3,8,0:vw1); (0,0,9,0:vw1); (0,1,9,0:vw1); (0,2,9,0:vw1); (0,3,9,0:vw1); (0,0,10,0:vw1); (0,1,10,0:vw1); (0,2,10,0:vw1); (0,3,10,0:vw1); (0,0,11,0:vw1); (0,1,11,0:vw1); (0,2,11,0:vw1); (0,3,11,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_accvgpr_read_b32 v[vgprValuC+23], acc2           // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+24], acc6           // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+25], acc10          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+26], acc14          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+27], acc18          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+28], acc22          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+29], acc26          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+30], acc30          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+31], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+32], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+33], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+34], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+35], acc50          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+36], acc54          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+37], acc58          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+38], acc62          // copy acc to vreg[47]
s_nop 1                                            // 2 wait states required before reading vgpr

/* apply mask, calc new C and issue writes */
v_mov_b32 v18, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v19, 0x7fff0000                          // fp32 Nan
v_mov_b32 v20, 0x7fff                              // rounding bias for bfloat16
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v23, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v24, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v25, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v26, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v27, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v28, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v29, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v30, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v31, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v32, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v33, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v34, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v35, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v36, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v37, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v38, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Partials Write Batch #3 (d1,d0,vc1,vc0) = */
/*      (0,0,12,0:vw1); (0,1,12,0:vw1); (0,2,12,0:vw1); (0,3,12,0:vw1); (0,0,13,0:vw1); (0,1,13,0:vw1); (0,2,13,0:vw1); (0,3,13,0:vw1); (0,0,14,0:vw1); (0,1,14,0:vw1); (0,2,14,0:vw1); (0,3,14,0:vw1); (0,0,15,0:vw1); (0,1,15,0:vw1); (0,2,15,0:vw1); (0,3,15,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_accvgpr_read_b32 v[vgprValuC+23], acc3           // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+24], acc7           // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+25], acc11          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+26], acc15          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+27], acc19          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+28], acc23          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+29], acc27          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+30], acc31          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+31], acc35          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+32], acc39          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+33], acc43          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+34], acc47          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+35], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+36], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+37], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+38], acc63          // copy acc to vreg[63]
s_nop 1                                            // 2 wait states required before reading vgpr

/* apply mask, calc new C and issue writes */
v_mov_b32 v18, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v19, 0x7fff0000                          // fp32 Nan
v_mov_b32 v20, 0x7fff                              // rounding bias for bfloat16
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v23, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v24, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v25, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v26, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v27, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v28, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v29, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v30, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v31, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v32, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v33, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v34, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v35, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v36, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v37, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 1024                             // Inc sgpr offset
buffer_store_dword v38, v21, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_waitcnt vmcnt(0)                                 // wait for data store
s_barrier                                          // store all data before setting flag
s_lshl_b32 s8, s[sgprStreamKIdx], 2                // flag offset based on CTA index
v_readfirstlane_b32 s64, v[vgprSerial]             // Wave 0 updates flags
s_cmp_eq_u32 s64, 0                                // Check for wave 0
s_cbranch_scc0 label_SK_SkipFlagSet                // Skip flag set
s_mov_b32 s64, 1                                   // flag data
s_store_dword s64, s[sgprAddressFlags:sgprAddressFlags+1], s8 glc // set flag
label_SK_SkipFlagSet:
s_waitcnt lgkmcnt(0)                               // wait for flag
s_branch label_GW_End_1                            // jump to end
label_GW_End_1:
label_SK_CloseLoop:
s_cmp_ge_u32 s[sgprStreamKIter], s[sgprStreamKIterEnd] // Check if done all StreamK iterations
s_cbranch_scc1 label_NoBranch_IXPKU979JKZCQDH3     // Only branch on scc0
s_getpc_b64 s[70:71]                               // addr of next instr
s_add_i32 s72, label_PersistentLoopStart, 4        // target branch offset
s_abs_i32 s72, s72                                 // abs offset
s_sub_u32 s70, s70, s72                            // sub target branch offset
s_subb_u32 s71, s71, 0                             // sub high and carry
s_setpc_b64 s[70:71]                               // branch to label_PersistentLoopStart
label_NoBranch_IXPKU979JKZCQDH3:
label_KernelEnd:
s_endpgm                                           // Kernel End
label_ASM_End:  /// The end of the kernel
