
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
.text
.protected Custom_Cijk_Alik_Bljk_HHS_BH_UserArgs_MT128x16x128_MI16x16x1_SN_K1_MIWT2_1_triple_buffer
.globl Custom_Cijk_Alik_Bljk_HHS_BH_UserArgs_MT128x16x128_MI16x16x1_SN_K1_MIWT2_1_triple_buffer
.p2align 8
.type Custom_Cijk_Alik_Bljk_HHS_BH_UserArgs_MT128x16x128_MI16x16x1_SN_K1_MIWT2_1_triple_buffer,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Custom_Cijk_Alik_Bljk_HHS_BH_UserArgs_MT128x16x128_MI16x16x1_SN_K1_MIWT2_1_triple_buffer
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_accum_offset 256 // accvgpr offset
  .amdhsa_next_free_vgpr 264 // vgprs
  .amdhsa_next_free_sgpr 78 // sgprs
  .amdhsa_group_segment_fixed_size 39424 // lds bytes
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
/* Num AccVGPR=8 */
/* Num SGPR   =78 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 8 x 1 */
/* SubGroup= 16 x 16 */
/* VectorWidthA=2 */
/* VectorWidthB=1 */
/* GlobalReadVectorWidthA=8, GlobalReadVectorWidthB=8 */
/* DirectToLdsA=False */
/* DirectToLdsB=False */
/* UseSgprForGRO=1 */
.amdgpu_metadata
---
custom.config:
   ProblemType:
      OperationType: GEMM
      DataType: h
      DestDataType: h
      ComputeDataType: s
      HighPrecisionAccumulate: True
      TransposeA: 1
      TransposeB: 0
      UseBeta: True
      Batched: True
   AssertFree0ElementMultiple: 16
   AssertFree1ElementMultiple: 1
   AssertSummationElementMultiple: 128
   PrefetchLocalRead: 2
   DepthU: 128
   GlobalReadVectorWidthA: 8
   GlobalReadVectorWidthB: 8
   1LDSBuffer: 1
   NonTemporalA: 4
   WorkGroupMapping: 1
   InternalSupportParams: {KernArgsVersion: 0, SupportCustomWGM: True, SupportUserGSU: True, SupportCustomStaggerU: True}
   NoReject: 1
amdhsa.version:
  - 1
  - 1
amdhsa.kernels:
  - .name: Custom_Cijk_Alik_Bljk_HHS_BH_UserArgs_MT128x16x128_MI16x16x1_SN_K1_MIWT2_1_triple_buffer
    .symbol: 'Custom_Cijk_Alik_Bljk_HHS_BH_UserArgs_MT128x16x128_MI16x16x1_SN_K1_MIWT2_1_triple_buffer.kd'
    .language:                   OpenCL C
    .language_version:
      - 2
      - 0
    .args:
      - .name:            SizesFree0
        .size:            4
        .offset:          0
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree1
        .size:            4
        .offset:          4
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree2
        .size:            4
        .offset:          8
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesSum0
        .size:            4
        .offset:          12
        .value_kind:      by_value
        .value_type:      u32
      - .name:            Gemm info
        .size:            4
        .offset:          16
        .value_kind:      by_value
        .value_type:      u32
      - .name:            kernel info
        .size:            4
        .offset:          20
        .value_kind:      by_value
        .value_type:      u32
      - .name:            D
        .size:            8
        .offset:          24
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            C
        .size:            8
        .offset:          32
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            A
        .size:            8
        .offset:          40
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            B
        .size:            8
        .offset:          48
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            strideD0
        .size:            4
        .offset:          56
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideD1
        .size:            4
        .offset:          60
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC0
        .size:            4
        .offset:          64
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC1
        .size:            4
        .offset:          68
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA0
        .size:            4
        .offset:          72
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA1
        .size:            4
        .offset:          76
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB0
        .size:            4
        .offset:          80
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB1
        .size:            4
        .offset:          84
        .value_kind:      by_value
        .value_type:      u32
      - .name:            alpha
        .size:            4
        .offset:          88
        .value_kind:      by_value
        .value_type:      f32
      - .name:            beta
        .size:            4
        .offset:          92
        .value_kind:      by_value
        .value_type:      f32
    .group_segment_fixed_size:   39424
    .kernarg_segment_align:      8
    .kernarg_segment_size:       96
    .max_flat_workgroup_size:    256
    .private_segment_fixed_size: 0
    .sgpr_count:                 78
    .sgpr_spill_count:           0
    .vgpr_count:                 256
    .vgpr_spill_count:           0
    .wavefront_size:             64
...
.end_amdgpu_metadata
Custom_Cijk_Alik_Bljk_HHS_BH_UserArgs_MT128x16x128_MI16x16x1_SN_K1_MIWT2_1_triple_buffer:
label_ASM_Start:  /// Main body of the asm kernel

/* Magic div and mod functions */
.macro V_MAGIC_DIV dstIdx:req dividend:req magicNumber:req magicShift:req magicA:req
    v_mul_hi_u32 v[\dstIdx+1] \dividend \magicNumber
    v_mul_lo_u32 v[\dstIdx+0] \dividend \magicA
    v_add_u32 v[\dstIdx+0] v[\dstIdx+0] v[\dstIdx+1]
    v_lshrrev_b32 v[\dstIdx+0] \magicShift v[\dstIdx+0]
.endm

/******************************************/
/* VGPR Assignments                       */
/******************************************/
/* ValuC range: [0-0), serializedStore enabled */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprValuA_X0_I0, 0
.set vgprValuA_X1_I0, 4
.set vgprValuA_X2_I0, 8
.set vgprValuA_X3_I0, 12
.set vgprValuA_X4_I0, 16
.set vgprValuA_X5_I0, 20
.set vgprValuA_X6_I0, 24
.set vgprValuA_X7_I0, 28
.set vgprValuB_X0_I0, 32
.set vgprValuB_X1_I0, 34
.set vgprValuB_X2_I0, 36
.set vgprValuB_X3_I0, 38
.set vgprValuB_X4_I0, 40
.set vgprValuB_X5_I0, 42
.set vgprValuB_X6_I0, 44
.set vgprValuB_X7_I0, 46
.set vgprLocalWriteAddrA, 48
.set vgprLocalWriteAddrB, 49
.set vgprGlobalReadOffsetA, 50
.set vgprGlobalReadOffsetB, 51
.set vgprG2LA, 52
.set vgprG2LB, 84
.set vgprLocalReadAddrA, 88
.set vgprLocalReadAddrB, 89
.set vgprSerial, 90

/* extra set of vgpr for triple buffer*/
.set vgprG2LA1, 92
.set vgprG2LB1, 124

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
.set sgprAddressD, 28
.set sgprAddressC, 30
.set sgprAddressA, 32
.set sgprAddressB, 34
.set sgprStridesD, 36
.set sgprStridesC, 38
.set sgprStridesA, 40
.set sgprStridesB, 42
.set sgprAlpha, 44
.set sgprBeta, 45
.set sgprGSU, 46

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
.set MT1, 16
.set DepthU, 128
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
/* hex: 0x00020000                        */
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
.set Srd127_96, 0x00020000

/* Global Offset A */
.macro GLOBAL_OFFSET_A vgprAddr:req vgprOffsetL:req vgprOffset0I:req vgprTmp:req
    v_mul_lo_u32 v[\vgprTmp+0] s[sgprStrideA0I] v[\vgprOffset0I] // mul d1 lower
    v_add_co_u32 v[\vgprAddr+0] vcc v[\vgprOffsetL] v[\vgprTmp+0] // accumulate K lower
    v_add_u32 v[\vgprAddr+0] 0x8 v[\vgprAddr+0]      // add prepad for pointer shift
    v_lshlrev_b32 v[\vgprAddr+0] 0x1 v[\vgprAddr+0]  // offset *= bytes/element
.endm

/* Global Offset B */
.macro GLOBAL_OFFSET_B vgprAddr:req vgprOffsetL:req vgprOffset1J:req vgprTmp:req
    v_mul_lo_u32 v[\vgprTmp+0] s[sgprStrideB1J] v[\vgprOffset1J] // mul d1 lower
    v_add_co_u32 v[\vgprAddr+0] vcc v[\vgprOffsetL] v[\vgprTmp+0] // accumulate K lower
    v_add_u32 v[\vgprAddr+0] 0x8 v[\vgprAddr+0]      // add prepad for pointer shift
    v_lshlrev_b32 v[\vgprAddr+0] 0x1 v[\vgprAddr+0]  // offset *= bytes/element
.endm

/* Dynamic Scalar Divide: vQuotient=vDividend/vDivisor; vRemainder=vDividend%vDivisor; */
.macro DYNAMIC_VECTOR_DIVIDE vQuotient vRemainder vDividend vDivisor vTmp0 vTmp1 sTmp
    v_cvt_f32_u32 v[\vQuotient] v[\vDivisor]
    v_rcp_f32 v[\vQuotient] v[\vQuotient]
    v_mul_f32 v[\vQuotient] 0x4f800000 v[\vQuotient]
    v_cvt_u32_f32 v[\vQuotient] v[\vQuotient]
    v_mul_lo_u32 v[\vRemainder] v[\vDivisor] v[\vQuotient]
    v_mul_hi_u32 v[\vTmp0] v[\vDivisor] v[\vQuotient]
    v_sub_co_u32 v[\vTmp1] vcc 0x0 v[\vRemainder]
    v_cmp_ne_i32 s[\sTmp:\sTmp+1] 0x0 v[\vTmp0]
    v_cndmask_b32 v[\vRemainder] v[\vTmp1] v[\vRemainder] s[\sTmp:\sTmp+1]
    v_mul_hi_u32 v[\vRemainder] v[\vRemainder] v[\vQuotient]
    v_sub_co_u32 v[\vTmp0] vcc v[\vQuotient] v[\vRemainder]
    v_add_co_u32 v[\vQuotient] vcc v[\vQuotient] v[\vRemainder]
    v_cndmask_b32 v[\vQuotient] v[\vQuotient] v[\vTmp0] s[\sTmp:\sTmp+1]
    v_mul_hi_u32 v[\vQuotient] v[\vQuotient] v[\vDividend]
    v_mul_lo_u32 v[\vRemainder] v[\vQuotient] v[\vDivisor]
    v_sub_co_u32 v[\vTmp0] vcc v[\vDividend] v[\vRemainder]
    v_cmp_ge_u32 s[\sTmp:\sTmp+1] v[\vDividend] v[\vRemainder]
    v_add_co_u32 v[\vRemainder] vcc 0x1 v[\vQuotient]
    v_add_co_u32 v[\vTmp1] vcc -1 v[\vQuotient]
    v_cmp_le_u32 vcc v[\vDivisor] v[\vTmp0]
    s_and_b64 vcc s[\sTmp:\sTmp+1] vcc
    v_cndmask_b32 v[\vQuotient] v[\vQuotient] v[\vRemainder] vcc
    v_cndmask_b32 v[\vQuotient] v[\vTmp1] v[\vQuotient] s[\sTmp:\sTmp+1]
    v_cmp_ne_i32 vcc 0x0 v[\vDivisor]
    v_cndmask_b32 v[\vQuotient] -1 v[\vQuotient] vcc // final result
    v_mul_lo_u32 v[\vRemainder] v[\vQuotient] v[\vDivisor]
    v_sub_co_u32 v[\vRemainder] vcc v[\vDividend] v[\vRemainder] // final result
.endm

/******************************************/
/* Allocate Resources                     */
/******************************************/

/* Load num of Gemms */
s_load_dword s47, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x0

/* Load GSU data */
s_load_dword s[sgprGSU], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x4
s_waitcnt lgkmcnt(0)
s_lshr_b32 s48, s47, 0x1e                          // Get arg type
s_and_b32 s47, 0x3fffffff, s47                     // Get nums of gemm
s_cmp_eq_u32 s48, 0                                // Is kernel args
s_cbranch_scc0 label_HBMArgs
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], 0x8 // Shift common args
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0x0

/* Load Kernel Args */
s_load_dwordx16 s[24:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x0
s_load_dwordx4 s[40:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x40
s_load_dwordx2 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x50
s_waitcnt lgkmcnt(0)
s_branch label_LoadArgsEnd
label_HBMArgs:

/* Load address of kernel arguments */
s_load_dwordx2 s[sgprKernArgAddress:sgprKernArgAddress+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x8
s_waitcnt lgkmcnt(0)                               // wait for args to load
label_LoadArgsEnd:
s_branch label_common_kernel_entry

/* pad 41 snops to satisfy 0x100 code size for Preload Backward Compatibility Prologue */
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
s_nop 0
s_nop 0
s_nop 0
s_nop 0
label_Preload_Offset_Start:
s_and_b32 s47, 0x3fffffff, s2                      // Get nums of gemm
s_lshr_b32 s48, s2, 0x1e                           // Get arg type
s_mov_b32 s[sgprGSU], s3                           // Preload internal args
s_cmp_eq_u32 s48, 0                                // Is kernel args
s_cbranch_scc0 label_Preload_HBMArgs
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], 0x8 // Shift common args
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0x0

/* Load Kernel Args */
s_load_dword s33, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x24
s_load_dwordx2 s[34:35], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x28
s_load_dwordx8 s[36:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x30
s_load_dwordx2 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x50
s_mov_b32 s24, s4                                  // move preload data to correct sgpr
s_mov_b32 s25, s5                                  // move preload data to correct sgpr
s_mov_b32 s26, s6                                  // move preload data to correct sgpr
s_mov_b32 s27, s7                                  // move preload data to correct sgpr
s_mov_b32 s28, s8                                  // move preload data to correct sgpr
s_mov_b32 s29, s9                                  // move preload data to correct sgpr
s_mov_b32 s30, s10                                 // move preload data to correct sgpr
s_mov_b32 s31, s11                                 // move preload data to correct sgpr
s_mov_b32 s32, s12                                 // move preload data to correct sgpr
s_branch label_Preload_LoadArgsEnd
label_Preload_HBMArgs:
s_mov_b64 s[sgprKernArgAddress:sgprKernArgAddress+1], s[4:5] // Load address of kernel arguments
label_Preload_LoadArgsEnd:
label_common_kernel_entry:  /// for both preload/non-preload common code
s_mov_b32 s[sgprWorkGroup0+0], s13                 // restore workgroup id
s_mov_b32 s[sgprWorkGroup0+1], s14                 // restore workgroup id
s_mov_b32 s[sgprWorkGroup0+2], s15                 // restore workgroup id
s_and_b32 s[sgprWGM], s[sgprGSU], 0xff00           // Restore WGM
s_lshr_b32 s[sgprWGM], s[sgprWGM], 0x8
s_and_b32 s[sgprStaggerU], s[sgprGSU], 0xffff0000  // Restore StaggerU related vars
s_lshr_b32 s[sgprStaggerU], s[sgprStaggerU], 0x10
s_and_b32 s[sgprGSU], s[sgprGSU], 0xff             // Restore GSU
s_mov_b32 s[sgprArgType], s48
s_mov_b32 m0, 0x9a00                               // LDS clamp at 39424 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id
s_cmp_eq_u32 s48, 0
s_cbranch_scc0 label_MultiGemm
/* init: add vgpr [0...48) to pool */
/* init: add vgpr [0...0) to pool */
/* init: add agpr [0...8) to pool */

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
/* lr0I */
v_and_b32 v1, 63, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v0, 15, v1                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v0, 0x7, v0                          // 1. N offset: nOffset = nIdx * nStride(128)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
v_lshlrev_b32 v0, 0x1, v0                          // 4. apply VectorWidth: bnOffset = bnOffset * vw(2)
v_and_b32 v1, 63, v[vgprSerial]                    // 5. thread id in wave: wtid = tid % wavelength(64)
v_lshrrev_b32 v1, 4, v1                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshlrev_b32 v1, 0x3, v1                          // 5. K offset: lrKOffset = kIdx * mStride(8)
v_add_u32 v0, v1, v0                               // 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v1, 6, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
v_and_b32 v1, 3, v1                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(4)
v_lshlrev_b32 v1, 0xc, v1                          // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(4096)
v_add_u32 v0, v1, v0                               // 7. final local read offset: flrOffset = lrOffset + WOffset
/* lr1J */
v_and_b32 v2, 63, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v1, 0x7, v1                          // 1. N offset: nOffset = nIdx * nStride(128)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_and_b32 v2, 63, v[vgprSerial]                    // 5. thread id in wave: wtid = tid % wavelength(64)
v_lshrrev_b32 v2, 4, v2                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshlrev_b32 v2, 0x3, v2                          // 5. K offset: lrKOffset = kIdx * mStride(8)
v_add_u32 v1, v2, v1                               // 6. offset in wave: lrOffset = bnOffset + lrKOffset

/* local read addresses: final offsets a */
v_lshrrev_b32 v2, 6, v[vgprSerial]                 // v2 = v[vgprSerial] / 64
v_lshrrev_b32 v2, 2, v2                            // LSU offset: Get LSU wave_id
s_mov_b32 s49, 128                                 // LSU offset: stride = lsuStride(128) when umlds==True
v_mul_lo_u32 v2, s49, v2                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_lshl_u32 v[vgprLocalReadAddrA], v2, v0, 0x1  // Final Offset: offset = (lro0+lsuoffset)*bpeDS
v_lshrrev_b32 v3, 9, v[vgprLocalReadAddrA]         // Final Offset: padding 32 per block 512
v_lshlrev_b32 v3, 0x5, v3                          // Final Offset: padding 32 per block 512
v_add_u32 v[vgprLocalReadAddrA], v3, v[vgprLocalReadAddrA] // Final Offset: add padding 32 per block 512

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 6, v[vgprSerial]                 // v0 = v[vgprSerial] / 64
v_lshrrev_b32 v0, 2, v0                            // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(128) when umlds==True (dup assign opt.)
v_mul_lo_u32 v0, s49, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_lshl_u32 v[vgprLocalReadAddrB], v0, v1, 0x1  // Final Offset: offset = (lro1+lsuoffset)*bpeDS
v_lshrrev_b32 v2, 8, v[vgprLocalReadAddrB]         // Final Offset: padding 32 per block 256
v_lshlrev_b32 v2, 0x5, v2                          // Final Offset: padding 32 per block 256
v_add_u32 v[vgprLocalReadAddrB], v2, v[vgprLocalReadAddrB] // Final Offset: add padding 32 per block 256

/* local read addresses: declare addresses a */
/* N/A */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, 0x8800, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 16 */
/* v1 = A-unroll = serial%LVCA */
v_lshrrev_b32 v0, 4, v[vgprSerial]                 // v0 = v[vgprSerial] / 16
v_and_b32 v1, 15, v[vgprSerial]                    // v1 = v[vgprSerial] % 16
/* unroll *= glvw */
v_lshlrev_b32 v1, 0x3, v1                          // v1 = v1 * 8
v_mov_b32 v4, v1                                   // copy for GlobalSplitU
/* LVCB = 16 */
/* v3 = B-unroll = serial%LVCB */
v_lshrrev_b32 v2, 4, v[vgprSerial]                 // v2 = v[vgprSerial] / 16
v_and_b32 v3, 15, v[vgprSerial]                    // v3 = v[vgprSerial] % 16
/* unroll *= glvw */
v_lshlrev_b32 v3, 0x3, v3                          // v3 = v3 * 8
v_mov_b32 v5, v3                                   // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v4 */
/* lwaUnrollAssignmentB = v5 */

/* local write addresses: first offset a */
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x80, v0     // lwAL**(DepthU_Compute + PAD)
v_add_lshl_u32 v[vgprLocalWriteAddrA], v4, v[vgprLocalWriteAddrA], 0x1 // lwFOA = (lwAA + lwAL*(DepthU+PAD))*bpeDS
v_lshrrev_b32 v6, 9, v[vgprLocalWriteAddrA]        // padding 32 per block 512
v_lshlrev_b32 v6, 0x5, v6                          // padding 32 per block 512
v_add_u32 v[vgprLocalWriteAddrA], v6, v[vgprLocalWriteAddrA] // add padding 32 per block 512

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x80, v2     // lwBL**(DepthU_Compute + PAD)
v_add_lshl_u32 v[vgprLocalWriteAddrB], v5, v[vgprLocalWriteAddrB], 0x1 // lwFOB = (lwBB + lwBL*(DepthU+PAD))*bpeDS
v_lshrrev_b32 v6, 8, v[vgprLocalWriteAddrB]        // padding 32 per block 256
v_lshlrev_b32 v6, 0x5, v6                          // padding 32 per block 256
v_add_u32 v[vgprLocalWriteAddrB], v6, v[vgprLocalWriteAddrB] // add padding 32 per block 256
v_add_co_u32 v[vgprLocalWriteAddrB], vcc, 0x8800, v[vgprLocalWriteAddrB] // lwFOB = lwB1J + lwBL*MT1J + LDS_OFFSET_B=34816
v_mov_b32 v8, MT0                                  // set MT0 into sgpr
v_mov_b32 v7, s[sgprSizesFree+0]                   // set Free0 size
v_cvt_f32_u32 v6, v8                               // v6 = ceil(v7 / v8)
v_rcp_iflag_f32 v6, v6                             // v6 = ceil(v7 / v8)
v_cvt_f32_u32 v9, v7                               // v6 = ceil(v7 / v8)
v_mul_f32 v6, v6, v9                               // v6 = ceil(v7 / v8)
v_cvt_u32_f32 v6, v6                               // v6 = ceil(v7 / v8)
v_mul_u32_u24 v9, v6, v8                           // v6 = ceil(v7 / v8)
v_sub_u32 v9, v7, v9                               // v6 = ceil(v7 / v8)
v_cmp_ne_u32 vcc, v9, 0                            // v6 = ceil(v7 / v8)
v_addc_co_u32 v6, vcc, v6, 0, vcc                  // ceil
v_mov_b32 v8, MT1                                  // set MT1 into sgpr
v_mov_b32 v7, s[sgprSizesFree+1]                   // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v6      // set back to numWorkGroup0
v_cvt_f32_u32 v6, v8                               // v6 = ceil(v7 / v8)
v_rcp_iflag_f32 v6, v6                             // v6 = ceil(v7 / v8)
v_cvt_f32_u32 v9, v7                               // v6 = ceil(v7 / v8)
v_mul_f32 v6, v6, v9                               // v6 = ceil(v7 / v8)
v_cvt_u32_f32 v6, v6                               // v6 = ceil(v7 / v8)
v_mul_u32_u24 v9, v6, v8                           // v6 = ceil(v7 / v8)
v_sub_u32 v9, v7, v9                               // v6 = ceil(v7 / v8)
v_cmp_ne_u32 vcc, v9, 0                            // v6 = ceil(v7 / v8)
v_addc_co_u32 v6, vcc, v6, 0, vcc                  // ceil
s_nop 0                                            // 1 wait states
v_readfirstlane_b32 s[sgprNumWorkGroups1], v6      // set back to numWorkGroup1
s_waitcnt lgkmcnt(0)                               // wait for 44/0 bytes of kern args
s_branch label_MultiGemmEnd
label_MultiGemm:

/* Check if custom structure pointer is null */
s_cmp_eq_u32 s[sgprArgType], 2                     // ArgType == 2 ?
s_cbranch_scc1 label_IsExternalValid               // branch if ArgType == 2
s_mov_b32 s15, 88
s_mul_i32 s54, s47, 4
s_mov_b64 s[48:49], s[sgprKernArgAddress:sgprKernArgAddress+1]
s_branch label_IsExternalValidEnd
label_IsExternalValid:
s_mov_b32 s15, 196
s_mov_b32 s54, 0x0
s_mov_b64 s[48:49], s[sgprKernArgAddress:sgprKernArgAddress+1]
label_IsExternalValidEnd:

/* Grouped Gemm:: prefetch 1 arg load */
s_mov_b32 s14, 1
s_mov_b32 s55, 0
s_load_dwordx4 s[24:27], s[48:49], s54
s_cmpk_eq_u32 s47, 1                               // if gemm_count is 1?
s_cbranch_scc1 label_wgTable_noLoadLoop

/* Grouped Gemm:: accumulate numTiles for each gemm */
/* Grouped Gemm:: loop start */
label_Loop_GemmCount:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s52, s24, 7                             // s52 = s24 / 128
s_and_b32 s50, 127, s24                            // s50 = s24 % 128
s_addc_u32 s52, s52, 0x0
s_lshr_b32 s53, s25, 4                             // s53 = s25 / 16
s_and_b32 s50, 15, s25                             // s50 = s25 % 16
s_addc_u32 s53, s53, 0x0
s_mul_i32 s52, s52, s53
s_mul_i32 s52, s52, s26
s_mul_i32 s52, s52, s[sgprGSU]
s_add_u32 s55, s55, s52
s_cmp_lt_u32 s[sgprWorkGroup0], s55
s_cbranch_scc1 label_FOUND
s_add_u32 s54, s54, s15
s_load_dwordx4 s[24:27], s[48:49], s54
s_add_u32 s14, s14, 1
s_cmp_lt_u32 s14, s47
s_cbranch_scc1 label_Loop_GemmCount

/* Grouped Gemm:: noLoadLoop */
label_wgTable_noLoadLoop:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s52, s24, 7                             // s52 = s24 / 128
s_and_b32 s50, 127, s24                            // s50 = s24 % 128
s_addc_u32 s52, s52, 0x0
s_lshr_b32 s53, s25, 4                             // s53 = s25 / 16
s_and_b32 s50, 15, s25                             // s50 = s25 % 16
s_addc_u32 s53, s53, 0x0
s_mul_i32 s52, s52, s53
s_mul_i32 s52, s52, s26
s_mul_i32 s52, s52, s[sgprGSU]
s_add_u32 s55, s55, s52

/* Grouped Gemm:: gemmIndex found */
label_FOUND:
s_sub_u32 s49, s14, 1
s_sub_u32 s48, s55, s52
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s48
/* Check if custom structure pointer is null */
s_cmp_eq_u32 s[sgprArgType], 2                     // ArgType == 2 ?
s_cbranch_scc1 label_LoadExternalStruct            // branch if ArgType == 2

/* Grouped Gemm: offset argument address to gemm */
/* Grouped Gemm: offset address from wg_table_start to args_start */
s_lshl2_add_u32 s[sgprKernArgAddress], s47, s[sgprKernArgAddress]
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0x0
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s49, s49, 88
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s49
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0x0

/* Load Kernel Args */
s_load_dwordx16 s[28:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x10
s_load_dwordx2 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x50
s_branch label_LoadExternalStructEnd
label_LoadExternalStruct:
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s49, s49, 196
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s49
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0x0
s_load_dwordx16 s[28:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x10
s_load_dword s44, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x50
// Read Beta
s_load_dword s45, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x60
label_LoadExternalStructEnd:
/* init: add vgpr [0...48) to pool */
/* init: add vgpr [0...0) to pool */
/* init: add agpr [0...8) to pool */

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
/* lr0I */
v_and_b32 v1, 63, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v0, 15, v1                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v0, 0x7, v0                          // 1. N offset: nOffset = nIdx * nStride(128)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
v_lshlrev_b32 v0, 0x1, v0                          // 4. apply VectorWidth: bnOffset = bnOffset * vw(2)
v_and_b32 v1, 63, v[vgprSerial]                    // 5. thread id in wave: wtid = tid % wavelength(64)
v_lshrrev_b32 v1, 4, v1                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshlrev_b32 v1, 0x3, v1                          // 5. K offset: lrKOffset = kIdx * mStride(8)
v_add_u32 v0, v1, v0                               // 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v1, 6, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
v_and_b32 v1, 3, v1                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(4)
v_lshlrev_b32 v1, 0xc, v1                          // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(4096)
v_add_u32 v0, v1, v0                               // 7. final local read offset: flrOffset = lrOffset + WOffset
/* lr1J */
v_and_b32 v2, 63, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v1, 0x7, v1                          // 1. N offset: nOffset = nIdx * nStride(128)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_and_b32 v2, 63, v[vgprSerial]                    // 5. thread id in wave: wtid = tid % wavelength(64)
v_lshrrev_b32 v2, 4, v2                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshlrev_b32 v2, 0x3, v2                          // 5. K offset: lrKOffset = kIdx * mStride(8)
v_add_u32 v1, v2, v1                               // 6. offset in wave: lrOffset = bnOffset + lrKOffset

/* local read addresses: final offsets a */
v_lshrrev_b32 v2, 6, v[vgprSerial]                 // v2 = v[vgprSerial] / 64
v_lshrrev_b32 v2, 2, v2                            // LSU offset: Get LSU wave_id
s_mov_b32 s49, 128                                 // LSU offset: stride = lsuStride(128) when umlds==True
v_mul_lo_u32 v2, s49, v2                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_lshl_u32 v[vgprLocalReadAddrA], v2, v0, 0x1  // Final Offset: offset = (lro0+lsuoffset)*bpeDS
v_lshrrev_b32 v3, 9, v[vgprLocalReadAddrA]         // Final Offset: padding 32 per block 512
v_lshlrev_b32 v3, 0x5, v3                          // Final Offset: padding 32 per block 512
v_add_u32 v[vgprLocalReadAddrA], v3, v[vgprLocalReadAddrA] // Final Offset: add padding 32 per block 512

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 6, v[vgprSerial]                 // v0 = v[vgprSerial] / 64
v_lshrrev_b32 v0, 2, v0                            // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(128) when umlds==True (dup assign opt.)
v_mul_lo_u32 v0, s49, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_lshl_u32 v[vgprLocalReadAddrB], v0, v1, 0x1  // Final Offset: offset = (lro1+lsuoffset)*bpeDS
v_lshrrev_b32 v2, 8, v[vgprLocalReadAddrB]         // Final Offset: padding 32 per block 256
v_lshlrev_b32 v2, 0x5, v2                          // Final Offset: padding 32 per block 256
v_add_u32 v[vgprLocalReadAddrB], v2, v[vgprLocalReadAddrB] // Final Offset: add padding 32 per block 256

/* local read addresses: declare addresses a */
/* N/A */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, 0x8800, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 16 */
/* v1 = A-unroll = serial%LVCA */
v_lshrrev_b32 v0, 4, v[vgprSerial]                 // v0 = v[vgprSerial] / 16
v_and_b32 v1, 15, v[vgprSerial]                    // v1 = v[vgprSerial] % 16
/* unroll *= glvw */
v_lshlrev_b32 v1, 0x3, v1                          // v1 = v1 * 8
v_mov_b32 v4, v1                                   // copy for GlobalSplitU
/* LVCB = 16 */
/* v3 = B-unroll = serial%LVCB */
v_lshrrev_b32 v2, 4, v[vgprSerial]                 // v2 = v[vgprSerial] / 16
v_and_b32 v3, 15, v[vgprSerial]                    // v3 = v[vgprSerial] % 16
/* unroll *= glvw */
v_lshlrev_b32 v3, 0x3, v3                          // v3 = v3 * 8
v_mov_b32 v5, v3                                   // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v4 */
/* lwaUnrollAssignmentB = v5 */

/* local write addresses: first offset a */
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x80, v0     // lwAL**(DepthU_Compute + PAD)
v_add_lshl_u32 v[vgprLocalWriteAddrA], v4, v[vgprLocalWriteAddrA], 0x1 // lwFOA = (lwAA + lwAL*(DepthU+PAD))*bpeDS
v_lshrrev_b32 v6, 9, v[vgprLocalWriteAddrA]        // padding 32 per block 512
v_lshlrev_b32 v6, 0x5, v6                          // padding 32 per block 512
v_add_u32 v[vgprLocalWriteAddrA], v6, v[vgprLocalWriteAddrA] // add padding 32 per block 512

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x80, v2     // lwBL**(DepthU_Compute + PAD)
v_add_lshl_u32 v[vgprLocalWriteAddrB], v5, v[vgprLocalWriteAddrB], 0x1 // lwFOB = (lwBB + lwBL*(DepthU+PAD))*bpeDS
v_lshrrev_b32 v6, 8, v[vgprLocalWriteAddrB]        // padding 32 per block 256
v_lshlrev_b32 v6, 0x5, v6                          // padding 32 per block 256
v_add_u32 v[vgprLocalWriteAddrB], v6, v[vgprLocalWriteAddrB] // add padding 32 per block 256
v_add_co_u32 v[vgprLocalWriteAddrB], vcc, 0x8800, v[vgprLocalWriteAddrB] // lwFOB = lwB1J + lwBL*MT1J + LDS_OFFSET_B=34816
v_mov_b32 v8, MT0                                  // set MT0 into sgpr
v_mov_b32 v7, s[sgprSizesFree+0]                   // set Free0 size
v_cvt_f32_u32 v6, v8                               // v6 = ceil(v7 / v8)
v_rcp_iflag_f32 v6, v6                             // v6 = ceil(v7 / v8)
v_cvt_f32_u32 v9, v7                               // v6 = ceil(v7 / v8)
v_mul_f32 v6, v6, v9                               // v6 = ceil(v7 / v8)
v_cvt_u32_f32 v6, v6                               // v6 = ceil(v7 / v8)
v_mul_u32_u24 v9, v6, v8                           // v6 = ceil(v7 / v8)
v_sub_u32 v9, v7, v9                               // v6 = ceil(v7 / v8)
v_cmp_ne_u32 vcc, v9, 0                            // v6 = ceil(v7 / v8)
v_addc_co_u32 v6, vcc, v6, 0, vcc                  // ceil
v_mov_b32 v8, MT1                                  // set MT1 into sgpr
v_mov_b32 v7, s[sgprSizesFree+1]                   // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v6      // set back to numWorkGroup0
v_cvt_f32_u32 v6, v8                               // v6 = ceil(v7 / v8)
v_rcp_iflag_f32 v6, v6                             // v6 = ceil(v7 / v8)
v_cvt_f32_u32 v9, v7                               // v6 = ceil(v7 / v8)
v_mul_f32 v6, v6, v9                               // v6 = ceil(v7 / v8)
v_cvt_u32_f32 v6, v6                               // v6 = ceil(v7 / v8)
v_mul_u32_u24 v9, v6, v8                           // v6 = ceil(v7 / v8)
v_sub_u32 v9, v7, v9                               // v6 = ceil(v7 / v8)
v_cmp_ne_u32 vcc, v9, 0                            // v6 = ceil(v7 / v8)
v_addc_co_u32 v6, vcc, v6, 0, vcc                  // ceil
s_nop 0                                            // 1 wait states
v_readfirstlane_b32 s[sgprNumWorkGroups1], v6      // set back to numWorkGroup1
s_waitcnt lgkmcnt(0)                               // wait for 44/0 bytes of kern args

/* Early stop if N(SizeFreeJ) == 0 */
s_cmp_eq_u32 s[sgprSizeJ], 0x0
s_cbranch_scc0 label_NoEarlyStop_N0
label_EarlyStop_if_N_is_0:
s_endpgm
label_NoEarlyStop_N0:

/* Grouped Gemm: remap wg from 1D(idxWG012) to 3D(wg2,wg1,wg0) */
/* wg2 = idxWG012 * smallMagicNumber(1/(numWG0*numWG1)) */
s_mul_i32 s48, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_mul_i32 s48, s48, s[sgprGSU]
v_cvt_f32_u32 v6, s48                              // s48 = s[sgprWorkGroup0] / s48
v_rcp_iflag_f32 v6, v6                             // s48 = s[sgprWorkGroup0] / s48
v_cvt_f32_u32 v7, s[sgprWorkGroup0]                // s48 = s[sgprWorkGroup0] / s48
v_mul_f32 v6, v6, v7                               // s48 = s[sgprWorkGroup0] / s48
v_cvt_u32_f32 v6, v6                               // s48 = s[sgprWorkGroup0] / s48
v_mul_u32_u24 v7, v6, s48                          // s48 = s[sgprWorkGroup0] / s48
v_sub_u32 v7, s[sgprWorkGroup0], v7                // s48 = s[sgprWorkGroup0] / s48
v_cmpx_eq_u32 exec, v7, s48                        // s48 = s[sgprWorkGroup0] / s48
v_add_u32 v6, 1, v6                                // s48 = s[sgprWorkGroup0] / s48
s_mov_b64 exec, -1                                 // s48 = s[sgprWorkGroup0] / s48
v_readfirstlane_b32 s48, v6
s_mov_b32 s[sgprWorkGroup2], s48
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s48, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s48, s48, s[sgprWorkGroup2]
s_mul_i32 s48, s48, s[sgprGSU]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s48
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v6, s[sgprNumWorkGroups0]            // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v6, v6                             // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_f32_u32 v7, s[sgprWorkGroup0]                // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_f32 v6, v6, v7                               // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_u32_f32 v6, v6                               // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_u32_u24 v7, v6, s[sgprNumWorkGroups0]        // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_sub_u32 v7, s[sgprWorkGroup0], v7                // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmpx_eq_u32 exec, v7, s[sgprNumWorkGroups0]      // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_add_u32 v6, 1, v6                                // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b64 exec, -1                                 // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_readfirstlane_b32 s48, v6
s_mov_b32 s[sgprWorkGroup1], s48
/* wg0 = idxWG01 - wg1 * numWG0 */
s_mul_i32 s48, s[sgprWorkGroup1], s[sgprNumWorkGroups0]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s48

/* Early stop if wg exceed */
s_cmp_ge_u32 s[sgprWorkGroup2], s[sgprSizesFree+2]
s_cbranch_scc0 label_NoEarlyStop_wgExceed
label_EarlyStop_if_wg_exceed:
s_endpgm
label_NoEarlyStop_wgExceed:

label_MultiGemmEnd:
.set sgprSrdA, 48
.set sgprSrdB, 52
.set sgprShadowLimitA, 56
.set sgprShadowLimitB, 58
.set sgprStaggerUIter, 47
.set sgprWrapUA, 60
.set sgprWrapUB, 62
.set sgprGlobalReadIncsA, 64
.set sgprGlobalReadIncsB, 65
.set sgprScalarGlobalReadOffsetA, 66
s_sub_u32 s[sgprAddressA+0], s[sgprAddressA+0], 16 // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressA+1], s[sgprAddressA+1], 0 // pre-pad to make room for possible pointer shift
s_sub_u32 s[sgprAddressB+0], s[sgprAddressB+0], 16 // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressB+1], s[sgprAddressB+1], 0 // pre-pad to make room for possible pointer shift

/* Short circuit condition if Alpha == 0, then sumDims=0 */
v_cmp_eq_f32 vcc, s[sgprAlpha], 0.0                // s[Alpha] == 0.0f ?
s_cbranch_vccz label_AlphaNonZero                  // branch if s[Alpha] != 0
s_mov_b32 s[sgprSizesSum+0], 0x0                   // Set summation dim=0 if Alpha == 0
label_AlphaNonZero:

/******************************************/
/* Begin setupNewTile                     */
/******************************************/

/* global read addresses: work-group */
/* graWorkGroup mapping */
s_cmp_eq_u32 s[sgprGSU], 1                         // GSU == 1 ?
s_cbranch_scc1 label_GSU                           // branch if GSU == 1
// GSU-not-WGMapRR :nwg1 = (size1J + MT1J - 1) / MT1J;
v_cvt_f32_u32 v6, s[sgprGSU]                       // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s[sgprGSU]
v_rcp_iflag_f32 v6, v6                             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s[sgprGSU]
v_cvt_f32_u32 v7, s[sgprWorkGroup1]                // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s[sgprGSU]
v_mul_f32 v6, v6, v7                               // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s[sgprGSU]
v_cvt_u32_f32 v6, v6                               // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s[sgprGSU]
v_mul_u32_u24 v7, v6, s[sgprGSU]                   // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s[sgprGSU]
v_sub_u32 v7, s[sgprWorkGroup1], v7                // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s[sgprGSU]
v_cmpx_eq_u32 exec, v7, s[sgprGSU]                 // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s[sgprGSU]
v_add_u32 v6, 1, v6                                // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s[sgprGSU]
v_mov_b32 v7, 0                                    // s[sgprGSUSumIdx] = s[sgprWorkGroup1] % s[sgprGSU]
s_mov_b64 exec, -1                                 // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s[sgprGSU]
v_readfirstlane_b32 s[sgprWorkGroup1], v6
v_readfirstlane_b32 s[sgprGSUSumIdx], v7
s_mov_b32 s[sgprGSULog2BpeC], 1
s_mov_b32 s[sgprGSULog2BpeD], 2
s_branch label_GSU_End
label_GSU:
s_mov_b64 s[sgprGSUSumIdx:sgprGSUSumIdx+1], 0      // Set GSUSumIdx to 0
s_mov_b32 s[sgprGSULog2BpeC], 1
s_mov_b32 s[sgprGSULog2BpeD], 1
label_GSU_End:
s_cmp_le_u32 s[sgprWGM], 1                         // WGM <= 1 ?
s_cbranch_scc1 label_WGM                           // branch if WGM <= 1
v_cvt_f32_u32 v6, s[sgprWGM]                       // WGM
v_rcp_iflag_f32 v6, v6                             // WGM
v_cvt_f32_u32 v7, s[sgprWorkGroup1]                // WGM
v_mul_f32 v6, v6, v7                               // WGM
v_cvt_u32_f32 v6, v6                               // WGM
v_mul_u32_u24 v7, v6, s[sgprWGM]                   // WGM
v_sub_u32 v7, s[sgprWorkGroup1], v7                // WGM
v_cmpx_eq_u32 exec, v7, s[sgprWGM]                 // WGM
v_add_u32 v6, 1, v6                                // WGM
s_mov_b64 exec, -1                                 // WGM
v_readfirstlane_b32 s73, v6
s_mul_i32 s76, s73, s[sgprWGM]                     // quotient * non-magic divisor
s_sub_u32 s76, s[sgprWorkGroup1], s76              // WorkGroup1=remainder
s_mul_i32 s76, s76, s[sgprNumWorkGroups0]          // (wg1 % WGM)*nwg0
s_add_u32 s76, s76, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*nwg0
v_cvt_f32_u32 v6, s[sgprWGM]                       // WGM
v_rcp_iflag_f32 v6, v6                             // WGM
v_cvt_f32_u32 v7, s[sgprNumWorkGroups1]            // WGM
v_mul_f32 v6, v6, v7                               // WGM
v_cvt_u32_f32 v6, v6                               // WGM
v_mul_u32_u24 v7, v6, s[sgprWGM]                   // WGM
v_sub_u32 v7, s[sgprNumWorkGroups1], v7            // WGM
v_cmpx_eq_u32 exec, v7, s[sgprWGM]                 // WGM
v_add_u32 v6, 1, v6                                // WGM
s_mov_b64 exec, -1                                 // WGM
v_readfirstlane_b32 s74, v6
s_mul_i32 s75, s[sgprWGM], s74                     // quotient * non-magic divisor
s_sub_u32 s75, s[sgprNumWorkGroups1], s75          // WorkGroup1=remainder
s_cmp_eq_u32 s75, 0                                // remainder == 0 ?
s_cmov_b32 s75, s[sgprWGM]                         // remainder = WGM if remainder == 0
s_cmp_ge_u32 s73, s74                              // blockId >= numFullBlocks ?
s_cselect_b32 s74, s75, s[sgprWGM]
v_cvt_f32_u32 v6, s74                              // s[sgprWorkGroup0] = s76 / s74
v_rcp_iflag_f32 v6, v6                             // s[sgprWorkGroup0] = s76 / s74
v_cvt_f32_u32 v7, s76                              // s[sgprWorkGroup0] = s76 / s74
v_mul_f32 v6, v6, v7                               // s[sgprWorkGroup0] = s76 / s74
v_cvt_u32_f32 v6, v6                               // s[sgprWorkGroup0] = s76 / s74
v_mul_u32_u24 v7, v6, s74                          // s[sgprWorkGroup0] = s76 / s74
v_sub_u32 v7, s76, v7                              // s[sgprWorkGroup0] = s76 / s74
v_cmpx_eq_u32 exec, v7, s74                        // s[sgprWorkGroup0] = s76 / s74
v_add_u32 v6, 1, v6                                // s[sgprWorkGroup0] = s76 / s74
v_mov_b32 v7, 0                                    // s[sgprWorkGroup1] = s76 % s74
s_mov_b64 exec, -1                                 // s[sgprWorkGroup0] = s76 / s74
v_readfirstlane_b32 s[sgprWorkGroup0], v6
v_readfirstlane_b32 s[sgprWorkGroup1], v7
s_mul_i32 s73, s73, s[sgprWGM]                     // blockId * WGM
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s73 // wg1 += blockId * WGM
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

/* global read addresses: tile offsets b */

/* global read addresses: unroll offsets a */

/* global read addresses: unroll offsets b */

/* global read addresses: final offsets a */
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+0,  1,  0, 6 // gROA_0_0_0_0
s_mul_i32 s[sgprScalarGlobalReadOffsetA+0], s[sgprStrideA0I], 16 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetA+0], s[sgprScalarGlobalReadOffsetA+0], 0x1 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetA+1], s[sgprStrideA0I], 32 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetA+1], s[sgprScalarGlobalReadOffsetA+1], 0x1 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetA+2], s[sgprStrideA0I], 48 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetA+2], s[sgprScalarGlobalReadOffsetA+2], 0x1 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetA+3], s[sgprStrideA0I], 64 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetA+3], s[sgprScalarGlobalReadOffsetA+3], 0x1 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetA+4], s[sgprStrideA0I], 80 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetA+4], s[sgprScalarGlobalReadOffsetA+4], 0x1 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetA+5], s[sgprStrideA0I], 96 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetA+5], s[sgprScalarGlobalReadOffsetA+5], 0x1 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetA+6], s[sgprStrideA0I], 112 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetA+6], s[sgprScalarGlobalReadOffsetA+6], 0x1 // scalar offset *= bytes/element

/* global read addresses: final offsets b */
GLOBAL_OFFSET_B vgprGlobalReadOffsetB+0,  3,  2, 6 // gROB_0_0_0_0

/* global read addresses: addresses a */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s77, s[sgprWorkGroup0], 128           // WorkGroup[01] * MT
s_mul_i32 s76, s[sgprWorkGroup0], 128              // WorkGroup[01] * MT
s_mul_hi_u32 s77, s76, s[sgprStrideA0I]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s76, s76, s[sgprStrideA0I]               // tlu=0, scaled tile-offset by stride
s_mul_hi_u32 s75, 128, s[sgprGSUSumIdx]            // gsuOffset = DepthU*bpeGR*GSUSumIdx
s_mul_i32 s74, 128, s[sgprGSUSumIdx]               // gsuOffset = DepthU*bpeGR*GSUSumIdx
s_add_u32 s76, s76, s74                            // accum GsuOffset term to tilestart
s_addc_u32 s77, s77, s75                           // accum GsuOffset term to tilestart
s_mov_b32 s[sgprShadowLimitA+0], 1                 // Init tensor size
s_mov_b32 s[sgprShadowLimitA+1], 0                 // init tensor size
s_sub_u32 s74, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s75, constStrideAL, s74               // stride x (size-1)
s_mul_i32 s74, constStrideAL, s74                  // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s74 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s75 // sum tensor size
s_sub_u32 s74, s[sgprSizeI], 1                     // (size-1)
s_mul_hi_u32 s75, s[sgprStrideA0I], s74            // stride x (size-1)
s_mul_i32 s74, s[sgprStrideA0I], s74               // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s74 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s75 // sum tensor size
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s76 // sub tileStart
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s77 // sub tileStart
s_lshl_b64 s[sgprShadowLimitA:sgprShadowLimitA+1], s[sgprShadowLimitA:sgprShadowLimitA+1], 0x1 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_mul_hi_u32 s75, s[sgprStrideAK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s74, s[sgprStrideAK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s76, s76, s74                            // accum wg term to tilestart
s_addc_u32 s77, s77, s75                           // accum wg term to tilestart
s_lshl_b64 s[76:77], s[76:77], 0x1                 // tileStart *= BPE
s_add_u32 s[sgprSrdA+0], s[sgprAddressA+0], s76    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprAddressA+1], s77   // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdA+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: addresses b */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s77, s[sgprWorkGroup1], 16            // WorkGroup[01] * MT
s_mul_i32 s76, s[sgprWorkGroup1], 16               // WorkGroup[01] * MT
s_mul_hi_u32 s77, s76, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s76, s76, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_mul_hi_u32 s75, 128, s[sgprGSUSumIdx]            // gsuOffset = DepthU*bpeGR*GSUSumIdx
s_mul_i32 s74, 128, s[sgprGSUSumIdx]               // gsuOffset = DepthU*bpeGR*GSUSumIdx
s_add_u32 s76, s76, s74                            // accum GsuOffset term to tilestart
s_addc_u32 s77, s77, s75                           // accum GsuOffset term to tilestart
s_mov_b32 s[sgprShadowLimitB+0], 1                 // Init tensor size
s_mov_b32 s[sgprShadowLimitB+1], 0                 // init tensor size
s_sub_u32 s74, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s75, constStrideBL, s74               // stride x (size-1)
s_mul_i32 s74, constStrideBL, s74                  // stride x (size-1)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s74 // sum tensor size
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s75 // sum tensor size
s_sub_u32 s74, s[sgprSizeJ], 1                     // (size-1)
s_mul_hi_u32 s75, s[sgprStrideB1J], s74            // stride x (size-1)
s_mul_i32 s74, s[sgprStrideB1J], s74               // stride x (size-1)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s74 // sum tensor size
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s75 // sum tensor size
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s76 // sub tileStart
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s77 // sub tileStart
s_lshl_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], 0x1 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_mul_hi_u32 s75, s[sgprStrideBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s74, s[sgprStrideBK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s76, s76, s74                            // accum wg term to tilestart
s_addc_u32 s77, s77, s75                           // accum wg term to tilestart
s_lshl_b64 s[76:77], s[76:77], 0x1                 // tileStart *= BPE
s_add_u32 s[sgprSrdB+0], s[sgprAddressB+0], s76    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprAddressB+1], s77   // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdB+3], Srd127_96                 // Set bits 127_96 in SRD
s_mul_i32 s73, s[sgprGSU], DepthU*BpeAGR
s_mov_b32 s[sgprGlobalReadIncsA+0], s73            // incrA (unrollIdx)

/* global read addresses: increments b */
s_mul_i32 s73, s[sgprGSU], DepthU*BpeBGR
s_mov_b32 s[sgprGlobalReadIncsB+0], s73            // incrB (unrollIdx)
/* declare loop num iterations */
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 7 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 128
s_cmp_eq_u32 s[sgprGSU], 1                         // GSU == 1 ?
s_cbranch_scc1 label_GSU_1                         // branch if GSU == 1
v_cvt_f32_u32 v0, s[sgprGSU]                       // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSU]
v_rcp_iflag_f32 v0, v0                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSU]
v_cvt_f32_u32 v1, s[sgprLoopCounterL]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSU]
v_mul_f32 v0, v0, v1                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSU]
v_cvt_u32_f32 v0, v0                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSU]
v_mul_u32_u24 v1, v0, s[sgprGSU]                   // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSU]
v_sub_u32 v1, s[sgprLoopCounterL], v1              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSU]
v_cmpx_eq_u32 exec, v1, s[sgprGSU]                 // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSU]
v_add_u32 v0, 1, v0                                // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSU]
v_mov_b32 v1, 0                                    // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSU]
s_mov_b64 exec, -1                                 // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSU]
v_readfirstlane_b32 s[sgprLoopCounterL], v0
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v1
s_add_u32 s74, 1, s[sgprLoopCounterL]              // tmp<-numIterMyWg+
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cmov_b32 s[sgprLoopCounterL], s74                // numIterMyWg++ if needed
label_GSU_1:
s_mov_b32 s[sgprOrigLoopCounter], s[sgprLoopCounterL] // copy loop counter
s_and_b32 s76, s[sgprStaggerU], 0x1f00
s_lshr_b32 s76, s76, 0x8
s_and_b32 s77, s[sgprStaggerU], 0xe000
s_and_b32 s[sgprStaggerU], s[sgprStaggerU], 0xff
s_mov_b32 s74, s[sgprStaggerU]                     // init staggerU
label_beginStaggerUIter:
s_lshl_b32 s75, s74, s76                           // shift by StaggerUStride
s_cmp_ge_u32 s[sgprOrigLoopCounter], s75           // loopCount >= current shift Count
s_cbranch_scc1 label_endStaggerUIter               // jump to end
s_lshr_b32 s74, s74, 1                             // step down to smaller stagger
s_branch label_beginStaggerUIter                   // jump to begin
label_endStaggerUIter:
s_sub_u32 s75, s74, 1                              // staggerU mask
s_cmp_ge_u32 s74, 1                                // if current staggerU >= 1
s_cselect_b32 s[sgprStaggerUIter], s75, 0          // set Mask
s_cmp_eq_u32 s77, 0x0
s_cbranch_scc1 label_StaggerUMapping_1
s_mov_b32 s74, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping_1:
s_cmp_eq_u32 s77, 0x2000
s_cbranch_scc1 label_StaggerUMapping_2
s_mov_b32 s74, s[sgprWorkGroup1]
s_branch label_staggerInputEnd
label_StaggerUMapping_2:
s_cmp_eq_u32 s77, 0x4000
s_cbranch_scc1 label_StaggerUMapping_3
s_mov_b32 s74, -0x1
s_branch label_staggerInputEnd
label_StaggerUMapping_3:
s_cmp_eq_u32 s77, 0x6000
s_cbranch_scc1 label_StaggerUMapping_4
s_mul_i32 s75, s[sgprNumWorkGroups0], s[sgprWorkGroup1]
s_add_u32 s74, s74, s75
s_add_u32 s74, s74, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping_4:
s_cmp_eq_u32 s77, 0x8000
s_cbranch_scc1 label_staggerInputEnd
s_mov_b32 s74, -0x1
s_branch label_staggerInputEnd
label_staggerInputEnd:
s_and_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s74 // Compute actual stagger start for this tile
s_lshl_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s76 // shift by StaggerUStride

/* SRDs += (StaggerUIter) * GlobalReadIncsA+0 */
s_mul_hi_i32 s75, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_i32 s74, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_hi_i32 s[sgprWrapUA+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUA+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0], s[sgprWrapUA+0] // remove one iteration
s_subb_u32 s[sgprWrapUA+1], 0, s[sgprWrapUA+1]     // remove one iteration
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s74        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s75       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s74 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s75 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* SRDs += (StaggerUIter) * GlobalReadIncsB+0 */
s_mul_hi_i32 s75, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_i32 s74, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_hi_i32 s[sgprWrapUB+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUB+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0], s[sgprWrapUB+0] // remove one iteration
s_subb_u32 s[sgprWrapUB+1], 0, s[sgprWrapUB+1]     // remove one iteration
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s74        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s75       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s74 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s75 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_add_u32 s[sgprStaggerUIter], s[sgprStaggerUIter], 2 // Subtract (PGR-1); StaggerUIter now contains target iteration to wrap
/* local read addresses: init pointers a */

/* localReadInitPointers */
/* local read addresses: init pointers b */

/* localReadInitPointers */

/* first prefetch using set0 registers: GR0 */

/* prefetch: global -> local */
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?
s_cbranch_scc1 label_ShadowInitStart               // skip to ShadowInitStart iter b/c numIter==0

buffer_load_dwordx4 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 nt // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprG2LA+4:vgprG2LA+4+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:0 nt // G -> Reg 0_0_1_0
buffer_load_dwordx4 v[vgprG2LA+8:vgprG2LA+8+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:0 nt // G -> Reg 0_0_2_0
buffer_load_dwordx4 v[vgprG2LA+12:vgprG2LA+12+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:0 nt // G -> Reg 0_0_3_0
buffer_load_dwordx4 v[vgprG2LA+16:vgprG2LA+16+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+3] offen offset:0 nt // G -> Reg 0_0_4_0
buffer_load_dwordx4 v[vgprG2LA+20:vgprG2LA+20+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+4] offen offset:0 nt // G -> Reg 0_0_5_0
buffer_load_dwordx4 v[vgprG2LA+24:vgprG2LA+24+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+5] offen offset:0 nt // G -> Reg 0_0_6_0
buffer_load_dwordx4 v[vgprG2LA+28:vgprG2LA+28+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+6] offen offset:0 nt // G -> Reg 0_0_7_0
buffer_load_dwordx4 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0

/* global read inc A loopL */
s_add_u32 s76, s[sgprLoopCounterL], 1              // remove pf(1)
s_cmp_eq_u32 s[sgprStaggerUIter], s76              // Is this wrapIter? (pf)
s_cselect_b32 s74, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s75, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s74        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s75       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s74 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s75 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s76, s[sgprLoopCounterL], 1              // remove pf(1)
s_cmp_eq_u32 s[sgprStaggerUIter], s76              // Is this wrapIter? (pf)
s_cselect_b32 s74, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s75, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s74        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s75       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s74 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s75 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32


/* 2nd set of PGR */
s_cmp_eq_u32 s[sgprLoopCounterL], 0x1              // PGR=3 but only 1 loop
s_cbranch_scc1 label_skipPGR1_0                    // PGR=3 but only 1 loop


buffer_load_dwordx4 v[vgprG2LA1+0:vgprG2LA1+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 nt // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprG2LA1+4:vgprG2LA1+4+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:0 nt // G -> Reg 0_0_1_0
buffer_load_dwordx4 v[vgprG2LA1+8:vgprG2LA1+8+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:0 nt // G -> Reg 0_0_2_0
buffer_load_dwordx4 v[vgprG2LA1+12:vgprG2LA1+12+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:0 nt // G -> Reg 0_0_3_0
buffer_load_dwordx4 v[vgprG2LA1+16:vgprG2LA1+16+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+3] offen offset:0 nt // G -> Reg 0_0_4_0
buffer_load_dwordx4 v[vgprG2LA1+20:vgprG2LA1+20+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+4] offen offset:0 nt // G -> Reg 0_0_5_0
buffer_load_dwordx4 v[vgprG2LA1+24:vgprG2LA1+24+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+5] offen offset:0 nt // G -> Reg 0_0_6_0
buffer_load_dwordx4 v[vgprG2LA1+28:vgprG2LA1+28+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+6] offen offset:0 nt // G -> Reg 0_0_7_0
buffer_load_dwordx4 v[vgprG2LB1+0:vgprG2LB1+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0

/* global read inc A loopL */
s_add_u32 s76, s[sgprLoopCounterL], 1              // remove pf(1)
s_cmp_eq_u32 s[sgprStaggerUIter], s76              // Is this wrapIter? (pf)
s_cselect_b32 s74, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s75, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s74        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s75       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s74 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s75 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s76, s[sgprLoopCounterL], 1              // remove pf(1)
s_cmp_eq_u32 s[sgprStaggerUIter], s76              // Is this wrapIter? (pf)
s_cselect_b32 s74, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s75, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s74        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s75       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s74 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s75 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32

label_skipPGR1_0:

/******************************************/
/* End setupNewTile                       */
/******************************************/
label_ShadowInitStart:
s_mov_b32 s[sgprSrdD+0], s[sgprAddressD+0]         // init SRD base address (lower)
s_mov_b32 s[sgprSrdD+1], s[sgprAddressD+1]         // init SRD base address (upper) + other fields
s_mov_b32 s[sgprSrdD+2], 0x80000000
s_mov_b32 s[sgprSrdD+3], Srd127_96                 // Set bits 127_96 in post-loop SRD

s_mov_b32 s[sgprSrdC+0], s[sgprAddressC+0]         // init SRD base address (lower)
s_mov_b32 s[sgprSrdC+1], s[sgprAddressC+1]         // init SRD base address (upper) + other fields
s_mov_b32 s[sgprSrdC+2], 0x80000000
s_mov_b32 s[sgprSrdC+3], Srd127_96                 // Set bits 127_96 in post-loop SRD


s_mul_i32 s76, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_mul_hi_u32 s75, s76, s[sgprStrideC1J]            // ScaleC s76 by Stride
s_mul_i32 s74, s76, s[sgprStrideC1J]               // ScaleC s76 by Stride
s_lshl_b64 s[74:75], s[74:75], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprAddressC+0], s74    // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprAddressC+1], s75   // add hi to SRD
s_mul_hi_u32 s75, s76, s[sgprStrideD1J]            // ScaleD s76 by Stride
s_mul_i32 s74, s76, s[sgprStrideD1J]               // ScaleD s76 by Stride
s_lshl_b64 s[74:75], s[74:75], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprAddressD+0], s74    // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprAddressD+1], s75   // add hi to SRD

s_mul_hi_u32 s75, s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride
s_mul_i32 s74, s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride
s_lshl_b64 s[74:75], s[74:75], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s74        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s75       // add hi to SRD
s_mul_hi_u32 s75, s[sgprWorkGroup2], s[sgprStrideDK] // ScaleD s[sgprWorkGroup2] by Stride
s_mul_i32 s74, s[sgprWorkGroup2], s[sgprStrideDK]  // ScaleD s[sgprWorkGroup2] by Stride
s_lshl_b64 s[74:75], s[74:75], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s74        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s75       // add hi to SRD

s_cmp_eq_u32 s[sgprGSU], 1                         // GSU == 1 ?
s_cbranch_scc1 label_GSU_2                         // branch if GSU == 1
// GSU Output Buffer offset: Free0 + (Free1-1)*StrideC1J + (Free2-1)*StrideCK * GSUIdx * bpe%s
s_mul_hi_u32 s75, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_mul_i32 s74, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_sub_u32 s73, s[sgprSizesFree+1], 1               // Free1
s_mul_i32 s73, s73, s[sgprGSUSumIdx]               // Free1
s_mul_hi_u32 s76, s73, s[sgprStrideC1J]            // Free1
s_mul_i32 s73, s73, s[sgprStrideC1J]               // Free1
s_add_u32 s74, s74, s73                            // Free1
s_addc_u32 s75, s75, s76                           // Free1
s_sub_u32 s73, s[sgprSizesFree+2], 1               // Free2
s_mul_i32 s73, s73, s[sgprGSUSumIdx]               // Free2
s_mul_hi_u32 s76, s73, s[sgprStrideCK]             // Free2
s_mul_i32 s73, s73, s[sgprStrideCK]                // Free2
s_add_u32 s74, s74, s73                            // Free2
s_addc_u32 s75, s75, s76                           // Free2
s_lshl_b64 s[74:75], s[74:75], 2                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s74        // add lo GSU offset to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s75       // add hi GSU offset to SRD
label_GSU_2:
.set sgprGSULog2BpeC, UNDEF

/* initC: remove ValuC vgpr buffer [0...0) from pool */

/* initC: remove acc vgpr buffer [0...8) from pool */

/* initC: remove ValuA/B vgpr buffer [0...48) from pool */
v_accvgpr_write acc0, 0x0                          // initC
v_accvgpr_write acc1, 0x0                          // initC
v_accvgpr_write acc2, 0x0                          // initC
v_accvgpr_write acc3, 0x0                          // initC
v_accvgpr_write acc4, 0x0                          // initC
v_accvgpr_write acc5, 0x0                          // initC
v_accvgpr_write acc6, 0x0                          // initC
v_accvgpr_write acc7, 0x0                          // initC
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?

/* after InitC, skip to end of prefetch last iter if numIter==0 */
s_cbranch_scc0 label_NoBranch_LMI4T0VTC74OFN9W_0   // Only branch on scc1
s_getpc_b64 s[74:75]                               // addr of next instr
s_add_i32 s76, label_PrefetchGlobalLastIterEnd, 0x4 // target branch offset
s_add_u32 s74, s74, s76                            // add target branch offset
s_addc_u32 s75, s75, 0                             // add high and carry
s_setpc_b64 s[74:75]                               // branch to label_PrefetchGlobalLastIterEnd
label_NoBranch_LMI4T0VTC74OFN9W_0:


/* 2nd set of PGR */
s_cmp_eq_u32 s[sgprLoopCounterL], 0x1              // PGR=3 but only 1 loop
s_cbranch_scc1 label_PGR1_vmcnt0                    // PGR=3 but only 1 loop

s_waitcnt vmcnt(9)                                 // 8wait for global read
s_branch label_1st_local_write

label_PGR1_vmcnt0:
s_waitcnt vmcnt(0)                                 // 8wait for global read

label_1st_local_write:


/* local write a */
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+4:vgprG2LA+4+3] offset:4352 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 4352
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+8:vgprG2LA+8+3] offset:8704 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 8704
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+12:vgprG2LA+12+3] offset:13056 // lwoA_0_0_3_0 = (0*LSCA)*(MT0I+PAD) + (3*LSPA) = 13056
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+16:vgprG2LA+16+3] offset:17408 // lwoA_0_0_4_0 = (0*LSCA)*(MT0I+PAD) + (4*LSPA) = 17408
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+20:vgprG2LA+20+3] offset:21760 // lwoA_0_0_5_0 = (0*LSCA)*(MT0I+PAD) + (5*LSPA) = 21760
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+24:vgprG2LA+24+3] offset:26112 // lwoA_0_0_6_0 = (0*LSCA)*(MT0I+PAD) + (6*LSPA) = 26112
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+28:vgprG2LA+28+3] offset:30464 // lwoA_0_0_7_0 = (0*LSCA)*(MT0I+PAD) + (7*LSPA) = 30464

/* local write b */
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0

/* local write swap a */

/* local write swap b */
s_cmp_le_u32 s[sgprLoopCounterL], 0x2              // PGR=3 but only 2 loop
s_cbranch_scc1 label_skipPGR2_0                    // PGR=3 but only 2 loop

buffer_load_dwordx4 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 nt // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprG2LA+4:vgprG2LA+4+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:0 nt // G -> Reg 0_0_1_0
buffer_load_dwordx4 v[vgprG2LA+8:vgprG2LA+8+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:0 nt // G -> Reg 0_0_2_0
buffer_load_dwordx4 v[vgprG2LA+12:vgprG2LA+12+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:0 nt // G -> Reg 0_0_3_0
buffer_load_dwordx4 v[vgprG2LA+16:vgprG2LA+16+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+3] offen offset:0 nt // G -> Reg 0_0_4_0
buffer_load_dwordx4 v[vgprG2LA+20:vgprG2LA+20+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+4] offen offset:0 nt // G -> Reg 0_0_5_0
buffer_load_dwordx4 v[vgprG2LA+24:vgprG2LA+24+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+5] offen offset:0 nt // G -> Reg 0_0_6_0
buffer_load_dwordx4 v[vgprG2LA+28:vgprG2LA+28+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+6] offen offset:0 nt // G -> Reg 0_0_7_0
buffer_load_dwordx4 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0


label_skipPGR2_0:

s_waitcnt lgkmcnt(0)                               // 0prefetch wait for local write
// Skip force waitcnt0
s_barrier

/* local read prefetch a */
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read prefetch b */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read inc a */
/* N/A, lro->32 */
/* self.localReadDoCntA 1 self.localReadDoCntB 1 */

/* local read inc b */
/* N/A, lro->32 */
/* self.localReadDoCntA 1 self.localReadDoCntB 1 */

/* local read prefetch a */
ds_read_b128 v[vgprValuA_X2_I0+0:vgprValuA_X2_I0+0+3], v[vgprLocalReadAddrA] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
ds_read_b128 v[vgprValuA_X2_I0+4:vgprValuA_X2_I0+4+3], v[vgprLocalReadAddrA] offset:320 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=2 iui=0

/* local read prefetch b */
ds_read_b128 v[vgprValuB_X2_I0+0:vgprValuB_X2_I0+0+3], v[vgprLocalReadAddrB] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0

/* local read inc a */
/* N/A, lro->64 */
/* self.localReadDoCntA 2 self.localReadDoCntB 2 */

/* local read inc b */
/* N/A, lro->64 */
/* self.localReadDoCntA 2 self.localReadDoCntB 2 */

/******************************************/
/* Unrolled Loop(s) - Begin               */
/******************************************/
label_openLoopL:
s_cmp_eq_u32 s[sgprLoopCounterL], 0x1              // LoopCounterL < EndCounter
s_cbranch_scc1 label_last_NLL                      // PGR=3 but only 1 loop iterations

s_cmp_eq_u32 s[sgprLoopCounterL], 0x2              // LoopCounterL < EndCounter
s_cbranch_scc1 label_oddexit_2ndNLL                // PGR=3 but only 2 loop iterations

/* jump to NLL if only 3 iters */
s_cmp_le_u32 s[sgprLoopCounterL], 0x3              // LoopCounterL < EndCounter
s_cbranch_scc1 label_LoopEndL_evenexit             // do not enter LoopL
label_LoopBeginL:

/******************************************/
/* Unrolled Loop 1/2 - Begin              */
/******************************************/

/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */

/* iter 0 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(3)                               // wait for prior local read local write old=0, new=3 newLW=0 newLR=3
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:1  */
ds_read_b128 v[vgprValuA_X4_I0+0:vgprValuA_X4_I0+0+3], v[vgprLocalReadAddrA] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
ds_read_b128 v[vgprValuB_X4_I0+0:vgprValuB_X4_I0+0+3], v[vgprLocalReadAddrB] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
ds_read_b128 v[vgprValuA_X4_I0+4:vgprValuA_X4_I0+4+3], v[vgprLocalReadAddrA] offset:384 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=4 iui=0

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s74, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s75, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s74        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s75       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s74 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s75 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-2 numReadsIterA=1 skipReadsIterA=2 readsPerIterA=2 */
/* dataAtIterB=-2 numReadsIterB=1 skipReadsIterB=2 readsPerIterB=1 */

/* iter 1 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:2  */
ds_read_b128 v[vgprValuA_X6_I0+0:vgprValuA_X6_I0+0+3], v[vgprLocalReadAddrA] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
ds_read_b128 v[vgprValuB_X6_I0+0:vgprValuB_X6_I0+0+3], v[vgprLocalReadAddrB] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
s_cselect_b32 s74, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=8
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X0_I0+0+2+0:vgprValuA_X0_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:3  */
ds_read_b128 v[vgprValuA_X6_I0+4:vgprValuA_X6_I0+4+3], v[vgprLocalReadAddrA] offset:448 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=6 iui=0
s_cselect_b32 s75, s[sgprWrapUB+1], 0              // incUpper <- ?
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X0_I0+4+2+0:vgprValuA_X0_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-2 numReadsIterA=2 skipReadsIterA=3 readsPerIterA=2 */
/* dataAtIterB=-2 numReadsIterB=2 skipReadsIterB=3 readsPerIterB=1 */

/* iter 2 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:4  */
/* localReadsVacancy: latencyLeft 2 */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s74        // gra SRD += inc(lower)
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:5  */
/* localReadsVacancy: latencyLeft 2 */
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s75       // gra SRD += inc(upper)
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+4+0+0:vgprValuA_X2_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=2 skipReadsIterA=2 readsPerIterA=2 */
/* dataAtIterB=-1 numReadsIterB=2 skipReadsIterB=2 readsPerIterB=1 */

/* iter 3 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:6  */
/* localReadsVacancy: latencyLeft 2 */
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s74 // limit -= inc)
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X2_I0+0+2+0:vgprValuA_X2_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:7  */
/* localReadsVacancy: latencyLeft 2 */
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s75 // limit -= inc)
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X2_I0+4+2+0:vgprValuA_X2_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=2 skipReadsIterA=2 readsPerIterA=2 */
/* dataAtIterB=-1 numReadsIterB=2 skipReadsIterB=2 readsPerIterB=1 */

/* iter 4 (swap and reset local write pointers iteration)  */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:8  */
/* schedule remaining localreads for 1LDSB */
/* localReadsVacancy: latencyLeft 2 */
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X4_I0+0+0+0:vgprValuB_X4_I0+0+0+0+1], v[vgprValuA_X4_I0+0+0+0:vgprValuA_X4_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:9  */
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(17)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA1+0:vgprG2LA1+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0
buffer_load_dwordx4 v[vgprG2LA1+0:vgprG2LA1+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 nt // G -> Reg 0_0_0_0
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(17)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA1+4:vgprG2LA1+4+3] offset:4352 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 4352
buffer_load_dwordx4 v[vgprG2LA1+4:vgprG2LA1+4+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:0 nt // G -> Reg 0_0_1_0
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(17)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA1+8:vgprG2LA1+8+3] offset:8704 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 8704
buffer_load_dwordx4 v[vgprG2LA1+8:vgprG2LA1+8+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:0 nt // G -> Reg 0_0_2_0
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(17)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA1+12:vgprG2LA1+12+3] offset:13056 // lwoA_0_0_3_0 = (0*LSCA)*(MT0I+PAD) + (3*LSPA) = 13056
buffer_load_dwordx4 v[vgprG2LA1+12:vgprG2LA1+12+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:0 nt // G -> Reg 0_0_3_0
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(17)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA1+16:vgprG2LA1+16+3] offset:17408 // lwoA_0_0_4_0 = (0*LSCA)*(MT0I+PAD) + (4*LSPA) = 17408
buffer_load_dwordx4 v[vgprG2LA1+16:vgprG2LA1+16+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+3] offen offset:0 nt // G -> Reg 0_0_4_0
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(17)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA1+20:vgprG2LA1+20+3] offset:21760 // lwoA_0_0_5_0 = (0*LSCA)*(MT0I+PAD) + (5*LSPA) = 21760
buffer_load_dwordx4 v[vgprG2LA1+20:vgprG2LA1+20+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+4] offen offset:0 nt // G -> Reg 0_0_5_0
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(17)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA1+24:vgprG2LA1+24+3] offset:26112 // lwoA_0_0_6_0 = (0*LSCA)*(MT0I+PAD) + (6*LSPA) = 26112
buffer_load_dwordx4 v[vgprG2LA1+24:vgprG2LA1+24+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+5] offen offset:0 nt // G -> Reg 0_0_6_0
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(17)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA1+28:vgprG2LA1+28+3] offset:30464 // lwoA_0_0_7_0 = (0*LSCA)*(MT0I+PAD) + (7*LSPA) = 30464
buffer_load_dwordx4 v[vgprG2LA1+28:vgprG2LA1+28+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+6] offen offset:0 nt // G -> Reg 0_0_7_0
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(17)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB1+0:vgprG2LB1+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
buffer_load_dwordx4 v[vgprG2LB1+0:vgprG2LB1+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0

/* local write swap offsets a */

/* local write swap offsets b */
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X4_I0+0+0+0:vgprValuB_X4_I0+0+0+0+1], v[vgprValuA_X4_I0+4+0+0:vgprValuA_X4_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=2 */
/* dataAtIterB=0 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=1 */

/* iter 5 (reset local read pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:10  */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=9 newLR=3
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X4_I0+0+2+0:vgprValuB_X4_I0+0+2+0+1], v[vgprValuA_X4_I0+0+2+0:vgprValuA_X4_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:11  */

/* local read swap offsets a */

/* local read swap offsets b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X4_I0+0+2+0:vgprValuB_X4_I0+0+2+0+1], v[vgprValuA_X4_I0+4+2+0:vgprValuA_X4_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=2 */
/* dataAtIterB=0 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=1 */

/* iter 6 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:12  */
s_waitcnt lgkmcnt(0)                               // 3wait for local write
// Skip force waitcnt0
s_barrier
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X6_I0+0+0+0:vgprValuB_X6_I0+0+0+0+1], v[vgprValuA_X6_I0+0+0+0:vgprValuA_X6_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:13  */
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X6_I0+0+0+0:vgprValuB_X6_I0+0+0+0+1], v[vgprValuA_X6_I0+4+0+0:vgprValuA_X6_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=1 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=2 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=1 */

/* iter 7 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:14  */
ds_read_b128 v[vgprValuA_X2_I0+0:vgprValuA_X2_I0+0+3], v[vgprLocalReadAddrA] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
ds_read_b128 v[vgprValuB_X2_I0+0:vgprValuB_X2_I0+0+3], v[vgprLocalReadAddrB] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
s_waitcnt lgkmcnt(14)                              // wait for prior local read local write old=0, new=14 newLW=9 newLR=5
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X6_I0+0+2+0:vgprValuB_X6_I0+0+2+0+1], v[vgprValuA_X6_I0+0+2+0:vgprValuA_X6_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:15  */
ds_read_b128 v[vgprValuA_X2_I0+4:vgprValuA_X2_I0+4+3], v[vgprLocalReadAddrA] offset:320 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X6_I0+0+2+0:vgprValuB_X6_I0+0+2+0+1], v[vgprValuA_X6_I0+4+2+0:vgprValuA_X6_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=2 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=2 readsPerIterA=2 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=2 readsPerIterB=1 */

/******************************************/
/* Unrolled Loop - End 1/2                */
/******************************************/

/* closeLoop loopL finalLoop=0 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x3              // counterL==2
s_cbranch_scc1 label_LoopEndL_oddexit              // exit LoopL

/******************************************/
/* Unrolled Loop 2/2 - Begin              */
/******************************************/

/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */

/* iter 0 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(3)                               // wait for prior local read local write old=0, new=3 newLW=0 newLR=3
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:1  */
ds_read_b128 v[vgprValuA_X4_I0+0:vgprValuA_X4_I0+0+3], v[vgprLocalReadAddrA] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
ds_read_b128 v[vgprValuB_X4_I0+0:vgprValuB_X4_I0+0+3], v[vgprLocalReadAddrB] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
ds_read_b128 v[vgprValuA_X4_I0+4:vgprValuA_X4_I0+4+3], v[vgprLocalReadAddrA] offset:384 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=4 iui=0

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s74, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s75, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s74        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s75       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s74 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s75 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-2 numReadsIterA=1 skipReadsIterA=2 readsPerIterA=2 */
/* dataAtIterB=-2 numReadsIterB=1 skipReadsIterB=2 readsPerIterB=1 */

/* iter 1 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:2  */
ds_read_b128 v[vgprValuA_X6_I0+0:vgprValuA_X6_I0+0+3], v[vgprLocalReadAddrA] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
ds_read_b128 v[vgprValuB_X6_I0+0:vgprValuB_X6_I0+0+3], v[vgprLocalReadAddrB] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
s_cselect_b32 s74, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=8
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X0_I0+0+2+0:vgprValuA_X0_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:3  */
ds_read_b128 v[vgprValuA_X6_I0+4:vgprValuA_X6_I0+4+3], v[vgprLocalReadAddrA] offset:448 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=6 iui=0
s_cselect_b32 s75, s[sgprWrapUB+1], 0              // incUpper <- ?
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X0_I0+4+2+0:vgprValuA_X0_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-2 numReadsIterA=2 skipReadsIterA=3 readsPerIterA=2 */
/* dataAtIterB=-2 numReadsIterB=2 skipReadsIterB=3 readsPerIterB=1 */

/* iter 2 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:4  */
/* localReadsVacancy: latencyLeft 2 */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s74        // gra SRD += inc(lower)
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:5  */
/* localReadsVacancy: latencyLeft 2 */
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s75       // gra SRD += inc(upper)
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+4+0+0:vgprValuA_X2_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=2 skipReadsIterA=2 readsPerIterA=2 */
/* dataAtIterB=-1 numReadsIterB=2 skipReadsIterB=2 readsPerIterB=1 */

/* iter 3 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:6  */
/* localReadsVacancy: latencyLeft 2 */
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s74 // limit -= inc)
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X2_I0+0+2+0:vgprValuA_X2_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:7  */
/* localReadsVacancy: latencyLeft 2 */
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s75 // limit -= inc)
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X2_I0+4+2+0:vgprValuA_X2_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=2 skipReadsIterA=2 readsPerIterA=2 */
/* dataAtIterB=-1 numReadsIterB=2 skipReadsIterB=2 readsPerIterB=1 */

/* iter 4 (swap and reset local write pointers iteration)  */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:8  */
/* schedule remaining localreads for 1LDSB */
/* localReadsVacancy: latencyLeft 2 */
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X4_I0+0+0+0:vgprValuB_X4_I0+0+0+0+1], v[vgprValuA_X4_I0+0+0+0:vgprValuA_X4_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:9  */
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(17)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0
buffer_load_dwordx4 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 nt // G -> Reg 0_0_0_0
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(17)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+4:vgprG2LA+4+3] offset:4352 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 4352
buffer_load_dwordx4 v[vgprG2LA+4:vgprG2LA+4+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:0 nt // G -> Reg 0_0_1_0
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(17)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+8:vgprG2LA+8+3] offset:8704 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 8704
buffer_load_dwordx4 v[vgprG2LA+8:vgprG2LA+8+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:0 nt // G -> Reg 0_0_2_0
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(17)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+12:vgprG2LA+12+3] offset:13056 // lwoA_0_0_3_0 = (0*LSCA)*(MT0I+PAD) + (3*LSPA) = 13056
buffer_load_dwordx4 v[vgprG2LA+12:vgprG2LA+12+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:0 nt // G -> Reg 0_0_3_0
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(17)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+16:vgprG2LA+16+3] offset:17408 // lwoA_0_0_4_0 = (0*LSCA)*(MT0I+PAD) + (4*LSPA) = 17408
buffer_load_dwordx4 v[vgprG2LA+16:vgprG2LA+16+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+3] offen offset:0 nt // G -> Reg 0_0_4_0
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(17)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+20:vgprG2LA+20+3] offset:21760 // lwoA_0_0_5_0 = (0*LSCA)*(MT0I+PAD) + (5*LSPA) = 21760
buffer_load_dwordx4 v[vgprG2LA+20:vgprG2LA+20+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+4] offen offset:0 nt // G -> Reg 0_0_5_0
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(17)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+24:vgprG2LA+24+3] offset:26112 // lwoA_0_0_6_0 = (0*LSCA)*(MT0I+PAD) + (6*LSPA) = 26112
buffer_load_dwordx4 v[vgprG2LA+24:vgprG2LA+24+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+5] offen offset:0 nt // G -> Reg 0_0_6_0
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(17)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+28:vgprG2LA+28+3] offset:30464 // lwoA_0_0_7_0 = (0*LSCA)*(MT0I+PAD) + (7*LSPA) = 30464
buffer_load_dwordx4 v[vgprG2LA+28:vgprG2LA+28+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+6] offen offset:0 nt // G -> Reg 0_0_7_0
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(17)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
buffer_load_dwordx4 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0

/* local write swap offsets a */

/* local write swap offsets b */
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X4_I0+0+0+0:vgprValuB_X4_I0+0+0+0+1], v[vgprValuA_X4_I0+4+0+0:vgprValuA_X4_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=2 */
/* dataAtIterB=0 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=1 */

/* iter 5 (reset local read pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:10  */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=9 newLR=3
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X4_I0+0+2+0:vgprValuB_X4_I0+0+2+0+1], v[vgprValuA_X4_I0+0+2+0:vgprValuA_X4_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:11  */

/* local read swap offsets a */

/* local read swap offsets b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X4_I0+0+2+0:vgprValuB_X4_I0+0+2+0+1], v[vgprValuA_X4_I0+4+2+0:vgprValuA_X4_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=2 */
/* dataAtIterB=0 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=1 */

/* iter 6 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:12  */
s_waitcnt lgkmcnt(0)                               // 3wait for local write
// Skip force waitcnt0
s_barrier
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X6_I0+0+0+0:vgprValuB_X6_I0+0+0+0+1], v[vgprValuA_X6_I0+0+0+0:vgprValuA_X6_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:13  */
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X6_I0+0+0+0:vgprValuB_X6_I0+0+0+0+1], v[vgprValuA_X6_I0+4+0+0:vgprValuA_X6_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=1 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=2 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=1 */

/* iter 7 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:14  */
ds_read_b128 v[vgprValuA_X2_I0+0:vgprValuA_X2_I0+0+3], v[vgprLocalReadAddrA] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
ds_read_b128 v[vgprValuB_X2_I0+0:vgprValuB_X2_I0+0+3], v[vgprLocalReadAddrB] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
s_waitcnt lgkmcnt(14)                              // wait for prior local read local write old=0, new=14 newLW=9 newLR=5
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X6_I0+0+2+0:vgprValuB_X6_I0+0+2+0+1], v[vgprValuA_X6_I0+0+2+0:vgprValuA_X6_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:15  */
ds_read_b128 v[vgprValuA_X2_I0+4:vgprValuA_X2_I0+4+3], v[vgprLocalReadAddrA] offset:320 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X6_I0+0+2+0:vgprValuB_X6_I0+0+2+0+1], v[vgprValuA_X6_I0+4+2+0:vgprValuA_X6_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=2 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=2 readsPerIterA=2 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=2 readsPerIterB=1 */

/******************************************/
/* Unrolled Loop - End 2/2 (final)        */
/******************************************/

/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x3              // counterL==2
s_cbranch_scc0 label_LoopBeginL                    // restart LoopL


/* EVEN EXIT */

label_LoopEndL_evenexit:  /// unroll loop eveniter exit


/******************************************/
/* Ord. NoGlobalLoadLoop - 1/2           */
/******************************************/

/* iter 0 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(3)                               // wait for prior local read local write old=0, new=3 newLW=0 newLR=3
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:1  */
ds_read_b128 v[vgprValuA_X4_I0+0:vgprValuA_X4_I0+0+3], v[vgprLocalReadAddrA] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
ds_read_b128 v[vgprValuB_X4_I0+0:vgprValuB_X4_I0+0+3], v[vgprLocalReadAddrB] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
ds_read_b128 v[vgprValuA_X4_I0+4:vgprValuA_X4_I0+4+3], v[vgprLocalReadAddrA] offset:384 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=4 iui=0

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s74, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s75, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s74        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s75       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s74 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s75 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-2 numReadsIterA=1 skipReadsIterA=2 readsPerIterA=2 */
/* dataAtIterB=-2 numReadsIterB=1 skipReadsIterB=2 readsPerIterB=1 */

/* iter 1 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:2  */
ds_read_b128 v[vgprValuA_X6_I0+0:vgprValuA_X6_I0+0+3], v[vgprLocalReadAddrA] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
ds_read_b128 v[vgprValuB_X6_I0+0:vgprValuB_X6_I0+0+3], v[vgprLocalReadAddrB] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
s_cselect_b32 s74, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=8
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X0_I0+0+2+0:vgprValuA_X0_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:3  */
ds_read_b128 v[vgprValuA_X6_I0+4:vgprValuA_X6_I0+4+3], v[vgprLocalReadAddrA] offset:448 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=6 iui=0
s_cselect_b32 s75, s[sgprWrapUB+1], 0              // incUpper <- ?
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X0_I0+4+2+0:vgprValuA_X0_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-2 numReadsIterA=2 skipReadsIterA=3 readsPerIterA=2 */
/* dataAtIterB=-2 numReadsIterB=2 skipReadsIterB=3 readsPerIterB=1 */

/* iter 2 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:4  */
/* localReadsVacancy: latencyLeft 2 */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s74        // gra SRD += inc(lower)
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:5  */
/* localReadsVacancy: latencyLeft 2 */
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s75       // gra SRD += inc(upper)
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+4+0+0:vgprValuA_X2_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=2 skipReadsIterA=2 readsPerIterA=2 */
/* dataAtIterB=-1 numReadsIterB=2 skipReadsIterB=2 readsPerIterB=1 */

/* iter 3 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:6  */
/* localReadsVacancy: latencyLeft 2 */
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s74 // limit -= inc)
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X2_I0+0+2+0:vgprValuA_X2_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:7  */
/* localReadsVacancy: latencyLeft 2 */
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s75 // limit -= inc)
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X2_I0+4+2+0:vgprValuA_X2_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=2 skipReadsIterA=2 readsPerIterA=2 */
/* dataAtIterB=-1 numReadsIterB=2 skipReadsIterB=2 readsPerIterB=1 */

/* iter 4 (swap and reset local write pointers iteration)  */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:8  */
/* schedule remaining localreads for 1LDSB */
/* localReadsVacancy: latencyLeft 2 */
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X4_I0+0+0+0:vgprValuB_X4_I0+0+0+0+1], v[vgprValuA_X4_I0+0+0+0:vgprValuA_X4_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:9  */
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(17)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA1+0:vgprG2LA1+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(16)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA1+4:vgprG2LA1+4+3] offset:4352 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 4352
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(15)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA1+8:vgprG2LA1+8+3] offset:8704 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 8704
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(14)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA1+12:vgprG2LA1+12+3] offset:13056 // lwoA_0_0_3_0 = (0*LSCA)*(MT0I+PAD) + (3*LSPA) = 13056
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(13)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA1+16:vgprG2LA1+16+3] offset:17408 // lwoA_0_0_4_0 = (0*LSCA)*(MT0I+PAD) + (4*LSPA) = 17408
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(12)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA1+20:vgprG2LA1+20+3] offset:21760 // lwoA_0_0_5_0 = (0*LSCA)*(MT0I+PAD) + (5*LSPA) = 21760
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(11)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA1+24:vgprG2LA1+24+3] offset:26112 // lwoA_0_0_6_0 = (0*LSCA)*(MT0I+PAD) + (6*LSPA) = 26112
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(10)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA1+28:vgprG2LA1+28+3] offset:30464 // lwoA_0_0_7_0 = (0*LSCA)*(MT0I+PAD) + (7*LSPA) = 30464
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(9)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB1+0:vgprG2LB1+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0

/* local write swap offsets a */

/* local write swap offsets b */
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X4_I0+0+0+0:vgprValuB_X4_I0+0+0+0+1], v[vgprValuA_X4_I0+4+0+0:vgprValuA_X4_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=2 */
/* dataAtIterB=0 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=1 */

/* iter 5 (reset local read pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:10  */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=9 newLR=3
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X4_I0+0+2+0:vgprValuB_X4_I0+0+2+0+1], v[vgprValuA_X4_I0+0+2+0:vgprValuA_X4_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:11  */

/* local read swap offsets a */

/* local read swap offsets b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X4_I0+0+2+0:vgprValuB_X4_I0+0+2+0+1], v[vgprValuA_X4_I0+4+2+0:vgprValuA_X4_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=2 */
/* dataAtIterB=0 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=1 */

/* iter 6 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:12  */
s_waitcnt lgkmcnt(0)                               // 3wait for local write
// Skip force waitcnt0
s_barrier
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X6_I0+0+0+0:vgprValuB_X6_I0+0+0+0+1], v[vgprValuA_X6_I0+0+0+0:vgprValuA_X6_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:13  */
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X6_I0+0+0+0:vgprValuB_X6_I0+0+0+0+1], v[vgprValuA_X6_I0+4+0+0:vgprValuA_X6_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=1 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=2 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=1 */

/* iter 7 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:14  */
ds_read_b128 v[vgprValuA_X2_I0+0:vgprValuA_X2_I0+0+3], v[vgprLocalReadAddrA] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
ds_read_b128 v[vgprValuB_X2_I0+0:vgprValuB_X2_I0+0+3], v[vgprLocalReadAddrB] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
s_waitcnt lgkmcnt(14)                              // wait for prior local read local write old=0, new=14 newLW=9 newLR=5
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X6_I0+0+2+0:vgprValuB_X6_I0+0+2+0+1], v[vgprValuA_X6_I0+0+2+0:vgprValuA_X6_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:15  */
ds_read_b128 v[vgprValuA_X2_I0+4:vgprValuA_X2_I0+4+3], v[vgprLocalReadAddrA] offset:320 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X6_I0+0+2+0:vgprValuB_X6_I0+0+2+0+1], v[vgprValuA_X6_I0+4+2+0:vgprValuA_X6_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=2 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=2 readsPerIterA=2 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=2 readsPerIterB=1 */



/* 2nd NLL in even exit : LW VGPR */
/* ====================================== */

label_evenexit_2ndNLL:
/* iter 0 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(3)                               // wait for prior local read local write old=0, new=3 newLW=0 newLR=3
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:1  */
ds_read_b128 v[vgprValuA_X4_I0+0:vgprValuA_X4_I0+0+3], v[vgprLocalReadAddrA] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
ds_read_b128 v[vgprValuB_X4_I0+0:vgprValuB_X4_I0+0+3], v[vgprLocalReadAddrB] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
ds_read_b128 v[vgprValuA_X4_I0+4:vgprValuA_X4_I0+4+3], v[vgprLocalReadAddrA] offset:384 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=4 iui=0

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s74, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s75, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s74        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s75       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s74 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s75 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-2 numReadsIterA=1 skipReadsIterA=2 readsPerIterA=2 */
/* dataAtIterB=-2 numReadsIterB=1 skipReadsIterB=2 readsPerIterB=1 */

/* iter 1 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:2  */
ds_read_b128 v[vgprValuA_X6_I0+0:vgprValuA_X6_I0+0+3], v[vgprLocalReadAddrA] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
ds_read_b128 v[vgprValuB_X6_I0+0:vgprValuB_X6_I0+0+3], v[vgprLocalReadAddrB] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
s_cselect_b32 s74, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=8
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X0_I0+0+2+0:vgprValuA_X0_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:3  */
ds_read_b128 v[vgprValuA_X6_I0+4:vgprValuA_X6_I0+4+3], v[vgprLocalReadAddrA] offset:448 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=6 iui=0
s_cselect_b32 s75, s[sgprWrapUB+1], 0              // incUpper <- ?
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X0_I0+4+2+0:vgprValuA_X0_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-2 numReadsIterA=2 skipReadsIterA=3 readsPerIterA=2 */
/* dataAtIterB=-2 numReadsIterB=2 skipReadsIterB=3 readsPerIterB=1 */

/* iter 2 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:4  */
/* localReadsVacancy: latencyLeft 2 */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s74        // gra SRD += inc(lower)
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:5  */
/* localReadsVacancy: latencyLeft 2 */
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s75       // gra SRD += inc(upper)
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+4+0+0:vgprValuA_X2_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=2 skipReadsIterA=2 readsPerIterA=2 */
/* dataAtIterB=-1 numReadsIterB=2 skipReadsIterB=2 readsPerIterB=1 */

/* iter 3 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:6  */
/* localReadsVacancy: latencyLeft 2 */
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s74 // limit -= inc)
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X2_I0+0+2+0:vgprValuA_X2_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:7  */
/* localReadsVacancy: latencyLeft 2 */
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s75 // limit -= inc)
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X2_I0+4+2+0:vgprValuA_X2_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=2 skipReadsIterA=2 readsPerIterA=2 */
/* dataAtIterB=-1 numReadsIterB=2 skipReadsIterB=2 readsPerIterB=1 */

/* iter 4 (swap and reset local write pointers iteration)  */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:8  */
/* schedule remaining localreads for 1LDSB */
/* localReadsVacancy: latencyLeft 2 */
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X4_I0+0+0+0:vgprValuB_X4_I0+0+0+0+1], v[vgprValuA_X4_I0+0+0+0:vgprValuA_X4_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:9  */
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(8)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+4:vgprG2LA+4+3] offset:4352 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 4352
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(6)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+8:vgprG2LA+8+3] offset:8704 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 8704
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(5)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+12:vgprG2LA+12+3] offset:13056 // lwoA_0_0_3_0 = (0*LSCA)*(MT0I+PAD) + (3*LSPA) = 13056
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(4)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+16:vgprG2LA+16+3] offset:17408 // lwoA_0_0_4_0 = (0*LSCA)*(MT0I+PAD) + (4*LSPA) = 17408
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(3)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+20:vgprG2LA+20+3] offset:21760 // lwoA_0_0_5_0 = (0*LSCA)*(MT0I+PAD) + (5*LSPA) = 21760
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(2)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+24:vgprG2LA+24+3] offset:26112 // lwoA_0_0_6_0 = (0*LSCA)*(MT0I+PAD) + (6*LSPA) = 26112
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(1)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+28:vgprG2LA+28+3] offset:30464 // lwoA_0_0_7_0 = (0*LSCA)*(MT0I+PAD) + (7*LSPA) = 30464
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(0)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0

/* local write swap offsets a */

/* local write swap offsets b */
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X4_I0+0+0+0:vgprValuB_X4_I0+0+0+0+1], v[vgprValuA_X4_I0+4+0+0:vgprValuA_X4_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=2 */
/* dataAtIterB=0 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=1 */

/* iter 5 (reset local read pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:10  */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=9 newLR=3
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X4_I0+0+2+0:vgprValuB_X4_I0+0+2+0+1], v[vgprValuA_X4_I0+0+2+0:vgprValuA_X4_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:11  */

/* local read swap offsets a */

/* local read swap offsets b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X4_I0+0+2+0:vgprValuB_X4_I0+0+2+0+1], v[vgprValuA_X4_I0+4+2+0:vgprValuA_X4_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=2 */
/* dataAtIterB=0 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=1 */

/* iter 6 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:12  */
s_waitcnt lgkmcnt(0)                               // 3wait for local write
// Skip force waitcnt0
s_barrier
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X6_I0+0+0+0:vgprValuB_X6_I0+0+0+0+1], v[vgprValuA_X6_I0+0+0+0:vgprValuA_X6_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:13  */
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X6_I0+0+0+0:vgprValuB_X6_I0+0+0+0+1], v[vgprValuA_X6_I0+4+0+0:vgprValuA_X6_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=1 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=2 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=1 */

/* iter 7 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:14  */
ds_read_b128 v[vgprValuA_X2_I0+0:vgprValuA_X2_I0+0+3], v[vgprLocalReadAddrA] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
ds_read_b128 v[vgprValuB_X2_I0+0:vgprValuB_X2_I0+0+3], v[vgprLocalReadAddrB] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
s_waitcnt lgkmcnt(14)                              // wait for prior local read local write old=0, new=14 newLW=9 newLR=5
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X6_I0+0+2+0:vgprValuB_X6_I0+0+2+0+1], v[vgprValuA_X6_I0+0+2+0:vgprValuA_X6_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:15  */
ds_read_b128 v[vgprValuA_X2_I0+4:vgprValuA_X2_I0+4+3], v[vgprLocalReadAddrA] offset:320 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X6_I0+0+2+0:vgprValuB_X6_I0+0+2+0+1], v[vgprValuA_X6_I0+4+2+0:vgprValuA_X6_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=2 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=2 readsPerIterA=2 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=2 readsPerIterB=1 */


/* Last NLL is common for both odd and even exit */

s_branch label_last_NLL



label_LoopEndL_oddexit:  /// unroll loop odditer exit

/* Select high bank of LDS */
label_LoopEndL:

/* Before NLL: Check VGPR.checkin for INT8 LW */

/******************************************/
/* Ord. NoGlobalLoadLoop - Begin          */
/******************************************/

/* iter 0 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(3)                               // wait for prior local read local write old=0, new=3 newLW=0 newLR=3
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:1  */
ds_read_b128 v[vgprValuA_X4_I0+0:vgprValuA_X4_I0+0+3], v[vgprLocalReadAddrA] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
ds_read_b128 v[vgprValuB_X4_I0+0:vgprValuB_X4_I0+0+3], v[vgprLocalReadAddrB] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
ds_read_b128 v[vgprValuA_X4_I0+4:vgprValuA_X4_I0+4+3], v[vgprLocalReadAddrA] offset:384 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=4 iui=0

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s74, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s75, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s74        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s75       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s74 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s75 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-2 numReadsIterA=1 skipReadsIterA=2 readsPerIterA=2 */
/* dataAtIterB=-2 numReadsIterB=1 skipReadsIterB=2 readsPerIterB=1 */

/* iter 1 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:2  */
ds_read_b128 v[vgprValuA_X6_I0+0:vgprValuA_X6_I0+0+3], v[vgprLocalReadAddrA] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
ds_read_b128 v[vgprValuB_X6_I0+0:vgprValuB_X6_I0+0+3], v[vgprLocalReadAddrB] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
s_cselect_b32 s74, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=8
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X0_I0+0+2+0:vgprValuA_X0_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:3  */
ds_read_b128 v[vgprValuA_X6_I0+4:vgprValuA_X6_I0+4+3], v[vgprLocalReadAddrA] offset:448 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=6 iui=0
s_cselect_b32 s75, s[sgprWrapUB+1], 0              // incUpper <- ?
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X0_I0+4+2+0:vgprValuA_X0_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-2 numReadsIterA=2 skipReadsIterA=3 readsPerIterA=2 */
/* dataAtIterB=-2 numReadsIterB=2 skipReadsIterB=3 readsPerIterB=1 */

/* iter 2 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:4  */
/* localReadsVacancy: latencyLeft 2 */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s74        // gra SRD += inc(lower)
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:5  */
/* localReadsVacancy: latencyLeft 2 */
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s75       // gra SRD += inc(upper)
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+4+0+0:vgprValuA_X2_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=2 skipReadsIterA=2 readsPerIterA=2 */
/* dataAtIterB=-1 numReadsIterB=2 skipReadsIterB=2 readsPerIterB=1 */

/* iter 3 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:6  */
/* localReadsVacancy: latencyLeft 2 */
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s74 // limit -= inc)
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X2_I0+0+2+0:vgprValuA_X2_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:7  */
/* localReadsVacancy: latencyLeft 2 */
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s75 // limit -= inc)
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X2_I0+4+2+0:vgprValuA_X2_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=2 skipReadsIterA=2 readsPerIterA=2 */
/* dataAtIterB=-1 numReadsIterB=2 skipReadsIterB=2 readsPerIterB=1 */

/* iter 4 (swap and reset local write pointers iteration)  */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:8  */
/* schedule remaining localreads for 1LDSB */
/* localReadsVacancy: latencyLeft 2 */
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X4_I0+0+0+0:vgprValuB_X4_I0+0+0+0+1], v[vgprValuA_X4_I0+0+0+0:vgprValuA_X4_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:9  */
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(17)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(16)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+4:vgprG2LA+4+3] offset:4352 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 4352
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(15)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+8:vgprG2LA+8+3] offset:8704 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 8704
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(14)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+12:vgprG2LA+12+3] offset:13056 // lwoA_0_0_3_0 = (0*LSCA)*(MT0I+PAD) + (3*LSPA) = 13056
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(13)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+16:vgprG2LA+16+3] offset:17408 // lwoA_0_0_4_0 = (0*LSCA)*(MT0I+PAD) + (4*LSPA) = 17408
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(12)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+20:vgprG2LA+20+3] offset:21760 // lwoA_0_0_5_0 = (0*LSCA)*(MT0I+PAD) + (5*LSPA) = 21760
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(11)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+24:vgprG2LA+24+3] offset:26112 // lwoA_0_0_6_0 = (0*LSCA)*(MT0I+PAD) + (6*LSPA) = 26112
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(10)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+28:vgprG2LA+28+3] offset:30464 // lwoA_0_0_7_0 = (0*LSCA)*(MT0I+PAD) + (7*LSPA) = 30464
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(9)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0

/* local write swap offsets a */

/* local write swap offsets b */
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X4_I0+0+0+0:vgprValuB_X4_I0+0+0+0+1], v[vgprValuA_X4_I0+4+0+0:vgprValuA_X4_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=2 */
/* dataAtIterB=0 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=1 */

/* iter 5 (reset local read pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:10  */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=9 newLR=3
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X4_I0+0+2+0:vgprValuB_X4_I0+0+2+0+1], v[vgprValuA_X4_I0+0+2+0:vgprValuA_X4_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:11  */

/* local read swap offsets a */

/* local read swap offsets b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X4_I0+0+2+0:vgprValuB_X4_I0+0+2+0+1], v[vgprValuA_X4_I0+4+2+0:vgprValuA_X4_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=2 */
/* dataAtIterB=0 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=1 */

/* iter 6 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:12  */
s_waitcnt lgkmcnt(0)                               // 3wait for local write
// Skip force waitcnt0
s_barrier
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X6_I0+0+0+0:vgprValuB_X6_I0+0+0+0+1], v[vgprValuA_X6_I0+0+0+0:vgprValuA_X6_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:13  */
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X6_I0+0+0+0:vgprValuB_X6_I0+0+0+0+1], v[vgprValuA_X6_I0+4+0+0:vgprValuA_X6_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=1 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=2 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=1 */

/* iter 7 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:14  */
ds_read_b128 v[vgprValuA_X2_I0+0:vgprValuA_X2_I0+0+3], v[vgprLocalReadAddrA] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
ds_read_b128 v[vgprValuB_X2_I0+0:vgprValuB_X2_I0+0+3], v[vgprLocalReadAddrB] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
s_waitcnt lgkmcnt(14)                              // wait for prior local read local write old=0, new=14 newLW=9 newLR=5
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X6_I0+0+2+0:vgprValuB_X6_I0+0+2+0+1], v[vgprValuA_X6_I0+0+2+0:vgprValuA_X6_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:15  */
ds_read_b128 v[vgprValuA_X2_I0+4:vgprValuA_X2_I0+4+3], v[vgprLocalReadAddrA] offset:320 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X6_I0+0+2+0:vgprValuB_X6_I0+0+2+0+1], v[vgprValuA_X6_I0+4+2+0:vgprValuA_X6_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=2 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=2 readsPerIterA=2 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=2 readsPerIterB=1 */


/* 2nd NLL */
/* ================ */

label_oddexit_2ndNLL:


/* iter 0 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(3)                               // wait for prior local read local write old=0, new=3 newLW=0 newLR=3
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:1  */
ds_read_b128 v[vgprValuA_X4_I0+0:vgprValuA_X4_I0+0+3], v[vgprLocalReadAddrA] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
ds_read_b128 v[vgprValuB_X4_I0+0:vgprValuB_X4_I0+0+3], v[vgprLocalReadAddrB] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
ds_read_b128 v[vgprValuA_X4_I0+4:vgprValuA_X4_I0+4+3], v[vgprLocalReadAddrA] offset:384 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=4 iui=0

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s74, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s75, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s74        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s75       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s74 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s75 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-2 numReadsIterA=1 skipReadsIterA=2 readsPerIterA=2 */
/* dataAtIterB=-2 numReadsIterB=1 skipReadsIterB=2 readsPerIterB=1 */

/* iter 1 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:2  */
ds_read_b128 v[vgprValuA_X6_I0+0:vgprValuA_X6_I0+0+3], v[vgprLocalReadAddrA] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
ds_read_b128 v[vgprValuB_X6_I0+0:vgprValuB_X6_I0+0+3], v[vgprLocalReadAddrB] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
s_cselect_b32 s74, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=8
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X0_I0+0+2+0:vgprValuA_X0_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:3  */
ds_read_b128 v[vgprValuA_X6_I0+4:vgprValuA_X6_I0+4+3], v[vgprLocalReadAddrA] offset:448 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=6 iui=0
s_cselect_b32 s75, s[sgprWrapUB+1], 0              // incUpper <- ?
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X0_I0+4+2+0:vgprValuA_X0_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-2 numReadsIterA=2 skipReadsIterA=3 readsPerIterA=2 */
/* dataAtIterB=-2 numReadsIterB=2 skipReadsIterB=3 readsPerIterB=1 */

/* iter 2 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:4  */
/* localReadsVacancy: latencyLeft 2 */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s74        // gra SRD += inc(lower)
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:5  */
/* localReadsVacancy: latencyLeft 2 */
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s75       // gra SRD += inc(upper)
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+4+0+0:vgprValuA_X2_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=2 skipReadsIterA=2 readsPerIterA=2 */
/* dataAtIterB=-1 numReadsIterB=2 skipReadsIterB=2 readsPerIterB=1 */

/* iter 3 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:6  */
/* localReadsVacancy: latencyLeft 2 */
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s74 // limit -= inc)
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X2_I0+0+2+0:vgprValuA_X2_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:7  */
/* localReadsVacancy: latencyLeft 2 */
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s75 // limit -= inc)
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X2_I0+4+2+0:vgprValuA_X2_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=2 skipReadsIterA=2 readsPerIterA=2 */
/* dataAtIterB=-1 numReadsIterB=2 skipReadsIterB=2 readsPerIterB=1 */

/* iter 4 (swap and reset local write pointers iteration)  */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:8  */
/* schedule remaining localreads for 1LDSB */
/* localReadsVacancy: latencyLeft 2 */
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X4_I0+0+0+0:vgprValuB_X4_I0+0+0+0+1], v[vgprValuA_X4_I0+0+0+0:vgprValuA_X4_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:9  */
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(8)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA1+0:vgprG2LA1+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA1+4:vgprG2LA1+4+3] offset:4352 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 4352
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(6)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA1+8:vgprG2LA1+8+3] offset:8704 // lwoA_0_0_2_0 = (0*LSCA)*(MT0I+PAD) + (2*LSPA) = 8704
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(5)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA1+12:vgprG2LA1+12+3] offset:13056 // lwoA_0_0_3_0 = (0*LSCA)*(MT0I+PAD) + (3*LSPA) = 13056
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(4)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA1+16:vgprG2LA1+16+3] offset:17408 // lwoA_0_0_4_0 = (0*LSCA)*(MT0I+PAD) + (4*LSPA) = 17408
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(3)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA1+20:vgprG2LA1+20+3] offset:21760 // lwoA_0_0_5_0 = (0*LSCA)*(MT0I+PAD) + (5*LSPA) = 21760
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(2)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA1+24:vgprG2LA1+24+3] offset:26112 // lwoA_0_0_6_0 = (0*LSCA)*(MT0I+PAD) + (6*LSPA) = 26112
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(1)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA1+28:vgprG2LA1+28+3] offset:30464 // lwoA_0_0_7_0 = (0*LSCA)*(MT0I+PAD) + (7*LSPA) = 30464
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(0)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB1+0:vgprG2LB1+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0

/* local write swap offsets a */

/* local write swap offsets b */
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X4_I0+0+0+0:vgprValuB_X4_I0+0+0+0+1], v[vgprValuA_X4_I0+4+0+0:vgprValuA_X4_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=2 */
/* dataAtIterB=0 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=1 */

/* iter 5 (reset local read pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:10  */
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=12 newLW=9 newLR=3
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X4_I0+0+2+0:vgprValuB_X4_I0+0+2+0+1], v[vgprValuA_X4_I0+0+2+0:vgprValuA_X4_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:11  */

/* local read swap offsets a */

/* local read swap offsets b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X4_I0+0+2+0:vgprValuB_X4_I0+0+2+0+1], v[vgprValuA_X4_I0+4+2+0:vgprValuA_X4_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=2 */
/* dataAtIterB=0 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=1 */

/* iter 6 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:12  */
s_waitcnt lgkmcnt(0)                               // 3wait for local write
// Skip force waitcnt0
s_barrier
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X6_I0+0+0+0:vgprValuB_X6_I0+0+0+0+1], v[vgprValuA_X6_I0+0+0+0:vgprValuA_X6_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:13  */
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X6_I0+0+0+0:vgprValuB_X6_I0+0+0+0+1], v[vgprValuA_X6_I0+4+0+0:vgprValuA_X6_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=1 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=2 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=1 */

/* iter 7 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:14  */
ds_read_b128 v[vgprValuA_X2_I0+0:vgprValuA_X2_I0+0+3], v[vgprLocalReadAddrA] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
ds_read_b128 v[vgprValuB_X2_I0+0:vgprValuB_X2_I0+0+3], v[vgprLocalReadAddrB] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
s_waitcnt lgkmcnt(14)                              // wait for prior local read local write old=0, new=14 newLW=9 newLR=5
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X6_I0+0+2+0:vgprValuB_X6_I0+0+2+0+1], v[vgprValuA_X6_I0+0+2+0:vgprValuA_X6_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:15  */
ds_read_b128 v[vgprValuA_X2_I0+4:vgprValuA_X2_I0+4+3], v[vgprLocalReadAddrA] offset:320 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X6_I0+0+2+0:vgprValuB_X6_I0+0+2+0+1], v[vgprValuA_X6_I0+4+2+0:vgprValuA_X6_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=2 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=2 readsPerIterA=2 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=2 readsPerIterB=1 */


label_last_NLL:

label_toPGR1_0:
s_cmp_eq_u32 s[sgprGSU], 1                         // GSU == 1 ?
s_cbranch_scc0 label_GSU_3                         // branch if GSU != 1

/******************************************/
/* Opt. NoLoadLoop - Begin                */
/******************************************/
s_cmpk_eq_u32 s[sgprBeta], 0x0                     // Beta == 0
s_cbranch_scc0 label_OptNLL_End                    // Branch if Beta is not zero

s_cmp_eq_u32 s[sgprAlpha], 1.0                     // Alpha == 1.0 ?
s_cbranch_scc0 label_OptNLL_End                    // branch if alpha != 1

s_and_b32 s74, 127, s[sgprSizeI]                   // s74 = s[sgprSizeI] % 128
s_add_u32 s75, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s75                // wg0 >= nwg0-1 ?
s_cselect_b32 s74, s74, 0                          // set rMT0
s_cmpk_gt_u32 s74, 0x0                             // rMT0 > 0
s_cbranch_scc1 label_OptNLL_End                    // jump if edges required
s_and_b32 s74, 15, s[sgprSizeJ]                    // s74 = s[sgprSizeJ] % 16
s_add_u32 s75, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s75                // wg1 >= nwg1-1
s_cselect_b32 s74, s74, 0                          // set rMT1
s_cmpk_gt_u32 s74, 0x0                             // rMT1 > 0
s_cbranch_scc1 label_OptNLL_End                    // jump if edges required


/* iter 0 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(3)                               // wait for prior local read local write old=0, new=3 newLW=0 newLR=3
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:1  */
ds_read_b128 v[vgprValuA_X4_I0+0:vgprValuA_X4_I0+0+3], v[vgprLocalReadAddrA] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
ds_read_b128 v[vgprValuB_X4_I0+0:vgprValuB_X4_I0+0+3], v[vgprLocalReadAddrB] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
ds_read_b128 v[vgprValuA_X4_I0+4:vgprValuA_X4_I0+4+3], v[vgprLocalReadAddrA] offset:384 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=4 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-2 numReadsIterA=1 skipReadsIterA=2 readsPerIterA=2 */
/* dataAtIterB=-2 numReadsIterB=1 skipReadsIterB=2 readsPerIterB=1 */

/* iter 1 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:2  */
ds_read_b128 v[vgprValuA_X6_I0+0:vgprValuA_X6_I0+0+3], v[vgprLocalReadAddrA] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
ds_read_b128 v[vgprValuB_X6_I0+0:vgprValuB_X6_I0+0+3], v[vgprLocalReadAddrB] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=8
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X0_I0+0+2+0:vgprValuA_X0_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:3  */
ds_read_b128 v[vgprValuA_X6_I0+4:vgprValuA_X6_I0+4+3], v[vgprLocalReadAddrA] offset:448 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=6 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X0_I0+4+2+0:vgprValuA_X0_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-2 numReadsIterA=2 skipReadsIterA=3 readsPerIterA=2 */
/* dataAtIterB=-2 numReadsIterB=2 skipReadsIterB=3 readsPerIterB=1 */

/* iter 2 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:4  */
/* localReadsVacancy: latencyLeft 2 */
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:5  */
/* localReadsVacancy: latencyLeft 2 */
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+4+0+0:vgprValuA_X2_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=2 skipReadsIterA=2 readsPerIterA=2 */
/* dataAtIterB=-1 numReadsIterB=2 skipReadsIterB=2 readsPerIterB=1 */

/* iter 3 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:6  */
/* localReadsVacancy: latencyLeft 2 */
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X2_I0+0+2+0:vgprValuA_X2_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:7  */
/* localReadsVacancy: latencyLeft 2 */
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X2_I0+4+2+0:vgprValuA_X2_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=2 skipReadsIterA=2 readsPerIterA=2 */
/* dataAtIterB=-1 numReadsIterB=2 skipReadsIterB=2 readsPerIterB=1 */

/* iter 4 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:8  */
/* schedule remaining localreads for 1LDSB */
/* localReadsVacancy: latencyLeft 2 */
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X4_I0+0+0+0:vgprValuB_X4_I0+0+0+0+1], v[vgprValuA_X4_I0+0+0+0:vgprValuA_X4_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:9  */
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X4_I0+0+0+0:vgprValuB_X4_I0+0+0+0+1], v[vgprValuA_X4_I0+4+0+0:vgprValuA_X4_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=2 */
/* dataAtIterB=0 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=1 */

/* iter 5 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:10  */
s_waitcnt lgkmcnt(3)                               // wait for prior local read local write old=0, new=3 newLW=0 newLR=3
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X4_I0+0+2+0:vgprValuB_X4_I0+0+2+0+1], v[vgprValuA_X4_I0+0+2+0:vgprValuA_X4_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:11  */
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X4_I0+0+2+0:vgprValuB_X4_I0+0+2+0+1], v[vgprValuA_X4_I0+4+2+0:vgprValuA_X4_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=2 */
/* dataAtIterB=0 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=1 */

/* iter 6 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:12  */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X6_I0+0+0+0:vgprValuB_X6_I0+0+0+0+1], v[vgprValuA_X6_I0+0+0+0:vgprValuA_X6_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:13  */
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X6_I0+0+0+0:vgprValuB_X6_I0+0+0+0+1], v[vgprValuA_X6_I0+4+0+0:vgprValuA_X6_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=0 readsPerIterB=1 */

/* iter 7 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:14  */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X6_I0+0+2+0:vgprValuB_X6_I0+0+2+0+1], v[vgprValuA_X6_I0+0+2+0:vgprValuA_X6_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:15  */
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X6_I0+0+2+0:vgprValuB_X6_I0+0+2+0+1], v[vgprValuA_X6_I0+4+2+0:vgprValuA_X6_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=0 readsPerIterB=1 */
/* Stores for OptNLL */
label_Summation_End_OptNLL:
/* endSummation: add vgpr [0...90) to pool */
/* load store sgprs */

/* Mapping of Acc register -> C Vgpr register */
/* computeStoreVgprs */
v_lshrrev_b32 v4, 6, v[vgprSerial]                 // v4 = v[vgprSerial] / 64
v_lshrrev_b32 v5, 2, v4                            // v5 = v4 / 4
v_mul_lo_u32 v5, 0x10, v5                          // wave coordination offset 1
v_and_b32 v1, 63, v[vgprSerial]                    // v1 = v[vgprSerial] % 64
v_lshrrev_b32 v1, 4, v1                            // v1 = v1 / 16
v_lshlrev_b32 v1, 0x2, v1                          // thread0 * continuous_output
v_add_lshl_u32 v1, v5, v1, 0                       // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v2, v1, s[sgprStrideC1J]              //  offset 1
v_mul_lo_u32 v3, v1, s[sgprStrideD1J]              //  offset 1
v_and_b32 v0, 3, v4                                // v0 = v4 % 4
v_mul_lo_u32 v0, 0x10, v0                          // wave coordination offset 0
v_and_b32 v5, 15, v[vgprSerial]                    // v5 = v[vgprSerial] % 16
v_add_lshl_u32 v0, v5, v0, 1                       // coordination 0 = vwA * (wave_id0 + tid0)
s_mul_i32 s8, 128, s[sgprWorkGroup0]               // wgp0 * MT0
v_add_u32 v0, s8, v0                               // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s8, 16, s[sgprWorkGroup1]                // wgp1 * MT1
v_add_u32 v1, s8, v1                               // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1

/******************************************/
/* Global Write Elements                  */
/******************************************/
label_GW_B0_E0:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=122 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 biasDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw2); (0,0,1,0:vw2); (0,0,2,0:vw2); (0,0,3,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_lshl_u32 v6, v3, v0, 0x1                     // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0
v_accvgpr_read_b32 v[vgprValuC+8], acc0            // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+9], acc4            // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+10], acc1           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+11], acc5           // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+12], acc2           // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+13], acc6           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+14], acc3           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+15], acc7           // copy acc to vreg[7]

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v[vgprValuC+8], v[vgprValuC+8]       // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+9], v[vgprValuC+9]       // convert C to fp16
v_pack_b32_f16 v8, v[vgprValuC+8], v[vgprValuC+9]  // Pack with neighbor
buffer_store_dword v8, v6, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+10], v[vgprValuC+10]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+11], v[vgprValuC+11]     // convert C to fp16
v_pack_b32_f16 v10, v[vgprValuC+10], v[vgprValuC+11] // Pack with neighbor
s_lshl_b32 s12, s[sgprStrideD1J], 1                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v10, v6, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+12], v[vgprValuC+12]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+13], v[vgprValuC+13]     // convert C to fp16
v_pack_b32_f16 v12, v[vgprValuC+12], v[vgprValuC+13] // Pack with neighbor
s_lshl_b32 s12, s[sgprStrideD1J], 1                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v12, v6, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+14], v[vgprValuC+14]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+15], v[vgprValuC+15]     // convert C to fp16
v_pack_b32_f16 v14, v[vgprValuC+14], v[vgprValuC+15] // Pack with neighbor
s_lshl_b32 s12, s[sgprStrideD1J], 1                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v14, v6, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_End:

s_endpgm                                           // Kernel End

label_OptNLL_End:
label_GSU_3:

/******************************************/
/* Ord. NoLoadLoop - Begin                */
/******************************************/

/* iter 0 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(3)                               // wait for prior local read local write old=0, new=3 newLW=0 newLR=3
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:1  */
ds_read_b128 v[vgprValuA_X4_I0+0:vgprValuA_X4_I0+0+3], v[vgprLocalReadAddrA] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
ds_read_b128 v[vgprValuB_X4_I0+0:vgprValuB_X4_I0+0+3], v[vgprLocalReadAddrB] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
ds_read_b128 v[vgprValuA_X4_I0+4:vgprValuA_X4_I0+4+3], v[vgprLocalReadAddrA] offset:384 // L -> Reg lro=64 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=4 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-2 numReadsIterA=1 skipReadsIterA=2 readsPerIterA=2 */
/* dataAtIterB=-2 numReadsIterB=1 skipReadsIterB=2 readsPerIterB=1 */

/* iter 1 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:2  */
ds_read_b128 v[vgprValuA_X6_I0+0:vgprValuA_X6_I0+0+3], v[vgprLocalReadAddrA] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
ds_read_b128 v[vgprValuB_X6_I0+0:vgprValuB_X6_I0+0+3], v[vgprLocalReadAddrB] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=8
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X0_I0+0+2+0:vgprValuA_X0_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:3  */
ds_read_b128 v[vgprValuA_X6_I0+4:vgprValuA_X6_I0+4+3], v[vgprLocalReadAddrA] offset:448 // L -> Reg lro=96 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=6 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X0_I0+4+2+0:vgprValuA_X0_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-2 numReadsIterA=2 skipReadsIterA=3 readsPerIterA=2 */
/* dataAtIterB=-2 numReadsIterB=2 skipReadsIterB=3 readsPerIterB=1 */

/* iter 2 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:4  */
/* localReadsVacancy: latencyLeft 2 */
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:5  */
/* localReadsVacancy: latencyLeft 2 */
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+4+0+0:vgprValuA_X2_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=2 skipReadsIterA=2 readsPerIterA=2 */
/* dataAtIterB=-1 numReadsIterB=2 skipReadsIterB=2 readsPerIterB=1 */

/* iter 3 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:6  */
/* localReadsVacancy: latencyLeft 2 */
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X2_I0+0+2+0:vgprValuA_X2_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:7  */
/* localReadsVacancy: latencyLeft 2 */
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X2_I0+4+2+0:vgprValuA_X2_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=2 skipReadsIterA=2 readsPerIterA=2 */
/* dataAtIterB=-1 numReadsIterB=2 skipReadsIterB=2 readsPerIterB=1 */

/* iter 4 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:8  */
/* schedule remaining localreads for 1LDSB */
/* localReadsVacancy: latencyLeft 2 */
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X4_I0+0+0+0:vgprValuB_X4_I0+0+0+0+1], v[vgprValuA_X4_I0+0+0+0:vgprValuA_X4_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:9  */
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X4_I0+0+0+0:vgprValuB_X4_I0+0+0+0+1], v[vgprValuA_X4_I0+4+0+0:vgprValuA_X4_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=2 */
/* dataAtIterB=0 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=1 */

/* iter 5 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:10  */
s_waitcnt lgkmcnt(3)                               // wait for prior local read local write old=0, new=3 newLW=0 newLR=3
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X4_I0+0+2+0:vgprValuB_X4_I0+0+2+0+1], v[vgprValuA_X4_I0+0+2+0:vgprValuA_X4_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:11  */
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X4_I0+0+2+0:vgprValuB_X4_I0+0+2+0+1], v[vgprValuA_X4_I0+4+2+0:vgprValuA_X4_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=2 */
/* dataAtIterB=0 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=1 */

/* iter 6 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:12  */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X6_I0+0+0+0:vgprValuB_X6_I0+0+0+0+1], v[vgprValuA_X6_I0+0+0+0:vgprValuA_X6_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:13  */
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X6_I0+0+0+0:vgprValuB_X6_I0+0+0+0+1], v[vgprValuA_X6_I0+4+0+0:vgprValuA_X6_I0+4+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=0 readsPerIterB=1 */

/* iter 7 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:14  */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X6_I0+0+2+0:vgprValuB_X6_I0+0+2+0+1], v[vgprValuA_X6_I0+0+2+0:vgprValuA_X6_I0+0+2+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:15  */
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X6_I0+0+2+0:vgprValuB_X6_I0+0+2+0+1], v[vgprValuA_X6_I0+4+2+0:vgprValuA_X6_I0+4+2+0+1], acc[4:7] // left value = acc[4+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=2 skipReadsIterA=0 readsPerIterA=2 */
/* dataAtIterB=1 numReadsIterB=2 skipReadsIterB=0 readsPerIterB=1 */
label_PrefetchGlobalLastIterEnd:
label_Summation_End_D9WAH3W6Q43Z6YGN_0:
/* endSummation: add vgpr [0...90) to pool */
.set sgprWGM, UNDEF
.set sgprLoopCounterL, UNDEF
.set sgprOrigLoopCounter, UNDEF
.set sgprAddressA, UNDEF
.set sgprAddressB, UNDEF
.set sgprStridesA, UNDEF
.set sgprStridesB, UNDEF
.set sgprStaggerUIter, UNDEF
.set sgprSrdA, UNDEF
.set sgprSrdB, UNDEF
.set sgprShadowLimitA, UNDEF
.set sgprShadowLimitB, UNDEF
.set sgprWrapUA, UNDEF
.set sgprWrapUB, UNDEF
.set sgprGlobalReadIncsA, UNDEF
.set sgprGlobalReadIncsB, UNDEF
.set sgprScalarGlobalReadOffsetA, UNDEF
/* load store sgprs */

/* Mapping of Acc register -> C Vgpr register */

/* not-LocalSplitU: global write indices */
/* computeStoreVgprs */
v_lshrrev_b32 v4, 6, v[vgprSerial]                 // v4 = v[vgprSerial] / 64
v_lshrrev_b32 v5, 2, v4                            // v5 = v4 / 4
v_mul_lo_u32 v5, 0x10, v5                          // wave coordination offset 1
v_and_b32 v1, 63, v[vgprSerial]                    // v1 = v[vgprSerial] % 64
v_lshrrev_b32 v1, 4, v1                            // v1 = v1 / 16
v_lshlrev_b32 v1, 0x2, v1                          // thread0 * continuous_output
v_add_lshl_u32 v1, v5, v1, 0                       // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v2, v1, s[sgprStrideC1J]              //  offset 1
v_mul_lo_u32 v3, v1, s[sgprStrideD1J]              //  offset 1
v_and_b32 v0, 3, v4                                // v0 = v4 % 4
v_mul_lo_u32 v0, 0x10, v0                          // wave coordination offset 0
v_and_b32 v5, 15, v[vgprSerial]                    // v5 = v[vgprSerial] % 16
v_add_lshl_u32 v0, v5, v0, 1                       // coordination 0 = vwA * (wave_id0 + tid0)
s_mul_i32 s8, 128, s[sgprWorkGroup0]               // wgp0 * MT0
v_add_u32 v0, s8, v0                               // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s8, 16, s[sgprWorkGroup1]                // wgp1 * MT1
v_add_u32 v1, s8, v1                               // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1

/* not-LocalSplitU: global write */

/******************************************/
/* Global Write Elements                  */
/******************************************/
s_cmp_eq_u32 s[sgprGSU], 1                         // GSU == 1 ?
s_cbranch_scc1 label_GSU_4                         // branch if GSU == 1
s_and_b32 s32, 127, s[sgprSizeI]                   // s32 = s[sgprSizeI] % 128
s_add_u32 s33, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s33                // wg0 >= nwg0-1 ?
s_cselect_b32 s32, s32, 0                          // set rMT0
s_cmpk_gt_u32 s32, 0x0                             // rMT0 > 0
s_cbranch_scc1 label_GW_B0_E1                      // jump if edges required
s_and_b32 s32, 15, s[sgprSizeJ]                    // s32 = s[sgprSizeJ] % 16
s_add_u32 s33, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s33                // wg1 >= nwg1-1
s_cselect_b32 s32, s32, 0                          // set rMT1
s_cmpk_gt_u32 s32, 0x0                             // rMT1 > 0
s_cbranch_scc1 label_GW_B0_E1                      // jump if edges required
label_GW_B0_E0_1:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=122 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 biasDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw2); (0,0,1,0:vw2); (0,0,2,0:vw2); (0,0,3,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_lshl_u32 v6, v3, v0, 0x2                     // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0
v_accvgpr_read_b32 v[vgprValuC+8], acc0            // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+9], acc4            // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+10], acc1           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+11], acc5           // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+12], acc2           // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+13], acc6           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+14], acc3           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+15], acc7           // copy acc to vreg[7]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0)] */

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[8:9], v6, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx2 v[10:11], v6, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx2 v[12:13], v6, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx2 v[14:15], v6, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_E1:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=82 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 biasDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw2); (0,0,1,0:vw2); (0,0,2,0:vw2); (0,0,3,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v18, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[48:49], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[52:53], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v6, v3, v0, 0x2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v6, v18, v6, s[52:53]                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[52:53], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v7, v3, v0, 0x2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v7, v18, v7, s[52:53]                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[52:53], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v12, v3, v0, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v12, v18, v12, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[52:53], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v13, v3, v0, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v13, v18, v13, s[52:53]              // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+8], acc0            // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+9], acc4            // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+10], acc1           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+11], acc5           // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+14], acc2           // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+15], acc6           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+16], acc3           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+17], acc7           // copy acc to vreg[7]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0)] */

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[8:9], v6, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx2 v[10:11], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx2 v[14:15], v12, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx2 v[16:17], v13, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_End_1:
s_getpc_b64 s[32:33]                               // addr of next instr
s_add_i32 s34, label_KernelEnd, 0x4                // target branch offset
s_add_u32 s32, s32, s34                            // add target branch offset
s_addc_u32 s33, s33, 0                             // add high and carry
s_setpc_b64 s[32:33]                               // branch to label_KernelEnd
label_GSU_4:
s_cmpk_eq_u32 s[sgprBeta], 0x0                     // Beta == 0
s_cbranch_scc0 label_GW_Beta_2                     // Branch if Beta is not zero

s_and_b32 s32, 127, s[sgprSizeI]                   // s32 = s[sgprSizeI] % 128
s_add_u32 s33, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s33                // wg0 >= nwg0-1 ?
s_cselect_b32 s32, s32, 0                          // set rMT0
s_cmpk_gt_u32 s32, 0x0                             // rMT0 > 0
s_cbranch_scc1 label_GW_B0_E1_1                    // jump if edges required
s_and_b32 s32, 15, s[sgprSizeJ]                    // s32 = s[sgprSizeJ] % 16
s_add_u32 s33, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s33                // wg1 >= nwg1-1
s_cselect_b32 s32, s32, 0                          // set rMT1
s_cmpk_gt_u32 s32, 0x0                             // rMT1 > 0
s_cbranch_scc1 label_GW_B0_E1_1                    // jump if edges required
label_GW_B0_E0_2:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=122 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 biasDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw2); (0,0,1,0:vw2); (0,0,2,0:vw2); (0,0,3,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_lshl_u32 v6, v3, v0, 0x1                     // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0
v_accvgpr_read_b32 v[vgprValuC+8], acc0            // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+9], acc4            // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+10], acc1           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+11], acc5           // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+12], acc2           // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+13], acc6           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+14], acc3           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+15], acc7           // copy acc to vreg[7]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0)] */
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // *= alpha
v_mul_f32 v[vgprValuC+9], s[sgprAlpha], v[vgprValuC+9] // *= alpha
v_mul_f32 v[vgprValuC+10], s[sgprAlpha], v[vgprValuC+10] // *= alpha
v_mul_f32 v[vgprValuC+11], s[sgprAlpha], v[vgprValuC+11] // *= alpha
v_mul_f32 v[vgprValuC+12], s[sgprAlpha], v[vgprValuC+12] // *= alpha
v_mul_f32 v[vgprValuC+13], s[sgprAlpha], v[vgprValuC+13] // *= alpha
v_mul_f32 v[vgprValuC+14], s[sgprAlpha], v[vgprValuC+14] // *= alpha
v_mul_f32 v[vgprValuC+15], s[sgprAlpha], v[vgprValuC+15] // *= alpha

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v[vgprValuC+8], v[vgprValuC+8]       // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+9], v[vgprValuC+9]       // convert C to fp16
v_pack_b32_f16 v8, v[vgprValuC+8], v[vgprValuC+9]  // Pack with neighbor
buffer_store_dword v8, v6, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+10], v[vgprValuC+10]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+11], v[vgprValuC+11]     // convert C to fp16
v_pack_b32_f16 v10, v[vgprValuC+10], v[vgprValuC+11] // Pack with neighbor
s_lshl_b32 s12, s[sgprStrideD1J], 1                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v10, v6, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+12], v[vgprValuC+12]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+13], v[vgprValuC+13]     // convert C to fp16
v_pack_b32_f16 v12, v[vgprValuC+12], v[vgprValuC+13] // Pack with neighbor
s_lshl_b32 s12, s[sgprStrideD1J], 1                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v12, v6, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+14], v[vgprValuC+14]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+15], v[vgprValuC+15]     // convert C to fp16
v_pack_b32_f16 v14, v[vgprValuC+14], v[vgprValuC+15] // Pack with neighbor
s_lshl_b32 s12, s[sgprStrideD1J], 1                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v14, v6, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_2                            // jump to end
label_GW_B0_E1_1:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=82 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 biasDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw2); (0,0,1,0:vw2); (0,0,2,0:vw2); (0,0,3,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v18, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[48:49], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[52:53], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v6, v3, v0, 0x1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v6, v18, v6, s[52:53]                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[52:53], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v7, v3, v0, 0x1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v7, v18, v7, s[52:53]                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[52:53], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v12, v3, v0, 0x1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v12, v18, v12, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[52:53], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v13, v3, v0, 0x1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v13, v18, v13, s[52:53]              // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+8], acc0            // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+9], acc4            // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+10], acc1           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+11], acc5           // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+14], acc2           // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+15], acc6           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+16], acc3           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+17], acc7           // copy acc to vreg[7]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0)] */
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // *= alpha
v_mul_f32 v[vgprValuC+9], s[sgprAlpha], v[vgprValuC+9] // *= alpha
v_mul_f32 v[vgprValuC+10], s[sgprAlpha], v[vgprValuC+10] // *= alpha
v_mul_f32 v[vgprValuC+11], s[sgprAlpha], v[vgprValuC+11] // *= alpha
v_mul_f32 v[vgprValuC+14], s[sgprAlpha], v[vgprValuC+14] // *= alpha
v_mul_f32 v[vgprValuC+15], s[sgprAlpha], v[vgprValuC+15] // *= alpha
v_mul_f32 v[vgprValuC+16], s[sgprAlpha], v[vgprValuC+16] // *= alpha
v_mul_f32 v[vgprValuC+17], s[sgprAlpha], v[vgprValuC+17] // *= alpha

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v[vgprValuC+8], v[vgprValuC+8]       // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+9], v[vgprValuC+9]       // convert C to fp16
v_pack_b32_f16 v8, v[vgprValuC+8], v[vgprValuC+9]  // Pack with neighbor
buffer_store_dword v8, v6, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+10], v[vgprValuC+10]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+11], v[vgprValuC+11]     // convert C to fp16
v_pack_b32_f16 v10, v[vgprValuC+10], v[vgprValuC+11] // Pack with neighbor
buffer_store_dword v10, v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+14], v[vgprValuC+14]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+15], v[vgprValuC+15]     // convert C to fp16
v_pack_b32_f16 v14, v[vgprValuC+14], v[vgprValuC+15] // Pack with neighbor
buffer_store_dword v14, v12, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+16], v[vgprValuC+16]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+17], v[vgprValuC+17]     // convert C to fp16
v_pack_b32_f16 v16, v[vgprValuC+16], v[vgprValuC+17] // Pack with neighbor
buffer_store_dword v16, v13, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_2                            // jump to end
label_GW_Beta_2:
s_and_b32 s32, 127, s[sgprSizeI]                   // s32 = s[sgprSizeI] % 128
s_add_u32 s33, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s33                // wg0 >= nwg0-1 ?
s_cselect_b32 s32, s32, 0                          // set rMT0
s_cmpk_gt_u32 s32, 0x0                             // rMT0 > 0
s_cbranch_scc1 label_GW_B1_E1                      // jump if edges required
s_and_b32 s32, 15, s[sgprSizeJ]                    // s32 = s[sgprSizeJ] % 16
s_add_u32 s33, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s33                // wg1 >= nwg1-1
s_cselect_b32 s32, s32, 0                          // set rMT1
s_cmpk_gt_u32 s32, 0x0                             // rMT1 > 0
s_cbranch_scc1 label_GW_B1_E1                      // jump if edges required
label_GW_B1_E0:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=80 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 biasDim=0 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw2); (0,0,1,0:vw2); (0,0,2,0:vw2); (0,0,3,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v7, v2, v0, 0x1                     // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0
buffer_load_dword v8, v7, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
s_lshl_b32 s12, s[sgprStrideC1J], 1                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dword v9, v7, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
s_lshl_b32 s12, s[sgprStrideC1J], 1                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dword v14, v7, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
s_lshl_b32 s12, s[sgprStrideC1J], 1                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dword v15, v7, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v6, v3, v0, 0x1                     // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0
v_accvgpr_read_b32 v[vgprValuC+10], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+11], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+12], acc1           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+13], acc5           // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+16], acc2           // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+17], acc6           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+18], acc3           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+19], acc7           // copy acc to vreg[7]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0)] */
v_mul_f32 v[vgprValuC+10], s[sgprAlpha], v[vgprValuC+10] // *= alpha
v_mul_f32 v[vgprValuC+11], s[sgprAlpha], v[vgprValuC+11] // *= alpha
v_mul_f32 v[vgprValuC+12], s[sgprAlpha], v[vgprValuC+12] // *= alpha
v_mul_f32 v[vgprValuC+13], s[sgprAlpha], v[vgprValuC+13] // *= alpha
v_mul_f32 v[vgprValuC+16], s[sgprAlpha], v[vgprValuC+16] // *= alpha
v_mul_f32 v[vgprValuC+17], s[sgprAlpha], v[vgprValuC+17] // *= alpha
v_mul_f32 v[vgprValuC+18], s[sgprAlpha], v[vgprValuC+18] // *= alpha
v_mul_f32 v[vgprValuC+19], s[sgprAlpha], v[vgprValuC+19] // *= alpha

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(3)                                 // vmcnt(3) = 4 - 1 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+10], s[sgprBeta], v8, v[vgprValuC+10] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+11], s[sgprBeta], v8, v[vgprValuC+11] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+10], v[vgprValuC+10]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+11], v[vgprValuC+11]     // convert C to fp16
v_pack_b32_f16 v10, v[vgprValuC+10], v[vgprValuC+11] // Pack with neighbor
buffer_store_dword v10, v6, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(3)                                 // vmcnt(2) = 4 - 2 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+12], s[sgprBeta], v9, v[vgprValuC+12] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+13], s[sgprBeta], v9, v[vgprValuC+13] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+12], v[vgprValuC+12]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+13], v[vgprValuC+13]     // convert C to fp16
v_pack_b32_f16 v12, v[vgprValuC+12], v[vgprValuC+13] // Pack with neighbor
s_lshl_b32 s12, s[sgprStrideD1J], 1                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v12, v6, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(3)                                 // vmcnt(1) = 4 - 3 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+16], s[sgprBeta], v14, v[vgprValuC+16] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+17], s[sgprBeta], v14, v[vgprValuC+17] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+16], v[vgprValuC+16]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+17], v[vgprValuC+17]     // convert C to fp16
v_pack_b32_f16 v16, v[vgprValuC+16], v[vgprValuC+17] // Pack with neighbor
s_lshl_b32 s12, s[sgprStrideD1J], 1                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v16, v6, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(3)                                 // vmcnt(0) = 4 - 4 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+18], s[sgprBeta], v15, v[vgprValuC+18] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+19], s[sgprBeta], v15, v[vgprValuC+19] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+18], v[vgprValuC+18]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+19], v[vgprValuC+19]     // convert C to fp16
v_pack_b32_f16 v18, v[vgprValuC+18], v[vgprValuC+19] // Pack with neighbor
s_lshl_b32 s12, s[sgprStrideD1J], 1                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v18, v6, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_2                            // jump to end
label_GW_B1_E1:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=62 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 biasDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw2); (0,0,1,0:vw2); (0,0,2,0:vw2); (0,0,3,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v22, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[48:49], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[52:53], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v6, v2, v0, 0x1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v6, v22, v6, s[52:53]                // LDC clip if OOB. offset
buffer_load_dword v7, v6, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v6, v3, v0, 0x1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v6, v22, v6, s[52:53]                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[52:53], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v10, v2, v0, 0x1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v10, v22, v10, s[52:53]              // LDC clip if OOB. offset
buffer_load_dword v11, v10, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v10, v3, v0, 0x1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v10, v22, v10, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[52:53], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v14, v2, v0, 0x1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v14, v22, v14, s[52:53]              // LDC clip if OOB. offset
buffer_load_dword v15, v14, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v14, v3, v0, 0x1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v14, v22, v14, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[52:53], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v18, v2, v0, 0x1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v18, v22, v18, s[52:53]              // LDC clip if OOB. offset
buffer_load_dword v19, v18, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v18, v3, v0, 0x1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v18, v22, v18, s[52:53]              // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+8], acc0            // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+9], acc4            // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+12], acc1           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+13], acc5           // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+16], acc2           // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+17], acc6           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+20], acc3           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+21], acc7           // copy acc to vreg[7]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0)] */
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // *= alpha
v_mul_f32 v[vgprValuC+9], s[sgprAlpha], v[vgprValuC+9] // *= alpha
v_mul_f32 v[vgprValuC+12], s[sgprAlpha], v[vgprValuC+12] // *= alpha
v_mul_f32 v[vgprValuC+13], s[sgprAlpha], v[vgprValuC+13] // *= alpha
v_mul_f32 v[vgprValuC+16], s[sgprAlpha], v[vgprValuC+16] // *= alpha
v_mul_f32 v[vgprValuC+17], s[sgprAlpha], v[vgprValuC+17] // *= alpha
v_mul_f32 v[vgprValuC+20], s[sgprAlpha], v[vgprValuC+20] // *= alpha
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+21] // *= alpha
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fma_mix_f32 v[vgprValuC+8], s[sgprBeta], v7, v[vgprValuC+8] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+9], s[sgprBeta], v7, v[vgprValuC+9] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+8], v[vgprValuC+8]       // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+9], v[vgprValuC+9]       // convert C to fp16
v_pack_b32_f16 v8, v[vgprValuC+8], v[vgprValuC+9]  // Pack with neighbor
buffer_store_dword v8, v6, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+12], s[sgprBeta], v11, v[vgprValuC+12] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+13], s[sgprBeta], v11, v[vgprValuC+13] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+12], v[vgprValuC+12]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+13], v[vgprValuC+13]     // convert C to fp16
v_pack_b32_f16 v12, v[vgprValuC+12], v[vgprValuC+13] // Pack with neighbor
buffer_store_dword v12, v10, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+16], s[sgprBeta], v15, v[vgprValuC+16] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+17], s[sgprBeta], v15, v[vgprValuC+17] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+16], v[vgprValuC+16]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+17], v[vgprValuC+17]     // convert C to fp16
v_pack_b32_f16 v16, v[vgprValuC+16], v[vgprValuC+17] // Pack with neighbor
buffer_store_dword v16, v14, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+20], s[sgprBeta], v19, v[vgprValuC+20] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+21], s[sgprBeta], v19, v[vgprValuC+21] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+20], v[vgprValuC+20]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+21], v[vgprValuC+21]     // convert C to fp16
v_pack_b32_f16 v20, v[vgprValuC+20], v[vgprValuC+21] // Pack with neighbor
buffer_store_dword v20, v18, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_2                            // jump to end
label_GW_End_2:
label_KernelEnd:
s_endpgm                                           // Kernel End
label_ASM_End:  /// The end of the kernel
