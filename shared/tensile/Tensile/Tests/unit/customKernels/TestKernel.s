/***********************************************************************************/
/*                                                                                 */
/* Copyright (C) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.       */
/*                                                                                 */
/* Permission is hereby granted, free of charge, to any person obtaining a copy    */
/* of this software and associated documentation files (the "Software"), to deal   */
/* in the Software without restriction, including without limitation the rights    */
/* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell       */
/* copies of the Software, and to permit persons to whom the Software is           */
/* furnished to do so, subject to the following conditions:                        */
/*                                                                                 */
/* The above copyright notice and this permission notice shall be included in      */
/* all copies or substantial portions of the Software.                             */
/*                                                                                 */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR      */
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        */
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE     */
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER          */
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   */
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE   */
/* SOFTWARE.                                                                       */
/*                                                                                 */
/***********************************************************************************/

/******************************************/
/* Function Prefix                        */
/******************************************/



/******************************************/
/* Begin Kernel                           */
/******************************************/

// Component.Signature.SignatureCOV3
.amdgcn_target "amdgcn-amd-amdhsa--gfx900"
.text
.protected TestKernel
.globl TestKernel
.p2align 8
.type TestKernel,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel TestKernel
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 115 // vgprs
  .amdhsa_next_free_sgpr 102 // sgprs
  .amdhsa_group_segment_fixed_size 14336 // lds bytes
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_system_vgpr_workitem_id 0
.end_amdhsa_kernel
.text

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 8 x 8 */
/* SubGroup= 8 x 16 */
/* VectorWidth=4 */
/* GlobalLoadVectorWidthA=4, GlobalLoadVectorWidthB=4 */
/* DirectToLdsA=False */
/* DirectToLdsB=False */
/* UseSgprForGRO=1 */
.amdgpu_metadata
---
custom.config:
  ProblemType:
    OperationType: GEMM
    DataType: s
    TransposeA: False
    TransposeB: False
    UseBeta: True
    Batched: True
  LoopDoWhile: False
  WorkGroupMapping:  1
  ThreadTile: [ 8, 8 ]
  WorkGroup: [  8, 16,  1 ]
  DepthU: 8
  VectorWidth: 4
  AssertSizeEqual: {3: 512}
  AssertSizeMultiple: {0: 128, 1: 128}

amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: TestKernel
    .symbol: 'TestKernel.kd'
    .language:                   OpenCL C
    .language_version:
      - 2
      - 0
    .args:
      - .name:            sizeC
        .size:            8
        .offset:          0
        .value_kind:      by_value
        .value_type:      u64
      - .name:            sizeA
        .size:            8
        .offset:          8
        .value_kind:      by_value
        .value_type:      u64
      - .name:            sizeB
        .size:            8
        .offset:          16
        .value_kind:      by_value
        .value_type:      u64
      - .name:            D
        .size:            8
        .offset:          24
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            C
        .size:            8
        .offset:          32
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            A
        .size:            8
        .offset:          40
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            B
        .size:            8
        .offset:          48
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            alpha
        .size:            4
        .offset:          56
        .value_kind:      by_value
        .value_type:      f32
      - .name:            beta
        .size:            4
        .offset:          60
        .value_kind:      by_value
        .value_type:      f32
      - .name:            strideD0
        .size:            4
        .offset:          64
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideD1
        .size:            4
        .offset:          68
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC0
        .size:            4
        .offset:          72
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC1
        .size:            4
        .offset:          76
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA0
        .size:            4
        .offset:          80
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA1
        .size:            4
        .offset:          84
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB0
        .size:            4
        .offset:          88
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB1
        .size:            4
        .offset:          92
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree0
        .size:            4
        .offset:          96
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree1
        .size:            4
        .offset:          100
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree2
        .size:            4
        .offset:          104
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesSum0
        .size:            4
        .offset:          108
        .value_kind:      by_value
        .value_type:      u32
      - .name:            OrigStaggerUIter
        .size:            4
        .offset:          112
        .value_kind:      by_value
        .value_type:      i32
      - .name:            NumWorkGroups0
        .size:            4
        .offset:          116
        .value_kind:      by_value
        .value_type:      u32
      - .name:            NumWorkGroups1
        .size:            4
        .offset:          120
        .value_kind:      by_value
        .value_type:      u32
      - .name:            NumFullBlocks
        .size:            4
        .offset:          124
        .value_kind:      by_value
        .value_type:      u32
      - .name:            WgmRemainder1
        .size:            4
        .offset:          128
        .value_kind:      by_value
        .value_type:      u32
      - .name:            MagicNumberWgmRemainder1
        .size:            4
        .offset:          132
        .value_kind:      by_value
        .value_type:      u32
      - .name:            padding
        .size:            4
        .offset:          136
        .value_kind:      by_value
        .value_type:      u32
    .group_segment_fixed_size:   14336
    .kernarg_segment_align:      8
    .kernarg_segment_size:       144
    .max_flat_workgroup_size:    128
    .private_segment_fixed_size: 0
    .sgpr_count:                 102
    .sgpr_spill_count:           0
    .vgpr_count:                 115
    .vgpr_spill_count:           0
    .wavefront_size:             64
...
.end_amdgpu_metadata
TestKernel:

/******************************************/
/* Asm syntax workarounds                 */
/******************************************/
.macro _v_add_co_u32 dst:req, cc:req, src0:req, src1:req, dpp=
   v_add_co_u32 \dst, \cc, \src0, \src1 \dpp
.endm

.macro _v_add_u32 dst:req, src0:req, src1:req, dpp=
   v_add_u32 \dst, \src0, \src1 \dpp
.endm

.macro _v_sub_co_u32 dst:req, cc:req, src0:req, src1:req, dpp=
   v_sub_co_u32 \dst, \cc, \src0, \src1 \dpp
.endm

.macro _v_sub_u32 dst:req, src0:req, src1:req, dpp=
   v_sub_u32 \dst, \src0, \src1 \dpp
.endm

.macro _v_addc_co_u32 dst:req, ccOut:req, src0:req, ccIn:req, src1:req, dpp=
   v_addc_co_u32 \dst, \ccOut, \src0, \ccIn, \src1 \dpp
.endm

.macro _v_add_lshl_u32 dst:req, src0:req, src1:req, shiftCnt:req
    v_add_lshl_u32 \dst, \src0, \src1, \shiftCnt
.endm

.macro _v_lshl_add_u32 dst:req, src0:req, src1:req, shiftCnt:req
    v_lshl_add_u32 \dst, \src0, \src1, \shiftCnt
.endm

.macro _v_cmpx_lt_i16 dst, src0, src1=
   v_cmpx_lt_i16 \dst, \src0, \src1
.endm

.macro _v_cmpx_lt_i32 dst, src0, src1=
   v_cmpx_lt_i32 \dst, \src0, \src1
.endm

.macro _v_cmpx_lt_i64 dst, src0, src1=
   v_cmpx_lt_i64 \dst, \src0, \src1
.endm

.macro _v_cmpx_lt_u16 dst, src0, src1=
   v_cmpx_lt_u16 \dst, \src0, \src1
.endm

.macro _v_cmpx_lt_u32 dst, src0, src1=
   v_cmpx_lt_u32 \dst, \src0, \src1
.endm

.macro _v_cmpx_lt_u64 dst, src0, src1=
   v_cmpx_lt_u64 \dst, \src0, \src1
.endm

.macro _v_cmpx_eq_i16 dst, src0, src1=
   v_cmpx_eq_i16 \dst, \src0, \src1
.endm

.macro _v_cmpx_eq_i32 dst, src0, src1=
   v_cmpx_eq_i32 \dst, \src0, \src1
.endm

.macro _v_cmpx_eq_i64 dst, src0, src1=
   v_cmpx_eq_i64 \dst, \src0, \src1
.endm

.macro _v_cmpx_eq_u16 dst, src0, src1=
   v_cmpx_eq_u16 \dst, \src0, \src1
.endm

.macro _v_cmpx_eq_u32 dst, src0, src1=
   v_cmpx_eq_u32 \dst, \src0, \src1
.endm

.macro _v_cmpx_eq_u64 dst, src0, src1=
   v_cmpx_eq_u64 \dst, \src0, \src1
.endm

.macro _v_cmpx_le_i16 dst, src0, src1=
   v_cmpx_le_i16 \dst, \src0, \src1
.endm

.macro _v_cmpx_le_i32 dst, src0, src1=
   v_cmpx_le_i32 \dst, \src0, \src1
.endm

.macro _v_cmpx_le_i64 dst, src0, src1=
   v_cmpx_le_i64 \dst, \src0, \src1
.endm

.macro _v_cmpx_le_u16 dst, src0, src1=
   v_cmpx_le_u16 \dst, \src0, \src1
.endm

.macro _v_cmpx_le_u32 dst, src0, src1=
   v_cmpx_le_u32 \dst, \src0, \src1
.endm

.macro _v_cmpx_le_u64 dst, src0, src1=
   v_cmpx_le_u64 \dst, \src0, \src1
.endm

.macro _v_cmpx_gt_i16 dst, src0, src1=
   v_cmpx_gt_i16 \dst, \src0, \src1
.endm

.macro _v_cmpx_gt_i32 dst, src0, src1=
   v_cmpx_gt_i32 \dst, \src0, \src1
.endm

.macro _v_cmpx_gt_i64 dst, src0, src1=
   v_cmpx_gt_i64 \dst, \src0, \src1
.endm

.macro _v_cmpx_gt_u16 dst, src0, src1=
   v_cmpx_gt_u16 \dst, \src0, \src1
.endm

.macro _v_cmpx_gt_u32 dst, src0, src1=
   v_cmpx_gt_u32 \dst, \src0, \src1
.endm

.macro _v_cmpx_gt_u64 dst, src0, src1=
   v_cmpx_gt_u64 \dst, \src0, \src1
.endm

.macro _v_cmpx_ne_i16 dst, src0, src1=
   v_cmpx_ne_i16 \dst, \src0, \src1
.endm

.macro _v_cmpx_ne_i32 dst, src0, src1=
   v_cmpx_ne_i32 \dst, \src0, \src1
.endm

.macro _v_cmpx_ne_i64 dst, src0, src1=
   v_cmpx_ne_i64 \dst, \src0, \src1
.endm

.macro _v_cmpx_ne_u16 dst, src0, src1=
   v_cmpx_ne_u16 \dst, \src0, \src1
.endm

.macro _v_cmpx_ne_u32 dst, src0, src1=
   v_cmpx_ne_u32 \dst, \src0, \src1
.endm

.macro _v_cmpx_ne_u64 dst, src0, src1=
   v_cmpx_ne_u64 \dst, \src0, \src1
.endm

.macro _v_cmpx_lg_i16 dst, src0, src1=
   v_cmpx_lg_i16 \dst, \src0, \src1
.endm

.macro _v_cmpx_lg_i32 dst, src0, src1=
   v_cmpx_lg_i32 \dst, \src0, \src1
.endm

.macro _v_cmpx_lg_i64 dst, src0, src1=
   v_cmpx_lg_i64 \dst, \src0, \src1
.endm

.macro _v_cmpx_lg_u16 dst, src0, src1=
   v_cmpx_lg_u16 \dst, \src0, \src1
.endm

.macro _v_cmpx_lg_u32 dst, src0, src1=
   v_cmpx_lg_u32 \dst, \src0, \src1
.endm

.macro _v_cmpx_lg_u64 dst, src0, src1=
   v_cmpx_lg_u64 \dst, \src0, \src1
.endm

.macro _v_cmpx_ge_i16 dst, src0, src1=
   v_cmpx_ge_i16 \dst, \src0, \src1
.endm

.macro _v_cmpx_ge_i32 dst, src0, src1=
   v_cmpx_ge_i32 \dst, \src0, \src1
.endm

.macro _v_cmpx_ge_i64 dst, src0, src1=
   v_cmpx_ge_i64 \dst, \src0, \src1
.endm

.macro _v_cmpx_ge_u16 dst, src0, src1=
   v_cmpx_ge_u16 \dst, \src0, \src1
.endm

.macro _v_cmpx_ge_u32 dst, src0, src1=
   v_cmpx_ge_u32 \dst, \src0, \src1
.endm

.macro _v_cmpx_ge_u64 dst, src0, src1=
   v_cmpx_ge_u64 \dst, \src0, \src1
.endm

.macro _v_cmpx_o_i16 dst, src0, src1=
   v_cmpx_o_i16 \dst, \src0, \src1
.endm

.macro _v_cmpx_o_i32 dst, src0, src1=
   v_cmpx_o_i32 \dst, \src0, \src1
.endm

.macro _v_cmpx_o_i64 dst, src0, src1=
   v_cmpx_o_i64 \dst, \src0, \src1
.endm

.macro _v_cmpx_o_u16 dst, src0, src1=
   v_cmpx_o_u16 \dst, \src0, \src1
.endm

.macro _v_cmpx_o_u32 dst, src0, src1=
   v_cmpx_o_u32 \dst, \src0, \src1
.endm

.macro _v_cmpx_o_u64 dst, src0, src1=
   v_cmpx_o_u64 \dst, \src0, \src1
.endm

.macro _v_cmpx_u_i16 dst, src0, src1=
   v_cmpx_u_i16 \dst, \src0, \src1
.endm

.macro _v_cmpx_u_i32 dst, src0, src1=
   v_cmpx_u_i32 \dst, \src0, \src1
.endm

.macro _v_cmpx_u_i64 dst, src0, src1=
   v_cmpx_u_i64 \dst, \src0, \src1
.endm

.macro _v_cmpx_u_u16 dst, src0, src1=
   v_cmpx_u_u16 \dst, \src0, \src1
.endm

.macro _v_cmpx_u_u32 dst, src0, src1=
   v_cmpx_u_u32 \dst, \src0, \src1
.endm

.macro _v_cmpx_u_u64 dst, src0, src1=
   v_cmpx_u_u64 \dst, \src0, \src1
.endm

/******************************************/
/* Magic div and mod functions            */
/******************************************/
.macro V_MAGIC_DIV dstIdx:req, dividend:req, magicNumber:req, magicShift:req, magicA:req
    v_mul_hi_u32 v[\dstIdx+1], \dividend, \magicNumber
    v_mul_lo_u32 v[\dstIdx+0], \dividend, \magicA
    v_add_u32 v[\dstIdx+0], v[\dstIdx+0], v[\dstIdx+1]
    v_lshrrev_b32 v[\dstIdx+0], \magicShift, v[\dstIdx+0]
.endm

/******************************************/
/* VGPR Assignments                       */
/******************************************/
/* ValuC range: [0-64), ,  */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprValuA_X0_I0, 64
.set vgprValuA_X1_I0, 72
.set vgprG2LA, 100
.set vgprValuB_X0_I0, 80
.set vgprValuB_X1_I0, 88
.set vgprG2LB, 104
.set vgprLocalWriteAddrA, 96
.set vgprLocalWriteAddrB, 97
.set vgprGlobalReadOffsetA, 98
.set vgprGlobalReadOffsetB, 99
.set vgprLocalReadAddrA, 112
.set vgprLocalReadAddrB, 113
.set vgprSerial, 114
/* Num VGPR=115 */
/* Num AccVGPR=0 */

/******************************************/
/* SGPR Assignments                       */
/******************************************/
.set sgprKernArgAddress, 0
.set sgprWorkGroup0, 2
.set sgprWorkGroup1, 3
.set sgprWorkGroup2, 4
.set sgprLoopCounterL, 5
.set sgprOrigLoopCounter, 6
.set sgprSrdA, 8
.set sgprSrdB, 12
.set sgprSrdD, 16
.set sgprSrdC, 20
.set sgprTensor2dSizeA, 24
.set sgprTensor2dSizeB, 26
.set sgprAddressD, 28
.set sgprAddressC, 30
.set sgprAddressA, 32
.set sgprAddressB, 34
.set sgprAlpha, 36
.set sgprBeta, 37
.set sgprStridesD, 38
.set sgprStridesC, 40
.set sgprStridesA, 42
.set sgprStridesB, 44
.set sgprSizesFree, 46
.set sgprSizesSum, 49
.set sgprOrigStaggerUIter, 50
.set sgprNumWorkGroups0, 51
.set sgprNumWorkGroups1, 52
.set sgprNumFullBlocks, 53
.set sgprWgmRemainder1, 54
.set sgprMagicNumberWgmRemainder1, 55
.set sgprShadowLimitA, 38
.set sgprShadowLimitB, 56
.set sgprStaggerUIter, 7
.set sgprWrapUA, 58
.set sgprWrapUB, 60
.set sgprGlobalReadIncsA, 62
.set sgprGlobalReadIncsB, 63
.set sgprScalarGlobalReadOffsetB, 64
.set sgprWaveId, 65
/* max SGPR=102 */

/* Size Assignments */
.set sgprSizeI, sgprSizesFree+0
.set sgprSizeJ, sgprSizesFree+1
.set sgprSizeK, sgprSizesFree+2
.set sgprSizeL, sgprSizesSum+0

/* Stride Assignments */
.set constStrideD0I, 1
.set sgprStrideD1J, sgprStridesC+0
.set sgprStrideDK, sgprStridesC+1
.set constStrideC0I, 1
.set sgprStrideC1J, sgprStridesC+0
.set sgprStrideCK, sgprStridesC+1
.set constStrideA0I, 1
.set sgprStrideAL, sgprStridesA+0
.set sgprStrideAK, sgprStridesA+1
.set constStrideBL, 1
.set sgprStrideB1J, sgprStridesB+0
.set sgprStrideBK, sgprStridesB+1

.set MT0, 64
.set MT1, 128
.set DepthU, 8
.set GSU, 1
.set BpeA, 4
.set BpeALog2, 2
.set BpeB, 4
.set BpeBLog2, 2
/* Number of elements to shift-left SRD */
.set SrdShiftLeftA, 4
.set SrdShiftLeftB, 4
/* 2GB limit - set offsets to -1 to exceed this and clamp */
.set BufferLimit, 0x80000000
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
.macro GLOBAL_OFFSET_A vgprAddr:req vgprOffset0I:req vgprOffsetL:req vgprTmp:req
v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideAL], v[\vgprOffsetL] // mul d1 lower
_v_add_co_u32 v[\vgprAddr+0], vcc, v[\vgprOffset0I], v[\vgprTmp+0] // accumulate K lower
_v_add_u32 v[\vgprAddr+0], 0x4, v[\vgprAddr+0]     // add prepad for pointer shift
v_lshlrev_b32 v[\vgprAddr+0], 0x2, v[\vgprAddr+0]  // offset *= bytes/element
.endm

/* Global Offset B */
.macro GLOBAL_OFFSET_B vgprAddr:req vgprOffsetL:req vgprOffset1J:req vgprTmp:req
v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideB1J], v[\vgprOffset1J] // mul d1 lower
_v_add_co_u32 v[\vgprAddr+0], vcc, v[\vgprOffsetL], v[\vgprTmp+0] // accumulate K lower
_v_add_u32 v[\vgprAddr+0], 0x4, v[\vgprAddr+0]     // add prepad for pointer shift
v_lshlrev_b32 v[\vgprAddr+0], 0x2, v[\vgprAddr+0]  // offset *= bytes/element
.endm

/******************************************/
/* Dynamic Scalar Divide: vQuotient=vDividend/vDivisor; vRemainder=vDividend%vDivisor; */
/******************************************/
.macro DYNAMIC_VECTOR_DIVIDE vQuotient vRemainder vDividend vDivisor vTmp0 vTmp1 sTmp
v_cvt_f32_u32 v[\vQuotient], v[\vDivisor]          //
v_rcp_f32 v[\vQuotient], v[\vQuotient]             //
v_mul_f32 v[\vQuotient], 0x4f800000, v[\vQuotient] //
v_cvt_u32_f32 v[\vQuotient], v[\vQuotient]         //
v_mul_lo_u32 v[\vRemainder], v[\vDivisor], v[\vQuotient] //
v_mul_hi_u32 v[\vTmp0], v[\vDivisor], v[\vQuotient] //
_v_sub_co_u32 v[\vTmp1], vcc, 0x0, v[\vRemainder]  //
v_cmp_ne_i32 s[\sTmp:\sTmp+1], 0x0, v[\vTmp0]      //
v_cndmask_b32 v[\vRemainder], v[\vTmp1], v[\vRemainder], s[\sTmp:\sTmp+1] //
v_mul_hi_u32 v[\vRemainder], v[\vRemainder], v[\vQuotient] //
_v_sub_co_u32 v[\vTmp0], vcc, v[\vQuotient], v[\vRemainder] //
_v_add_co_u32 v[\vQuotient], vcc, v[\vQuotient], v[\vRemainder] //
v_cndmask_b32 v[\vQuotient], v[\vQuotient], v[\vTmp0], s[\sTmp:\sTmp+1] //
v_mul_hi_u32 v[\vQuotient], v[\vQuotient], v[\vDividend] //
v_mul_lo_u32 v[\vRemainder], v[\vQuotient], v[\vDivisor] //
_v_sub_co_u32 v[\vTmp0], vcc, v[\vDividend], v[\vRemainder] //
v_cmp_ge_u32 s[\sTmp:\sTmp+1], v[\vDividend], v[\vRemainder] //
_v_add_co_u32 v[\vRemainder], vcc, 0x1, v[\vQuotient] //
_v_add_co_u32 v[\vTmp1], vcc, -1, v[\vQuotient]    //
v_cmp_le_u32 vcc, v[\vDivisor], v[\vTmp0]          //
s_and_b64 vcc, s[\sTmp:\sTmp+1], vcc               //
v_cndmask_b32 v[\vQuotient], v[\vQuotient], v[\vRemainder], vcc //
v_cndmask_b32 v[\vQuotient], v[\vTmp1], v[\vQuotient], s[\sTmp:\sTmp+1] //
v_cmp_ne_i32 vcc, 0x0, v[\vDivisor]                //
v_cndmask_b32 v[\vQuotient], -1, v[\vQuotient], vcc // final result
v_mul_lo_u32 v[\vRemainder], v[\vQuotient], v[\vDivisor] //
_v_sub_co_u32 v[\vRemainder], vcc, v[\vDividend], v[\vRemainder] // final result
.endm

/******************************************/
/* 8x8 thread-tile                        */
/******************************************/
.macro MAC_8x8_X0
v_mac_f32 v[vgprValuC+0+0*8], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0]
s_setprio 1 // Raise priority while processing macs
v_mac_f32 v[vgprValuC+1+0*8], v[vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+0]
v_mac_f32 v[vgprValuC+2+0*8], v[vgprValuA_X0_I0+2], v[vgprValuB_X0_I0+0]
v_mac_f32 v[vgprValuC+3+0*8], v[vgprValuA_X0_I0+3], v[vgprValuB_X0_I0+0]
v_mac_f32 v[vgprValuC+4+0*8], v[vgprValuA_X0_I0+4], v[vgprValuB_X0_I0+0]
v_mac_f32 v[vgprValuC+5+0*8], v[vgprValuA_X0_I0+5], v[vgprValuB_X0_I0+0]
v_mac_f32 v[vgprValuC+6+0*8], v[vgprValuA_X0_I0+6], v[vgprValuB_X0_I0+0]
v_mac_f32 v[vgprValuC+7+0*8], v[vgprValuA_X0_I0+7], v[vgprValuB_X0_I0+0]
v_mac_f32 v[vgprValuC+0+1*8], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+1]
v_mac_f32 v[vgprValuC+1+1*8], v[vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+1]
v_mac_f32 v[vgprValuC+2+1*8], v[vgprValuA_X0_I0+2], v[vgprValuB_X0_I0+1]
v_mac_f32 v[vgprValuC+3+1*8], v[vgprValuA_X0_I0+3], v[vgprValuB_X0_I0+1]
v_mac_f32 v[vgprValuC+4+1*8], v[vgprValuA_X0_I0+4], v[vgprValuB_X0_I0+1]
v_mac_f32 v[vgprValuC+5+1*8], v[vgprValuA_X0_I0+5], v[vgprValuB_X0_I0+1]
v_mac_f32 v[vgprValuC+6+1*8], v[vgprValuA_X0_I0+6], v[vgprValuB_X0_I0+1]
v_mac_f32 v[vgprValuC+7+1*8], v[vgprValuA_X0_I0+7], v[vgprValuB_X0_I0+1]
v_mac_f32 v[vgprValuC+0+2*8], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+2]
v_mac_f32 v[vgprValuC+1+2*8], v[vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+2]
v_mac_f32 v[vgprValuC+2+2*8], v[vgprValuA_X0_I0+2], v[vgprValuB_X0_I0+2]
v_mac_f32 v[vgprValuC+3+2*8], v[vgprValuA_X0_I0+3], v[vgprValuB_X0_I0+2]
v_mac_f32 v[vgprValuC+4+2*8], v[vgprValuA_X0_I0+4], v[vgprValuB_X0_I0+2]
v_mac_f32 v[vgprValuC+5+2*8], v[vgprValuA_X0_I0+5], v[vgprValuB_X0_I0+2]
v_mac_f32 v[vgprValuC+6+2*8], v[vgprValuA_X0_I0+6], v[vgprValuB_X0_I0+2]
v_mac_f32 v[vgprValuC+7+2*8], v[vgprValuA_X0_I0+7], v[vgprValuB_X0_I0+2]
v_mac_f32 v[vgprValuC+0+3*8], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+3]
v_mac_f32 v[vgprValuC+1+3*8], v[vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+3]
v_mac_f32 v[vgprValuC+2+3*8], v[vgprValuA_X0_I0+2], v[vgprValuB_X0_I0+3]
v_mac_f32 v[vgprValuC+3+3*8], v[vgprValuA_X0_I0+3], v[vgprValuB_X0_I0+3]
v_mac_f32 v[vgprValuC+4+3*8], v[vgprValuA_X0_I0+4], v[vgprValuB_X0_I0+3]
v_mac_f32 v[vgprValuC+5+3*8], v[vgprValuA_X0_I0+5], v[vgprValuB_X0_I0+3]
v_mac_f32 v[vgprValuC+6+3*8], v[vgprValuA_X0_I0+6], v[vgprValuB_X0_I0+3]
v_mac_f32 v[vgprValuC+7+3*8], v[vgprValuA_X0_I0+7], v[vgprValuB_X0_I0+3]
v_mac_f32 v[vgprValuC+0+4*8], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+4]
v_mac_f32 v[vgprValuC+1+4*8], v[vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+4]
v_mac_f32 v[vgprValuC+2+4*8], v[vgprValuA_X0_I0+2], v[vgprValuB_X0_I0+4]
v_mac_f32 v[vgprValuC+3+4*8], v[vgprValuA_X0_I0+3], v[vgprValuB_X0_I0+4]
v_mac_f32 v[vgprValuC+4+4*8], v[vgprValuA_X0_I0+4], v[vgprValuB_X0_I0+4]
v_mac_f32 v[vgprValuC+5+4*8], v[vgprValuA_X0_I0+5], v[vgprValuB_X0_I0+4]
v_mac_f32 v[vgprValuC+6+4*8], v[vgprValuA_X0_I0+6], v[vgprValuB_X0_I0+4]
v_mac_f32 v[vgprValuC+7+4*8], v[vgprValuA_X0_I0+7], v[vgprValuB_X0_I0+4]
v_mac_f32 v[vgprValuC+0+5*8], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+5]
v_mac_f32 v[vgprValuC+1+5*8], v[vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+5]
v_mac_f32 v[vgprValuC+2+5*8], v[vgprValuA_X0_I0+2], v[vgprValuB_X0_I0+5]
v_mac_f32 v[vgprValuC+3+5*8], v[vgprValuA_X0_I0+3], v[vgprValuB_X0_I0+5]
v_mac_f32 v[vgprValuC+4+5*8], v[vgprValuA_X0_I0+4], v[vgprValuB_X0_I0+5]
v_mac_f32 v[vgprValuC+5+5*8], v[vgprValuA_X0_I0+5], v[vgprValuB_X0_I0+5]
v_mac_f32 v[vgprValuC+6+5*8], v[vgprValuA_X0_I0+6], v[vgprValuB_X0_I0+5]
v_mac_f32 v[vgprValuC+7+5*8], v[vgprValuA_X0_I0+7], v[vgprValuB_X0_I0+5]
v_mac_f32 v[vgprValuC+0+6*8], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+6]
v_mac_f32 v[vgprValuC+1+6*8], v[vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+6]
v_mac_f32 v[vgprValuC+2+6*8], v[vgprValuA_X0_I0+2], v[vgprValuB_X0_I0+6]
v_mac_f32 v[vgprValuC+3+6*8], v[vgprValuA_X0_I0+3], v[vgprValuB_X0_I0+6]
v_mac_f32 v[vgprValuC+4+6*8], v[vgprValuA_X0_I0+4], v[vgprValuB_X0_I0+6]
v_mac_f32 v[vgprValuC+5+6*8], v[vgprValuA_X0_I0+5], v[vgprValuB_X0_I0+6]
v_mac_f32 v[vgprValuC+6+6*8], v[vgprValuA_X0_I0+6], v[vgprValuB_X0_I0+6]
v_mac_f32 v[vgprValuC+7+6*8], v[vgprValuA_X0_I0+7], v[vgprValuB_X0_I0+6]
v_mac_f32 v[vgprValuC+0+7*8], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+7]
v_mac_f32 v[vgprValuC+1+7*8], v[vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+7]
v_mac_f32 v[vgprValuC+2+7*8], v[vgprValuA_X0_I0+2], v[vgprValuB_X0_I0+7]
v_mac_f32 v[vgprValuC+3+7*8], v[vgprValuA_X0_I0+3], v[vgprValuB_X0_I0+7]
v_mac_f32 v[vgprValuC+4+7*8], v[vgprValuA_X0_I0+4], v[vgprValuB_X0_I0+7]
v_mac_f32 v[vgprValuC+5+7*8], v[vgprValuA_X0_I0+5], v[vgprValuB_X0_I0+7]
v_mac_f32 v[vgprValuC+6+7*8], v[vgprValuA_X0_I0+6], v[vgprValuB_X0_I0+7]
v_mac_f32 v[vgprValuC+7+7*8], v[vgprValuA_X0_I0+7], v[vgprValuB_X0_I0+7]
s_setprio 0 // Reset priority after macs
.endm
.macro MAC_8x8_X1
v_mac_f32 v[vgprValuC+0+0*8], v[vgprValuA_X1_I0+0], v[vgprValuB_X1_I0+0]
s_setprio 1 // Raise priority while processing macs
v_mac_f32 v[vgprValuC+1+0*8], v[vgprValuA_X1_I0+1], v[vgprValuB_X1_I0+0]
v_mac_f32 v[vgprValuC+2+0*8], v[vgprValuA_X1_I0+2], v[vgprValuB_X1_I0+0]
v_mac_f32 v[vgprValuC+3+0*8], v[vgprValuA_X1_I0+3], v[vgprValuB_X1_I0+0]
v_mac_f32 v[vgprValuC+4+0*8], v[vgprValuA_X1_I0+4], v[vgprValuB_X1_I0+0]
v_mac_f32 v[vgprValuC+5+0*8], v[vgprValuA_X1_I0+5], v[vgprValuB_X1_I0+0]
v_mac_f32 v[vgprValuC+6+0*8], v[vgprValuA_X1_I0+6], v[vgprValuB_X1_I0+0]
v_mac_f32 v[vgprValuC+7+0*8], v[vgprValuA_X1_I0+7], v[vgprValuB_X1_I0+0]
v_mac_f32 v[vgprValuC+0+1*8], v[vgprValuA_X1_I0+0], v[vgprValuB_X1_I0+1]
v_mac_f32 v[vgprValuC+1+1*8], v[vgprValuA_X1_I0+1], v[vgprValuB_X1_I0+1]
v_mac_f32 v[vgprValuC+2+1*8], v[vgprValuA_X1_I0+2], v[vgprValuB_X1_I0+1]
v_mac_f32 v[vgprValuC+3+1*8], v[vgprValuA_X1_I0+3], v[vgprValuB_X1_I0+1]
v_mac_f32 v[vgprValuC+4+1*8], v[vgprValuA_X1_I0+4], v[vgprValuB_X1_I0+1]
v_mac_f32 v[vgprValuC+5+1*8], v[vgprValuA_X1_I0+5], v[vgprValuB_X1_I0+1]
v_mac_f32 v[vgprValuC+6+1*8], v[vgprValuA_X1_I0+6], v[vgprValuB_X1_I0+1]
v_mac_f32 v[vgprValuC+7+1*8], v[vgprValuA_X1_I0+7], v[vgprValuB_X1_I0+1]
v_mac_f32 v[vgprValuC+0+2*8], v[vgprValuA_X1_I0+0], v[vgprValuB_X1_I0+2]
v_mac_f32 v[vgprValuC+1+2*8], v[vgprValuA_X1_I0+1], v[vgprValuB_X1_I0+2]
v_mac_f32 v[vgprValuC+2+2*8], v[vgprValuA_X1_I0+2], v[vgprValuB_X1_I0+2]
v_mac_f32 v[vgprValuC+3+2*8], v[vgprValuA_X1_I0+3], v[vgprValuB_X1_I0+2]
v_mac_f32 v[vgprValuC+4+2*8], v[vgprValuA_X1_I0+4], v[vgprValuB_X1_I0+2]
v_mac_f32 v[vgprValuC+5+2*8], v[vgprValuA_X1_I0+5], v[vgprValuB_X1_I0+2]
v_mac_f32 v[vgprValuC+6+2*8], v[vgprValuA_X1_I0+6], v[vgprValuB_X1_I0+2]
v_mac_f32 v[vgprValuC+7+2*8], v[vgprValuA_X1_I0+7], v[vgprValuB_X1_I0+2]
v_mac_f32 v[vgprValuC+0+3*8], v[vgprValuA_X1_I0+0], v[vgprValuB_X1_I0+3]
v_mac_f32 v[vgprValuC+1+3*8], v[vgprValuA_X1_I0+1], v[vgprValuB_X1_I0+3]
v_mac_f32 v[vgprValuC+2+3*8], v[vgprValuA_X1_I0+2], v[vgprValuB_X1_I0+3]
v_mac_f32 v[vgprValuC+3+3*8], v[vgprValuA_X1_I0+3], v[vgprValuB_X1_I0+3]
v_mac_f32 v[vgprValuC+4+3*8], v[vgprValuA_X1_I0+4], v[vgprValuB_X1_I0+3]
v_mac_f32 v[vgprValuC+5+3*8], v[vgprValuA_X1_I0+5], v[vgprValuB_X1_I0+3]
v_mac_f32 v[vgprValuC+6+3*8], v[vgprValuA_X1_I0+6], v[vgprValuB_X1_I0+3]
v_mac_f32 v[vgprValuC+7+3*8], v[vgprValuA_X1_I0+7], v[vgprValuB_X1_I0+3]
v_mac_f32 v[vgprValuC+0+4*8], v[vgprValuA_X1_I0+0], v[vgprValuB_X1_I0+4]
v_mac_f32 v[vgprValuC+1+4*8], v[vgprValuA_X1_I0+1], v[vgprValuB_X1_I0+4]
v_mac_f32 v[vgprValuC+2+4*8], v[vgprValuA_X1_I0+2], v[vgprValuB_X1_I0+4]
v_mac_f32 v[vgprValuC+3+4*8], v[vgprValuA_X1_I0+3], v[vgprValuB_X1_I0+4]
v_mac_f32 v[vgprValuC+4+4*8], v[vgprValuA_X1_I0+4], v[vgprValuB_X1_I0+4]
v_mac_f32 v[vgprValuC+5+4*8], v[vgprValuA_X1_I0+5], v[vgprValuB_X1_I0+4]
v_mac_f32 v[vgprValuC+6+4*8], v[vgprValuA_X1_I0+6], v[vgprValuB_X1_I0+4]
v_mac_f32 v[vgprValuC+7+4*8], v[vgprValuA_X1_I0+7], v[vgprValuB_X1_I0+4]
v_mac_f32 v[vgprValuC+0+5*8], v[vgprValuA_X1_I0+0], v[vgprValuB_X1_I0+5]
v_mac_f32 v[vgprValuC+1+5*8], v[vgprValuA_X1_I0+1], v[vgprValuB_X1_I0+5]
v_mac_f32 v[vgprValuC+2+5*8], v[vgprValuA_X1_I0+2], v[vgprValuB_X1_I0+5]
v_mac_f32 v[vgprValuC+3+5*8], v[vgprValuA_X1_I0+3], v[vgprValuB_X1_I0+5]
v_mac_f32 v[vgprValuC+4+5*8], v[vgprValuA_X1_I0+4], v[vgprValuB_X1_I0+5]
v_mac_f32 v[vgprValuC+5+5*8], v[vgprValuA_X1_I0+5], v[vgprValuB_X1_I0+5]
v_mac_f32 v[vgprValuC+6+5*8], v[vgprValuA_X1_I0+6], v[vgprValuB_X1_I0+5]
v_mac_f32 v[vgprValuC+7+5*8], v[vgprValuA_X1_I0+7], v[vgprValuB_X1_I0+5]
v_mac_f32 v[vgprValuC+0+6*8], v[vgprValuA_X1_I0+0], v[vgprValuB_X1_I0+6]
v_mac_f32 v[vgprValuC+1+6*8], v[vgprValuA_X1_I0+1], v[vgprValuB_X1_I0+6]
v_mac_f32 v[vgprValuC+2+6*8], v[vgprValuA_X1_I0+2], v[vgprValuB_X1_I0+6]
v_mac_f32 v[vgprValuC+3+6*8], v[vgprValuA_X1_I0+3], v[vgprValuB_X1_I0+6]
v_mac_f32 v[vgprValuC+4+6*8], v[vgprValuA_X1_I0+4], v[vgprValuB_X1_I0+6]
v_mac_f32 v[vgprValuC+5+6*8], v[vgprValuA_X1_I0+5], v[vgprValuB_X1_I0+6]
v_mac_f32 v[vgprValuC+6+6*8], v[vgprValuA_X1_I0+6], v[vgprValuB_X1_I0+6]
v_mac_f32 v[vgprValuC+7+6*8], v[vgprValuA_X1_I0+7], v[vgprValuB_X1_I0+6]
v_mac_f32 v[vgprValuC+0+7*8], v[vgprValuA_X1_I0+0], v[vgprValuB_X1_I0+7]
v_mac_f32 v[vgprValuC+1+7*8], v[vgprValuA_X1_I0+1], v[vgprValuB_X1_I0+7]
v_mac_f32 v[vgprValuC+2+7*8], v[vgprValuA_X1_I0+2], v[vgprValuB_X1_I0+7]
v_mac_f32 v[vgprValuC+3+7*8], v[vgprValuA_X1_I0+3], v[vgprValuB_X1_I0+7]
v_mac_f32 v[vgprValuC+4+7*8], v[vgprValuA_X1_I0+4], v[vgprValuB_X1_I0+7]
v_mac_f32 v[vgprValuC+5+7*8], v[vgprValuA_X1_I0+5], v[vgprValuB_X1_I0+7]
v_mac_f32 v[vgprValuC+6+7*8], v[vgprValuA_X1_I0+6], v[vgprValuB_X1_I0+7]
v_mac_f32 v[vgprValuC+7+7*8], v[vgprValuA_X1_I0+7], v[vgprValuB_X1_I0+7]
s_setprio 0 // Reset priority after macs
.endm



/******************************************/
/* Allocate Resources                     */
/******************************************/

s_mov_b32 m0, 0x3800                               // LDS clamp at 14336 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id

/* Load Kernel Args */
s_load_dwordx16 s[24:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x8 //
s_load_dwordx16 s[40:55], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x48 //
s_waitcnt lgkmcnt(0)                               // wait for 136 bytes of kern args
v_lshrrev_b32  v0, 0x6, v[vgprSerial]              // Wavefront Serial Id
v_readfirstlane_b32 s[sgprWaveId], v0              // WaveId

/* Short circuit condition if Alpha == 0, then sumDims=0 */
v_cmp_eq_f32 vcc, s[sgprAlpha], 0.0                // Alpha == 0.0f ?
s_cbranch_vccz label_AlphaNonZero                  // branch if alpha != 0
s_mov_b32 s[sgprSizesSum+0], 0x0                   // Set summation dim=0 if Alpha == 0
label_AlphaNonZero:


/******************************************/
/* Local Read Addresses                   */
/******************************************/


/* local read addresses: tile assignments a */

/*lr0I = serial % SG0I*/
v_lshrrev_b32 v0, 3, v[vgprSerial]                 // v0 = v[vgprSerial] / 8
v_and_b32 v1, 7, v[vgprSerial]                     // v1 = v[vgprSerial] % 8


/* local read addresses: tile assignments b */

/*lr1J = (serial / SG1J) % SG1J*/
v_lshrrev_b32 v2, 4, v0                            // v2 = v0 / 16
v_and_b32 v3, 15, v0                               // v3 = v0 % 16


/* local read addresses: final offsets a */

v_lshrrev_b32 v0, 7, v[vgprSerial]                 // LSU offset: sgid = Serial / subGroup(128)
s_mov_b32 s66, 64                                  // LSU offset: stirde = MT0(64) + PAD0(0)
v_mul_lo_u32 v0, s66, v0                           // LSU offset: lsuoffset = sgid*(MT0+PAD)
v_lshlrev_b32 v1, 2, v1                            // Final Offset: lrAOffset * VW
_v_add_lshl_u32 v[vgprLocalReadAddrA], v0, v1, 0x2 // Final Offset: offset = (lro0*VW+lsuoffset)*bpe


/* local read addresses: final offsets b */

v_lshrrev_b32 v0, 7, v[vgprSerial]                 // LSU offset: sgid = Serial / subGroup(128)
s_mov_b32 s66, 128                                 // LSU offset: stirde = MT1(128) + PAD1(0)
v_mul_lo_u32 v0, s66, v0                           // LSU offset: lsuoffset = sgid*(MT1+PAD)
v_lshlrev_b32 v3, 2, v3                            // Final Offset: lrBOffset * VW
_v_add_lshl_u32 v[vgprLocalReadAddrB], v0, v3, 0x2 // Final Offset: offset = (lro1*VW+lsuoffset)*bpe


/* local read addresses: declare addresses a */

/* N/A */


/* local read addresses: declare addresses b */

_v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, 0x800, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)



/******************************************/
/* Begin setupNewTile, isPap=False           */
/******************************************/


/* global read addresses: work-group */

/* graWorkGroup mapping */


/* global read addresses: tile offset assignment a */

/* LVCA = 16 */
/* v0 = (local)groA-tile = serial%LVCA (note (wgA*MTA) will be added to SRD) */
/* v1 = groA-unroll = serial/LVCA */
v_lshrrev_b32 v1, 4, v[vgprSerial]                 // v1 = v[vgprSerial] / 16
v_and_b32 v0, 15, v[vgprSerial]                    // v0 = v[vgprSerial] % 16
/* gro-tile *= glvw */
v_lshlrev_b32 v0, 2, v0                            // v0 = v0 * 4


/* global read addresses: tile offset assignment b */

/* LVCB = 2 */
/* v2 = (local)groB-tile = serial/LVCB (note (wgB*MTB) will be added to SRD) */
/* v3 = groB-unroll = serial%LVCB */
v_lshrrev_b32 v2, 1, v[vgprSerial]                 // v2 = v[vgprSerial] / 2
v_and_b32 v3, 1, v[vgprSerial]                     // v3 = v[vgprSerial] % 2
/* gro-unroll *= glvw */
v_lshlrev_b32 v3, 2, v3                            // v3 = v3 * 4


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



/* global read addresses: branch a */



/* global read addresses: branch b */



/* global read addresses: final offsets a */

GLOBAL_OFFSET_A vgprGlobalReadOffsetA+0,  0,  1, 4 // gROA_0_0_0_0


/* global read addresses: final offsets b */

GLOBAL_OFFSET_B vgprGlobalReadOffsetB+0,  3,  2, 4 // gROB_0_0_0_0
s_mul_i32 s[sgprScalarGlobalReadOffsetB+0], s[sgprStrideB1J], 64 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetB+0], s[sgprScalarGlobalReadOffsetB+0], 0x2 // scalar offset *= bytes/element


/* global read addresses: addresses a */

/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s69, s[sgprWorkGroup0], 64            // WorkGroup[01] * MT
s_mul_i32 s68, s[sgprWorkGroup0], 64               // WorkGroup[01] * MT
s_sub_u32 s[sgprShadowLimitA+0], s[sgprTensor2dSizeA], s68 // sub tileStart
s_subb_u32 s[sgprShadowLimitA+1], s[sgprTensor2dSizeA+1], s69 // sub tileStart
s_lshl_b64 s[sgprShadowLimitA:sgprShadowLimitA+1], s[sgprShadowLimitA:sgprShadowLimitA+1], 0x2 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_mul_hi_u32 s67, s[sgprStrideAK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s66, s[sgprStrideAK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s68, s68, s66                            // accum wg term to tilestart
s_addc_u32 s69, s69, s67                           // accum wg term to tilestart
s_lshl_b64 s[68:69], s[68:69], 2                   // tileStart *= BPE
s_add_u32 s[sgprSrdA+0], s[sgprAddressA+0], s68    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprAddressA+1], s69   // SRD base = Address+ tileStart1
s_sub_u32 s[sgprSrdA+0], s[sgprSrdA+0], 16         // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0         // pre-pad to make room for possible pointer shift
s_mov_b32 s[sgprSrdA+3], Srd127_96                 // Set bits 127_96 in SRD


/* global read addresses: addresses b */

/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s69, s[sgprWorkGroup1], 128           // WorkGroup[01] * MT
s_mul_i32 s68, s[sgprWorkGroup1], 128              // WorkGroup[01] * MT
s_mul_hi_u32 s69, s68, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s68, s68, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_sub_u32 s[sgprShadowLimitB+0], s[sgprTensor2dSizeB], s68 // sub tileStart
s_subb_u32 s[sgprShadowLimitB+1], s[sgprTensor2dSizeB+1], s69 // sub tileStart
s_lshl_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], 0x2 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_mul_hi_u32 s67, s[sgprStrideBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s66, s[sgprStrideBK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s68, s68, s66                            // accum wg term to tilestart
s_addc_u32 s69, s69, s67                           // accum wg term to tilestart
s_lshl_b64 s[68:69], s[68:69], 2                   // tileStart *= BPE
s_add_u32 s[sgprSrdB+0], s[sgprAddressB+0], s68    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprAddressB+1], s69   // SRD base = Address+ tileStart1
s_sub_u32 s[sgprSrdB+0], s[sgprSrdB+0], 16         // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0         // pre-pad to make room for possible pointer shift
s_mov_b32 s[sgprSrdB+3], Srd127_96                 // Set bits 127_96 in SRD


/* global read addresses: increments a */

s_mul_i32 s[sgprGlobalReadIncsA+0], DepthU*BpeA, s[sgprStrideAL] // incrA unrollIdx)


/* global read addresses: increments b */

s_mov_b32 s[sgprGlobalReadIncsB+0], DepthU*BpeB    // incrB (unrollIdx)


/******************************************/
/* Local Write Addresses                  */
/******************************************/

/* lwaTileAssignmentA = v0 */

/* lwaTileAssignmentB = v2 */

/* lwaUnrollAssignmentA = v1 */

/* lwaUnrollAssignmentB = v3 */


/* local write addresses: first offset a */

v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x40, v1     // lwAL**(MTA + PAD)
_v_add_lshl_u32 v[vgprLocalWriteAddrA], v0, v[vgprLocalWriteAddrA], 0x2 // lwFOA = (lwAA + lwAL*(MT0I+PAD))*bpe


/* local write addresses: first offset b */

v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x80, v3     // lwBL**(MTB + PAD)
_v_add_lshl_u32 v[vgprLocalWriteAddrB], v2, v[vgprLocalWriteAddrB], 0x2 // lwFOB = (lwBB + lwBL*(MT1J+PAD))*bpe
_v_add_co_u32 v[vgprLocalWriteAddrB], vcc, 0x800, v[vgprLocalWriteAddrB] // lwFOB = lwB1J + lwBL*MT1J + LDS_OFFSET_B=512*4







/* declare loop num iterations */


s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 3 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 8
s_mov_b32 s[sgprOrigLoopCounter], s[sgprLoopCounterL] // copy loop counter

s_and_b32 s[sgprStaggerUIter], s[sgprOrigStaggerUIter], s[sgprWorkGroup0] // Compute actual stagger start for this tile
s_lshl_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], 3 // shift by StaggerUStride


/* SRDs += (StaggerUIter) * GlobalReadIncsA+0 */
s_mul_hi_i32 s67, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_i32 s66, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_hi_i32 s[sgprWrapUA+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUA+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0], s[sgprWrapUA+0] // remove one iteration
s_subb_u32 s[sgprWrapUA+1], 0, s[sgprWrapUA+1]     // remove one iteration
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s66        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], s67      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32


/* SRDs += (StaggerUIter) * GlobalReadIncsB+0 */
s_mul_hi_i32 s67, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_i32 s66, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_hi_i32 s[sgprWrapUB+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUB+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0], s[sgprWrapUB+0] // remove one iteration
s_subb_u32 s[sgprWrapUB+1], 0, s[sgprWrapUB+1]     // remove one iteration
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s66        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], s67      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32
s_add_u32 s[sgprStaggerUIter], s[sgprStaggerUIter], 2 // Subtract (PGR-1); StaggerUIter now contains target iteration to wrap

/* local read addresses: init pointers a */


/* localReadInitPointers */

/* local read addresses: init pointers b */


/* localReadInitPointers */


/* prefetch: global -> local */

s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?
s_cbranch_scc1 ShadowInitStart_9                   // skip to ShadowInitStart iter b/c numIter==0


buffer_load_dwordx4 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_0_0


buffer_load_dwordx4 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0], offen offset:0 // G -> Reg 0_0_1_0


/* global read inc A loopL */
s_add_u32 s68, s[sgprLoopCounterL], 1              // remove pf(1)
s_cmp_eq_u32 s[sgprStaggerUIter], s68              // Is this wrapIter? (pf)
s_cselect_b32 s66, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s67, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s66        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], s67      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s68, s[sgprLoopCounterL], 1              // remove pf(1)
s_cmp_eq_u32 s[sgprStaggerUIter], s68              // Is this wrapIter? (pf)
s_cselect_b32 s66, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s67, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s66        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], s67      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32


/******************************************/
/* End setupNewTile, isPap=False             */
/******************************************/

ShadowInitStart_9: //

s_mov_b32 s[sgprSrdD+0], s[sgprAddressD+0]         // init SRD base address (lower)
s_mov_b32 s[sgprSrdD+1], s[sgprAddressD+1]         // init SRD base address (upper) + other fields
s_mov_b32 s[sgprSrdD+2], 0x80000000                //
s_mov_b32 s[sgprSrdD+3], Srd127_96                 // Set bits 127_96 in post-loop SRD

s_mov_b32 s[sgprSrdC+0], s[sgprAddressC+0]         // init SRD base address (lower)
s_mov_b32 s[sgprSrdC+1], s[sgprAddressC+1]         // init SRD base address (upper) + other fields
s_mov_b32 s[sgprSrdC+2], 0x80000000                //
s_mov_b32 s[sgprSrdC+3], Srd127_96                 // Set bits 127_96 in post-loop SRD


s_mul_i32 s68, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_mul_hi_u32 s67, s68, s[sgprStrideC1J]            // CScale s68 by Stride
s_mul_i32 s66, s68, s[sgprStrideC1J]               // CScale s68 by Stride
s_lshl_b64 s[66:67], s[66:67], 2                   // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s66        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s67       // add hi to SRD
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s67       // add hi to SRD

s_mul_hi_u32 s67, s[sgprWorkGroup2], s[sgprStrideCK] // CScale s[sgprWorkGroup2] by Stride
s_mul_i32 s66, s[sgprWorkGroup2], s[sgprStrideCK]  // CScale s[sgprWorkGroup2] by Stride
s_lshl_b64 s[66:67], s[66:67], 2                   // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s66        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s67       // add hi to SRD
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s67       // add hi to SRD



/* initC: remove C-tile 0-64 from pool */

/* initC: remove AB-tile 64-96 from pool */
v_mov_b32 v[vgprValuC+0], 0x0                      // initC
v_mov_b32 v[vgprValuC+1], 0x0                      // initC
v_mov_b32 v[vgprValuC+2], 0x0                      // initC
v_mov_b32 v[vgprValuC+3], 0x0                      // initC
v_mov_b32 v[vgprValuC+4], 0x0                      // initC
v_mov_b32 v[vgprValuC+5], 0x0                      // initC
v_mov_b32 v[vgprValuC+6], 0x0                      // initC
v_mov_b32 v[vgprValuC+7], 0x0                      // initC
v_mov_b32 v[vgprValuC+8], 0x0                      // initC
v_mov_b32 v[vgprValuC+9], 0x0                      // initC
v_mov_b32 v[vgprValuC+10], 0x0                     // initC
v_mov_b32 v[vgprValuC+11], 0x0                     // initC
v_mov_b32 v[vgprValuC+12], 0x0                     // initC
v_mov_b32 v[vgprValuC+13], 0x0                     // initC
v_mov_b32 v[vgprValuC+14], 0x0                     // initC
v_mov_b32 v[vgprValuC+15], 0x0                     // initC
v_mov_b32 v[vgprValuC+16], 0x0                     // initC
v_mov_b32 v[vgprValuC+17], 0x0                     // initC
v_mov_b32 v[vgprValuC+18], 0x0                     // initC
v_mov_b32 v[vgprValuC+19], 0x0                     // initC
v_mov_b32 v[vgprValuC+20], 0x0                     // initC
v_mov_b32 v[vgprValuC+21], 0x0                     // initC
v_mov_b32 v[vgprValuC+22], 0x0                     // initC
v_mov_b32 v[vgprValuC+23], 0x0                     // initC
v_mov_b32 v[vgprValuC+24], 0x0                     // initC
v_mov_b32 v[vgprValuC+25], 0x0                     // initC
v_mov_b32 v[vgprValuC+26], 0x0                     // initC
v_mov_b32 v[vgprValuC+27], 0x0                     // initC
v_mov_b32 v[vgprValuC+28], 0x0                     // initC
v_mov_b32 v[vgprValuC+29], 0x0                     // initC
v_mov_b32 v[vgprValuC+30], 0x0                     // initC
v_mov_b32 v[vgprValuC+31], 0x0                     // initC
v_mov_b32 v[vgprValuC+32], 0x0                     // initC
v_mov_b32 v[vgprValuC+33], 0x0                     // initC
v_mov_b32 v[vgprValuC+34], 0x0                     // initC
v_mov_b32 v[vgprValuC+35], 0x0                     // initC
v_mov_b32 v[vgprValuC+36], 0x0                     // initC
v_mov_b32 v[vgprValuC+37], 0x0                     // initC
v_mov_b32 v[vgprValuC+38], 0x0                     // initC
v_mov_b32 v[vgprValuC+39], 0x0                     // initC
v_mov_b32 v[vgprValuC+40], 0x0                     // initC
v_mov_b32 v[vgprValuC+41], 0x0                     // initC
v_mov_b32 v[vgprValuC+42], 0x0                     // initC
v_mov_b32 v[vgprValuC+43], 0x0                     // initC
v_mov_b32 v[vgprValuC+44], 0x0                     // initC
v_mov_b32 v[vgprValuC+45], 0x0                     // initC
v_mov_b32 v[vgprValuC+46], 0x0                     // initC
v_mov_b32 v[vgprValuC+47], 0x0                     // initC
v_mov_b32 v[vgprValuC+48], 0x0                     // initC
v_mov_b32 v[vgprValuC+49], 0x0                     // initC
v_mov_b32 v[vgprValuC+50], 0x0                     // initC
v_mov_b32 v[vgprValuC+51], 0x0                     // initC
v_mov_b32 v[vgprValuC+52], 0x0                     // initC
v_mov_b32 v[vgprValuC+53], 0x0                     // initC
v_mov_b32 v[vgprValuC+54], 0x0                     // initC
v_mov_b32 v[vgprValuC+55], 0x0                     // initC
v_mov_b32 v[vgprValuC+56], 0x0                     // initC
v_mov_b32 v[vgprValuC+57], 0x0                     // initC
v_mov_b32 v[vgprValuC+58], 0x0                     // initC
v_mov_b32 v[vgprValuC+59], 0x0                     // initC
v_mov_b32 v[vgprValuC+60], 0x0                     // initC
v_mov_b32 v[vgprValuC+61], 0x0                     // initC
v_mov_b32 v[vgprValuC+62], 0x0                     // initC
v_mov_b32 v[vgprValuC+63], 0x0                     // initC

s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?

/* after InitC, skip to end of prefetch last iter if numIter==0 */
s_cbranch_scc0 label_NoBranch_10                   // Only branch on scc1
s_getpc_B64 s[66:67]                               // addr of next instr
s_add_u32 s68, PrefetchGlobalLastIterEnd_4, 0x4    // target branch offset
s_add_u32 s66, s66, s68                            // add target branch offset
s_addc_u32 s67, 0, s67                             // add high and carry
s_setpc_b64 s[66:67]                               // branch to PrefetchGlobalLastIterEnd_4
label_NoBranch_10:

s_waitcnt vmcnt(0)                                 // 8wait for global read


/* local write a */
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0

/* local write b */
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+0] offset:0 // lwoB_0_0_0_0 = (0 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+1] offset:512 // lwoB_0_1_0_0 = (1 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 512
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+2] offset:1024 // lwoB_0_2_0_0 = (2 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 1024
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+3] offset:1536 // lwoB_0_3_0_0 = (3 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 1536
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+4] offset:256 // lwoB_0_0_1_0 = (0 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 256
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+5] offset:768 // lwoB_0_1_1_0 = (1 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 768
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+6] offset:1280 // lwoB_0_2_1_0 = (2 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 1280
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+7] offset:1792 // lwoB_0_3_1_0 = (3 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 1792


/* local write swap a */


/* (EPS=1) local write swap internal offset -> 8192 */


/* local write swap b */


/* (EPS=1) local write swap internal offset -> 8192 */



s_waitcnt lgkmcnt(0)                               // 0prefetch wait for local write

s_barrier //


/* local read prefetch a */

ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=8 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=8 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0


/* local read prefetch b */

ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0


/* local read inc a */

/* N/A, lro->64 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */


/* local read inc b */

/* N/A, lro->128 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */



/******************************************/
/* Unrolled Loop(s) - Begin               */
/******************************************/

openLoopL_11:
s_cmp_le_u32 s[sgprLoopCounterL], 0x1              // LoopCounterL < EndCounter
s_cbranch_scc1 LoopEndL_2                          // don't enter LoopL
LoopBeginL_1:


/******************************************/
/* Unrolled Loop 1/2 - Begin              */
/******************************************/

label_0012: // LoopCopy1





/* iter 0 */


/* local read a */
ds_read_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:256 // L -> Reg lro=64 swapByteOffset=0 ti=8 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:384 // L -> Reg lro=64 swapByteOffset=0 ti=8 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
buffer_load_dwordx4 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_0_0

/* local read b */
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:512 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:768 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read increment a */
/* N/A, lro->128 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->256 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=4, new=4 newLW=0 newLR=0
MAC_8x8_X0

/* iter 1 */


/* local read a */
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:512 // L -> Reg lro=128 swapByteOffset=0 ti=8 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:640 // L -> Reg lro=128 swapByteOffset=0 ti=8 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
buffer_load_dwordx4 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0], offen offset:0 // G -> Reg 0_0_1_0

/* local read b */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:1024 // L -> Reg lro=256 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:1280 // L -> Reg lro=256 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->192 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->384 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=4, new=4 newLW=0 newLR=0
MAC_8x8_X1

/* iter 2 */


/* local read a */
ds_read_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:768 // L -> Reg lro=192 swapByteOffset=0 ti=8 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:896 // L -> Reg lro=192 swapByteOffset=0 ti=8 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s66, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s67, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s66        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], s67      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

/* local read b */
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:1536 // L -> Reg lro=384 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:1792 // L -> Reg lro=384 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read increment a */
/* N/A, lro->256 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->512 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=4, new=4 newLW=0 newLR=0
MAC_8x8_X0

/* iter 3 */


/* local read a */
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:1024 // L -> Reg lro=256 swapByteOffset=0 ti=8 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:1152 // L -> Reg lro=256 swapByteOffset=0 ti=8 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s66, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s67, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s66        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], s67      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32

/* local read b */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:2048 // L -> Reg lro=512 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:2304 // L -> Reg lro=512 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->320 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->640 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=4, new=4 newLW=0 newLR=0
MAC_8x8_X1

/* iter 4 */


/* local read a */
ds_read_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:1280 // L -> Reg lro=320 swapByteOffset=0 ti=8 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:1408 // L -> Reg lro=320 swapByteOffset=0 ti=8 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read b */
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:2560 // L -> Reg lro=640 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:2816 // L -> Reg lro=640 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read increment a */
/* N/A, lro->384 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->768 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(2)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:8192 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 8192
s_waitcnt lgkmcnt(5)                               // wait for prior local read local write old=4, new=5 newLW=0 newLR=0
MAC_8x8_X0

/* iter 5 */


/* local read a */
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:1536 // L -> Reg lro=384 swapByteOffset=0 ti=8 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:1664 // L -> Reg lro=384 swapByteOffset=0 ti=8 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:3072 // L -> Reg lro=768 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:3328 // L -> Reg lro=768 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->448 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->896 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
/* sched write - iter 5 writesPerItem=4 */
s_waitcnt vmcnt(1)                                 // wait for global read before writing to local
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+0] offset:8192 // lwoB_0_0_0_0 = (0 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 8192
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+1] offset:8704 // lwoB_0_1_0_0 = (1 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 8704
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+2] offset:9216 // lwoB_0_2_0_0 = (2 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 9216
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+3] offset:9728 // lwoB_0_3_0_0 = (3 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 9728
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=4, new=8 newLW=0 newLR=0
MAC_8x8_X1

/* iter 6 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */


/* local read a */
ds_read_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:1792 // L -> Reg lro=448 swapByteOffset=0 ti=8 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:1920 // L -> Reg lro=448 swapByteOffset=0 ti=8 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read b */
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:3584 // L -> Reg lro=896 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:3840 // L -> Reg lro=896 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
/* sched write - iter 6 writesPerItem=4 */
s_waitcnt vmcnt(0)                                 // wait for global read before writing to local
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+4] offset:8448 // lwoB_0_0_1_0 = (0 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 8448
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+5] offset:8960 // lwoB_0_1_1_0 = (1 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 8960
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+6] offset:9472 // lwoB_0_2_1_0 = (2 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 9472
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+7] offset:9984 // lwoB_0_3_1_0 = (3 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 9984

/* local write swap offsets a */

/* (EPS=1) local write swap internal offset -> 0 */

/* local write swap offsets b */

/* (EPS=1) local write swap internal offset -> 0 */

/* local read swap offsets a */

/* local read swap internal offset -> 8192 */

/* local read swap offsets b */

/* local read swap internal offset -> 8192 */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=13, new=8 newLW=0 newLR=0
MAC_8x8_X0

/* iter 7 */

s_waitcnt lgkmcnt(0)                               // 3wait for local write
s_barrier //

/* local read a */
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:8192 // L -> Reg lro=0 swapByteOffset=8192 ti=8 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:8320 // L -> Reg lro=0 swapByteOffset=8192 ti=8 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:8192 // L -> Reg lro=0 swapByteOffset=8192 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:8448 // L -> Reg lro=0 swapByteOffset=8192 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->64 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->128 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=4, new=4 newLW=0 newLR=0
MAC_8x8_X1



/******************************************/
/* Unrolled Loop - End 1/2                */
/******************************************/


/* closeLoop loopL finalLoop=0 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x1              // counterL==1
s_cbranch_scc1 LoopEndL_oddexit_3                  // exit LoopL


/******************************************/
/* Unrolled Loop 2/2 - Begin              */
/******************************************/

label_0013: // LoopCopy2





/* iter 0 */


/* local read a */
ds_read_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:8448 // L -> Reg lro=64 swapByteOffset=8192 ti=8 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:8576 // L -> Reg lro=64 swapByteOffset=8192 ti=8 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
buffer_load_dwordx4 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_0_0

/* local read b */
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:8704 // L -> Reg lro=128 swapByteOffset=8192 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:8960 // L -> Reg lro=128 swapByteOffset=8192 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read increment a */
/* N/A, lro->128 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->256 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=4, new=4 newLW=0 newLR=0
MAC_8x8_X0

/* iter 1 */


/* local read a */
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:8704 // L -> Reg lro=128 swapByteOffset=8192 ti=8 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:8832 // L -> Reg lro=128 swapByteOffset=8192 ti=8 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
buffer_load_dwordx4 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0], offen offset:0 // G -> Reg 0_0_1_0

/* local read b */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:9216 // L -> Reg lro=256 swapByteOffset=8192 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:9472 // L -> Reg lro=256 swapByteOffset=8192 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->192 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->384 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=4, new=4 newLW=0 newLR=0
MAC_8x8_X1

/* iter 2 */


/* local read a */
ds_read_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:8960 // L -> Reg lro=192 swapByteOffset=8192 ti=8 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:9088 // L -> Reg lro=192 swapByteOffset=8192 ti=8 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s66, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s67, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s66        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], s67      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

/* local read b */
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:9728 // L -> Reg lro=384 swapByteOffset=8192 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:9984 // L -> Reg lro=384 swapByteOffset=8192 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read increment a */
/* N/A, lro->256 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->512 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=4, new=4 newLW=0 newLR=0
MAC_8x8_X0

/* iter 3 */


/* local read a */
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:9216 // L -> Reg lro=256 swapByteOffset=8192 ti=8 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:9344 // L -> Reg lro=256 swapByteOffset=8192 ti=8 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s66, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s67, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s66        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], s67      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32

/* local read b */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:10240 // L -> Reg lro=512 swapByteOffset=8192 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:10496 // L -> Reg lro=512 swapByteOffset=8192 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->320 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->640 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=4, new=4 newLW=0 newLR=0
MAC_8x8_X1

/* iter 4 */


/* local read a */
ds_read_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:9472 // L -> Reg lro=320 swapByteOffset=8192 ti=8 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:9600 // L -> Reg lro=320 swapByteOffset=8192 ti=8 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read b */
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:10752 // L -> Reg lro=640 swapByteOffset=8192 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:11008 // L -> Reg lro=640 swapByteOffset=8192 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read increment a */
/* N/A, lro->384 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->768 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(2)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0
s_waitcnt lgkmcnt(5)                               // wait for prior local read local write old=4, new=5 newLW=0 newLR=0
MAC_8x8_X0

/* iter 5 */


/* local read a */
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:9728 // L -> Reg lro=384 swapByteOffset=8192 ti=8 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:9856 // L -> Reg lro=384 swapByteOffset=8192 ti=8 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:11264 // L -> Reg lro=768 swapByteOffset=8192 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:11520 // L -> Reg lro=768 swapByteOffset=8192 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->448 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->896 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
/* sched write - iter 5 writesPerItem=4 */
s_waitcnt vmcnt(1)                                 // wait for global read before writing to local
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+0] offset:0 // lwoB_0_0_0_0 = (0 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+1] offset:512 // lwoB_0_1_0_0 = (1 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 512
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+2] offset:1024 // lwoB_0_2_0_0 = (2 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 1024
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+3] offset:1536 // lwoB_0_3_0_0 = (3 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 1536
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=4, new=8 newLW=0 newLR=0
MAC_8x8_X1

/* iter 6 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */


/* local read a */
ds_read_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:9984 // L -> Reg lro=448 swapByteOffset=8192 ti=8 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:10112 // L -> Reg lro=448 swapByteOffset=8192 ti=8 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read b */
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:11776 // L -> Reg lro=896 swapByteOffset=8192 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:12032 // L -> Reg lro=896 swapByteOffset=8192 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
/* sched write - iter 6 writesPerItem=4 */
s_waitcnt vmcnt(0)                                 // wait for global read before writing to local
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+4] offset:256 // lwoB_0_0_1_0 = (0 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 256
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+5] offset:768 // lwoB_0_1_1_0 = (1 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 768
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+6] offset:1280 // lwoB_0_2_1_0 = (2 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 1280
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+7] offset:1792 // lwoB_0_3_1_0 = (3 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 1792

/* local write swap offsets a */

/* (EPS=1) local write swap internal offset -> 8192 */

/* local write swap offsets b */

/* (EPS=1) local write swap internal offset -> 8192 */

/* local read swap offsets a */

/* local read swap internal offset -> 0 */

/* local read swap offsets b */

/* local read swap internal offset -> 0 */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=13, new=8 newLW=0 newLR=0
MAC_8x8_X0

/* iter 7 */

s_waitcnt lgkmcnt(0)                               // 3wait for local write
s_barrier //

/* local read a */
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=8 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=8 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->64 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->128 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=4, new=4 newLW=0 newLR=0
MAC_8x8_X1



/******************************************/
/* Unrolled Loop - End 2/2 (final)        */
/******************************************/


/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x1              // counterL==1
s_cbranch_scc0 LoopBeginL_1                        // restart LoopL
s_branch LoopEndL_2                                // exit unroll loopL (and skip oddexit)
LoopEndL_oddexit_3: // unroll loop odditer exit

/* Select high bank of LDS */
v_xor_b32 v[vgprLocalReadAddrA], 0x2000, v[vgprLocalReadAddrA] // swap Red Blk
v_xor_b32 v[vgprLocalReadAddrB], 0x2000, v[vgprLocalReadAddrB] // swap Red Blk
LoopEndL_2:


/******************************************/
/* Opt. NoLoadLoop - Begin                  */
/******************************************/

s_cmpk_eq_u32 s[sgprBeta], 0x0                     // Beta == 0
s_cbranch_scc0 OptNLL_End_14                       // Branch if Beta is not zero

s_cmp_eq_u32 s[sgprAlpha], 1.0                     // Alpha == 1.0 ?
s_cbranch_scc0 OptNLL_End_14                       // branch if alpha != 1

s_and_b32 s66, 63, s[sgprSizeI]                    // s66 = s[sgprSizeI] % 64
s_add_u32 s68, -0x1, s[sgprNumWorkGroups0]         //
s_cmp_ge_u32 s[sgprWorkGroup0], s68                // wg0 >= nwg0-1 ?
s_cselect_b32 s66, s66, 0                          // set rMT0
s_cmpk_gt_u32 s66, 0x0                             // rMT0 > 0
s_cbranch_scc1 OptNLL_End_14                       // jump if edges required
s_and_b32 s66, 127, s[sgprSizeJ]                   // s66 = s[sgprSizeJ] % 128
s_add_u32 s68, -0x1, s[sgprNumWorkGroups1]         //
s_cmp_ge_u32 s[sgprWorkGroup1], s68                // wg1 >= nwg1-1
s_cselect_b32 s66, s66, 0                          // set rMT1
s_cmpk_gt_u32 s66, 0x0                             // rMT1 > 0
s_cbranch_scc1 OptNLL_End_14                       // jump if edges required

s_and_b32 s67, 7, s[sgprSizesSum+0]                // s67 = s[sgprSizesSum+0] % 8
s_cmp_eq_u32 s67, 0x0                              // numIterL == 0
s_cbranch_scc0 OptNLL_End_14                       // skip if tail loop required



/* iter 0 (last unrolled loop) */


/* local read a */
ds_read_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:256 // L -> Reg lro=64 swapByteOffset=0 ti=8 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:384 // L -> Reg lro=64 swapByteOffset=0 ti=8 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read b */
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:512 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:768 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read increment a */
/* N/A, lro->128 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->256 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(4)                               // 7wait for local read old=4, new=4 newLW=0 newLR=0
MAC_8x8_X0

/* iter 1 (last unrolled loop) */


/* local read a */
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:512 // L -> Reg lro=128 swapByteOffset=0 ti=8 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:640 // L -> Reg lro=128 swapByteOffset=0 ti=8 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:1024 // L -> Reg lro=256 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:1280 // L -> Reg lro=256 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->192 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->384 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(4)                               // 7wait for local read old=4, new=4 newLW=0 newLR=0
MAC_8x8_X1

/* iter 2 (last unrolled loop) */


/* local read a */
ds_read_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:768 // L -> Reg lro=192 swapByteOffset=0 ti=8 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:896 // L -> Reg lro=192 swapByteOffset=0 ti=8 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read b */
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:1536 // L -> Reg lro=384 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:1792 // L -> Reg lro=384 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read increment a */
/* N/A, lro->256 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->512 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(4)                               // 7wait for local read old=4, new=4 newLW=0 newLR=0
MAC_8x8_X0

/* iter 3 (last unrolled loop) */


/* local read a */
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:1024 // L -> Reg lro=256 swapByteOffset=0 ti=8 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:1152 // L -> Reg lro=256 swapByteOffset=0 ti=8 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:2048 // L -> Reg lro=512 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:2304 // L -> Reg lro=512 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->320 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->640 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(4)                               // 7wait for local read old=4, new=4 newLW=0 newLR=0
MAC_8x8_X1

/* iter 4 (last unrolled loop) */


/* local read a */
ds_read_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:1280 // L -> Reg lro=320 swapByteOffset=0 ti=8 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:1408 // L -> Reg lro=320 swapByteOffset=0 ti=8 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read b */
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:2560 // L -> Reg lro=640 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:2816 // L -> Reg lro=640 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read increment a */
/* N/A, lro->384 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->768 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(4)                               // 7wait for local read old=4, new=4 newLW=0 newLR=0
MAC_8x8_X0

/* iter 5 (last unrolled loop) */


/* local read a */
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:1536 // L -> Reg lro=384 swapByteOffset=0 ti=8 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:1664 // L -> Reg lro=384 swapByteOffset=0 ti=8 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:3072 // L -> Reg lro=768 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:3328 // L -> Reg lro=768 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->448 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->896 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(4)                               // 7wait for local read old=4, new=4 newLW=0 newLR=0
MAC_8x8_X1

/* iter 6 (last unrolled loop) */


/* local read a */
ds_read_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:1792 // L -> Reg lro=448 swapByteOffset=0 ti=8 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:1920 // L -> Reg lro=448 swapByteOffset=0 ti=8 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read b */
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:3584 // L -> Reg lro=896 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:3840 // L -> Reg lro=896 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
s_waitcnt lgkmcnt(4)                               // 7wait for local read old=4, new=4 newLW=0 newLR=0
MAC_8x8_X0

/* iter 7 (last unrolled loop) */

s_waitcnt lgkmcnt(0)                               // 7wait for local read old=0, new=0 newLW=0 newLR=0
MAC_8x8_X1
/* Stores for OptNLL */
Summation_End_15:
/* endSummation: add vgpr [64...112) to pool */
.set NumFullBlocks, UNDEF
.set WgmRemainder1, UNDEF
.set MagicNumberWgmRemainder1, UNDEF
.set ShadowLimitB, UNDEF
.set WrapUA, UNDEF
.set WrapUB, UNDEF
.set GlobalReadIncsA, UNDEF
.set GlobalReadIncsB, UNDEF
.set ScalarGlobalReadOffsetB, UNDEF
.set WaveId, UNDEF
/* computeStoreVgprs */
v_lshrrev_b32 v65, 3, v[vgprSerial]                // v65 = v[vgprSerial] / 8
v_and_b32 v64, 7, v[vgprSerial]                    // v64 = v[vgprSerial] % 8
v_lshlrev_b32 v64, 2, v64                          // v64 = v64 * 4
v_lshlrev_b32 v65, 2, v65                          // v65 = v65 * 4
v_mul_lo_u32 v66, v65, s[sgprStrideC1J]            // rowStart vgpr

s_mul_i32 s54, 0x40, s[sgprWorkGroup0]             // s54 = wg0*MT0
_v_add_co_u32 v64, vcc, s54, v64                   // coord0 = tid0*VW + wg0*MT0
s_mul_i32 s56, 0x80, s[sgprWorkGroup1]             // <- wg1*MT1
_v_add_co_u32 v65, vcc, s56, v65                   // coord1 = tid1*VW + wg1*MT1
GW_B0_E0_18:

/* edge=0, allocate 2 sgpr. perBatch=2 perElement=0 elementsPerBatch=16 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSharedMask=1 optSrdIncForRow=1 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,1,0,0:vw4); (0,0,1,0:vw4); (0,1,1,0:vw4); (0,0,2,0:vw4); (0,1,2,0:vw4); (0,0,3,0:vw4); (0,1,3,0:vw4); (1,0,0,0:vw4); (1,1,0,0:vw4); (1,0,1,0:vw4); (1,1,1,0:vw4); (1,0,2,0:vw4); (1,1,2,0:vw4); (1,0,3,0:vw4); (1,1,3,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
_v_add_lshl_u32 v69, v66, v64, 0x2                 // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=64, coord0Vgpr=64
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
/* (d1,vc1,d0,vc0)=(0,2,1,0) */
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
/* (d1,vc1,d0,vc0)=(0,3,1,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
/* (d1,vc1,d0,vc0)=(1,1,0,0) */
/* (d1,vc1,d0,vc0)=(1,1,1,0) */
/* (d1,vc1,d0,vc0)=(1,2,0,0) */
/* (d1,vc1,d0,vc0)=(1,2,1,0) */
/* (d1,vc1,d0,vc0)=(1,3,0,0) */
/* (d1,vc1,d0,vc0)=(1,3,1,0) */

/* apply mask, calc new C and issue writes */
buffer_store_dwordx4 v[0:3], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dwordx4 v[4:7], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
s_lshl_b32  s54, s[sgprStrideD1J], 2               // incToNextRow: Scale by BPE
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[8:11], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dwordx4 v[12:15], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
s_lshl_b32  s54, s[sgprStrideD1J], 2               // incToNextRow: Scale by BPE
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[16:19], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dwordx4 v[20:23], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
s_lshl_b32  s54, s[sgprStrideD1J], 2               // incToNextRow: Scale by BPE
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[24:27], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dwordx4 v[28:31], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
s_mul_i32 s54, s[sgprStrideD1J], 244               // scale StrideD *= numRows(61) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[32:35], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dwordx4 v[36:39], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
s_lshl_b32  s54, s[sgprStrideD1J], 2               // incToNextRow: Scale by BPE
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[40:43], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dwordx4 v[44:47], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
s_lshl_b32  s54, s[sgprStrideD1J], 2               // incToNextRow: Scale by BPE
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[48:51], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dwordx4 v[52:55], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
s_lshl_b32  s54, s[sgprStrideD1J], 2               // incToNextRow: Scale by BPE
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[56:59], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dwordx4 v[60:63], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
s_branch label_GW_End_20                           // jump to end
label_GW_End_20:

s_endpgm                                           // Kernel End
OptNLL_End_14:


/******************************************/
/* Ord. NoLoadLoop - Begin                  */
/******************************************/




/* iter 0 (last unrolled loop) */


/* local read a */
ds_read_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:256 // L -> Reg lro=64 swapByteOffset=0 ti=8 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:384 // L -> Reg lro=64 swapByteOffset=0 ti=8 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read b */
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:512 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:768 // L -> Reg lro=128 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read increment a */
/* N/A, lro->128 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->256 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(4)                               // 7wait for local read old=4, new=4 newLW=0 newLR=0
MAC_8x8_X0

/* iter 1 (last unrolled loop) */


/* local read a */
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:512 // L -> Reg lro=128 swapByteOffset=0 ti=8 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:640 // L -> Reg lro=128 swapByteOffset=0 ti=8 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:1024 // L -> Reg lro=256 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:1280 // L -> Reg lro=256 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->192 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->384 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(4)                               // 7wait for local read old=4, new=4 newLW=0 newLR=0
MAC_8x8_X1

/* iter 2 (last unrolled loop) */


/* local read a */
ds_read_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:768 // L -> Reg lro=192 swapByteOffset=0 ti=8 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:896 // L -> Reg lro=192 swapByteOffset=0 ti=8 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read b */
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:1536 // L -> Reg lro=384 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:1792 // L -> Reg lro=384 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read increment a */
/* N/A, lro->256 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->512 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(4)                               // 7wait for local read old=4, new=4 newLW=0 newLR=0
MAC_8x8_X0

/* iter 3 (last unrolled loop) */


/* local read a */
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:1024 // L -> Reg lro=256 swapByteOffset=0 ti=8 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:1152 // L -> Reg lro=256 swapByteOffset=0 ti=8 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:2048 // L -> Reg lro=512 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:2304 // L -> Reg lro=512 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->320 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->640 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(4)                               // 7wait for local read old=4, new=4 newLW=0 newLR=0
MAC_8x8_X1

/* iter 4 (last unrolled loop) */


/* local read a */
ds_read_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:1280 // L -> Reg lro=320 swapByteOffset=0 ti=8 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:1408 // L -> Reg lro=320 swapByteOffset=0 ti=8 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read b */
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:2560 // L -> Reg lro=640 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:2816 // L -> Reg lro=640 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read increment a */
/* N/A, lro->384 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->768 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(4)                               // 7wait for local read old=4, new=4 newLW=0 newLR=0
MAC_8x8_X0

/* iter 5 (last unrolled loop) */


/* local read a */
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:1536 // L -> Reg lro=384 swapByteOffset=0 ti=8 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:1664 // L -> Reg lro=384 swapByteOffset=0 ti=8 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:3072 // L -> Reg lro=768 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:3328 // L -> Reg lro=768 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->448 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */

/* local read increment b */
/* N/A, lro->896 */
/* self.localReadDoCntA 0 self.localReadDoCntB 0 */
s_waitcnt lgkmcnt(4)                               // 7wait for local read old=4, new=4 newLW=0 newLR=0
MAC_8x8_X1

/* iter 6 (last unrolled loop) */


/* local read a */
ds_read_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:1792 // L -> Reg lro=448 swapByteOffset=0 ti=8 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:1920 // L -> Reg lro=448 swapByteOffset=0 ti=8 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0

/* local read b */
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB] offset:3584 // L -> Reg lro=896 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB] offset:3840 // L -> Reg lro=896 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
s_waitcnt lgkmcnt(4)                               // 7wait for local read old=4, new=4 newLW=0 newLR=0
MAC_8x8_X0

/* iter 7 (last unrolled loop) */

s_waitcnt lgkmcnt(0)                               // 7wait for local read old=0, new=0 newLW=0 newLR=0
MAC_8x8_X1
PrefetchGlobalLastIterEnd_4:


/******************************************/
/* Tail Loop                              */
/******************************************/


/* local write reset offsets a */



/* local write reset offsets b */



//numIterL = (((sizeL % LOCAL_DEPTHU) + LOCAL_SPLITU - 1) / LOCAL_SPLITU)
s_and_b32 s[sgprLoopCounterL], 7, s[sgprSizesSum+0] // s[sgprLoopCounterL] = s[sgprSizesSum+0] % 8
s_cmp_eq_u32 s[sgprLoopCounterL], 0x0              // numIterL == 0
s_mov_b32 s[sgprOrigLoopCounter], 0                // repurpose to count each localRead increment
s_cbranch_scc1 SkipTailLoopL_7                     // skip to end of tail loop b/c numIter==0


/* remove stagger offsets for tail loop */

s_sub_i32 s66, 3, s[sgprStaggerUIter]              //
s_mul_hi_i32 s67, s66, s[sgprGlobalReadIncsA+0]    // start offset S in bytes
s_mul_i32 s66, s66, s[sgprGlobalReadIncsA+0]       // start offset S in bytes
s_sub_u32 s66, s66, s[sgprWrapUA]                  // S - WrapU
s_subb_u32 s67, s67, s[sgprWrapUA+1]               // S - WrapU
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s66        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], s67      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

s_sub_i32 s66, 3, s[sgprStaggerUIter]              //
s_mul_hi_i32 s67, s66, s[sgprGlobalReadIncsB+0]    // start offset S in bytes
s_mul_i32 s66, s66, s[sgprGlobalReadIncsB+0]       // start offset S in bytes
s_sub_u32 s66, s66, s[sgprWrapUB]                  // S - WrapU
s_subb_u32 s67, s67, s[sgprWrapUB+1]               // S - WrapU
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s66        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], s67      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32


/* Update M0 for DTLDS */



/* global read a */

/* g2l=0, load component 0 */
buffer_load_dword v[vgprG2LA+0+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
buffer_load_dword v[vgprG2LA+0+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:4 // load one buffer value
/* g2l=0, load component 2 */
buffer_load_dword v[vgprG2LA+0+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:8 // load one buffer value
/* g2l=0, load component 3 */
buffer_load_dword v[vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:12 // load one buffer value


/* Update M0 for DTLDS */



/* global read b */

/* g2l=0, load component 0 */
buffer_load_dword v[vgprG2LB+0+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
buffer_load_dword v[vgprG2LB+0+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:4 // load one buffer value
/* g2l=0, load component 2 */
buffer_load_dword v[vgprG2LB+0+2], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:8 // load one buffer value
/* g2l=0, load component 3 */
buffer_load_dword v[vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:12 // load one buffer value
/* g2l=4, load component 0 */
buffer_load_dword v[vgprG2LB+4+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0], offen offset:0 // load one buffer value
/* g2l=4, load component 1 */
buffer_load_dword v[vgprG2LB+4+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0], offen offset:4 // load one buffer value
/* g2l=4, load component 2 */
buffer_load_dword v[vgprG2LB+4+2], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0], offen offset:8 // load one buffer value
/* g2l=4, load component 3 */
buffer_load_dword v[vgprG2LB+4+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0], offen offset:12 // load one buffer value

s_waitcnt vmcnt(0)                                 // 2wait for global read

s_barrier //




/* local write a */

ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0


/* local write b */

ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+0] offset:0 // lwoB_0_0_0_0 = (0 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+1] offset:512 // lwoB_0_1_0_0 = (1 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 512
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+2] offset:1024 // lwoB_0_2_0_0 = (2 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 1024
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+3] offset:1536 // lwoB_0_3_0_0 = (3 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 1536
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+4] offset:256 // lwoB_0_0_1_0 = (0 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 256
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+5] offset:768 // lwoB_0_1_1_0 = (1 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 768
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+6] offset:1280 // lwoB_0_2_1_0 = (2 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 1280
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+7] offset:1792 // lwoB_0_3_1_0 = (3 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 1792


/* Recalc local read offsets */


s_waitcnt lgkmcnt(0)                               // 5wait for local write

s_barrier //


/* local read reset offsets a */


/* localReadResetOffsets */
/* handled internally */
v_and_b32 v[vgprLocalReadAddrA], 0x1fff, v[vgprLocalReadAddrA] // reset Red,Blk -> Red


/* local read reset offsets b */


/* localReadResetOffsets */
/* handled internally */
v_and_b32 v[vgprLocalReadAddrB], 0x1fff, v[vgprLocalReadAddrB] // reset Red,Blk -> Red


/* local read init pointers a */


/* localReadInitPointers */


/* local read init pointers b */


/* localReadInitPointers */


/* tail loop: macs */

TailLoopBeginL_5:


/* local read a */

ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=8 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=8 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0


/* local read b */

ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0


/* local read inc a */

s_mov_b32 s66, 0x100                               // inc
_v_add_co_u32 v[vgprLocalReadAddrA], vcc, s66, v[vgprLocalReadAddrA] // lrA += 256 (LSU*(MT+PAD)*bpe)


/* local read inc b */

s_mov_b32 s66, 0x200                               // inc
_v_add_co_u32 v[vgprLocalReadAddrB], vcc, s66, v[vgprLocalReadAddrB] // lrB += 512 (LSU*(MT+PAD)*bpe)

s_waitcnt lgkmcnt(0)                               // 4wait for local read

MAC_8x8_X0

/* closeLoop loopL finalLoop=1 tailLoop=1 */
s_sub_i32 s[sgprLoopCounterL], s[sgprLoopCounterL], 0x1 // dec counterL (tailLoop)
s_add_u32 s[sgprOrigLoopCounter], s[sgprOrigLoopCounter], 0x1 // inc counterL
s_cmp_le_i32 s[sgprLoopCounterL], 0x0              // counterL<=0
s_cbranch_scc0 TailLoopBeginL_5                    // restart LoopL
TailLoopEndL_6:

SkipTailLoopL_7:

Summation_End_23:
/* endSummation: add vgpr [64...112) to pool */
.set NumFullBlocks, UNDEF
.set WgmRemainder1, UNDEF
.set MagicNumberWgmRemainder1, UNDEF
.set ShadowLimitB, UNDEF
.set WrapUA, UNDEF
.set WrapUB, UNDEF
.set GlobalReadIncsA, UNDEF
.set GlobalReadIncsB, UNDEF
.set ScalarGlobalReadOffsetB, UNDEF
.set WaveId, UNDEF



/* not-LocalSplitU: global write indices */

/* computeStoreVgprs */
v_lshrrev_b32 v65, 3, v[vgprSerial]                // v65 = v[vgprSerial] / 8
v_and_b32 v64, 7, v[vgprSerial]                    // v64 = v[vgprSerial] % 8
v_lshlrev_b32 v64, 2, v64                          // v64 = v64 * 4
v_lshlrev_b32 v65, 2, v65                          // v65 = v65 * 4
v_mul_lo_u32 v66, v65, s[sgprStrideC1J]            // rowStart vgpr

s_mul_i32 s54, 0x40, s[sgprWorkGroup0]             // s54 = wg0*MT0
_v_add_co_u32 v64, vcc, s54, v64                   // coord0 = tid0*VW + wg0*MT0
s_mul_i32 s56, 0x80, s[sgprWorkGroup1]             // <- wg1*MT1
_v_add_co_u32 v65, vcc, s56, v65                   // coord1 = tid1*VW + wg1*MT1


/* not-LocalSplitU: global write */

s_cmpk_eq_u32 s[sgprBeta], 0x0                     // Beta == 0
s_cbranch_scc0 GW_Beta_38                          // Branch if Beta is not zero

s_and_b32 s54, 63, s[sgprSizeI]                    // s54 = s[sgprSizeI] % 64
s_add_u32 s56, -0x1, s[sgprNumWorkGroups0]         //
s_cmp_ge_u32 s[sgprWorkGroup0], s56                // wg0 >= nwg0-1 ?
s_cselect_b32 s54, s54, 0                          // set rMT0
s_cmpk_gt_u32 s54, 0x0                             // rMT0 > 0
s_cbranch_scc1 GW_B0_E1_29                         // jump if edges required
s_and_b32 s54, 127, s[sgprSizeJ]                   // s54 = s[sgprSizeJ] % 128
s_add_u32 s56, -0x1, s[sgprNumWorkGroups1]         //
s_cmp_ge_u32 s[sgprWorkGroup1], s56                // wg1 >= nwg1-1
s_cselect_b32 s54, s54, 0                          // set rMT1
s_cmpk_gt_u32 s54, 0x0                             // rMT1 > 0
s_cbranch_scc1 GW_B0_E1_29                         // jump if edges required
GW_B0_E0_26:

/* edge=0, allocate 2 sgpr. perBatch=2 perElement=0 elementsPerBatch=16 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSharedMask=1 optSrdIncForRow=1 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,1,0,0:vw4); (0,0,1,0:vw4); (0,1,1,0:vw4); (0,0,2,0:vw4); (0,1,2,0:vw4); (0,0,3,0:vw4); (0,1,3,0:vw4); (1,0,0,0:vw4); (1,1,0,0:vw4); (1,0,1,0:vw4); (1,1,1,0:vw4); (1,0,2,0:vw4); (1,1,2,0:vw4); (1,0,3,0:vw4); (1,1,3,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
_v_add_lshl_u32 v69, v66, v64, 0x2                 // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=64, coord0Vgpr=64
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
/* (d1,vc1,d0,vc0)=(0,2,1,0) */
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
/* (d1,vc1,d0,vc0)=(0,3,1,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
/* (d1,vc1,d0,vc0)=(1,1,0,0) */
/* (d1,vc1,d0,vc0)=(1,1,1,0) */
/* (d1,vc1,d0,vc0)=(1,2,0,0) */
/* (d1,vc1,d0,vc0)=(1,2,1,0) */
/* (d1,vc1,d0,vc0)=(1,3,0,0) */
/* (d1,vc1,d0,vc0)=(1,3,1,0) */

/* rC *= alpha batchEements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 1, 1, 0), (0, 0, 2, 0), (0, 1, 2, 0), (0, 0, 3, 0), (0, 1, 3, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 0, 1, 0), (1, 1, 1, 0), (1, 0, 2, 0), (1, 1, 2, 0), (1, 0, 3, 0), (1, 1, 3, 0)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha
v_mul_f32 v[vgprValuC+1], s[sgprAlpha], v[vgprValuC+1] // *= alpha
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha
v_mul_f32 v[vgprValuC+3], s[sgprAlpha], v[vgprValuC+3] // *= alpha
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // *= alpha
v_mul_f32 v[vgprValuC+5], s[sgprAlpha], v[vgprValuC+5] // *= alpha
v_mul_f32 v[vgprValuC+6], s[sgprAlpha], v[vgprValuC+6] // *= alpha
v_mul_f32 v[vgprValuC+7], s[sgprAlpha], v[vgprValuC+7] // *= alpha
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // *= alpha
v_mul_f32 v[vgprValuC+9], s[sgprAlpha], v[vgprValuC+9] // *= alpha
v_mul_f32 v[vgprValuC+10], s[sgprAlpha], v[vgprValuC+10] // *= alpha
v_mul_f32 v[vgprValuC+11], s[sgprAlpha], v[vgprValuC+11] // *= alpha
v_mul_f32 v[vgprValuC+12], s[sgprAlpha], v[vgprValuC+12] // *= alpha
v_mul_f32 v[vgprValuC+13], s[sgprAlpha], v[vgprValuC+13] // *= alpha
v_mul_f32 v[vgprValuC+14], s[sgprAlpha], v[vgprValuC+14] // *= alpha
v_mul_f32 v[vgprValuC+15], s[sgprAlpha], v[vgprValuC+15] // *= alpha
v_mul_f32 v[vgprValuC+16], s[sgprAlpha], v[vgprValuC+16] // *= alpha
v_mul_f32 v[vgprValuC+17], s[sgprAlpha], v[vgprValuC+17] // *= alpha
v_mul_f32 v[vgprValuC+18], s[sgprAlpha], v[vgprValuC+18] // *= alpha
v_mul_f32 v[vgprValuC+19], s[sgprAlpha], v[vgprValuC+19] // *= alpha
v_mul_f32 v[vgprValuC+20], s[sgprAlpha], v[vgprValuC+20] // *= alpha
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+21] // *= alpha
v_mul_f32 v[vgprValuC+22], s[sgprAlpha], v[vgprValuC+22] // *= alpha
v_mul_f32 v[vgprValuC+23], s[sgprAlpha], v[vgprValuC+23] // *= alpha
v_mul_f32 v[vgprValuC+24], s[sgprAlpha], v[vgprValuC+24] // *= alpha
v_mul_f32 v[vgprValuC+25], s[sgprAlpha], v[vgprValuC+25] // *= alpha
v_mul_f32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+26] // *= alpha
v_mul_f32 v[vgprValuC+27], s[sgprAlpha], v[vgprValuC+27] // *= alpha
v_mul_f32 v[vgprValuC+28], s[sgprAlpha], v[vgprValuC+28] // *= alpha
v_mul_f32 v[vgprValuC+29], s[sgprAlpha], v[vgprValuC+29] // *= alpha
v_mul_f32 v[vgprValuC+30], s[sgprAlpha], v[vgprValuC+30] // *= alpha
v_mul_f32 v[vgprValuC+31], s[sgprAlpha], v[vgprValuC+31] // *= alpha
v_mul_f32 v[vgprValuC+32], s[sgprAlpha], v[vgprValuC+32] // *= alpha
v_mul_f32 v[vgprValuC+33], s[sgprAlpha], v[vgprValuC+33] // *= alpha
v_mul_f32 v[vgprValuC+34], s[sgprAlpha], v[vgprValuC+34] // *= alpha
v_mul_f32 v[vgprValuC+35], s[sgprAlpha], v[vgprValuC+35] // *= alpha
v_mul_f32 v[vgprValuC+36], s[sgprAlpha], v[vgprValuC+36] // *= alpha
v_mul_f32 v[vgprValuC+37], s[sgprAlpha], v[vgprValuC+37] // *= alpha
v_mul_f32 v[vgprValuC+38], s[sgprAlpha], v[vgprValuC+38] // *= alpha
v_mul_f32 v[vgprValuC+39], s[sgprAlpha], v[vgprValuC+39] // *= alpha
v_mul_f32 v[vgprValuC+40], s[sgprAlpha], v[vgprValuC+40] // *= alpha
v_mul_f32 v[vgprValuC+41], s[sgprAlpha], v[vgprValuC+41] // *= alpha
v_mul_f32 v[vgprValuC+42], s[sgprAlpha], v[vgprValuC+42] // *= alpha
v_mul_f32 v[vgprValuC+43], s[sgprAlpha], v[vgprValuC+43] // *= alpha
v_mul_f32 v[vgprValuC+44], s[sgprAlpha], v[vgprValuC+44] // *= alpha
v_mul_f32 v[vgprValuC+45], s[sgprAlpha], v[vgprValuC+45] // *= alpha
v_mul_f32 v[vgprValuC+46], s[sgprAlpha], v[vgprValuC+46] // *= alpha
v_mul_f32 v[vgprValuC+47], s[sgprAlpha], v[vgprValuC+47] // *= alpha
v_mul_f32 v[vgprValuC+48], s[sgprAlpha], v[vgprValuC+48] // *= alpha
v_mul_f32 v[vgprValuC+49], s[sgprAlpha], v[vgprValuC+49] // *= alpha
v_mul_f32 v[vgprValuC+50], s[sgprAlpha], v[vgprValuC+50] // *= alpha
v_mul_f32 v[vgprValuC+51], s[sgprAlpha], v[vgprValuC+51] // *= alpha
v_mul_f32 v[vgprValuC+52], s[sgprAlpha], v[vgprValuC+52] // *= alpha
v_mul_f32 v[vgprValuC+53], s[sgprAlpha], v[vgprValuC+53] // *= alpha
v_mul_f32 v[vgprValuC+54], s[sgprAlpha], v[vgprValuC+54] // *= alpha
v_mul_f32 v[vgprValuC+55], s[sgprAlpha], v[vgprValuC+55] // *= alpha
v_mul_f32 v[vgprValuC+56], s[sgprAlpha], v[vgprValuC+56] // *= alpha
v_mul_f32 v[vgprValuC+57], s[sgprAlpha], v[vgprValuC+57] // *= alpha
v_mul_f32 v[vgprValuC+58], s[sgprAlpha], v[vgprValuC+58] // *= alpha
v_mul_f32 v[vgprValuC+59], s[sgprAlpha], v[vgprValuC+59] // *= alpha
v_mul_f32 v[vgprValuC+60], s[sgprAlpha], v[vgprValuC+60] // *= alpha
v_mul_f32 v[vgprValuC+61], s[sgprAlpha], v[vgprValuC+61] // *= alpha
v_mul_f32 v[vgprValuC+62], s[sgprAlpha], v[vgprValuC+62] // *= alpha
v_mul_f32 v[vgprValuC+63], s[sgprAlpha], v[vgprValuC+63] // *= alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx4 v[0:3], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dwordx4 v[4:7], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
s_lshl_b32  s54, s[sgprStrideD1J], 2               // incToNextRow: Scale by BPE
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[8:11], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dwordx4 v[12:15], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
s_lshl_b32  s54, s[sgprStrideD1J], 2               // incToNextRow: Scale by BPE
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[16:19], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dwordx4 v[20:23], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
s_lshl_b32  s54, s[sgprStrideD1J], 2               // incToNextRow: Scale by BPE
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[24:27], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dwordx4 v[28:31], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
s_mul_i32 s54, s[sgprStrideD1J], 244               // scale StrideD *= numRows(61) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[32:35], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dwordx4 v[36:39], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
s_lshl_b32  s54, s[sgprStrideD1J], 2               // incToNextRow: Scale by BPE
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[40:43], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dwordx4 v[44:47], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
s_lshl_b32  s54, s[sgprStrideD1J], 2               // incToNextRow: Scale by BPE
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[48:51], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dwordx4 v[52:55], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
s_lshl_b32  s54, s[sgprStrideD1J], 2               // incToNextRow: Scale by BPE
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[56:59], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dwordx4 v[60:63], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
s_branch label_GW_End_37                           // jump to end
GW_B0_E1_29:

/* edge=1, allocate 48 sgpr. perBatch=6 perElement=2 elementsPerBatch=21 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1); (0,1,0,0:vw1); (0,1,0,1:vw1); (0,1,0,2:vw1); (0,1,0,3:vw1); (0,0,1,0:vw1); (0,0,1,1:vw1); (0,0,1,2:vw1); (0,0,1,3:vw1); (0,1,1,0:vw1); (0,1,1,1:vw1); (0,1,1,2:vw1); (0,1,1,3:vw1); (0,0,2,0:vw1); (0,0,2,1:vw1); (0,0,2,2:vw1); (0,0,2,3:vw1); (0,1,2,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[54:55], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[60:61], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v69, v66, v64, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v69, -1, v69, s[60:61]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
_v_add_co_u32 v67, vcc, v64, 1                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[62:63], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v70, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v70, -1, v70, s[62:63]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
_v_add_co_u32 v67, vcc, v64, 2                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v71, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v71, -1, v71, s[64:65]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
_v_add_co_u32 v67, vcc, v64, 3                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[66:67], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v72, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v72, -1, v72, s[66:67]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
_v_add_co_u32 v67, vcc, v64, 32                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[68:69], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v73, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v73, -1, v73, s[68:69]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,1) */
_v_add_co_u32 v67, vcc, v64, 33                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v74, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, -1, v74, s[70:71]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,2) */
_v_add_co_u32 v67, vcc, v64, 34                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v75, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v75, -1, v75, s[72:73]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,3) */
_v_add_co_u32 v67, vcc, v64, 35                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v76, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v76, -1, v76, s[74:75]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
_v_add_co_u32 v65, vcc, v65, 1                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
_v_add_u32 v66, v66, s[sgprStrideC1J]              // ROWINC- Move cinRowPtr to next row
v_cmp_lt_u32 s[54:55], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[76:77], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
_v_add_lshl_u32 v77, v66, v64, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v77, -1, v77, s[76:77]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,1) */
_v_add_co_u32 v67, vcc, v64, 1                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[78:79], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
_v_add_lshl_u32 v78, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v78, -1, v78, s[78:79]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,2) */
_v_add_co_u32 v67, vcc, v64, 2                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[80:81], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
_v_add_lshl_u32 v79, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v79, -1, v79, s[80:81]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,3) */
_v_add_co_u32 v67, vcc, v64, 3                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[82:83], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
_v_add_lshl_u32 v80, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, -1, v80, s[82:83]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
_v_add_co_u32 v67, vcc, v64, 32                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[84:85], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v81, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v81, -1, v81, s[84:85]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,1) */
_v_add_co_u32 v67, vcc, v64, 33                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[86:87], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
_v_add_lshl_u32 v82, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v82, -1, v82, s[86:87]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,2) */
_v_add_co_u32 v67, vcc, v64, 34                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[88:89], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
_v_add_lshl_u32 v83, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v83, -1, v83, s[88:89]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,3) */
_v_add_co_u32 v67, vcc, v64, 35                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[90:91], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
_v_add_lshl_u32 v84, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v84, -1, v84, s[90:91]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
_v_add_co_u32 v65, vcc, v65, 1                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
_v_add_u32 v66, v66, s[sgprStrideC1J]              // ROWINC- Move cinRowPtr to next row
v_cmp_lt_u32 s[54:55], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[92:93], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v85, v66, v64, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v85, -1, v85, s[92:93]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,1) */
_v_add_co_u32 v67, vcc, v64, 1                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[94:95], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
_v_add_lshl_u32 v86, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, -1, v86, s[94:95]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,2) */
_v_add_co_u32 v67, vcc, v64, 2                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[96:97], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
_v_add_lshl_u32 v87, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v87, -1, v87, s[96:97]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,3) */
_v_add_co_u32 v67, vcc, v64, 3                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[98:99], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
_v_add_lshl_u32 v88, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v88, -1, v88, s[98:99]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,0) */
_v_add_co_u32 v67, vcc, v64, 32                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[100:101], v65, s[sgprSizeJ]         // coord1 < size1
s_and_b64 s[100:101], s[54:55], s[100:101]         // in0 && in1
_v_add_lshl_u32 v89, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v89, -1, v89, s[100:101]             // clip if OOB. offset

/* rC *= alpha batchEements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 0, 2), (0, 1, 0, 3), (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 1, 2), (0, 0, 1, 3), (0, 1, 1, 0), (0, 1, 1, 1), (0, 1, 1, 2), (0, 1, 1, 3), (0, 0, 2, 0), (0, 0, 2, 1), (0, 0, 2, 2), (0, 0, 2, 3), (0, 1, 2, 0)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha
v_mul_f32 v[vgprValuC+1], s[sgprAlpha], v[vgprValuC+1] // *= alpha
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha
v_mul_f32 v[vgprValuC+3], s[sgprAlpha], v[vgprValuC+3] // *= alpha
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // *= alpha
v_mul_f32 v[vgprValuC+5], s[sgprAlpha], v[vgprValuC+5] // *= alpha
v_mul_f32 v[vgprValuC+6], s[sgprAlpha], v[vgprValuC+6] // *= alpha
v_mul_f32 v[vgprValuC+7], s[sgprAlpha], v[vgprValuC+7] // *= alpha
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // *= alpha
v_mul_f32 v[vgprValuC+9], s[sgprAlpha], v[vgprValuC+9] // *= alpha
v_mul_f32 v[vgprValuC+10], s[sgprAlpha], v[vgprValuC+10] // *= alpha
v_mul_f32 v[vgprValuC+11], s[sgprAlpha], v[vgprValuC+11] // *= alpha
v_mul_f32 v[vgprValuC+12], s[sgprAlpha], v[vgprValuC+12] // *= alpha
v_mul_f32 v[vgprValuC+13], s[sgprAlpha], v[vgprValuC+13] // *= alpha
v_mul_f32 v[vgprValuC+14], s[sgprAlpha], v[vgprValuC+14] // *= alpha
v_mul_f32 v[vgprValuC+15], s[sgprAlpha], v[vgprValuC+15] // *= alpha
v_mul_f32 v[vgprValuC+16], s[sgprAlpha], v[vgprValuC+16] // *= alpha
v_mul_f32 v[vgprValuC+17], s[sgprAlpha], v[vgprValuC+17] // *= alpha
v_mul_f32 v[vgprValuC+18], s[sgprAlpha], v[vgprValuC+18] // *= alpha
v_mul_f32 v[vgprValuC+19], s[sgprAlpha], v[vgprValuC+19] // *= alpha
v_mul_f32 v[vgprValuC+20], s[sgprAlpha], v[vgprValuC+20] // *= alpha

/* apply mask, calc new C and issue writes */
buffer_store_dword v0, v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v1, v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v2, v71, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v3, v72, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v4, v73, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v5, v74, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v6, v75, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v7, v76, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v8, v77, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v9, v78, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v10, v79, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v11, v80, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v12, v81, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v13, v82, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v14, v83, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v15, v84, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v16, v85, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v17, v86, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v18, v87, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v19, v88, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v20, v89, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (0,1,2,1:vw1); (0,1,2,2:vw1); (0,1,2,3:vw1); (0,0,3,0:vw1); (0,0,3,1:vw1); (0,0,3,2:vw1); (0,0,3,3:vw1); (0,1,3,0:vw1); (0,1,3,1:vw1); (0,1,3,2:vw1); (0,1,3,3:vw1); (1,0,0,0:vw1); (1,0,0,1:vw1); (1,0,0,2:vw1); (1,0,0,3:vw1); (1,1,0,0:vw1); (1,1,0,1:vw1); (1,1,0,2:vw1); (1,1,0,3:vw1); (1,0,1,0:vw1); (1,0,1,1:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,2,1,1) */
_v_add_co_u32 v67, vcc, v64, 33                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[60:61], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v69, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v69, -1, v69, s[60:61]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,2) */
_v_add_co_u32 v67, vcc, v64, 34                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[62:63], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v70, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v70, -1, v70, s[62:63]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,3) */
_v_add_co_u32 v67, vcc, v64, 35                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v71, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v71, -1, v71, s[64:65]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
_v_add_co_u32 v65, vcc, v65, 1                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
_v_add_u32 v66, v66, s[sgprStrideC1J]              // ROWINC- Move cinRowPtr to next row
v_cmp_lt_u32 s[54:55], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[66:67], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v72, v66, v64, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v72, -1, v72, s[66:67]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,1) */
_v_add_co_u32 v67, vcc, v64, 1                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[68:69], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v73, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v73, -1, v73, s[68:69]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,2) */
_v_add_co_u32 v67, vcc, v64, 2                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v74, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, -1, v74, s[70:71]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,3) */
_v_add_co_u32 v67, vcc, v64, 3                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v75, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v75, -1, v75, s[72:73]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,0) */
_v_add_co_u32 v67, vcc, v64, 32                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v76, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v76, -1, v76, s[74:75]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,1) */
_v_add_co_u32 v67, vcc, v64, 33                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[76:77], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
_v_add_lshl_u32 v77, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v77, -1, v77, s[76:77]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,2) */
_v_add_co_u32 v67, vcc, v64, 34                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[78:79], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
_v_add_lshl_u32 v78, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v78, -1, v78, s[78:79]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,3) */
_v_add_co_u32 v67, vcc, v64, 35                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[80:81], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
_v_add_lshl_u32 v79, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v79, -1, v79, s[80:81]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
_v_add_co_u32 v65, vcc, v65, 61                    // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s54, s[sgprStrideC1J], 61                // scale stride
_v_add_u32 v66, v66, s54                           // ROWINC- Move cinRowPtr to next row
v_cmp_lt_u32 s[54:55], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[82:83], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
_v_add_lshl_u32 v80, v66, v64, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, -1, v80, s[82:83]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,1) */
_v_add_co_u32 v67, vcc, v64, 1                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[84:85], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v81, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v81, -1, v81, s[84:85]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,2) */
_v_add_co_u32 v67, vcc, v64, 2                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[86:87], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
_v_add_lshl_u32 v82, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v82, -1, v82, s[86:87]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,3) */
_v_add_co_u32 v67, vcc, v64, 3                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[88:89], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
_v_add_lshl_u32 v83, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v83, -1, v83, s[88:89]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
_v_add_co_u32 v67, vcc, v64, 32                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[90:91], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
_v_add_lshl_u32 v84, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v84, -1, v84, s[90:91]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,1) */
_v_add_co_u32 v67, vcc, v64, 33                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[92:93], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v85, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v85, -1, v85, s[92:93]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,2) */
_v_add_co_u32 v67, vcc, v64, 34                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[94:95], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
_v_add_lshl_u32 v86, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, -1, v86, s[94:95]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,3) */
_v_add_co_u32 v67, vcc, v64, 35                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[96:97], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
_v_add_lshl_u32 v87, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v87, -1, v87, s[96:97]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,1,0,0) */
_v_add_co_u32 v65, vcc, v65, 1                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
_v_add_u32 v66, v66, s[sgprStrideC1J]              // ROWINC- Move cinRowPtr to next row
v_cmp_lt_u32 s[54:55], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[98:99], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
_v_add_lshl_u32 v88, v66, v64, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v88, -1, v88, s[98:99]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,1,0,1) */
_v_add_co_u32 v67, vcc, v64, 1                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[100:101], v65, s[sgprSizeJ]         // coord1 < size1
s_and_b64 s[100:101], s[54:55], s[100:101]         // in0 && in1
_v_add_lshl_u32 v89, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v89, -1, v89, s[100:101]             // clip if OOB. offset

/* rC *= alpha batchEements=[(0, 1, 2, 1), (0, 1, 2, 2), (0, 1, 2, 3), (0, 0, 3, 0), (0, 0, 3, 1), (0, 0, 3, 2), (0, 0, 3, 3), (0, 1, 3, 0), (0, 1, 3, 1), (0, 1, 3, 2), (0, 1, 3, 3), (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 0, 2), (1, 0, 0, 3), (1, 1, 0, 0), (1, 1, 0, 1), (1, 1, 0, 2), (1, 1, 0, 3), (1, 0, 1, 0), (1, 0, 1, 1)] */
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+21] // *= alpha
v_mul_f32 v[vgprValuC+22], s[sgprAlpha], v[vgprValuC+22] // *= alpha
v_mul_f32 v[vgprValuC+23], s[sgprAlpha], v[vgprValuC+23] // *= alpha
v_mul_f32 v[vgprValuC+24], s[sgprAlpha], v[vgprValuC+24] // *= alpha
v_mul_f32 v[vgprValuC+25], s[sgprAlpha], v[vgprValuC+25] // *= alpha
v_mul_f32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+26] // *= alpha
v_mul_f32 v[vgprValuC+27], s[sgprAlpha], v[vgprValuC+27] // *= alpha
v_mul_f32 v[vgprValuC+28], s[sgprAlpha], v[vgprValuC+28] // *= alpha
v_mul_f32 v[vgprValuC+29], s[sgprAlpha], v[vgprValuC+29] // *= alpha
v_mul_f32 v[vgprValuC+30], s[sgprAlpha], v[vgprValuC+30] // *= alpha
v_mul_f32 v[vgprValuC+31], s[sgprAlpha], v[vgprValuC+31] // *= alpha
v_mul_f32 v[vgprValuC+32], s[sgprAlpha], v[vgprValuC+32] // *= alpha
v_mul_f32 v[vgprValuC+33], s[sgprAlpha], v[vgprValuC+33] // *= alpha
v_mul_f32 v[vgprValuC+34], s[sgprAlpha], v[vgprValuC+34] // *= alpha
v_mul_f32 v[vgprValuC+35], s[sgprAlpha], v[vgprValuC+35] // *= alpha
v_mul_f32 v[vgprValuC+36], s[sgprAlpha], v[vgprValuC+36] // *= alpha
v_mul_f32 v[vgprValuC+37], s[sgprAlpha], v[vgprValuC+37] // *= alpha
v_mul_f32 v[vgprValuC+38], s[sgprAlpha], v[vgprValuC+38] // *= alpha
v_mul_f32 v[vgprValuC+39], s[sgprAlpha], v[vgprValuC+39] // *= alpha
v_mul_f32 v[vgprValuC+40], s[sgprAlpha], v[vgprValuC+40] // *= alpha
v_mul_f32 v[vgprValuC+41], s[sgprAlpha], v[vgprValuC+41] // *= alpha

/* apply mask, calc new C and issue writes */
buffer_store_dword v21, v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v22, v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v23, v71, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v24, v72, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v25, v73, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v26, v74, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v27, v75, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v28, v76, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v29, v77, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v30, v78, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v31, v79, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v32, v80, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v33, v81, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v34, v82, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v35, v83, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v36, v84, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v37, v85, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v38, v86, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v39, v87, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v40, v88, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v41, v89, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #2 (d1,d0,vc1,vc0) = */
/*    (1,0,1,2:vw1); (1,0,1,3:vw1); (1,1,1,0:vw1); (1,1,1,1:vw1); (1,1,1,2:vw1); (1,1,1,3:vw1); (1,0,2,0:vw1); (1,0,2,1:vw1); (1,0,2,2:vw1); (1,0,2,3:vw1); (1,1,2,0:vw1); (1,1,2,1:vw1); (1,1,2,2:vw1); (1,1,2,3:vw1); (1,0,3,0:vw1); (1,0,3,1:vw1); (1,0,3,2:vw1); (1,0,3,3:vw1); (1,1,3,0:vw1); (1,1,3,1:vw1); (1,1,3,2:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(1,1,0,2) */
_v_add_co_u32 v67, vcc, v64, 2                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[60:61], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v69, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v69, -1, v69, s[60:61]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,1,0,3) */
_v_add_co_u32 v67, vcc, v64, 3                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[62:63], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v70, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v70, -1, v70, s[62:63]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,1,1,0) */
_v_add_co_u32 v67, vcc, v64, 32                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v71, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v71, -1, v71, s[64:65]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,1,1,1) */
_v_add_co_u32 v67, vcc, v64, 33                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[66:67], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v72, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v72, -1, v72, s[66:67]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,1,1,2) */
_v_add_co_u32 v67, vcc, v64, 34                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[68:69], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v73, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v73, -1, v73, s[68:69]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,1,1,3) */
_v_add_co_u32 v67, vcc, v64, 35                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v74, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, -1, v74, s[70:71]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,2,0,0) */
_v_add_co_u32 v65, vcc, v65, 1                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
_v_add_u32 v66, v66, s[sgprStrideC1J]              // ROWINC- Move cinRowPtr to next row
v_cmp_lt_u32 s[54:55], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v75, v66, v64, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v75, -1, v75, s[72:73]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,2,0,1) */
_v_add_co_u32 v67, vcc, v64, 1                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v76, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v76, -1, v76, s[74:75]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,2,0,2) */
_v_add_co_u32 v67, vcc, v64, 2                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[76:77], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
_v_add_lshl_u32 v77, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v77, -1, v77, s[76:77]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,2,0,3) */
_v_add_co_u32 v67, vcc, v64, 3                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[78:79], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
_v_add_lshl_u32 v78, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v78, -1, v78, s[78:79]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,2,1,0) */
_v_add_co_u32 v67, vcc, v64, 32                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[80:81], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
_v_add_lshl_u32 v79, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v79, -1, v79, s[80:81]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,2,1,1) */
_v_add_co_u32 v67, vcc, v64, 33                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[82:83], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
_v_add_lshl_u32 v80, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, -1, v80, s[82:83]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,2,1,2) */
_v_add_co_u32 v67, vcc, v64, 34                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[84:85], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v81, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v81, -1, v81, s[84:85]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,2,1,3) */
_v_add_co_u32 v67, vcc, v64, 35                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[86:87], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
_v_add_lshl_u32 v82, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v82, -1, v82, s[86:87]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,3,0,0) */
_v_add_co_u32 v65, vcc, v65, 1                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
_v_add_u32 v66, v66, s[sgprStrideC1J]              // ROWINC- Move cinRowPtr to next row
v_cmp_lt_u32 s[54:55], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[88:89], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
_v_add_lshl_u32 v83, v66, v64, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v83, -1, v83, s[88:89]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,3,0,1) */
_v_add_co_u32 v67, vcc, v64, 1                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[90:91], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
_v_add_lshl_u32 v84, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v84, -1, v84, s[90:91]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,3,0,2) */
_v_add_co_u32 v67, vcc, v64, 2                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[92:93], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v85, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v85, -1, v85, s[92:93]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,3,0,3) */
_v_add_co_u32 v67, vcc, v64, 3                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[94:95], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
_v_add_lshl_u32 v86, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, -1, v86, s[94:95]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,3,1,0) */
_v_add_co_u32 v67, vcc, v64, 32                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[96:97], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
_v_add_lshl_u32 v87, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v87, -1, v87, s[96:97]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,3,1,1) */
_v_add_co_u32 v67, vcc, v64, 33                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[98:99], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
_v_add_lshl_u32 v88, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v88, -1, v88, s[98:99]               // clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,3,1,2) */
_v_add_co_u32 v67, vcc, v64, 34                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[100:101], v65, s[sgprSizeJ]         // coord1 < size1
s_and_b64 s[100:101], s[54:55], s[100:101]         // in0 && in1
_v_add_lshl_u32 v89, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v89, -1, v89, s[100:101]             // clip if OOB. offset

/* rC *= alpha batchEements=[(1, 0, 1, 2), (1, 0, 1, 3), (1, 1, 1, 0), (1, 1, 1, 1), (1, 1, 1, 2), (1, 1, 1, 3), (1, 0, 2, 0), (1, 0, 2, 1), (1, 0, 2, 2), (1, 0, 2, 3), (1, 1, 2, 0), (1, 1, 2, 1), (1, 1, 2, 2), (1, 1, 2, 3), (1, 0, 3, 0), (1, 0, 3, 1), (1, 0, 3, 2), (1, 0, 3, 3), (1, 1, 3, 0), (1, 1, 3, 1), (1, 1, 3, 2)] */
v_mul_f32 v[vgprValuC+42], s[sgprAlpha], v[vgprValuC+42] // *= alpha
v_mul_f32 v[vgprValuC+43], s[sgprAlpha], v[vgprValuC+43] // *= alpha
v_mul_f32 v[vgprValuC+44], s[sgprAlpha], v[vgprValuC+44] // *= alpha
v_mul_f32 v[vgprValuC+45], s[sgprAlpha], v[vgprValuC+45] // *= alpha
v_mul_f32 v[vgprValuC+46], s[sgprAlpha], v[vgprValuC+46] // *= alpha
v_mul_f32 v[vgprValuC+47], s[sgprAlpha], v[vgprValuC+47] // *= alpha
v_mul_f32 v[vgprValuC+48], s[sgprAlpha], v[vgprValuC+48] // *= alpha
v_mul_f32 v[vgprValuC+49], s[sgprAlpha], v[vgprValuC+49] // *= alpha
v_mul_f32 v[vgprValuC+50], s[sgprAlpha], v[vgprValuC+50] // *= alpha
v_mul_f32 v[vgprValuC+51], s[sgprAlpha], v[vgprValuC+51] // *= alpha
v_mul_f32 v[vgprValuC+52], s[sgprAlpha], v[vgprValuC+52] // *= alpha
v_mul_f32 v[vgprValuC+53], s[sgprAlpha], v[vgprValuC+53] // *= alpha
v_mul_f32 v[vgprValuC+54], s[sgprAlpha], v[vgprValuC+54] // *= alpha
v_mul_f32 v[vgprValuC+55], s[sgprAlpha], v[vgprValuC+55] // *= alpha
v_mul_f32 v[vgprValuC+56], s[sgprAlpha], v[vgprValuC+56] // *= alpha
v_mul_f32 v[vgprValuC+57], s[sgprAlpha], v[vgprValuC+57] // *= alpha
v_mul_f32 v[vgprValuC+58], s[sgprAlpha], v[vgprValuC+58] // *= alpha
v_mul_f32 v[vgprValuC+59], s[sgprAlpha], v[vgprValuC+59] // *= alpha
v_mul_f32 v[vgprValuC+60], s[sgprAlpha], v[vgprValuC+60] // *= alpha
v_mul_f32 v[vgprValuC+61], s[sgprAlpha], v[vgprValuC+61] // *= alpha
v_mul_f32 v[vgprValuC+62], s[sgprAlpha], v[vgprValuC+62] // *= alpha

/* apply mask, calc new C and issue writes */
buffer_store_dword v42, v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v43, v70, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v44, v71, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v45, v72, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v46, v73, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v47, v74, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v48, v75, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v49, v76, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v50, v77, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v51, v78, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v52, v79, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v53, v80, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v54, v81, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v55, v82, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v56, v83, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v57, v84, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v58, v85, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v59, v86, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v60, v87, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v61, v88, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_dword v62, v89, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #3 (d1,d0,vc1,vc0) = */
/*    (1,1,3,3:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(1,3,1,3) */
_v_add_co_u32 v67, vcc, v64, 35                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[60:61], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v69, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v69, -1, v69, s[60:61]               // clip if OOB. offset

/* rC *= alpha batchEements=[(1, 1, 3, 3)] */
v_mul_f32 v[vgprValuC+63], s[sgprAlpha], v[vgprValuC+63] // *= alpha

/* apply mask, calc new C and issue writes */
buffer_store_dword v63, v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
s_branch label_GW_End_37                           // jump to end
GW_Beta_38:
s_and_b32 s54, 63, s[sgprSizeI]                    // s54 = s[sgprSizeI] % 64
s_add_u32 s56, -0x1, s[sgprNumWorkGroups0]         //
s_cmp_ge_u32 s[sgprWorkGroup0], s56                // wg0 >= nwg0-1 ?
s_cselect_b32 s54, s54, 0                          // set rMT0
s_cmpk_gt_u32 s54, 0x0                             // rMT0 > 0
s_cbranch_scc1 GW_B1_E1_36                         // jump if edges required
s_and_b32 s54, 127, s[sgprSizeJ]                   // s54 = s[sgprSizeJ] % 128
s_add_u32 s56, -0x1, s[sgprNumWorkGroups1]         //
s_cmp_ge_u32 s[sgprWorkGroup1], s56                // wg1 >= nwg1-1
s_cselect_b32 s54, s54, 0                          // set rMT1
s_cmpk_gt_u32 s54, 0x0                             // rMT1 > 0
s_cbranch_scc1 GW_B1_E1_36                         // jump if edges required
GW_B1_E0_33:

/* edge=0, allocate 2 sgpr. perBatch=2 perElement=0 elementsPerBatch=10 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSharedMask=1 optSrdIncForRow=1 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,1,0,0:vw4); (0,0,1,0:vw4); (0,1,1,0:vw4); (0,0,2,0:vw4); (0,1,2,0:vw4); (0,0,3,0:vw4); (0,1,3,0:vw4); (1,0,0,0:vw4); (1,1,0,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
_v_add_lshl_u32 v69, v66, v64, 0x2                 // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=64, coord0Vgpr=64
buffer_load_dwordx4 v[70:73], v69, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
buffer_load_dwordx4 v[74:77], v69, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
s_lshl_b32  s54, s[sgprStrideC1J], 2               // incToNextRow: Scale by BPE
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[78:81], v69, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
buffer_load_dwordx4 v[82:85], v69, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
s_lshl_b32  s54, s[sgprStrideC1J], 2               // incToNextRow: Scale by BPE
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[86:89], v69, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,2,1,0) */
buffer_load_dwordx4 v[90:93], v69, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
s_lshl_b32  s54, s[sgprStrideC1J], 2               // incToNextRow: Scale by BPE
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[94:97], v69, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,1,0) */
buffer_load_dwordx4 v[98:101], v69, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
s_mul_i32 s54, s[sgprStrideC1J], 244               // scale StrideC *= numRows(61) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[102:105], v69, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
buffer_load_dwordx4 v[106:109], v69, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc

/* rC *= alpha batchEements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 1, 1, 0), (0, 0, 2, 0), (0, 1, 2, 0), (0, 0, 3, 0), (0, 1, 3, 0), (1, 0, 0, 0), (1, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha
v_mul_f32 v[vgprValuC+1], s[sgprAlpha], v[vgprValuC+1] // *= alpha
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha
v_mul_f32 v[vgprValuC+3], s[sgprAlpha], v[vgprValuC+3] // *= alpha
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // *= alpha
v_mul_f32 v[vgprValuC+5], s[sgprAlpha], v[vgprValuC+5] // *= alpha
v_mul_f32 v[vgprValuC+6], s[sgprAlpha], v[vgprValuC+6] // *= alpha
v_mul_f32 v[vgprValuC+7], s[sgprAlpha], v[vgprValuC+7] // *= alpha
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // *= alpha
v_mul_f32 v[vgprValuC+9], s[sgprAlpha], v[vgprValuC+9] // *= alpha
v_mul_f32 v[vgprValuC+10], s[sgprAlpha], v[vgprValuC+10] // *= alpha
v_mul_f32 v[vgprValuC+11], s[sgprAlpha], v[vgprValuC+11] // *= alpha
v_mul_f32 v[vgprValuC+12], s[sgprAlpha], v[vgprValuC+12] // *= alpha
v_mul_f32 v[vgprValuC+13], s[sgprAlpha], v[vgprValuC+13] // *= alpha
v_mul_f32 v[vgprValuC+14], s[sgprAlpha], v[vgprValuC+14] // *= alpha
v_mul_f32 v[vgprValuC+15], s[sgprAlpha], v[vgprValuC+15] // *= alpha
v_mul_f32 v[vgprValuC+16], s[sgprAlpha], v[vgprValuC+16] // *= alpha
v_mul_f32 v[vgprValuC+17], s[sgprAlpha], v[vgprValuC+17] // *= alpha
v_mul_f32 v[vgprValuC+18], s[sgprAlpha], v[vgprValuC+18] // *= alpha
v_mul_f32 v[vgprValuC+19], s[sgprAlpha], v[vgprValuC+19] // *= alpha
v_mul_f32 v[vgprValuC+20], s[sgprAlpha], v[vgprValuC+20] // *= alpha
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+21] // *= alpha
v_mul_f32 v[vgprValuC+22], s[sgprAlpha], v[vgprValuC+22] // *= alpha
v_mul_f32 v[vgprValuC+23], s[sgprAlpha], v[vgprValuC+23] // *= alpha
v_mul_f32 v[vgprValuC+24], s[sgprAlpha], v[vgprValuC+24] // *= alpha
v_mul_f32 v[vgprValuC+25], s[sgprAlpha], v[vgprValuC+25] // *= alpha
v_mul_f32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+26] // *= alpha
v_mul_f32 v[vgprValuC+27], s[sgprAlpha], v[vgprValuC+27] // *= alpha
v_mul_f32 v[vgprValuC+28], s[sgprAlpha], v[vgprValuC+28] // *= alpha
v_mul_f32 v[vgprValuC+29], s[sgprAlpha], v[vgprValuC+29] // *= alpha
v_mul_f32 v[vgprValuC+30], s[sgprAlpha], v[vgprValuC+30] // *= alpha
v_mul_f32 v[vgprValuC+31], s[sgprAlpha], v[vgprValuC+31] // *= alpha
v_mul_f32 v[vgprValuC+32], s[sgprAlpha], v[vgprValuC+32] // *= alpha
v_mul_f32 v[vgprValuC+33], s[sgprAlpha], v[vgprValuC+33] // *= alpha
v_mul_f32 v[vgprValuC+34], s[sgprAlpha], v[vgprValuC+34] // *= alpha
v_mul_f32 v[vgprValuC+35], s[sgprAlpha], v[vgprValuC+35] // *= alpha
v_mul_f32 v[vgprValuC+36], s[sgprAlpha], v[vgprValuC+36] // *= alpha
v_mul_f32 v[vgprValuC+37], s[sgprAlpha], v[vgprValuC+37] // *= alpha
v_mul_f32 v[vgprValuC+38], s[sgprAlpha], v[vgprValuC+38] // *= alpha
v_mul_f32 v[vgprValuC+39], s[sgprAlpha], v[vgprValuC+39] // *= alpha

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 10 - 0 + 0 - 1
v_mac_f32 v[vgprValuC+0], v70, s[sgprBeta]         // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+1], v71, s[sgprBeta]         // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+2], v72, s[sgprBeta]         // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+3], v73, s[sgprBeta]         // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[0:3], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 10 - 1 + 1 - 1
v_mac_f32 v[vgprValuC+4], v74, s[sgprBeta]         // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+5], v75, s[sgprBeta]         // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+6], v76, s[sgprBeta]         // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+7], v77, s[sgprBeta]         // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[4:7], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 10 - 2 + 2 - 1
v_mac_f32 v[vgprValuC+8], v78, s[sgprBeta]         // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+9], v79, s[sgprBeta]         // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+10], v80, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+11], v81, s[sgprBeta]        // finalSum = sum*alpha + C*beta
s_lshl_b32  s54, s[sgprStrideD1J], 2               // incToNextRow: Scale by BPE
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[8:11], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 10 - 3 + 3 - 1
v_mac_f32 v[vgprValuC+12], v82, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+13], v83, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+14], v84, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+15], v85, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[12:15], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 10 - 4 + 4 - 1
v_mac_f32 v[vgprValuC+16], v86, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+17], v87, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+18], v88, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+19], v89, s[sgprBeta]        // finalSum = sum*alpha + C*beta
s_lshl_b32  s54, s[sgprStrideD1J], 2               // incToNextRow: Scale by BPE
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[16:19], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 10 - 5 + 5 - 1
v_mac_f32 v[vgprValuC+20], v90, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+21], v91, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+22], v92, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+23], v93, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[20:23], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 10 - 6 + 6 - 1
v_mac_f32 v[vgprValuC+24], v94, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+25], v95, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+26], v96, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+27], v97, s[sgprBeta]        // finalSum = sum*alpha + C*beta
s_lshl_b32  s54, s[sgprStrideD1J], 2               // incToNextRow: Scale by BPE
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[24:27], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 10 - 7 + 7 - 1
v_mac_f32 v[vgprValuC+28], v98, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+29], v99, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+30], v100, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+31], v101, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[28:31], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 10 - 8 + 8 - 1
v_mac_f32 v[vgprValuC+32], v102, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+33], v103, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+34], v104, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+35], v105, s[sgprBeta]       // finalSum = sum*alpha + C*beta
s_mul_i32 s54, s[sgprStrideD1J], 244               // scale StrideD *= numRows(61) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[32:35], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 10 - 9 + 9 - 1
v_mac_f32 v[vgprValuC+36], v106, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+37], v107, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+38], v108, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+39], v109, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[36:39], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
/* optSingleColVgpr=1 optSharedColVgpr=0 optSharedMask=1 optSrdIncForRow=1 */

/******************************************/
/* Global Write Beta Batch #1 (d1,d0,vc1,vc0) = */
/*    (1,0,1,0:vw4); (1,1,1,0:vw4); (1,0,2,0:vw4); (1,1,2,0:vw4); (1,0,3,0:vw4); (1,1,3,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(1,1,0,0) */
s_lshl_b32  s54, s[sgprStrideC1J], 2               // incToNextRow: Scale by BPE
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[70:73], v69, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,1,1,0) */
buffer_load_dwordx4 v[74:77], v69, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,2,0,0) */
s_lshl_b32  s54, s[sgprStrideC1J], 2               // incToNextRow: Scale by BPE
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[78:81], v69, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,2,1,0) */
buffer_load_dwordx4 v[82:85], v69, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,3,0,0) */
s_lshl_b32  s54, s[sgprStrideC1J], 2               // incToNextRow: Scale by BPE
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[86:89], v69, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,3,1,0) */
buffer_load_dwordx4 v[90:93], v69, s[sgprSrdC:sgprSrdC+3], 0, offen offset:128 // load C for beta calc

/* rC *= alpha batchEements=[(1, 0, 1, 0), (1, 1, 1, 0), (1, 0, 2, 0), (1, 1, 2, 0), (1, 0, 3, 0), (1, 1, 3, 0)] */
v_mul_f32 v[vgprValuC+40], s[sgprAlpha], v[vgprValuC+40] // *= alpha
v_mul_f32 v[vgprValuC+41], s[sgprAlpha], v[vgprValuC+41] // *= alpha
v_mul_f32 v[vgprValuC+42], s[sgprAlpha], v[vgprValuC+42] // *= alpha
v_mul_f32 v[vgprValuC+43], s[sgprAlpha], v[vgprValuC+43] // *= alpha
v_mul_f32 v[vgprValuC+44], s[sgprAlpha], v[vgprValuC+44] // *= alpha
v_mul_f32 v[vgprValuC+45], s[sgprAlpha], v[vgprValuC+45] // *= alpha
v_mul_f32 v[vgprValuC+46], s[sgprAlpha], v[vgprValuC+46] // *= alpha
v_mul_f32 v[vgprValuC+47], s[sgprAlpha], v[vgprValuC+47] // *= alpha
v_mul_f32 v[vgprValuC+48], s[sgprAlpha], v[vgprValuC+48] // *= alpha
v_mul_f32 v[vgprValuC+49], s[sgprAlpha], v[vgprValuC+49] // *= alpha
v_mul_f32 v[vgprValuC+50], s[sgprAlpha], v[vgprValuC+50] // *= alpha
v_mul_f32 v[vgprValuC+51], s[sgprAlpha], v[vgprValuC+51] // *= alpha
v_mul_f32 v[vgprValuC+52], s[sgprAlpha], v[vgprValuC+52] // *= alpha
v_mul_f32 v[vgprValuC+53], s[sgprAlpha], v[vgprValuC+53] // *= alpha
v_mul_f32 v[vgprValuC+54], s[sgprAlpha], v[vgprValuC+54] // *= alpha
v_mul_f32 v[vgprValuC+55], s[sgprAlpha], v[vgprValuC+55] // *= alpha
v_mul_f32 v[vgprValuC+56], s[sgprAlpha], v[vgprValuC+56] // *= alpha
v_mul_f32 v[vgprValuC+57], s[sgprAlpha], v[vgprValuC+57] // *= alpha
v_mul_f32 v[vgprValuC+58], s[sgprAlpha], v[vgprValuC+58] // *= alpha
v_mul_f32 v[vgprValuC+59], s[sgprAlpha], v[vgprValuC+59] // *= alpha
v_mul_f32 v[vgprValuC+60], s[sgprAlpha], v[vgprValuC+60] // *= alpha
v_mul_f32 v[vgprValuC+61], s[sgprAlpha], v[vgprValuC+61] // *= alpha
v_mul_f32 v[vgprValuC+62], s[sgprAlpha], v[vgprValuC+62] // *= alpha
v_mul_f32 v[vgprValuC+63], s[sgprAlpha], v[vgprValuC+63] // *= alpha

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(5)                                 // wait C (interleaved) 5 = 6 - 0 + 0 - 1
v_mac_f32 v[vgprValuC+40], v70, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+41], v71, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+42], v72, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+43], v73, s[sgprBeta]        // finalSum = sum*alpha + C*beta
s_lshl_b32  s54, s[sgprStrideD1J], 2               // incToNextRow: Scale by BPE
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[40:43], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(5)                                 // wait C (interleaved) 5 = 6 - 1 + 1 - 1
v_mac_f32 v[vgprValuC+44], v74, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+45], v75, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+46], v76, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+47], v77, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[44:47], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(5)                                 // wait C (interleaved) 5 = 6 - 2 + 2 - 1
v_mac_f32 v[vgprValuC+48], v78, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+49], v79, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+50], v80, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+51], v81, s[sgprBeta]        // finalSum = sum*alpha + C*beta
s_lshl_b32  s54, s[sgprStrideD1J], 2               // incToNextRow: Scale by BPE
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[48:51], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(5)                                 // wait C (interleaved) 5 = 6 - 3 + 3 - 1
v_mac_f32 v[vgprValuC+52], v82, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+53], v83, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+54], v84, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+55], v85, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[52:55], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D

s_waitcnt vmcnt(5)                                 // wait C (interleaved) 5 = 6 - 4 + 4 - 1
v_mac_f32 v[vgprValuC+56], v86, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+57], v87, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+58], v88, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+59], v89, s[sgprBeta]        // finalSum = sum*alpha + C*beta
s_lshl_b32  s54, s[sgprStrideD1J], 2               // incToNextRow: Scale by BPE
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[56:59], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(5)                                 // wait C (interleaved) 5 = 6 - 5 + 5 - 1
v_mac_f32 v[vgprValuC+60], v90, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+61], v91, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+62], v92, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_mac_f32 v[vgprValuC+63], v93, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[60:63], v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:128 // store D
s_branch label_GW_End_37                           // jump to end
GW_B1_E1_36:

/* edge=1, allocate 48 sgpr. perBatch=6 perElement=2 elementsPerBatch=21 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1); (0,1,0,0:vw1); (0,1,0,1:vw1); (0,1,0,2:vw1); (0,1,0,3:vw1); (0,0,1,0:vw1); (0,0,1,1:vw1); (0,0,1,2:vw1); (0,0,1,3:vw1); (0,1,1,0:vw1); (0,1,1,1:vw1); (0,1,1,2:vw1); (0,1,1,3:vw1); (0,0,2,0:vw1); (0,0,2,1:vw1); (0,0,2,2:vw1); (0,0,2,3:vw1); (0,1,2,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[54:55], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[60:61], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v69, v66, v64, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v69, -1, v69, s[60:61]               // clip if OOB. offset
buffer_load_dword v70, v69, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
_v_add_co_u32 v67, vcc, v64, 1                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[62:63], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v71, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v71, -1, v71, s[62:63]               // clip if OOB. offset
buffer_load_dword v72, v71, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
_v_add_co_u32 v67, vcc, v64, 2                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v73, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v73, -1, v73, s[64:65]               // clip if OOB. offset
buffer_load_dword v74, v73, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
_v_add_co_u32 v67, vcc, v64, 3                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[66:67], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v75, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v75, -1, v75, s[66:67]               // clip if OOB. offset
buffer_load_dword v76, v75, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
_v_add_co_u32 v67, vcc, v64, 32                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[68:69], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v77, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v77, -1, v77, s[68:69]               // clip if OOB. offset
buffer_load_dword v78, v77, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,1,1) */
_v_add_co_u32 v67, vcc, v64, 33                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v79, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v79, -1, v79, s[70:71]               // clip if OOB. offset
buffer_load_dword v80, v79, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,1,2) */
_v_add_co_u32 v67, vcc, v64, 34                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v81, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v81, -1, v81, s[72:73]               // clip if OOB. offset
buffer_load_dword v82, v81, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,1,3) */
_v_add_co_u32 v67, vcc, v64, 35                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v83, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v83, -1, v83, s[74:75]               // clip if OOB. offset
buffer_load_dword v84, v83, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
_v_add_co_u32 v65, vcc, v65, 1                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
_v_add_u32 v66, v66, s[sgprStrideC1J]              // ROWINC- Move cinRowPtr to next row
v_cmp_lt_u32 s[54:55], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[76:77], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
_v_add_lshl_u32 v85, v66, v64, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v85, -1, v85, s[76:77]               // clip if OOB. offset
buffer_load_dword v86, v85, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,1,0,1) */
_v_add_co_u32 v67, vcc, v64, 1                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[78:79], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
_v_add_lshl_u32 v87, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v87, -1, v87, s[78:79]               // clip if OOB. offset
buffer_load_dword v88, v87, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,1,0,2) */
_v_add_co_u32 v67, vcc, v64, 2                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[80:81], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
_v_add_lshl_u32 v89, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v89, -1, v89, s[80:81]               // clip if OOB. offset
buffer_load_dword v90, v89, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,1,0,3) */
_v_add_co_u32 v67, vcc, v64, 3                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[82:83], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
_v_add_lshl_u32 v91, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v91, -1, v91, s[82:83]               // clip if OOB. offset
buffer_load_dword v92, v91, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
_v_add_co_u32 v67, vcc, v64, 32                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[84:85], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v93, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v93, -1, v93, s[84:85]               // clip if OOB. offset
buffer_load_dword v94, v93, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,1,1,1) */
_v_add_co_u32 v67, vcc, v64, 33                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[86:87], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
_v_add_lshl_u32 v95, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v95, -1, v95, s[86:87]               // clip if OOB. offset
buffer_load_dword v96, v95, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,1,1,2) */
_v_add_co_u32 v67, vcc, v64, 34                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[88:89], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
_v_add_lshl_u32 v97, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v97, -1, v97, s[88:89]               // clip if OOB. offset
buffer_load_dword v98, v97, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,1,1,3) */
_v_add_co_u32 v67, vcc, v64, 35                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[90:91], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
_v_add_lshl_u32 v99, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v99, -1, v99, s[90:91]               // clip if OOB. offset
buffer_load_dword v100, v99, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
_v_add_co_u32 v65, vcc, v65, 1                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
_v_add_u32 v66, v66, s[sgprStrideC1J]              // ROWINC- Move cinRowPtr to next row
v_cmp_lt_u32 s[54:55], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[92:93], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v101, v66, v64, 0x2                // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v101, -1, v101, s[92:93]             // clip if OOB. offset
buffer_load_dword v102, v101, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,2,0,1) */
_v_add_co_u32 v67, vcc, v64, 1                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[94:95], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
_v_add_lshl_u32 v103, v66, v67, 0x2                // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v103, -1, v103, s[94:95]             // clip if OOB. offset
buffer_load_dword v104, v103, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,2,0,2) */
_v_add_co_u32 v67, vcc, v64, 2                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[96:97], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
_v_add_lshl_u32 v105, v66, v67, 0x2                // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v105, -1, v105, s[96:97]             // clip if OOB. offset
buffer_load_dword v106, v105, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,2,0,3) */
_v_add_co_u32 v67, vcc, v64, 3                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[98:99], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
_v_add_lshl_u32 v107, v66, v67, 0x2                // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v107, -1, v107, s[98:99]             // clip if OOB. offset
buffer_load_dword v108, v107, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,2,1,0) */
_v_add_co_u32 v67, vcc, v64, 32                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[100:101], v65, s[sgprSizeJ]         // coord1 < size1
s_and_b64 s[100:101], s[54:55], s[100:101]         // in0 && in1
_v_add_lshl_u32 v109, v66, v67, 0x2                // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v109, -1, v109, s[100:101]           // clip if OOB. offset
buffer_load_dword v110, v109, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc

/* rC *= alpha batchEements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 0, 2), (0, 1, 0, 3), (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 1, 2), (0, 0, 1, 3), (0, 1, 1, 0), (0, 1, 1, 1), (0, 1, 1, 2), (0, 1, 1, 3), (0, 0, 2, 0), (0, 0, 2, 1), (0, 0, 2, 2), (0, 0, 2, 3), (0, 1, 2, 0)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha
v_mul_f32 v[vgprValuC+1], s[sgprAlpha], v[vgprValuC+1] // *= alpha
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha
v_mul_f32 v[vgprValuC+3], s[sgprAlpha], v[vgprValuC+3] // *= alpha
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // *= alpha
v_mul_f32 v[vgprValuC+5], s[sgprAlpha], v[vgprValuC+5] // *= alpha
v_mul_f32 v[vgprValuC+6], s[sgprAlpha], v[vgprValuC+6] // *= alpha
v_mul_f32 v[vgprValuC+7], s[sgprAlpha], v[vgprValuC+7] // *= alpha
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // *= alpha
v_mul_f32 v[vgprValuC+9], s[sgprAlpha], v[vgprValuC+9] // *= alpha
v_mul_f32 v[vgprValuC+10], s[sgprAlpha], v[vgprValuC+10] // *= alpha
v_mul_f32 v[vgprValuC+11], s[sgprAlpha], v[vgprValuC+11] // *= alpha
v_mul_f32 v[vgprValuC+12], s[sgprAlpha], v[vgprValuC+12] // *= alpha
v_mul_f32 v[vgprValuC+13], s[sgprAlpha], v[vgprValuC+13] // *= alpha
v_mul_f32 v[vgprValuC+14], s[sgprAlpha], v[vgprValuC+14] // *= alpha
v_mul_f32 v[vgprValuC+15], s[sgprAlpha], v[vgprValuC+15] // *= alpha
v_mul_f32 v[vgprValuC+16], s[sgprAlpha], v[vgprValuC+16] // *= alpha
v_mul_f32 v[vgprValuC+17], s[sgprAlpha], v[vgprValuC+17] // *= alpha
v_mul_f32 v[vgprValuC+18], s[sgprAlpha], v[vgprValuC+18] // *= alpha
v_mul_f32 v[vgprValuC+19], s[sgprAlpha], v[vgprValuC+19] // *= alpha
v_mul_f32 v[vgprValuC+20], s[sgprAlpha], v[vgprValuC+20] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_mac_f32 v[vgprValuC+0], v70, s[sgprBeta]         // finalSum = sum*alpha + C*beta
buffer_store_dword v0, v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+1], v72, s[sgprBeta]         // finalSum = sum*alpha + C*beta
buffer_store_dword v1, v71, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+2], v74, s[sgprBeta]         // finalSum = sum*alpha + C*beta
buffer_store_dword v2, v73, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+3], v76, s[sgprBeta]         // finalSum = sum*alpha + C*beta
buffer_store_dword v3, v75, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+4], v78, s[sgprBeta]         // finalSum = sum*alpha + C*beta
buffer_store_dword v4, v77, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+5], v80, s[sgprBeta]         // finalSum = sum*alpha + C*beta
buffer_store_dword v5, v79, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+6], v82, s[sgprBeta]         // finalSum = sum*alpha + C*beta
buffer_store_dword v6, v81, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+7], v84, s[sgprBeta]         // finalSum = sum*alpha + C*beta
buffer_store_dword v7, v83, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+8], v86, s[sgprBeta]         // finalSum = sum*alpha + C*beta
buffer_store_dword v8, v85, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+9], v88, s[sgprBeta]         // finalSum = sum*alpha + C*beta
buffer_store_dword v9, v87, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+10], v90, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v10, v89, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+11], v92, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v11, v91, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+12], v94, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v12, v93, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+13], v96, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v13, v95, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+14], v98, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v14, v97, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+15], v100, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v15, v99, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+16], v102, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v16, v101, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+17], v104, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v17, v103, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+18], v106, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v18, v105, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+19], v108, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v19, v107, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+20], v110, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v20, v109, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (0,1,2,1:vw1); (0,1,2,2:vw1); (0,1,2,3:vw1); (0,0,3,0:vw1); (0,0,3,1:vw1); (0,0,3,2:vw1); (0,0,3,3:vw1); (0,1,3,0:vw1); (0,1,3,1:vw1); (0,1,3,2:vw1); (0,1,3,3:vw1); (1,0,0,0:vw1); (1,0,0,1:vw1); (1,0,0,2:vw1); (1,0,0,3:vw1); (1,1,0,0:vw1); (1,1,0,1:vw1); (1,1,0,2:vw1); (1,1,0,3:vw1); (1,0,1,0:vw1); (1,0,1,1:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,2,1,1) */
_v_add_co_u32 v67, vcc, v64, 33                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[60:61], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v69, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v69, -1, v69, s[60:61]               // clip if OOB. offset
buffer_load_dword v70, v69, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,2,1,2) */
_v_add_co_u32 v67, vcc, v64, 34                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[62:63], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v71, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v71, -1, v71, s[62:63]               // clip if OOB. offset
buffer_load_dword v72, v71, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,2,1,3) */
_v_add_co_u32 v67, vcc, v64, 35                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v73, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v73, -1, v73, s[64:65]               // clip if OOB. offset
buffer_load_dword v74, v73, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
_v_add_co_u32 v65, vcc, v65, 1                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
_v_add_u32 v66, v66, s[sgprStrideC1J]              // ROWINC- Move cinRowPtr to next row
v_cmp_lt_u32 s[54:55], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[66:67], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v75, v66, v64, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v75, -1, v75, s[66:67]               // clip if OOB. offset
buffer_load_dword v76, v75, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,0,1) */
_v_add_co_u32 v67, vcc, v64, 1                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[68:69], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v77, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v77, -1, v77, s[68:69]               // clip if OOB. offset
buffer_load_dword v78, v77, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,0,2) */
_v_add_co_u32 v67, vcc, v64, 2                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v79, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v79, -1, v79, s[70:71]               // clip if OOB. offset
buffer_load_dword v80, v79, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,0,3) */
_v_add_co_u32 v67, vcc, v64, 3                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v81, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v81, -1, v81, s[72:73]               // clip if OOB. offset
buffer_load_dword v82, v81, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,1,0) */
_v_add_co_u32 v67, vcc, v64, 32                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v83, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v83, -1, v83, s[74:75]               // clip if OOB. offset
buffer_load_dword v84, v83, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,1,1) */
_v_add_co_u32 v67, vcc, v64, 33                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[76:77], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
_v_add_lshl_u32 v85, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v85, -1, v85, s[76:77]               // clip if OOB. offset
buffer_load_dword v86, v85, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,1,2) */
_v_add_co_u32 v67, vcc, v64, 34                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[78:79], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
_v_add_lshl_u32 v87, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v87, -1, v87, s[78:79]               // clip if OOB. offset
buffer_load_dword v88, v87, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,1,3) */
_v_add_co_u32 v67, vcc, v64, 35                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[80:81], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
_v_add_lshl_u32 v89, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v89, -1, v89, s[80:81]               // clip if OOB. offset
buffer_load_dword v90, v89, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
_v_add_co_u32 v65, vcc, v65, 61                    // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s54, s[sgprStrideC1J], 61                // scale stride
_v_add_u32 v66, v66, s54                           // ROWINC- Move cinRowPtr to next row
v_cmp_lt_u32 s[54:55], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[82:83], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
_v_add_lshl_u32 v91, v66, v64, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v91, -1, v91, s[82:83]               // clip if OOB. offset
buffer_load_dword v92, v91, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,0,1) */
_v_add_co_u32 v67, vcc, v64, 1                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[84:85], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v93, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v93, -1, v93, s[84:85]               // clip if OOB. offset
buffer_load_dword v94, v93, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,0,2) */
_v_add_co_u32 v67, vcc, v64, 2                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[86:87], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
_v_add_lshl_u32 v95, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v95, -1, v95, s[86:87]               // clip if OOB. offset
buffer_load_dword v96, v95, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,0,3) */
_v_add_co_u32 v67, vcc, v64, 3                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[88:89], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
_v_add_lshl_u32 v97, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v97, -1, v97, s[88:89]               // clip if OOB. offset
buffer_load_dword v98, v97, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
_v_add_co_u32 v67, vcc, v64, 32                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[90:91], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
_v_add_lshl_u32 v99, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v99, -1, v99, s[90:91]               // clip if OOB. offset
buffer_load_dword v100, v99, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,1,1) */
_v_add_co_u32 v67, vcc, v64, 33                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[92:93], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v101, v66, v67, 0x2                // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v101, -1, v101, s[92:93]             // clip if OOB. offset
buffer_load_dword v102, v101, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,1,2) */
_v_add_co_u32 v67, vcc, v64, 34                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[94:95], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
_v_add_lshl_u32 v103, v66, v67, 0x2                // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v103, -1, v103, s[94:95]             // clip if OOB. offset
buffer_load_dword v104, v103, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,1,3) */
_v_add_co_u32 v67, vcc, v64, 35                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[96:97], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
_v_add_lshl_u32 v105, v66, v67, 0x2                // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v105, -1, v105, s[96:97]             // clip if OOB. offset
buffer_load_dword v106, v105, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,1,0,0) */
_v_add_co_u32 v65, vcc, v65, 1                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
_v_add_u32 v66, v66, s[sgprStrideC1J]              // ROWINC- Move cinRowPtr to next row
v_cmp_lt_u32 s[54:55], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[98:99], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
_v_add_lshl_u32 v107, v66, v64, 0x2                // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v107, -1, v107, s[98:99]             // clip if OOB. offset
buffer_load_dword v108, v107, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,1,0,1) */
_v_add_co_u32 v67, vcc, v64, 1                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[100:101], v65, s[sgprSizeJ]         // coord1 < size1
s_and_b64 s[100:101], s[54:55], s[100:101]         // in0 && in1
_v_add_lshl_u32 v109, v66, v67, 0x2                // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v109, -1, v109, s[100:101]           // clip if OOB. offset
buffer_load_dword v110, v109, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc

/* rC *= alpha batchEements=[(0, 1, 2, 1), (0, 1, 2, 2), (0, 1, 2, 3), (0, 0, 3, 0), (0, 0, 3, 1), (0, 0, 3, 2), (0, 0, 3, 3), (0, 1, 3, 0), (0, 1, 3, 1), (0, 1, 3, 2), (0, 1, 3, 3), (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 0, 2), (1, 0, 0, 3), (1, 1, 0, 0), (1, 1, 0, 1), (1, 1, 0, 2), (1, 1, 0, 3), (1, 0, 1, 0), (1, 0, 1, 1)] */
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+21] // *= alpha
v_mul_f32 v[vgprValuC+22], s[sgprAlpha], v[vgprValuC+22] // *= alpha
v_mul_f32 v[vgprValuC+23], s[sgprAlpha], v[vgprValuC+23] // *= alpha
v_mul_f32 v[vgprValuC+24], s[sgprAlpha], v[vgprValuC+24] // *= alpha
v_mul_f32 v[vgprValuC+25], s[sgprAlpha], v[vgprValuC+25] // *= alpha
v_mul_f32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+26] // *= alpha
v_mul_f32 v[vgprValuC+27], s[sgprAlpha], v[vgprValuC+27] // *= alpha
v_mul_f32 v[vgprValuC+28], s[sgprAlpha], v[vgprValuC+28] // *= alpha
v_mul_f32 v[vgprValuC+29], s[sgprAlpha], v[vgprValuC+29] // *= alpha
v_mul_f32 v[vgprValuC+30], s[sgprAlpha], v[vgprValuC+30] // *= alpha
v_mul_f32 v[vgprValuC+31], s[sgprAlpha], v[vgprValuC+31] // *= alpha
v_mul_f32 v[vgprValuC+32], s[sgprAlpha], v[vgprValuC+32] // *= alpha
v_mul_f32 v[vgprValuC+33], s[sgprAlpha], v[vgprValuC+33] // *= alpha
v_mul_f32 v[vgprValuC+34], s[sgprAlpha], v[vgprValuC+34] // *= alpha
v_mul_f32 v[vgprValuC+35], s[sgprAlpha], v[vgprValuC+35] // *= alpha
v_mul_f32 v[vgprValuC+36], s[sgprAlpha], v[vgprValuC+36] // *= alpha
v_mul_f32 v[vgprValuC+37], s[sgprAlpha], v[vgprValuC+37] // *= alpha
v_mul_f32 v[vgprValuC+38], s[sgprAlpha], v[vgprValuC+38] // *= alpha
v_mul_f32 v[vgprValuC+39], s[sgprAlpha], v[vgprValuC+39] // *= alpha
v_mul_f32 v[vgprValuC+40], s[sgprAlpha], v[vgprValuC+40] // *= alpha
v_mul_f32 v[vgprValuC+41], s[sgprAlpha], v[vgprValuC+41] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_mac_f32 v[vgprValuC+21], v70, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v21, v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+22], v72, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v22, v71, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+23], v74, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v23, v73, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+24], v76, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v24, v75, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+25], v78, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v25, v77, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+26], v80, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v26, v79, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+27], v82, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v27, v81, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+28], v84, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v28, v83, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+29], v86, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v29, v85, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+30], v88, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v30, v87, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+31], v90, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v31, v89, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+32], v92, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v32, v91, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+33], v94, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v33, v93, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+34], v96, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v34, v95, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+35], v98, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v35, v97, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+36], v100, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v36, v99, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+37], v102, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v37, v101, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+38], v104, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v38, v103, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+39], v106, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v39, v105, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+40], v108, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v40, v107, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+41], v110, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v41, v109, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #2 (d1,d0,vc1,vc0) = */
/*    (1,0,1,2:vw1); (1,0,1,3:vw1); (1,1,1,0:vw1); (1,1,1,1:vw1); (1,1,1,2:vw1); (1,1,1,3:vw1); (1,0,2,0:vw1); (1,0,2,1:vw1); (1,0,2,2:vw1); (1,0,2,3:vw1); (1,1,2,0:vw1); (1,1,2,1:vw1); (1,1,2,2:vw1); (1,1,2,3:vw1); (1,0,3,0:vw1); (1,0,3,1:vw1); (1,0,3,2:vw1); (1,0,3,3:vw1); (1,1,3,0:vw1); (1,1,3,1:vw1); (1,1,3,2:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(1,1,0,2) */
_v_add_co_u32 v67, vcc, v64, 2                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[60:61], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v69, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v69, -1, v69, s[60:61]               // clip if OOB. offset
buffer_load_dword v70, v69, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,1,0,3) */
_v_add_co_u32 v67, vcc, v64, 3                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[62:63], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v71, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v71, -1, v71, s[62:63]               // clip if OOB. offset
buffer_load_dword v72, v71, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,1,1,0) */
_v_add_co_u32 v67, vcc, v64, 32                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v73, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v73, -1, v73, s[64:65]               // clip if OOB. offset
buffer_load_dword v74, v73, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,1,1,1) */
_v_add_co_u32 v67, vcc, v64, 33                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[66:67], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v75, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v75, -1, v75, s[66:67]               // clip if OOB. offset
buffer_load_dword v76, v75, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,1,1,2) */
_v_add_co_u32 v67, vcc, v64, 34                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[68:69], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v77, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v77, -1, v77, s[68:69]               // clip if OOB. offset
buffer_load_dword v78, v77, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,1,1,3) */
_v_add_co_u32 v67, vcc, v64, 35                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v79, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v79, -1, v79, s[70:71]               // clip if OOB. offset
buffer_load_dword v80, v79, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,2,0,0) */
_v_add_co_u32 v65, vcc, v65, 1                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
_v_add_u32 v66, v66, s[sgprStrideC1J]              // ROWINC- Move cinRowPtr to next row
v_cmp_lt_u32 s[54:55], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v81, v66, v64, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v81, -1, v81, s[72:73]               // clip if OOB. offset
buffer_load_dword v82, v81, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,2,0,1) */
_v_add_co_u32 v67, vcc, v64, 1                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[74:75], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v83, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v83, -1, v83, s[74:75]               // clip if OOB. offset
buffer_load_dword v84, v83, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,2,0,2) */
_v_add_co_u32 v67, vcc, v64, 2                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[76:77], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
_v_add_lshl_u32 v85, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v85, -1, v85, s[76:77]               // clip if OOB. offset
buffer_load_dword v86, v85, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,2,0,3) */
_v_add_co_u32 v67, vcc, v64, 3                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[78:79], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
_v_add_lshl_u32 v87, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v87, -1, v87, s[78:79]               // clip if OOB. offset
buffer_load_dword v88, v87, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,2,1,0) */
_v_add_co_u32 v67, vcc, v64, 32                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[80:81], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
_v_add_lshl_u32 v89, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v89, -1, v89, s[80:81]               // clip if OOB. offset
buffer_load_dword v90, v89, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,2,1,1) */
_v_add_co_u32 v67, vcc, v64, 33                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[82:83], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
_v_add_lshl_u32 v91, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v91, -1, v91, s[82:83]               // clip if OOB. offset
buffer_load_dword v92, v91, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,2,1,2) */
_v_add_co_u32 v67, vcc, v64, 34                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[84:85], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v93, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v93, -1, v93, s[84:85]               // clip if OOB. offset
buffer_load_dword v94, v93, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,2,1,3) */
_v_add_co_u32 v67, vcc, v64, 35                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[86:87], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
_v_add_lshl_u32 v95, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v95, -1, v95, s[86:87]               // clip if OOB. offset
buffer_load_dword v96, v95, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,3,0,0) */
_v_add_co_u32 v65, vcc, v65, 1                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
_v_add_u32 v66, v66, s[sgprStrideC1J]              // ROWINC- Move cinRowPtr to next row
v_cmp_lt_u32 s[54:55], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[88:89], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
_v_add_lshl_u32 v97, v66, v64, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v97, -1, v97, s[88:89]               // clip if OOB. offset
buffer_load_dword v98, v97, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,3,0,1) */
_v_add_co_u32 v67, vcc, v64, 1                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[90:91], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
_v_add_lshl_u32 v99, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v99, -1, v99, s[90:91]               // clip if OOB. offset
buffer_load_dword v100, v99, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,3,0,2) */
_v_add_co_u32 v67, vcc, v64, 2                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[92:93], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v101, v66, v67, 0x2                // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v101, -1, v101, s[92:93]             // clip if OOB. offset
buffer_load_dword v102, v101, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,3,0,3) */
_v_add_co_u32 v67, vcc, v64, 3                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[94:95], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
_v_add_lshl_u32 v103, v66, v67, 0x2                // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v103, -1, v103, s[94:95]             // clip if OOB. offset
buffer_load_dword v104, v103, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,3,1,0) */
_v_add_co_u32 v67, vcc, v64, 32                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[96:97], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
_v_add_lshl_u32 v105, v66, v67, 0x2                // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v105, -1, v105, s[96:97]             // clip if OOB. offset
buffer_load_dword v106, v105, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,3,1,1) */
_v_add_co_u32 v67, vcc, v64, 33                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[98:99], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
_v_add_lshl_u32 v107, v66, v67, 0x2                // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v107, -1, v107, s[98:99]             // clip if OOB. offset
buffer_load_dword v108, v107, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,3,1,2) */
_v_add_co_u32 v67, vcc, v64, 34                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[100:101], v65, s[sgprSizeJ]         // coord1 < size1
s_and_b64 s[100:101], s[54:55], s[100:101]         // in0 && in1
_v_add_lshl_u32 v109, v66, v67, 0x2                // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v109, -1, v109, s[100:101]           // clip if OOB. offset
buffer_load_dword v110, v109, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc

/* rC *= alpha batchEements=[(1, 0, 1, 2), (1, 0, 1, 3), (1, 1, 1, 0), (1, 1, 1, 1), (1, 1, 1, 2), (1, 1, 1, 3), (1, 0, 2, 0), (1, 0, 2, 1), (1, 0, 2, 2), (1, 0, 2, 3), (1, 1, 2, 0), (1, 1, 2, 1), (1, 1, 2, 2), (1, 1, 2, 3), (1, 0, 3, 0), (1, 0, 3, 1), (1, 0, 3, 2), (1, 0, 3, 3), (1, 1, 3, 0), (1, 1, 3, 1), (1, 1, 3, 2)] */
v_mul_f32 v[vgprValuC+42], s[sgprAlpha], v[vgprValuC+42] // *= alpha
v_mul_f32 v[vgprValuC+43], s[sgprAlpha], v[vgprValuC+43] // *= alpha
v_mul_f32 v[vgprValuC+44], s[sgprAlpha], v[vgprValuC+44] // *= alpha
v_mul_f32 v[vgprValuC+45], s[sgprAlpha], v[vgprValuC+45] // *= alpha
v_mul_f32 v[vgprValuC+46], s[sgprAlpha], v[vgprValuC+46] // *= alpha
v_mul_f32 v[vgprValuC+47], s[sgprAlpha], v[vgprValuC+47] // *= alpha
v_mul_f32 v[vgprValuC+48], s[sgprAlpha], v[vgprValuC+48] // *= alpha
v_mul_f32 v[vgprValuC+49], s[sgprAlpha], v[vgprValuC+49] // *= alpha
v_mul_f32 v[vgprValuC+50], s[sgprAlpha], v[vgprValuC+50] // *= alpha
v_mul_f32 v[vgprValuC+51], s[sgprAlpha], v[vgprValuC+51] // *= alpha
v_mul_f32 v[vgprValuC+52], s[sgprAlpha], v[vgprValuC+52] // *= alpha
v_mul_f32 v[vgprValuC+53], s[sgprAlpha], v[vgprValuC+53] // *= alpha
v_mul_f32 v[vgprValuC+54], s[sgprAlpha], v[vgprValuC+54] // *= alpha
v_mul_f32 v[vgprValuC+55], s[sgprAlpha], v[vgprValuC+55] // *= alpha
v_mul_f32 v[vgprValuC+56], s[sgprAlpha], v[vgprValuC+56] // *= alpha
v_mul_f32 v[vgprValuC+57], s[sgprAlpha], v[vgprValuC+57] // *= alpha
v_mul_f32 v[vgprValuC+58], s[sgprAlpha], v[vgprValuC+58] // *= alpha
v_mul_f32 v[vgprValuC+59], s[sgprAlpha], v[vgprValuC+59] // *= alpha
v_mul_f32 v[vgprValuC+60], s[sgprAlpha], v[vgprValuC+60] // *= alpha
v_mul_f32 v[vgprValuC+61], s[sgprAlpha], v[vgprValuC+61] // *= alpha
v_mul_f32 v[vgprValuC+62], s[sgprAlpha], v[vgprValuC+62] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_mac_f32 v[vgprValuC+42], v70, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v42, v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+43], v72, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v43, v71, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+44], v74, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v44, v73, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+45], v76, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v45, v75, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+46], v78, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v46, v77, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+47], v80, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v47, v79, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+48], v82, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v48, v81, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+49], v84, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v49, v83, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+50], v86, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v50, v85, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+51], v88, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v51, v87, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+52], v90, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v52, v89, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+53], v92, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v53, v91, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+54], v94, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v54, v93, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+55], v96, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v55, v95, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+56], v98, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v56, v97, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+57], v100, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v57, v99, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+58], v102, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v58, v101, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+59], v104, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v59, v103, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+60], v106, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v60, v105, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+61], v108, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v61, v107, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_mac_f32 v[vgprValuC+62], v110, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v62, v109, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #3 (d1,d0,vc1,vc0) = */
/*    (1,1,3,3:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(1,3,1,3) */
_v_add_co_u32 v67, vcc, v64, 35                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v67, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[60:61], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v69, v66, v67, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v69, -1, v69, s[60:61]               // clip if OOB. offset
buffer_load_dword v70, v69, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C for beta calc

/* rC *= alpha batchEements=[(1, 1, 3, 3)] */
v_mul_f32 v[vgprValuC+63], s[sgprAlpha], v[vgprValuC+63] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_mac_f32 v[vgprValuC+63], v70, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v63, v69, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
s_branch label_GW_End_37                           // jump to end
label_GW_End_37:

label_0039:  /// KernelEnd
s_endpgm                                           // Kernel End
