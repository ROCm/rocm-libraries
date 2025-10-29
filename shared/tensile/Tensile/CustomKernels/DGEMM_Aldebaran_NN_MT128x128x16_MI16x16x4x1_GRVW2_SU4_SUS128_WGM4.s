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
.amdgcn_target "amdgcn-amd-amdhsa--gfx90a"
.text
.protected DGEMM_Aldebaran_NN_MT128x128x16_MI16x16x4x1_GRVW2_SU4_SUS128_WGM4
.globl DGEMM_Aldebaran_NN_MT128x128x16_MI16x16x4x1_GRVW2_SU4_SUS128_WGM4
.p2align 8
.type DGEMM_Aldebaran_NN_MT128x128x16_MI16x16x4x1_GRVW2_SU4_SUS128_WGM4,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel DGEMM_Aldebaran_NN_MT128x128x16_MI16x16x4x1_GRVW2_SU4_SUS128_WGM4
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_accum_offset 256 // accvgpr offset
  .amdhsa_next_free_vgpr 256 // vgprs
  .amdhsa_next_free_sgpr 73 // sgprs
  .amdhsa_group_segment_fixed_size 32768 // lds bytes
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
/* SubGroup= 16 x 16 */
/* VectorWidth=2 */
/* GlobalLoadVectorWidthA=2, GlobalLoadVectorWidthB=2 */
/* DirectToLdsA=False */
/* DirectToLdsB=False */
/* UseSgprForGRO=False */
.amdgpu_metadata
---
custom.config:
   ProblemType:
      OperationType: GEMM
      DataType: D
      TransposeA: False
      TransposeB: False
      UseBeta: True
      Batched: True
   MatrixInstruction: [ 16, 16, 4, 1 ]
   ThreadTile: [ 2, 128 ]
   WorkGroup: [ 64, 4, 1 ]
   DepthU: 16
   VectorWidth: 2
   SourceSwap: 1
   GlobalReadVectorWidth: 2
   StaggerUStride: 128
   StaggerU: 4
   WorkGroupMapping: 4
   AssertSizeMultiple: {3: 32}
amdhsa.version:
  - 1
  - 1
amdhsa.kernels:
  - .name: DGEMM_Aldebaran_NN_MT128x128x16_MI16x16x4x1_GRVW2_SU4_SUS128_WGM4
    .symbol: 'DGEMM_Aldebaran_NN_MT128x128x16_MI16x16x4x1_GRVW2_SU4_SUS128_WGM4.kd'
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
        .value_type:      f64
        .address_space:   generic
      - .name:            C
        .size:            8
        .offset:          32
        .value_kind:      global_buffer
        .value_type:      f64
        .address_space:   generic
      - .name:            A
        .size:            8
        .offset:          40
        .value_kind:      global_buffer
        .value_type:      f64
        .address_space:   generic
      - .name:            B
        .size:            8
        .offset:          48
        .value_kind:      global_buffer
        .value_type:      f64
        .address_space:   generic
      - .name:            alpha
        .size:            8
        .offset:          56
        .value_kind:      by_value
        .value_type:      f64
      - .name:            beta
        .size:            8
        .offset:          64
        .value_kind:      by_value
        .value_type:      f64
      - .name:            strideD0
        .size:            4
        .offset:          72
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideD1
        .size:            4
        .offset:          76
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC0
        .size:            4
        .offset:          80
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC1
        .size:            4
        .offset:          84
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA0
        .size:            4
        .offset:          88
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA1
        .size:            4
        .offset:          92
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB0
        .size:            4
        .offset:          96
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB1
        .size:            4
        .offset:          100
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree0
        .size:            4
        .offset:          104
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree1
        .size:            4
        .offset:          108
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree2
        .size:            4
        .offset:          112
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesSum0
        .size:            4
        .offset:          116
        .value_kind:      by_value
        .value_type:      u32
      - .name:            OrigStaggerUIter
        .size:            4
        .offset:          120
        .value_kind:      by_value
        .value_type:      i32
      - .name:            NumWorkGroups0
        .size:            4
        .offset:          124
        .value_kind:      by_value
        .value_type:      u32
      - .name:            NumWorkGroups1
        .size:            4
        .offset:          128
        .value_kind:      by_value
        .value_type:      u32
      - .name:            NumFullBlocks
        .size:            4
        .offset:          132
        .value_kind:      by_value
        .value_type:      u32
      - .name:            WgmRemainder1
        .size:            4
        .offset:          136
        .value_kind:      by_value
        .value_type:      u32
      - .name:            MagicNumberWgmRemainder1
        .size:            4
        .offset:          140
        .value_kind:      by_value
        .value_type:      u32
      - .name:            padding
        .size:            4
        .offset:          144
        .value_kind:      by_value
        .value_type:      u32
    .group_segment_fixed_size:   32768
    .kernarg_segment_align:      8
    .kernarg_segment_size:       152
    .max_flat_workgroup_size:    256
    .private_segment_fixed_size: 0
    .sgpr_count:                 73
    .sgpr_spill_count:           0
    .vgpr_count:                 256
    .vgpr_spill_count:           0
    .wavefront_size:             64
...
.end_amdgpu_metadata
DGEMM_Aldebaran_NN_MT128x128x16_MI16x16x4x1_GRVW2_SU4_SUS128_WGM4:

/******************************************/
/* Asm syntax workarounds                 */
/******************************************/
.macro _v_add_co_u32 dst:req, cc:req, src0:req, src1:req, dpp=
   v_add_co_u32 \dst, \cc, \src0, \src1 \dpp
.endm

.macro _v_add_u32 dst:req, src0:req, src1:req, dpp=
   v_add_u32 \dst, \src0, \src1 \dpp
.endm

.macro _v_add_i32 dst:req, src0:req, src1:req, dpp=
   v_add_i32 \dst, \src0, \src1 \dpp
.endm

.macro _v_addc_co_u32 dst:req, ccOut:req, src0:req, ccIn:req, src1:req, dpp=
   v_addc_co_u32 \dst, \ccOut, \src0, \ccIn, \src1 \dpp
.endm

.macro _v_sub_co_u32 dst:req, cc:req, src0:req, src1:req, dpp=
   v_sub_co_u32 \dst, \cc, \src0, \src1 \dpp
.endm

.macro _v_sub_u32 dst:req, src0:req, src1:req, dpp=
   v_sub_u32 \dst, \src0, \src1 \dpp
.endm

.macro _v_sub_i32 dst:req, src0:req, src1:req, dpp=
   v_sub_i32 \dst, \src0, \src1 \dpp
.endm

.macro _v_add_lshl_u32 dst:req, src0:req, src1:req, shiftCnt:req
    v_add_lshl_u32 \dst, \src0, \src1, \shiftCnt
.endm

.macro _v_lshl_add_u32 dst:req, src0:req, src1:req, shiftCnt:req
    v_lshl_add_u32 \dst, \src0, \src1, \shiftCnt
.endm

.macro _v_lshl_or_b32 dst:req, src0:req, shiftCnt:req, src1:req
    v_lshl_or_b32 \dst, \src0, \shiftCnt, \src1
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
.macro _v_mac_f32 c:req, a:req, b:req
    v_mac_f32 \c, \a, \b
.endmacro

/******************************************/
/* Magic div and mod functions            */
/******************************************/
.macro V_MAGIC_DIV dstIdx:req, dividend:req, magicNumber:req, magicShift:req, magicA:req
    v_mul_hi_u32 v[\dstIdx+1], \dividend, \magicNumber
    v_mul_lo_u32 v[\dstIdx+0], \dividend, \magicA
    _v_add_u32 v[\dstIdx+0], v[\dstIdx+0], v[\dstIdx+1]
    v_lshrrev_b32 v[\dstIdx+0], \magicShift, v[\dstIdx+0]
.endm

/******************************************/
/* VGPR Assignments                       */
/******************************************/
/* ValuC range: [0-128), serializedStore enabled */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx */
.set vgprValuA_X0_I0, 128
.set vgprValuA_X1_I0, 132
.set vgprValuA_X2_I0, 136
.set vgprValuA_X3_I0, 140
.set vgprValuB_X0_I0, 144
.set vgprValuB_X1_I0, 160
.set vgprValuB_X2_I0, 176
.set vgprValuB_X3_I0, 192
.set vgprLocalWriteAddrA, 208
.set vgprLocalWriteAddrB, 209
.set vgprGlobalReadOffsetA, 210
.set vgprGlobalReadOffsetB, 214
.set vgprG2LA, 218
.set vgprValuA_X0_I1, 218
.set vgprValuA_X1_I1, 222
.set vgprValuA_X2_I1, 226
.set vgprValuA_X3_I1, 230
.set vgprG2LB, 234
.set vgprLocalReadAddrA, 250
.set vgprLocalReadAddrB, 251
.set vgprSerial, 252
/* Num VGPR=256 */
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
/* offsets pre-applied */
.set sgprAlpha, 44
.set sgprBeta, 46
.set sgprStridesD, 48
.set sgprStridesC, 50
.set sgprStridesA, 52
.set sgprStridesB, 54
.set sgprSizesFree, 56
.set sgprSizesSum, 59
.set sgprOrigStaggerUIter, 60
.set sgprNumWorkGroups0, 61
.set sgprNumWorkGroups1, 62
.set sgprNumFullBlocks, 63
.set sgprWgmRemainder1, 64
.set sgprMagicNumberWgmRemainder1, 65
.set sgprShadowLimitA, 36
.set sgprShadowLimitB, 38
.set sgprStaggerUIter, 7
.set sgprWrapUA, 40
.set sgprWrapUB, 42
.set sgprGlobalReadIncsA, 66
.set sgprGlobalReadIncsB, 67
/* max SGPR=73 */

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
.set DepthU, 16
.set GSU, 1
.set BpeA, 8
.set BpeALog2, 3
.set BpeB, 8
.set BpeBLog2, 3
/* Number of elements to shift-left SRD */
.set SrdShiftLeftA, 2
.set SrdShiftLeftB, 2
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
.macro GLOBAL_OFFSET_A vgprAddr:req vgprOffset0I:req vgprOffsetL:req vgprTmp:req
v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideAL], v[\vgprOffsetL] // mul d1 lower
_v_add_co_u32 v[\vgprAddr+0], vcc, v[\vgprOffset0I], v[\vgprTmp+0] // accumulate K lower
_v_add_u32 v[\vgprAddr+0], 0x2, v[\vgprAddr+0]     // add prepad for pointer shift
v_lshlrev_b32 v[\vgprAddr+0], 0x3, v[\vgprAddr+0]  // offset *= bytes/element
.endm

/* Global Offset B */
.macro GLOBAL_OFFSET_B vgprAddr:req vgprOffsetL:req vgprOffset1J:req vgprTmp:req
v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideB1J], v[\vgprOffset1J] // mul d1 lower
_v_add_co_u32 v[\vgprAddr+0], vcc, v[\vgprOffsetL], v[\vgprTmp+0] // accumulate K lower
_v_add_u32 v[\vgprAddr+0], 0x2, v[\vgprAddr+0]     // add prepad for pointer shift
v_lshlrev_b32 v[\vgprAddr+0], 0x3, v[\vgprAddr+0]  // offset *= bytes/element
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
/* Allocate Resources                     */
/******************************************/

s_setprio 3                                        // optimization store
s_mov_b32 m0, 0x9000                               // LDS clamp at 36864 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id

/* Load Kernel Args */
s_load_dwordx16 s[24:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x0 //
s_load_dwordx16 s[48:63], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x40 //
s_load_dwordx2 s[64:65], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x80 //
s_waitcnt lgkmcnt(0)                               // wait for 160 bytes of kern args
s_mov_b32 s44, s36
s_mov_b32 s45, s37
s_mov_b32 s46, s38
s_mov_b32 s47, s39

/* Short circuit condition if Alpha == 0, then sumDims=0 */
v_cmp_eq_f64 vcc, s[sgprAlpha:sgprAlpha+1], 0.0    // Alpha == 0.0 ?
s_cbranch_vccz label_AlphaNonZero                  // branch if Alpha != 0
s_mov_b32 s[sgprSizesSum+0], 0x0                   // Set summation dim=0 if Alpha == 0
label_AlphaNonZero:


/******************************************/
/* Local Read Addresses                   */
/******************************************/


/* local read addresses: tile assignments a/b */

/*lr0I*/
v_and_b32 v2, 63, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
                                                   // 1. N offset: nOffset = nIdx * nStride(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v0, 4, v2                            // 2. block offset: bnIdx = wtid / dividedForBlkId(16)
v_and_b32 v0, 0, v0                                // 2. block offset: bnIdx = bnIdx % num1DBlocks(1)
v_lshlrev_b32 v0, 0x4, v0                          // 2. block offset: bnOffset = bnIdx * strideBlock(16)
_v_add_u32 v1, v0, v1                              // 3. add N and block offset: bnOffset = block and N offset
v_lshlrev_b32 v1, 0x1, v1                          // 3. apply VectorWidth: bnOffset = bnOffset * vw(2)
v_lshrrev_b32 v2, 4, v2                            // 4. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshlrev_b32 v2, 0x8, v2                          // 4. K offset: lrKOffset = kIdx * mStride(256)
_v_add_u32 v1, v2, v1                              // 5. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v0, 6, v[vgprSerial]                 // 6. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
v_and_b32 v0, 3, v0                                // 6. wave offset in M dimen: wtid0 = wtid / num1DWaves(4)
v_lshlrev_b32 v0, 0x5, v0                          // 6. wave offset in M dimen: wOffset = wtid0 * W0Stride(32)
_v_add_u32 v1, v0, v1                              // 7. final local read offset: flrOffset = lrOffset + WOffset
/*lr1J*/
v_and_b32 v3, 63, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v2, 15, v3                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v2, 0x4, v2                          // 1. N offset: nOffset = nIdx * nStride(16)
v_lshrrev_b32 v0, 4, v3                            // 2. block offset: bnIdx = wtid / dividedForBlkId(16)
v_and_b32 v0, 0, v0                                // 2. block offset: bnIdx = bnIdx % num1DBlocks(1)
v_lshlrev_b32 v0, 0x8, v0                          // 2. block offset: bnOffset = bnIdx * strideBlock(256)
_v_add_u32 v2, v0, v2                              // 3. add N and block offset: bnOffset = block and N offset
                                                   // 3. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v3, 4, v3                            // 4. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshlrev_b32 v3, 0x1, v3                          // 4. K offset: lrKOffset = kIdx * mStride(2)
_v_add_u32 v2, v3, v2                              // 5. offset in wave: lrOffset = bnOffset + lrKOffset


/* local read addresses: final offsets a */

// v_lshrrev_b32 v0, 8, v[vgprSerial]                 // LSU offset: sgid = Serial / subGroup(256)
// s_mov_b32 s68, 128                                 // LSU offset: stirde = MT0(128) + PAD0(0)
// v_mul_lo_u32 v0, s68, v0                           // LSU offset: lsuoffset = sgid*(MT0+PAD)
// _v_add_lshl_u32 v[vgprLocalReadAddrA], v0, v1, 0x3 // Final Offset: offset = (lro0*VW+lsuoffset)*bpe


/* local read addresses: final offsets b */

v_lshrrev_b32 v0, 8, v[vgprSerial]                 // LSU offset: sgid = Serial / subGroup(256)
s_mov_b32 s68, 128                                 // LSU offset: stirde = MT1(128) + PAD1(0)
v_mul_lo_u32 v0, s68, v0                           // LSU offset: lsuoffset = sgid*(MT1+PAD)
_v_add_lshl_u32 v[vgprLocalReadAddrB], v0, v2, 0x3 // Final Offset: offset = (lro1*VW+lsuoffset)*bpe
v_lshrrev_b32 v1, 7, v[vgprLocalReadAddrB]         // Final Offset: padding 4 per block 128
v_lshlrev_b32 v1, 0x5, v1                          // Final Offset: padding 4 per block 128
_v_add_u32 v[vgprLocalReadAddrB], v1, v[vgprLocalReadAddrB] // Final Offset: add padding 4 per block 128


/* local read addresses: declare addresses a */

/* N/A */


/* local read addresses: declare addresses b */

// _v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, 0x4000, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)



/******************************************/
/* Begin setupNewTile, isPap=False           */
/******************************************/


/* global read addresses: work-group */

/* graWorkGroup mapping */
s_mov_b32 s71, 0x20000001L                         // magic number for WGM==4
s_mul_hi_u32 s69, s[sgprWorkGroup1], s71           // s_magic mul
s_mul_i32 s68, s[sgprWorkGroup1], s71              // s_magic mul
s_lshr_b64 s[68:69], s[68:69], 31                  // sMagicDiv
s_mul_i32 s69, s68, 4                              // quotient * non-magic divisor
s_sub_u32 s69, s[sgprWorkGroup1], s69              // WorkGroup1=remainder
s_mul_i32 s69, s69, s[sgprNumWorkGroups0]          // (wg1 % WGM)*nwg0
s_add_u32 s69, s69, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*nwg0
s_cmp_ge_u32 s68, s[sgprNumFullBlocks]             // blockId >= numFullBlocks ?
s_cmov_b32 s71, s[sgprMagicNumberWgmRemainder1]    //
s_cselect_b32 s70, s[sgprWgmRemainder1], 4         //
s_mul_hi_u32 s3, s69, s71                          // s_magic mul
s_mul_i32 s2, s69, s71                             // s_magic mul
s_lshr_b64 s[2:3], s[2:3], 31                      // sMagicDiv
s_mul_i32 s[sgprWorkGroup1], s[sgprWorkGroup0], s70 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup1], s69, s[sgprWorkGroup1] // WorkGroup1=remainder
s_mul_i32 s68, s68, 4                              // blockId * WGM
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s68 // wg1 += blockId * WGM


/* global read addresses: tile offset assignment a */

/* LVCA = 64 */
/* v0 = (local)groA-tile = serial%LVCA (note (wgA*MTA) will be added to SRD) */
/* v1 = groA-unroll = serial/LVCA */
v_and_b32 v0, 15, v[vgprSerial]     // v0 = v[vgprSerial] % 16
v_lshlrev_b32 v0, 0x1, v0           // v0 = v0 * 2
v_lshrrev_b32 v1, 6, v[vgprSerial]  // v1 = v[vgprSerial] / 64
v_lshlrev_b32 v1, 5, v1             // v1 = v1 * 32
v_add_u32 v0, v0, v1                // v0 = v0 + v1

v_and_b32 v1, 63, v[vgprSerial]     // v1 = (v[vgprSerial] % 64) / 16
v_lshrrev_b32 v1, 4, v1             // v1 = (v[vgprSerial] % 64) / 16
v_lshlrev_b32 v1, 1, v1             // v1 = v1 * 2

// v_lshrrev_b32 v1, 6, v[vgprSerial]                 // v1 = v[vgprSerial] / 64
// v_and_b32 v0, 63, v[vgprSerial]                    // v0 = v[vgprSerial] % 64
// /* gro-tile *= glvw */
// v_lshlrev_b32 v0, 0x1, v0                          // v0 = v0 * 2


/* global read addresses: tile offset assignment b */

/* LVCB = 8 */
/* v2 = (local)groB-tile = serial/LVCB (note (wgB*MTB) will be added to SRD) */
/* v3 = groB-unroll = serial%LVCB */
v_and_b32 v6, 63, v[vgprSerial]                    // v6 = v[vgprSerial] % 64
v_lshrrev_b32 v2, 3, v6                            // v2 = v6 / 8
v_and_b32 v3, 7, v6                                // v3 = v6 % 8
v_readfirstlane_b32 s68, v[vgprSerial]             // WaveIdxWavefrontWidth
s_lshr_b32 s68, s68, 0x6                           // WaveId
s_mul_i32 s68, s68, 32                             // Global Read Wave: each wave loads continuous lsp(8)*nrp(4) columns
_v_add_u32 v2, s68, v2                             // Global Read Wave: add back to cloumn index
/* gro-unroll *= glvw */
v_lshlrev_b32 v3, 0x1, v3                          // v3 = v3 * 2


/* global read addresses: unroll assignment a */

/* v1 */


/* global read addresses: unroll assignment b */

/* v3 */


/* global read addresses: other free assignments */

/* s[sgprWorkGroup2] */


/* global read addresses: tile offsets a */

v_mov_b32 v4, v0                                   // groA0I_0


/* global read addresses: tile offsets b */

v_mov_b32 v5, v2                                   // groB1J_0
_v_add_co_u32 v6, vcc, 8, v5                       // groB1J_1 += LSPB
_v_add_co_u32 v7, vcc, 8, v6                       // groB1J_2 += LSPB
_v_add_co_u32 v8, vcc, 8, v7                       // groB1J_3 += LSPB


/* global read addresses: unroll offsets a */

v_mov_b32 v9, v1                                   // groAL_0
_v_add_co_u32 v10, vcc, 1, v9                      // groAL_1 + LSPA
_v_add_co_u32 v11, vcc, 8, v9                     // groAL_2 + LSPA
_v_add_co_u32 v12, vcc, 9, v9                     // groAL_3 + LSPA


/* global read addresses: unroll offsets b */

v_mov_b32 v13, v3                                  // groBL_0


/* global read addresses: shift a */

s_mul_i32 s68, s[sgprWorkGroup0], 128              // WorkGroup[01] * MT
s_sub_u32 s68, s[sgprSizeI], s68                   // edge = Size0I - WG*MT
s_sub_u32 s68, s68, 2                              // edge -= margin(2)
v_mov_b32 v14, s68                                 // edge vgpr = Size0I- WG*MT - margin(2)
_v_add_co_u32 v15, vcc, v14, 2                     // shiftedEdge = edge + srdShiftLeft(2)
_v_add_co_u32 v16, vcc, v4, 2                      // shiftedOffset = offset + srdShiftLeft(2)
v_cmp_lt_u32 s[68:69], v16, v15                    // shiftedOffset < shiftedEdge
v_cndmask_b32 v4, v14, v4, s[68:69]                // offset = (offset < edge) ? offset(v4) : edge(v14)


/* global read addresses: final offsets a */

GLOBAL_OFFSET_A vgprGlobalReadOffsetA+0,  4,  9, 14 // gROA_0_0_0_0
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+1,  4, 10, 14 // gROA_0_0_1_0
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+2,  4, 11, 14 // gROA_0_0_2_0
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+3,  4, 12, 14 // gROA_0_0_3_0


/* global read addresses: final offsets b */

GLOBAL_OFFSET_B vgprGlobalReadOffsetB+0, 13,  5, 9 // gROB_0_0_0_0
GLOBAL_OFFSET_B vgprGlobalReadOffsetB+1, 13,  6, 9 // gROB_0_0_1_0
GLOBAL_OFFSET_B vgprGlobalReadOffsetB+2, 13,  7, 9 // gROB_0_0_2_0
GLOBAL_OFFSET_B vgprGlobalReadOffsetB+3, 13,  8, 9 // gROB_0_0_3_0


/* global read addresses: addresses a */

/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s71, s[sgprWorkGroup0], 128           // WorkGroup[01] * MT
s_mul_i32 s70, s[sgprWorkGroup0], 128              // WorkGroup[01] * MT
s_sub_u32 s[sgprShadowLimitA+0], s[sgprTensor2dSizeA], s70 // sub tileStart
s_subb_u32 s[sgprShadowLimitA+1], s[sgprTensor2dSizeA+1], s71 // sub tileStart
s_lshl_b64 s[sgprShadowLimitA:sgprShadowLimitA+1], s[sgprShadowLimitA:sgprShadowLimitA+1], 0x3 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_mul_hi_u32 s69, s[sgprStrideAK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s68, s[sgprStrideAK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s70, s70, s68                            // accum wg term to tilestart
s_addc_u32 s71, s71, s69                           // accum wg term to tilestart
s_lshl_b64 s[70:71], s[70:71], 0x3                 // tileStart *= BPE
s_add_u32 s[sgprSrdA+0], s[sgprAddressA+0], s70    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprAddressA+1], s71   // SRD base = Address+ tileStart1
s_sub_u32 s[sgprSrdA+0], s[sgprSrdA+0], 16         // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0         // pre-pad to make room for possible pointer shift
s_mov_b32 s[sgprSrdA+3], Srd127_96                 // Set bits 127_96 in SRD


/* global read addresses: addresses b */

/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s71, s[sgprWorkGroup1], 128           // WorkGroup[01] * MT
s_mul_i32 s70, s[sgprWorkGroup1], 128              // WorkGroup[01] * MT
s_mul_hi_u32 s71, s70, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s70, s70, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_sub_u32 s[sgprShadowLimitB+0], s[sgprTensor2dSizeB], s70 // sub tileStart
s_subb_u32 s[sgprShadowLimitB+1], s[sgprTensor2dSizeB+1], s71 // sub tileStart
s_lshl_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], 0x3 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_mul_hi_u32 s69, s[sgprStrideBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s68, s[sgprStrideBK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s70, s70, s68                            // accum wg term to tilestart
s_addc_u32 s71, s71, s69                           // accum wg term to tilestart
s_lshl_b64 s[70:71], s[70:71], 0x3                 // tileStart *= BPE
s_add_u32 s[sgprSrdB+0], s[sgprAddressB+0], s70    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprAddressB+1], s71   // SRD base = Address+ tileStart1
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

// v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x80, v1     // lwAL**(MTA + PAD)
// _v_add_lshl_u32 v[vgprLocalWriteAddrA], v0, v[vgprLocalWriteAddrA], 0x3 // lwFOA = (lwAA + lwAL*(MT0I+PAD))*bpe


/* local write addresses: first offset b */

v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x10, v2     // lwBL**(DepthU_Compute + PAD)
_v_add_lshl_u32 v[vgprLocalWriteAddrB], v3, v[vgprLocalWriteAddrB], 0x3 // lwFOB = (lwBB + lwBL*(DepthU+PAD))*bpe
v_lshrrev_b32 v3, 7, v[vgprLocalWriteAddrB]        // padding 4 per block 128
v_lshlrev_b32 v3, 0x5, v3                          // padding 4 per block 128
_v_add_u32 v[vgprLocalWriteAddrB], v3, v[vgprLocalWriteAddrB] // add padding 4 per block 128
// _v_add_co_u32 v[vgprLocalWriteAddrB], vcc, 0x4000, v[vgprLocalWriteAddrB] // lwFOB = lwB1J + lwBL*MT1J + LDS_OFFSET_B=2048*8







/* declare loop num iterations */


s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 4 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 16
s_mov_b32 s[sgprOrigLoopCounter], s[sgprLoopCounterL] // copy loop counter

s_and_b32 s[sgprStaggerUIter], s[sgprOrigStaggerUIter], s[sgprWorkGroup0] // Compute actual stagger start for this tile
s_lshl_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], 0 // shift by StaggerUStride


/* SRDs += (StaggerUIter) * GlobalReadIncsA+0 */
s_mul_hi_i32 s69, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_i32 s68, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_hi_i32 s[sgprWrapUA+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUA+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0], s[sgprWrapUA+0] // remove one iteration
s_subb_u32 s[sgprWrapUA+1], 0, s[sgprWrapUA+1]     // remove one iteration
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s68        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], s69      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s68 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s69 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32


/* SRDs += (StaggerUIter) * GlobalReadIncsB+0 */
s_mul_hi_i32 s69, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_i32 s68, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_hi_i32 s[sgprWrapUB+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUB+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0], s[sgprWrapUB+0] // remove one iteration
s_subb_u32 s[sgprWrapUB+1], 0, s[sgprWrapUB+1]     // remove one iteration
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s68        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], s69      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s68 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s69 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32
s_add_u32 s[sgprStaggerUIter], s[sgprStaggerUIter], 2 // Subtract (PGR-1); StaggerUIter now contains target iteration to wrap

/* local read addresses: init pointers a */


/* localReadInitPointers */

/* local read addresses: init pointers b */


/* localReadInitPointers */


/* prefetch: global -> local */

s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?
// s_setprio 0                                        // optimization store
s_cbranch_scc1 ShadowInitStart_9                   // skip to ShadowInitStart iter b/c numIter==0

buffer_load_dwordx4 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_1_0
buffer_load_dwordx4 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_2_0
buffer_load_dwordx4 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_3_0

buffer_load_dwordx4 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_1_0
buffer_load_dwordx4 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_2_0
buffer_load_dwordx4 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_3_0




/* global read inc A loopL */
s_add_u32 s70, s[sgprLoopCounterL], 1              // remove pf(1)
s_cmp_eq_u32 s[sgprStaggerUIter], s70              // Is this wrapIter? (pf)
s_cselect_b32 s68, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s69, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s68        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], s69      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s68 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s69 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s70, s[sgprLoopCounterL], 1              // remove pf(1)
s_cmp_eq_u32 s[sgprStaggerUIter], s70              // Is this wrapIter? (pf)
s_cselect_b32 s68, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s69, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s68        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], s69      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s68 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s69 // limit -= inc)
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


s_mul_i32 s70, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_mul_hi_u32 s69, s70, s[sgprStrideC1J]            // CScale s70 by Stride
s_mul_i32 s68, s70, s[sgprStrideC1J]               // CScale s70 by Stride
s_lshl_b64 s[68:69], s[68:69], 3                   // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s68        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s69       // add hi to SRD
s_mul_hi_u32 s69, s70, s[sgprStrideD1J]            // Scale s70 by Stride
s_mul_i32 s68, s70, s[sgprStrideD1J]               // Scale s70 by Stride
s_lshl_b64 s[68:69], s[68:69], 3                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s68        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s69       // add hi to SRD

s_mul_hi_u32 s69, s[sgprWorkGroup2], s[sgprStrideCK] // CScale s[sgprWorkGroup2] by Stride
s_mul_i32 s68, s[sgprWorkGroup2], s[sgprStrideCK]  // CScale s[sgprWorkGroup2] by Stride
s_lshl_b64 s[68:69], s[68:69], 3                   // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s68        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s69       // add hi to SRD
s_mul_hi_u32 s69, s[sgprWorkGroup2], s[sgprStrideDK] // Scale s[sgprWorkGroup2] by Stride
s_mul_i32 s68, s[sgprWorkGroup2], s[sgprStrideDK]  // Scale s[sgprWorkGroup2] by Stride
s_lshl_b64 s[68:69], s[68:69], 3                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s68        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s69       // add hi to SRD



/* initC: remove C-tile 0-128 from pool */
v_mov_b32 v208, 15728640                           // set out-of-bound addr
ds_read_b32 v[vgprValuC+0], v208, offset:0         // initC
ds_read_b32 v[vgprValuC+1], v208, offset:0         // initC
ds_read_b32 v[vgprValuC+2], v208, offset:0         // initC
ds_read_b32 v[vgprValuC+3], v208, offset:0         // initC
ds_read_b32 v[vgprValuC+4], v208, offset:0         // initC
ds_read_b32 v[vgprValuC+5], v208, offset:0         // initC
ds_read_b32 v[vgprValuC+6], v208, offset:0         // initC
ds_read_b32 v[vgprValuC+7], v208, offset:0         // initC
ds_read_b32 v[vgprValuC+8], v208, offset:0         // initC
ds_read_b32 v[vgprValuC+9], v208, offset:0         // initC
ds_read_b32 v[vgprValuC+10], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+11], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+12], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+13], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+14], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+15], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+16], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+17], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+18], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+19], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+20], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+21], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+22], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+23], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+24], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+25], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+26], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+27], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+28], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+29], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+30], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+31], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+32], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+33], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+34], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+35], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+36], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+37], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+38], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+39], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+40], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+41], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+42], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+43], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+44], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+45], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+46], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+47], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+48], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+49], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+50], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+51], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+52], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+53], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+54], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+55], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+56], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+57], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+58], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+59], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+60], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+61], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+62], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+63], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+64], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+65], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+66], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+67], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+68], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+69], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+70], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+71], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+72], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+73], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+74], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+75], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+76], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+77], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+78], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+79], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+80], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+81], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+82], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+83], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+84], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+85], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+86], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+87], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+88], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+89], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+90], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+91], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+92], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+93], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+94], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+95], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+96], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+97], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+98], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+99], v208, offset:0        // initC
ds_read_b32 v[vgprValuC+100], v208, offset:0       // initC
ds_read_b32 v[vgprValuC+101], v208, offset:0       // initC
ds_read_b32 v[vgprValuC+102], v208, offset:0       // initC
ds_read_b32 v[vgprValuC+103], v208, offset:0       // initC
ds_read_b32 v[vgprValuC+104], v208, offset:0       // initC
ds_read_b32 v[vgprValuC+105], v208, offset:0       // initC
ds_read_b32 v[vgprValuC+106], v208, offset:0       // initC
ds_read_b32 v[vgprValuC+107], v208, offset:0       // initC
ds_read_b32 v[vgprValuC+108], v208, offset:0       // initC
ds_read_b32 v[vgprValuC+109], v208, offset:0       // initC
ds_read_b32 v[vgprValuC+110], v208, offset:0       // initC
ds_read_b32 v[vgprValuC+111], v208, offset:0       // initC
ds_read_b32 v[vgprValuC+112], v208, offset:0       // initC
ds_read_b32 v[vgprValuC+113], v208, offset:0       // initC
ds_read_b32 v[vgprValuC+114], v208, offset:0       // initC
ds_read_b32 v[vgprValuC+115], v208, offset:0       // initC
ds_read_b32 v[vgprValuC+116], v208, offset:0       // initC
ds_read_b32 v[vgprValuC+117], v208, offset:0       // initC
ds_read_b32 v[vgprValuC+118], v208, offset:0       // initC
ds_read_b32 v[vgprValuC+119], v208, offset:0       // initC
ds_read_b32 v[vgprValuC+120], v208, offset:0       // initC
ds_read_b32 v[vgprValuC+121], v208, offset:0       // initC
ds_read_b32 v[vgprValuC+122], v208, offset:0       // initC
ds_read_b32 v[vgprValuC+123], v208, offset:0       // initC
ds_read_b32 v[vgprValuC+124], v208, offset:0       // initC
ds_read_b32 v[vgprValuC+125], v208, offset:0       // initC
ds_read_b32 v[vgprValuC+126], v208, offset:0       // initC
ds_read_b32 v[vgprValuC+127], v208, offset:0       // initC

/* initC: remove AB-tile 128-208 from pool */

s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?

/* after InitC, skip to end of prefetch last iter if numIter==0 */
s_cbranch_scc0 label_NoBranch_10                   // Only branch on scc1
s_getpc_B64 s[68:69]                               // addr of next instr
s_add_i32 s70, PrefetchGlobalLastIterEnd_4, 0x4    // target branch offset
s_cmp_ge_i32 s70, 0x0                              // check positive or negative
s_cbranch_scc1 label_Positive_11                   // jump when positive
s_abs_i32 s70, s70                                 // abs offset
s_sub_u32 s68, s68, s70                            // sub target branch offset
s_subb_u32 s69, s69, 0                             // sub high and carry
s_setpc_b64 s[68:69]                               // branch to PrefetchGlobalLastIterEnd_4
label_Positive_11:
s_add_u32 s68, s68, s70                            // add target branch offset
s_addc_u32 s69, s69, 0                             // add high and carry
s_setpc_b64 s[68:69]                               // branch to PrefetchGlobalLastIterEnd_4
label_NoBranch_10:

s_waitcnt vmcnt(4)                                 // lgkmcnt=-1 vmcnt=08wait for global read

/* local write b */
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+4:vgprG2LB+4+3] offset:1280 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 1280
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+8:vgprG2LB+8+3] offset:2560 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 2560
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+12:vgprG2LB+12+3] offset:3840 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 3840


/* local write swap a */



/* local write swap b */




s_cmp_eq_u32 s[sgprLoopCounterL] 0x1               // PGR=2 but only 1 loop
s_cbranch_scc1 label_0012                          // PGR=2 but only 1 loop


buffer_load_dwordx4 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_1_0
buffer_load_dwordx4 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_2_0
buffer_load_dwordx4 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_3_0

buffer_load_dwordx4 v[vgprValuA_X0_I1+0:vgprValuA_X0_I1+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprValuA_X0_I1+4:vgprValuA_X0_I1+4+3], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_1_0
buffer_load_dwordx4 v[vgprValuA_X0_I1+8:vgprValuA_X0_I1+8+3], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_2_0
buffer_load_dwordx4 v[vgprValuA_X0_I1+12:vgprValuA_X0_I1+12+3], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_3_0

label_0012:                                        //

s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-10prefetch wait for local write

// Skip force waitcnt0
s_barrier //


/* local read prefetch b */

ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:2560 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:5120 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB] offset:7680 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB] offset:10240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB] offset:12800 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=5 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB] offset:15360 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=6 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB] offset:17920 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=7 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0


/* local read inc a */

/* N/A, lro->128 */
/* self.localReadDoCntA 1 self.localReadDoCntB 1 */


/* local read inc b */

/* N/A, lro->8 */
/* self.localReadDoCntA 1 self.localReadDoCntB 1 */



/******************************************/
/* Unrolled Loop(s) - Begin               */
/******************************************/

openLoopL_13:
s_cmp_eq_u32 s[sgprLoopCounterL], 0x1              // LoopCounterL < EndCounter
s_cbranch_scc1 label_0014                          // PGR=2 but only 1 loop, toPGR1
s_cmp_le_u32 s[sgprLoopCounterL], 0x2              // LoopCounterL < EndCounter
s_cbranch_scc1 LoopEndL_2                          // do not enter LoopL
LoopBeginL_1:


/******************************************/
/* Unrolled Loop 1/2 - Begin              */
/******************************************/


/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */





/* iter 0 */

/*  grEndMfmaIndex:6, lwStartMfmaIndex:18, lwEndMfmaIndex:48  */
/*  numMfmaForLR:13, barrierMfmaIndex:50  */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(0) vmcnt(8)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_mfma_f64_16x16x4f64 v[0:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[0:7]
/*  mfmaIndex:1  */
ds_read_b128 v[vgprValuB_X2_I0+0:vgprValuB_X2_I0+0+3], v[vgprLocalReadAddrB] offset:64 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s68, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s69, s[sgprWrapUA+1], 0              // incUpper <- ?
v_mfma_f64_16x16x4f64 v[8:15], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[8:15]
/*  mfmaIndex:2  */
ds_read_b128 v[vgprValuB_X2_I0+4:vgprValuB_X2_I0+4+3], v[vgprLocalReadAddrB] offset:2624 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s68        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], s69      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s68 // limit -= inc)
v_mfma_f64_16x16x4f64 v[16:23], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[16:23]
/*  mfmaIndex:3  */
ds_read_b128 v[vgprValuB_X2_I0+8:vgprValuB_X2_I0+8+3], v[vgprLocalReadAddrB] offset:5184 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s69 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32
v_mfma_f64_16x16x4f64 v[24:31], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[24:31]
/*  mfmaIndex:4  */
s_setprio 0                                        // store optimization
ds_read_b128 v[vgprValuB_X2_I0+12:vgprValuB_X2_I0+12+3], v[vgprLocalReadAddrB] offset:7744 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s68, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s69, s[sgprWrapUB+1], 0              // incUpper <- ?
v_mfma_f64_16x16x4f64 v[32:39], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[32:39]
/*  mfmaIndex:5  */
ds_read_b128 v[vgprValuB_X2_I0+16:vgprValuB_X2_I0+16+3], v[vgprLocalReadAddrB] offset:10304 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s68        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], s69      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s68 // limit -= inc)
v_mfma_f64_16x16x4f64 v[40:47], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[40:47]
/*  mfmaIndex:6  */
ds_read_b128 v[vgprValuB_X2_I0+20:vgprValuB_X2_I0+20+3], v[vgprLocalReadAddrB] offset:12864 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=5 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s69 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32
v_mfma_f64_16x16x4f64 v[48:55], v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[48:55]
/*  mfmaIndex:7  */
ds_read_b128 v[vgprValuB_X2_I0+24:vgprValuB_X2_I0+24+3], v[vgprLocalReadAddrB] offset:15424 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=6 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_f64_16x16x4f64 v[56:63], v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[56:63]
/*  mfmaIndex:8  */
ds_read_b128 v[vgprValuB_X2_I0+28:vgprValuB_X2_I0+28+3], v[vgprLocalReadAddrB] offset:17984 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=7 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_f64_16x16x4f64 v[64:71], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[64:71]
/*  mfmaIndex:9  */
/* localReadsVacancy: letencyLeft 1 */
v_mfma_f64_16x16x4f64 v[72:79], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[72:79]
/*  mfmaIndex:10  */
v_mfma_f64_16x16x4f64 v[80:87], v[vgprValuB_X0_I0+20+0+0:vgprValuB_X0_I0+20+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[80:87]
/*  mfmaIndex:11  */
v_mfma_f64_16x16x4f64 v[88:95], v[vgprValuB_X0_I0+20+0+0:vgprValuB_X0_I0+20+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[88:95]
/*  mfmaIndex:12  */
s_setprio 3                                        // store optimization
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[96:103], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[96:103]
/*  mfmaIndex:13  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[104:111], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[104:111]
/*  mfmaIndex:14  */
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)                               //
s_barrier                                          //
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[112:119], v[vgprValuB_X0_I0+28+0+0:vgprValuB_X0_I0+28+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[112:119]
/*  mfmaIndex:15  */
s_waitcnt vmcnt(7)                                 // lgkmcnt=-1 vmcnt=7wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[120:127], v[vgprValuB_X0_I0+28+0+0:vgprValuB_X0_I0+28+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[120:127]
buffer_load_dwordx4 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_0_0
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=1 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=8 */


/* iter 1 */

/*  grEndMfmaIndex:6, lwStartMfmaIndex:18, lwEndMfmaIndex:48  */
/*  numMfmaForLR:13, barrierMfmaIndex:50  */
/*  mfmaIndex:16  */
/* localReadsVacancy: letencyLeft 5 */
s_waitcnt lgkmcnt(1)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=1, new=1 newLW=0 newLR=0
v_mfma_f64_16x16x4f64 v[0:7], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], v[0:7]
/*  mfmaIndex:17  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[8:15], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], v[8:15]
/*  mfmaIndex:18  */
v_mfma_f64_16x16x4f64 v[16:23], v[vgprValuB_X0_I0+4+2+0:vgprValuB_X0_I0+4+2+0+1], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], v[16:23]
/*  mfmaIndex:19  */
v_mfma_f64_16x16x4f64 v[24:31], v[vgprValuB_X0_I0+4+2+0:vgprValuB_X0_I0+4+2+0+1], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], v[24:31]
/*  mfmaIndex:20  */
v_mfma_f64_16x16x4f64 v[32:39], v[vgprValuB_X0_I0+8+2+0:vgprValuB_X0_I0+8+2+0+1], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], v[32:39]
/*  mfmaIndex:21  */
s_waitcnt vmcnt(7)                                 // lgkmcnt=-1 vmcnt=7wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+4:vgprG2LB+4+3] offset:1280 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 1280
v_mfma_f64_16x16x4f64 v[40:47], v[vgprValuB_X0_I0+8+2+0:vgprValuB_X0_I0+8+2+0+1], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], v[40:47]
/*  mfmaIndex:22  */
buffer_load_dwordx4 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_1_0
v_mfma_f64_16x16x4f64 v[48:55], v[vgprValuB_X0_I0+12+2+0:vgprValuB_X0_I0+12+2+0+1], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], v[48:55]
/*  mfmaIndex:23  */
v_mfma_f64_16x16x4f64 v[56:63], v[vgprValuB_X0_I0+12+2+0:vgprValuB_X0_I0+12+2+0+1], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], v[56:63]
/*  mfmaIndex:24  */
v_mfma_f64_16x16x4f64 v[64:71], v[vgprValuB_X0_I0+16+2+0:vgprValuB_X0_I0+16+2+0+1], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], v[64:71]
/*  mfmaIndex:25  */
v_mfma_f64_16x16x4f64 v[72:79], v[vgprValuB_X0_I0+16+2+0:vgprValuB_X0_I0+16+2+0+1], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], v[72:79]
/*  mfmaIndex:26  */
v_mfma_f64_16x16x4f64 v[80:87], v[vgprValuB_X0_I0+20+2+0:vgprValuB_X0_I0+20+2+0+1], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], v[80:87]
/*  mfmaIndex:27  */
s_waitcnt vmcnt(7)                                 // lgkmcnt=-1 vmcnt=7wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+8:vgprG2LB+8+3] offset:2560 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 2560
v_mfma_f64_16x16x4f64 v[88:95], v[vgprValuB_X0_I0+20+2+0:vgprValuB_X0_I0+20+2+0+1], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], v[88:95]
/*  mfmaIndex:28  */
buffer_load_dwordx4 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_2_0
v_mfma_f64_16x16x4f64 v[96:103], v[vgprValuB_X0_I0+24+2+0:vgprValuB_X0_I0+24+2+0+1], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], v[96:103]
/*  mfmaIndex:29  */
v_mfma_f64_16x16x4f64 v[104:111], v[vgprValuB_X0_I0+24+2+0:vgprValuB_X0_I0+24+2+0+1], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], v[104:111]
/*  mfmaIndex:30  */
v_mfma_f64_16x16x4f64 v[112:119], v[vgprValuB_X0_I0+28+2+0:vgprValuB_X0_I0+28+2+0+1], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], v[112:119]
/*  mfmaIndex:31  */
v_mfma_f64_16x16x4f64 v[120:127], v[vgprValuB_X0_I0+28+2+0:vgprValuB_X0_I0+28+2+0+1], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], v[120:127]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=1 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=8 */


/* iter 2 (reset local read pointers iteration)  (swap local read pointers iteration)  */

/*  grEndMfmaIndex:6, lwStartMfmaIndex:18, lwEndMfmaIndex:48  */
/*  numMfmaForLR:13, barrierMfmaIndex:50  */
/*  mfmaIndex:32  */
s_waitcnt lgkmcnt(3)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=4 newLW=4 newLR=0
v_mfma_f64_16x16x4f64 v[0:7], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], v[0:7]
/*  mfmaIndex:33  */
s_waitcnt vmcnt(7)                                 // lgkmcnt=-1 vmcnt=7wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+12:vgprG2LB+12+3] offset:3840 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 3840
v_mfma_f64_16x16x4f64 v[8:15], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+2+0+0:vgprValuA_X2_I0+2+0+0+1], v[8:15]
/*  mfmaIndex:34  */
buffer_load_dwordx4 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_3_0
v_mfma_f64_16x16x4f64 v[16:23], v[vgprValuB_X2_I0+4+0+0:vgprValuB_X2_I0+4+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], v[16:23]
/*  mfmaIndex:35  */
v_mfma_f64_16x16x4f64 v[24:31], v[vgprValuB_X2_I0+4+0+0:vgprValuB_X2_I0+4+0+0+1], v[vgprValuA_X2_I0+2+0+0:vgprValuA_X2_I0+2+0+0+1], v[24:31]
/*  mfmaIndex:36  */
s_setprio 0                                        // store optimization
v_mfma_f64_16x16x4f64 v[32:39], v[vgprValuB_X2_I0+8+0+0:vgprValuB_X2_I0+8+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], v[32:39]
/*  mfmaIndex:37  */
v_mfma_f64_16x16x4f64 v[40:47], v[vgprValuB_X2_I0+8+0+0:vgprValuB_X2_I0+8+0+0+1], v[vgprValuA_X2_I0+2+0+0:vgprValuA_X2_I0+2+0+0+1], v[40:47]
/*  mfmaIndex:38  */
v_mfma_f64_16x16x4f64 v[48:55], v[vgprValuB_X2_I0+12+0+0:vgprValuB_X2_I0+12+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], v[48:55]
/*  mfmaIndex:39  */
v_mfma_f64_16x16x4f64 v[56:63], v[vgprValuB_X2_I0+12+0+0:vgprValuB_X2_I0+12+0+0+1], v[vgprValuA_X2_I0+2+0+0:vgprValuA_X2_I0+2+0+0+1], v[56:63]
/*  mfmaIndex:40  */
v_mfma_f64_16x16x4f64 v[64:71], v[vgprValuB_X2_I0+16+0+0:vgprValuB_X2_I0+16+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], v[64:71]
/*  mfmaIndex:41  */
v_mfma_f64_16x16x4f64 v[72:79], v[vgprValuB_X2_I0+16+0+0:vgprValuB_X2_I0+16+0+0+1], v[vgprValuA_X2_I0+2+0+0:vgprValuA_X2_I0+2+0+0+1], v[72:79]
/*  mfmaIndex:42  */
v_mfma_f64_16x16x4f64 v[80:87], v[vgprValuB_X2_I0+20+0+0:vgprValuB_X2_I0+20+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], v[80:87]
/*  mfmaIndex:43  */
v_mfma_f64_16x16x4f64 v[88:95], v[vgprValuB_X2_I0+20+0+0:vgprValuB_X2_I0+20+0+0+1], v[vgprValuA_X2_I0+2+0+0:vgprValuA_X2_I0+2+0+0+1], v[88:95]
/*  mfmaIndex:44  */
s_setprio 3                                        // store optimization
s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-13wait for local write
s_barrier //
v_mfma_f64_16x16x4f64 v[96:103], v[vgprValuB_X2_I0+24+0+0:vgprValuB_X2_I0+24+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], v[96:103]
/*  mfmaIndex:45  */
v_mfma_f64_16x16x4f64 v[104:111], v[vgprValuB_X2_I0+24+0+0:vgprValuB_X2_I0+24+0+0+1], v[vgprValuA_X2_I0+2+0+0:vgprValuA_X2_I0+2+0+0+1], v[104:111]
/*  mfmaIndex:46  */
v_mfma_f64_16x16x4f64 v[112:119], v[vgprValuB_X2_I0+28+0+0:vgprValuB_X2_I0+28+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], v[112:119]
/*  mfmaIndex:47  */

/* local read swap offsets a */

/* local read swap offsets b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_mfma_f64_16x16x4f64 v[120:127], v[vgprValuB_X2_I0+28+0+0:vgprValuB_X2_I0+28+0+0+1], v[vgprValuA_X2_I0+2+0+0:vgprValuA_X2_I0+2+0+0+1], v[120:127]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=3 skipReadsIterA=1 readsPerIterA=1 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=8 */


/* iter 3 (swap and reset local write pointers iteration)  */

/*  grEndMfmaIndex:6, lwStartMfmaIndex:18, lwEndMfmaIndex:48  */
/*  numMfmaForLR:13, barrierMfmaIndex:50  */
/*  mfmaIndex:48  */

/* local write swap offsets a */

/* local write swap offsets b */
s_waitcnt lgkmcnt(4)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=4 newLW=4 newLR=0
v_mfma_f64_16x16x4f64 v[0:7], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+1], v[0:7]
/*  mfmaIndex:49  */
v_mfma_f64_16x16x4f64 v[8:15], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X3_I0+2+0+0:vgprValuA_X3_I0+2+0+0+1], v[8:15]
/*  mfmaIndex:50  */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f64_16x16x4f64 v[16:23], v[vgprValuB_X2_I0+4+2+0:vgprValuB_X2_I0+4+2+0+1], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+1], v[16:23]
/*  mfmaIndex:51  */
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:2560 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f64_16x16x4f64 v[24:31], v[vgprValuB_X2_I0+4+2+0:vgprValuB_X2_I0+4+2+0+1], v[vgprValuA_X3_I0+2+0+0:vgprValuA_X3_I0+2+0+0+1], v[24:31]
/*  mfmaIndex:52  */
buffer_load_dwordx4 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_0_0
v_mfma_f64_16x16x4f64 v[32:39], v[vgprValuB_X2_I0+8+2+0:vgprValuB_X2_I0+8+2+0+1], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+1], v[32:39]
/*  mfmaIndex:53  */
ds_read_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:5120 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f64_16x16x4f64 v[40:47], v[vgprValuB_X2_I0+8+2+0:vgprValuB_X2_I0+8+2+0+1], v[vgprValuA_X3_I0+2+0+0:vgprValuA_X3_I0+2+0+0+1], v[40:47]
/*  mfmaIndex:54  */
ds_read_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB] offset:7680 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f64_16x16x4f64 v[48:55], v[vgprValuB_X2_I0+12+2+0:vgprValuB_X2_I0+12+2+0+1], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+1], v[48:55]
/*  mfmaIndex:55  */
ds_read_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB] offset:10240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f64_16x16x4f64 v[56:63], v[vgprValuB_X2_I0+12+2+0:vgprValuB_X2_I0+12+2+0+1], v[vgprValuA_X3_I0+2+0+0:vgprValuA_X3_I0+2+0+0+1], v[56:63]
/*  mfmaIndex:56  */
buffer_load_dwordx4 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_1_0
v_mfma_f64_16x16x4f64 v[64:71], v[vgprValuB_X2_I0+16+2+0:vgprValuB_X2_I0+16+2+0+1], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+1], v[64:71]
/*  mfmaIndex:57  */
ds_read_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB] offset:12800 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=5 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f64_16x16x4f64 v[72:79], v[vgprValuB_X2_I0+16+2+0:vgprValuB_X2_I0+16+2+0+1], v[vgprValuA_X3_I0+2+0+0:vgprValuA_X3_I0+2+0+0+1], v[72:79]
/*  mfmaIndex:58  */
ds_read_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB] offset:15360 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=6 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f64_16x16x4f64 v[80:87], v[vgprValuB_X2_I0+20+2+0:vgprValuB_X2_I0+20+2+0+1], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+1], v[80:87]
/*  mfmaIndex:59  */
ds_read_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB] offset:17920 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=7 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f64_16x16x4f64 v[88:95], v[vgprValuB_X2_I0+20+2+0:vgprValuB_X2_I0+20+2+0+1], v[vgprValuA_X3_I0+2+0+0:vgprValuA_X3_I0+2+0+0+1], v[88:95]
/*  mfmaIndex:60  */
buffer_load_dwordx4 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_2_0
v_mfma_f64_16x16x4f64 v[96:103], v[vgprValuB_X2_I0+24+2+0:vgprValuB_X2_I0+24+2+0+1], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+1], v[96:103]
/*  mfmaIndex:61  */
v_mfma_f64_16x16x4f64 v[104:111], v[vgprValuB_X2_I0+24+2+0:vgprValuB_X2_I0+24+2+0+1], v[vgprValuA_X3_I0+2+0+0:vgprValuA_X3_I0+2+0+0+1], v[104:111]
/*  mfmaIndex:62  */
v_mfma_f64_16x16x4f64 v[112:119], v[vgprValuB_X2_I0+28+2+0:vgprValuB_X2_I0+28+2+0+1], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+1], v[112:119]
/*  mfmaIndex:63  */
v_mfma_f64_16x16x4f64 v[120:127], v[vgprValuB_X2_I0+28+2+0:vgprValuB_X2_I0+28+2+0+1], v[vgprValuA_X3_I0+2+0+0:vgprValuA_X3_I0+2+0+0+1], v[120:127]
buffer_load_dwordx4 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_3_0
/* numPrefetchIter=1 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=1 readsPerIterA=1 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=8 */




/******************************************/
/* Unrolled Loop - End                    */
/******************************************/

s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL

/******************************************/
/* Unrolled Loop 2/2 - Begin              */
/******************************************/


/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */





/* iter 0 */

/*  grEndMfmaIndex:6, lwStartMfmaIndex:18, lwEndMfmaIndex:48  */
/*  numMfmaForLR:13, barrierMfmaIndex:50  */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(0) vmcnt(8)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_mfma_f64_16x16x4f64 v[0:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I1+0+0+0:vgprValuA_X0_I1+0+0+0+1], v[0:7]
/*  mfmaIndex:1  */
ds_read_b128 v[vgprValuB_X2_I0+0:vgprValuB_X2_I0+0+3], v[vgprLocalReadAddrB] offset:64 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s68, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s69, s[sgprWrapUA+1], 0              // incUpper <- ?
v_mfma_f64_16x16x4f64 v[8:15], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I1+2+0+0:vgprValuA_X0_I1+2+0+0+1], v[8:15]
/*  mfmaIndex:2  */
ds_read_b128 v[vgprValuB_X2_I0+4:vgprValuB_X2_I0+4+3], v[vgprLocalReadAddrB] offset:2624 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s68        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], s69      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s68 // limit -= inc)
v_mfma_f64_16x16x4f64 v[16:23], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I1+0+0+0:vgprValuA_X0_I1+0+0+0+1], v[16:23]
/*  mfmaIndex:3  */
ds_read_b128 v[vgprValuB_X2_I0+8:vgprValuB_X2_I0+8+3], v[vgprLocalReadAddrB] offset:5184 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s69 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32
v_mfma_f64_16x16x4f64 v[24:31], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I1+2+0+0:vgprValuA_X0_I1+2+0+0+1], v[24:31]
/*  mfmaIndex:4  */
s_setprio 0                                        // store optimization
ds_read_b128 v[vgprValuB_X2_I0+12:vgprValuB_X2_I0+12+3], v[vgprLocalReadAddrB] offset:7744 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s68, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s69, s[sgprWrapUB+1], 0              // incUpper <- ?
v_mfma_f64_16x16x4f64 v[32:39], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+1], v[vgprValuA_X0_I1+0+0+0:vgprValuA_X0_I1+0+0+0+1], v[32:39]
/*  mfmaIndex:5  */
ds_read_b128 v[vgprValuB_X2_I0+16:vgprValuB_X2_I0+16+3], v[vgprLocalReadAddrB] offset:10304 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s68        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], s69      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s68 // limit -= inc)
v_mfma_f64_16x16x4f64 v[40:47], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+1], v[vgprValuA_X0_I1+2+0+0:vgprValuA_X0_I1+2+0+0+1], v[40:47]
/*  mfmaIndex:6  */
ds_read_b128 v[vgprValuB_X2_I0+20:vgprValuB_X2_I0+20+3], v[vgprLocalReadAddrB] offset:12864 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=5 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s69 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32
v_mfma_f64_16x16x4f64 v[48:55], v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+1], v[vgprValuA_X0_I1+0+0+0:vgprValuA_X0_I1+0+0+0+1], v[48:55]
/*  mfmaIndex:7  */
ds_read_b128 v[vgprValuB_X2_I0+24:vgprValuB_X2_I0+24+3], v[vgprLocalReadAddrB] offset:15424 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=6 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_f64_16x16x4f64 v[56:63], v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+1], v[vgprValuA_X0_I1+2+0+0:vgprValuA_X0_I1+2+0+0+1], v[56:63]
/*  mfmaIndex:8  */
ds_read_b128 v[vgprValuB_X2_I0+28:vgprValuB_X2_I0+28+3], v[vgprLocalReadAddrB] offset:17984 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=7 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_f64_16x16x4f64 v[64:71], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+1], v[vgprValuA_X0_I1+0+0+0:vgprValuA_X0_I1+0+0+0+1], v[64:71]
/*  mfmaIndex:9  */
/* localReadsVacancy: letencyLeft 1 */
v_mfma_f64_16x16x4f64 v[72:79], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+1], v[vgprValuA_X0_I1+2+0+0:vgprValuA_X0_I1+2+0+0+1], v[72:79]
/*  mfmaIndex:10  */
v_mfma_f64_16x16x4f64 v[80:87], v[vgprValuB_X0_I0+20+0+0:vgprValuB_X0_I0+20+0+0+1], v[vgprValuA_X0_I1+0+0+0:vgprValuA_X0_I1+0+0+0+1], v[80:87]
/*  mfmaIndex:11  */
v_mfma_f64_16x16x4f64 v[88:95], v[vgprValuB_X0_I0+20+0+0:vgprValuB_X0_I0+20+0+0+1], v[vgprValuA_X0_I1+2+0+0:vgprValuA_X0_I1+2+0+0+1], v[88:95]
/*  mfmaIndex:12  */
s_setprio 3                                        // store optimization
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[96:103], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+1], v[vgprValuA_X0_I1+0+0+0:vgprValuA_X0_I1+0+0+0+1], v[96:103]
/*  mfmaIndex:13  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[104:111], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+1], v[vgprValuA_X0_I1+2+0+0:vgprValuA_X0_I1+2+0+0+1], v[104:111]
/*  mfmaIndex:14  */
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)                               //
s_barrier                                          //
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[112:119], v[vgprValuB_X0_I0+28+0+0:vgprValuB_X0_I0+28+0+0+1], v[vgprValuA_X0_I1+0+0+0:vgprValuA_X0_I1+0+0+0+1], v[112:119]
/*  mfmaIndex:15  */
s_waitcnt vmcnt(7)                                 // lgkmcnt=-1 vmcnt=7wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[120:127], v[vgprValuB_X0_I0+28+0+0:vgprValuB_X0_I0+28+0+0+1], v[vgprValuA_X0_I1+2+0+0:vgprValuA_X0_I1+2+0+0+1], v[120:127]
buffer_load_dwordx4 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_0_0
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=1 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=8 */


/* iter 1 */

/*  grEndMfmaIndex:6, lwStartMfmaIndex:18, lwEndMfmaIndex:48  */
/*  numMfmaForLR:13, barrierMfmaIndex:50  */
/*  mfmaIndex:16  */
/* localReadsVacancy: letencyLeft 5 */
s_waitcnt lgkmcnt(1)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=1, new=1 newLW=0 newLR=0
v_mfma_f64_16x16x4f64 v[0:7], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X1_I1+0+0+0:vgprValuA_X1_I1+0+0+0+1], v[0:7]
/*  mfmaIndex:17  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[8:15], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X1_I1+2+0+0:vgprValuA_X1_I1+2+0+0+1], v[8:15]
/*  mfmaIndex:18  */
v_mfma_f64_16x16x4f64 v[16:23], v[vgprValuB_X0_I0+4+2+0:vgprValuB_X0_I0+4+2+0+1], v[vgprValuA_X1_I1+0+0+0:vgprValuA_X1_I1+0+0+0+1], v[16:23]
/*  mfmaIndex:19  */
v_mfma_f64_16x16x4f64 v[24:31], v[vgprValuB_X0_I0+4+2+0:vgprValuB_X0_I0+4+2+0+1], v[vgprValuA_X1_I1+2+0+0:vgprValuA_X1_I1+2+0+0+1], v[24:31]
/*  mfmaIndex:20  */
v_mfma_f64_16x16x4f64 v[32:39], v[vgprValuB_X0_I0+8+2+0:vgprValuB_X0_I0+8+2+0+1], v[vgprValuA_X1_I1+0+0+0:vgprValuA_X1_I1+0+0+0+1], v[32:39]
/*  mfmaIndex:21  */
s_waitcnt vmcnt(7)                                 // lgkmcnt=-1 vmcnt=7wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+4:vgprG2LB+4+3] offset:1280 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 1280
v_mfma_f64_16x16x4f64 v[40:47], v[vgprValuB_X0_I0+8+2+0:vgprValuB_X0_I0+8+2+0+1], v[vgprValuA_X1_I1+2+0+0:vgprValuA_X1_I1+2+0+0+1], v[40:47]
/*  mfmaIndex:22  */
buffer_load_dwordx4 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_1_0
v_mfma_f64_16x16x4f64 v[48:55], v[vgprValuB_X0_I0+12+2+0:vgprValuB_X0_I0+12+2+0+1], v[vgprValuA_X1_I1+0+0+0:vgprValuA_X1_I1+0+0+0+1], v[48:55]
/*  mfmaIndex:23  */
v_mfma_f64_16x16x4f64 v[56:63], v[vgprValuB_X0_I0+12+2+0:vgprValuB_X0_I0+12+2+0+1], v[vgprValuA_X1_I1+2+0+0:vgprValuA_X1_I1+2+0+0+1], v[56:63]
/*  mfmaIndex:24  */
v_mfma_f64_16x16x4f64 v[64:71], v[vgprValuB_X0_I0+16+2+0:vgprValuB_X0_I0+16+2+0+1], v[vgprValuA_X1_I1+0+0+0:vgprValuA_X1_I1+0+0+0+1], v[64:71]
/*  mfmaIndex:25  */
v_mfma_f64_16x16x4f64 v[72:79], v[vgprValuB_X0_I0+16+2+0:vgprValuB_X0_I0+16+2+0+1], v[vgprValuA_X1_I1+2+0+0:vgprValuA_X1_I1+2+0+0+1], v[72:79]
/*  mfmaIndex:26  */
v_mfma_f64_16x16x4f64 v[80:87], v[vgprValuB_X0_I0+20+2+0:vgprValuB_X0_I0+20+2+0+1], v[vgprValuA_X1_I1+0+0+0:vgprValuA_X1_I1+0+0+0+1], v[80:87]
/*  mfmaIndex:27  */
s_waitcnt vmcnt(7)                                 // lgkmcnt=-1 vmcnt=7wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+8:vgprG2LB+8+3] offset:2560 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 2560
v_mfma_f64_16x16x4f64 v[88:95], v[vgprValuB_X0_I0+20+2+0:vgprValuB_X0_I0+20+2+0+1], v[vgprValuA_X1_I1+2+0+0:vgprValuA_X1_I1+2+0+0+1], v[88:95]
/*  mfmaIndex:28  */
buffer_load_dwordx4 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_2_0
v_mfma_f64_16x16x4f64 v[96:103], v[vgprValuB_X0_I0+24+2+0:vgprValuB_X0_I0+24+2+0+1], v[vgprValuA_X1_I1+0+0+0:vgprValuA_X1_I1+0+0+0+1], v[96:103]
/*  mfmaIndex:29  */
v_mfma_f64_16x16x4f64 v[104:111], v[vgprValuB_X0_I0+24+2+0:vgprValuB_X0_I0+24+2+0+1], v[vgprValuA_X1_I1+2+0+0:vgprValuA_X1_I1+2+0+0+1], v[104:111]
/*  mfmaIndex:30  */
v_mfma_f64_16x16x4f64 v[112:119], v[vgprValuB_X0_I0+28+2+0:vgprValuB_X0_I0+28+2+0+1], v[vgprValuA_X1_I1+0+0+0:vgprValuA_X1_I1+0+0+0+1], v[112:119]
/*  mfmaIndex:31  */
v_mfma_f64_16x16x4f64 v[120:127], v[vgprValuB_X0_I0+28+2+0:vgprValuB_X0_I0+28+2+0+1], v[vgprValuA_X1_I1+2+0+0:vgprValuA_X1_I1+2+0+0+1], v[120:127]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=1 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=8 */


/* iter 2 (reset local read pointers iteration)  (swap local read pointers iteration)  */

/*  grEndMfmaIndex:6, lwStartMfmaIndex:18, lwEndMfmaIndex:48  */
/*  numMfmaForLR:13, barrierMfmaIndex:50  */
/*  mfmaIndex:32  */
s_waitcnt lgkmcnt(3)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=4 newLW=4 newLR=0
v_mfma_f64_16x16x4f64 v[0:7], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I1+0+0+0:vgprValuA_X2_I1+0+0+0+1], v[0:7]
/*  mfmaIndex:33  */
s_waitcnt vmcnt(7)                                 // lgkmcnt=-1 vmcnt=7wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+12:vgprG2LB+12+3] offset:3840 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 3840
v_mfma_f64_16x16x4f64 v[8:15], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I1+2+0+0:vgprValuA_X2_I1+2+0+0+1], v[8:15]
/*  mfmaIndex:34  */
buffer_load_dwordx4 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_3_0
v_mfma_f64_16x16x4f64 v[16:23], v[vgprValuB_X2_I0+4+0+0:vgprValuB_X2_I0+4+0+0+1], v[vgprValuA_X2_I1+0+0+0:vgprValuA_X2_I1+0+0+0+1], v[16:23]
/*  mfmaIndex:35  */
v_mfma_f64_16x16x4f64 v[24:31], v[vgprValuB_X2_I0+4+0+0:vgprValuB_X2_I0+4+0+0+1], v[vgprValuA_X2_I1+2+0+0:vgprValuA_X2_I1+2+0+0+1], v[24:31]
/*  mfmaIndex:36  */
s_setprio 0                                        // store optimization
v_mfma_f64_16x16x4f64 v[32:39], v[vgprValuB_X2_I0+8+0+0:vgprValuB_X2_I0+8+0+0+1], v[vgprValuA_X2_I1+0+0+0:vgprValuA_X2_I1+0+0+0+1], v[32:39]
/*  mfmaIndex:37  */
v_mfma_f64_16x16x4f64 v[40:47], v[vgprValuB_X2_I0+8+0+0:vgprValuB_X2_I0+8+0+0+1], v[vgprValuA_X2_I1+2+0+0:vgprValuA_X2_I1+2+0+0+1], v[40:47]
/*  mfmaIndex:38  */
v_mfma_f64_16x16x4f64 v[48:55], v[vgprValuB_X2_I0+12+0+0:vgprValuB_X2_I0+12+0+0+1], v[vgprValuA_X2_I1+0+0+0:vgprValuA_X2_I1+0+0+0+1], v[48:55]
/*  mfmaIndex:39  */
v_mfma_f64_16x16x4f64 v[56:63], v[vgprValuB_X2_I0+12+0+0:vgprValuB_X2_I0+12+0+0+1], v[vgprValuA_X2_I1+2+0+0:vgprValuA_X2_I1+2+0+0+1], v[56:63]
/*  mfmaIndex:40  */
v_mfma_f64_16x16x4f64 v[64:71], v[vgprValuB_X2_I0+16+0+0:vgprValuB_X2_I0+16+0+0+1], v[vgprValuA_X2_I1+0+0+0:vgprValuA_X2_I1+0+0+0+1], v[64:71]
/*  mfmaIndex:41  */
v_mfma_f64_16x16x4f64 v[72:79], v[vgprValuB_X2_I0+16+0+0:vgprValuB_X2_I0+16+0+0+1], v[vgprValuA_X2_I1+2+0+0:vgprValuA_X2_I1+2+0+0+1], v[72:79]
/*  mfmaIndex:42  */
v_mfma_f64_16x16x4f64 v[80:87], v[vgprValuB_X2_I0+20+0+0:vgprValuB_X2_I0+20+0+0+1], v[vgprValuA_X2_I1+0+0+0:vgprValuA_X2_I1+0+0+0+1], v[80:87]
/*  mfmaIndex:43  */
v_mfma_f64_16x16x4f64 v[88:95], v[vgprValuB_X2_I0+20+0+0:vgprValuB_X2_I0+20+0+0+1], v[vgprValuA_X2_I1+2+0+0:vgprValuA_X2_I1+2+0+0+1], v[88:95]
/*  mfmaIndex:44  */
s_setprio 3                                        // store optimization
s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-13wait for local write
s_barrier //
v_mfma_f64_16x16x4f64 v[96:103], v[vgprValuB_X2_I0+24+0+0:vgprValuB_X2_I0+24+0+0+1], v[vgprValuA_X2_I1+0+0+0:vgprValuA_X2_I1+0+0+0+1], v[96:103]
/*  mfmaIndex:45  */
v_mfma_f64_16x16x4f64 v[104:111], v[vgprValuB_X2_I0+24+0+0:vgprValuB_X2_I0+24+0+0+1], v[vgprValuA_X2_I1+2+0+0:vgprValuA_X2_I1+2+0+0+1], v[104:111]
/*  mfmaIndex:46  */
v_mfma_f64_16x16x4f64 v[112:119], v[vgprValuB_X2_I0+28+0+0:vgprValuB_X2_I0+28+0+0+1], v[vgprValuA_X2_I1+0+0+0:vgprValuA_X2_I1+0+0+0+1], v[112:119]
/*  mfmaIndex:47  */

/* local read swap offsets a */

/* local read swap offsets b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_mfma_f64_16x16x4f64 v[120:127], v[vgprValuB_X2_I0+28+0+0:vgprValuB_X2_I0+28+0+0+1], v[vgprValuA_X2_I1+2+0+0:vgprValuA_X2_I1+2+0+0+1], v[120:127]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=3 skipReadsIterA=1 readsPerIterA=1 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=8 */


/* iter 3 (swap and reset local write pointers iteration)  */

/*  grEndMfmaIndex:6, lwStartMfmaIndex:18, lwEndMfmaIndex:48  */
/*  numMfmaForLR:13, barrierMfmaIndex:50  */
/*  mfmaIndex:48  */

/* local write swap offsets a */

/* local write swap offsets b */
s_waitcnt lgkmcnt(4)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=4 newLW=4 newLR=0
v_mfma_f64_16x16x4f64 v[0:7], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X3_I1+0+0+0:vgprValuA_X3_I1+0+0+0+1], v[0:7]
/*  mfmaIndex:49  */
v_mfma_f64_16x16x4f64 v[8:15], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X3_I1+2+0+0:vgprValuA_X3_I1+2+0+0+1], v[8:15]
/*  mfmaIndex:50  */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f64_16x16x4f64 v[16:23], v[vgprValuB_X2_I0+4+2+0:vgprValuB_X2_I0+4+2+0+1], v[vgprValuA_X3_I1+0+0+0:vgprValuA_X3_I1+0+0+0+1], v[16:23]
/*  mfmaIndex:51  */
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:2560 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f64_16x16x4f64 v[24:31], v[vgprValuB_X2_I0+4+2+0:vgprValuB_X2_I0+4+2+0+1], v[vgprValuA_X3_I1+2+0+0:vgprValuA_X3_I1+2+0+0+1], v[24:31]
/*  mfmaIndex:52  */
buffer_load_dwordx4 v[vgprValuA_X0_I1+0:vgprValuA_X0_I1+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_0_0
v_mfma_f64_16x16x4f64 v[32:39], v[vgprValuB_X2_I0+8+2+0:vgprValuB_X2_I0+8+2+0+1], v[vgprValuA_X3_I1+0+0+0:vgprValuA_X3_I1+0+0+0+1], v[32:39]
/*  mfmaIndex:53  */
ds_read_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:5120 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f64_16x16x4f64 v[40:47], v[vgprValuB_X2_I0+8+2+0:vgprValuB_X2_I0+8+2+0+1], v[vgprValuA_X3_I1+2+0+0:vgprValuA_X3_I1+2+0+0+1], v[40:47]
/*  mfmaIndex:54  */
ds_read_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB] offset:7680 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f64_16x16x4f64 v[48:55], v[vgprValuB_X2_I0+12+2+0:vgprValuB_X2_I0+12+2+0+1], v[vgprValuA_X3_I1+0+0+0:vgprValuA_X3_I1+0+0+0+1], v[48:55]
/*  mfmaIndex:55  */
ds_read_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB] offset:10240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f64_16x16x4f64 v[56:63], v[vgprValuB_X2_I0+12+2+0:vgprValuB_X2_I0+12+2+0+1], v[vgprValuA_X3_I1+2+0+0:vgprValuA_X3_I1+2+0+0+1], v[56:63]
/*  mfmaIndex:56  */
buffer_load_dwordx4 v[vgprValuA_X0_I1+4:vgprValuA_X0_I1+4+3], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_1_0
v_mfma_f64_16x16x4f64 v[64:71], v[vgprValuB_X2_I0+16+2+0:vgprValuB_X2_I0+16+2+0+1], v[vgprValuA_X3_I1+0+0+0:vgprValuA_X3_I1+0+0+0+1], v[64:71]
/*  mfmaIndex:57  */
ds_read_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB] offset:12800 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=5 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f64_16x16x4f64 v[72:79], v[vgprValuB_X2_I0+16+2+0:vgprValuB_X2_I0+16+2+0+1], v[vgprValuA_X3_I1+2+0+0:vgprValuA_X3_I1+2+0+0+1], v[72:79]
/*  mfmaIndex:58  */
ds_read_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB] offset:15360 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=6 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f64_16x16x4f64 v[80:87], v[vgprValuB_X2_I0+20+2+0:vgprValuB_X2_I0+20+2+0+1], v[vgprValuA_X3_I1+0+0+0:vgprValuA_X3_I1+0+0+0+1], v[80:87]
/*  mfmaIndex:59  */
ds_read_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB] offset:17920 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=7 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f64_16x16x4f64 v[88:95], v[vgprValuB_X2_I0+20+2+0:vgprValuB_X2_I0+20+2+0+1], v[vgprValuA_X3_I1+2+0+0:vgprValuA_X3_I1+2+0+0+1], v[88:95]
/*  mfmaIndex:60  */
buffer_load_dwordx4 v[vgprValuA_X0_I1+8:vgprValuA_X0_I1+8+3], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_2_0
v_mfma_f64_16x16x4f64 v[96:103], v[vgprValuB_X2_I0+24+2+0:vgprValuB_X2_I0+24+2+0+1], v[vgprValuA_X3_I1+0+0+0:vgprValuA_X3_I1+0+0+0+1], v[96:103]
/*  mfmaIndex:61  */
v_mfma_f64_16x16x4f64 v[104:111], v[vgprValuB_X2_I0+24+2+0:vgprValuB_X2_I0+24+2+0+1], v[vgprValuA_X3_I1+2+0+0:vgprValuA_X3_I1+2+0+0+1], v[104:111]
/*  mfmaIndex:62  */
v_mfma_f64_16x16x4f64 v[112:119], v[vgprValuB_X2_I0+28+2+0:vgprValuB_X2_I0+28+2+0+1], v[vgprValuA_X3_I1+0+0+0:vgprValuA_X3_I1+0+0+0+1], v[112:119]
/*  mfmaIndex:63  */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_u32 s[sgprLoopCounterL], 0x2              // counterL==2
v_mfma_f64_16x16x4f64 v[120:127], v[vgprValuB_X2_I0+28+2+0:vgprValuB_X2_I0+28+2+0+1], v[vgprValuA_X3_I1+2+0+0:vgprValuA_X3_I1+2+0+0+1], v[120:127]
buffer_load_dwordx4 v[vgprValuA_X0_I1+12:vgprValuA_X0_I1+12+3], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_3_0
/* numPrefetchIter=1 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=1 readsPerIterA=1 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=8 */




/******************************************/
/* Unrolled Loop - End                    */
/******************************************/


/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_cbranch_scc0 LoopBeginL_1                        // restart LoopL
LoopEndL_oddexit_3: // unroll loop odditer exit
LoopEndL_2:


/* Before NLL: Check VGPR.checkin for INT8 LW */




/******************************************/
/*  NoGlobalLoadLoop - Begin              */
/******************************************/

s_setprio 0                                        // store optimization

/* iter 0 */

/*  grEndMfmaIndex:6, lwStartMfmaIndex:18, lwEndMfmaIndex:48  */
/*  numMfmaForLR:13, barrierMfmaIndex:50  */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(0) vmcnt(8)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_mfma_f64_16x16x4f64 v[0:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[0:7]
/*  mfmaIndex:1  */
ds_read_b128 v[vgprValuB_X2_I0+0:vgprValuB_X2_I0+0+3], v[vgprLocalReadAddrB] offset:64 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s68, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s69, s[sgprWrapUA+1], 0              // incUpper <- ?
v_mfma_f64_16x16x4f64 v[8:15], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[8:15]
/*  mfmaIndex:2  */
ds_read_b128 v[vgprValuB_X2_I0+4:vgprValuB_X2_I0+4+3], v[vgprLocalReadAddrB] offset:2624 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s68        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], s69      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s68 // limit -= inc)
v_mfma_f64_16x16x4f64 v[16:23], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[16:23]
/*  mfmaIndex:3  */
ds_read_b128 v[vgprValuB_X2_I0+8:vgprValuB_X2_I0+8+3], v[vgprLocalReadAddrB] offset:5184 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s69 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32
v_mfma_f64_16x16x4f64 v[24:31], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[24:31]
/*  mfmaIndex:4  */
ds_read_b128 v[vgprValuB_X2_I0+12:vgprValuB_X2_I0+12+3], v[vgprLocalReadAddrB] offset:7744 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s68, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s69, s[sgprWrapUB+1], 0              // incUpper <- ?
v_mfma_f64_16x16x4f64 v[32:39], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[32:39]
/*  mfmaIndex:5  */
ds_read_b128 v[vgprValuB_X2_I0+16:vgprValuB_X2_I0+16+3], v[vgprLocalReadAddrB] offset:10304 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s68        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], s69      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s68 // limit -= inc)
v_mfma_f64_16x16x4f64 v[40:47], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[40:47]
/*  mfmaIndex:6  */
ds_read_b128 v[vgprValuB_X2_I0+20:vgprValuB_X2_I0+20+3], v[vgprLocalReadAddrB] offset:12864 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=5 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s69 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32
v_mfma_f64_16x16x4f64 v[48:55], v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[48:55]
/*  mfmaIndex:7  */
ds_read_b128 v[vgprValuB_X2_I0+24:vgprValuB_X2_I0+24+3], v[vgprLocalReadAddrB] offset:15424 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=6 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_f64_16x16x4f64 v[56:63], v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[56:63]
/*  mfmaIndex:8  */
ds_read_b128 v[vgprValuB_X2_I0+28:vgprValuB_X2_I0+28+3], v[vgprLocalReadAddrB] offset:17984 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=7 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_f64_16x16x4f64 v[64:71], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[64:71]
/*  mfmaIndex:9  */
/* localReadsVacancy: letencyLeft 1 */
v_mfma_f64_16x16x4f64 v[72:79], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[72:79]
/*  mfmaIndex:10  */
v_mfma_f64_16x16x4f64 v[80:87], v[vgprValuB_X0_I0+20+0+0:vgprValuB_X0_I0+20+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[80:87]
/*  mfmaIndex:11  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[88:95], v[vgprValuB_X0_I0+20+0+0:vgprValuB_X0_I0+20+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[88:95]
/*  mfmaIndex:12  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[96:103], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[96:103]
/*  mfmaIndex:13  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[104:111], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[104:111]
/*  mfmaIndex:14  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[112:119], v[vgprValuB_X0_I0+28+0+0:vgprValuB_X0_I0+28+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[112:119]
/*  mfmaIndex:15  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[120:127], v[vgprValuB_X0_I0+28+0+0:vgprValuB_X0_I0+28+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[120:127]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=1 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=8 */


/* iter 1 */

/*  grEndMfmaIndex:6, lwStartMfmaIndex:18, lwEndMfmaIndex:48  */
/*  numMfmaForLR:13, barrierMfmaIndex:50  */
/*  mfmaIndex:16  */
/* localReadsVacancy: letencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=1, new=1 newLW=0 newLR=0
v_mfma_f64_16x16x4f64 v[0:7], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], v[0:7]
/*  mfmaIndex:17  */
/* localReadsVacancy: letencyLeft 5 */
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)                               //
s_barrier                                          //
v_mfma_f64_16x16x4f64 v[8:15], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], v[8:15]
/*  mfmaIndex:18  */
v_mfma_f64_16x16x4f64 v[16:23], v[vgprValuB_X0_I0+4+2+0:vgprValuB_X0_I0+4+2+0+1], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], v[16:23]
/*  mfmaIndex:19  */
v_mfma_f64_16x16x4f64 v[24:31], v[vgprValuB_X0_I0+4+2+0:vgprValuB_X0_I0+4+2+0+1], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], v[24:31]
/*  mfmaIndex:20  */
v_mfma_f64_16x16x4f64 v[32:39], v[vgprValuB_X0_I0+8+2+0:vgprValuB_X0_I0+8+2+0+1], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], v[32:39]
/*  mfmaIndex:21  */
v_mfma_f64_16x16x4f64 v[40:47], v[vgprValuB_X0_I0+8+2+0:vgprValuB_X0_I0+8+2+0+1], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], v[40:47]
/*  mfmaIndex:22  */
v_mfma_f64_16x16x4f64 v[48:55], v[vgprValuB_X0_I0+12+2+0:vgprValuB_X0_I0+12+2+0+1], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], v[48:55]
/*  mfmaIndex:23  */
v_mfma_f64_16x16x4f64 v[56:63], v[vgprValuB_X0_I0+12+2+0:vgprValuB_X0_I0+12+2+0+1], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], v[56:63]
/*  mfmaIndex:24  */
v_mfma_f64_16x16x4f64 v[64:71], v[vgprValuB_X0_I0+16+2+0:vgprValuB_X0_I0+16+2+0+1], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], v[64:71]
/*  mfmaIndex:25  */
v_mfma_f64_16x16x4f64 v[72:79], v[vgprValuB_X0_I0+16+2+0:vgprValuB_X0_I0+16+2+0+1], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], v[72:79]
/*  mfmaIndex:26  */
v_mfma_f64_16x16x4f64 v[80:87], v[vgprValuB_X0_I0+20+2+0:vgprValuB_X0_I0+20+2+0+1], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], v[80:87]
/*  mfmaIndex:27  */
v_mfma_f64_16x16x4f64 v[88:95], v[vgprValuB_X0_I0+20+2+0:vgprValuB_X0_I0+20+2+0+1], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], v[88:95]
/*  mfmaIndex:28  */
v_mfma_f64_16x16x4f64 v[96:103], v[vgprValuB_X0_I0+24+2+0:vgprValuB_X0_I0+24+2+0+1], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], v[96:103]
/*  mfmaIndex:29  */
v_mfma_f64_16x16x4f64 v[104:111], v[vgprValuB_X0_I0+24+2+0:vgprValuB_X0_I0+24+2+0+1], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], v[104:111]
/*  mfmaIndex:30  */
v_mfma_f64_16x16x4f64 v[112:119], v[vgprValuB_X0_I0+28+2+0:vgprValuB_X0_I0+28+2+0+1], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], v[112:119]
/*  mfmaIndex:31  */
v_mfma_f64_16x16x4f64 v[120:127], v[vgprValuB_X0_I0+28+2+0:vgprValuB_X0_I0+28+2+0+1], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], v[120:127]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=1 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=8 */


/* iter 2 (reset local read pointers iteration)  (swap local read pointers iteration)  */

/*  grEndMfmaIndex:6, lwStartMfmaIndex:18, lwEndMfmaIndex:48  */
/*  numMfmaForLR:13, barrierMfmaIndex:50  */
/*  mfmaIndex:32  */
s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=4 newLW=4 newLR=0
v_mfma_f64_16x16x4f64 v[0:7], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], v[0:7]
/*  mfmaIndex:33  */
v_mfma_f64_16x16x4f64 v[8:15], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+2+0+0:vgprValuA_X2_I0+2+0+0+1], v[8:15]
/*  mfmaIndex:34  */
v_mfma_f64_16x16x4f64 v[16:23], v[vgprValuB_X2_I0+4+0+0:vgprValuB_X2_I0+4+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], v[16:23]
/*  mfmaIndex:35  */
v_mfma_f64_16x16x4f64 v[24:31], v[vgprValuB_X2_I0+4+0+0:vgprValuB_X2_I0+4+0+0+1], v[vgprValuA_X2_I0+2+0+0:vgprValuA_X2_I0+2+0+0+1], v[24:31]
/*  mfmaIndex:36  */
v_mfma_f64_16x16x4f64 v[32:39], v[vgprValuB_X2_I0+8+0+0:vgprValuB_X2_I0+8+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], v[32:39]
/*  mfmaIndex:37  */
v_mfma_f64_16x16x4f64 v[40:47], v[vgprValuB_X2_I0+8+0+0:vgprValuB_X2_I0+8+0+0+1], v[vgprValuA_X2_I0+2+0+0:vgprValuA_X2_I0+2+0+0+1], v[40:47]
/*  mfmaIndex:38  */
v_mfma_f64_16x16x4f64 v[48:55], v[vgprValuB_X2_I0+12+0+0:vgprValuB_X2_I0+12+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], v[48:55]
/*  mfmaIndex:39  */
v_mfma_f64_16x16x4f64 v[56:63], v[vgprValuB_X2_I0+12+0+0:vgprValuB_X2_I0+12+0+0+1], v[vgprValuA_X2_I0+2+0+0:vgprValuA_X2_I0+2+0+0+1], v[56:63]
/*  mfmaIndex:40  */
v_mfma_f64_16x16x4f64 v[64:71], v[vgprValuB_X2_I0+16+0+0:vgprValuB_X2_I0+16+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], v[64:71]
/*  mfmaIndex:41  */
s_waitcnt vmcnt(7)                                 // lgkmcnt=-1 vmcnt=3wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
v_mfma_f64_16x16x4f64 v[72:79], v[vgprValuB_X2_I0+16+0+0:vgprValuB_X2_I0+16+0+0+1], v[vgprValuA_X2_I0+2+0+0:vgprValuA_X2_I0+2+0+0+1], v[72:79]
/*  mfmaIndex:42  */
v_mfma_f64_16x16x4f64 v[80:87], v[vgprValuB_X2_I0+20+0+0:vgprValuB_X2_I0+20+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], v[80:87]
/*  mfmaIndex:43  */
s_waitcnt vmcnt(6)                                 // lgkmcnt=-1 vmcnt=2wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+4:vgprG2LB+4+3] offset:1280 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 1280
v_mfma_f64_16x16x4f64 v[88:95], v[vgprValuB_X2_I0+20+0+0:vgprValuB_X2_I0+20+0+0+1], v[vgprValuA_X2_I0+2+0+0:vgprValuA_X2_I0+2+0+0+1], v[88:95]
/*  mfmaIndex:44  */
v_mfma_f64_16x16x4f64 v[96:103], v[vgprValuB_X2_I0+24+0+0:vgprValuB_X2_I0+24+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], v[96:103]
/*  mfmaIndex:45  */
s_waitcnt vmcnt(5)                                 // lgkmcnt=-1 vmcnt=1wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+8:vgprG2LB+8+3] offset:2560 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 2560
v_mfma_f64_16x16x4f64 v[104:111], v[vgprValuB_X2_I0+24+0+0:vgprValuB_X2_I0+24+0+0+1], v[vgprValuA_X2_I0+2+0+0:vgprValuA_X2_I0+2+0+0+1], v[104:111]
/*  mfmaIndex:46  */
v_mfma_f64_16x16x4f64 v[112:119], v[vgprValuB_X2_I0+28+0+0:vgprValuB_X2_I0+28+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], v[112:119]
/*  mfmaIndex:47  */
s_waitcnt vmcnt(4)                                 // lgkmcnt=-1 vmcnt=0wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+12:vgprG2LB+12+3] offset:3840 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 3840

/* local read swap offsets a */

/* local read swap offsets b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_mfma_f64_16x16x4f64 v[120:127], v[vgprValuB_X2_I0+28+0+0:vgprValuB_X2_I0+28+0+0+1], v[vgprValuA_X2_I0+2+0+0:vgprValuA_X2_I0+2+0+0+1], v[120:127]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=3 skipReadsIterA=1 readsPerIterA=1 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=8 */


/* iter 3 (swap and reset local write pointers iteration)  */

/*  grEndMfmaIndex:6, lwStartMfmaIndex:18, lwEndMfmaIndex:48  */
/*  numMfmaForLR:13, barrierMfmaIndex:50  */
/*  mfmaIndex:48  */

/* local write swap offsets a */

/* local write swap offsets b */
s_waitcnt lgkmcnt(4)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=4 newLW=4 newLR=0
v_mfma_f64_16x16x4f64 v[0:7], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+1], v[0:7]
/*  mfmaIndex:49  */
v_mfma_f64_16x16x4f64 v[8:15], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X3_I0+2+0+0:vgprValuA_X3_I0+2+0+0+1], v[8:15]
/*  mfmaIndex:50  */
v_mfma_f64_16x16x4f64 v[16:23], v[vgprValuB_X2_I0+4+2+0:vgprValuB_X2_I0+4+2+0+1], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+1], v[16:23]
/*  mfmaIndex:51  */
s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-13wait for local write
// Skip force waitcnt0
s_barrier //
v_mfma_f64_16x16x4f64 v[24:31], v[vgprValuB_X2_I0+4+2+0:vgprValuB_X2_I0+4+2+0+1], v[vgprValuA_X3_I0+2+0+0:vgprValuA_X3_I0+2+0+0+1], v[24:31]
/*  mfmaIndex:52  */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f64_16x16x4f64 v[32:39], v[vgprValuB_X2_I0+8+2+0:vgprValuB_X2_I0+8+2+0+1], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+1], v[32:39]
/*  mfmaIndex:53  */
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:2560 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f64_16x16x4f64 v[40:47], v[vgprValuB_X2_I0+8+2+0:vgprValuB_X2_I0+8+2+0+1], v[vgprValuA_X3_I0+2+0+0:vgprValuA_X3_I0+2+0+0+1], v[40:47]
/*  mfmaIndex:54  */
ds_read_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:5120 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f64_16x16x4f64 v[48:55], v[vgprValuB_X2_I0+12+2+0:vgprValuB_X2_I0+12+2+0+1], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+1], v[48:55]
/*  mfmaIndex:55  */
ds_read_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB] offset:7680 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f64_16x16x4f64 v[56:63], v[vgprValuB_X2_I0+12+2+0:vgprValuB_X2_I0+12+2+0+1], v[vgprValuA_X3_I0+2+0+0:vgprValuA_X3_I0+2+0+0+1], v[56:63]
/*  mfmaIndex:56  */
ds_read_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB] offset:10240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f64_16x16x4f64 v[64:71], v[vgprValuB_X2_I0+16+2+0:vgprValuB_X2_I0+16+2+0+1], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+1], v[64:71]
/*  mfmaIndex:57  */
ds_read_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB] offset:12800 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=5 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f64_16x16x4f64 v[72:79], v[vgprValuB_X2_I0+16+2+0:vgprValuB_X2_I0+16+2+0+1], v[vgprValuA_X3_I0+2+0+0:vgprValuA_X3_I0+2+0+0+1], v[72:79]
/*  mfmaIndex:58  */
ds_read_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB] offset:15360 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=6 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f64_16x16x4f64 v[80:87], v[vgprValuB_X2_I0+20+2+0:vgprValuB_X2_I0+20+2+0+1], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+1], v[80:87]
/*  mfmaIndex:59  */
ds_read_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB] offset:17920 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=7 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f64_16x16x4f64 v[88:95], v[vgprValuB_X2_I0+20+2+0:vgprValuB_X2_I0+20+2+0+1], v[vgprValuA_X3_I0+2+0+0:vgprValuA_X3_I0+2+0+0+1], v[88:95]
/*  mfmaIndex:60  */
v_mfma_f64_16x16x4f64 v[96:103], v[vgprValuB_X2_I0+24+2+0:vgprValuB_X2_I0+24+2+0+1], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+1], v[96:103]
/*  mfmaIndex:61  */
v_mfma_f64_16x16x4f64 v[104:111], v[vgprValuB_X2_I0+24+2+0:vgprValuB_X2_I0+24+2+0+1], v[vgprValuA_X3_I0+2+0+0:vgprValuA_X3_I0+2+0+0+1], v[104:111]
/*  mfmaIndex:62  */
v_mfma_f64_16x16x4f64 v[112:119], v[vgprValuB_X2_I0+28+2+0:vgprValuB_X2_I0+28+2+0+1], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+1], v[112:119]
/*  mfmaIndex:63  */
v_mfma_f64_16x16x4f64 v[120:127], v[vgprValuB_X2_I0+28+2+0:vgprValuB_X2_I0+28+2+0+1], v[vgprValuA_X3_I0+2+0+0:vgprValuA_X3_I0+2+0+0+1], v[120:127]
/* numPrefetchIter=1 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=1 readsPerIterA=1 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=8 */

label_0014:


/******************************************/
/* Opt. NoLoadLoop Without PAP - Begin         */
/******************************************/

s_mov_b32 s68, s[sgprBeta+0]                       // tmp = Beta[0]
s_or_b32 s68, s[sgprBeta+1], s68                   // tmp |= Beta[1]
s_cmpk_eq_u32 s68, 0x0                             // Beta == 0
s_cbranch_scc0 OptNLL_End_16                       // Branch if Beta is not zero

s_mov_b32 s68, 0                                   // Low part of double 1.0
s_mov_b32 s69, 0x3ff00000                          // High part of double 1.0
s_cmp_eq_u64 s[sgprAlpha:sgprAlpha+1], s[68:69]    // Alpha == 1.0 ?
s_cbranch_scc0 OptNLL_End_16                       // branch if alpha != 1

s_and_b32 s68, 127, s[sgprSizeI]                   // s68 = s[sgprSizeI] % 128
s_add_u32 s69, -0x1, s[sgprNumWorkGroups0]         //
s_cmp_ge_u32 s[sgprWorkGroup0], s69                // wg0 >= nwg0-1 ?
s_cselect_b32 s68, s68, 0                          // set rMT0
s_cmpk_gt_u32 s68, 0x0                             // rMT0 > 0
s_cbranch_scc1 OptNLL_End_16                       // jump if edges required
s_and_b32 s68, 127, s[sgprSizeJ]                   // s68 = s[sgprSizeJ] % 128
s_add_u32 s69, -0x1, s[sgprNumWorkGroups1]         //
s_cmp_ge_u32 s[sgprWorkGroup1], s69                // wg1 >= nwg1-1
s_cselect_b32 s68, s68, 0                          // set rMT1
s_cmpk_gt_u32 s68, 0x0                             // rMT1 > 0
s_cbranch_scc1 OptNLL_End_16                       // jump if edges required

s_and_b32 s69, 15, s[sgprSizesSum+0]               // s69 = s[sgprSizesSum+0] % 16
s_cmp_eq_u32 s69, 0x0                              // numIterL == 0
s_cbranch_scc0 OptNLL_End_16                       // skip if tail loop required



/* iter 0 (last unrolled loop) */

/*  grEndMfmaIndex:0, lwStartMfmaIndex:48, lwEndMfmaIndex:48  */
/*  numMfmaForLR:13, barrierMfmaIndex:50  */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(0) vmcnt(3)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_mfma_f64_16x16x4f64 v[0:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I1+0+0+0:vgprValuA_X0_I1+0+0+0+1], v[0:7]
/*  mfmaIndex:1  */
ds_read_b128 v[vgprValuB_X2_I0+0:vgprValuB_X2_I0+0+3], v[vgprLocalReadAddrB] offset:64 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_f64_16x16x4f64 v[8:15], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I1+2+0+0:vgprValuA_X0_I1+2+0+0+1], v[8:15]
/*  mfmaIndex:2  */
ds_read_b128 v[vgprValuB_X2_I0+4:vgprValuB_X2_I0+4+3], v[vgprLocalReadAddrB] offset:2624 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_f64_16x16x4f64 v[16:23], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I1+0+0+0:vgprValuA_X0_I1+0+0+0+1], v[16:23]
/*  mfmaIndex:3  */
ds_read_b128 v[vgprValuB_X2_I0+8:vgprValuB_X2_I0+8+3], v[vgprLocalReadAddrB] offset:5184 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_f64_16x16x4f64 v[24:31], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I1+2+0+0:vgprValuA_X0_I1+2+0+0+1], v[24:31]
/*  mfmaIndex:4  */
ds_read_b128 v[vgprValuB_X2_I0+12:vgprValuB_X2_I0+12+3], v[vgprLocalReadAddrB] offset:7744 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_f64_16x16x4f64 v[32:39], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+1], v[vgprValuA_X0_I1+0+0+0:vgprValuA_X0_I1+0+0+0+1], v[32:39]
/*  mfmaIndex:5  */
ds_read_b128 v[vgprValuB_X2_I0+16:vgprValuB_X2_I0+16+3], v[vgprLocalReadAddrB] offset:10304 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_f64_16x16x4f64 v[40:47], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+1], v[vgprValuA_X0_I1+2+0+0:vgprValuA_X0_I1+2+0+0+1], v[40:47]
/*  mfmaIndex:6  */
ds_read_b128 v[vgprValuB_X2_I0+20:vgprValuB_X2_I0+20+3], v[vgprLocalReadAddrB] offset:12864 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=5 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_f64_16x16x4f64 v[48:55], v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+1], v[vgprValuA_X0_I1+0+0+0:vgprValuA_X0_I1+0+0+0+1], v[48:55]
/*  mfmaIndex:7  */
ds_read_b128 v[vgprValuB_X2_I0+24:vgprValuB_X2_I0+24+3], v[vgprLocalReadAddrB] offset:15424 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=6 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_f64_16x16x4f64 v[56:63], v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+1], v[vgprValuA_X0_I1+2+0+0:vgprValuA_X0_I1+2+0+0+1], v[56:63]
/*  mfmaIndex:8  */
ds_read_b128 v[vgprValuB_X2_I0+28:vgprValuB_X2_I0+28+3], v[vgprLocalReadAddrB] offset:17984 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=7 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_f64_16x16x4f64 v[64:71], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+1], v[vgprValuA_X0_I1+0+0+0:vgprValuA_X0_I1+0+0+0+1], v[64:71]
/*  mfmaIndex:9  */
/* localReadsVacancy: letencyLeft 1 */
v_mfma_f64_16x16x4f64 v[72:79], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+1], v[vgprValuA_X0_I1+2+0+0:vgprValuA_X0_I1+2+0+0+1], v[72:79]
/*  mfmaIndex:10  */
v_mfma_f64_16x16x4f64 v[80:87], v[vgprValuB_X0_I0+20+0+0:vgprValuB_X0_I0+20+0+0+1], v[vgprValuA_X0_I1+0+0+0:vgprValuA_X0_I1+0+0+0+1], v[80:87]
/*  mfmaIndex:11  */
v_mfma_f64_16x16x4f64 v[88:95], v[vgprValuB_X0_I0+20+0+0:vgprValuB_X0_I0+20+0+0+1], v[vgprValuA_X0_I1+2+0+0:vgprValuA_X0_I1+2+0+0+1], v[88:95]
/*  mfmaIndex:12  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[96:103], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+1], v[vgprValuA_X0_I1+0+0+0:vgprValuA_X0_I1+0+0+0+1], v[96:103]
/*  mfmaIndex:13  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[104:111], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+1], v[vgprValuA_X0_I1+2+0+0:vgprValuA_X0_I1+2+0+0+1], v[104:111]
/*  mfmaIndex:14  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[112:119], v[vgprValuB_X0_I0+28+0+0:vgprValuB_X0_I0+28+0+0+1], v[vgprValuA_X0_I1+0+0+0:vgprValuA_X0_I1+0+0+0+1], v[112:119]
/*  mfmaIndex:15  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[120:127], v[vgprValuB_X0_I0+28+0+0:vgprValuB_X0_I0+28+0+0+1], v[vgprValuA_X0_I1+2+0+0:vgprValuA_X0_I1+2+0+0+1], v[120:127]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=1 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=8 */


/* iter 1 (last unrolled loop) */

/*  grEndMfmaIndex:0, lwStartMfmaIndex:48, lwEndMfmaIndex:48  */
/*  numMfmaForLR:13, barrierMfmaIndex:50  */
/*  mfmaIndex:16  */
/* localReadsVacancy: letencyLeft 5 */
s_waitcnt lgkmcnt(0) vmcnt(2)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=1, new=1 newLW=0 newLR=0
v_mfma_f64_16x16x4f64 v[0:7], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X1_I1+0+0+0:vgprValuA_X1_I1+0+0+0+1], v[0:7]
/*  mfmaIndex:17  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[8:15], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X1_I1+2+0+0:vgprValuA_X1_I1+2+0+0+1], v[8:15]
/*  mfmaIndex:18  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[16:23], v[vgprValuB_X0_I0+4+2+0:vgprValuB_X0_I0+4+2+0+1], v[vgprValuA_X1_I1+0+0+0:vgprValuA_X1_I1+0+0+0+1], v[16:23]
/*  mfmaIndex:19  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[24:31], v[vgprValuB_X0_I0+4+2+0:vgprValuB_X0_I0+4+2+0+1], v[vgprValuA_X1_I1+2+0+0:vgprValuA_X1_I1+2+0+0+1], v[24:31]
/*  mfmaIndex:20  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[32:39], v[vgprValuB_X0_I0+8+2+0:vgprValuB_X0_I0+8+2+0+1], v[vgprValuA_X1_I1+0+0+0:vgprValuA_X1_I1+0+0+0+1], v[32:39]
/*  mfmaIndex:21  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[40:47], v[vgprValuB_X0_I0+8+2+0:vgprValuB_X0_I0+8+2+0+1], v[vgprValuA_X1_I1+2+0+0:vgprValuA_X1_I1+2+0+0+1], v[40:47]
/*  mfmaIndex:22  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[48:55], v[vgprValuB_X0_I0+12+2+0:vgprValuB_X0_I0+12+2+0+1], v[vgprValuA_X1_I1+0+0+0:vgprValuA_X1_I1+0+0+0+1], v[48:55]
/*  mfmaIndex:23  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[56:63], v[vgprValuB_X0_I0+12+2+0:vgprValuB_X0_I0+12+2+0+1], v[vgprValuA_X1_I1+2+0+0:vgprValuA_X1_I1+2+0+0+1], v[56:63]
/*  mfmaIndex:24  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[64:71], v[vgprValuB_X0_I0+16+2+0:vgprValuB_X0_I0+16+2+0+1], v[vgprValuA_X1_I1+0+0+0:vgprValuA_X1_I1+0+0+0+1], v[64:71]
/*  mfmaIndex:25  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[72:79], v[vgprValuB_X0_I0+16+2+0:vgprValuB_X0_I0+16+2+0+1], v[vgprValuA_X1_I1+2+0+0:vgprValuA_X1_I1+2+0+0+1], v[72:79]
/*  mfmaIndex:26  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[80:87], v[vgprValuB_X0_I0+20+2+0:vgprValuB_X0_I0+20+2+0+1], v[vgprValuA_X1_I1+0+0+0:vgprValuA_X1_I1+0+0+0+1], v[80:87]
/*  mfmaIndex:27  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[88:95], v[vgprValuB_X0_I0+20+2+0:vgprValuB_X0_I0+20+2+0+1], v[vgprValuA_X1_I1+2+0+0:vgprValuA_X1_I1+2+0+0+1], v[88:95]
/*  mfmaIndex:28  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[96:103], v[vgprValuB_X0_I0+24+2+0:vgprValuB_X0_I0+24+2+0+1], v[vgprValuA_X1_I1+0+0+0:vgprValuA_X1_I1+0+0+0+1], v[96:103]
/*  mfmaIndex:29  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[104:111], v[vgprValuB_X0_I0+24+2+0:vgprValuB_X0_I0+24+2+0+1], v[vgprValuA_X1_I1+2+0+0:vgprValuA_X1_I1+2+0+0+1], v[104:111]
/*  mfmaIndex:30  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[112:119], v[vgprValuB_X0_I0+28+2+0:vgprValuB_X0_I0+28+2+0+1], v[vgprValuA_X1_I1+0+0+0:vgprValuA_X1_I1+0+0+0+1], v[112:119]
/*  mfmaIndex:31  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[120:127], v[vgprValuB_X0_I0+28+2+0:vgprValuB_X0_I0+28+2+0+1], v[vgprValuA_X1_I1+2+0+0:vgprValuA_X1_I1+2+0+0+1], v[120:127]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=1 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=8 */


/* iter 2 (last unrolled loop) */

/*  grEndMfmaIndex:0, lwStartMfmaIndex:48, lwEndMfmaIndex:48  */
/*  numMfmaForLR:13, barrierMfmaIndex:50  */
/*  mfmaIndex:32  */
/* localReadsVacancy: letencyLeft 5 */
s_waitcnt lgkmcnt(0) vmcnt(1)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_mfma_f64_16x16x4f64 v[0:7], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I1+0+0+0:vgprValuA_X2_I1+0+0+0+1], v[0:7]
/*  mfmaIndex:33  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[8:15], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I1+2+0+0:vgprValuA_X2_I1+2+0+0+1], v[8:15]
/*  mfmaIndex:34  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[16:23], v[vgprValuB_X2_I0+4+0+0:vgprValuB_X2_I0+4+0+0+1], v[vgprValuA_X2_I1+0+0+0:vgprValuA_X2_I1+0+0+0+1], v[16:23]
/*  mfmaIndex:35  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[24:31], v[vgprValuB_X2_I0+4+0+0:vgprValuB_X2_I0+4+0+0+1], v[vgprValuA_X2_I1+2+0+0:vgprValuA_X2_I1+2+0+0+1], v[24:31]
/*  mfmaIndex:36  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[32:39], v[vgprValuB_X2_I0+8+0+0:vgprValuB_X2_I0+8+0+0+1], v[vgprValuA_X2_I1+0+0+0:vgprValuA_X2_I1+0+0+0+1], v[32:39]
/*  mfmaIndex:37  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[40:47], v[vgprValuB_X2_I0+8+0+0:vgprValuB_X2_I0+8+0+0+1], v[vgprValuA_X2_I1+2+0+0:vgprValuA_X2_I1+2+0+0+1], v[40:47]
/*  mfmaIndex:38  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[48:55], v[vgprValuB_X2_I0+12+0+0:vgprValuB_X2_I0+12+0+0+1], v[vgprValuA_X2_I1+0+0+0:vgprValuA_X2_I1+0+0+0+1], v[48:55]
/*  mfmaIndex:39  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[56:63], v[vgprValuB_X2_I0+12+0+0:vgprValuB_X2_I0+12+0+0+1], v[vgprValuA_X2_I1+2+0+0:vgprValuA_X2_I1+2+0+0+1], v[56:63]
/*  mfmaIndex:40  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[64:71], v[vgprValuB_X2_I0+16+0+0:vgprValuB_X2_I0+16+0+0+1], v[vgprValuA_X2_I1+0+0+0:vgprValuA_X2_I1+0+0+0+1], v[64:71]
/*  mfmaIndex:41  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[72:79], v[vgprValuB_X2_I0+16+0+0:vgprValuB_X2_I0+16+0+0+1], v[vgprValuA_X2_I1+2+0+0:vgprValuA_X2_I1+2+0+0+1], v[72:79]
/*  mfmaIndex:42  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[80:87], v[vgprValuB_X2_I0+20+0+0:vgprValuB_X2_I0+20+0+0+1], v[vgprValuA_X2_I1+0+0+0:vgprValuA_X2_I1+0+0+0+1], v[80:87]
/*  mfmaIndex:43  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[88:95], v[vgprValuB_X2_I0+20+0+0:vgprValuB_X2_I0+20+0+0+1], v[vgprValuA_X2_I1+2+0+0:vgprValuA_X2_I1+2+0+0+1], v[88:95]
/*  mfmaIndex:44  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[96:103], v[vgprValuB_X2_I0+24+0+0:vgprValuB_X2_I0+24+0+0+1], v[vgprValuA_X2_I1+0+0+0:vgprValuA_X2_I1+0+0+0+1], v[96:103]
/*  mfmaIndex:45  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[104:111], v[vgprValuB_X2_I0+24+0+0:vgprValuB_X2_I0+24+0+0+1], v[vgprValuA_X2_I1+2+0+0:vgprValuA_X2_I1+2+0+0+1], v[104:111]
/*  mfmaIndex:46  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[112:119], v[vgprValuB_X2_I0+28+0+0:vgprValuB_X2_I0+28+0+0+1], v[vgprValuA_X2_I1+0+0+0:vgprValuA_X2_I1+0+0+0+1], v[112:119]
/*  mfmaIndex:47  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[120:127], v[vgprValuB_X2_I0+28+0+0:vgprValuB_X2_I0+28+0+0+1], v[vgprValuA_X2_I1+2+0+0:vgprValuA_X2_I1+2+0+0+1], v[120:127]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=3 skipReadsIterA=1 readsPerIterA=1 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=8 */


/* iter 3 (last unrolled loop) */

/*  grEndMfmaIndex:0, lwStartMfmaIndex:48, lwEndMfmaIndex:48  */
/*  numMfmaForLR:13, barrierMfmaIndex:50  */
/*  mfmaIndex:48  */
s_waitcnt lgkmcnt(0) vmcnt(0)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_mfma_f64_16x16x4f64 v[0:7], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X3_I1+0+0+0:vgprValuA_X3_I1+0+0+0+1], v[0:7]
/*  mfmaIndex:49  */
v_mfma_f64_16x16x4f64 v[8:15], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X3_I1+2+0+0:vgprValuA_X3_I1+2+0+0+1], v[8:15]
/*  mfmaIndex:50  */
v_mfma_f64_16x16x4f64 v[16:23], v[vgprValuB_X2_I0+4+2+0:vgprValuB_X2_I0+4+2+0+1], v[vgprValuA_X3_I1+0+0+0:vgprValuA_X3_I1+0+0+0+1], v[16:23]
/*  mfmaIndex:51  */
v_mfma_f64_16x16x4f64 v[24:31], v[vgprValuB_X2_I0+4+2+0:vgprValuB_X2_I0+4+2+0+1], v[vgprValuA_X3_I1+2+0+0:vgprValuA_X3_I1+2+0+0+1], v[24:31]
/*  mfmaIndex:52  */
v_mfma_f64_16x16x4f64 v[32:39], v[vgprValuB_X2_I0+8+2+0:vgprValuB_X2_I0+8+2+0+1], v[vgprValuA_X3_I1+0+0+0:vgprValuA_X3_I1+0+0+0+1], v[32:39]
/*  mfmaIndex:53  */
v_mfma_f64_16x16x4f64 v[40:47], v[vgprValuB_X2_I0+8+2+0:vgprValuB_X2_I0+8+2+0+1], v[vgprValuA_X3_I1+2+0+0:vgprValuA_X3_I1+2+0+0+1], v[40:47]
/*  mfmaIndex:54  */
v_mfma_f64_16x16x4f64 v[48:55], v[vgprValuB_X2_I0+12+2+0:vgprValuB_X2_I0+12+2+0+1], v[vgprValuA_X3_I1+0+0+0:vgprValuA_X3_I1+0+0+0+1], v[48:55]
/*  mfmaIndex:55  */
v_mfma_f64_16x16x4f64 v[56:63], v[vgprValuB_X2_I0+12+2+0:vgprValuB_X2_I0+12+2+0+1], v[vgprValuA_X3_I1+2+0+0:vgprValuA_X3_I1+2+0+0+1], v[56:63]
/*  mfmaIndex:56  */
v_mfma_f64_16x16x4f64 v[64:71], v[vgprValuB_X2_I0+16+2+0:vgprValuB_X2_I0+16+2+0+1], v[vgprValuA_X3_I1+0+0+0:vgprValuA_X3_I1+0+0+0+1], v[64:71]
/*  mfmaIndex:57  */
v_mfma_f64_16x16x4f64 v[72:79], v[vgprValuB_X2_I0+16+2+0:vgprValuB_X2_I0+16+2+0+1], v[vgprValuA_X3_I1+2+0+0:vgprValuA_X3_I1+2+0+0+1], v[72:79]
/*  mfmaIndex:58  */
v_mfma_f64_16x16x4f64 v[80:87], v[vgprValuB_X2_I0+20+2+0:vgprValuB_X2_I0+20+2+0+1], v[vgprValuA_X3_I1+0+0+0:vgprValuA_X3_I1+0+0+0+1], v[80:87]
/*  mfmaIndex:59  */
v_mfma_f64_16x16x4f64 v[88:95], v[vgprValuB_X2_I0+20+2+0:vgprValuB_X2_I0+20+2+0+1], v[vgprValuA_X3_I1+2+0+0:vgprValuA_X3_I1+2+0+0+1], v[88:95]
/*  mfmaIndex:60  */
v_mfma_f64_16x16x4f64 v[96:103], v[vgprValuB_X2_I0+24+2+0:vgprValuB_X2_I0+24+2+0+1], v[vgprValuA_X3_I1+0+0+0:vgprValuA_X3_I1+0+0+0+1], v[96:103]
/*  mfmaIndex:61  */
v_mfma_f64_16x16x4f64 v[104:111], v[vgprValuB_X2_I0+24+2+0:vgprValuB_X2_I0+24+2+0+1], v[vgprValuA_X3_I1+2+0+0:vgprValuA_X3_I1+2+0+0+1], v[104:111]
/*  mfmaIndex:62  */
v_mfma_f64_16x16x4f64 v[112:119], v[vgprValuB_X2_I0+28+2+0:vgprValuB_X2_I0+28+2+0+1], v[vgprValuA_X3_I1+0+0+0:vgprValuA_X3_I1+0+0+0+1], v[112:119]
/*  mfmaIndex:63  */
v_mfma_f64_16x16x4f64 v[120:127], v[vgprValuB_X2_I0+28+2+0:vgprValuB_X2_I0+28+2+0+1], v[vgprValuA_X3_I1+2+0+0:vgprValuA_X3_I1+2+0+0+1], v[120:127]
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=0 readsPerIterA=1 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=8 */

/* Stores for OptNLL */
Summation_End_OptNLL_17:
s_setprio 0                                        // optimization store
/* endSummation: add vgpr [128...132) to pool (vgprValuA_X0_I0) */
/* endSummation: add vgpr [144...160) to pool (vgprValuB_X0_I0) */
/* endSummation: add vgpr [208...252) to pool */
.set NumFullBlocks, UNDEF
.set WgmRemainder1, UNDEF
.set MagicNumberWgmRemainder1, UNDEF
.set ShadowLimitA, UNDEF
.set ShadowLimitB, UNDEF
.set WrapUA, UNDEF
.set WrapUB, UNDEF
.set GlobalReadIncsA, UNDEF
.set GlobalReadIncsB, UNDEF

/* Mapping of Acc register -> C Vgpr register */

/* Multiply MI out register with Alpha -> C Vgpr register */
/* computeStoreVgprs */
v_lshrrev_b32 v144, 6, v[vgprSerial]               // v144 = v[vgprSerial] / 64
v_lshrrev_b32 v145, 2, v144                        // v145 = v144 / 4
v_mul_lo_u32 v145, 0x10, v145                      // wave coordination offset 1
v_and_b32 v129, 63, v[vgprSerial]                  // v129 = v[vgprSerial] % 64
v_lshrrev_b32 v129, 4, v129                        // v129 = v129 / 16
                                                   // thread0 * continuous_output (multiplier is 1, do nothing)
v_add_u32 v129, v145, v129                         // coordination 1 = wave_id1 + tid1
v_mul_lo_u32 v130, v129, s[sgprStrideC1J]          //  offset 1
v_mul_lo_u32 v131, v129, s[sgprStrideD1J]          //  offset 1
v_and_b32 v128, 3, v144                            // v128 = v144 % 4
v_mul_lo_u32 v128, 0x10, v128                      // wave coordination offset 0
v_and_b32 v145, 15, v[vgprSerial]                  // v145 = v[vgprSerial] % 16
_v_add_lshl_u32 v128, v145, v128, 1                // coordination 0 = wave_id0 + tid0
s_mul_i32 s63, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_add_u32 v128, s63, v128                          // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s63, 128, s[sgprWorkGroup1]              // wgp1 * MT1
v_add_u32 v129, s63, v129                          // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1
GW_B0_E0_20:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=1 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
_v_add_lshl_u32 v146, v131, v128, 0x3              // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=128, coord0Vgpr=128
v_mov_b32 v[vgprValuC+148], v[vgprValuC+0] // copy MI out reg to vreg[0]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+1] // copy MI out reg to vreg[1]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+8] // copy MI out reg to vreg[2]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+9] // copy MI out reg to vreg[3]

/* apply mask, calc new C and issue writes */
buffer_store_dwordx4 v[148:151], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #1 (d1,d0,vc1,vc0) = */
/*    (1,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_mov_b32 v[vgprValuC+148], v[vgprValuC+2] // copy MI out reg to vreg[4]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+3] // copy MI out reg to vreg[5]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+10] // copy MI out reg to vreg[6]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+11] // copy MI out reg to vreg[7]

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[148:151], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #2 (d1,d0,vc1,vc0) = */
/*    (2,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_mov_b32 v[vgprValuC+148], v[vgprValuC+4] // copy MI out reg to vreg[8]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+5] // copy MI out reg to vreg[9]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+12] // copy MI out reg to vreg[10]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+13] // copy MI out reg to vreg[11]

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[148:151], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #3 (d1,d0,vc1,vc0) = */
/*    (3,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_mov_b32 v[vgprValuC+148], v[vgprValuC+6] // copy MI out reg to vreg[12]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+7] // copy MI out reg to vreg[13]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+14] // copy MI out reg to vreg[14]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+15] // copy MI out reg to vreg[15]

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[148:151], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #4 (d1,d0,vc1,vc0) = */
/*    (4,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_mov_b32 v[vgprValuC+148], v[vgprValuC+16] // copy MI out reg to vreg[16]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+17] // copy MI out reg to vreg[17]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+24] // copy MI out reg to vreg[18]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+25] // copy MI out reg to vreg[19]

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[148:151], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #5 (d1,d0,vc1,vc0) = */
/*    (5,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
v_mov_b32 v[vgprValuC+148], v[vgprValuC+18] // copy MI out reg to vreg[20]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+19] // copy MI out reg to vreg[21]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+26] // copy MI out reg to vreg[22]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+27] // copy MI out reg to vreg[23]

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[148:151], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #6 (d1,d0,vc1,vc0) = */
/*    (6,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
v_mov_b32 v[vgprValuC+148], v[vgprValuC+20] // copy MI out reg to vreg[24]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+21] // copy MI out reg to vreg[25]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+28] // copy MI out reg to vreg[26]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+29] // copy MI out reg to vreg[27]

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[148:151], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #7 (d1,d0,vc1,vc0) = */
/*    (7,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
v_mov_b32 v[vgprValuC+148], v[vgprValuC+22] // copy MI out reg to vreg[28]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+23] // copy MI out reg to vreg[29]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+30] // copy MI out reg to vreg[30]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+31] // copy MI out reg to vreg[31]

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[148:151], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #8 (d1,d0,vc1,vc0) = */
/*    (8,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
v_mov_b32 v[vgprValuC+148], v[vgprValuC+32] // copy MI out reg to vreg[32]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+33] // copy MI out reg to vreg[33]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+40] // copy MI out reg to vreg[34]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+41] // copy MI out reg to vreg[35]

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[148:151], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #9 (d1,d0,vc1,vc0) = */
/*    (9,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
v_mov_b32 v[vgprValuC+148], v[vgprValuC+34] // copy MI out reg to vreg[36]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+35] // copy MI out reg to vreg[37]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+42] // copy MI out reg to vreg[38]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+43] // copy MI out reg to vreg[39]

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[148:151], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #10 (d1,d0,vc1,vc0) = */
/*    (10,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
v_mov_b32 v[vgprValuC+148], v[vgprValuC+36] // copy MI out reg to vreg[40]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+37] // copy MI out reg to vreg[41]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+44] // copy MI out reg to vreg[42]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+45] // copy MI out reg to vreg[43]

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[148:151], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #11 (d1,d0,vc1,vc0) = */
/*    (11,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
v_mov_b32 v[vgprValuC+148], v[vgprValuC+38] // copy MI out reg to vreg[44]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+39] // copy MI out reg to vreg[45]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+46] // copy MI out reg to vreg[46]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+47] // copy MI out reg to vreg[47]

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[148:151], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #12 (d1,d0,vc1,vc0) = */
/*    (12,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
v_mov_b32 v[vgprValuC+148], v[vgprValuC+48] // copy MI out reg to vreg[48]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+49] // copy MI out reg to vreg[49]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+56] // copy MI out reg to vreg[50]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+57] // copy MI out reg to vreg[51]

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[148:151], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #13 (d1,d0,vc1,vc0) = */
/*    (13,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
v_mov_b32 v[vgprValuC+148], v[vgprValuC+50] // copy MI out reg to vreg[52]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+51] // copy MI out reg to vreg[53]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+58] // copy MI out reg to vreg[54]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+59] // copy MI out reg to vreg[55]

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[148:151], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #14 (d1,d0,vc1,vc0) = */
/*    (14,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
v_mov_b32 v[vgprValuC+148], v[vgprValuC+52] // copy MI out reg to vreg[56]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+53] // copy MI out reg to vreg[57]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+60] // copy MI out reg to vreg[58]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+61] // copy MI out reg to vreg[59]

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[148:151], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #15 (d1,d0,vc1,vc0) = */
/*    (15,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
v_mov_b32 v[vgprValuC+148], v[vgprValuC+54] // copy MI out reg to vreg[60]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+55] // copy MI out reg to vreg[61]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+62] // copy MI out reg to vreg[62]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+63] // copy MI out reg to vreg[63]

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[148:151], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #16 (d1,d0,vc1,vc0) = */
/*    (16,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
v_mov_b32 v[vgprValuC+148], v[vgprValuC+64] // copy MI out reg to vreg[64]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+65] // copy MI out reg to vreg[65]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+72] // copy MI out reg to vreg[66]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+73] // copy MI out reg to vreg[67]

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[148:151], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #17 (d1,d0,vc1,vc0) = */
/*    (17,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
v_mov_b32 v[vgprValuC+148], v[vgprValuC+66] // copy MI out reg to vreg[68]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+67] // copy MI out reg to vreg[69]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+74] // copy MI out reg to vreg[70]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+75] // copy MI out reg to vreg[71]

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[148:151], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #18 (d1,d0,vc1,vc0) = */
/*    (18,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
v_mov_b32 v[vgprValuC+148], v[vgprValuC+68] // copy MI out reg to vreg[72]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+69] // copy MI out reg to vreg[73]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+76] // copy MI out reg to vreg[74]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+77] // copy MI out reg to vreg[75]

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[148:151], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #19 (d1,d0,vc1,vc0) = */
/*    (19,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
v_mov_b32 v[vgprValuC+148], v[vgprValuC+70] // copy MI out reg to vreg[76]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+71] // copy MI out reg to vreg[77]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+78] // copy MI out reg to vreg[78]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+79] // copy MI out reg to vreg[79]

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[148:151], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #20 (d1,d0,vc1,vc0) = */
/*    (20,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
v_mov_b32 v[vgprValuC+148], v[vgprValuC+80] // copy MI out reg to vreg[80]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+81] // copy MI out reg to vreg[81]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+88] // copy MI out reg to vreg[82]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+89] // copy MI out reg to vreg[83]

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[148:151], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #21 (d1,d0,vc1,vc0) = */
/*    (21,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
v_mov_b32 v[vgprValuC+148], v[vgprValuC+82] // copy MI out reg to vreg[84]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+83] // copy MI out reg to vreg[85]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+90] // copy MI out reg to vreg[86]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+91] // copy MI out reg to vreg[87]

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[148:151], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #22 (d1,d0,vc1,vc0) = */
/*    (22,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
v_mov_b32 v[vgprValuC+148], v[vgprValuC+84] // copy MI out reg to vreg[88]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+85] // copy MI out reg to vreg[89]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+92] // copy MI out reg to vreg[90]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+93] // copy MI out reg to vreg[91]

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[148:151], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #23 (d1,d0,vc1,vc0) = */
/*    (23,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
v_mov_b32 v[vgprValuC+148], v[vgprValuC+86] // copy MI out reg to vreg[92]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+87] // copy MI out reg to vreg[93]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+94] // copy MI out reg to vreg[94]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+95] // copy MI out reg to vreg[95]

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[148:151], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #24 (d1,d0,vc1,vc0) = */
/*    (24,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
v_mov_b32 v[vgprValuC+148], v[vgprValuC+96] // copy MI out reg to vreg[96]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+97] // copy MI out reg to vreg[97]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+104] // copy MI out reg to vreg[98]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+105] // copy MI out reg to vreg[99]

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[148:151], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #25 (d1,d0,vc1,vc0) = */
/*    (25,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
v_mov_b32 v[vgprValuC+148], v[vgprValuC+98] // copy MI out reg to vreg[100]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+99] // copy MI out reg to vreg[101]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+106] // copy MI out reg to vreg[102]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+107] // copy MI out reg to vreg[103]

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[148:151], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #26 (d1,d0,vc1,vc0) = */
/*    (26,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
v_mov_b32 v[vgprValuC+148], v[vgprValuC+100] // copy MI out reg to vreg[104]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+101] // copy MI out reg to vreg[105]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+108] // copy MI out reg to vreg[106]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+109] // copy MI out reg to vreg[107]

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[148:151], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #27 (d1,d0,vc1,vc0) = */
/*    (27,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
v_mov_b32 v[vgprValuC+148], v[vgprValuC+102] // copy MI out reg to vreg[108]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+103] // copy MI out reg to vreg[109]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+110] // copy MI out reg to vreg[110]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+111] // copy MI out reg to vreg[111]

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[148:151], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #28 (d1,d0,vc1,vc0) = */
/*    (28,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
v_mov_b32 v[vgprValuC+148], v[vgprValuC+112] // copy MI out reg to vreg[112]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+113] // copy MI out reg to vreg[113]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+120] // copy MI out reg to vreg[114]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+121] // copy MI out reg to vreg[115]

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[148:151], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #29 (d1,d0,vc1,vc0) = */
/*    (29,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
v_mov_b32 v[vgprValuC+148], v[vgprValuC+114] // copy MI out reg to vreg[116]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+115] // copy MI out reg to vreg[117]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+122] // copy MI out reg to vreg[118]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+123] // copy MI out reg to vreg[119]

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[148:151], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #30 (d1,d0,vc1,vc0) = */
/*    (30,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
v_mov_b32 v[vgprValuC+148], v[vgprValuC+116] // copy MI out reg to vreg[120]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+117] // copy MI out reg to vreg[121]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+124] // copy MI out reg to vreg[122]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+125] // copy MI out reg to vreg[123]

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[148:151], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #31 (d1,d0,vc1,vc0) = */
/*    (31,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
v_mov_b32 v[vgprValuC+148], v[vgprValuC+118] // copy MI out reg to vreg[124]
v_mov_b32 v[vgprValuC+149], v[vgprValuC+119] // copy MI out reg to vreg[125]
v_mov_b32 v[vgprValuC+150], v[vgprValuC+126] // copy MI out reg to vreg[126]
v_mov_b32 v[vgprValuC+151], v[vgprValuC+127] // copy MI out reg to vreg[127]

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[148:151], v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_22                           // jump to end
label_GW_End_22:

s_endpgm                                           // Kernel End
OptNLL_End_16:


/******************************************/
/* Ord. NoLoadLoop - Begin                */
/******************************************/




/* iter 0 (last unrolled loop) */

/*  grEndMfmaIndex:0, lwStartMfmaIndex:48, lwEndMfmaIndex:48  */
/*  numMfmaForLR:13, barrierMfmaIndex:50  */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(0) vmcnt(3)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_mfma_f64_16x16x4f64 v[0:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I1+0+0+0:vgprValuA_X0_I1+0+0+0+1], v[0:7]
/*  mfmaIndex:1  */
ds_read_b128 v[vgprValuB_X2_I0+0:vgprValuB_X2_I0+0+3], v[vgprLocalReadAddrB] offset:64 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_f64_16x16x4f64 v[8:15], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I1+2+0+0:vgprValuA_X0_I1+2+0+0+1], v[8:15]
/*  mfmaIndex:2  */
ds_read_b128 v[vgprValuB_X2_I0+4:vgprValuB_X2_I0+4+3], v[vgprLocalReadAddrB] offset:2624 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_f64_16x16x4f64 v[16:23], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I1+0+0+0:vgprValuA_X0_I1+0+0+0+1], v[16:23]
/*  mfmaIndex:3  */
ds_read_b128 v[vgprValuB_X2_I0+8:vgprValuB_X2_I0+8+3], v[vgprLocalReadAddrB] offset:5184 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_f64_16x16x4f64 v[24:31], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I1+2+0+0:vgprValuA_X0_I1+2+0+0+1], v[24:31]
/*  mfmaIndex:4  */
ds_read_b128 v[vgprValuB_X2_I0+12:vgprValuB_X2_I0+12+3], v[vgprLocalReadAddrB] offset:7744 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_f64_16x16x4f64 v[32:39], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+1], v[vgprValuA_X0_I1+0+0+0:vgprValuA_X0_I1+0+0+0+1], v[32:39]
/*  mfmaIndex:5  */
ds_read_b128 v[vgprValuB_X2_I0+16:vgprValuB_X2_I0+16+3], v[vgprLocalReadAddrB] offset:10304 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_f64_16x16x4f64 v[40:47], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+1], v[vgprValuA_X0_I1+2+0+0:vgprValuA_X0_I1+2+0+0+1], v[40:47]
/*  mfmaIndex:6  */
ds_read_b128 v[vgprValuB_X2_I0+20:vgprValuB_X2_I0+20+3], v[vgprLocalReadAddrB] offset:12864 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=5 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_f64_16x16x4f64 v[48:55], v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+1], v[vgprValuA_X0_I1+0+0+0:vgprValuA_X0_I1+0+0+0+1], v[48:55]
/*  mfmaIndex:7  */
ds_read_b128 v[vgprValuB_X2_I0+24:vgprValuB_X2_I0+24+3], v[vgprLocalReadAddrB] offset:15424 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=6 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_f64_16x16x4f64 v[56:63], v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+1], v[vgprValuA_X0_I1+2+0+0:vgprValuA_X0_I1+2+0+0+1], v[56:63]
/*  mfmaIndex:8  */
ds_read_b128 v[vgprValuB_X2_I0+28:vgprValuB_X2_I0+28+3], v[vgprLocalReadAddrB] offset:17984 // L -> Reg lro=8 swapByteOffset=0 ti=16 vIdx=7 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_f64_16x16x4f64 v[64:71], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+1], v[vgprValuA_X0_I1+0+0+0:vgprValuA_X0_I1+0+0+0+1], v[64:71]
/*  mfmaIndex:9  */
/* localReadsVacancy: letencyLeft 1 */
v_mfma_f64_16x16x4f64 v[72:79], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+1], v[vgprValuA_X0_I1+2+0+0:vgprValuA_X0_I1+2+0+0+1], v[72:79]
/*  mfmaIndex:10  */
v_mfma_f64_16x16x4f64 v[80:87], v[vgprValuB_X0_I0+20+0+0:vgprValuB_X0_I0+20+0+0+1], v[vgprValuA_X0_I1+0+0+0:vgprValuA_X0_I1+0+0+0+1], v[80:87]
/*  mfmaIndex:11  */
v_mfma_f64_16x16x4f64 v[88:95], v[vgprValuB_X0_I0+20+0+0:vgprValuB_X0_I0+20+0+0+1], v[vgprValuA_X0_I1+2+0+0:vgprValuA_X0_I1+2+0+0+1], v[88:95]
/*  mfmaIndex:12  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[96:103], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+1], v[vgprValuA_X0_I1+0+0+0:vgprValuA_X0_I1+0+0+0+1], v[96:103]
/*  mfmaIndex:13  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[104:111], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+1], v[vgprValuA_X0_I1+2+0+0:vgprValuA_X0_I1+2+0+0+1], v[104:111]
/*  mfmaIndex:14  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[112:119], v[vgprValuB_X0_I0+28+0+0:vgprValuB_X0_I0+28+0+0+1], v[vgprValuA_X0_I1+0+0+0:vgprValuA_X0_I1+0+0+0+1], v[112:119]
/*  mfmaIndex:15  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[120:127], v[vgprValuB_X0_I0+28+0+0:vgprValuB_X0_I0+28+0+0+1], v[vgprValuA_X0_I1+2+0+0:vgprValuA_X0_I1+2+0+0+1], v[120:127]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=1 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=8 */


/* iter 1 (last unrolled loop) */

/*  grEndMfmaIndex:0, lwStartMfmaIndex:48, lwEndMfmaIndex:48  */
/*  numMfmaForLR:13, barrierMfmaIndex:50  */
/*  mfmaIndex:16  */
/* localReadsVacancy: letencyLeft 5 */
s_waitcnt lgkmcnt(0) vmcnt(2)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=1, new=1 newLW=0 newLR=0
v_mfma_f64_16x16x4f64 v[0:7], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X1_I1+0+0+0:vgprValuA_X1_I1+0+0+0+1], v[0:7]
/*  mfmaIndex:17  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[8:15], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X1_I1+2+0+0:vgprValuA_X1_I1+2+0+0+1], v[8:15]
/*  mfmaIndex:18  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[16:23], v[vgprValuB_X0_I0+4+2+0:vgprValuB_X0_I0+4+2+0+1], v[vgprValuA_X1_I1+0+0+0:vgprValuA_X1_I1+0+0+0+1], v[16:23]
/*  mfmaIndex:19  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[24:31], v[vgprValuB_X0_I0+4+2+0:vgprValuB_X0_I0+4+2+0+1], v[vgprValuA_X1_I1+2+0+0:vgprValuA_X1_I1+2+0+0+1], v[24:31]
/*  mfmaIndex:20  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[32:39], v[vgprValuB_X0_I0+8+2+0:vgprValuB_X0_I0+8+2+0+1], v[vgprValuA_X1_I1+0+0+0:vgprValuA_X1_I1+0+0+0+1], v[32:39]
/*  mfmaIndex:21  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[40:47], v[vgprValuB_X0_I0+8+2+0:vgprValuB_X0_I0+8+2+0+1], v[vgprValuA_X1_I1+2+0+0:vgprValuA_X1_I1+2+0+0+1], v[40:47]
/*  mfmaIndex:22  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[48:55], v[vgprValuB_X0_I0+12+2+0:vgprValuB_X0_I0+12+2+0+1], v[vgprValuA_X1_I1+0+0+0:vgprValuA_X1_I1+0+0+0+1], v[48:55]
/*  mfmaIndex:23  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[56:63], v[vgprValuB_X0_I0+12+2+0:vgprValuB_X0_I0+12+2+0+1], v[vgprValuA_X1_I1+2+0+0:vgprValuA_X1_I1+2+0+0+1], v[56:63]
/*  mfmaIndex:24  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[64:71], v[vgprValuB_X0_I0+16+2+0:vgprValuB_X0_I0+16+2+0+1], v[vgprValuA_X1_I1+0+0+0:vgprValuA_X1_I1+0+0+0+1], v[64:71]
/*  mfmaIndex:25  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[72:79], v[vgprValuB_X0_I0+16+2+0:vgprValuB_X0_I0+16+2+0+1], v[vgprValuA_X1_I1+2+0+0:vgprValuA_X1_I1+2+0+0+1], v[72:79]
/*  mfmaIndex:26  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[80:87], v[vgprValuB_X0_I0+20+2+0:vgprValuB_X0_I0+20+2+0+1], v[vgprValuA_X1_I1+0+0+0:vgprValuA_X1_I1+0+0+0+1], v[80:87]
/*  mfmaIndex:27  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[88:95], v[vgprValuB_X0_I0+20+2+0:vgprValuB_X0_I0+20+2+0+1], v[vgprValuA_X1_I1+2+0+0:vgprValuA_X1_I1+2+0+0+1], v[88:95]
/*  mfmaIndex:28  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[96:103], v[vgprValuB_X0_I0+24+2+0:vgprValuB_X0_I0+24+2+0+1], v[vgprValuA_X1_I1+0+0+0:vgprValuA_X1_I1+0+0+0+1], v[96:103]
/*  mfmaIndex:29  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[104:111], v[vgprValuB_X0_I0+24+2+0:vgprValuB_X0_I0+24+2+0+1], v[vgprValuA_X1_I1+2+0+0:vgprValuA_X1_I1+2+0+0+1], v[104:111]
/*  mfmaIndex:30  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[112:119], v[vgprValuB_X0_I0+28+2+0:vgprValuB_X0_I0+28+2+0+1], v[vgprValuA_X1_I1+0+0+0:vgprValuA_X1_I1+0+0+0+1], v[112:119]
/*  mfmaIndex:31  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[120:127], v[vgprValuB_X0_I0+28+2+0:vgprValuB_X0_I0+28+2+0+1], v[vgprValuA_X1_I1+2+0+0:vgprValuA_X1_I1+2+0+0+1], v[120:127]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=1 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=8 */


/* iter 2 (last unrolled loop) */

/*  grEndMfmaIndex:0, lwStartMfmaIndex:48, lwEndMfmaIndex:48  */
/*  numMfmaForLR:13, barrierMfmaIndex:50  */
/*  mfmaIndex:32  */
/* localReadsVacancy: letencyLeft 5 */
s_waitcnt lgkmcnt(0) vmcnt(1)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_mfma_f64_16x16x4f64 v[0:7], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I1+0+0+0:vgprValuA_X2_I1+0+0+0+1], v[0:7]
/*  mfmaIndex:33  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[8:15], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I1+2+0+0:vgprValuA_X2_I1+2+0+0+1], v[8:15]
/*  mfmaIndex:34  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[16:23], v[vgprValuB_X2_I0+4+0+0:vgprValuB_X2_I0+4+0+0+1], v[vgprValuA_X2_I1+0+0+0:vgprValuA_X2_I1+0+0+0+1], v[16:23]
/*  mfmaIndex:35  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[24:31], v[vgprValuB_X2_I0+4+0+0:vgprValuB_X2_I0+4+0+0+1], v[vgprValuA_X2_I1+2+0+0:vgprValuA_X2_I1+2+0+0+1], v[24:31]
/*  mfmaIndex:36  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[32:39], v[vgprValuB_X2_I0+8+0+0:vgprValuB_X2_I0+8+0+0+1], v[vgprValuA_X2_I1+0+0+0:vgprValuA_X2_I1+0+0+0+1], v[32:39]
/*  mfmaIndex:37  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[40:47], v[vgprValuB_X2_I0+8+0+0:vgprValuB_X2_I0+8+0+0+1], v[vgprValuA_X2_I1+2+0+0:vgprValuA_X2_I1+2+0+0+1], v[40:47]
/*  mfmaIndex:38  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[48:55], v[vgprValuB_X2_I0+12+0+0:vgprValuB_X2_I0+12+0+0+1], v[vgprValuA_X2_I1+0+0+0:vgprValuA_X2_I1+0+0+0+1], v[48:55]
/*  mfmaIndex:39  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[56:63], v[vgprValuB_X2_I0+12+0+0:vgprValuB_X2_I0+12+0+0+1], v[vgprValuA_X2_I1+2+0+0:vgprValuA_X2_I1+2+0+0+1], v[56:63]
/*  mfmaIndex:40  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[64:71], v[vgprValuB_X2_I0+16+0+0:vgprValuB_X2_I0+16+0+0+1], v[vgprValuA_X2_I1+0+0+0:vgprValuA_X2_I1+0+0+0+1], v[64:71]
/*  mfmaIndex:41  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[72:79], v[vgprValuB_X2_I0+16+0+0:vgprValuB_X2_I0+16+0+0+1], v[vgprValuA_X2_I1+2+0+0:vgprValuA_X2_I1+2+0+0+1], v[72:79]
/*  mfmaIndex:42  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[80:87], v[vgprValuB_X2_I0+20+0+0:vgprValuB_X2_I0+20+0+0+1], v[vgprValuA_X2_I1+0+0+0:vgprValuA_X2_I1+0+0+0+1], v[80:87]
/*  mfmaIndex:43  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[88:95], v[vgprValuB_X2_I0+20+0+0:vgprValuB_X2_I0+20+0+0+1], v[vgprValuA_X2_I1+2+0+0:vgprValuA_X2_I1+2+0+0+1], v[88:95]
/*  mfmaIndex:44  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[96:103], v[vgprValuB_X2_I0+24+0+0:vgprValuB_X2_I0+24+0+0+1], v[vgprValuA_X2_I1+0+0+0:vgprValuA_X2_I1+0+0+0+1], v[96:103]
/*  mfmaIndex:45  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[104:111], v[vgprValuB_X2_I0+24+0+0:vgprValuB_X2_I0+24+0+0+1], v[vgprValuA_X2_I1+2+0+0:vgprValuA_X2_I1+2+0+0+1], v[104:111]
/*  mfmaIndex:46  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[112:119], v[vgprValuB_X2_I0+28+0+0:vgprValuB_X2_I0+28+0+0+1], v[vgprValuA_X2_I1+0+0+0:vgprValuA_X2_I1+0+0+0+1], v[112:119]
/*  mfmaIndex:47  */
/* localReadsVacancy: letencyLeft 5 */
v_mfma_f64_16x16x4f64 v[120:127], v[vgprValuB_X2_I0+28+0+0:vgprValuB_X2_I0+28+0+0+1], v[vgprValuA_X2_I1+2+0+0:vgprValuA_X2_I1+2+0+0+1], v[120:127]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=3 skipReadsIterA=1 readsPerIterA=1 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=8 */


/* iter 3 (last unrolled loop) */

/*  grEndMfmaIndex:0, lwStartMfmaIndex:48, lwEndMfmaIndex:48  */
/*  numMfmaForLR:13, barrierMfmaIndex:50  */
/*  mfmaIndex:48  */
s_waitcnt lgkmcnt(0) vmcnt(0)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_mfma_f64_16x16x4f64 v[0:7], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X3_I1+0+0+0:vgprValuA_X3_I1+0+0+0+1], v[0:7]
/*  mfmaIndex:49  */
v_mfma_f64_16x16x4f64 v[8:15], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X3_I1+2+0+0:vgprValuA_X3_I1+2+0+0+1], v[8:15]
/*  mfmaIndex:50  */
v_mfma_f64_16x16x4f64 v[16:23], v[vgprValuB_X2_I0+4+2+0:vgprValuB_X2_I0+4+2+0+1], v[vgprValuA_X3_I1+0+0+0:vgprValuA_X3_I1+0+0+0+1], v[16:23]
/*  mfmaIndex:51  */
v_mfma_f64_16x16x4f64 v[24:31], v[vgprValuB_X2_I0+4+2+0:vgprValuB_X2_I0+4+2+0+1], v[vgprValuA_X3_I1+2+0+0:vgprValuA_X3_I1+2+0+0+1], v[24:31]
/*  mfmaIndex:52  */
v_mfma_f64_16x16x4f64 v[32:39], v[vgprValuB_X2_I0+8+2+0:vgprValuB_X2_I0+8+2+0+1], v[vgprValuA_X3_I1+0+0+0:vgprValuA_X3_I1+0+0+0+1], v[32:39]
/*  mfmaIndex:53  */
v_mfma_f64_16x16x4f64 v[40:47], v[vgprValuB_X2_I0+8+2+0:vgprValuB_X2_I0+8+2+0+1], v[vgprValuA_X3_I1+2+0+0:vgprValuA_X3_I1+2+0+0+1], v[40:47]
/*  mfmaIndex:54  */
v_mfma_f64_16x16x4f64 v[48:55], v[vgprValuB_X2_I0+12+2+0:vgprValuB_X2_I0+12+2+0+1], v[vgprValuA_X3_I1+0+0+0:vgprValuA_X3_I1+0+0+0+1], v[48:55]
/*  mfmaIndex:55  */
v_mfma_f64_16x16x4f64 v[56:63], v[vgprValuB_X2_I0+12+2+0:vgprValuB_X2_I0+12+2+0+1], v[vgprValuA_X3_I1+2+0+0:vgprValuA_X3_I1+2+0+0+1], v[56:63]
/*  mfmaIndex:56  */
v_mfma_f64_16x16x4f64 v[64:71], v[vgprValuB_X2_I0+16+2+0:vgprValuB_X2_I0+16+2+0+1], v[vgprValuA_X3_I1+0+0+0:vgprValuA_X3_I1+0+0+0+1], v[64:71]
/*  mfmaIndex:57  */
v_mfma_f64_16x16x4f64 v[72:79], v[vgprValuB_X2_I0+16+2+0:vgprValuB_X2_I0+16+2+0+1], v[vgprValuA_X3_I1+2+0+0:vgprValuA_X3_I1+2+0+0+1], v[72:79]
/*  mfmaIndex:58  */
v_mfma_f64_16x16x4f64 v[80:87], v[vgprValuB_X2_I0+20+2+0:vgprValuB_X2_I0+20+2+0+1], v[vgprValuA_X3_I1+0+0+0:vgprValuA_X3_I1+0+0+0+1], v[80:87]
/*  mfmaIndex:59  */
v_mfma_f64_16x16x4f64 v[88:95], v[vgprValuB_X2_I0+20+2+0:vgprValuB_X2_I0+20+2+0+1], v[vgprValuA_X3_I1+2+0+0:vgprValuA_X3_I1+2+0+0+1], v[88:95]
/*  mfmaIndex:60  */
v_mfma_f64_16x16x4f64 v[96:103], v[vgprValuB_X2_I0+24+2+0:vgprValuB_X2_I0+24+2+0+1], v[vgprValuA_X3_I1+0+0+0:vgprValuA_X3_I1+0+0+0+1], v[96:103]
/*  mfmaIndex:61  */
v_mfma_f64_16x16x4f64 v[104:111], v[vgprValuB_X2_I0+24+2+0:vgprValuB_X2_I0+24+2+0+1], v[vgprValuA_X3_I1+2+0+0:vgprValuA_X3_I1+2+0+0+1], v[104:111]
/*  mfmaIndex:62  */
v_mfma_f64_16x16x4f64 v[112:119], v[vgprValuB_X2_I0+28+2+0:vgprValuB_X2_I0+28+2+0+1], v[vgprValuA_X3_I1+0+0+0:vgprValuA_X3_I1+0+0+0+1], v[112:119]
/*  mfmaIndex:63  */
v_mfma_f64_16x16x4f64 v[120:127], v[vgprValuB_X2_I0+28+2+0:vgprValuB_X2_I0+28+2+0+1], v[vgprValuA_X3_I1+2+0+0:vgprValuA_X3_I1+2+0+0+1], v[120:127]
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=0 readsPerIterA=1 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=8 */

PrefetchGlobalLastIterEnd_4:


/******************************************/
/* Tail Loop                              */
/******************************************/


/* local write reset offsets a */



/* local write reset offsets b */


/* tail loop: add vgpr [132...136) to pool (vgprValuA_X1_I0) */
/* tail loop: add vgpr [136...140) to pool (vgprValuA_X2_I0) */
/* tail loop: add vgpr [140...144) to pool (vgprValuA_X3_I0) */
/* tail loop: add vgpr [160...176) to pool (vgprValuB_X1_I0) */
/* tail loop: add vgpr [176...192) to pool (vgprValuB_X2_I0) */
/* tail loop: add vgpr [192...208) to pool (vgprValuB_X3_I0) */

//numIterL = (((sizeL % LOCAL_DEPTHU) + LOCAL_SPLITU - 1) / LOCAL_SPLITU)
s_and_b32 s[sgprLoopCounterL], 15, s[sgprSizesSum+0] // s[sgprLoopCounterL] = s[sgprSizesSum+0] % 16
s_cmp_eq_u32 s[sgprLoopCounterL], 0x0              // numIterL == 0
s_mov_b32 s[sgprOrigLoopCounter], 0                // repurpose to count each localRead increment
s_cbranch_scc1 SkipTailLoopL_7                     // skip to end of tail loop b/c numIter==0


/* remove stagger offsets for tail loop */

s_sub_i32 s68, 3, s[sgprStaggerUIter]              //
s_mul_hi_i32 s69, s68, s[sgprGlobalReadIncsA+0]    // start offset S in bytes
s_mul_i32 s68, s68, s[sgprGlobalReadIncsA+0]       // start offset S in bytes
s_sub_u32 s68, s68, s[sgprWrapUA]                  // S - WrapU
s_subb_u32 s69, s69, s[sgprWrapUA+1]               // S - WrapU
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s68        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], s69      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s68 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s69 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

s_sub_i32 s68, 3, s[sgprStaggerUIter]              //
s_mul_hi_i32 s69, s68, s[sgprGlobalReadIncsB+0]    // start offset S in bytes
s_mul_i32 s68, s68, s[sgprGlobalReadIncsB+0]       // start offset S in bytes
s_sub_u32 s68, s68, s[sgprWrapUB]                  // S - WrapU
s_subb_u32 s69, s69, s[sgprWrapUB+1]               // S - WrapU
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s68        // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], s69      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s68 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s69 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32


/* Update M0 for DTLDS */



/* global read a */

/* g2l=0, load component 0 */
buffer_load_dwordx2 v[vgprG2LA+0+0:vgprG2LA+0+0+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
buffer_load_dwordx2 v[vgprG2LA+0+2:vgprG2LA+0+2+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:8 // load one buffer value
/* g2l=4, load component 0 */
buffer_load_dwordx2 v[vgprG2LA+4+0:vgprG2LA+4+0+1], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=4, load component 1 */
buffer_load_dwordx2 v[vgprG2LA+4+2:vgprG2LA+4+2+1], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0, offen offset:8 // load one buffer value
/* g2l=8, load component 0 */
buffer_load_dwordx2 v[vgprG2LA+8+0:vgprG2LA+8+0+1], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=8, load component 1 */
buffer_load_dwordx2 v[vgprG2LA+8+2:vgprG2LA+8+2+1], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0, offen offset:8 // load one buffer value
/* g2l=12, load component 0 */
buffer_load_dwordx2 v[vgprG2LA+12+0:vgprG2LA+12+0+1], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=12, load component 1 */
buffer_load_dwordx2 v[vgprG2LA+12+2:vgprG2LA+12+2+1], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0, offen offset:8 // load one buffer value


/* Update M0 for DTLDS */



/* global read b */

/* g2l=0, load component 0 */
buffer_load_dwordx2 v[vgprG2LB+0+0:vgprG2LB+0+0+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
buffer_load_dwordx2 v[vgprG2LB+0+2:vgprG2LB+0+2+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:8 // load one buffer value
/* g2l=4, load component 0 */
buffer_load_dwordx2 v[vgprG2LB+4+0:vgprG2LB+4+0+1], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=4, load component 1 */
buffer_load_dwordx2 v[vgprG2LB+4+2:vgprG2LB+4+2+1], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:8 // load one buffer value
/* g2l=8, load component 0 */
buffer_load_dwordx2 v[vgprG2LB+8+0:vgprG2LB+8+0+1], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=8, load component 1 */
buffer_load_dwordx2 v[vgprG2LB+8+2:vgprG2LB+8+2+1], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0, offen offset:8 // load one buffer value
/* g2l=12, load component 0 */
buffer_load_dwordx2 v[vgprG2LB+12+0:vgprG2LB+12+0+1], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=12, load component 1 */
buffer_load_dwordx2 v[vgprG2LB+12+2:vgprG2LB+12+2+1], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0, offen offset:8 // load one buffer value

s_waitcnt vmcnt(0)                                 // lgkmcnt=-1 vmcnt=02wait for global read

// Skip force waitcnt0
s_barrier //




/* local write a */

ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+4:vgprG2LA+4+3] offset:4096 // lwoA_0_0_1_0 = (0*LSCA) + (1*LSPA)(*MT0I+PAD) = 4096
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+8:vgprG2LA+8+3] offset:8192 // lwoA_0_0_2_0 = (0*LSCA) + (2*LSPA)(*MT0I+PAD) = 8192
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+12:vgprG2LA+12+3] offset:12288 // lwoA_0_0_3_0 = (0*LSCA) + (3*LSPA)(*MT0I+PAD) = 12288


/* local write b */

ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+4:vgprG2LB+4+3] offset:1280 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 1280
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+8:vgprG2LB+8+3] offset:2560 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 2560
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+12:vgprG2LB+12+3] offset:3840 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 3840


/* Recalc local read offsets */

/*lr0I*/
v_and_b32 v134, 63, v[vgprSerial]                  // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v133, 15, v134                           // 1. N offset: nIdx = wtid % MI_N(16)
                                                   // 1. N offset: nOffset = nIdx * nStride(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v132, 4, v134                        // 2. block offset: bnIdx = wtid / dividedForBlkId(16)
v_and_b32 v132, 0, v132                            // 2. block offset: bnIdx = bnIdx % num1DBlocks(1)
v_lshlrev_b32 v132, 0x4, v132                      // 2. block offset: bnOffset = bnIdx * strideBlock(16)
_v_add_u32 v133, v132, v133                        // 3. add N and block offset: bnOffset = block and N offset
v_lshlrev_b32 v133, 0x1, v133                      // 3. apply VectorWidth: bnOffset = bnOffset * vw(2)
v_lshrrev_b32 v134, 4, v134                        // 4. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshlrev_b32 v134, 0x7, v134                      // 4. K offset: lrKOffset = kIdx * mStride(128)
_v_add_u32 v133, v134, v133                        // 5. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v132, 6, v[vgprSerial]               // 6. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
v_and_b32 v132, 3, v132                            // 6. wave offset in M dimen: wtid0 = wtid / num1DWaves(4)
v_lshlrev_b32 v132, 0x5, v132                      // 6. wave offset in M dimen: wOffset = wtid0 * W0Stride(32)
_v_add_u32 v133, v132, v133                        // 7. final local read offset: flrOffset = lrOffset + WOffset
/*lr1J*/
v_and_b32 v135, 63, v[vgprSerial]                  // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v134, 15, v135                           // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v134, 0x4, v134                      // 1. N offset: nOffset = nIdx * nStride(16)
v_lshrrev_b32 v132, 4, v135                        // 2. block offset: bnIdx = wtid / dividedForBlkId(16)
v_and_b32 v132, 0, v132                            // 2. block offset: bnIdx = bnIdx % num1DBlocks(1)
v_lshlrev_b32 v132, 0x8, v132                      // 2. block offset: bnOffset = bnIdx * strideBlock(256)
_v_add_u32 v134, v132, v134                        // 3. add N and block offset: bnOffset = block and N offset
                                                   // 3. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v135, 4, v135                        // 4. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
                                                   // 4. K offset: lrKOffset = kIdx * mStride(1) (multiplier is 1, do nothing)
_v_add_u32 v134, v135, v134                        // 5. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v132, 8, v[vgprSerial]               // LSU offset: sgid = Serial / subGroup(256)
s_mov_b32 s68, 128                                 // LSU offset: stirde = MT0(128) + PAD0(0)
v_mul_lo_u32 v132, s68, v132                       // LSU offset: lsuoffset = sgid*(MT0+PAD)
_v_add_lshl_u32 v[vgprLocalReadAddrA], v132, v133, 0x3 // Final Offset: offset = (lro0*VW+lsuoffset)*bpe
/* N/A */
v_lshrrev_b32 v132, 8, v[vgprSerial]               // LSU offset: sgid = Serial / subGroup(256)
s_mov_b32 s68, 128                                 // LSU offset: stirde = MT1(128) + PAD1(0)
v_mul_lo_u32 v132, s68, v132                       // LSU offset: lsuoffset = sgid*(MT1+PAD)
_v_add_lshl_u32 v[vgprLocalReadAddrB], v132, v134, 0x3 // Final Offset: offset = (lro1*VW+lsuoffset)*bpe
v_lshrrev_b32 v133, 7, v[vgprLocalReadAddrB]       // Final Offset: padding 4 per block 128
v_lshlrev_b32 v133, 0x5, v133                      // Final Offset: padding 4 per block 128
_v_add_u32 v[vgprLocalReadAddrB], v133, v[vgprLocalReadAddrB] // Final Offset: add padding 4 per block 128
_v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, 0x4000, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-15wait for local write

// Skip force waitcnt0
s_barrier //


/* local read reset offsets a */



/* local read reset offsets b */



/* local read init pointers a */


/* localReadInitPointers */


/* local read init pointers b */


/* localReadInitPointers */


/* tail loop: macs */

TailLoopBeginL_5:


/* local read a */

ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0


/* local read b */

ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB] offset:2560 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrB] offset:5120 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrB] offset:7680 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b64 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+1], v[vgprLocalReadAddrB] offset:10240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b64 v[vgprValuB_X0_I0+10:vgprValuB_X0_I0+10+1], v[vgprLocalReadAddrB] offset:12800 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=5 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b64 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+1], v[vgprLocalReadAddrB] offset:15360 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=6 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b64 v[vgprValuB_X0_I0+14:vgprValuB_X0_I0+14+1], v[vgprLocalReadAddrB] offset:17920 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=7 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0


/* local read inc a */

s_mov_b32 s68, 0x1000                              // inc
_v_add_co_u32 v[vgprLocalReadAddrA], vcc, s68, v[vgprLocalReadAddrA] // lrA += 4096 (LSU*(MT+PAD)*bpe)


/* local read inc b */

s_mov_b32 s68, 0x20                                // inc
_v_add_co_u32 v[vgprLocalReadAddrB], vcc, s68, v[vgprLocalReadAddrB] // lrB += 32 (LSU*(MT+PAD)*bpe)

s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-14wait for local read


v_and_b32 v132, 63, v[vgprSerial]                  // v132 = v[vgprSerial] % 64
v_lshrrev_b32 v132, 4, v132                        // v132 = v132 / 16
                                                   // v132 = v132 * 1 (multiplier is 1, do nothing)
v_cmp_ge_i32 s[68:69], v132, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+0+0], v[vgprValuA_X0_I0+0+0], 0x0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+2+0], v[vgprValuA_X0_I0+2+0], 0x0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+0+0], v[vgprValuB_X0_I0+0+0], 0x0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+2+0], v[vgprValuB_X0_I0+2+0], 0x0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+4+0], v[vgprValuB_X0_I0+4+0], 0x0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+6+0], v[vgprValuB_X0_I0+6+0], 0x0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+8+0], v[vgprValuB_X0_I0+8+0], 0x0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+10+0], v[vgprValuB_X0_I0+10+0], 0x0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+12+0], v[vgprValuB_X0_I0+12+0], 0x0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+14+0], v[vgprValuB_X0_I0+14+0], 0x0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+0+1], v[vgprValuA_X0_I0+0+1], 0x0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+2+1], v[vgprValuA_X0_I0+2+1], 0x0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+0+1], v[vgprValuB_X0_I0+0+1], 0x0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+2+1], v[vgprValuB_X0_I0+2+1], 0x0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+4+1], v[vgprValuB_X0_I0+4+1], 0x0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+6+1], v[vgprValuB_X0_I0+6+1], 0x0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+8+1], v[vgprValuB_X0_I0+8+1], 0x0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+10+1], v[vgprValuB_X0_I0+10+1], 0x0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+12+1], v[vgprValuB_X0_I0+12+1], 0x0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+14+1], v[vgprValuB_X0_I0+14+1], 0x0, s[68:69] // set 0 if K_idx >= sizeL
s_nop 1
v_mfma_f64_16x16x4f64 v[0:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[0:7]
v_mfma_f64_16x16x4f64 v[8:15], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[8:15]
v_mfma_f64_16x16x4f64 v[16:23], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[16:23]
v_mfma_f64_16x16x4f64 v[24:31], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[24:31]
v_mfma_f64_16x16x4f64 v[32:39], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[32:39]
v_mfma_f64_16x16x4f64 v[40:47], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[40:47]
v_mfma_f64_16x16x4f64 v[48:55], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[48:55]
v_mfma_f64_16x16x4f64 v[56:63], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[56:63]
v_mfma_f64_16x16x4f64 v[64:71], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[64:71]
v_mfma_f64_16x16x4f64 v[72:79], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[72:79]
v_mfma_f64_16x16x4f64 v[80:87], v[vgprValuB_X0_I0+10+0+0:vgprValuB_X0_I0+10+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[80:87]
v_mfma_f64_16x16x4f64 v[88:95], v[vgprValuB_X0_I0+10+0+0:vgprValuB_X0_I0+10+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[88:95]
v_mfma_f64_16x16x4f64 v[96:103], v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[96:103]
v_mfma_f64_16x16x4f64 v[104:111], v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[104:111]
v_mfma_f64_16x16x4f64 v[112:119], v[vgprValuB_X0_I0+14+0+0:vgprValuB_X0_I0+14+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[112:119]
v_mfma_f64_16x16x4f64 v[120:127], v[vgprValuB_X0_I0+14+0+0:vgprValuB_X0_I0+14+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[120:127]


/* closeLoop loopL finalLoop=1 tailLoop=1 */
s_sub_i32 s[sgprLoopCounterL], s[sgprLoopCounterL], 0x4 // dec counterL (tailLoop)
s_add_u32 s[sgprOrigLoopCounter], s[sgprOrigLoopCounter], 0x4 // inc counterL
s_cmp_le_i32 s[sgprLoopCounterL], 0x0              // counterL<=0
s_cbranch_scc0 TailLoopBeginL_5                    // restart LoopL
TailLoopEndL_6:

SkipTailLoopL_7:

Summation_End_25:
s_setprio 0                                        // optimization store
/* endSummation: add vgpr [128...132) to pool (vgprValuA_X0_I0) */
/* endSummation: add vgpr [144...160) to pool (vgprValuB_X0_I0) */
/* endSummation: add vgpr [208...252) to pool */
.set NumFullBlocks, UNDEF
.set WgmRemainder1, UNDEF
.set MagicNumberWgmRemainder1, UNDEF
.set ShadowLimitA, UNDEF
.set ShadowLimitB, UNDEF
.set WrapUA, UNDEF
.set WrapUB, UNDEF
.set GlobalReadIncsA, UNDEF
.set GlobalReadIncsB, UNDEF

/* Mapping of Acc register -> C Vgpr register */

/* Multiply MI out register with Alpha -> C Vgpr register */

// TODO in Generator
// skip shift vector if M % 2 == 0
s_and_b32 s63, 0x1, s[sgprSizeI]
s_cbranch_scc0 label_0029                                // done shifting

/* shift vector components d0 */

v_mov_b32 v131, s[sgprWorkGroup0]                  //
v_mul_i32_i24 v131, -0x80, v131                    // wg*MT
_v_add_co_u32 v131, vcc, s[sgprSizesFree+0], v131  // wgMT = Size - wg*MT
v_mov_b32 v132, 0x80                               // MT
v_cmp_lt_u32 s[64:65], v131, v132                  // wgMT < MT
v_cndmask_b32 v131, v132, v131, s[64:65]           // wgMT = (wgMT < MT) ? wgMT : MT
v_lshrrev_b32 v133, 6, v[vgprSerial]               // v133 = v[vgprSerial] / 64
v_and_b32 v133, 3, v133                            // v133 = v133 % 4
v_lshrrev_b32 v134, 5, v131                        // v134 = v131 / 32
v_and_b32 v134, 3, v134                            // v134 = v134 % 4
v_cmp_eq_u32 s[64:65], v134, v133                  // wave_id == block_belong_to_wave?
v_cndmask_b32 v131, v132, v131, s[64:65]           // wgMT = (wgMT < MT) ? wgMT : MT

/* mbReg: which mb block need to shift, mb(matrixInstCoal(16) * VectorWidth(2)) */
v_lshrrev_b32 v132, 5, v131                        // v132 = v131 / 32
v_lshlrev_b32 v134, 0x0, v133                      // v134 = v133 * 1
_v_sub_u32 v132, v132, v134                        //

/* gbReg: glvw block id */
v_lshrrev_b32 v134, 1, v131                        // v134 = v131 / 2

/* tgbReg: glvw block id */
v_lshrrev_b32 v135, 0, v[vgprSerial]               // v135 = v[vgprSerial] / 1
v_and_b32 v135, 15, v135                           // v135 = v135 % 16
v_lshlrev_b32 v135, 0x1, v135                      // v135 = v135 * 2
v_lshrrev_b32 v135, 1, v135                        // v135 = v135 / 2
v_lshlrev_b32 v133, 0x4, v133                      // v133 = v133 * 16
_v_add_co_u32 v135, vcc, v133, v135                // tgbReg = (tid_coal * continOut) / GLVW
_v_sub_u32 v134, v134, v135                        //

/* vwReg: glvw in which vw block? */
v_and_b32 v133, 1, v131                            // permute register between threads
v_lshrrev_b32 v133, 1, v133                        // permute register between threads

/* rReg : reminder of M_size % GlobalLoadVectorWidth */
v_and_b32 v135, 1, v131                            // v135 = v131 % 2
v_cmp_eq_u32 vcc, v135, 0x1                        // wgMT%VW == 1
s_cbranch_vccnz label_0026                         // branch to shift d0 r=1
s_branch label_0029                                // no shifting

/******************************************/
/* shift d0 r=1                           */
/******************************************/
label_0026:
v_cmp_eq_u32 vcc, v132, 0x0                        //
s_cbranch_vccnz label_0027                         // branch to shift d0 r1 mb0

/******************************************/
/* shift d0 r=1 mb=0                      */
/******************************************/
label_0027: // r1 mb0
v_cmp_eq_u32 vcc, v133, 0x0                        //
s_cbranch_vccnz label_0028                         // branch to shift d0 r1 mb0 vw0

/******************************************/
/* shift d0 r=1 mb=0 vw0                  */
/******************************************/
label_0028: // r1 mb0 vw0
s_mov_b32 s64, 0                                   //
v_cmpx_eq_u32 s[64:65], v134, s64                  // is thread in edge glvw region
v_and_b32 v128, 63, v[vgprSerial]                  // permute register between threads
v_lshlrev_b32 v128, 2, v128                        // permute register between threads
v_mov_b32 v135, v8                                 // glvw 1 mb 0 tt1 0 r 0
v_mov_b32 v0, v135                                 //
v_mov_b32 v135, v9                                 // glvw 1 mb 0 tt1 0 r 1
v_mov_b32 v1, v135                                 //
v_mov_b32 v135, v10                                // glvw 1 mb 0 tt1 1 r 0
v_mov_b32 v2, v135                                 //
v_mov_b32 v135, v11                                // glvw 1 mb 0 tt1 1 r 1
v_mov_b32 v3, v135                                 //
v_mov_b32 v135, v12                                // glvw 1 mb 0 tt1 2 r 0
v_mov_b32 v4, v135                                 //
v_mov_b32 v135, v13                                // glvw 1 mb 0 tt1 2 r 1
v_mov_b32 v5, v135                                 //
v_mov_b32 v135, v14                                // glvw 1 mb 0 tt1 3 r 0
v_mov_b32 v6, v135                                 //
v_mov_b32 v135, v15                                // glvw 1 mb 0 tt1 3 r 1
v_mov_b32 v7, v135                                 //
v_mov_b32 v135, v24                                // glvw 1 mb 0 tt1 4 r 0
v_mov_b32 v16, v135                                //
v_mov_b32 v135, v25                                // glvw 1 mb 0 tt1 4 r 1
v_mov_b32 v17, v135                                //
v_mov_b32 v135, v26                                // glvw 1 mb 0 tt1 5 r 0
v_mov_b32 v18, v135                                //
v_mov_b32 v135, v27                                // glvw 1 mb 0 tt1 5 r 1
v_mov_b32 v19, v135                                //
v_mov_b32 v135, v28                                // glvw 1 mb 0 tt1 6 r 0
v_mov_b32 v20, v135                                //
v_mov_b32 v135, v29                                // glvw 1 mb 0 tt1 6 r 1
v_mov_b32 v21, v135                                //
v_mov_b32 v135, v30                                // glvw 1 mb 0 tt1 7 r 0
v_mov_b32 v22, v135                                //
v_mov_b32 v135, v31                                // glvw 1 mb 0 tt1 7 r 1
v_mov_b32 v23, v135                                //
v_mov_b32 v135, v40                                // glvw 1 mb 0 tt1 8 r 0
v_mov_b32 v32, v135                                //
v_mov_b32 v135, v41                                // glvw 1 mb 0 tt1 8 r 1
v_mov_b32 v33, v135                                //
v_mov_b32 v135, v42                                // glvw 1 mb 0 tt1 9 r 0
v_mov_b32 v34, v135                                //
v_mov_b32 v135, v43                                // glvw 1 mb 0 tt1 9 r 1
v_mov_b32 v35, v135                                //
v_mov_b32 v135, v44                                // glvw 1 mb 0 tt1 10 r 0
v_mov_b32 v36, v135                                //
v_mov_b32 v135, v45                                // glvw 1 mb 0 tt1 10 r 1
v_mov_b32 v37, v135                                //
v_mov_b32 v135, v46                                // glvw 1 mb 0 tt1 11 r 0
v_mov_b32 v38, v135                                //
v_mov_b32 v135, v47                                // glvw 1 mb 0 tt1 11 r 1
v_mov_b32 v39, v135                                //
v_mov_b32 v135, v56                                // glvw 1 mb 0 tt1 12 r 0
v_mov_b32 v48, v135                                //
v_mov_b32 v135, v57                                // glvw 1 mb 0 tt1 12 r 1
v_mov_b32 v49, v135                                //
v_mov_b32 v135, v58                                // glvw 1 mb 0 tt1 13 r 0
v_mov_b32 v50, v135                                //
v_mov_b32 v135, v59                                // glvw 1 mb 0 tt1 13 r 1
v_mov_b32 v51, v135                                //
v_mov_b32 v135, v60                                // glvw 1 mb 0 tt1 14 r 0
v_mov_b32 v52, v135                                //
v_mov_b32 v135, v61                                // glvw 1 mb 0 tt1 14 r 1
v_mov_b32 v53, v135                                //
v_mov_b32 v135, v62                                // glvw 1 mb 0 tt1 15 r 0
v_mov_b32 v54, v135                                //
v_mov_b32 v135, v63                                // glvw 1 mb 0 tt1 15 r 1
v_mov_b32 v55, v135                                //
v_mov_b32 v135, v72                                // glvw 1 mb 0 tt1 16 r 0
v_mov_b32 v64, v135                                //
v_mov_b32 v135, v73                                // glvw 1 mb 0 tt1 16 r 1
v_mov_b32 v65, v135                                //
v_mov_b32 v135, v74                                // glvw 1 mb 0 tt1 17 r 0
v_mov_b32 v66, v135                                //
v_mov_b32 v135, v75                                // glvw 1 mb 0 tt1 17 r 1
v_mov_b32 v67, v135                                //
v_mov_b32 v135, v76                                // glvw 1 mb 0 tt1 18 r 0
v_mov_b32 v68, v135                                //
v_mov_b32 v135, v77                                // glvw 1 mb 0 tt1 18 r 1
v_mov_b32 v69, v135                                //
v_mov_b32 v135, v78                                // glvw 1 mb 0 tt1 19 r 0
v_mov_b32 v70, v135                                //
v_mov_b32 v135, v79                                // glvw 1 mb 0 tt1 19 r 1
v_mov_b32 v71, v135                                //
v_mov_b32 v135, v88                                // glvw 1 mb 0 tt1 20 r 0
v_mov_b32 v80, v135                                //
v_mov_b32 v135, v89                                // glvw 1 mb 0 tt1 20 r 1
v_mov_b32 v81, v135                                //
v_mov_b32 v135, v90                                // glvw 1 mb 0 tt1 21 r 0
v_mov_b32 v82, v135                                //
v_mov_b32 v135, v91                                // glvw 1 mb 0 tt1 21 r 1
v_mov_b32 v83, v135                                //
v_mov_b32 v135, v92                                // glvw 1 mb 0 tt1 22 r 0
v_mov_b32 v84, v135                                //
v_mov_b32 v135, v93                                // glvw 1 mb 0 tt1 22 r 1
v_mov_b32 v85, v135                                //
v_mov_b32 v135, v94                                // glvw 1 mb 0 tt1 23 r 0
v_mov_b32 v86, v135                                //
v_mov_b32 v135, v95                                // glvw 1 mb 0 tt1 23 r 1
v_mov_b32 v87, v135                                //
v_mov_b32 v135, v104                               // glvw 1 mb 0 tt1 24 r 0
v_mov_b32 v96, v135                                //
v_mov_b32 v135, v105                               // glvw 1 mb 0 tt1 24 r 1
v_mov_b32 v97, v135                                //
v_mov_b32 v135, v106                               // glvw 1 mb 0 tt1 25 r 0
v_mov_b32 v98, v135                                //
v_mov_b32 v135, v107                               // glvw 1 mb 0 tt1 25 r 1
v_mov_b32 v99, v135                                //
v_mov_b32 v135, v108                               // glvw 1 mb 0 tt1 26 r 0
v_mov_b32 v100, v135                               //
v_mov_b32 v135, v109                               // glvw 1 mb 0 tt1 26 r 1
v_mov_b32 v101, v135                               //
v_mov_b32 v135, v110                               // glvw 1 mb 0 tt1 27 r 0
v_mov_b32 v102, v135                               //
v_mov_b32 v135, v111                               // glvw 1 mb 0 tt1 27 r 1
v_mov_b32 v103, v135                               //
v_mov_b32 v135, v120                               // glvw 1 mb 0 tt1 28 r 0
v_mov_b32 v112, v135                               //
v_mov_b32 v135, v121                               // glvw 1 mb 0 tt1 28 r 1
v_mov_b32 v113, v135                               //
v_mov_b32 v135, v122                               // glvw 1 mb 0 tt1 29 r 0
v_mov_b32 v114, v135                               //
v_mov_b32 v135, v123                               // glvw 1 mb 0 tt1 29 r 1
v_mov_b32 v115, v135                               //
v_mov_b32 v135, v124                               // glvw 1 mb 0 tt1 30 r 0
v_mov_b32 v116, v135                               //
v_mov_b32 v135, v125                               // glvw 1 mb 0 tt1 30 r 1
v_mov_b32 v117, v135                               //
v_mov_b32 v135, v126                               // glvw 1 mb 0 tt1 31 r 0
v_mov_b32 v118, v135                               //
v_mov_b32 v135, v127                               // glvw 1 mb 0 tt1 31 r 1
v_mov_b32 v119, v135                               //
s_mov_b64 s[64:65], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[64:65]                    // all threads active
s_branch label_0029                                // done shifting

label_0029: // end shift0



/* not-LocalSplitU: global write indices */

/* computeStoreVgprs */
v_lshrrev_b32 v132, 6, v[vgprSerial]               // v132 = v[vgprSerial] / 64
v_lshrrev_b32 v133, 2, v132                        // v133 = v132 / 4
v_mul_lo_u32 v133, 0x10, v133                      // wave coordination offset 1
v_and_b32 v129, 63, v[vgprSerial]                  // v129 = v[vgprSerial] % 64
v_lshrrev_b32 v129, 4, v129                        // v129 = v129 / 16
                                                   // thread0 * continuous_output (multiplier is 1, do nothing)
v_add_u32 v129, v133, v129                         // coordination 1 = wave_id1 + tid1
v_mul_lo_u32 v130, v129, s[sgprStrideC1J]          //  offset 1
v_mul_lo_u32 v131, v129, s[sgprStrideD1J]          //  offset 1
v_and_b32 v128, 3, v132                            // v128 = v132 % 4
v_mul_lo_u32 v128, 0x10, v128                      // wave coordination offset 0
v_and_b32 v133, 15, v[vgprSerial]                  // v133 = v[vgprSerial] % 16
_v_add_lshl_u32 v128, v133, v128, 1                // coordination 0 = wave_id0 + tid0
s_mul_i32 s63, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_add_u32 v128, s63, v128                          // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s63, 128, s[sgprWorkGroup1]              // wgp1 * MT1
v_add_u32 v129, s63, v129                          // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1


/* not-LocalSplitU: global write */

s_mov_b32 s64, s[sgprBeta+0]                       // tmp = Beta[0]
s_or_b32 s64, s[sgprBeta+1], s64                   // tmp |= Beta[1]
s_cmpk_eq_u32 s64, 0x0                             // Beta == 0
s_cbranch_scc0 GW_Beta_46                          // Branch if Beta is not zero

s_and_b32 s64, 127, s[sgprSizeI]                   // s64 = s[sgprSizeI] % 128
s_add_u32 s65, -0x1, s[sgprNumWorkGroups0]         //
s_cmp_ge_u32 s[sgprWorkGroup0], s65                // wg0 >= nwg0-1 ?
s_cselect_b32 s64, s64, 0                          // set rMT0
s_cmpk_gt_u32 s64, 0x0                             // rMT0 > 0
s_cbranch_scc1 GW_B0_E1_37                         // jump if edges required
s_and_b32 s64, 127, s[sgprSizeJ]                   // s64 = s[sgprSizeJ] % 128
s_add_u32 s65, -0x1, s[sgprNumWorkGroups1]         //
s_cmp_ge_u32 s[sgprWorkGroup1], s65                // wg1 >= nwg1-1
s_cselect_b32 s64, s64, 0                          // set rMT1
s_cmpk_gt_u32 s64, 0x0                             // rMT1 > 0
s_cbranch_scc1 GW_B0_E1_37                         // jump if edges required
GW_B0_E0_34:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=1 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
_v_add_lshl_u32 v134, v131, v128, 0x3              // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=128, coord0Vgpr=128

/* rC *= alpha batchEements=[(0, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+0:vgprValuC+0+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+8:vgprValuC+8+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx4 v[136:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #1 (d1,d0,vc1,vc0) = */
/*    (1,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */

/* rC *= alpha batchEements=[(1, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+2:vgprValuC+2+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+10:vgprValuC+10+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[136:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #2 (d1,d0,vc1,vc0) = */
/*    (2,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(2,0,0,0) */

/* rC *= alpha batchEements=[(2, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+4:vgprValuC+4+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+12:vgprValuC+12+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[136:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #3 (d1,d0,vc1,vc0) = */
/*    (3,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(3,0,0,0) */

/* rC *= alpha batchEements=[(3, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+6:vgprValuC+6+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+14:vgprValuC+14+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[136:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #4 (d1,d0,vc1,vc0) = */
/*    (4,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(4,0,0,0) */

/* rC *= alpha batchEements=[(4, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+16:vgprValuC+16+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+24:vgprValuC+24+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[136:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #5 (d1,d0,vc1,vc0) = */
/*    (5,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(5,0,0,0) */

/* rC *= alpha batchEements=[(5, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+18:vgprValuC+18+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+26:vgprValuC+26+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[136:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #6 (d1,d0,vc1,vc0) = */
/*    (6,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(6,0,0,0) */

/* rC *= alpha batchEements=[(6, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+20:vgprValuC+20+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+28:vgprValuC+28+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[136:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #7 (d1,d0,vc1,vc0) = */
/*    (7,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(7,0,0,0) */

/* rC *= alpha batchEements=[(7, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+22:vgprValuC+22+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+30:vgprValuC+30+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[136:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #8 (d1,d0,vc1,vc0) = */
/*    (8,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(8,0,0,0) */

/* rC *= alpha batchEements=[(8, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+32:vgprValuC+32+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+40:vgprValuC+40+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[136:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #9 (d1,d0,vc1,vc0) = */
/*    (9,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(9,0,0,0) */

/* rC *= alpha batchEements=[(9, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+34:vgprValuC+34+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+42:vgprValuC+42+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[136:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #10 (d1,d0,vc1,vc0) = */
/*    (10,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(10,0,0,0) */

/* rC *= alpha batchEements=[(10, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+36:vgprValuC+36+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+44:vgprValuC+44+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[136:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #11 (d1,d0,vc1,vc0) = */
/*    (11,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(11,0,0,0) */

/* rC *= alpha batchEements=[(11, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+38:vgprValuC+38+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+46:vgprValuC+46+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[136:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #12 (d1,d0,vc1,vc0) = */
/*    (12,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(12,0,0,0) */

/* rC *= alpha batchEements=[(12, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+48:vgprValuC+48+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+56:vgprValuC+56+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[136:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #13 (d1,d0,vc1,vc0) = */
/*    (13,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(13,0,0,0) */

/* rC *= alpha batchEements=[(13, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+50:vgprValuC+50+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+58:vgprValuC+58+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[136:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #14 (d1,d0,vc1,vc0) = */
/*    (14,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(14,0,0,0) */

/* rC *= alpha batchEements=[(14, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+52:vgprValuC+52+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+60:vgprValuC+60+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[136:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #15 (d1,d0,vc1,vc0) = */
/*    (15,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(15,0,0,0) */

/* rC *= alpha batchEements=[(15, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+54:vgprValuC+54+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+62:vgprValuC+62+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[136:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #16 (d1,d0,vc1,vc0) = */
/*    (16,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(16,0,0,0) */

/* rC *= alpha batchEements=[(16, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+64:vgprValuC+64+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+72:vgprValuC+72+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[136:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #17 (d1,d0,vc1,vc0) = */
/*    (17,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(17,0,0,0) */

/* rC *= alpha batchEements=[(17, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+66:vgprValuC+66+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+74:vgprValuC+74+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[136:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #18 (d1,d0,vc1,vc0) = */
/*    (18,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(18,0,0,0) */

/* rC *= alpha batchEements=[(18, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+68:vgprValuC+68+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+76:vgprValuC+76+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[136:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #19 (d1,d0,vc1,vc0) = */
/*    (19,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(19,0,0,0) */

/* rC *= alpha batchEements=[(19, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+70:vgprValuC+70+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+78:vgprValuC+78+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[136:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #20 (d1,d0,vc1,vc0) = */
/*    (20,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(20,0,0,0) */

/* rC *= alpha batchEements=[(20, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+80:vgprValuC+80+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+88:vgprValuC+88+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[136:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #21 (d1,d0,vc1,vc0) = */
/*    (21,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(21,0,0,0) */

/* rC *= alpha batchEements=[(21, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+82:vgprValuC+82+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+90:vgprValuC+90+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[136:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #22 (d1,d0,vc1,vc0) = */
/*    (22,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(22,0,0,0) */

/* rC *= alpha batchEements=[(22, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+84:vgprValuC+84+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+92:vgprValuC+92+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[136:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #23 (d1,d0,vc1,vc0) = */
/*    (23,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(23,0,0,0) */

/* rC *= alpha batchEements=[(23, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+86:vgprValuC+86+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+94:vgprValuC+94+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[136:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #24 (d1,d0,vc1,vc0) = */
/*    (24,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(24,0,0,0) */

/* rC *= alpha batchEements=[(24, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+96:vgprValuC+96+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+104:vgprValuC+104+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[136:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #25 (d1,d0,vc1,vc0) = */
/*    (25,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(25,0,0,0) */

/* rC *= alpha batchEements=[(25, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+98:vgprValuC+98+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+106:vgprValuC+106+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[136:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #26 (d1,d0,vc1,vc0) = */
/*    (26,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(26,0,0,0) */

/* rC *= alpha batchEements=[(26, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+100:vgprValuC+100+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+108:vgprValuC+108+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[136:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #27 (d1,d0,vc1,vc0) = */
/*    (27,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(27,0,0,0) */

/* rC *= alpha batchEements=[(27, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+102:vgprValuC+102+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+110:vgprValuC+110+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[136:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #28 (d1,d0,vc1,vc0) = */
/*    (28,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(28,0,0,0) */

/* rC *= alpha batchEements=[(28, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+112:vgprValuC+112+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+120:vgprValuC+120+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[136:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #29 (d1,d0,vc1,vc0) = */
/*    (29,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(29,0,0,0) */

/* rC *= alpha batchEements=[(29, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+114:vgprValuC+114+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+122:vgprValuC+122+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[136:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #30 (d1,d0,vc1,vc0) = */
/*    (30,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(30,0,0,0) */

/* rC *= alpha batchEements=[(30, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+116:vgprValuC+116+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+124:vgprValuC+124+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[136:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Batch #31 (d1,d0,vc1,vc0) = */
/*    (31,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(31,0,0,0) */

/* rC *= alpha batchEements=[(31, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+118:vgprValuC+118+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+126:vgprValuC+126+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[136:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_45                           // jump to end
GW_B0_E1_37:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=1 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(0, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+0:vgprValuC+0+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (0,0,0,1:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(0, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+8:vgprValuC+8+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #2 (d1,d0,vc1,vc0) = */
/*    (1,0,0,0:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(1, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+2:vgprValuC+2+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #3 (d1,d0,vc1,vc0) = */
/*    (1,0,0,1:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(1,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(1, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+10:vgprValuC+10+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #4 (d1,d0,vc1,vc0) = */
/*    (2,0,0,0:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(2, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+4:vgprValuC+4+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #5 (d1,d0,vc1,vc0) = */
/*    (2,0,0,1:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(2,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(2, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+12:vgprValuC+12+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #6 (d1,d0,vc1,vc0) = */
/*    (3,0,0,0:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(3, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+6:vgprValuC+6+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #7 (d1,d0,vc1,vc0) = */
/*    (3,0,0,1:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(3,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(3, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+14:vgprValuC+14+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #8 (d1,d0,vc1,vc0) = */
/*    (4,0,0,0:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(4, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+16:vgprValuC+16+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #9 (d1,d0,vc1,vc0) = */
/*    (4,0,0,1:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(4,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(4, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+24:vgprValuC+24+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #10 (d1,d0,vc1,vc0) = */
/*    (5,0,0,0:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(5, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+18:vgprValuC+18+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #11 (d1,d0,vc1,vc0) = */
/*    (5,0,0,1:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(5,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(5, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+26:vgprValuC+26+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #12 (d1,d0,vc1,vc0) = */
/*    (6,0,0,0:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(6, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+20:vgprValuC+20+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #13 (d1,d0,vc1,vc0) = */
/*    (6,0,0,1:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(6,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(6, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+28:vgprValuC+28+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #14 (d1,d0,vc1,vc0) = */
/*    (7,0,0,0:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(7, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+22:vgprValuC+22+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #15 (d1,d0,vc1,vc0) = */
/*    (7,0,0,1:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(7,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(7, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+30:vgprValuC+30+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #16 (d1,d0,vc1,vc0) = */
/*    (8,0,0,0:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(8, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+32:vgprValuC+32+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #17 (d1,d0,vc1,vc0) = */
/*    (8,0,0,1:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(8,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(8, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+40:vgprValuC+40+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #18 (d1,d0,vc1,vc0) = */
/*    (9,0,0,0:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(9, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+34:vgprValuC+34+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #19 (d1,d0,vc1,vc0) = */
/*    (9,0,0,1:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(9,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(9, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+42:vgprValuC+42+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #20 (d1,d0,vc1,vc0) = */
/*    (10,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(10, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+36:vgprValuC+36+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #21 (d1,d0,vc1,vc0) = */
/*    (10,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(10,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(10, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+44:vgprValuC+44+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #22 (d1,d0,vc1,vc0) = */
/*    (11,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(11, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+38:vgprValuC+38+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #23 (d1,d0,vc1,vc0) = */
/*    (11,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(11,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(11, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+46:vgprValuC+46+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #24 (d1,d0,vc1,vc0) = */
/*    (12,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(12, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+48:vgprValuC+48+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #25 (d1,d0,vc1,vc0) = */
/*    (12,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(12,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(12, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+56:vgprValuC+56+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #26 (d1,d0,vc1,vc0) = */
/*    (13,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(13, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+50:vgprValuC+50+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #27 (d1,d0,vc1,vc0) = */
/*    (13,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(13,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(13, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+58:vgprValuC+58+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #28 (d1,d0,vc1,vc0) = */
/*    (14,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(14, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+52:vgprValuC+52+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #29 (d1,d0,vc1,vc0) = */
/*    (14,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(14,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(14, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+60:vgprValuC+60+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #30 (d1,d0,vc1,vc0) = */
/*    (15,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(15, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+54:vgprValuC+54+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #31 (d1,d0,vc1,vc0) = */
/*    (15,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(15,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(15, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+62:vgprValuC+62+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #32 (d1,d0,vc1,vc0) = */
/*    (16,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(16, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+64:vgprValuC+64+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #33 (d1,d0,vc1,vc0) = */
/*    (16,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(16,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(16, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+72:vgprValuC+72+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #34 (d1,d0,vc1,vc0) = */
/*    (17,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(17, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+66:vgprValuC+66+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #35 (d1,d0,vc1,vc0) = */
/*    (17,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(17,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(17, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+74:vgprValuC+74+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #36 (d1,d0,vc1,vc0) = */
/*    (18,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(18, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+68:vgprValuC+68+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #37 (d1,d0,vc1,vc0) = */
/*    (18,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(18,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(18, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+76:vgprValuC+76+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #38 (d1,d0,vc1,vc0) = */
/*    (19,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(19, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+70:vgprValuC+70+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #39 (d1,d0,vc1,vc0) = */
/*    (19,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(19,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(19, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+78:vgprValuC+78+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #40 (d1,d0,vc1,vc0) = */
/*    (20,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(20, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+80:vgprValuC+80+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #41 (d1,d0,vc1,vc0) = */
/*    (20,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(20,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(20, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+88:vgprValuC+88+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #42 (d1,d0,vc1,vc0) = */
/*    (21,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(21, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+82:vgprValuC+82+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #43 (d1,d0,vc1,vc0) = */
/*    (21,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(21,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(21, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+90:vgprValuC+90+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #44 (d1,d0,vc1,vc0) = */
/*    (22,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(22, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+84:vgprValuC+84+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #45 (d1,d0,vc1,vc0) = */
/*    (22,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(22,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(22, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+92:vgprValuC+92+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #46 (d1,d0,vc1,vc0) = */
/*    (23,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(23, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+86:vgprValuC+86+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #47 (d1,d0,vc1,vc0) = */
/*    (23,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(23,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(23, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+94:vgprValuC+94+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #48 (d1,d0,vc1,vc0) = */
/*    (24,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(24, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+96:vgprValuC+96+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #49 (d1,d0,vc1,vc0) = */
/*    (24,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(24,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(24, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+104:vgprValuC+104+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #50 (d1,d0,vc1,vc0) = */
/*    (25,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(25, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+98:vgprValuC+98+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #51 (d1,d0,vc1,vc0) = */
/*    (25,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(25,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(25, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+106:vgprValuC+106+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #52 (d1,d0,vc1,vc0) = */
/*    (26,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(26, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+100:vgprValuC+100+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #53 (d1,d0,vc1,vc0) = */
/*    (26,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(26,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(26, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+108:vgprValuC+108+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #54 (d1,d0,vc1,vc0) = */
/*    (27,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(27, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+102:vgprValuC+102+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #55 (d1,d0,vc1,vc0) = */
/*    (27,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(27,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(27, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+110:vgprValuC+110+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #56 (d1,d0,vc1,vc0) = */
/*    (28,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(28, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+112:vgprValuC+112+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #57 (d1,d0,vc1,vc0) = */
/*    (28,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(28,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(28, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+120:vgprValuC+120+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #58 (d1,d0,vc1,vc0) = */
/*    (29,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(29, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+114:vgprValuC+114+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #59 (d1,d0,vc1,vc0) = */
/*    (29,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(29,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(29, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+122:vgprValuC+122+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #60 (d1,d0,vc1,vc0) = */
/*    (30,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(30, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+116:vgprValuC+116+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #61 (d1,d0,vc1,vc0) = */
/*    (30,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(30,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(30, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+124:vgprValuC+124+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #62 (d1,d0,vc1,vc0) = */
/*    (31,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(31, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+118:vgprValuC+118+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Edge Batch #63 (d1,d0,vc1,vc0) = */
/*    (31,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(31,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(31, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+136:vgprValuC+136+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+126:vgprValuC+126+1] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[136:137], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_45                           // jump to end
GW_Beta_46:
s_and_b32 s64, 127, s[sgprSizeI]                   // s64 = s[sgprSizeI] % 128
s_add_u32 s65, -0x1, s[sgprNumWorkGroups0]         //
s_cmp_ge_u32 s[sgprWorkGroup0], s65                // wg0 >= nwg0-1 ?
s_cselect_b32 s64, s64, 0                          // set rMT0
s_cmpk_gt_u32 s64, 0x0                             // rMT0 > 0
s_cbranch_scc1 GW_B1_E1_44                         // jump if edges required
s_and_b32 s64, 127, s[sgprSizeJ]                   // s64 = s[sgprSizeJ] % 128
s_add_u32 s65, -0x1, s[sgprNumWorkGroups1]         //
s_cmp_ge_u32 s[sgprWorkGroup1], s65                // wg1 >= nwg1-1
s_cselect_b32 s64, s64, 0                          // set rMT1
s_cmpk_gt_u32 s64, 0x0                             // rMT1 > 0
s_cbranch_scc1 GW_B1_E1_44                         // jump if edges required
GW_B1_E0_41:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=1 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(0, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+0:vgprValuC+0+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+8:vgprValuC+8+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
_v_add_lshl_u32 v135, v130, v128, 0x3              // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=128, coord0Vgpr=128
_v_add_lshl_u32 v134, v131, v128, 0x3              // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=128, coord0Vgpr=128
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Batch #1 (d1,d0,vc1,vc0) = */
/*    (1,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(1, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+2:vgprValuC+2+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+10:vgprValuC+10+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
s_mul_i32 s64, s[sgprStrideC1J], 32                // scale StrideC *= numRows(4) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Batch #2 (d1,d0,vc1,vc0) = */
/*    (2,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(2, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+4:vgprValuC+4+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+12:vgprValuC+12+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
s_mul_i32 s64, s[sgprStrideC1J], 32                // scale StrideC *= numRows(4) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Batch #3 (d1,d0,vc1,vc0) = */
/*    (3,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(3, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+6:vgprValuC+6+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+14:vgprValuC+14+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
s_mul_i32 s64, s[sgprStrideC1J], 32                // scale StrideC *= numRows(4) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Batch #4 (d1,d0,vc1,vc0) = */
/*    (4,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(4, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+16:vgprValuC+16+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+24:vgprValuC+24+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
s_mul_i32 s64, s[sgprStrideC1J], 32                // scale StrideC *= numRows(4) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Batch #5 (d1,d0,vc1,vc0) = */
/*    (5,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(5, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+18:vgprValuC+18+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+26:vgprValuC+26+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
s_mul_i32 s64, s[sgprStrideC1J], 32                // scale StrideC *= numRows(4) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Batch #6 (d1,d0,vc1,vc0) = */
/*    (6,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(6, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+20:vgprValuC+20+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+28:vgprValuC+28+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
s_mul_i32 s64, s[sgprStrideC1J], 32                // scale StrideC *= numRows(4) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Batch #7 (d1,d0,vc1,vc0) = */
/*    (7,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(7, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+22:vgprValuC+22+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+30:vgprValuC+30+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
s_mul_i32 s64, s[sgprStrideC1J], 32                // scale StrideC *= numRows(4) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Batch #8 (d1,d0,vc1,vc0) = */
/*    (8,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(8, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+32:vgprValuC+32+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+40:vgprValuC+40+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
s_mul_i32 s64, s[sgprStrideC1J], 32                // scale StrideC *= numRows(4) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Batch #9 (d1,d0,vc1,vc0) = */
/*    (9,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(9, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+34:vgprValuC+34+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+42:vgprValuC+42+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
s_mul_i32 s64, s[sgprStrideC1J], 32                // scale StrideC *= numRows(4) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Batch #10 (d1,d0,vc1,vc0) = */
/*    (10,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(10, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+36:vgprValuC+36+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+44:vgprValuC+44+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
s_mul_i32 s64, s[sgprStrideC1J], 32                // scale StrideC *= numRows(4) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Batch #11 (d1,d0,vc1,vc0) = */
/*    (11,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(11, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+38:vgprValuC+38+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+46:vgprValuC+46+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
s_mul_i32 s64, s[sgprStrideC1J], 32                // scale StrideC *= numRows(4) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Batch #12 (d1,d0,vc1,vc0) = */
/*    (12,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(12, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+48:vgprValuC+48+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+56:vgprValuC+56+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
s_mul_i32 s64, s[sgprStrideC1J], 32                // scale StrideC *= numRows(4) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Batch #13 (d1,d0,vc1,vc0) = */
/*    (13,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(13, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+50:vgprValuC+50+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+58:vgprValuC+58+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
s_mul_i32 s64, s[sgprStrideC1J], 32                // scale StrideC *= numRows(4) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Batch #14 (d1,d0,vc1,vc0) = */
/*    (14,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(14, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+52:vgprValuC+52+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+60:vgprValuC+60+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
s_mul_i32 s64, s[sgprStrideC1J], 32                // scale StrideC *= numRows(4) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Batch #15 (d1,d0,vc1,vc0) = */
/*    (15,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(15, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+54:vgprValuC+54+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+62:vgprValuC+62+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
s_mul_i32 s64, s[sgprStrideC1J], 32                // scale StrideC *= numRows(4) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Batch #16 (d1,d0,vc1,vc0) = */
/*    (16,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(16, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+64:vgprValuC+64+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+72:vgprValuC+72+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
s_mul_i32 s64, s[sgprStrideC1J], 32                // scale StrideC *= numRows(4) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Batch #17 (d1,d0,vc1,vc0) = */
/*    (17,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(17, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+66:vgprValuC+66+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+74:vgprValuC+74+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
s_mul_i32 s64, s[sgprStrideC1J], 32                // scale StrideC *= numRows(4) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Batch #18 (d1,d0,vc1,vc0) = */
/*    (18,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(18, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+68:vgprValuC+68+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+76:vgprValuC+76+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
s_mul_i32 s64, s[sgprStrideC1J], 32                // scale StrideC *= numRows(4) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Batch #19 (d1,d0,vc1,vc0) = */
/*    (19,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(19, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+70:vgprValuC+70+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+78:vgprValuC+78+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
s_mul_i32 s64, s[sgprStrideC1J], 32                // scale StrideC *= numRows(4) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Batch #20 (d1,d0,vc1,vc0) = */
/*    (20,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(20, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+80:vgprValuC+80+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+88:vgprValuC+88+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
s_mul_i32 s64, s[sgprStrideC1J], 32                // scale StrideC *= numRows(4) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Batch #21 (d1,d0,vc1,vc0) = */
/*    (21,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(21, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+82:vgprValuC+82+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+90:vgprValuC+90+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
s_mul_i32 s64, s[sgprStrideC1J], 32                // scale StrideC *= numRows(4) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Batch #22 (d1,d0,vc1,vc0) = */
/*    (22,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(22, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+84:vgprValuC+84+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+92:vgprValuC+92+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
s_mul_i32 s64, s[sgprStrideC1J], 32                // scale StrideC *= numRows(4) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Batch #23 (d1,d0,vc1,vc0) = */
/*    (23,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(23, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+86:vgprValuC+86+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+94:vgprValuC+94+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
s_mul_i32 s64, s[sgprStrideC1J], 32                // scale StrideC *= numRows(4) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Batch #24 (d1,d0,vc1,vc0) = */
/*    (24,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(24, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+96:vgprValuC+96+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+104:vgprValuC+104+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
s_mul_i32 s64, s[sgprStrideC1J], 32                // scale StrideC *= numRows(4) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Batch #25 (d1,d0,vc1,vc0) = */
/*    (25,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(25, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+98:vgprValuC+98+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+106:vgprValuC+106+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
s_mul_i32 s64, s[sgprStrideC1J], 32                // scale StrideC *= numRows(4) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Batch #26 (d1,d0,vc1,vc0) = */
/*    (26,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(26, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+100:vgprValuC+100+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+108:vgprValuC+108+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
s_mul_i32 s64, s[sgprStrideC1J], 32                // scale StrideC *= numRows(4) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Batch #27 (d1,d0,vc1,vc0) = */
/*    (27,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(27, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+102:vgprValuC+102+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+110:vgprValuC+110+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
s_mul_i32 s64, s[sgprStrideC1J], 32                // scale StrideC *= numRows(4) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Batch #28 (d1,d0,vc1,vc0) = */
/*    (28,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(28, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+112:vgprValuC+112+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+120:vgprValuC+120+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
s_mul_i32 s64, s[sgprStrideC1J], 32                // scale StrideC *= numRows(4) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Batch #29 (d1,d0,vc1,vc0) = */
/*    (29,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(29, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+114:vgprValuC+114+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+122:vgprValuC+122+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
s_mul_i32 s64, s[sgprStrideC1J], 32                // scale StrideC *= numRows(4) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Batch #30 (d1,d0,vc1,vc0) = */
/*    (30,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(30, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+116:vgprValuC+116+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+124:vgprValuC+124+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
s_mul_i32 s64, s[sgprStrideC1J], 32                // scale StrideC *= numRows(4) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Batch #31 (d1,d0,vc1,vc0) = */
/*    (31,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(31, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+118:vgprValuC+118+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+126:vgprValuC+126+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
s_mul_i32 s64, s[sgprStrideC1J], 32                // scale StrideC *= numRows(4) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
s_mul_i32 s64, s[sgprStrideD1J], 32                // scale StrideD *= numRows(4) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s64       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_45                           // jump to end
GW_B1_E1_44:

// TODO in Generator
// wider store if M % 2 == 0
s_and_b32 s63, 0x1, s[sgprSizeI]
s_cbranch_scc0 GW_B1_E1_VW2                                // done shifting

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=1 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(0, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+0:vgprValuC+0+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (0,0,0,1:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(0, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+8:vgprValuC+8+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #2 (d1,d0,vc1,vc0) = */
/*    (1,0,0,0:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(1, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+2:vgprValuC+2+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #3 (d1,d0,vc1,vc0) = */
/*    (1,0,0,1:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(1, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+10:vgprValuC+10+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(1,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #4 (d1,d0,vc1,vc0) = */
/*    (2,0,0,0:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(2, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+4:vgprValuC+4+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #5 (d1,d0,vc1,vc0) = */
/*    (2,0,0,1:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(2, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+12:vgprValuC+12+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(2,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #6 (d1,d0,vc1,vc0) = */
/*    (3,0,0,0:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(3, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+6:vgprValuC+6+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #7 (d1,d0,vc1,vc0) = */
/*    (3,0,0,1:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(3, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+14:vgprValuC+14+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(3,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #8 (d1,d0,vc1,vc0) = */
/*    (4,0,0,0:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(4, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+16:vgprValuC+16+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #9 (d1,d0,vc1,vc0) = */
/*    (4,0,0,1:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(4, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+24:vgprValuC+24+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(4,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #10 (d1,d0,vc1,vc0) = */
/*    (5,0,0,0:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(5, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+18:vgprValuC+18+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #11 (d1,d0,vc1,vc0) = */
/*    (5,0,0,1:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(5, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+26:vgprValuC+26+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(5,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #12 (d1,d0,vc1,vc0) = */
/*    (6,0,0,0:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(6, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+20:vgprValuC+20+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #13 (d1,d0,vc1,vc0) = */
/*    (6,0,0,1:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(6, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+28:vgprValuC+28+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(6,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #14 (d1,d0,vc1,vc0) = */
/*    (7,0,0,0:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(7, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+22:vgprValuC+22+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #15 (d1,d0,vc1,vc0) = */
/*    (7,0,0,1:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(7, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+30:vgprValuC+30+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(7,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #16 (d1,d0,vc1,vc0) = */
/*    (8,0,0,0:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(8, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+32:vgprValuC+32+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #17 (d1,d0,vc1,vc0) = */
/*    (8,0,0,1:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(8, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+40:vgprValuC+40+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(8,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #18 (d1,d0,vc1,vc0) = */
/*    (9,0,0,0:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(9, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+34:vgprValuC+34+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #19 (d1,d0,vc1,vc0) = */
/*    (9,0,0,1:vw1)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(9, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+42:vgprValuC+42+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(9,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #20 (d1,d0,vc1,vc0) = */
/*    (10,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(10, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+36:vgprValuC+36+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #21 (d1,d0,vc1,vc0) = */
/*    (10,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(10, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+44:vgprValuC+44+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(10,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #22 (d1,d0,vc1,vc0) = */
/*    (11,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(11, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+38:vgprValuC+38+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #23 (d1,d0,vc1,vc0) = */
/*    (11,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(11, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+46:vgprValuC+46+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(11,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #24 (d1,d0,vc1,vc0) = */
/*    (12,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(12, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+48:vgprValuC+48+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #25 (d1,d0,vc1,vc0) = */
/*    (12,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(12, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+56:vgprValuC+56+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(12,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #26 (d1,d0,vc1,vc0) = */
/*    (13,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(13, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+50:vgprValuC+50+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #27 (d1,d0,vc1,vc0) = */
/*    (13,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(13, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+58:vgprValuC+58+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(13,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #28 (d1,d0,vc1,vc0) = */
/*    (14,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(14, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+52:vgprValuC+52+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #29 (d1,d0,vc1,vc0) = */
/*    (14,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(14, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+60:vgprValuC+60+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(14,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #30 (d1,d0,vc1,vc0) = */
/*    (15,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(15, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+54:vgprValuC+54+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #31 (d1,d0,vc1,vc0) = */
/*    (15,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(15, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+62:vgprValuC+62+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(15,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #32 (d1,d0,vc1,vc0) = */
/*    (16,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(16, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+64:vgprValuC+64+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #33 (d1,d0,vc1,vc0) = */
/*    (16,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(16, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+72:vgprValuC+72+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(16,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #34 (d1,d0,vc1,vc0) = */
/*    (17,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(17, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+66:vgprValuC+66+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #35 (d1,d0,vc1,vc0) = */
/*    (17,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(17, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+74:vgprValuC+74+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(17,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #36 (d1,d0,vc1,vc0) = */
/*    (18,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(18, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+68:vgprValuC+68+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #37 (d1,d0,vc1,vc0) = */
/*    (18,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(18, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+76:vgprValuC+76+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(18,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #38 (d1,d0,vc1,vc0) = */
/*    (19,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(19, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+70:vgprValuC+70+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #39 (d1,d0,vc1,vc0) = */
/*    (19,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(19, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+78:vgprValuC+78+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(19,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #40 (d1,d0,vc1,vc0) = */
/*    (20,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(20, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+80:vgprValuC+80+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #41 (d1,d0,vc1,vc0) = */
/*    (20,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(20, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+88:vgprValuC+88+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(20,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #42 (d1,d0,vc1,vc0) = */
/*    (21,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(21, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+82:vgprValuC+82+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #43 (d1,d0,vc1,vc0) = */
/*    (21,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(21, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+90:vgprValuC+90+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(21,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #44 (d1,d0,vc1,vc0) = */
/*    (22,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(22, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+84:vgprValuC+84+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #45 (d1,d0,vc1,vc0) = */
/*    (22,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(22, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+92:vgprValuC+92+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(22,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #46 (d1,d0,vc1,vc0) = */
/*    (23,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(23, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+86:vgprValuC+86+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #47 (d1,d0,vc1,vc0) = */
/*    (23,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(23, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+94:vgprValuC+94+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(23,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #48 (d1,d0,vc1,vc0) = */
/*    (24,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(24, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+96:vgprValuC+96+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #49 (d1,d0,vc1,vc0) = */
/*    (24,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(24, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+104:vgprValuC+104+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(24,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #50 (d1,d0,vc1,vc0) = */
/*    (25,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(25, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+98:vgprValuC+98+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #51 (d1,d0,vc1,vc0) = */
/*    (25,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(25, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+106:vgprValuC+106+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(25,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #52 (d1,d0,vc1,vc0) = */
/*    (26,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(26, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+100:vgprValuC+100+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #53 (d1,d0,vc1,vc0) = */
/*    (26,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(26, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+108:vgprValuC+108+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(26,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #54 (d1,d0,vc1,vc0) = */
/*    (27,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(27, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+102:vgprValuC+102+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #55 (d1,d0,vc1,vc0) = */
/*    (27,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(27, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+110:vgprValuC+110+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(27,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #56 (d1,d0,vc1,vc0) = */
/*    (28,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(28, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+112:vgprValuC+112+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #57 (d1,d0,vc1,vc0) = */
/*    (28,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(28, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+120:vgprValuC+120+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(28,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #58 (d1,d0,vc1,vc0) = */
/*    (29,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(29, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+114:vgprValuC+114+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #59 (d1,d0,vc1,vc0) = */
/*    (29,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(29, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+122:vgprValuC+122+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(29,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #60 (d1,d0,vc1,vc0) = */
/*    (30,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(30, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+116:vgprValuC+116+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #61 (d1,d0,vc1,vc0) = */
/*    (30,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(30, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+124:vgprValuC+124+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(30,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #62 (d1,d0,vc1,vc0) = */
/*    (31,0,0,0:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(31, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+118:vgprValuC+118+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #63 (d1,d0,vc1,vc0) = */
/*    (31,0,0,1:vw1)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(31, 0, 0, 1)] */
v_mul_f64 v[vgprValuC+138:vgprValuC+138+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+126:vgprValuC+126+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(31,0,0,1) */
_v_add_co_u32 v132, vcc, v128, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[64:65], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v132, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx2 v[136:137], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+138:vgprValuC+138+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+138:vgprValuC+138+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx2 v[138:139], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_45                           // jump to end

GW_B1_E1_VW2:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=1 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(0, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+0:vgprValuC+0+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+8:vgprValuC+8+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (1,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(1, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+2:vgprValuC+2+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+10:vgprValuC+10+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #2 (d1,d0,vc1,vc0) = */
/*    (2,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(2, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+4:vgprValuC+4+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+12:vgprValuC+12+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #3 (d1,d0,vc1,vc0) = */
/*    (3,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(3, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+6:vgprValuC+6+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+14:vgprValuC+14+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #4 (d1,d0,vc1,vc0) = */
/*    (4,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(4, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+16:vgprValuC+16+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+24:vgprValuC+24+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #5 (d1,d0,vc1,vc0) = */
/*    (5,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(5, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+18:vgprValuC+18+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+26:vgprValuC+26+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #6 (d1,d0,vc1,vc0) = */
/*    (6,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(6, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+20:vgprValuC+20+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+28:vgprValuC+28+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #7 (d1,d0,vc1,vc0) = */
/*    (7,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(7, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+22:vgprValuC+22+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+30:vgprValuC+30+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #8 (d1,d0,vc1,vc0) = */
/*    (8,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(8, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+32:vgprValuC+32+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+40:vgprValuC+40+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #9 (d1,d0,vc1,vc0) = */
/*    (9,0,0,0:vw2)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(9, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+34:vgprValuC+34+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+42:vgprValuC+42+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #10 (d1,d0,vc1,vc0) = */
/*    (10,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(10, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+36:vgprValuC+36+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+44:vgprValuC+44+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #11 (d1,d0,vc1,vc0) = */
/*    (11,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(11, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+38:vgprValuC+38+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+46:vgprValuC+46+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #12 (d1,d0,vc1,vc0) = */
/*    (12,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(12, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+48:vgprValuC+48+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+56:vgprValuC+56+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #13 (d1,d0,vc1,vc0) = */
/*    (13,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(13, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+50:vgprValuC+50+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+58:vgprValuC+58+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #14 (d1,d0,vc1,vc0) = */
/*    (14,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(14, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+52:vgprValuC+52+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+60:vgprValuC+60+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #15 (d1,d0,vc1,vc0) = */
/*    (15,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(15, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+54:vgprValuC+54+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+62:vgprValuC+62+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #16 (d1,d0,vc1,vc0) = */
/*    (16,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(16, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+64:vgprValuC+64+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+72:vgprValuC+72+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(16,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #17 (d1,d0,vc1,vc0) = */
/*    (17,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(17, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+66:vgprValuC+66+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+74:vgprValuC+74+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(17,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #18 (d1,d0,vc1,vc0) = */
/*    (18,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(18, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+68:vgprValuC+68+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+76:vgprValuC+76+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(18,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #19 (d1,d0,vc1,vc0) = */
/*    (19,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(19, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+70:vgprValuC+70+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+78:vgprValuC+78+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(19,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #20 (d1,d0,vc1,vc0) = */
/*    (20,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(20, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+80:vgprValuC+80+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+88:vgprValuC+88+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(20,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #21 (d1,d0,vc1,vc0) = */
/*    (21,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(21, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+82:vgprValuC+82+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+90:vgprValuC+90+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(21,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #22 (d1,d0,vc1,vc0) = */
/*    (22,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(22, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+84:vgprValuC+84+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+92:vgprValuC+92+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(22,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #23 (d1,d0,vc1,vc0) = */
/*    (23,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(23, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+86:vgprValuC+86+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+94:vgprValuC+94+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(23,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #24 (d1,d0,vc1,vc0) = */
/*    (24,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(24, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+96:vgprValuC+96+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+104:vgprValuC+104+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(24,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #25 (d1,d0,vc1,vc0) = */
/*    (25,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(25, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+98:vgprValuC+98+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+106:vgprValuC+106+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(25,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #26 (d1,d0,vc1,vc0) = */
/*    (26,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(26, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+100:vgprValuC+100+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+108:vgprValuC+108+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(26,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #27 (d1,d0,vc1,vc0) = */
/*    (27,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(27, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+102:vgprValuC+102+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+110:vgprValuC+110+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(27,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #28 (d1,d0,vc1,vc0) = */
/*    (28,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(28, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+112:vgprValuC+112+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+120:vgprValuC+120+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(28,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #29 (d1,d0,vc1,vc0) = */
/*    (29,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(29, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+114:vgprValuC+114+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+122:vgprValuC+122+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(29,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #30 (d1,d0,vc1,vc0) = */
/*    (30,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(30, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+116:vgprValuC+116+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+124:vgprValuC+124+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(30,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */
s_sleep 5 // optimization: sync and wait
s_barrier

/******************************************/
/* Global Write Beta Edge Batch #31 (d1,d0,vc1,vc0) = */
/*    (31,0,0,0:vw2)                      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */

/* rC *= alpha batchEements=[(31, 0, 0, 0)] */
v_mul_f64 v[vgprValuC+140:vgprValuC+140+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+118:vgprValuC+118+1] // Multiply MI out reg with alpha
v_mul_f64 v[vgprValuC+142:vgprValuC+142+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+126:vgprValuC+126+1] // Multiply MI out reg with alpha
/* (d1,vc1,d0,vc0)=(31,0,0,0) */
_v_add_co_u32 v129, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s64, s[sgprStrideC1J], 4                 // scale stride
_v_add_u32 v130, v130, s64                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s64, s[sgprStrideD1J], 4                 // scale stride
_v_add_u32 v131, v131, s64                         // Move coutRowPtr to next row
v_cmp_lt_u32 s[64:65], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[38:39], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[38:39], s[64:65], s[38:39]             // in0 && in1
_v_add_lshl_u32 v135, v130, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, -1, v135, s[38:39]             // LDC clip if OOB. offset
_v_add_lshl_u32 v134, v131, v128, 0x3              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[38:39]             // LDD clip if OOB. offset
buffer_load_dwordx4 v[136:139], v135, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0,  glc slc // load C for beta calc
s_sleep 5 // optimization: sync and wait
s_barrier
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_fma_f64 v[vgprValuC+140:vgprValuC+140+1], v[136:137], s[sgprBeta:sgprBeta+1], v[vgprValuC+140:vgprValuC+140+1] // finalSum = sum*alpha + C*beta
v_fma_f64 v[vgprValuC+142:vgprValuC+142+1], v[138:139], s[sgprBeta:sgprBeta+1], v[vgprValuC+142:vgprValuC+142+1] // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[140:143], v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0,  glc slc // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_45                           // jump to end
label_GW_End_45:

label_0047:  /// KernelEnd
s_endpgm                                           // Kernel End
